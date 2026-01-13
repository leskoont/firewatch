from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from functools import partial

import cv2
import numpy as np

log = logging.getLogger(__name__)

# udp gives smeared gray frames on wifi
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def open_capture(source: str, timeout_ms: int = 10_000) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    # without these a dead RTSP server blocks read() for ~30 s
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms]
    return cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)


class VideoStream:
    """Yields every `every`-th frame from a file, webcam or network stream.

    Live sources are read in a background thread that keeps only the newest frame,
    so slow inference drops frames instead of accumulating latency. Broken live
    streams are reopened after `reconnect_delay` seconds.
    """

    def __init__(
        self,
        source: str,
        live: bool,
        every: int = 1,
        reconnect_delay: float = 5.0,
        loop: bool = False,
        capture_factory: Callable[[str], cv2.VideoCapture] = open_capture,
        read_timeout: float = 10.0,
    ):
        self.source = source
        self.live = live
        self.every = max(1, every)
        self.reconnect_delay = reconnect_delay
        self.loop = loop
        self.read_timeout = read_timeout
        self._open = capture_factory

        self.connected = False
        self.reconnects = 0
        self.error: str | None = None
        self.fps: float | None = None

        self._cond = threading.Condition()
        self._latest: tuple[int, np.ndarray] | None = None

    def frames(self, stop: threading.Event) -> Iterator[tuple[int, np.ndarray]]:
        if self.live:
            yield from self._live_frames(stop)
        else:
            yield from self._file_frames(stop)

    def _file_frames(self, stop: threading.Event) -> Iterator[tuple[int, np.ndarray]]:
        cap = self._open(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {self.source}")
        self.connected = True
        self.fps = cap.get(cv2.CAP_PROP_FPS) or None
        idx, rewinds = -1, 0
        try:
            while not stop.is_set():
                idx += 1
                skip = idx % self.every
                ok, frame = (cap.grab(), None) if skip else cap.read()
                if not ok:
                    if not self.loop:
                        break
                    if rewinds >= 3:
                        raise RuntimeError(f"{self.source}: no frames after {rewinds} rewinds")
                    if rewinds:
                        stop.wait(1.0)
                    rewinds += 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                rewinds = 0
                if not skip:
                    yield idx, frame
        finally:
            cap.release()
            self.connected = False

    def _live_frames(self, stop: threading.Event) -> Iterator[tuple[int, np.ndarray]]:
        # the reader gets its own event so closing this generator never touches the caller's one
        halt = threading.Event()
        self._latest = None
        reader = threading.Thread(target=self._reader, args=(halt,), daemon=True)
        reader.start()
        last_idx = -1
        try:
            while not stop.is_set():
                with self._cond:
                    self._cond.wait_for(partial(self._has_new, stop, last_idx), timeout=1.0)
                    item = self._latest
                if item is None or item[0] == last_idx:
                    continue
                last_idx = item[0]
                yield item
        finally:
            halt.set()
            with self._cond:
                self._cond.notify_all()
            reader.join(timeout=5)

    def _has_new(self, stop: threading.Event, last_idx: int) -> bool:
        return stop.is_set() or (self._latest is not None and self._latest[0] != last_idx)

    def _reader(self, stop: threading.Event) -> None:
        self._idx = -1
        while not stop.is_set():
            try:
                self._read_until_lost(stop)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                if error != self.error:
                    log.exception("%s: reader failed", self.source)
                self.error = error
            self.connected = False
            if not stop.is_set():
                log.warning("%s: %s, reconnect in %gs", self.source, self.error, self.reconnect_delay)
                self.reconnects += 1
                stop.wait(self.reconnect_delay)

    def _read_until_lost(self, stop: threading.Event) -> None:
        cap = self._open(self.source)
        try:
            if not cap.isOpened():
                # FFmpeg prints the real reason (401, 404...) only to stderr
                self.error = "cannot open stream, check the address, login and password"
                return
            log.info("%s: connected", self.source)
            self.connected, self.error = True, None
            self.fps = cap.get(cv2.CAP_PROP_FPS) or None
            last_ok = time.monotonic()
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    # some backends return False for a moment instead of blocking
                    if time.monotonic() - last_ok > self.read_timeout:
                        self.error = "no frames, stream lost"
                        return
                    stop.wait(0.05)
                    continue
                last_ok = time.monotonic()
                with self._cond:
                    # a reader that outlived its join() timeout must not feed the next one
                    if stop.is_set():
                        return
                    self._idx += 1
                    if self._idx % self.every == 0:
                        self._latest = (self._idx, frame)
                        self._cond.notify_all()
        finally:
            cap.release()
