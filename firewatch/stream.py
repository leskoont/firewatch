from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from functools import partial

import cv2
import numpy as np

log = logging.getLogger(__name__)

# udp gives smeared gray frames on wifi
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

_KEYS = r"(pas\w*|pwd|pw|secret|token|key|auth)="
# XMEye cameras put credentials in the path, /user=admin_password=xxx_channel=1,
# so "_" ends a value there. In the query a value runs to the next "&".
_XMEYE_SECRET_RE = re.compile(r"(?<=[/_])" + _KEYS + r"[^_/&?#]*", re.IGNORECASE)
_PATH_SECRET_RE = re.compile(r"(?<![/_])" + _KEYS + r"[^&]*", re.IGNORECASE)
_QUERY_SECRET_RE = re.compile(_KEYS + r"[^&]*", re.IGNORECASE)


def userinfo_end(url: str) -> int:
    """Index of the "@" that ends user:password in url, -1 if there is none."""
    start = url.find("://") + 3
    if start < 3:
        return -1
    slash = url.find("/", start)
    if slash < 0:
        return url.rfind("@", start)
    # FFmpeg takes the last "@" before the first "/". A raw "/" in the password moves
    # that "@" into the path, where it looks just like /live@main, so a path "@" counts
    # only after a user:pass segment or, with no "@" up front, before a host-like part.
    end = url.rfind("@", start, slash)
    _, colon, port = url[start:slash].partition(":")
    query = url.find("?", slash)
    path = url[slash:query] if query >= 0 else url[slash:]
    at = len(path)
    while (at := path.rfind("@", 0, at)) > 0:
        segment = path[path.rfind("/", 0, at) + 1 : at]
        host = path[at + 1 :].split("/", 1)[0]
        if ":" in segment or (end < 0 and colon and (not port.isdigit() or re.search(r"[.:\[]", host))):
            return slash + at
    return end


def mask_credentials(url: str) -> str:
    # local paths are left alone, data/videos/key=1.mp4 is just a file name
    if "://" not in url:
        return url
    end = userinfo_end(url)
    if end >= 0:
        url = url[: url.find("://") + 3] + "***" + url[end:]
    path, sep, query = url.partition("?")
    path = _PATH_SECRET_RE.sub(r"\1=***", _XMEYE_SECRET_RE.sub(r"\1=***", path))
    return path + sep + _QUERY_SECRET_RE.sub(r"\1=***", query)


def open_capture(source: str, timeout_ms: int = 10_000) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    # without these a dead RTSP server blocks read() for ~30 s
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms]
    return cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)


class VideoStream:
    """Yields every `every`-th frame from a file, webcam or network stream.

    Live sources are read in a background thread that keeps only the newest frame,
    so with slow inference old frames are dropped and latency stays flat. Broken
    live streams are reopened after `reconnect_delay` seconds.
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
        camera: str = "",
    ):
        self.source = source
        self.name = mask_credentials(source)
        if camera:
            self.name = f"camera {camera} ({self.name})"
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
        self._idx = -1

    def frames(self, stop: threading.Event) -> Iterator[tuple[int, np.ndarray]]:
        if self.live:
            yield from self._live_frames(stop)
        else:
            yield from self._file_frames(stop)

    def _file_frames(self, stop: threading.Event) -> Iterator[tuple[int, np.ndarray]]:
        cap = self._open(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {self.name}")
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
                        raise RuntimeError(f"{self.name}: no frames after {rewinds} rewinds")
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
            with self._cond:
                halt.set()
                self.connected = False
                self._cond.notify_all()
            reader.join(timeout=5)

    def _has_new(self, stop: threading.Event, last_idx: int) -> bool:
        return stop.is_set() or (self._latest is not None and self._latest[0] != last_idx)

    def _reader(self, halt: threading.Event) -> None:
        while not halt.is_set():
            try:
                error = self._read_until_lost(halt)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                if error != self.error:
                    log.exception("%s: reader failed", self.name)
            # halt is set under the same lock, so a reader that outlived its join() timeout
            # can't overwrite what the next session reports
            with self._cond:
                if halt.is_set():
                    return
                self.connected, self.error = False, error
                self.reconnects += 1
                log.warning("%s: %s, reconnect in %gs", self.name, error, self.reconnect_delay)
            halt.wait(self.reconnect_delay)

    def _read_until_lost(self, halt: threading.Event) -> str | None:
        cap = self._open(self.source)
        try:
            if not cap.isOpened():
                # FFmpeg prints the real reason (401, 404...) only to stderr
                return "cannot open stream: wrong address or credentials, or the camera is down"
            with self._cond:
                if halt.is_set():
                    return None
                self.connected, self.error = True, None
                self.fps = cap.get(cv2.CAP_PROP_FPS) or None
            log.info("%s: connected", self.name)
            last_ok = time.monotonic()
            while not halt.is_set():
                ok, frame = cap.read()
                if not ok:
                    # some backends return False for a moment instead of blocking
                    if time.monotonic() - last_ok > self.read_timeout:
                        return "no frames, stream lost"
                    halt.wait(0.05)
                    continue
                last_ok = time.monotonic()
                with self._cond:
                    if halt.is_set():
                        return None
                    self._idx += 1
                    if self._idx % self.every == 0:
                        self._latest = (self._idx, frame)
                        self._cond.notify_all()
            return None
        finally:
            cap.release()
