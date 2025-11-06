from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


class VideoStream:
    """Yields every `every`-th frame from a file, webcam or network stream.

    Broken live streams are reopened after `reconnect_delay` seconds.
    """

    def __init__(
        self,
        source: str,
        live: bool = False,
        every: int = 1,
        reconnect_delay: float = 5.0,
        capture_factory: Callable[[str], cv2.VideoCapture] = open_capture,
    ):
        self.source = source
        self.live = live
        self.every = max(1, every)
        self.reconnect_delay = reconnect_delay
        self._open = capture_factory

        self.connected = False
        self.reconnects = 0
        self.fps: float | None = None

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
        idx = -1
        try:
            while not stop.is_set():
                idx += 1
                skip = idx % self.every
                ok, frame = (cap.grab(), None) if skip else cap.read()
                if not ok:
                    break
                if not skip:
                    yield idx, frame
        finally:
            cap.release()
            self.connected = False

    def _live_frames(self, stop: threading.Event) -> Iterator[tuple[int, np.ndarray]]:
        # TODO: inference is slower than the camera, frames pile up in the decoder
        idx = -1
        while not stop.is_set():
            cap = self._open(self.source)
            try:
                if cap.isOpened():
                    log.info("%s: connected", self.source)
                    self.connected = True
                    self.fps = cap.get(cv2.CAP_PROP_FPS) or None
                    while not stop.is_set():
                        ok, frame = cap.read()
                        if not ok:
                            break
                        idx += 1
                        if idx % self.every == 0:
                            yield idx, frame
            finally:
                cap.release()
                self.connected = False
            if not stop.is_set():
                log.warning("%s: stream lost, reconnect in %gs", self.source, self.reconnect_delay)
                self.reconnects += 1
                stop.wait(self.reconnect_delay)
