from __future__ import annotations

import threading
from collections.abc import Callable, Iterator

import cv2
import numpy as np


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


class VideoStream:
    """Yields every `every`-th frame from a video file or webcam."""

    def __init__(
        self,
        source: str,
        every: int = 1,
        capture_factory: Callable[[str], cv2.VideoCapture] = open_capture,
    ):
        self.source = source
        self.every = max(1, every)
        self._open = capture_factory

        self.connected = False
        self.fps: float | None = None

    def frames(self, stop: threading.Event) -> Iterator[tuple[int, np.ndarray]]:
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
