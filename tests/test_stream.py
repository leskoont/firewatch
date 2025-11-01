import threading

import cv2
import numpy as np
import pytest

from firewatch.stream import VideoStream


class FakeCapture:
    """Minimal cv2.VideoCapture stand-in: frame i is filled with value i."""

    def __init__(self, n_frames: int, opened: bool = True):
        self.n = n_frames
        self.opened = opened
        self.pos = 0

    def isOpened(self):
        return self.opened

    def _next(self):
        if self.pos >= self.n:
            return False, None
        frame = np.full((4, 4, 3), self.pos % 256, np.uint8)
        self.pos += 1
        return True, frame

    def read(self):
        return self._next()

    def grab(self):
        return self._next()[0]

    def get(self, prop):
        return 25.0 if prop == cv2.CAP_PROP_FPS else 0.0

    def release(self):
        pass


def test_file_frames_are_skipped():
    vs = VideoStream("a.mp4", every=3, capture_factory=lambda s: FakeCapture(10))
    got = list(vs.frames(threading.Event()))
    assert [i for i, _ in got] == [0, 3, 6, 9]
    assert [int(f[0, 0, 0]) for _, f in got] == [0, 3, 6, 9]
    assert vs.fps == 25.0


def test_cannot_open():
    vs = VideoStream("a.mp4", capture_factory=lambda s: FakeCapture(0, opened=False))
    with pytest.raises(RuntimeError, match="cannot open"):
        list(vs.frames(threading.Event()))
