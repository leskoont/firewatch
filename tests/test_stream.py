import threading

import cv2
import numpy as np
import pytest

from firewatch.stream import VideoStream


class FakeCapture:
    """Minimal cv2.VideoCapture stand-in: frame i is filled with value i."""

    def __init__(self, n_frames: int, opened: bool = True, fail_after: int | None = None):
        self.n = n_frames
        self.opened = opened
        self.fail_after = fail_after
        self.pos = 0

    def isOpened(self):
        return self.opened

    def _next(self):
        if self.pos >= self.n or (self.fail_after is not None and self.pos >= self.fail_after):
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

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = int(value)

    def release(self):
        pass


def test_file_frames_are_skipped():
    vs = VideoStream("a.mp4", live=False, every=3, capture_factory=lambda s: FakeCapture(10))
    got = list(vs.frames(threading.Event()))
    assert [i for i, _ in got] == [0, 3, 6, 9]
    assert [int(f[0, 0, 0]) for _, f in got] == [0, 3, 6, 9]
    assert vs.fps == 25.0


def test_file_loop_rewinds():
    stop = threading.Event()
    vs = VideoStream("a.mp4", live=False, loop=True, capture_factory=lambda s: FakeCapture(3))
    values = []
    for _, frame in vs.frames(stop):
        values.append(int(frame[0, 0, 0]))
        if len(values) == 7:
            stop.set()
    assert values == [0, 1, 2, 0, 1, 2, 0]


def test_cannot_open():
    vs = VideoStream("a.mp4", live=False, capture_factory=lambda s: FakeCapture(0, opened=False))
    with pytest.raises(RuntimeError, match="cannot open"):
        list(vs.frames(threading.Event()))


def test_live_reconnects_after_failures():
    # first open fails, second stream dies after 3 frames, third one works
    caps = iter([FakeCapture(0, opened=False), FakeCapture(100, fail_after=3)])
    vs = VideoStream(
        "rtsp://cam",
        live=True,
        reconnect_delay=0.01,
        capture_factory=lambda s: next(caps, None) or FakeCapture(10**9),
    )
    stop = threading.Event()
    frames = 0
    for _ in vs.frames(stop):
        frames += 1
        if vs.reconnects >= 2 and frames >= 5:
            stop.set()
    assert vs.reconnects == 2
    assert frames >= 5


def test_file_loop_gives_up_on_empty_video():
    vs = VideoStream("a.mp4", live=False, loop=True, capture_factory=lambda s: FakeCapture(0))
    with pytest.raises(RuntimeError, match="no frames"):
        list(vs.frames(threading.Event()))


def test_live_generator_close_does_not_set_caller_event():
    vs = VideoStream("0", live=True, capture_factory=lambda s: FakeCapture(10_000))
    stop = threading.Event()
    gen = vs.frames(stop)
    next(gen)
    gen.close()
    assert not stop.is_set()
