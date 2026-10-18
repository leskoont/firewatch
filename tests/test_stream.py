import threading

import cv2
import numpy as np
import pytest

from firewatch.stream import VideoStream, mask_credentials


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


@pytest.mark.parametrize(
    ("url", "masked"),
    [
        ("rtsp://admin:secret@10.0.0.2:554/s", "rtsp://***@10.0.0.2:554/s"),
        ("rtsp://10.0.0.2/s", "rtsp://10.0.0.2/s"),
        ("rtsp://admin:p@ss@10.0.0.2/s", "rtsp://***@10.0.0.2/s"),
        ("rtsp://admin:p%40ss%2Fw0rd@10.0.0.2/s", "rtsp://***@10.0.0.2/s"),
        # an '@' in the path is not userinfo, unless the password has a raw '/'
        ("rtsp://u:p@10.0.0.2/live@main", "rtsp://***@10.0.0.2/live@main"),
        ("rtsp://10.0.0.2:554/live@main", "rtsp://10.0.0.2:554/live@main"),
        ("http://cam1.lan/stream@1.mjpg", "http://cam1.lan/stream@1.mjpg"),
        ("rtsp://admin:12/34@10.0.0.2/s", "rtsp://***@10.0.0.2/s"),
        ("rtsp://u:p@ss#1@10.0.0.2/s", "rtsp://***@10.0.0.2/s"),
        ("rtsp://admin:secret@10.0.0.2:554/admin:secret@x", "rtsp://***@x"),
        (
            "http://10.0.0.3/video.mjpg?user=me@corp.com&pwd=x",
            "http://10.0.0.3/video.mjpg?user=me@corp.com&pwd=***",
        ),
        ("http://10.0.0.3:8080/video?login=me@x.org", "http://10.0.0.3:8080/video?login=me@x.org"),
        ("rtsp://admin:pa/ss@10.0.0.2:554/s", "rtsp://***@10.0.0.2:554/s"),
        ("rtsp://admin:pa#ss@10.0.0.2:554/s", "rtsp://***@10.0.0.2:554/s"),
        ("rtsp://admin:pa?ss@10.0.0.2:554/s", "rtsp://***@10.0.0.2:554/s"),
        ("rtsp://admin:pw@[fe80::1]:554/s", "rtsp://***@[fe80::1]:554/s"),
        ("rtsp://[fe80::1:554/s", "rtsp://[fe80::1:554/s"),
        ("rtsp://admin:secret\n@10.0.0.2/s", "rtsp://***@10.0.0.2/s"),
        ("rtsp://admin:se cret@10.0.0.2/s", "rtsp://***@10.0.0.2/s"),
        ("RTSP://Admin:Secret@10.0.0.2/s", "RTSP://***@10.0.0.2/s"),
        ("rtsp://user@10.0.0.2/s", "rtsp://***@10.0.0.2/s"),
        (
            "rtsp://10.0.0.2:554/user=admin_password=secret_channel=1_stream=0.sdp",
            "rtsp://10.0.0.2:554/user=admin_password=***_channel=1_stream=0.sdp",
        ),
        (
            "rtsp://10.0.0.2:554/user=admin&password=secret&channel=1&stream=0.sdp?",
            "rtsp://10.0.0.2:554/user=admin&password=***&channel=1&stream=0.sdp?",
        ),
        ("http://cam/snap?loginuse=admin&loginpas=secret", "http://cam/snap?loginuse=admin&loginpas=***"),
        ("http://cam/s?pass=x", "http://cam/s?pass=***"),
        ("http://10.0.0.3/video.cgi?user=admin&pwd=secret", "http://10.0.0.3/video.cgi?user=admin&pwd=***"),
        (
            "rtsp://10.0.0.2/user=admin&password=sec_ret&channel=1",
            "rtsp://10.0.0.2/user=admin&password=***&channel=1",
        ),
        ("http://cam/mjpg?usr=admin&passwd=hun_ter2", "http://cam/mjpg?usr=admin&passwd=***"),
        ("http://cam/videostream.cgi?user=admin&pw=hunter2", "http://cam/videostream.cgi?user=admin&pw=***"),
        ("http://cam/mjpg?user=admin&secret=hunter2", "http://cam/mjpg?user=admin&secret=***"),
        ("https://u:tok@cam.example.com/mjpeg?token=abc.def", "https://***@cam.example.com/mjpeg?token=***"),
        ("http://cam/mjpg?Token=abc&fps=5", "http://cam/mjpg?Token=***&fps=5"),
        # in the query '_' is part of the key or the value, not a separator
        ("http://cam/snap.cgi?user=admin&login_pwd=sec_ret", "http://cam/snap.cgi?user=admin&login_pwd=***"),
        ("http://cam/cgi?cam_password=abc_def&x=1", "http://cam/cgi?cam_password=***&x=1"),
        ("http://cam/cgi?user_pass=abc_def", "http://cam/cgi?user_pass=***"),
        ("http://cam/cgi?a=1&_token=abc_def", "http://cam/cgi?a=1&_token=***"),
        ("rtsp://cam/s?keyframe=1&key=abc", "rtsp://cam/s?keyframe=1&key=***"),
        ("data/videos/key=1.mp4", "data/videos/key=1.mp4"),
        ("videos/a.mp4", "videos/a.mp4"),
        ("data/videos/a@b.mp4", "data/videos/a@b.mp4"),
        (r"C:\videos\me@home.mp4", r"C:\videos\me@home.mp4"),
        ("//nas/share/a@b.mp4", "//nas/share/a@b.mp4"),
        ("0", "0"),
    ],
)
def test_mask_credentials(url, masked):
    assert mask_credentials(url) == masked
    assert mask_credentials(masked) == masked


def test_open_failure_does_not_guess_the_reason():
    vs = VideoStream(
        "rtsp://cam",
        live=True,
        reconnect_delay=10,
        capture_factory=lambda s: FakeCapture(0, opened=False),
    )
    stop = threading.Event()
    gen = vs.frames(stop)
    threading.Timer(0.2, stop.set).start()
    assert list(gen) == []
    assert vs.error.startswith("cannot open stream")
    assert "password" not in vs.error


@pytest.mark.parametrize("opened", [False, True])
def test_stale_reader_keeps_its_hands_off_the_state(opened):
    # the session ends while the old reader is still inside open()
    old_session = threading.Event()

    def slow_open(source):
        old_session.set()
        return FakeCapture(10, opened=opened)

    vs = VideoStream("rtsp://cam", live=True, capture_factory=slow_open)
    vs.connected, vs.error, vs.fps = True, None, 12.0
    vs._reader(old_session)
    assert (vs.connected, vs.error, vs.fps, vs.reconnects) == (True, None, 12.0, 0)


def test_logs_name_the_camera(caplog):
    vs = VideoStream(
        "rtsp://admin:secret@10.0.0.2/s",
        live=True,
        reconnect_delay=10,
        capture_factory=lambda s: FakeCapture(0, opened=False),
        camera="gate",
    )
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    assert list(vs.frames(stop)) == []
    assert "camera gate (rtsp://***@10.0.0.2/s): cannot open stream" in caplog.text
    assert "secret" not in caplog.text


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


def test_live_reconnects_after_failures():
    # first open fails, second stream dies after 3 frames, third one works
    caps = iter([FakeCapture(0, opened=False), FakeCapture(100, fail_after=3)])
    vs = VideoStream(
        "rtsp://cam",
        live=True,
        reconnect_delay=0.01,
        read_timeout=0.05,
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


class CrashingCapture(FakeCapture):
    def read(self):
        raise RuntimeError("decoder died")


def test_live_reader_survives_exceptions():
    caps = iter([CrashingCapture(10)])
    vs = VideoStream(
        "rtsp://cam",
        live=True,
        reconnect_delay=0.01,
        capture_factory=lambda s: next(caps, None) or FakeCapture(10**9),
    )
    stop = threading.Event()
    for _ in vs.frames(stop):
        stop.set()
    assert vs.reconnects == 1


def test_live_generator_close_does_not_set_caller_event():
    vs = VideoStream("0", live=True, capture_factory=lambda s: FakeCapture(10_000))
    stop = threading.Event()
    gen = vs.frames(stop)
    next(gen)
    gen.close()
    assert not stop.is_set()
