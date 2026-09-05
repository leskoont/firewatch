import cv2
import numpy as np
import pytest

from firewatch.config import CameraConfig, Settings, StorageConfig
from firewatch.detector import Detection
from firewatch.monitor import CameraWorker, Monitor


def det(label, score, box=(10, 10, 50, 50)):
    return Detection(*box, score=score, class_id=int(label == "fire"), label=label)


class ScriptedDetector:
    """Returns a predefined list of detections per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, img, conf=None):
        self.calls.append(conf)
        return self.script.pop(0) if self.script else []


@pytest.fixture
def frame():
    return np.zeros((120, 160, 3), np.uint8)


def make_worker(tmp_path, detector, alerts, **cam):
    cfg = CameraConfig(name="cam1", source="video.mp4", window=4, min_hits=3, cooldown_s=600, **cam)
    storage = StorageConfig(dir=tmp_path, snapshots=True, clips=False)
    return CameraWorker(cfg, detector, storage, alerts.append)


def test_per_class_thresholds(tmp_path):
    w = make_worker(tmp_path, None, [], thresholds={"fire": 0.5, "smoke": 0.7})
    kept = w.filter([det("fire", 0.55), det("smoke", 0.6), det("smoke", 0.75), det("person", 0.99)])
    assert [(d.label, d.score) for d in kept] == [("fire", 0.55), ("smoke", 0.75)]


def test_alert_after_persistent_detection(tmp_path, frame):
    alerts = []
    w = make_worker(tmp_path, None, alerts)
    for dets in [[det("fire", 0.9)], [], [det("fire", 0.8)], [det("fire", 0.7)], [det("fire", 0.9)]]:
        w._process(frame, dets)

    assert len(alerts) == 1  # third hit inside the window, then cooldown
    a = alerts[0]
    assert a.camera == "cam1" and a.labels == ["fire"]
    assert "cam1" in a.caption and "0.70" in a.caption
    assert len(list(tmp_path.glob("cam1_*.jpg"))) == 1
    assert w.status.alerts == 1
    assert w.status.frames_processed == 5
    assert cv2.imdecode(np.frombuffer(a.jpeg, np.uint8), cv2.IMREAD_COLOR).any()  # boxes drawn


def test_alert_on_a_miss_frame_shows_last_boxes(tmp_path, frame):
    from firewatch.smoothing import AlertGate

    now = [0.0]
    alerts = []
    w = make_worker(tmp_path, None, alerts)
    w.gate = AlertGate(w.cfg.thresholds, window=3, min_hits=2, cooldown_s=10, clock=lambda: now[0])
    w._process(frame, [det("smoke", 0.9)])
    w._process(frame, [det("smoke", 0.8)])
    assert len(alerts) == 1

    # cooldown is over while the window still holds two hits, current frame is empty
    now[0] = 11.0
    w._process(frame, [])
    assert len(alerts) == 2
    assert [d.score for d in alerts[1].detections] == [0.8]
    assert alerts[1].jpeg.startswith(b"\xff\xd8")


def test_flicker_does_not_alert(tmp_path, frame):
    alerts = []
    w = make_worker(tmp_path, None, alerts)
    for i in range(20):
        w._process(frame, [det("smoke", 0.9)] if i % 3 == 0 else [])
    assert alerts == []


def test_worker_runs_over_video_file(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(12):
        writer.write(np.full((48, 64, 3), 80, np.uint8))
    writer.release()

    detector = ScriptedDetector([[det("fire", 0.9)]] * 12)
    alerts = []
    cfg = CameraConfig(name="file", source=str(path), process_every=2, window=3, min_hits=2)
    storage = StorageConfig(dir=tmp_path / "ev", clips=True, clip_after_s=0)
    w = CameraWorker(cfg, detector, storage, alerts.append)
    w.run()

    assert w.status.state == "stopped"
    assert w.status.frames_processed == 6
    assert len(alerts) == 1
    assert list((tmp_path / "ev").glob("*.mp4"))


def test_monitor_dispatches_to_notifier(tmp_path, frame):
    sent = []

    class Notifier:
        def send_photo(self, image, caption):
            sent.append(caption)
            return True

    settings = Settings(
        storage=StorageConfig(dir=tmp_path, snapshots=False),
        cameras=[
            CameraConfig(name="a", source="a.mp4", window=1, min_hits=1),
            CameraConfig(name="b", source="b.mp4", enabled=False),
        ],
    )
    m = Monitor(settings, detector=None, notifier=Notifier())
    assert [w.cfg.name for w in m.workers] == ["a"]
    m.sender.start()
    m.workers[0]._process(frame, [det("fire", 0.9)])
    m.stop()
    assert len(sent) == 1 and sent[0].startswith("[a] fire")


def test_worker_restarts_after_crash(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(4):
        writer.write(np.full((48, 64, 3), 80, np.uint8))
    writer.release()

    class Flaky:
        calls = 0

        def __call__(self, img, conf=None):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("CUDA error")
            return []

    cfg = CameraConfig(name="file", source=str(path), process_every=1)
    w = CameraWorker(cfg, Flaky(), StorageConfig(dir=tmp_path), lambda a: None)
    w.run()
    # one frame before the crash, then the whole file again
    assert w.status.frames_processed == 5
    assert w.status.state == "stopped"


def test_status_follows_stream_between_frames(tmp_path, frame):
    settings = Settings(
        storage=StorageConfig(dir=tmp_path),
        cameras=[CameraConfig(name="gate", source="rtsp://u:p@10.0.0.5/live")],
    )
    m = Monitor(settings, detector=None)
    w = m.workers[0]
    w.stream.connected = True
    w._process(frame, [])
    assert m.statuses()[0].state == "running"

    # camera went down, no new frames reach _process
    w.stream.connected = False
    w.stream.reconnects = 2
    st = m.statuses()[0]
    assert (st.state, st.connected, st.reconnects) == ("reconnecting", False, 2)
    assert st.source == "rtsp://***@10.0.0.5/live"
