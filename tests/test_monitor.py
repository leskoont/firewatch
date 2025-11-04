import cv2
import numpy as np
import pytest

from firewatch.config import CameraConfig, Settings
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


def make_worker(detector, alerts, **cam):
    cfg = CameraConfig(name="cam1", source="video.mp4", window=4, min_hits=3, cooldown_s=600, **cam)
    return CameraWorker(cfg, detector, alerts.append)


def test_per_class_thresholds():
    w = make_worker(None, [], thresholds={"fire": 0.5, "smoke": 0.7})
    kept = w.filter([det("fire", 0.55), det("smoke", 0.6), det("smoke", 0.75), det("person", 0.99)])
    assert [(d.label, d.score) for d in kept] == [("fire", 0.55), ("smoke", 0.75)]


def test_alert_after_persistent_detection(frame):
    alerts = []
    w = make_worker(None, alerts)
    for dets in [[det("fire", 0.9)], [], [det("fire", 0.8)], [det("fire", 0.7)], [det("fire", 0.9)]]:
        w._process(frame, dets)

    assert len(alerts) == 1  # third hit inside the window, then cooldown
    a = alerts[0]
    assert a.camera == "cam1" and a.labels == ["fire"]
    assert "cam1" in a.caption and "0.70" in a.caption
    assert cv2.imdecode(np.frombuffer(a.jpeg, np.uint8), cv2.IMREAD_COLOR).any()  # boxes drawn


def test_flicker_does_not_alert(frame):
    alerts = []
    w = make_worker(None, alerts)
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
    w = CameraWorker(cfg, detector, alerts.append)
    w.run()

    assert len(detector.calls) == 6
    assert len(alerts) == 1


def test_monitor_dispatches_to_notifier(frame):
    sent = []

    class Notifier:
        def send_photo(self, image, caption):
            sent.append(caption)
            return True

    settings = Settings(
        cameras=[
            CameraConfig(name="a", source="a.mp4", window=1, min_hits=1),
            CameraConfig(name="b", source="b.mp4", enabled=False),
        ],
    )
    m = Monitor(settings, detector=None, notifier=Notifier())
    assert [w.cfg.name for w in m.workers] == ["a"]
    m.workers[0]._process(frame, [det("fire", 0.9)])
    assert len(sent) == 1 and sent[0].startswith("[a] fire")
