import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from firewatch.api import create_app
from firewatch.detector import OnnxDetector
from firewatch.monitor import CameraStatus


class FakeMonitor:
    def statuses(self):
        return [CameraStatus(name="gate", source="rtsp://10.0.0.5/1", state="running", alerts=2)]


@pytest.fixture
def client(dummy_model):
    det = OnnxDetector(dummy_model, "CPUExecutionProvider")
    return TestClient(create_app(det, FakeMonitor(), model_name="dummy.onnx"))


def png(h=128, w=256):
    ok, buf = cv2.imencode(".png", np.zeros((h, w, 3), np.uint8))
    assert ok
    return buf.tobytes()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "dummy.onnx"
    assert body["classes"] == ["smoke", "fire"]
    assert "CPUExecutionProvider" in body["providers"]


def test_detect(client):
    resp = client.post("/detect", files={"file": ("x.png", png(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert (body["width"], body["height"]) == (256, 128)
    assert [d["label"] for d in body["detections"]] == ["fire", "smoke"]
    assert body["detections"][0]["box"] == [88.0, 24.0, 168.0, 104.0]
    assert body["inference_ms"] >= 0


def test_detect_conf_param(client):
    resp = client.post("/detect?conf=0.8", files={"file": ("x.png", png(), "image/png")})
    assert [d["label"] for d in resp.json()["detections"]] == ["fire"]


def test_detect_rejects_garbage(client):
    resp = client.post("/detect", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert resp.status_code == 400


def test_detect_requires_file(client):
    assert client.post("/detect").status_code == 422


def test_cameras(client):
    cams = client.get("/cameras").json()
    assert cams[0]["name"] == "gate"
    assert cams[0]["alerts"] == 2


def test_cameras_without_monitor(dummy_model):
    det = OnnxDetector(dummy_model, "CPUExecutionProvider")
    assert TestClient(create_app(det)).get("/cameras").json() == []
