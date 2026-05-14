import struct
import zlib

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from firewatch import api
from firewatch.api import create_app
from firewatch.detector import OnnxDetector
from firewatch.monitor import CameraStatus


class FakeMonitor:
    def __init__(self):
        self.stopped = 0

    def statuses(self):
        return [CameraStatus(name="gate", source="rtsp://10.0.0.5/1", state="running", alerts=2)]

    def stop(self):
        self.stopped += 1


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


def test_monitor_is_stopped_on_shutdown(dummy_model):
    det = OnnxDetector(dummy_model, "CPUExecutionProvider")
    monitor = FakeMonitor()
    with TestClient(create_app(det, monitor)) as c:
        assert c.get("/health").status_code == 200
        assert monitor.stopped == 0
    assert monitor.stopped == 1


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


@pytest.mark.parametrize("size", [(8000, 6000), (30000, 30000)])
def test_detect_rejects_huge_resolution(client, size):
    # a valid PNG header that claims a huge size, never decoded
    data = bytearray(png(8, 8))
    data[16:24] = struct.pack(">II", *size)
    data[29:33] = struct.pack(">I", zlib.crc32(bytes(data[12:29])))
    resp = client.post("/detect", files={"file": ("x.png", bytes(data), "image/png")})
    assert resp.status_code == 413


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 1000)
    resp = client.post("/detect", files={"file": ("x.png", b"0" * 2000, "image/png")})
    assert resp.status_code == 413

    # chunked, no Content-Length: counted while it arrives
    def chunked(payload):
        body = (
            b'--xx\r\nContent-Disposition: form-data; name="file"; filename="x.png"\r\n'
            b"Content-Type: image/png\r\n\r\n" + payload + b"\r\n--xx--\r\n"
        )
        headers = {"content-type": "multipart/form-data; boundary=xx"}
        return client.post("/detect", content=iter([body[:300], body[300:]]), headers=headers)

    assert chunked(b"0" * 2000).status_code == 413
    assert chunked(b"0" * 100).status_code == 400  # small enough, just not an image


def test_limit_is_only_on_detect(client):
    assert client.post("/health").status_code == 405


def test_detect_requires_file(client):
    assert client.post("/detect").status_code == 422


def test_cameras(client):
    cams = client.get("/cameras").json()
    assert cams[0]["name"] == "gate"
    assert cams[0]["alerts"] == 2


def test_cameras_without_monitor(dummy_model):
    det = OnnxDetector(dummy_model, "CPUExecutionProvider")
    assert TestClient(create_app(det)).get("/cameras").json() == []
