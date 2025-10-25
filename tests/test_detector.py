import numpy as np
import pytest

from firewatch.detector import OnnxDetector, resolve_providers


def test_reads_metadata(dummy_model):
    det = OnnxDetector(dummy_model, "CPUExecutionProvider")
    assert det.names == {0: "smoke", 1: "fire"}
    assert det.input_hw == (64, 64)
    assert det.providers == ["CPUExecutionProvider"]


def test_detections_in_source_coordinates(dummy_model):
    det = OnnxDetector(dummy_model, "CPUExecutionProvider", conf=0.25, iou=0.45)
    # 128x256 -> ratio 0.25, letterboxed to 64x32 with 16 px padding top and bottom
    img = np.zeros((128, 256, 3), np.uint8)
    dets = det(img)
    assert [d.label for d in dets] == ["fire", "smoke"]

    fire = dets[0]
    assert fire.score == pytest.approx(0.9)
    # head box (22, 22, 42, 42) in input pixels -> minus pad, divided by ratio
    assert (fire.x1, fire.y1, fire.x2, fire.y2) == pytest.approx((88, 24, 168, 104))
    assert fire.to_dict() == {"label": "fire", "class_id": 1, "score": 0.9, "box": [88.0, 24.0, 168.0, 104.0]}


def test_conf_override(dummy_model):
    det = OnnxDetector(dummy_model, "CPUExecutionProvider", conf=0.25)
    img = np.zeros((64, 64, 3), np.uint8)
    assert len(det(img)) == 2
    assert [d.label for d in det(img, conf=0.75)] == ["fire"]
    assert det(img, conf=0.95) == []


def test_names_override(dummy_model):
    det = OnnxDetector(dummy_model, "CPUExecutionProvider", names={0: "dym", 1: "ogon"})
    assert {d.label for d in det(np.zeros((64, 64, 3), np.uint8))} == {"dym", "ogon"}


def test_resolve_providers_falls_back_to_cpu():
    assert resolve_providers(["NoSuchExecutionProvider"]) == ["CPUExecutionProvider"]
    assert "CPUExecutionProvider" in resolve_providers("auto")
