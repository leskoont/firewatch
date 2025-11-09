from datetime import datetime

import cv2
import numpy as np

from firewatch.events import encode_jpeg, event_stem, save_snapshot


def test_event_stem():
    assert event_stem("cam1", datetime(2024, 5, 1, 13, 4, 5)) == "cam1_20240501_130405_000"
    assert event_stem("cam1", datetime(2024, 5, 1, 13, 4, 5, 250_000)).endswith("_250")
    assert event_stem("gate 2/left", datetime(2024, 5, 1)).startswith("gate_2_left_")


def test_encode_jpeg():
    assert encode_jpeg(np.zeros((10, 10, 3), np.uint8)).startswith(b"\xff\xd8")


def test_save_snapshot(tmp_path):
    p = save_snapshot(tmp_path / "ev", "x", np.full((10, 10, 3), 200, np.uint8))
    assert p.exists() and p.suffix == ".jpg"
    assert cv2.imread(str(p)).shape == (10, 10, 3)
