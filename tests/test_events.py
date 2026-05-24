from datetime import datetime

import cv2
import numpy as np
import pytest

from firewatch.events import ClipRecorder, encode_jpeg, event_stem, save_snapshot


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_event_stem():
    assert event_stem("cam1", datetime(2024, 5, 1, 13, 4, 5)) == "cam1_20240501_130405_000"
    assert event_stem("cam1", datetime(2024, 5, 1, 13, 4, 5, 250_000)).endswith("_250")


@pytest.mark.parametrize(
    ("name", "prefix"),
    [("gate 2/left", "gate_2_left_"), ('a:b*c?"d"', "a_b_c_d_"), ("c:\\cams\\x", "c_cams_x_")],
)
def test_event_stem_is_a_safe_file_name(name, prefix):
    assert event_stem(name, datetime(2024, 5, 1)).startswith(prefix)


def test_save_snapshot(tmp_path):
    jpeg = encode_jpeg(np.zeros((10, 10, 3), np.uint8))
    p = save_snapshot(tmp_path / "ev", "x", jpeg)
    assert p.exists() and p.suffix == ".jpg" and p.read_bytes() == jpeg
    assert save_snapshot(tmp_path / "ev", "y", b"") is None


def test_clip_recorder_keeps_pre_buffer_and_stops(tmp_path):
    clock = Clock()
    rec = ClipRecorder(before_s=1.0, after_s=1.0, clock=clock)
    for i in range(30):  # 3 s at 10 fps, only the last second is kept
        clock.t = i / 10
        rec.push(np.full((48, 64, 3), i, np.uint8))
    assert 10 <= len(rec._buffer) <= 11

    path = tmp_path / "clip.mp4"
    rec.start(path)
    assert rec.recording
    for i in range(30, 45):
        clock.t = i / 10
        rec.push(np.full((48, 64, 3), i, np.uint8))
    assert not rec.recording

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert 19 <= n <= 22  # ~1 s buffered + ~1 s after the trigger
    assert round(fps) == 10


def test_clip_recorder_downscales_wide_frames(tmp_path):
    rec = ClipRecorder(1.0, 0.0, clock=Clock(), max_width=320)
    rec.push(np.zeros((720, 1280, 3), np.uint8))
    assert rec._buffer[0][1].shape == (180, 320, 3)


def test_start_without_frames_is_noop(tmp_path):
    rec = ClipRecorder(1.0, 1.0, clock=Clock())
    rec.start(tmp_path / "x.mp4")
    assert not rec.recording
