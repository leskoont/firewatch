import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from firewatch.config import CameraConfig, load_settings

EXAMPLE = Path(__file__).parent.parent / "configs" / "cameras.example.yaml"


def write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


def test_telegram_enabled_without_token(tmp_path):
    path = write(
        tmp_path,
        """
        telegram:
          enabled: true
          token: ""
          chat_id: "1"
    """,
    )
    with pytest.raises(ValidationError, match="token"):
        load_settings(path)


def test_min_hits_above_window():
    with pytest.raises(ValidationError, match="min_hits"):
        CameraConfig(name="x", source="a.mp4", window=3, min_hits=5)


@pytest.mark.parametrize("thresholds", [{"fire": 40}, {"smoke": -0.1}])
def test_bad_thresholds(thresholds):
    with pytest.raises(ValidationError):
        CameraConfig(name="x", source="a.mp4", thresholds=thresholds)


def test_numeric_chat_id(tmp_path):
    cfg = load_settings(write(tmp_path, "telegram: {chat_id: -100500}"))
    assert cfg.telegram.chat_id == "-100500"


def test_duplicate_camera_names(tmp_path):
    path = write(
        tmp_path,
        """
        cameras:
          - {name: a, source: a.mp4}
          - {name: a, source: b.mp4}
    """,
    )
    with pytest.raises(ValidationError, match="duplicate"):
        load_settings(path)


@pytest.mark.parametrize(
    ("source", "live"),
    [("0", True), ("rtsp://u:p@host/s", True), ("http://host/mjpg", True), ("videos/a.mp4", False)],
)
def test_is_live(source, live):
    assert CameraConfig(name="x", source=source).is_live is live


def test_numeric_source(tmp_path):
    cfg = load_settings(write(tmp_path, "cameras: [{name: a, source: 0}]"))
    assert cfg.cameras[0].source == "0"


def test_example_config_parses():
    cfg = load_settings(EXAMPLE)
    assert not cfg.telegram.enabled
    assert [c.name for c in cfg.cameras] == ["warehouse", "demo-video", "webcam"]
    assert cfg.cameras[1].process_every == 2
