import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from firewatch.config import CameraConfig, expand_env, load_settings

EXAMPLE = Path(__file__).parent.parent / "configs" / "cameras.example.yaml"


def write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


def test_expand_env(monkeypatch):
    monkeypatch.setenv("FW_TOKEN", "abc")
    monkeypatch.delenv("FW_MISSING", raising=False)
    data = {"a": "${FW_TOKEN}", "b": ["x${FW_MISSING}y", "${FW_MISSING:-dflt}"], "c": 5}
    assert expand_env(data) == {"a": "abc", "b": ["xy", "dflt"], "c": 5}


def test_empty_env_takes_default(monkeypatch, caplog):
    monkeypatch.setenv("FW_EMPTY", "")
    monkeypatch.delenv("FW_MISSING", raising=False)
    assert expand_env("${FW_EMPTY:-x}") == "x"
    assert expand_env("${FW_MISSING}") == ""
    assert "FW_MISSING" in caplog.text


def test_defaults_are_merged_and_overridden(tmp_path):
    cfg = load_settings(
        write(
            tmp_path,
            """
        camera_defaults:
          process_every: 5
          cooldown_s: 120
        cameras:
          - name: a
            source: rtsp://cam/1
          - name: b
            source: 0
            process_every: 1
    """,
        )
    )
    a, b = cfg.cameras
    assert (a.process_every, a.cooldown_s) == (5, 120)
    assert (b.process_every, b.cooldown_s) == (1, 120)
    assert b.source == "0" and b.is_live
    assert a.thresholds == {"fire": 0.4, "smoke": 0.5}


def test_secrets_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TG_TOKEN", "123:abc")
    monkeypatch.setenv("TG_CHAT", "-100500")
    cfg = load_settings(
        write(
            tmp_path,
            """
        telegram:
          enabled: true
          token: ${TG_TOKEN}
          chat_id: ${TG_CHAT}
    """,
        )
    )
    assert cfg.telegram.enabled
    assert cfg.telegram.token == "123:abc"
    assert cfg.telegram.chat_id == "-100500"


def test_telegram_enabled_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    path = write(
        tmp_path,
        """
        telegram:
          enabled: true
          token: ${NOPE}
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


def test_example_config_parses(monkeypatch):
    monkeypatch.delenv("TELEGRAM_ENABLED", raising=False)
    cfg = load_settings(EXAMPLE)
    assert not cfg.telegram.enabled
    assert [c.name for c in cfg.cameras] == ["warehouse", "demo-video", "webcam"]
    assert cfg.cameras[1].process_every == 2 and cfg.cameras[1].loop


def test_thresholds_are_merged_per_class(tmp_path):
    cfg = load_settings(
        write(
            tmp_path,
            """
        camera_defaults:
          thresholds: {fire: 0.4, smoke: 0.5}
        cameras:
          - {name: a, source: a.mp4, thresholds: {smoke: 0.7}}
          - {name: b, source: b.mp4, thresholds: {smoke: null}}
    """,
        )
    )
    a, b = cfg.cameras
    assert a.thresholds == {"fire": 0.4, "smoke": 0.7}
    assert b.thresholds == {"fire": 0.4}
