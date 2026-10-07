import logging

import cv2
import numpy as np
import pytest

from firewatch.cli import main, setup_logging


def test_detect_output_without_suffix(dummy_model, tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    ok, buf = cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))
    (src / "frame").write_bytes(buf.tobytes())  # e.g. a dump from a camera, no extension
    (src / "b.png").write_bytes(buf.tobytes())

    out = tmp_path / "out"
    main(["detect", str(src / "frame"), str(src / "b.png"), "-m", str(dummy_model), "-o", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["b.png", "frame.jpg"]


def test_run_without_enabled_cameras_and_api(dummy_model, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    model = dummy_model.as_posix()
    cfg.write_text(f"model: {{path: '{model}'}}\ncameras: [{{name: a, source: a.mp4, enabled: false}}]\n")
    with pytest.raises(SystemExit, match="all cameras are disabled"):
        main(["run", "-c", str(cfg), "--no-api"])


def png_bytes():
    ok, buf = cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))
    return buf.tobytes()


def test_detect_names_differ_by_more_than_case(dummy_model, tmp_path):
    a, b = tmp_path / "a" / "Fire.png", tmp_path / "b" / "fire.png"
    for path in (a, b):
        path.parent.mkdir()
        path.write_bytes(png_bytes())

    out = tmp_path / "out"
    main(["detect", str(a), str(b), "-m", str(dummy_model), "-o", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["Fire.png", "fire_1.png"]


def test_detect_expands_wildcards(dummy_model, tmp_path, capsys):
    # what cmd.exe and PowerShell hand over: the pattern itself
    for name in ("a.png", "b.png", "notes.txt"):
        (tmp_path / name).write_bytes(png_bytes())
    out = tmp_path / "out"
    main(["detect", str(tmp_path / "*.png"), "-m", str(dummy_model), "-o", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["a.png", "b.png"]


def test_detect_encode_failure_is_not_counted(dummy_model, tmp_path, monkeypatch):
    (tmp_path / "a.png").write_bytes(png_bytes())
    monkeypatch.setattr(cv2, "imencode", lambda ext, img: (False, None))
    with pytest.raises(SystemExit, match="no images processed"):
        main(["detect", str(tmp_path / "a.png"), "-m", str(dummy_model), "-o", str(tmp_path / "out")])
    assert not any((tmp_path / "out").iterdir())


@pytest.mark.parametrize("pattern", ["*.jpg", "missing.png"])
def test_detect_fails_when_nothing_was_processed(dummy_model, tmp_path, pattern):
    with pytest.raises(SystemExit, match="no images processed"):
        main(["detect", str(tmp_path / pattern), "-m", str(dummy_model), "-o", str(tmp_path / "out")])


def test_setup_logging_quiets_httpx():
    setup_logging()
    assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)
