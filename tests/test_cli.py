import cv2
import numpy as np
import pytest

from firewatch.cli import main


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
