import cv2
import numpy as np

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
