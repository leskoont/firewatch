"""Export trained weights to ONNX for the onnxruntime-only runtime.

python scripts/export_onnx.py runs/yolo11s-dfire/weights/best.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out", type=Path, default=Path("models/firewatch-yolo11s.onnx"))
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    # fixed 1x3xHxW input, NMS stays outside the graph (done in firewatch.ops)
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
        half=False,
        nms=False,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if Path(exported).resolve() != args.out.resolve():
        shutil.copy2(exported, args.out)
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
