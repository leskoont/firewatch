"""Evaluate weights on a dataset split and save metrics as JSON.

python scripts/evaluate.py runs/yolo11s-dfire/weights/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path)
    ap.add_argument("--data", default="data/dfire/dfire.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, help="JSON file for metrics")
    args = ap.parse_args()

    model = YOLO(str(args.weights), task="detect")
    m = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=False,
        project=str(Path("runs/eval").resolve()),
        name=f"{args.weights.stem}-{args.split}",
        exist_ok=True,
    )

    box = m.box
    result = {
        "weights": str(args.weights),
        "split": args.split,
        "all": {"precision": box.mp, "recall": box.mr, "map50": box.map50, "map50_95": box.map},
        "classes": {},
        "speed_ms": m.speed,
    }
    for i, cls in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        result["classes"][m.names[int(cls)]] = {
            "precision": p,
            "recall": r,
            "map50": ap50,
            "map50_95": ap,
        }

    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
