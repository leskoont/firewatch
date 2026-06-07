"""Render a few test-split predictions for the README.

python scripts/render_examples.py models/firewatch-yolo11s.onnx --pick AoF07841 AoF07976 AoF07834 AoF07977
python scripts/render_examples.py models/firewatch-yolo11s.onnx      # choose automatically
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from firewatch.detector import OnnxDetector
from firewatch.draw import draw_detections


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--images", type=Path, default=Path("data/dfire/images/test"))
    ap.add_argument("--out", type=Path, default=Path("docs"))
    ap.add_argument("--pick", nargs="*", help="image stems to render")
    # PublicDataset* images come from third-party sources with their own terms,
    # AoF* are the D-Fire authors' own cameras
    ap.add_argument("--prefix", default="AoF", help="filename prefix for automatic choice")
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    det = OnnxDetector(args.model, conf=args.conf)
    if args.pick:
        files = [args.images / f"{s}.jpg" for s in args.pick]
    else:
        # prefer images where the model sees both classes, then one class only
        files = sorted(args.images.glob(f"{args.prefix}*.jpg"))
        random.Random(args.seed).shuffle(files)
        both, single = [], []
        for f in files[:400]:
            labels = {d.label for d in det(cv2.imread(str(f)))}
            if len(labels) == 2:
                both.append(f)
            elif labels:
                single.append(f)
        files = (both[: args.n // 2] + single)[: args.n]

    args.out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        dets = det(img)
        out = draw_detections(img, dets)
        h, w = out.shape[:2]
        if w > args.width:
            out = cv2.resize(out, (args.width, round(h * args.width / w)), interpolation=cv2.INTER_AREA)
        path = args.out / f"example_{i}.jpg"
        cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(f"{path}: {f.stem} -> {', '.join(f'{d.label} {d.score:.2f}' for d in dets) or 'nothing'}")


if __name__ == "__main__":
    main()
