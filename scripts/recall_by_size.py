# usage: python scripts/recall_by_size.py models/firewatch-yolo11s.onnx --out docs/recall_by_size.json
"""Box recall on the test split by box size, at the default per-camera thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from firewatch.detector import OnnxDetector
from firewatch.ops import box_iou

NAMES = {0: "smoke", 1: "fire"}
THRESHOLDS = {"fire": 0.40, "smoke": 0.50}  # CameraConfig defaults
BINS = [(0, 20), (20, 32), (32, 64), (64, 10_000)]


def load_labels(path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    lab = np.loadtxt(path, ndmin=2).reshape(-1, 5) if path.exists() else np.zeros((0, 5))
    cx, cy, bw, bh = lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h
    return lab[:, 0].astype(int), np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def found(gt_boxes: np.ndarray, pred: np.ndarray, iou_thr: float) -> np.ndarray:
    """Which ground-truth boxes get a prediction of their own (greedy, best IoU first)."""
    hit = np.zeros(len(gt_boxes), bool)
    if not len(pred) or not len(gt_boxes):
        return hit
    iou = np.stack([box_iou(g, pred) for g in gt_boxes])
    used = set()
    for flat in np.argsort(iou, axis=None)[::-1]:
        g, p = divmod(int(flat), iou.shape[1])
        if iou[g, p] < iou_thr:
            break
        if not hit[g] and p not in used:
            hit[g] = True
            used.add(p)
    return hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/dfire"))
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--providers", default="auto")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    det = OnnxDetector(args.model, args.providers, conf=min(THRESHOLDS.values()))
    sizes = {n: [] for n in NAMES.values()}
    hits = {n: [] for n in NAMES.values()}
    for f in sorted((args.data / "images/test").glob("*.jpg")):
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        gt_cls, gt_boxes = load_labels(args.data / "labels/test" / f"{f.stem}.txt", w, h)
        dets = det(img)
        scale = 640 / max(w, h)  # size as the network sees it after letterboxing
        for cid, name in NAMES.items():
            gt = gt_boxes[gt_cls == cid]
            kept = [d for d in dets if d.class_id == cid and d.score >= THRESHOLDS[name]]
            pred = np.array([(d.x1, d.y1, d.x2, d.y2) for d in kept], np.float32).reshape(-1, 4)
            sizes[name].append(np.maximum(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]) * scale)
            hits[name].append(found(gt, pred, args.iou))

    result = {"model": args.model.name, "thresholds": THRESHOLDS, "iou": args.iou}
    result["size"] = "longer side in px, after letterboxing to 640"
    for name in NAMES.values():
        s, ok = np.concatenate(sizes[name]), np.concatenate(hits[name])
        rows = {"all": {"boxes": len(s), "recall": round(float(ok.mean()), 3)}}
        for lo, hi in BINS:
            sel = (s >= lo) & (s < hi)
            key = f"{lo}+" if hi == BINS[-1][1] else f"{lo}-{hi}"
            rows[key] = {"boxes": int(sel.sum()), "recall": round(float(ok[sel].mean()), 3)}
        result[name] = rows
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
