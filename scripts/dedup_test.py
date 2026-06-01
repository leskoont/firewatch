"""How much of the D-Fire test split is a near-copy of train, and what the model scores
without those images.

D-Fire has many frames cut from the same videos, and the official test split is not
separated by video. A test image counts as unique when its dHash differs from every
train image in at least --min-dist bits (of 64).

python scripts/dedup_test.py models/firewatch-yolo11s.onnx --out docs/dedup_test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou

from firewatch.detector import OnnxDetector

NAMES = {0: "smoke", 1: "fire"}
# the defaults of CameraConfig.thresholds
THRESHOLDS = {"fire": 0.40, "smoke": 0.50}
IOUV = np.linspace(0.5, 0.95, 10)


def dhash(path: Path) -> int:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    return int(np.packbits(g[:, 1:] > g[:, :-1]).view(">u8")[0])


def min_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(len(a), np.uint8)
    for i in range(0, len(a), 512):
        out[i : i + 512] = np.bitwise_count(a[i : i + 512, None] ^ b[None, :]).min(1)
    return out


def hashes(images: Path, split: str) -> tuple[list[Path], np.ndarray]:
    files = sorted((images / split).glob("*.jpg"))
    return files, np.array([dhash(f) for f in files], np.uint64)


def load_labels(path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    lab = np.loadtxt(path, ndmin=2).reshape(-1, 5) if path.exists() else np.zeros((0, 5))
    cx, cy, bw, bh = lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return lab[:, 0].astype(int), boxes


def match(gt_cls, gt_boxes, pred) -> np.ndarray:
    # same greedy matching as ultralytics DetectionValidator
    tp = np.zeros((len(pred), len(IOUV)), bool)
    if not len(pred) or not len(gt_cls):
        return tp
    iou = box_iou(torch.tensor(gt_boxes, dtype=torch.float32), torch.tensor(pred[:, :4])).numpy()
    iou *= gt_cls[:, None] == pred[:, 5]
    for i, thr in enumerate(IOUV):
        m = np.argwhere(iou >= thr)
        if len(m) > 1:
            m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
        tp[m[:, 1], i] = True
    return tp


def metrics(rows: list[dict]) -> dict:
    tp, conf, pcls, tcls = (np.concatenate([r[k] for r in rows]) for k in ("tp", "conf", "pcls", "tcls"))
    res = ap_per_class(tp, conf, pcls, tcls, names=NAMES)
    p, r, ap, classes = res[2], res[3], res[5], res[6]
    out = {
        "images": len(rows),
        "all": {"precision": p.mean(), "recall": r.mean(), "map50": ap[:, 0].mean(), "map50_95": ap.mean()},
    }
    for i, c in enumerate(classes):
        out[NAMES[int(c)]] = {"precision": p[i], "recall": r[i], "map50": ap[i, 0], "map50_95": ap[i].mean()}

    # what a camera would do with one frame: alert or not, at the default thresholds
    op = {"thresholds": THRESHOLDS}
    for name in ("fire", "smoke"):
        with_gt = [r for r in rows if name in r["gt"]]
        hit = sum(r["max"][name] >= THRESHOLDS[name] for r in with_gt)
        op[f"{name}_image_recall"] = hit / len(with_gt)
        op[f"{name}_images"] = len(with_gt)
    background = [r for r in rows if not r["gt"]]
    false_alarms = sum(any(r["max"][n] >= t for n, t in THRESHOLDS.items()) for r in background)
    op["background_images"] = len(background)
    op["background_fp_rate"] = false_alarms / len(background)
    out["operating_point"] = op
    return out


def box_sizes(rows: list[dict]) -> dict:
    out = {}
    for cid, name in NAMES.items():
        sizes = np.concatenate([r["sizes"][r["tcls"] == cid] for r in rows])
        out[name] = {
            "boxes": len(sizes),
            "median_px": float(np.median(sizes)),
            "under_32px": float((sizes < 32).mean()),
            "under_20px": float((sizes < 20).mean()),
        }
    return out


def rounded(obj):
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, float | np.floating):
        return round(float(obj), 3)
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/dfire"))
    ap.add_argument("--min-dist", type=int, default=7)
    ap.add_argument("--providers", default="auto")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    images = args.data / "images"
    train_files, train_h = hashes(images, "train")
    test_files, test_h = hashes(images, "test")
    _, val_h = hashes(images, "val")
    test_d = min_distance(test_h, train_h)
    val_d = min_distance(val_h, train_h)

    leakage = {}
    for split, d in (("test", test_d), ("val", val_d)):
        leakage[split] = {f"dist<={t}": float((d <= t).mean()) for t in (0, 2, 4, 6)}
    print(json.dumps(rounded(leakage), indent=2))

    det = OnnxDetector(args.model, args.providers)
    rows = []
    for f, dist in zip(test_files, test_d, strict=True):
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        gt_cls, gt_boxes = load_labels(args.data / "labels/test" / f"{f.stem}.txt", w, h)
        dets = det(img, conf=0.001, iou=0.7)
        pred = np.array([[d.x1, d.y1, d.x2, d.y2, d.score, d.class_id] for d in dets], np.float32)
        pred = pred.reshape(-1, 6)
        # box size as the network sees it, after letterboxing to 640
        scale = 640 / max(w, h)
        rows.append(
            {
                "unique": dist >= args.min_dist,
                "tp": match(gt_cls, gt_boxes, pred),
                "conf": pred[:, 4],
                "pcls": pred[:, 5],
                "tcls": gt_cls,
                "gt": {NAMES[c] for c in gt_cls},
                "max": {n: max((d.score for d in dets if d.label == n), default=0.0) for n in THRESHOLDS},
                "sizes": np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0], gt_boxes[:, 3] - gt_boxes[:, 1]) * scale,
            }
        )

    unique = [r for r in rows if r["unique"]]
    result = rounded(
        {
            "model": args.model.name,
            "min_dist": args.min_dist,
            "leakage": leakage,
            "test_full": metrics(rows),
            "test_unique": metrics(unique),
            "boxes_full": box_sizes(rows),
            "boxes_unique": box_sizes(unique),
            "unique_images": [f.stem for f, r in zip(test_files, rows, strict=True) if r["unique"]],
        }
    )
    print(json.dumps({k: v for k, v in result.items() if k != "unique_images"}, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
