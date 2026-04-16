"""Compare the onnxruntime pipeline against ultralytics on real images.

Three checks:
  raw   - PyTorch head output vs ONNX head output on the same letterboxed tensor
  onnx  - ultralytics predict on the .onnx vs firewatch.OnnxDetector (same model,
          tests our pre/postprocessing; scores should match exactly, coordinates
          within about a pixel because ultralytics re-derives the padding with
          its own rounding when scaling boxes back)
  pt    - ultralytics predict on the .pt vs firewatch.OnnxDetector (end to end;
          small differences are expected: ultralytics pads .pt input to a stride
          multiple instead of a fixed square)

    python scripts/check_onnx.py runs/yolo11s-dfire/weights/best.pt models/firewatch-yolo11s.onnx
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from firewatch.detector import OnnxDetector
from firewatch.ops import box_iou, letterbox, to_tensor


def ul_boxes(result) -> np.ndarray:
    b = result.boxes
    if b is None or not len(b):
        return np.zeros((0, 6), np.float32)
    return np.concatenate(
        [b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1
    )


def fw_boxes(dets) -> np.ndarray:
    if not dets:
        return np.zeros((0, 6), np.float32)
    return np.array([[d.x1, d.y1, d.x2, d.y2, d.score, d.class_id] for d in dets], np.float32)


def match(ref: np.ndarray, ours: np.ndarray, iou_thr: float = 0.5):
    """Greedy one-to-one matching by class and IoU. Returns (ious, score_diffs, coord_diffs)."""
    used = set()
    ious, sdiff, cdiff = [], [], []
    for r in ref[np.argsort(-ref[:, 4])]:
        cand = [i for i in range(len(ours)) if i not in used and ours[i, 5] == r[5]]
        if not cand:
            continue
        iou = box_iou(r[:4], ours[cand, :4])
        j = int(iou.argmax())
        if iou[j] >= iou_thr:
            used.add(cand[j])
            ious.append(float(iou[j]))
            sdiff.append(abs(float(r[4] - ours[cand[j], 4])))
            cdiff.append(float(np.abs(r[:4] - ours[cand[j], :4]).max()))
    return ious, sdiff, cdiff


def summarize(name, pairs):
    n_ref = sum(len(r) for r, _ in pairs)
    n_ours = sum(len(o) for _, o in pairs)
    ious, sdiff, cdiff = [], [], []
    for r, o in pairs:
        i, s, c = match(r, o)
        ious += i
        sdiff += s
        cdiff += c
    out = {
        "ref_boxes": n_ref,
        "our_boxes": n_ours,
        "matched": len(ious),
        "mean_iou": round(float(np.mean(ious)), 5) if ious else None,
        "min_iou": round(float(np.min(ious)), 5) if ious else None,
        "max_score_diff": round(float(np.max(sdiff)), 5) if sdiff else None,
        "max_coord_diff_px": round(float(np.max(cdiff)), 3) if cdiff else None,
    }
    print(f"{name:>5}: {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pt", type=Path)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--images", type=Path, default=Path("data/dfire/images/test"))
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    files = sorted(args.images.glob("*.jpg"))
    random.Random(args.seed).shuffle(files)
    files = files[: args.n]

    ours = OnnxDetector(args.onnx, "CPUExecutionProvider", conf=args.conf, iou=args.iou)
    ul_onnx = YOLO(str(args.onnx), task="detect")
    ul_pt = YOLO(str(args.pt))
    torch_model = ul_pt.model.float().eval().fuse()
    kw = dict(conf=args.conf, iou=args.iou, imgsz=640, device="cpu", verbose=False)

    raw_box, raw_cls, pairs_onnx, pairs_pt = [], [], [], []
    for f in files:
        img = cv2.imread(str(f))
        padded, _ = letterbox(img, ours.input_hw)
        x = to_tensor(padded)
        with torch.no_grad():
            ref = torch_model(torch.from_numpy(x))[0].numpy()
        got = ours.session.run(None, {ours.input_name: x})[0]
        d = np.abs(ref - got)[0]
        raw_box.append(float(d[:4].max()))
        raw_cls.append(float(d[4:].max()))

        mine = fw_boxes(ours(img))
        pairs_onnx.append((ul_boxes(ul_onnx.predict(img, **kw)[0]), mine))
        pairs_pt.append((ul_boxes(ul_pt.predict(img, **kw)[0]), mine))

    report = {
        "images": len(files),
        "raw_max_abs_diff_box_px": round(max(raw_box), 6),
        "raw_max_abs_diff_score": round(max(raw_cls), 6),
    }
    print(
        f"  raw: max |torch - onnx| boxes {report['raw_max_abs_diff_box_px']} px, "
        f"scores {report['raw_max_abs_diff_score']}"
    )
    report["onnx"] = summarize("onnx", pairs_onnx)
    report["pt"] = summarize("pt", pairs_pt)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
