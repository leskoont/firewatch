"""Latency / throughput of the onnxruntime pipeline on this machine.

    python scripts/benchmark.py models/firewatch-yolo11s.onnx --providers cpu cuda

Times are per image, batch 1, measured over real test images (decode excluded).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from firewatch.detector import OnnxDetector
from firewatch.ops import letterbox, postprocess, scale_boxes, to_tensor

PROVIDERS = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}


def stats(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {
        "mean": round(float(a.mean()), 2),
        "p50": round(float(np.percentile(a, 50)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
    }


def bench(det: OnnxDetector, images: list[np.ndarray], warmup: int, runs: int) -> dict:
    for i in range(warmup):
        det(images[i % len(images)])

    pre, infer, post, total = [], [], [], []
    for i in range(runs):
        img = images[i % len(images)]
        t0 = time.perf_counter()
        padded, info = letterbox(img, det.input_hw)
        x = to_tensor(padded)
        t1 = time.perf_counter()
        raw = det.session.run(None, {det.input_name: x})[0][0]
        t2 = time.perf_counter()
        boxes = postprocess(raw, det.conf, det.iou)
        scale_boxes(boxes, info)
        t3 = time.perf_counter()
        pre.append((t1 - t0) * 1e3)
        infer.append((t2 - t1) * 1e3)
        post.append((t3 - t2) * 1e3)
        total.append((t3 - t0) * 1e3)

    s = stats(total)
    return {
        "preprocess_ms": stats(pre),
        "inference_ms": stats(infer),
        "postprocess_ms": stats(post),
        "total_ms": s,
        "fps": round(1000 / s["mean"], 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--images", type=Path, default=Path("data/dfire/images/test"))
    ap.add_argument("--providers", nargs="+", default=["cpu", "cuda"], choices=list(PROVIDERS))
    ap.add_argument("--threads", type=int, default=None, help="intra-op threads for CPU")
    ap.add_argument("-n", "--num-images", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    files = sorted(args.images.glob("*.jpg"))[: args.num_images]
    images = [cv2.imread(str(f)) for f in files]
    if not images:
        raise SystemExit(f"no images in {args.images}")

    sizes = np.array([i.shape[:2] for i in images])
    report = {
        "model": args.model.name,
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "images": len(images),
        "mean_image_wh": [int(sizes[:, 1].mean()), int(sizes[:, 0].mean())],
        "results": {},
    }
    available = ort.get_available_providers()
    for name in args.providers:
        prov = PROVIDERS[name]
        if prov not in available:
            print(f"{name}: provider not available, skipped")
            continue
        det = OnnxDetector(args.model, [prov], intra_op_threads=args.threads)
        if det.providers[0] != prov:
            print(f"{name}: session fell back to {det.providers}, skipped")
            continue
        r = bench(det, images, args.warmup, args.runs)
        report["results"][name] = r
        print(
            f"{name:>4}: total {r['total_ms']['mean']:.1f} ms (p95 {r['total_ms']['p95']:.1f}), "
            f"inference {r['inference_ms']['mean']:.1f} ms, pre {r['preprocess_ms']['mean']:.1f} ms, "
            f"post {r['postprocess_ms']['mean']:.2f} ms, {r['fps']} FPS"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
