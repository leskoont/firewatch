"""Train a detector from a YAML config.

python scripts/train.py                          # configs/train.yaml
python scripts/train.py -c my.yaml epochs=5 batch=8
python scripts/train.py --resume runs/yolo11s-dfire/weights/last.pt batch=8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"bad override {item!r}, expected key=value")
        out[key] = yaml.safe_load(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--resume", type=Path, help="continue an interrupted run from its last.pt")
    ap.add_argument(
        "--keep-cuda-cache",
        action="store_true",
        help="do not release cached VRAM between epochs (works around OOM on Windows/WDDM)",
    )
    ap.add_argument("overrides", nargs="*", help="key=value pairs on top of the config")
    args = ap.parse_args()

    if args.keep_cuda_cache:
        # Under WDDM, with other programs holding VRAM, re-allocating after ultralytics
        # empties the cache at the epoch boundary can fail even though memory is free.
        torch.cuda.empty_cache = lambda: None

    if args.resume:
        # ultralytics restores the arguments from the checkpoint and only lets a few
        # of them (batch, workers, device...) be changed
        model = YOLO(str(args.resume))
        model.train(resume=True, **parse_overrides(args.overrides))
        print(f"best weights: {Path(model.trainer.best)}")
        return

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg.update(parse_overrides(args.overrides))
    # ultralytics resolves relative project dirs against its own runs dir
    cfg["project"] = str(Path(cfg.get("project", "runs")).resolve())

    model = YOLO(cfg.pop("model"))
    model.train(**cfg)
    print(f"best weights: {Path(model.trainer.best)}")


if __name__ == "__main__":
    main()
