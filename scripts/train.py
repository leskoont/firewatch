"""Train a detector from a YAML config.

python scripts/train.py                          # configs/train.yaml
python scripts/train.py -c my.yaml epochs=5 batch=8
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    ap.add_argument("overrides", nargs="*", help="key=value pairs on top of the config")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg.update(parse_overrides(args.overrides))
    # ultralytics resolves relative project dirs against its own runs dir
    cfg["project"] = str(Path(cfg.get("project", "runs")).resolve())

    model = YOLO(cfg.pop("model"))
    model.train(**cfg)
    print(f"best weights: {Path(model.trainer.best)}")


if __name__ == "__main__":
    main()
