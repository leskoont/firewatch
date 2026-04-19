from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

from firewatch.config import Settings, load_settings
from firewatch.detector import OnnxDetector
from firewatch.draw import draw_detections

log = logging.getLogger("firewatch")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def require_model(path: Path) -> None:
    if not path.exists():
        sys.exit(f"model not found: {path}")


def load_config(path: str) -> Settings:
    try:
        return load_settings(path)
    except (OSError, ValueError, yaml.YAMLError) as e:
        sys.exit(f"{path}: {e}")


def build_detector(settings: Settings) -> OnnxDetector:
    m = settings.model
    require_model(m.path)
    det = OnnxDetector(m.path, m.providers, conf=m.conf, iou=m.iou, intra_op_threads=m.threads)
    log.info("model %s, providers %s, classes %s", m.path.name, det.providers, det.names)
    return det


def cmd_run(args) -> None:
    from firewatch.monitor import Monitor
    from firewatch.notify import TelegramNotifier

    settings = load_config(args.config)
    if not any(c.enabled for c in settings.cameras):
        sys.exit("no enabled cameras in config")
    detector = build_detector(settings)

    tg = settings.telegram
    notifier = TelegramNotifier(tg.token, tg.chat_id, tg.timeout) if tg.enabled else None
    if notifier is None:
        log.info("telegram disabled, alerts go to the log only")

    monitor = Monitor(settings, detector, notifier)
    monitor.start()
    signal.signal(signal.SIGTERM, lambda *_: monitor.stop())
    try:
        monitor.wait()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
    if notifier:
        notifier.close()


def cmd_detect(args) -> None:
    require_model(Path(args.model))
    detector = OnnxDetector(args.model, args.providers, conf=args.conf, iou=args.iou)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for p in map(Path, args.inputs):
        if p.is_dir():
            paths += sorted(x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS)
        else:
            paths.append(p)

    used: set[str] = set()
    for p in paths:
        # imread/imwrite choke on non-ASCII paths on Windows
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) if p.is_file() else None
        if img is None:
            log.warning("skip %s: cannot read", p)
            continue
        dets = detector(img)

        ext = p.suffix if p.suffix.lower() in IMAGE_EXTS else ".jpg"
        name, n = p.stem + ext, 1
        while name in used:
            name, n = f"{p.stem}_{n}{ext}", n + 1
        used.add(name)
        ok, buf = cv2.imencode(ext, draw_detections(img, dets))
        if ok:
            (out_dir / name).write_bytes(buf.tobytes())

        summary = ", ".join(f"{d.label} {d.score:.2f}" for d in dets) or "-"
        print(f"{p} -> {name}: {summary}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="firewatch")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="watch cameras from config, send alerts")
    p.add_argument("-c", "--config", default="configs/cameras.yaml")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("detect", help="run the model on images and save annotated copies")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-m", "--model", default="models/firewatch-yolo11s.onnx")
    p.add_argument("-o", "--output", default="runs/detect")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--providers", default="auto")
    p.set_defaults(func=cmd_detect)

    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    load_dotenv(".env")
    args.func(args)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


if __name__ == "__main__":
    main()
