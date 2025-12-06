"""Download D-Fire and convert it into an Ultralytics YOLO dataset.

D-Fire (https://github.com/gaiasd/DFireDataset, CC0 1.0) is distributed through
OneDrive, which does not allow scripted downloads. We take the same images and
labels from a parquet mirror on the Hugging Face Hub that keeps the original
train/test split (17221 / 4306 images). Validation is carved out of train
deterministically, the original test split is left untouched.

Layout produced:
    data/dfire/images/{train,val,test}/*.jpg
    data/dfire/labels/{train,val,test}/*.txt
    data/dfire/dfire.yaml
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import cv2
import httpx
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

HF_REPO = "badsaarow/d-fire"
HF_API = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/data"
HF_FILE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/data/{{name}}"

NAMES = {0: "smoke", 1: "fire"}


def list_shards(client: httpx.Client) -> list[tuple[str, int]]:
    resp = client.get(HF_API)
    resp.raise_for_status()
    return [
        (Path(item["path"]).name, item["size"]) for item in resp.json() if item["path"].endswith(".parquet")
    ]


def download(client: httpx.Client, name: str, size: int, dest: Path) -> Path:
    path = dest / name
    if path.exists() and path.stat().st_size == size:
        return path
    tmp = path.with_suffix(".part")
    with client.stream("GET", HF_FILE.format(name=name)) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f, tqdm(total=size, unit="B", unit_scale=True, desc=name) as bar:
            for chunk in resp.iter_bytes(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    if tmp.stat().st_size != size:
        raise RuntimeError(f"{name}: size mismatch after download")
    tmp.replace(path)
    return path


def is_val(stem: str, fraction: float) -> bool:
    h = int(hashlib.md5(stem.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < fraction


def clean_labels(text: str) -> tuple[list[str], int]:
    """Validate YOLO lines, clip coordinates. Returns (lines, dropped)."""
    lines, dropped = [], 0
    for raw in (text or "").splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
        except (ValueError, IndexError):
            dropped += 1
            continue
        if cls not in NAMES or len(parts) != 5:
            dropped += 1
            continue
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
            dropped += 1
            continue
        lines.append(f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return lines, dropped


def decode(image_bytes: bytes, max_size: int) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None or not max_size:
        return img
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def convert(shards: list[Path], out: Path, val_fraction: float, max_size: int) -> Counter:
    stats: Counter = Counter()
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for shard in shards:
        source_split = "test" if shard.name.startswith("test") else "train"
        pf = pq.ParquetFile(shard)
        batches = pf.iter_batches(batch_size=256)
        for batch in tqdm(batches, desc=shard.name, total=-(-pf.metadata.num_rows // 256)):
            for row in batch.to_pylist():
                stem = Path(row["filename"]).stem
                split = source_split
                if split == "train" and is_val(stem, val_fraction):
                    split = "val"

                img_path = out / "images" / split / f"{stem}.jpg"
                if not img_path.exists():
                    img = decode(row["image"]["bytes"], max_size)
                    if img is None:
                        stats["bad_image"] += 1
                        continue
                    cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                lines, dropped = clean_labels(row["label"])
                (out / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
                stats[f"{split}_images"] += 1
                stats["dropped_boxes"] += dropped
                if not lines:
                    stats[f"{split}_background"] += 1
                for line in lines:
                    stats[f"{split}_{NAMES[int(line[0])]}"] += 1
    return stats


def write_yaml(out: Path) -> Path:
    path = out / "dfire.yaml"
    names = "\n".join(f"  {k}: {v}" for k, v in NAMES.items())
    path.write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/dfire"), help="where parquet shards go")
    ap.add_argument("--out", type=Path, default=Path("data/dfire"))
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument(
        "--max-size",
        type=int,
        default=640,
        help="downscale so the long side is at most this (0 keeps originals); speeds up training a lot",
    )
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    args.raw.mkdir(parents=True, exist_ok=True)
    if args.skip_download:
        shards = sorted(args.raw.glob("*.parquet"))
    else:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            shards = [download(client, n, s, args.raw) for n, s in list_shards(client)]
    if not shards:
        raise SystemExit("no parquet shards found")

    stats = convert(sorted(shards), args.out, args.val_fraction, args.max_size)
    yaml_path = write_yaml(args.out)
    for key in sorted(stats):
        print(f"{key:>20}: {stats[key]}")
    print(f"dataset config: {yaml_path}")


if __name__ == "__main__":
    main()
