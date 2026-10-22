"""Fetch the exported ONNX model from GitHub Releases into models/."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import httpx

REPO = "leskoont/firewatch"
ASSET = "firewatch-yolo11s.onnx"
DEFAULT_TAG = "v0.1.0"
SHA256 = {
    "v0.1.0": "1ab25ae6a26e711542e5b5e9670611476719532bc00ef1bd1a1657c01f24c2b4",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=DEFAULT_TAG, help="release tag, or 'latest'")
    ap.add_argument("--out", type=Path, default=Path("models") / ASSET)
    args = ap.parse_args()

    if args.tag == "latest":
        url = f"https://github.com/{REPO}/releases/latest/download/{ASSET}"
    else:
        url = f"https://github.com/{REPO}/releases/download/{args.tag}/{ASSET}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".part")
    try:
        digest, tag = fetch(url, tmp)
        expected = SHA256.get(tag or args.tag)
        if expected and digest != expected:
            raise SystemExit(f"checksum mismatch for {url}\n  expected {expected}\n  got      {digest}")
        tmp.replace(args.out)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    if not expected:
        print(f"WARNING: no known checksum for {tag or args.tag}, sha256 {digest}")


def fetch(url: str, dest: Path) -> tuple[str, str | None]:
    """Download to dest, return the sha256 and the release tag the URL resolved to."""
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as resp:
        if resp.status_code == 404:
            raise SystemExit(f"not found: {url}\nsee https://github.com/{REPO}/releases")
        resp.raise_for_status()
        sha = hashlib.sha256()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(1 << 20):
                f.write(chunk)
                sha.update(chunk)
    # releases/latest redirects to releases/download/<tag>/..., which tells us what to verify against
    tag = None
    for r in resp.history:
        if m := re.search(r"/releases/download/([^/]+)/", r.headers.get("location", "")):
            tag = m.group(1)
    return sha.hexdigest(), tag


if __name__ == "__main__":
    main()
