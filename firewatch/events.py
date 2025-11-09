from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

_UNSAFE_RE = re.compile(r'[\\/:*?"<>|\s]+')


def event_stem(camera: str, when: datetime | None = None) -> str:
    when = when or datetime.now()
    camera = _UNSAFE_RE.sub("_", camera)
    return f"{camera}_{when:%Y%m%d_%H%M%S}_{when.microsecond // 1000:03d}"


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        log.error("jpeg encoding failed")
        return b""
    return buf.tobytes()


def save_snapshot(directory: Path, stem: str, image: np.ndarray) -> Path | None:
    path = directory / f"{stem}.jpg"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error("cannot save snaphot %s: %s", path, e)
        return None
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        log.error("cannot save snaphot %s", path)
        return None
    return path
