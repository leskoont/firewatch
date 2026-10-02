from __future__ import annotations

import logging
import os
import re
import time
import uuid
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

CLIP_MAX_WIDTH = 1280
_UNSAFE_RE = re.compile(r'[\\/:*?"<>|\s]+')


def event_stem(camera: str, when: datetime | None = None) -> str:
    when = when or datetime.now()
    camera = _UNSAFE_RE.sub("_", camera).strip("._") or "camera"
    return f"{camera}_{when:%Y%m%d_%H%M%S}_{when.microsecond // 1000:03d}"


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        log.error("jpeg encoding failed")
        return b""
    return buf.tobytes()


def save_snapshot(directory: Path, stem: str, jpeg: bytes) -> Path | None:
    if not jpeg:
        return None
    path = directory / f"{stem}.jpg"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg)
    except OSError as e:
        log.error("cannot save snapshot %s: %s", path, e)
        return None
    return path


# (backend, codec) that opened last time. Probing avc1 through the FFmpeg backend first
# makes Windows builds print OpenH264 errors before OpenCV falls back to Media Foundation,
# so that happens once and later clips go straight to what worked.
_writer_choice: tuple[int, str] | None = None


def _writer_backend(writer: cv2.VideoWriter) -> int:
    name = writer.getBackendName()
    for api in cv2.videoio_registry.getWriterBackends():
        if cv2.videoio_registry.getBackendName(api) == name:
            return api
    return cv2.CAP_ANY


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter | None:
    global _writer_choice
    if _writer_choice is not None:
        api, codec = _writer_choice
        writer = cv2.VideoWriter(str(path), api, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
        _writer_choice = None

    # H.264 plays in browsers and Telegram, mp4v doesn't; not every OpenCV build can encode it
    # (the Linux wheels from PyPI can't, so the Docker image skips avc1 via the env var)
    codecs = os.environ.get("FIREWATCH_CLIP_CODECS", "avc1,mp4v").split(",")
    for codec in filter(None, map(str.strip, codecs)):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            _writer_choice = (_writer_backend(writer), codec)
            return writer
        writer.release()
    return None


def _fit_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


class ClipRecorder:
    """Keeps a short history of processed frames and writes it out, plus what
    follows, when an event starts."""

    def __init__(
        self,
        before_s: float,
        after_s: float,
        clock: Callable[[], float] = time.monotonic,
        max_width: int = CLIP_MAX_WIDTH,
    ):
        self.before_s = before_s
        self.after_s = after_s
        self.max_width = max_width
        self._clock = clock
        self._buffer: deque[tuple[float, np.ndarray]] = deque()
        self._writer: cv2.VideoWriter | None = None
        self._size = (0, 0)
        self._stop_at = 0.0
        self._tmp: Path | None = None
        self.path: Path | None = None

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def push(self, frame: np.ndarray, t: float | None = None) -> None:
        now = self._clock() if t is None else t
        frame = _fit_width(frame, self.max_width)
        if self._writer is not None:
            if frame.shape[:2] == self._size[::-1]:
                self._writer.write(frame)
            if now >= self._stop_at:
                self._finish()
            return
        self._buffer.append((now, frame))
        while self._buffer and now - self._buffer[0][0] > self.before_s:
            self._buffer.popleft()

    def start(self, path: Path) -> bool:
        if self._writer is not None or not self._buffer:
            return False
        h, w = self._buffer[-1][1].shape[:2]
        self._size = (w, h)
        # cv2 can't open non-ASCII paths on Windows: write under an ASCII name, rename when done
        tmp = path.with_name(f".{uuid.uuid4().hex}.mp4")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.error("cannot record clip %s: %s", path, e)
            return False
        writer = open_writer(tmp, self._estimate_fps(), (w, h))
        if writer is None:
            log.error("cannot open video writer for %s", path)
            return False
        for _, frame in self._buffer:
            if frame.shape[:2] == (h, w):
                writer.write(frame)
        self._stop_at = self._buffer[-1][0] + self.after_s
        self._buffer.clear()
        self._writer = writer
        self._tmp, self.path = tmp, path
        return True

    def _estimate_fps(self) -> float:
        if len(self._buffer) < 2:
            return 5.0
        span = self._buffer[-1][0] - self._buffer[0][0]
        return max(1.0, min(30.0, (len(self._buffer) - 1) / span)) if span > 0 else 5.0

    def _finish(self) -> None:
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        try:
            self._tmp.replace(self.path)
            log.info("clip saved: %s", self.path)
        except OSError as e:
            log.error("cannot save clip %s: %s", self.path, e)

    def close(self) -> None:
        self._finish()
