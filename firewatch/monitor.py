from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from firewatch.config import CameraConfig, Settings
from firewatch.detector import Detection
from firewatch.draw import draw_detections
from firewatch.notify import TelegramNotifier
from firewatch.smoothing import AlertGate
from firewatch.stream import VideoStream

log = logging.getLogger(__name__)


@dataclass
class Alert:
    camera: str
    labels: list[str]
    detections: list[Detection]
    jpeg: bytes
    time: datetime

    @property
    def caption(self) -> str:
        best: dict[str, float] = {}
        for d in self.detections:
            best[d.label] = max(best.get(d.label, 0.0), d.score)
        found = ", ".join(f"{c} {best[c]:.2f}" if c in best else c for c in self.labels)
        return f"[{self.camera}] {found}\n{self.time:%Y-%m-%d %H:%M:%S}"


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


class CameraWorker(threading.Thread):
    def __init__(self, cfg: CameraConfig, detector, on_alert):
        super().__init__(name=f"camera-{cfg.name}", daemon=True)
        self.cfg = cfg
        self.detector = detector
        self.on_alert = on_alert
        self.stop_event = threading.Event()
        self.gate = AlertGate(cfg.thresholds, cfg.window, cfg.min_hits, cfg.cooldown_s)
        self.stream = VideoStream(
            cfg.source,
            live=cfg.is_live,
            every=cfg.process_every,
            reconnect_delay=cfg.reconnect_delay_s,
        )

    def filter(self, dets: list[Detection]) -> list[Detection]:
        th = self.cfg.thresholds
        return [d for d in dets if d.label in th and d.score >= th[d.label]]

    def run(self) -> None:
        log.info("camera %s: starting (%s)", self.cfg.name, self.cfg.source)
        try:
            for _, frame in self.stream.frames(self.stop_event):
                self._process(frame, self.filter(self.detector(frame)))
        except Exception:
            log.exception("camera %s failed", self.cfg.name)
        log.info("camera %s: stopped", self.cfg.name)

    def _process(self, frame: np.ndarray, dets: list[Detection]) -> None:
        due = self.gate.update(d.label for d in dets)
        if not due:
            return
        shown = [d for d in dets if d.label in due]
        alert = Alert(self.cfg.name, due, shown, encode_jpeg(draw_detections(frame, shown)), datetime.now())
        log.warning("ALERT %s", alert.caption.replace("\n", ", "))
        self.on_alert(alert)

    def stop(self) -> None:
        self.stop_event.set()


class Monitor:
    def __init__(self, settings: Settings, detector, notifier: TelegramNotifier | None = None):
        self.settings = settings
        self.detector = detector
        self.notifier = notifier
        self._send_lock = threading.Lock()
        self.workers = [
            CameraWorker(cam, detector, self._dispatch) for cam in settings.cameras if cam.enabled
        ]

    def _dispatch(self, alert: Alert) -> None:
        if self.notifier is None:
            return
        with self._send_lock:
            if not self.notifier.send_photo(alert.jpeg, alert.caption):
                self.notifier.send_text(alert.caption)

    def start(self) -> None:
        for w in self.workers:
            w.start()

    def stop(self, timeout: float = 5.0) -> None:
        for w in self.workers:
            w.stop()
        for w in self.workers:
            if w.is_alive():
                w.join(timeout)

    def wait(self) -> None:
        for w in self.workers:
            while w.is_alive():
                w.join(0.5)
