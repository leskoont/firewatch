from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from firewatch.config import CameraConfig, Settings, StorageConfig
from firewatch.detector import Detection
from firewatch.draw import draw_detections
from firewatch.events import encode_jpeg, event_stem, save_snapshot
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


class CameraWorker(threading.Thread):
    def __init__(self, cfg: CameraConfig, detector, storage: StorageConfig, on_alert):
        super().__init__(name=f"camera-{cfg.name}", daemon=True)
        self.cfg = cfg
        self.detector = detector
        self.storage = storage
        self.on_alert = on_alert
        self.stop_event = threading.Event()
        self.gate = AlertGate(cfg.thresholds, cfg.window, cfg.min_hits, cfg.cooldown_s)
        self.stream = VideoStream(
            cfg.source,
            live=cfg.is_live,
            every=cfg.process_every,
            reconnect_delay=cfg.reconnect_delay_s,
        )
        self.frames_processed = 0
        self._last_hit: dict[str, tuple[int, np.ndarray, list[Detection]]] = {}

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
        self.frames_processed += 1
        due = self.gate.update(d.label for d in dets)
        for label in {d.label for d in dets}:
            self._last_hit[label] = (self.frames_processed, frame, dets)
        if not due:
            return

        # the confirming frame itself may be a miss: show the latest frame where these classes were seen
        hits = [self._last_hit[c] for c in due]
        _, image, image_dets = max(hits, key=lambda h: h[0])
        shown = [d for d in image_dets if d.label in due]
        scored = [d for c in due for d in self._last_hit[c][2] if d.label == c]

        now = datetime.now()
        image = draw_detections(image, shown)
        alert = Alert(self.cfg.name, due, scored, encode_jpeg(image), now)
        stem = event_stem(self.cfg.name, now)
        jpg = self.storage.snapshots and save_snapshot(self.storage.dir, stem, image)
        log.warning("ALERT %s (%s)", alert.caption.replace("\n", ", "), jpg.name if jpg else "no snapshot")
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
            CameraWorker(cam, detector, settings.storage, self._dispatch)
            for cam in settings.cameras
            if cam.enabled
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
