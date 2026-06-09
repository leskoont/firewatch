from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from firewatch.config import CameraConfig, Settings, StorageConfig
from firewatch.detector import Detection
from firewatch.draw import draw_detections
from firewatch.events import ClipRecorder, encode_jpeg, event_stem, save_snapshot
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


@dataclass
class CameraStatus:
    name: str
    source: str
    state: str = "starting"
    connected: bool = False
    reconnects: int = 0
    frames_processed: int = 0
    processing_fps: float = 0.0
    last_frame_at: str | None = None
    last_alert_at: str | None = None
    alerts: int = 0
    last_detections: list[dict] = field(default_factory=list)
    error: str | None = None


class CameraWorker(threading.Thread):
    def __init__(self, cfg: CameraConfig, detector, storage: StorageConfig, on_alert):
        super().__init__(name=f"camera-{cfg.name}", daemon=True)
        self.cfg = cfg
        self.detector = detector
        self.storage = storage
        self.on_alert = on_alert
        self.stop_event = threading.Event()
        self.status = CameraStatus(name=cfg.name, source=cfg.source)
        self.gate = AlertGate(cfg.thresholds, cfg.window, cfg.min_hits, cfg.cooldown_s)
        self.stream = VideoStream(
            cfg.source,
            live=cfg.is_live,
            every=cfg.process_every,
            reconnect_delay=cfg.reconnect_delay_s,
            loop=cfg.loop,
        )
        self.recorder = ClipRecorder(storage.clip_before_s, storage.clip_after_s) if storage.clips else None
        self._last_hit: dict[str, tuple[int, np.ndarray, list[Detection]]] = {}

    def filter(self, dets: list[Detection]) -> list[Detection]:
        th = self.cfg.thresholds
        return [d for d in dets if d.label in th and d.score >= th[d.label]]

    def run(self) -> None:
        log.info("camera %s: starting (%s)", self.cfg.name, self.cfg.source)
        self.status.state = "connecting"
        delay = 1.0
        while not self.stop_event.is_set():
            processed = self.status.frames_processed
            try:
                self._watch()
                self.status.state = "stopped"
                break
            except Exception as e:
                if self.status.frames_processed > processed:
                    delay = 1.0
                log.exception("camera %s crashed, restarting in %.0fs", self.cfg.name, delay)
                self.status.state, self.status.error = "error", f"{type(e).__name__}: {e}"
                self.stop_event.wait(delay)
                delay = min(delay * 2, 60.0)
        if self.recorder:
            self.recorder.close()
        self.status.connected = False
        log.info("camera %s: stopped", self.cfg.name)

    def _watch(self) -> None:
        tick, frames = None, 0
        # if detection throws, the live reader must stop now rather than after the backoff
        with contextlib.closing(self.stream.frames(self.stop_event)) as stream:
            for idx, frame in stream:
                dets = self.filter(self.detector(frame))
                # for files the clip clock is the position in the video
                t = None if self.cfg.is_live else idx / (self.stream.fps or 25.0)
                self._process(frame, dets, t)

                frames += 1
                now = time.monotonic()
                if tick is None:
                    tick, frames = now, 0
                elif now - tick >= 2.0:
                    self.status.processing_fps = round(frames / (now - tick), 2)
                    tick, frames = now, 0

    def _process(self, frame: np.ndarray, dets: list[Detection], t: float | None = None) -> None:
        st = self.status
        st.state, st.error = "running", None
        st.frames_processed += 1
        st.last_frame_at = datetime.now().isoformat(timespec="seconds")
        st.last_detections = [d.to_dict() for d in dets]

        due = self.gate.update(d.label for d in dets)
        for label in {d.label for d in dets}:
            self._last_hit[label] = (st.frames_processed, frame, dets)
        if self.recorder:
            self.recorder.push(draw_detections(frame, dets) if dets else frame, t)
        if not due:
            return

        # the confirming frame itself may be a miss: show the latest frame where these classes were seen
        hits = [self._last_hit[c] for c in due]
        _, image, image_dets = max(hits, key=lambda h: h[0])
        shown = [d for d in image_dets if d.label in due]
        scored = [d for c in due for d in self._last_hit[c][2] if d.label == c]

        now = datetime.now()
        stem = event_stem(self.cfg.name, now)
        alert = Alert(self.cfg.name, due, scored, encode_jpeg(draw_detections(image, shown)), now)
        saved = []
        if self.storage.snapshots and save_snapshot(self.storage.dir, stem, alert.jpeg):
            saved.append(f"{stem}.jpg")
        if self.recorder and self.recorder.start(self.storage.dir / f"{stem}.mp4"):
            saved.append(f"{stem}.mp4")
        st.alerts += 1
        st.last_alert_at = now.isoformat(timespec="seconds")
        log.warning("ALERT %s (%s)", alert.caption.replace("\n", ", "), ", ".join(saved) or "nothing saved")
        self.on_alert(alert)

    def stop(self) -> None:
        self.stop_event.set()


class AlertSender(threading.Thread):
    """Sends alerts one at a time, so Telegram latency stays out of the camera loops."""

    def __init__(self, notifier):
        super().__init__(name="notify", daemon=True)
        self.notifier = notifier
        self._queue: queue.Queue[Alert | str | None] = queue.Queue()

    def put(self, item: Alert | str) -> None:
        self._queue.put(item)

    def run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                if isinstance(item, str):
                    self.notifier.send_text(item)
                elif not self.notifier.send_photo(item.jpeg, item.caption):
                    self.notifier.send_text(item.caption)
            except Exception:
                log.exception("notification failed")

    def stop(self, timeout: float = 10.0) -> None:
        self._queue.put(None)
        if self.is_alive():
            self.join(timeout)


class Monitor:
    def __init__(self, settings: Settings, detector, notifier: TelegramNotifier | None = None):
        self.settings = settings
        self.detector = detector
        self.sender = AlertSender(notifier) if notifier is not None else None
        self.workers = [
            CameraWorker(cam, detector, settings.storage, self._dispatch)
            for cam in settings.cameras
            if cam.enabled
        ]

    def _dispatch(self, alert: Alert) -> None:
        if self.sender is not None:
            self.sender.put(alert)

    def start(self) -> None:
        for w in self.workers:
            w.start()
        if self.sender is not None:
            self.sender.start()
            self.sender.put(f"firewatch запущен, камер: {len(self.workers)}")

    def stop(self, timeout: float = 5.0) -> None:
        for w in self.workers:
            w.stop()
        for w in self.workers:
            if w.is_alive():
                w.join(timeout)
        if self.sender is not None:
            self.sender.stop()

    def wait(self) -> None:
        for w in self.workers:
            while w.is_alive():
                w.join(0.5)

    def statuses(self) -> list[CameraStatus]:
        # the stream changes these between frames, e.g. while an RTSP camera is down
        for w in self.workers:
            st = w.status
            st.connected = w.stream.connected
            st.reconnects = w.stream.reconnects
            if st.state == "running" and not st.connected:
                st.state = "reconnecting"
            if st.state in ("connecting", "reconnecting"):
                st.error = w.stream.error
        return [w.status for w in self.workers]
