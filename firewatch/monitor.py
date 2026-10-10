from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime

import numpy as np

from firewatch.config import CameraConfig, Settings, StorageConfig
from firewatch.detector import Detection
from firewatch.draw import draw_detections
from firewatch.events import ClipRecorder, encode_jpeg, event_stem, save_snapshot
from firewatch.notify import TelegramNotifier
from firewatch.smoothing import AlertGate
from firewatch.stream import VideoStream, mask_credentials

log = logging.getLogger(__name__)

MAX_QUEUED_ALERTS = 20
# a fire alert that arrives 10 minutes late is noise
ALERT_MAX_AGE_S = 600.0


@dataclass
class Alert:
    camera: str
    labels: list[str]
    detections: list[Detection]
    jpeg: bytes
    time: datetime
    created: float = field(default_factory=time.monotonic)

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
    last_frame_age_s: float | None = None
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
        self.status = CameraStatus(name=cfg.name, source=mask_credentials(cfg.source))
        self.gate = AlertGate(cfg.thresholds, cfg.window, cfg.min_hits, cfg.cooldown_s)
        self.stream = VideoStream(
            cfg.source,
            live=cfg.is_live,
            every=cfg.process_every,
            reconnect_delay=cfg.reconnect_delay_s,
            loop=cfg.loop,
        )
        self.recorder = ClipRecorder(storage.clip_before_s, storage.clip_after_s) if storage.clips else None
        # counts from start too, so a camera that never connects goes stale in /health
        self.last_frame = time.monotonic()
        self._last_hit: dict[str, tuple[int, np.ndarray, list[Detection]]] = {}

    def filter(self, dets: list[Detection]) -> list[Detection]:
        th = self.cfg.thresholds
        return [d for d in dets if d.label in th and d.score >= th[d.label]]

    def run(self) -> None:
        log.info("camera %s: starting (%s)", self.cfg.name, self.status.source)
        self.status.state = "connecting"
        self.last_frame = time.monotonic()
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
        self.last_frame = time.monotonic()
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
    """Sends alerts one at a time, so Telegram latency stays out of the camera loops.

    While Telegram is down every alert can take a minute of retries, so the queue is
    bounded (oldest alerts go first) and alerts older than `max_age_s` are skipped.
    """

    def __init__(self, notifier, max_queued: int = MAX_QUEUED_ALERTS, max_age_s: float = ALERT_MAX_AGE_S):
        super().__init__(name="notify", daemon=True)
        self.notifier = notifier
        self.max_age_s = max_age_s
        self._queue: deque[Alert | str] = deque(maxlen=max_queued)
        self._cond = threading.Condition()
        self._stopping = False

    def put(self, item: Alert | str) -> None:
        with self._cond:
            if self._stopping:
                log.warning("shutting down, %s not sent", _describe(item))
                return
            if len(self._queue) == self._queue.maxlen:
                log.warning("notification queue is full, dropping %s", _describe(self._queue[0]))
            self._queue.append(item)
            self._cond.notify()

    def run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._queue or self._stopping)
                if not self._queue:
                    return
                item = self._queue.popleft()
            try:
                if isinstance(item, str):
                    self.notifier.send_text(item)
                else:
                    self._send(item)
            except Exception as e:
                if not self._stopping:
                    log.exception("notification failed")
                else:
                    # the notifier gets closed right after stop(), cutting off what's in flight
                    log.warning("shutting down, %s not sent (%s)", _describe(item), type(e).__name__)

    def _send(self, alert: Alert) -> None:
        n = self.notifier
        for attempt in (lambda: n.send_photo(alert.jpeg, alert.caption), lambda: n.send_text(alert.caption)):
            age = time.monotonic() - alert.created
            if age > self.max_age_s:
                log.error("alert from %s is %.0f s old, not sending it", alert.camera, age)
                return
            if attempt():
                return
        if self._stopping:
            log.warning("shutting down, %s not sent", _describe(alert))
            return
        # the cooldown is already armed by now, so a lost alert has to at least be loud
        log.error("notification failed: alert from %s was not delivered", alert.camera)

    def stop(self, timeout: float = 3.0) -> None:
        """Give the queue `timeout` seconds to drain, then drop the rest."""
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        if self.is_alive():
            self.join(timeout)
        with self._cond:
            left = len(self._queue)
            self._queue.clear()
        if left:
            log.warning("shutting down, %d notification(s) not sent", left)


def _describe(item: Alert | str) -> str:
    return "a text message" if isinstance(item, str) else f"alert from {item.camera} at {item.time:%H:%M:%S}"


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
        """Copies of the worker statuses with the stream's connection state filled in."""
        now = time.monotonic()
        result = []
        for w in self.workers:
            st, connected = w.status, w.stream.connected
            state, error = st.state, st.error
            if state == "running" and not connected:
                state = "reconnecting"
            if state in ("connecting", "reconnecting"):
                error = w.stream.error
            result.append(
                replace(
                    st,
                    state=state,
                    error=error,
                    connected=connected,
                    reconnects=w.stream.reconnects,
                    last_frame_age_s=round(now - w.last_frame, 1),
                )
            )
        return result
