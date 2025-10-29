from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable


class HitWindow:
    """Confirms a detection once it was seen in at least `min_hits` of the last `window` frames."""

    def __init__(self, window: int, min_hits: int):
        self.window = window
        self.min_hits = min_hits
        self._hits: deque[bool] = deque(maxlen=window)

    def update(self, hit: bool) -> bool:
        self._hits.append(hit)
        return self.hits >= self.min_hits

    @property
    def hits(self) -> int:
        return sum(self._hits)


class Cooldown:
    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic):
        self.seconds = seconds
        self._clock = clock
        self._last: float | None = None

    def ready(self) -> bool:
        return self._last is None or self._clock() - self._last >= self.seconds

    def trigger(self) -> None:
        self._last = self._clock()


class AlertGate:
    """Per-camera decision: which classes are confirmed, and whether to alert right now."""

    def __init__(
        self,
        classes: Iterable[str],
        window: int,
        min_hits: int,
        cooldown_s: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.windows = {c: HitWindow(window, min_hits) for c in classes}
        self.cooldown = Cooldown(cooldown_s, clock)

    def update(self, labels: Iterable[str]) -> list[str]:
        """Feed labels seen on the current frame. Returns the classes to alert about now."""
        present = set(labels)
        confirmed = [c for c, w in self.windows.items() if w.update(c in present)]
        if not confirmed or not self.cooldown.ready():
            return []
        self.cooldown.trigger()
        return confirmed
