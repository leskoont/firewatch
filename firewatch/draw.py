from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from firewatch.detector import Detection

COLORS = {"fire": (0, 69, 255), "smoke": (160, 160, 160)}
FALLBACK_COLOR = (0, 200, 255)


def draw_detections(img: np.ndarray, dets: Iterable[Detection]) -> np.ndarray:
    out = img.copy()
    thickness = max(2, round(sum(out.shape[:2]) / 600))
    font_scale = thickness / 3
    for d in dets:
        color = COLORS.get(d.label, FALLBACK_COLOR)
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        text = f"{d.label} {d.score:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = y1 - 4 if y1 - th - base - 4 > 0 else y1 + th + base + 4
        tx = max(0, min(x1, out.shape[1] - tw - 4))  # keep the label inside the frame
        cv2.rectangle(out, (tx, ty - th - base), (tx + tw + 4, ty + base // 2), color, -1)
        cv2.putText(
            out,
            text,
            (tx + 2, ty - base // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out
