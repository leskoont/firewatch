from __future__ import annotations

import ast
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from firewatch.ops import letterbox, postprocess, scale_boxes, to_tensor

log = logging.getLogger(__name__)

DEFAULT_NAMES = {0: "smoke", 1: "fire"}


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    label: str

    @property
    def box(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "class_id": self.class_id,
            "score": round(self.score, 4),
            "box": [round(v, 1) for v in (self.x1, self.y1, self.x2, self.y2)],
        }


def resolve_providers(requested: str | Sequence[str] = "auto") -> list[str]:
    available = ort.get_available_providers()
    if requested == "auto":
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif isinstance(requested, str):
        wanted = [requested]
    else:
        wanted = list(requested)
    providers = [p for p in wanted if p in available]
    if not providers:
        log.warning("none of %s available, falling back to CPU", wanted)
        providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
        # picks up CUDA/cuDNN from the nvidia-* wheels (onnxruntime-gpu[cuda,cudnn]),
        # so no system-wide CUDA toolkit is needed
        ort.preload_dlls()
    return providers


class OnnxDetector:
    def __init__(
        self,
        model_path: str | Path,
        providers: str | Sequence[str] = "auto",
        conf: float = 0.25,
        iou: float = 0.45,
        names: dict[int, str] | None = None,
        intra_op_threads: int | None = None,
    ):
        opts = ort.SessionOptions()
        if intra_op_threads:
            opts.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=resolve_providers(providers)
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        if not isinstance(h, int) or not isinstance(w, int):
            # dynamic axes: fall back to the export size stored in metadata, or 640
            h, w = self._meta_imgsz() or (640, 640)
        self.input_hw = (h, w)
        self.names = names or self._meta_names() or DEFAULT_NAMES
        self.conf = conf
        self.iou = iou

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def _meta(self) -> dict[str, str]:
        return self.session.get_modelmeta().custom_metadata_map

    def _meta_names(self) -> dict[int, str] | None:
        raw = self._meta().get("names")
        if not raw:
            return None
        return {int(k): str(v) for k, v in ast.literal_eval(raw).items()}

    def _meta_imgsz(self) -> tuple[int, int] | None:
        raw = self._meta().get("imgsz")
        if not raw:
            return None
        h, w = ast.literal_eval(raw)
        return int(h), int(w)

    def infer_raw(self, img_bgr: np.ndarray):
        padded, info = letterbox(img_bgr, self.input_hw)
        out = self.session.run(None, {self.input_name: to_tensor(padded)})[0]
        return out[0], info

    def __call__(
        self, img_bgr: np.ndarray, conf: float | None = None, iou: float | None = None
    ) -> list[Detection]:
        raw, info = self.infer_raw(img_bgr)
        conf = self.conf if conf is None else conf
        iou = self.iou if iou is None else iou
        dets = postprocess(raw, conf, iou)
        if not len(dets):
            return []
        dets = scale_boxes(dets, info)
        return [
            Detection(
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(s),
                int(c),
                self.names.get(int(c), str(int(c))),
            )
            for x1, y1, x2, y2, s, c in dets
        ]
