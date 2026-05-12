from __future__ import annotations

import time
from dataclasses import asdict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from firewatch import __version__


def create_app(detector, monitor=None, model_name: str = "") -> FastAPI:
    app = FastAPI(title="firewatch", version=__version__)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": model_name,
            "providers": detector.providers,
            "classes": list(detector.names.values()),
        }

    @app.post("/detect")
    def detect(
        file: UploadFile = File(...),
        conf: float | None = Query(None, ge=0.0, le=1.0),
    ):
        data = file.file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "cannot decode image")

        t0 = time.perf_counter()
        dets = detector(img, conf=conf)
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "width": img.shape[1],
            "height": img.shape[0],
            "inference_ms": round(elapsed, 1),
            "detections": [d.to_dict() for d in dets],
        }

    @app.get("/cameras")
    def cameras():
        if monitor is None:
            return []
        return [asdict(s) for s in monitor.statuses()]

    return app
