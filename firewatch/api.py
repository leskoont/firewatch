from __future__ import annotations

import asyncio
import io
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.routing import APIRoute
from PIL import Image, UnidentifiedImageError

from firewatch import __version__

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
# a few hundred KB of PNG can decode into gigabytes, so the size is checked from the header first
MAX_PIXELS = 40_000_000
STALE_AFTER_S = 60.0


def image_size(data: bytes) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except Image.DecompressionBombError as e:  # PIL's own limit, far above ours
        raise HTTPException(413, "image is too large") from e
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, "cannot decode image") from e


class LimitedBody(APIRoute):
    """Rejects bodies over MAX_UPLOAD_BYTES while they arrive, chunked ones included.

    Starlette spools the whole multipart body before the endpoint runs.
    """

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def limited(request: Request):
            size = request.headers.get("content-length", "")
            if size.isdigit() and int(size) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "image is too large")
            received = 0

            async def receive():
                nonlocal received
                message = await request.receive()
                received += len(message.get("body", b""))
                if received > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "image is too large")
                return message

            return await handler(Request(request.scope, receive))

        return limited


def create_app(detector, monitor=None, model_name: str = "") -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        # uvicorn re-raises SIGTERM once it is done, so code after uvicorn.run() may never run
        if monitor is not None:
            await asyncio.to_thread(monitor.stop)

    app = FastAPI(title="firewatch", version=__version__, lifespan=lifespan)
    upload = APIRouter(route_class=LimitedBody)

    @app.get("/health")
    def health(response: Response):
        body = {
            "status": "ok",
            "model": model_name,
            "providers": detector.providers,
            "classes": list(detector.names.values()),
        }
        if monitor is not None:
            statuses = monitor.statuses()
            body["cameras"] = len(statuses)
            body["last_frame_age_s"] = {s.name: s.last_frame_age_s for s in statuses}
            # a video file without `loop` that reached its end is done, not stuck
            stale = sorted(
                s.name
                for s in statuses
                if s.state != "stopped" and (s.last_frame_age_s is None or s.last_frame_age_s > STALE_AFTER_S)
            )
            if stale:
                body["status"] = "degraded"
                body["stale"] = stale
                response.status_code = 503
        return body

    @upload.post("/detect")
    def detect(
        file: UploadFile = File(...),
        conf: float | None = Query(None, ge=0.0, le=1.0),
    ):
        data = file.file.read()
        w, h = image_size(data)
        if w * h > MAX_PIXELS:
            raise HTTPException(413, f"image is too large: {w}x{h}")
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

    app.include_router(upload)
    return app
