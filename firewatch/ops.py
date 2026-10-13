from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# class offset for single-pass batched NMS, bigger than any 640 image side
_CLASS_OFFSET = 7680.0


@dataclass(frozen=True)
class LetterboxInfo:
    ratio: float
    pad: tuple[float, float]  # (left, top)
    src_shape: tuple[int, int]  # (h, w)


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize keeping aspect ratio and pad to new_shape (h, w).

    Mirrors ultralytics LetterBox(auto=False) so results match the exported model.
    """
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw = (new_shape[1] - new_w) / 2
    dh = (new_shape[0] - new_h) / 2

    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, LetterboxInfo(ratio=r, pad=(left, top), src_shape=(h, w))


def to_tensor(img_bgr: np.ndarray) -> np.ndarray:
    x = img_bgr[:, :, ::-1].transpose(2, 0, 1)
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x[None]


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    half_w = x[..., 2] / 2
    half_h = x[..., 3] / 2
    y[..., 0] = x[..., 0] - half_w
    y[..., 1] = x[..., 1] - half_h
    y[..., 2] = x[..., 0] + half_w
    y[..., 3] = x[..., 1] + half_h
    return y


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy NMS. Returns indices of kept boxes sorted by score."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def postprocess(
    pred: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    max_nms: int = 30000,
) -> np.ndarray:
    """Decode raw YOLOv8/11 head output for one image.

    pred: (4 + nc, num_anchors) with boxes as cx, cy, w, h in input pixels and
    per-class scores already passed through sigmoid.
    Returns (n, 6) array of x1, y1, x2, y2, score, class_id in input pixels.
    """
    pred = pred.T
    cls_scores = pred[:, 4:]
    class_ids = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(pred)), class_ids]

    mask = scores > conf_thres
    if not mask.any():
        return np.zeros((0, 6), dtype=np.float32)
    boxes = xywh2xyxy(pred[mask, :4])
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(scores) > max_nms:
        top = scores.argsort()[::-1][:max_nms]
        boxes, scores, class_ids = boxes[top], scores[top], class_ids[top]

    keep = nms(boxes + class_ids[:, None] * _CLASS_OFFSET, scores, iou_thres)[:max_det]
    out = np.concatenate([boxes[keep], scores[keep, None], class_ids[keep, None].astype(np.float32)], axis=1)
    return out.astype(np.float32)


def scale_boxes(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Map xyxy boxes from letterboxed input back to the source image."""
    out = boxes.copy()
    out[:, [0, 2]] -= info.pad[0]
    out[:, [1, 3]] -= info.pad[1]
    out[:, :4] /= info.ratio
    h, w = info.src_shape
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out
