import numpy as np
import pytest

from firewatch.ops import (
    letterbox,
    nms,
    postprocess,
    scale_boxes,
    to_tensor,
    xywh2xyxy,
)


@pytest.mark.parametrize(
    ("h", "w", "ratio", "pad"),
    [
        (480, 640, 1.0, (0, 80)),
        (1080, 1920, 1 / 3, (0, 140)),
        (640, 320, 1.0, (160, 0)),
        (100, 100, 6.4, (0, 0)),  # small images are scaled up, like ultralytics does
    ],
)
def test_letterbox_geometry(h, w, ratio, pad):
    img = np.full((h, w, 3), 255, np.uint8)
    out, info = letterbox(img, (640, 640))
    assert out.shape == (640, 640, 3)
    assert info.ratio == pytest.approx(ratio)
    assert info.pad == pad
    assert info.src_shape == (h, w)
    if pad[1]:
        assert (out[: pad[1] - 1] == 114).all()
        assert (out[pad[1] + 1 : 640 - pad[1] - 1] == 255).all()


def test_letterbox_odd_padding_sums_to_target():
    img = np.zeros((361, 640, 3), np.uint8)
    out, info = letterbox(img, (640, 640))
    assert out.shape[:2] == (640, 640)
    assert info.pad[1] in (139, 140)


def test_scale_boxes_inverts_letterbox():
    img = np.zeros((720, 1280, 3), np.uint8)
    _, info = letterbox(img, (640, 640))
    src = np.array([[100, 200, 300, 400, 0.9, 1]], np.float32)
    lb = src.copy()
    lb[:, :4] *= info.ratio
    lb[:, [0, 2]] += info.pad[0]
    lb[:, [1, 3]] += info.pad[1]
    np.testing.assert_allclose(scale_boxes(lb, info), src, atol=1e-3)


def test_scale_boxes_clips_to_image():
    img = np.zeros((100, 200, 3), np.uint8)
    _, info = letterbox(img, (640, 640))
    boxes = np.array([[-50, 0, 700, 640, 0.5, 0]], np.float32)
    out = scale_boxes(boxes, info)
    assert out[0, 0] == 0 and out[0, 2] == 200
    assert out[0, 1] == 0 and out[0, 3] == 100


def test_to_tensor_layout():
    img = np.zeros((4, 6, 3), np.uint8)
    img[..., 0] = 255  # blue channel in BGR
    x = to_tensor(img)
    assert x.shape == (1, 3, 4, 6)
    assert x.dtype == np.float32
    assert x[0, 2].max() == 1.0 and x[0, 0].max() == 0.0  # ends up as the last (B) channel in RGB


def test_xywh2xyxy():
    out = xywh2xyxy(np.array([[10, 20, 4, 6]], np.float32))
    np.testing.assert_array_equal(out, [[8, 17, 12, 23]])


def test_nms_keeps_best_of_overlapping():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], np.float32)
    scores = np.array([0.8, 0.9, 0.7], np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [1, 2]


def test_nms_threshold_is_exclusive():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 5]], np.float32)  # IoU exactly 0.5
    scores = np.array([0.9, 0.8], np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [0, 1]
    assert nms(boxes, scores, 0.49).tolist() == [0]


def _raw(rows):
    return np.asarray(rows, np.float32).T  # (4 + nc, anchors)


def test_postprocess_is_class_aware():
    raw = _raw(
        [
            [50, 50, 20, 20, 0.9, 0.0],
            [50, 50, 20, 20, 0.0, 0.8],  # same box, other class: must survive
            [51, 51, 20, 20, 0.6, 0.0],  # same class, suppressed
        ]
    )
    out = postprocess(raw, conf_thres=0.25, iou_thres=0.45)
    assert out.shape == (2, 6)
    assert out[:, 5].tolist() == [0, 1]
    np.testing.assert_allclose(out[0, :4], [40, 40, 60, 60])


def test_postprocess_conf_filter_and_empty():
    raw = _raw([[50, 50, 20, 20, 0.2, 0.1]])
    assert postprocess(raw, conf_thres=0.25).shape == (0, 6)
    assert len(postprocess(raw, conf_thres=0.15)) == 1


def test_postprocess_max_det():
    rows = [[i * 30 + 15, 15, 10, 10, 0.5 + i / 100, 0] for i in range(10)]
    out = postprocess(_raw(rows), conf_thres=0.1, max_det=3)
    assert len(out) == 3
    assert out[0, 4] == pytest.approx(0.59)
