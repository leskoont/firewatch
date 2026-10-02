from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

INPUT = 64


def make_head_output() -> np.ndarray:
    """Fixed raw head output (1, 4 + 2, 3) for a 64x64 input, classes smoke/fire."""
    anchors = [
        # cx, cy, w, h, smoke, fire
        [32, 32, 20, 20, 0.10, 0.90],
        [33, 33, 20, 20, 0.05, 0.80],  # overlaps the first one, same class -> NMS
        [10, 30, 8, 8, 0.70, 0.01],
    ]
    return np.asarray(anchors, np.float32).T[None]


def build_dummy_model(path, input_shape=(1, 3, INPUT, INPUT)) -> None:
    # output = const + 0 * sum(input): keeps the input in the graph, ignores its values
    const = numpy_helper.from_array(make_head_output(), "head")
    zero = numpy_helper.from_array(np.zeros((), np.float32), "zero")
    nodes = [
        helper.make_node("ReduceSum", ["images"], ["s"], keepdims=0),
        helper.make_node("Mul", ["s", "zero"], ["z"]),
        helper.make_node("Add", ["head", "z"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "dummy",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, list(input_shape))],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 6, 3])],
        initializer=[const, zero],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    for key, value in {"names": "{0: 'smoke', 1: 'fire'}", "imgsz": f"[{INPUT}, {INPUT}]"}.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture(scope="session")
def dummy_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "dummy.onnx"
    build_dummy_model(path)
    return path


@pytest.fixture
def dynamic_model(tmp_path):
    path = tmp_path / "dynamic.onnx"
    build_dummy_model(path, input_shape=(1, 3, "height", "width"))
    return path
