# /// script
# dependencies = ["onnx==1.23.2", "numpy==2.5.5"]
# ///
"""The graph the generic ggml plugin is tested with (test/fixtures/plugin/graph.onnx): no weight bytes, only external initializers the
test supplies at run time, so the plugin's host-weight path is what serves them.

    uv run tools/ggml_parity/make_plugin_fixture.py

inputs  x [B, 4] float, ids [B] int64 (data), mask [B] bool, k [] int64 (folded: it decides a shape)
weights W [4, 3] float32, E [10, 3] float16 (external; the plugin upcasts)
outputs s = x @ W + E[ids]           [B, 3]
        y = where(mask, s, 0)        [B, 3]
        head = s[:k]                 [k, 3]
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "test", "fixtures", "plugin")


def external(name, dtype, dims):
    t = TensorProto()  # no data at all: the bytes live with the host
    t.name, t.data_type = name, dtype
    t.dims.extend(dims)
    t.data_location = TensorProto.EXTERNAL
    for k, v in (("location", "host"), ("offset", "0"), ("length", "0")):
        e = t.external_data.add()
        e.key, e.value = k, v
    return t


nodes = [
    helper.make_node("MatMul", ["x", "W"], ["xw"]),
    helper.make_node("Gather", ["E", "ids"], ["e16"], axis=0),
    helper.make_node("Cast", ["e16"], ["e"], to=TensorProto.FLOAT),
    helper.make_node("Add", ["xw", "e"], ["s"]),
    helper.make_node("Unsqueeze", ["mask", "one"], ["mask2"]),
    helper.make_node("Where", ["mask2", "s", "zero"], ["y"]),
    helper.make_node("Reshape", ["k", "one"], ["k1"]),
    helper.make_node("Slice", ["s", "zero_i", "k1", "zero_i"], ["head"]),
]
graph = helper.make_graph(
    nodes, "plugin_fixture",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["B", 4]),
     helper.make_tensor_value_info("ids", TensorProto.INT64, ["B"]),
     helper.make_tensor_value_info("mask", TensorProto.BOOL, ["B"]),
     helper.make_tensor_value_info("k", TensorProto.INT64, [])],
    [helper.make_tensor_value_info("s", TensorProto.FLOAT, ["B", 3]),
     helper.make_tensor_value_info("y", TensorProto.FLOAT, ["B", 3]),
     helper.make_tensor_value_info("head", TensorProto.FLOAT, ["K", 3])],
    initializer=[external("W", TensorProto.FLOAT, [4, 3]), external("E", TensorProto.FLOAT16, [10, 3]),
                 helper.make_tensor("one", TensorProto.INT64, [1], [1]),
                 helper.make_tensor("zero_i", TensorProto.INT64, [1], [0]),
                 helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
os.makedirs(OUT, exist_ok=True)
onnx.save(model, os.path.join(OUT, "graph.onnx"))
print("wrote", os.path.normpath(os.path.join(OUT, "graph.onnx")))
