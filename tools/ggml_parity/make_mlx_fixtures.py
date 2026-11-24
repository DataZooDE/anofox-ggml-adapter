# /// script
# dependencies = ["onnx==1.23.2"]
# ///
"""Graphs the MLX interpreter must REFUSE, for test/cpp/test_mlx_goldens.cpp (ONNX Runtime has no golden for a refusal):

    uv run tools/ggml_parity/make_mlx_fixtures.py

split_num_outputs_negative_last.onnx: x [5] -> Split(num_outputs=4). ceil(5/4) = 2 leaves -1 for the last part, which the
ONNX reference and the ggml engine refuse (an uneven split may only shorten the LAST part, never drop below zero).
"""
import os

import onnx
from onnx import TensorProto, helper

out = os.path.join(os.path.dirname(__file__), "..", "..", "test", "fixtures", "mlx")
os.makedirs(out, exist_ok=True)
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [5])
ys = [helper.make_tensor_value_info(f"y{i}", TensorProto.FLOAT, None) for i in range(4)]
node = helper.make_node("Split", ["x"], [f"y{i}" for i in range(4)], axis=0, num_outputs=4)
model = helper.make_model(helper.make_graph([node], "split_negative_last", [x], ys), opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.save(model, os.path.join(out, "split_num_outputs_negative_last.onnx"))
print("wrote split_num_outputs_negative_last.onnx")
