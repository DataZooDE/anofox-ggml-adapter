#!/usr/bin/env bash
# The engine links no ONNX Runtime: onnxruntime may appear only in the Python golden tooling (tools/ggml_parity/*.py).
set -uo pipefail
cd "$(dirname "$0")/.."
hits=$(grep -rnIE 'onnxruntime|Ort::' src include cmake CMakeLists.txt test/cpp/*.cpp tools/ggml_parity/*.cpp 2>/dev/null | grep -viE 'oracle|golden|reference')
if [ -n "$hits" ]; then
	echo "FAIL: ONNX Runtime is referenced outside the Python tooling:"
	echo "$hits"
	exit 1
fi
echo "OK: no ONNX Runtime in the engine"
