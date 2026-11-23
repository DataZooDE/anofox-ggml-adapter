# anofox-ggml-adapter

ONNX inference graphs on [ggml](https://github.com/ggml-org/ggml): the engine that runs the models of the anofox DuckDB extensions
([anofox-tabfm](https://github.com/DataZooDE/anofox-tabfm): tabular foundation models, [anofox-decide](https://github.com/DataZooDE/anofox-decide):
decision models). It reads an ONNX graph with a small wire-format reader (no libonnx, no protobuf), evaluates everything that is known before
the data arrives on the host (shapes, constants), lowers the rest to a ggml graph and runs it on the CPU, or on a CUDA, HIP or other ggml GPU
device. Weights stay in the caller's buffers (or are uploaded once to a device); compiled graphs are cached per input-shape signature.

No ONNX Runtime is linked. ONNX Runtime is only the oracle of the per-op goldens (`tools/ggml_parity/gen_op_goldens.py`).

## Layout

| Path | What |
|---|---|
| `include/anofox_ggml_adapter/` | the API: `ggml_lower.hpp` (`GgmlExecutor`, `GgmlRunGraph`), `onnx_reader.hpp`, `shape_eval.hpp`, `ggml_ops.hpp` (the op table), `ggml_engine.hpp` (`GgmlRefusal`), `cpu_support.hpp` |
| `src/` | the engine |
| `cmake/anofox_ggml_adapter.cmake` | how a consumer builds it: the sources, `vendor/ggml` as static archives with hidden visibility, the `anofox_ggml` link target |
| `vendor/ggml` | ggml, a git submodule |
| `test/` | Catch2 tests and the per-op goldens (`test/fixtures/ggml_ops`) |
| `tools/ggml_parity/` | the golden generator (ONNX Runtime as oracle) and a standalone driver for real-weight graphs |

## Use it

```cmake
include(vendor/anofox-ggml-adapter/cmake/anofox_ggml_adapter.cmake)
list(APPEND MY_SOURCES ${ANOFOX_GGML_ADAPTER_SOURCES})
add_dependencies(my_target ggml)
target_link_libraries(my_target anofox_ggml)
```

```cpp
#include "anofox_ggml_adapter/ggml_lower.hpp"
using namespace anofox::ggml_adapter;

auto graph = ::anofox::onnxread::ParseModelFile("model.onnx");
GgmlExecutorConfig config;
config.threads = 8;
config.weights = my_weight_lookup;            // name, dims -> ExternalWeight{data, count, onnx_type}; the buffers must outlive the executor
GgmlExecutor executor(graph, config);
HostTensors outputs = executor.Run({{"x", {{1, 4}, {0.f, 1.f, 2.f, 3.f}}}}, GgmlRunOptions{});
```

## Rules

- **A new ONNX op or attribute form** needs a golden first (`tools/ggml_parity/gen_op_goldens.py`, ONNX Runtime is the oracle;
  regenerate with `uv run tools/ggml_parity/gen_op_goldens.py`), then the lowering in `src/ggml_lower.cpp` and its row in `ggml_ops.hpp`.
  `GgmlRefusal` names any op the engine lacks. Never compare a golden with a loose tolerance to make it pass: exact ops are bit-exact.
- The adapter knows nothing about its hosts: error texts take the host's advice (`GgmlExecutorConfig::out_of_memory_remedy`, the
  `remedy` of `GgmlRefusal`).

## Build and test

```bash
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ./build/adapter_tests
```

## History

Extracted from anofox-tabfm (`src/tabfm_ggml_lower.cpp` and friends) at commit dd2b4f6 (release v2026.11.23); see its
`docs/GGML_PLAN.md` for how the engine was built and verified op by op against ONNX Runtime.

## License

MIT. ggml is MIT (`vendor/ggml/LICENSE`).
