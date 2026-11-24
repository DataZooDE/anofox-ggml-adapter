# Changelog

## [Unreleased]

### Added
- The ggml engine of anofox-tabfm as a shared component: the ONNX reader, the host shape pass, the lowering and executor, the op table and
  its per-op goldens. Namespace `anofox::ggml_adapter` (the reader stays `anofox::onnxread`). Environment switches are
  `ANOFOX_GGML_PROFILE`, `ANOFOX_GGML_PROFILE_SHAPES` and `ANOFOX_GGML_NO_POOL` (the `TABFM_GGML_*` names still work).
- `GgmlExecutorConfig::out_of_memory_remedy` and the `remedy` argument of `GgmlRefusal`: the host's advice in the engine's errors.
- `UnsupportedCpuReason()` (from anofox-tabfm's session glue).
- `Flatten` and `Split` (the `split` input, the attribute and opset 18's `num_outputs`, zero-size parts); data inputs typed by the
  graph (int64 token ids, bool masks); `GgmlExecutor::InitializerCopyBytes()`.

- `CgroupCpuLimitFrom()`; `PhysicalCpuCores()` is capped by a container's cgroup CPU quota and counts cores by the kernel's
  `core_cpus_list` (device-tree arm64 restarts `core_id` per cluster, which the (package, core) count halved).
- `PhysicalCpuCores()` (devices.hpp): the physical cores this process may use, SMT siblings counted once (sysfs topology within the
  process's affinity on Linux, `hw.physicalcpu` on macOS, the processor-core records on Windows). The CPU engine is fastest at one
  thread per physical core; hosts cap their engine threads with it.
- The MLX graph interpreter runs `Flatten` and `Split` (the ops anofox-decide's graphs need), checked bit for bit against the ONNX
  Runtime goldens by the new `mlx_tests` (tools/mlx_plugin `-DANOFOX_MLX_TESTS=ON`; run in the MLX plugin CI job).

### Changed
- Attention is fused on the CPU: the exported scaled-dot-product pattern (`MatMul(q, kT)`, an optional scalar scale, an optional
  `Where(keep, 0, -max)` mask, `Softmax` over the last axis, float casts, `MatMul(., v)`) lowers to one `ggml_flash_attn_ext`
  instead of materialising the `[B, H, S, S]` scores. A real-valued additive bias, broadcast batch/head dimensions and the device
  path (the GPU plugins) keep the unfused lowering. A query row with every key masked (padding) now gives 0 instead of ORT's
  uniform average. `GgmlExecutor::GraphOpCounts()` reports the ops of a compiled graph (a test hook).
- The plugin layout fingerprint covers the plugin's function table too (a function added at the same ABI version is caught).
- CUDA discovery loads and initialises NVML once per process and keeps the devices it saw when a later probe fails: a GPU
  no longer vanishes mid-session (an RTX 4090 run lost its device between two models).
- The MLX interpreter refuses a `Split` whose `num_outputs` leaves a last part below zero, as ONNX and the ggml engine do.
- Windows loads a plugin with `LoadLibraryEx` restricted to the plugin's own directory and the system directories.
- The MLX interpreter returns a zero-size graph output empty instead of failing on its missing buffer.
- Float32 initializers inside the ONNX file are read in place instead of copied per compiled shape: the parsed graph must outlive
  the executor.
- A zero-size graph output is returned empty with its shape, as ONNX Runtime returns it.
- `Split` refuses an uneven equal split without `num_outputs`, and sizes given together with `num_outputs`.
