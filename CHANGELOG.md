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

- The MLX graph interpreter runs `Flatten` and `Split` (the ops anofox-decide's graphs need), checked bit for bit against the ONNX
  Runtime goldens by the new `mlx_tests` (tools/mlx_plugin `-DANOFOX_MLX_TESTS=ON`; run in the MLX plugin CI job).

### Changed
- The MLX interpreter returns a zero-size graph output empty instead of failing on its missing buffer.
- Float32 initializers inside the ONNX file are read in place instead of copied per compiled shape: the parsed graph must outlive
  the executor.
- A zero-size graph output is returned empty with its shape, as ONNX Runtime returns it.
- `Split` refuses an uneven equal split without `num_outputs`, and sizes given together with `num_outputs`.
