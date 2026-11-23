# Changelog

## [Unreleased]

### Added
- The ggml engine of anofox-tabfm as a shared component: the ONNX reader, the host shape pass, the lowering and executor, the op table and
  its per-op goldens. Namespace `anofox::ggml_adapter` (the reader stays `anofox::onnxread`). Environment switches are
  `ANOFOX_GGML_PROFILE`, `ANOFOX_GGML_PROFILE_SHAPES` and `ANOFOX_GGML_NO_POOL` (the `TABFM_GGML_*` names still work).
- `GgmlExecutorConfig::out_of_memory_remedy` and the `remedy` argument of `GgmlRefusal`: the host's advice in the engine's errors.
- `UnsupportedCpuReason()` (from anofox-tabfm's session glue).
