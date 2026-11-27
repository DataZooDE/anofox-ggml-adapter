# Changelog

## [Unreleased]

### Fixed
- ConstantOfShape whose output is too large to fold on the host (above 2^20 values) is filled on the device instead of failing with
  "op is not implemented". kumo-tabular fills [1, features, rows, 3, 64]: a table of 1000 rows with 5 or more features failed on the CPU
  and the ggml GPU plugin. Golden: constantofshape_over_fold_cap.

### Changed
- The CPU fuses attention over rank-5 tensors [B1, B2, H, S, D] as well (held as [B1*B2, H, S, D]): exaone-tabular's feature-context
  attentions ran unfused, their [heads, S, S] scores materialised and copied. Golden: sdpa_rank5.
- RMSNorm spelled out (Pow, ReduceMean, Add eps, Sqrt, Reciprocal, Mul) lowers to one ggml_rms_norm on every backend: tabfm-v1-1
  runs 341 of them per call. Golden: rmsnorm_pattern (ReduceMean's tolerance).
- The CPU's attention fusion looks through PyTorch's SDPA guard Where(IsNaN(p), 0, p) between Softmax and PV: tabfm-v1-1's 36
  attentions per call ran unfused (their [heads, S, S] scores materialised). Golden: sdpa_nan_guard.
- The attention fusion also accepts a 0/-max mask the shape pass knows on the host (built from shapes alone, as tabfm-v1-1's
  Where(Less(Range(..)), 0, -max)); it took only runtime masks. Golden: sdpa_known_mask.
- The exact-GELU fusion also matches the half inside, x * (0.5 * (erf(x / sqrt 2) + 1)), as kumo-tabular exports it (29 per call ran
  as Div, a custom Erf kernel, Add and two Mul). Golden: gelu_erf_pattern_half_inside.
- A batched activation times a 2-D weight runs as one matmul over all the activation's rows instead of one small matmul per batch
  (kumo-tabular's [25200, 4, 32] x [32, 64] ran at ~9 GFLOP/s). Golden: matmul_batched_activation_2d_weight.

## [2026.11.26]

### Changed
- MLX plugin: shape arithmetic (nodes whose operands are all small integer or boolean tensors) and the integer read-backs of Reshape,
  Slice, Expand and the like run on MLX's CPU stream. On the GPU stream each read-back waited for every layer kernel queued before it;
  on an M3 that was ~58% of the plugin's time. decide's 400-call julia-1 run: 24.5 s -> 11.0 s, results bit-identical.
- MLX plugin: 16-bit weights for a float32 graph are widened on the GPU stream (not the CPU), and MLX's buffer cache is released after
  load and after each run. Laya typed-decisions on an M3: resident set 1.87 -> 0.40 GB, physical footprint 2.19 -> 1.77 GB, no change
  in speed.

## [2026.11.25]

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
  instead of materialising the `[B, H, S, S]` scores, on the CPU (a GPU keeps the unfused attention: ROCm's flash-attention kernels
  moved probabilities by up to 5e-3 against the CPU, above the 1e-3 GPU bar). A real-valued additive
  bias and broadcast batch/head dimensions keep the unfused lowering. A query row with every key masked (padding) now gives 0 instead of ORT's
  uniform average. `GgmlExecutor::GraphOpCounts()` reports the ops of a compiled graph (a test hook).
  A query row's result does not depend on how many rows share the batch: fewer than 64 query rows are padded to the
  tiled kernel's 64 (ggml picks its flash-attention kernel by row count, and the kernels sum in different orders).
- The exact-GELU pattern (`Mul(Mul(x, Add(Erf(Div(x, sqrt 2)), 1)), 0.5)`) lowers to one `ggml_gelu_erf` on the CPU and on devices
  (five passes, three of them CPU-only kernels, before). It computes `erf(x * 0.70710678)`: within an ulp of the graph's argument,
  held to Erf's tolerance by ONNX Runtime goldens.
- An embedding lookup whose token ids come straight from a graph input is `ggml_get_rows` over int32 ids the executor converts per
  run (multi-threaded, and on a GPU no longer a copy of the whole table to the host every run). Negative ids wrap; NaN, fractional
  and out-of-range ids are refused with the node, element and bound named (the host kernel returned NaN).
- One CPU activation arena per executor, sized for the largest cached graph, instead of one per cached input shape; it replans on
  every switch of graph. `GgmlExecutor::ActivationBytes()` and `UploadedBytes()` as test hooks.
- No activation copies around the fused attention (K is read from the tensor the graph's Transpose copied; a permutation that undoes
  a view returns the view); a key-padding mask is cast to float16 before it is repeated.
- The elementwise custom kernel switches once per row and vectorises its contiguous case.
- The shape pass no longer decodes an inline initializer above the known-elements cap (a full graph's weights, every compile).
- Devices: inline float32 weights (a registered full graph) are uploaded once per executor, not on every run; a float32 initializer
  takes float16/bfloat16 data (the upload, or the MLX interpreter at create, upcasts it), so a host can pass its 16-bit checkpoint;
  running out of device memory for a shape's activations is reported as out of memory, with the host's remedy.
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
