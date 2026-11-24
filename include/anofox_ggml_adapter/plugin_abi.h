/*===----------------------------------------------------------------------===
 * anofox-ggml-adapter — the GPU backend plugin ABI (version 3).
 *
 * A GPU is never a build: the host extension (anofox-tabfm, anofox-decide) dlopens a plugin library at run time and talks to it
 * through this C table. One plugin per backend (ggml-cuda, ggml-rocm, mlx), shared by every host: the plugin knows ONNX graphs, not
 * models. The host describes everything model-specific:
 *
 *   - the graph (a file path, or the bytes of an embedded graph);
 *   - the weights, as named host buffers (float32, float16 or bfloat16) the host keeps alive until destroy; the plugin uploads
 *     them to the device (upcasting 16-bit ones) and never needs to know a checkpoint format or a tensor map;
 *   - the inputs of each run, as named tensors with an ONNX element type. A `fold` input is an integer the plugin folds into the
 *     compiled graph (a shape-determining scalar such as tabfm's train_size): it is part of the compile signature, not data.
 *   - the outputs it wants, by name.
 *
 * Memory: nothing allocated on one side is freed on the other. Outputs are allocated by the plugin and released with free_output;
 * every pointer the host passes stays owned by the host.
 *
 * ABI stability: ANOFOX_GGML_PLUGIN_ABI_VERSION is checked on load before any other field is read, and a mismatch is refused.
 * Version 3 replaces the per-model tables of anofox-tabfm's ABI v2 and anofox-decide's ABI v1 (precompile and mxr_source are gone;
 * they served MIGraphX only).
 *===----------------------------------------------------------------------===*/
#ifndef ANOFOX_GGML_PLUGIN_ABI_H
#define ANOFOX_GGML_PLUGIN_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ANOFOX_GGML_PLUGIN_ABI_VERSION 3
#define ANOFOX_GGML_PLUGIN_ENTRY_SYMBOL "AnofoxGgmlGetPluginApi"

#if defined(_WIN32)
#define ANOFOX_GGML_PLUGIN_EXPORT __declspec(dllexport)
#else
#define ANOFOX_GGML_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

typedef enum {
	ANOFOX_GGML_PLUGIN_OK = 0,
	ANOFOX_GGML_PLUGIN_ERROR = 1,
	ANOFOX_GGML_PLUGIN_OUT_OF_MEMORY = 2, /* the device ran out of memory: the host names its own remedy */
} AnofoxGgmlPluginStatus;

/* ONNX TensorProto element types used across the boundary */
#define ANOFOX_GGML_DTYPE_FLOAT 1
#define ANOFOX_GGML_DTYPE_UINT8 2 /* a bool tensor crosses as uint8: _Bool's size is implementation-defined */
#define ANOFOX_GGML_DTYPE_INT64 7
#define ANOFOX_GGML_DTYPE_BOOL 9 /* data is one uint8 per element */
#define ANOFOX_GGML_DTYPE_FLOAT16 10
#define ANOFOX_GGML_DTYPE_BFLOAT16 16

/* A weight the graph reads as an external initializer, by its initializer name. */
typedef struct {
	const char *name;
	int32_t dtype;      /* FLOAT, FLOAT16 or BFLOAT16 */
	const void *data;   /* host memory, owned by the host, alive until destroy */
	int64_t count;      /* number of elements */
	int32_t rank;       /* -1 when the host does not know it; otherwise the shape below (a plugin without a graph needs it) */
	const int64_t *shape;
} AnofoxGgmlWeight;

typedef struct {
	const char *graph_path;   /* the ONNX graph; ignored when graph_data is set */
	const void *graph_data;   /* or its bytes (an embedded graph); owned by the host, read during create only */
	size_t graph_size;
	const AnofoxGgmlWeight *weights; /* external initializers, by name; may be NULL for a graph with inline weights */
	size_t weight_count;
	const char *model;        /* the host's model name: lets a plugin pick a hand-ported fast path (MLX: "mitra") */
	const char *cache_dir;    /* where the plugin may keep compiled programs; may be "" */
	const char *arch;         /* "sm_89", "gfx1201", "apple-m3": diagnostics and kernel choice */
	const char *precision;    /* "fp32" */
	const char *out_of_memory_remedy; /* the host's advice, quoted in an out-of-memory error */
	int device_ordinal;
	int threads;              /* host threads the plugin may use for host-side work; 0 = its own default */
} AnofoxGgmlCreateParams;

typedef struct {
	const char *name;
	int32_t dtype;           /* FLOAT, INT64, BOOL/UINT8 */
	int32_t rank;
	const int64_t *shape;
	const void *data;
	int32_t fold;            /* 1: an integer the plugin folds into the compiled graph (part of the compile signature) */
} AnofoxGgmlTensorIn;

typedef struct {
	char *name;              /* allocated by the plugin */
	int32_t rank;
	int64_t *shape;          /* allocated by the plugin */
	float *data;             /* allocated by the plugin; every output is returned as float32 */
	int64_t count;
} AnofoxGgmlTensorOut;

typedef struct {
	AnofoxGgmlTensorOut *tensors; /* allocated by the plugin */
	size_t count;
} AnofoxGgmlRunOutput;

typedef struct {
	int abi_version; /* the ONLY field a host reads before checking it */
	const char *(*name)(void);
	void *(*create)(const AnofoxGgmlCreateParams *params, char *err, size_t err_len);
	/* The graph's inputs (initializers excluded), NULL-terminated, owned by the handle: the host checks its contract with them. */
	const char *const *(*graph_inputs)(void *handle);
	AnofoxGgmlPluginStatus (*run)(void *handle, const AnofoxGgmlTensorIn *inputs, size_t input_count, const char *const *outputs,
	                              size_t output_count, AnofoxGgmlRunOutput *result, char *err, size_t err_len);
	void (*free_output)(AnofoxGgmlRunOutput *result);
	void (*destroy)(void *handle);
} AnofoxGgmlPluginApi;

typedef const AnofoxGgmlPluginApi *(*AnofoxGgmlGetPluginApiFn)(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ANOFOX_GGML_PLUGIN_ABI_H */
