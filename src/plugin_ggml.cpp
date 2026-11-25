/*===----------------------------------------------------------------------===
 * plugin_ggml.cpp — the generic ggml GPU backend plugin (plugin_abi.h, version 3).
 *
 * The engine every host already runs on the CPU (ONNX graph -> ggml graph), pointed at a ggml GPU device: CUDA or HIP, chosen when this
 * library is built (tools/ggml_plugin). It knows ONNX graphs, not models: the host passes the graph, its weights by initializer name,
 * named inputs and the outputs it wants. Built with ANOFOX_GGML_PLUGIN_CPU_TEST it runs the same code on the CPU device, so the adapter's
 * CI tests the plugin itself, not a stand-in.
 *
 * Weights: the host's buffers are read in place; a GPU executor uploads them once (upcasting float16/bfloat16) and ops the GPU backend
 * cannot run stay on the CPU inside the same graph. The CPU test build upcasts 16-bit weights into its own arena (the CPU executor reads
 * float32 only).
 *
 * Precision: a CUDA GPU multiplies fp32 matrices in TF32 by default on Ampere and later (measured 2e-3 relative error per matmul). "fp32"
 * (the default) sets NVIDIA_TF32_OVERRIDE=0 before the CUDA runtime starts and pads matmuls past ggml's small-batch TF32 kernel
 * (GgmlExecutorConfig::strict_fp32); "tf32" leaves the hardware default. The environment switch is process-wide and latched by the first
 * CUDA initialisation. fp16/bf16 are refused (they would need a graph conversion; a silent fp32 run would be a lie).
 *
 * Every exception is caught at the API boundary and reported through the err buffer.
 *===----------------------------------------------------------------------===*/

#include "anofox_ggml_adapter/ggml_engine.hpp"
#include "anofox_ggml_adapter/ggml_lower.hpp"
#include "anofox_ggml_adapter/onnx_reader.hpp"
#include "anofox_ggml_adapter/plugin_abi.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using namespace ::anofox::ggml_adapter; // NOLINT

void SetError(char *err, size_t err_len, const std::string &message) {
	if (!err || !err_len) {
		return;
	}
	std::strncpy(err, message.c_str(), err_len - 1);
	err[err_len - 1] = '\0';
}

size_t ElementCount(const std::vector<int64_t> &dims) {
	size_t n = 1;
	for (auto d : dims) {
		n *= static_cast<size_t>(d);
	}
	return n;
}

float HalfToFloat(uint16_t h) {
	const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
	const int exp = (h >> 10) & 0x1f;
	const uint32_t man = h & 0x3ff;
	float v;
	if (exp == 0) {
		v = std::ldexp(static_cast<float>(man), -24);
	} else if (exp == 31) {
		v = man ? std::nanf("") : INFINITY;
	} else {
		v = std::ldexp(static_cast<float>(man | 0x400), exp - 25);
	}
	uint32_t bits;
	std::memcpy(&bits, &v, 4);
	bits |= sign;
	std::memcpy(&v, &bits, 4);
	return v;
}

float Bf16ToFloat(uint16_t h) {
	const uint32_t bits = static_cast<uint32_t>(h) << 16;
	float v;
	std::memcpy(&v, &bits, 4);
	return v;
}

void SetTf32(const std::string &precision) {
	if (precision != "fp32") {
		return;
	}
#if defined(_WIN32)
	if (!std::getenv("NVIDIA_TF32_OVERRIDE")) {
		_putenv_s("NVIDIA_TF32_OVERRIDE", "0");
	}
#else
	setenv("NVIDIA_TF32_OVERRIDE", "0", 0); // a value the user set explicitly wins
#endif
}

struct Backend {
	::anofox::onnxread::Graph graph; // must outlive the executor and stay in place (it reads inline weights from it)
	std::vector<std::string> input_names;
	std::vector<const char *> input_cstrs;
	std::unordered_map<std::string, int> input_types; // ONNX element type of each graph input
	std::unordered_map<std::string, ExternalWeight> weights;
	std::vector<std::vector<float>> upcast; // CPU test build: 16-bit weights as float32
	std::unique_ptr<GgmlExecutor> executor; // declared last: destroyed first
	std::mutex mutex;
	int device_ordinal = 0;
};

const char *PluginName(void) {
	return "ggml";
}

bool IsExternal(const ::anofox::onnxread::Tensor &t) {
	return t.location == ::anofox::onnxread::DataLocation::EXTERNAL ||
	       (t.raw.empty() && t.floats.empty() && t.int64s.empty() && t.int32s.empty() && ElementCount(t.dims) > 0);
}

void *PluginCreate(const AnofoxGgmlCreateParams *params, char *err, size_t err_len) {
	if (!params || ((!params->graph_path || !params->graph_path[0]) && !params->graph_data)) {
		SetError(err, err_len, "no graph given (graph_path or graph_data)");
		return nullptr;
	}
	try {
		const std::string precision = params->precision && params->precision[0] ? params->precision : "fp32";
		if (precision != "fp32" && precision != "tf32") {
			SetError(err, err_len,
			         "precision '" + precision + "' is not supported by the ggml GPU backend (fp32 or tf32); fp16/bf16 would need a graph conversion");
			return nullptr;
		}
		SetTf32(precision);

		auto backend = std::unique_ptr<Backend>(new Backend());
		backend->device_ordinal = params->device_ordinal;
		backend->graph = params->graph_data
		                     ? ::anofox::onnxread::ParseModelBytes(static_cast<const uint8_t *>(params->graph_data), params->graph_size, "graph")
		                     : ::anofox::onnxread::ParseModelFile(params->graph_path);
		std::vector<std::string> ops;
		for (const auto &n : backend->graph.nodes) {
			ops.push_back(n.op_type);
		}
		const std::string refusal = GgmlRefusal(ops, "");
		if (!refusal.empty()) {
			SetError(err, err_len, "the ggml GPU backend cannot run this graph: " + refusal);
			return nullptr;
		}

		// The host's weights, by initializer name: each must match the graph's declaration exactly.
		std::unordered_map<std::string, const AnofoxGgmlWeight *> given;
		for (size_t i = 0; i < params->weight_count; i++) {
			given[params->weights[i].name] = &params->weights[i];
		}
		std::set<std::string> initializer_names;
		for (auto &t : backend->graph.initializers) {
			initializer_names.insert(t.name);
			if (!IsExternal(t)) {
				continue;
			}
			const auto it = given.find(t.name);
			if (it == given.end()) {
				throw std::runtime_error("the host supplied no weight '" + t.name + "', which the graph reads");
			}
			const auto &w = *it->second;
			const size_t count = ElementCount(t.dims);
			if (static_cast<size_t>(w.count) != count) {
				throw std::runtime_error("the weight '" + t.name + "' has " + std::to_string(w.count) + " values, the graph's shape needs " +
				                         std::to_string(count));
			}
			if (w.rank >= 0 && (static_cast<size_t>(w.rank) != t.dims.size() || !std::equal(t.dims.begin(), t.dims.end(), w.shape))) {
				throw std::runtime_error("the weight '" + t.name + "' was supplied with another shape than the graph declares");
			}
			// a 16-bit checkpoint may feed a float32 initializer: the host passes its mapped checkpoint, the upload upcasts it
			const bool upcast = t.data_type == ANOFOX_GGML_DTYPE_FLOAT && (w.dtype == ANOFOX_GGML_DTYPE_FLOAT16 || w.dtype == ANOFOX_GGML_DTYPE_BFLOAT16);
			if (w.dtype != t.data_type && !upcast) {
				throw std::runtime_error("the weight '" + t.name + "' was supplied as ONNX type " + std::to_string(w.dtype) +
				                         ", the graph declares type " + std::to_string(t.data_type));
			}
			if (w.dtype != ANOFOX_GGML_DTYPE_FLOAT && w.dtype != ANOFOX_GGML_DTYPE_FLOAT16 && w.dtype != ANOFOX_GGML_DTYPE_BFLOAT16) {
				throw std::runtime_error("the weight '" + t.name + "' is ONNX type " + std::to_string(w.dtype) +
				                         "; the plugin reads float32, float16 and bfloat16 weights");
			}
#ifdef ANOFOX_GGML_PLUGIN_CPU_TEST
			if (w.dtype != ANOFOX_GGML_DTYPE_FLOAT) {
				std::vector<float> f(count);
				const auto *h = static_cast<const uint16_t *>(w.data);
				for (size_t i = 0; i < count; i++) {
					f[i] = w.dtype == ANOFOX_GGML_DTYPE_FLOAT16 ? HalfToFloat(h[i]) : Bf16ToFloat(h[i]);
				}
				backend->upcast.push_back(std::move(f));
				backend->weights[t.name] = {backend->upcast.back().data(), count, ANOFOX_GGML_DTYPE_FLOAT};
				t.data_type = ANOFOX_GGML_DTYPE_FLOAT; // the graph now reads the float32 copy (as the CPU session's injection does)
				continue;
			}
#endif
			backend->weights[t.name] = {static_cast<const float *>(w.data), count, static_cast<int>(w.dtype)};
		}
		for (const auto &name : backend->graph.inputs) {
			if (!initializer_names.count(name)) {
				backend->input_names.push_back(name);
				const auto type = backend->graph.input_types.find(name);
				backend->input_types[name] = type != backend->graph.input_types.end() ? type->second : ANOFOX_GGML_DTYPE_FLOAT;
			}
		}
		for (const auto &n : backend->input_names) {
			backend->input_cstrs.push_back(n.c_str());
		}
		backend->input_cstrs.push_back(nullptr);

		GgmlExecutorConfig config;
		// on a GPU the threads serve the CPU kernels the GPU backend cannot run (TopK, Where, the float64 islands)
		config.threads = params->threads > 0 ? params->threads : static_cast<int>(std::max(1u, std::min(8u, std::thread::hardware_concurrency())));
#ifndef ANOFOX_GGML_PLUGIN_CPU_TEST
		config.device = "gpu:" + std::to_string(params->device_ordinal);
		config.strict_fp32 = precision == "fp32";
#endif
		if (params->out_of_memory_remedy && params->out_of_memory_remedy[0]) {
			config.out_of_memory_remedy = params->out_of_memory_remedy;
		}
		config.cache_entries = 8;
		auto *self = backend.get();
		config.weights = [self](const std::string &name, const std::vector<int64_t> &dims) -> ExternalWeight {
			const auto it = self->weights.find(name);
			if (it == self->weights.end() || it->second.count != ElementCount(dims)) {
				return ExternalWeight();
			}
			return it->second;
		};
		backend->executor.reset(new GgmlExecutor(backend->graph, std::move(config)));
		// Upload now: a device that cannot hold the model fails at load, not inside the first query.
		backend->executor->PreloadWeights();
		return backend.release();
	} catch (const std::exception &e) {
		SetError(err, err_len, std::string("could not initialise the ggml backend on device ") + std::to_string(params->device_ordinal) + ": " + e.what());
		return nullptr;
	} catch (...) {
		SetError(err, err_len, "could not initialise the ggml backend: unknown error");
		return nullptr;
	}
}

const char *const *PluginGraphInputs(void *handle) {
	return static_cast<Backend *>(handle)->input_cstrs.data();
}

bool IsMemoryError(const std::string &what) {
	return what.find("out of memory") != std::string::npos || what.find("out of device memory") != std::string::npos ||
	       what.find("failed to allocate") != std::string::npos;
}

AnofoxGgmlPluginStatus PluginRun(void *handle, const AnofoxGgmlTensorIn *inputs, size_t input_count, const char *const *outputs,
                                 size_t output_count, AnofoxGgmlRunOutput *result, char *err, size_t err_len) {
	auto *backend = static_cast<Backend *>(handle);
	if (!backend || !result || (input_count && !inputs) || (output_count && !outputs)) {
		SetError(err, err_len, "null argument");
		return ANOFOX_GGML_PLUGIN_ERROR;
	}
	result->tensors = nullptr;
	result->count = 0;
	try {
		HostTensors in;
		GgmlRunOptions options;
		std::set<std::string> seen;
		for (size_t i = 0; i < input_count; i++) {
			const auto &t = inputs[i];
			const std::string name = t.name ? t.name : "";
			if (!backend->input_types.count(name)) {
				throw std::runtime_error("the graph declares no input '" + name + "'");
			}
			seen.insert(name);
			const std::vector<int64_t> shape(t.shape, t.shape + t.rank);
			const size_t count = ElementCount(shape);
			if (t.fold) {
				if (t.dtype != ANOFOX_GGML_DTYPE_INT64) {
					throw std::runtime_error("the folded input '" + name + "' must be int64");
				}
				const auto *v = static_cast<const int64_t *>(t.data);
				options.int_inputs[name] = {shape, std::vector<int64_t>(v, v + count)};
				continue;
			}
			HostTensor h;
			h.shape = shape;
			h.data.resize(count);
			switch (t.dtype) {
			case ANOFOX_GGML_DTYPE_FLOAT:
				std::memcpy(h.data.data(), t.data, count * sizeof(float));
				break;
			case ANOFOX_GGML_DTYPE_INT64: { // exact below 2^24; the graph's input type gives it integer semantics
				const auto *v = static_cast<const int64_t *>(t.data);
				for (size_t j = 0; j < count; j++) {
					h.data[j] = static_cast<float>(v[j]);
				}
				break;
			}
			case ANOFOX_GGML_DTYPE_BOOL:
			case ANOFOX_GGML_DTYPE_UINT8: {
				const auto *v = static_cast<const uint8_t *>(t.data);
				for (size_t j = 0; j < count; j++) {
					h.data[j] = v[j] ? 1.0f : 0.0f;
				}
				if (t.dtype == ANOFOX_GGML_DTYPE_BOOL) {
					options.bool_inputs.insert(name);
				}
				break;
			}
			default:
				throw std::runtime_error("the input '" + name + "' has ONNX type " + std::to_string(t.dtype) + ", which the plugin does not take");
			}
			in[name] = std::move(h);
		}
		for (const auto &name : backend->input_names) {
			if (!seen.count(name)) {
				throw std::runtime_error("the input '" + name + "', which the graph declares, was not given");
			}
		}
		for (size_t o = 0; o < output_count; o++) {
			options.outputs.push_back(outputs[o]);
		}
		HostTensors out;
		{
			std::lock_guard<std::mutex> guard(backend->mutex);
			out = backend->executor->Run(in, options);
		}
		result->tensors = static_cast<AnofoxGgmlTensorOut *>(std::calloc(std::max<size_t>(1, output_count), sizeof(AnofoxGgmlTensorOut)));
		if (!result->tensors) {
			throw std::bad_alloc();
		}
		for (size_t o = 0; o < output_count; o++) {
			const auto found = out.find(outputs[o]);
			if (found == out.end()) {
				throw std::runtime_error(std::string("the engine did not return the output '") + outputs[o] + "'");
			}
			const HostTensor &h = found->second;
			auto &t = result->tensors[result->count++];
			t.name = static_cast<char *>(std::malloc(std::strlen(outputs[o]) + 1));
			t.shape = static_cast<int64_t *>(std::malloc(sizeof(int64_t) * std::max<size_t>(1, h.shape.size())));
			t.data = static_cast<float *>(std::malloc(sizeof(float) * std::max<size_t>(1, h.data.size())));
			if (!t.name || !t.shape || !t.data) {
				throw std::bad_alloc();
			}
			std::strcpy(t.name, outputs[o]);
			std::memcpy(t.shape, h.shape.data(), sizeof(int64_t) * h.shape.size());
			std::memcpy(t.data, h.data.data(), sizeof(float) * h.data.size());
			t.rank = static_cast<int32_t>(h.shape.size());
			t.count = static_cast<int64_t>(h.data.size());
		}
		return ANOFOX_GGML_PLUGIN_OK;
	} catch (const std::bad_alloc &) {
		SetError(err, err_len, "host allocation failed: out of memory");
		return ANOFOX_GGML_PLUGIN_OUT_OF_MEMORY;
	} catch (const std::exception &e) {
		const std::string what = e.what();
		SetError(err, err_len, "ggml inference failed on device " + std::to_string(backend->device_ordinal) + ": " + what);
		return IsMemoryError(what) ? ANOFOX_GGML_PLUGIN_OUT_OF_MEMORY : ANOFOX_GGML_PLUGIN_ERROR;
	} catch (...) {
		SetError(err, err_len, "ggml inference failed: unknown error");
		return ANOFOX_GGML_PLUGIN_ERROR;
	}
}

void PluginFreeOutput(AnofoxGgmlRunOutput *result) {
	if (!result || !result->tensors) {
		return;
	}
	for (size_t i = 0; i < result->count; i++) {
		std::free(result->tensors[i].name);
		std::free(result->tensors[i].shape);
		std::free(result->tensors[i].data);
	}
	std::free(result->tensors);
	result->tensors = nullptr;
	result->count = 0;
}

void PluginDestroy(void *handle) {
	delete static_cast<Backend *>(handle);
}

const AnofoxGgmlPluginApi kApi = {ANOFOX_GGML_PLUGIN_ABI_VERSION, ANOFOX_GGML_PLUGIN_LAYOUT, PluginName, PluginCreate, PluginGraphInputs,
                                  PluginRun,                      PluginFreeOutput, PluginDestroy};

} // namespace

extern "C" ANOFOX_GGML_PLUGIN_EXPORT const AnofoxGgmlPluginApi *AnofoxGgmlGetPluginApi(void) {
	return &kApi;
}
