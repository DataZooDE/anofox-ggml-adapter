/*===----------------------------------------------------------------------===
 * A backend plugin that computes nothing real, so the LOADER and the ABI can be tested without a GPU.
 *
 * Deterministic functions of its inputs, so a test asserts the data made the round trip rather than that nothing threw:
 *   y = 2 * x + sum(ids)        (x: float input, ids: int64 data input)
 *   k = the folded input "k"    (proves the fold flag and integer values cross)
 *   w = the first weight, as float32 (proves host weights cross, float16 upcast included)
 *   m = the mask "mask" as 0/1 floats (bool crosses as one byte per element)
 * Triggers: arch "refuse" fails create; model "oom" makes run report out of memory; model "fail" makes run fail.
 * Built twice; ANOFOX_FAKE_PLUGIN_BAD_ABI builds the library with another ABI version, so the refusal is tested for real.
 *===----------------------------------------------------------------------===*/

#include "anofox_ggml_adapter/plugin_abi.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct FakeBackend {
	std::string model;
	std::vector<float> weight; // the first weight, as float32
	std::vector<const char *> inputs {"x", "ids", "k", "mask", nullptr};
};

void SetError(char *err, size_t err_len, const char *message) {
	if (err && err_len) {
		std::strncpy(err, message, err_len - 1);
		err[err_len - 1] = '\0';
	}
}

float HalfToFloat(uint16_t h) {
	const int sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
	float v = exp == 0 ? std::ldexp(static_cast<float>(man), -24) : std::ldexp(static_cast<float>(man | 0x400), exp - 25);
	return sign ? -v : v;
}

const char *PluginName(void) {
	return "fake";
}

void *PluginCreate(const AnofoxGgmlCreateParams *params, char *err, size_t err_len) {
	if (!params || ((!params->graph_path || !*params->graph_path) && !params->graph_data)) {
		SetError(err, err_len, "no graph supplied");
		return nullptr;
	}
	if (params->arch && std::strcmp(params->arch, "refuse") == 0) {
		SetError(err, err_len, "no device matching arch 'refuse'");
		return nullptr;
	}
	auto *b = new FakeBackend();
	b->model = params->model ? params->model : "";
	if (params->weight_count > 0) {
		const auto &w = params->weights[0];
		for (int64_t i = 0; i < w.count; i++) {
			b->weight.push_back(w.dtype == ANOFOX_GGML_DTYPE_FLOAT16 ? HalfToFloat(static_cast<const uint16_t *>(w.data)[i])
			                                                         : static_cast<const float *>(w.data)[i]);
		}
	}
	return b;
}

const char *const *PluginGraphInputs(void *handle) {
	return static_cast<FakeBackend *>(handle)->inputs.data();
}

const AnofoxGgmlTensorIn *Find(const AnofoxGgmlTensorIn *inputs, size_t n, const char *name) {
	for (size_t i = 0; i < n; i++) {
		if (std::strcmp(inputs[i].name, name) == 0) {
			return &inputs[i];
		}
	}
	return nullptr;
}

int64_t Count(const AnofoxGgmlTensorIn &t) {
	int64_t c = 1;
	for (int32_t i = 0; i < t.rank; i++) {
		c *= t.shape[i];
	}
	return c;
}

void Fill(AnofoxGgmlTensorOut &o, const char *name, std::vector<int64_t> shape, const std::vector<float> &values) {
	o.name = static_cast<char *>(std::malloc(std::strlen(name) + 1));
	std::strcpy(o.name, name);
	o.rank = static_cast<int32_t>(shape.size());
	o.shape = static_cast<int64_t *>(std::malloc(sizeof(int64_t) * (shape.empty() ? 1 : shape.size())));
	for (size_t i = 0; i < shape.size(); i++) {
		o.shape[i] = shape[i];
	}
	o.count = static_cast<int64_t>(values.size());
	o.data = static_cast<float *>(std::malloc(sizeof(float) * (values.empty() ? 1 : values.size())));
	for (size_t i = 0; i < values.size(); i++) {
		o.data[i] = values[i];
	}
}

AnofoxGgmlPluginStatus PluginRun(void *handle, const AnofoxGgmlTensorIn *inputs, size_t input_count, const char *const *outputs,
                                 size_t output_count, AnofoxGgmlRunOutput *result, char *err, size_t err_len) {
	auto *b = static_cast<FakeBackend *>(handle);
	result->tensors = nullptr;
	result->count = 0;
	if (b->model == "oom") {
		SetError(err, err_len, "cudaMalloc failed: out of memory");
		return ANOFOX_GGML_PLUGIN_OUT_OF_MEMORY;
	}
	if (b->model == "fail") {
		SetError(err, err_len, "kernel launch failed");
		return ANOFOX_GGML_PLUGIN_ERROR;
	}
	result->tensors = static_cast<AnofoxGgmlTensorOut *>(std::calloc(output_count ? output_count : 1, sizeof(AnofoxGgmlTensorOut)));
	for (size_t o = 0; o < output_count; o++) {
		const std::string want = outputs[o];
		auto &out = result->tensors[result->count];
		if (want == "y") {
			const auto *x = Find(inputs, input_count, "x");
			const auto *ids = Find(inputs, input_count, "ids");
			if (!x || x->dtype != ANOFOX_GGML_DTYPE_FLOAT || !ids || ids->dtype != ANOFOX_GGML_DTYPE_INT64) {
				SetError(err, err_len, "y needs a float x and an int64 ids");
				return ANOFOX_GGML_PLUGIN_ERROR;
			}
			float sum = 0;
			for (int64_t i = 0; i < Count(*ids); i++) {
				sum += static_cast<float>(static_cast<const int64_t *>(ids->data)[i]);
			}
			std::vector<float> v;
			for (int64_t i = 0; i < Count(*x); i++) {
				v.push_back(2 * static_cast<const float *>(x->data)[i] + sum);
			}
			Fill(out, "y", std::vector<int64_t>(x->shape, x->shape + x->rank), v);
		} else if (want == "k") {
			const auto *k = Find(inputs, input_count, "k");
			if (!k || !k->fold || k->dtype != ANOFOX_GGML_DTYPE_INT64) {
				SetError(err, err_len, "k must be a folded int64");
				return ANOFOX_GGML_PLUGIN_ERROR;
			}
			Fill(out, "k", {}, {static_cast<float>(*static_cast<const int64_t *>(k->data))});
		} else if (want == "w") {
			Fill(out, "w", {static_cast<int64_t>(b->weight.size())}, b->weight);
		} else if (want == "m") {
			const auto *m = Find(inputs, input_count, "mask");
			if (!m || m->dtype != ANOFOX_GGML_DTYPE_BOOL) {
				SetError(err, err_len, "mask must be bool");
				return ANOFOX_GGML_PLUGIN_ERROR;
			}
			std::vector<float> v;
			for (int64_t i = 0; i < Count(*m); i++) {
				v.push_back(static_cast<const uint8_t *>(m->data)[i] ? 1.0f : 0.0f);
			}
			Fill(out, "m", std::vector<int64_t>(m->shape, m->shape + m->rank), v);
		} else {
			continue; // an output this "graph" does not have: the host must notice it is missing
		}
		result->count++;
	}
	return ANOFOX_GGML_PLUGIN_OK;
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
	delete static_cast<FakeBackend *>(handle);
}

#ifdef ANOFOX_FAKE_PLUGIN_BAD_ABI
constexpr int kAbi = ANOFOX_GGML_PLUGIN_ABI_VERSION + 1;
#else
constexpr int kAbi = ANOFOX_GGML_PLUGIN_ABI_VERSION;
#endif
#ifdef ANOFOX_FAKE_PLUGIN_BAD_LAYOUT
const uint32_t kLayout = ANOFOX_GGML_PLUGIN_LAYOUT + 1; // a plugin built from another revision of the header
#else
const uint32_t kLayout = ANOFOX_GGML_PLUGIN_LAYOUT;
#endif

const AnofoxGgmlPluginApi kApi = {kAbi, kLayout, PluginName, PluginCreate, PluginGraphInputs, PluginRun, PluginFreeOutput, PluginDestroy};

} // namespace

extern "C" ANOFOX_GGML_PLUGIN_EXPORT const AnofoxGgmlPluginApi *AnofoxGgmlGetPluginApi(void) {
	return &kApi;
}
