// The generic ggml plugin (src/plugin_ggml.cpp), built in its CPU test mode so CI runs the real plugin end to end without a GPU: the
// same create/run code the CUDA and ROCm builds run, the same host-weight path, only the executor's device differs.

#include "catch.hpp"

#include "anofox_ggml_adapter/plugin_host.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>

using namespace anofox::ggml_adapter;

#ifndef ANOFOX_GGML_CPU_PLUGIN
#error "ANOFOX_GGML_CPU_PLUGIN must name the CPU test build of the ggml plugin"
#endif

namespace {

uint16_t FloatToHalf(float f) { // exact for the small integers and halves used here
	uint32_t x;
	std::memcpy(&x, &f, 4);
	const uint32_t sign = (x >> 16) & 0x8000;
	const int exp = static_cast<int>((x >> 23) & 0xff) - 127 + 15;
	const uint32_t man = (x >> 13) & 0x3ff;
	if ((x & 0x7fffffff) == 0) {
		return static_cast<uint16_t>(sign);
	}
	return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | man);
}

struct Fixture {
	std::vector<float> W;     // [4, 3]
	std::vector<float> E;     // [10, 3], as float
	std::vector<uint16_t> E16; // the same, as float16
	Fixture() {
		for (int i = 0; i < 12; i++) {
			W.push_back(0.25f * static_cast<float>(i) - 1.0f);
		}
		for (int i = 0; i < 30; i++) {
			E.push_back(0.5f * static_cast<float>(i % 7) - 1.5f);
			E16.push_back(FloatToHalf(E.back()));
		}
	}
	PluginCreate Params() const {
		PluginCreate p;
		p.graph_path = "test/fixtures/plugin/graph.onnx";
		p.weights = {{"W", ANOFOX_GGML_DTYPE_FLOAT, W.data(), 12}, {"E", ANOFOX_GGML_DTYPE_FLOAT16, E16.data(), 30}};
		p.model = "fixture";
		p.threads = 2;
		return p;
	}
	std::vector<float> S(const std::vector<float> &x, const std::vector<int64_t> &ids) const {
		std::vector<float> s;
		for (size_t b = 0; b < ids.size(); b++) {
			for (int j = 0; j < 3; j++) {
				float v = 0;
				for (int i = 0; i < 4; i++) {
					v += x[b * 4 + i] * W[i * 3 + j];
				}
				s.push_back(v + E[static_cast<size_t>(ids[b]) * 3 + j]);
			}
		}
		return s;
	}
};

std::vector<PluginTensor> Inputs(const std::vector<float> &x, const std::vector<int64_t> &ids, const std::vector<uint8_t> &mask,
                                 const int64_t &k) {
	const int64_t b = static_cast<int64_t>(ids.size());
	return {{"x", ANOFOX_GGML_DTYPE_FLOAT, {b, 4}, x.data()},
	        {"ids", ANOFOX_GGML_DTYPE_INT64, {b}, ids.data()},
	        {"mask", ANOFOX_GGML_DTYPE_BOOL, {b}, mask.data()},
	        {"k", ANOFOX_GGML_DTYPE_INT64, {}, &k, true}};
}

} // namespace

TEST_CASE("plugin_ggml: the generic plugin runs a graph whose weights the host supplies by name", "[adapter][plugin]") {
	Fixture f;
	auto session = PluginSession::Open(ANOFOX_GGML_CPU_PLUGIN, f.Params());
	REQUIRE(session->Name() == "ggml");
	REQUIRE(session->GraphInputs() == std::vector<std::string> {"x", "ids", "mask", "k"});

	const std::vector<float> x = {1, 0, 2, -1, 0.5f, 0.5f, 0, 3, -2, 1, 1, 1};
	const std::vector<int64_t> ids = {9, 0, 4};
	const std::vector<uint8_t> mask = {1, 0, 1};
	const int64_t k = 2;
	const auto out = session->Run(Inputs(x, ids, mask, k), {"s", "y", "head"});
	const auto s = f.S(x, ids);
	REQUIRE(out.at("s").shape == std::vector<int64_t> {3, 3});
	for (size_t i = 0; i < s.size(); i++) {
		REQUIRE(std::fabs(out.at("s").data[i] - s[i]) <= 1e-5f);
		const bool kept = mask[i / 3] != 0;
		REQUIRE(out.at("y").data[i] == (kept ? out.at("s").data[i] : 0.0f));
	}
	// the folded k decides head's shape: a different k is a different compiled graph, not a wrong slice
	REQUIRE(out.at("head").shape == std::vector<int64_t> {2, 3});
	const int64_t k1 = 1;
	REQUIRE(session->Run(Inputs(x, ids, mask, k1), {"head"}).at("head").shape == std::vector<int64_t> {1, 3});
}

TEST_CASE("plugin_ggml: the graph may come as bytes, and a missing or misshapen weight fails create by name", "[adapter][plugin]") {
	Fixture f;
	std::ifstream file("test/fixtures/plugin/graph.onnx", std::ios::binary);
	const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	auto p = f.Params();
	p.graph_path.clear();
	p.graph_data = bytes.data();
	p.graph_size = bytes.size();
	auto session = PluginSession::Open(ANOFOX_GGML_CPU_PLUGIN, p);
	REQUIRE(session->GraphInputs().size() == 4);

	auto missing = f.Params();
	missing.weights.pop_back();
	try {
		PluginSession::Open(ANOFOX_GGML_CPU_PLUGIN, missing);
		FAIL("expected a refusal");
	} catch (const PluginError &e) {
		REQUIRE(e.kind == PluginError::Kind::CREATE_FAILED);
		REQUIRE(e.detail.find("'E'") != std::string::npos);
	}
	auto wrong = f.Params();
	wrong.weights[0].count = 11;
	try {
		PluginSession::Open(ANOFOX_GGML_CPU_PLUGIN, wrong);
		FAIL("expected a refusal");
	} catch (const PluginError &e) {
		REQUIRE(e.kind == PluginError::Kind::CREATE_FAILED);
		REQUIRE(e.detail.find("'W'") != std::string::npos);
	}
	auto bf = f.Params();
	bf.precision = "fp16";
	try {
		PluginSession::Open(ANOFOX_GGML_CPU_PLUGIN, bf);
		FAIL("expected a refusal");
	} catch (const PluginError &e) {
		REQUIRE(e.detail.find("fp16") != std::string::npos);
	}
}

TEST_CASE("plugin_ggml: an input the graph does not declare, or a missing one, is named", "[adapter][plugin]") {
	Fixture f;
	auto session = PluginSession::Open(ANOFOX_GGML_CPU_PLUGIN, f.Params());
	const std::vector<float> x(4, 1.0f);
	const std::vector<int64_t> ids = {1};
	const std::vector<uint8_t> mask = {1};
	const int64_t k = 1;
	auto in = Inputs(x, ids, mask, k);
	in.pop_back(); // no k
	try {
		session->Run(in, {"s"});
		FAIL("expected a refusal");
	} catch (const PluginError &e) {
		REQUIRE(e.kind == PluginError::Kind::RUN_FAILED);
		REQUIRE(e.detail.find("'k'") != std::string::npos);
	}
	in = Inputs(x, ids, mask, k);
	in.push_back({"extra", ANOFOX_GGML_DTYPE_FLOAT, {1}, x.data()});
	try {
		session->Run(in, {"s"});
		FAIL("expected a refusal");
	} catch (const PluginError &e) {
		REQUIRE(e.detail.find("'extra'") != std::string::npos);
	}
}
