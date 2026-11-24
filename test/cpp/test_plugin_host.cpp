// The host side of the plugin ABI, against a real dlopen of the fake plugin fixture (no GPU needed).

#include "catch.hpp"

#include "anofox_ggml_adapter/plugin_host.hpp"

#include <cstring>
#include <functional>

using namespace anofox::ggml_adapter;

#ifndef ANOFOX_FAKE_PLUGIN
#error "ANOFOX_FAKE_PLUGIN must name the fixture library"
#endif

namespace {

PluginCreate Params() {
	PluginCreate p;
	p.graph_path = "graph.onnx"; // the fake plugin reads nothing; a real one reads this file
	p.arch = "sm_89";
	return p;
}

PluginError::Kind KindOf(const std::function<void()> &fn) {
	try {
		fn();
	} catch (const PluginError &e) {
		return e.kind;
	}
	FAIL("expected a PluginError");
	return PluginError::Kind::RUN_FAILED;
}

} // namespace

TEST_CASE("plugin_host: named tensors, folded integers, bool masks and host weights cross the boundary", "[adapter][plugin]") {
	auto params = Params();
	const uint16_t half_weights[3] = {0x3c00, 0x4000, 0xbc00}; // 1, 2, -1 as float16
	params.weights.push_back({"w0", ANOFOX_GGML_DTYPE_FLOAT16, half_weights, 3});
	auto session = PluginSession::Open(ANOFOX_FAKE_PLUGIN, params);
	REQUIRE(session->Name() == "fake");
	REQUIRE(session->GraphInputs() == std::vector<std::string> {"x", "ids", "k", "mask"});

	const float x[4] = {0.5f, 1.0f, -2.0f, 3.0f};
	const int64_t ids[3] = {1, 2, 262143}; // a 256k-vocabulary id crosses as int64, not float
	const int64_t k = 17;
	const uint8_t mask[2] = {1, 0};
	std::vector<PluginTensor> in = {{"x", ANOFOX_GGML_DTYPE_FLOAT, {2, 2}, x},
	                                {"ids", ANOFOX_GGML_DTYPE_INT64, {1, 3}, ids},
	                                {"k", ANOFOX_GGML_DTYPE_INT64, {}, &k, true},
	                                {"mask", ANOFOX_GGML_DTYPE_BOOL, {2}, mask}};
	const auto out = session->Run(in, {"y", "k", "w", "m"});
	REQUIRE(out.at("y").shape == std::vector<int64_t> {2, 2});
	REQUIRE(out.at("y").data == std::vector<float> {1.0f + 262146, 2.0f + 262146, -4.0f + 262146, 6.0f + 262146});
	REQUIRE(out.at("k").shape.empty());
	REQUIRE(out.at("k").data == std::vector<float> {17.0f});
	REQUIRE(out.at("w").data == std::vector<float> {1.0f, 2.0f, -1.0f}); // float16 upcast by the plugin
	REQUIRE(out.at("m").data == std::vector<float> {1.0f, 0.0f});
}

TEST_CASE("plugin_host: every failure is a PluginError with a kind the host can word", "[adapter][plugin]") {
	SECTION("a missing file is not loadable, with the loader's reason") {
		std::string why;
		REQUIRE_FALSE(PluginLoadable("/nonexistent/libnope.so", &why));
		REQUIRE_FALSE(why.empty());
		REQUIRE(KindOf([] { PluginSession::Open("/nonexistent/libnope.so", Params()); }) == PluginError::Kind::NOT_LOADABLE);
	}
	SECTION("a library built against another ABI is refused before any other field is read") {
		std::string why;
		REQUIRE_FALSE(PluginLoadable(ANOFOX_FAKE_PLUGIN_BAD_ABI, &why));
		REQUIRE(why.find("ABI version " + std::to_string(ANOFOX_GGML_PLUGIN_ABI_VERSION + 1)) != std::string::npos);
		try {
			PluginSession::Open(ANOFOX_FAKE_PLUGIN_BAD_ABI, Params());
			FAIL("expected a refusal");
		} catch (const PluginError &e) {
			REQUIRE(e.kind == PluginError::Kind::ABI_MISMATCH);
			REQUIRE(e.plugin_abi == ANOFOX_GGML_PLUGIN_ABI_VERSION + 1);
		}
	}
	SECTION("a library that is no plugin") {
		REQUIRE(KindOf([] { PluginSession::Open(ANOFOX_NOT_A_PLUGIN, Params()); }) == PluginError::Kind::NOT_A_PLUGIN);
	}
	SECTION("create refuses: the plugin's reason and name are kept") {
		auto p = Params();
		p.arch = "refuse";
		try {
			PluginSession::Open(ANOFOX_FAKE_PLUGIN, p);
			FAIL("expected a refusal");
		} catch (const PluginError &e) {
			REQUIRE(e.kind == PluginError::Kind::CREATE_FAILED);
			REQUIRE(e.backend == "fake");
			REQUIRE(e.detail == "no device matching arch 'refuse'");
		}
	}
	SECTION("a run that runs out of device memory is told apart from any other failure") {
		auto p = Params();
		p.model = "oom";
		auto oom = PluginSession::Open(ANOFOX_FAKE_PLUGIN, p);
		REQUIRE(KindOf([&] { oom->Run({}, {"y"}); }) == PluginError::Kind::OUT_OF_MEMORY);
		p.model = "fail";
		auto fail = PluginSession::Open(ANOFOX_FAKE_PLUGIN, p);
		REQUIRE(KindOf([&] { fail->Run({}, {"y"}); }) == PluginError::Kind::RUN_FAILED);
	}
	SECTION("an output the plugin does not return is an error, not a missing map entry") {
		auto session = PluginSession::Open(ANOFOX_FAKE_PLUGIN, Params());
		REQUIRE(KindOf([&] { session->Run({}, {"nope"}); }) == PluginError::Kind::RUN_FAILED);
	}
	SECTION("a plugin is loadable") {
		REQUIRE(PluginLoadable(ANOFOX_FAKE_PLUGIN));
	}
}
