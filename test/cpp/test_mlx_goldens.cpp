//===----------------------------------------------------------------------===//
// The MLX graph interpreter (src/mlxgraph) against the same ONNX Runtime goldens the ggml engine is held to
// (test/fixtures/ggml_ops, from tools/ggml_parity/gen_op_goldens.py). Built only on Apple Silicon, by
// tools/mlx_plugin with -DANOFOX_MLX_TESTS=ON (target mlx_tests); run it from the repository root.
//
// It checks the ops the interpreter gained for anofox-decide's graphs (Flatten, Split). Both only move data, so they
// are compared bit for bit: every golden case of theirs must run, and must match exactly.
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "anofox_ggml_adapter/onnx_reader.hpp"
#include "mlxgraph/mlx_graph.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

namespace fs = std::filesystem;
using anofox::mlxgraph::Graph;
using anofox::mlxgraph::Tensor;

namespace {

struct GoldenTensor {
	std::string dtype;
	Tensor t;
};

GoldenTensor ParseTensor(std::istringstream &line) {
	GoldenTensor g;
	std::string dims;
	line >> g.dtype >> dims;
	g.t.dtype = g.dtype == "f32" ? 1 : g.dtype == "i64" ? 7 : 9;
	if (dims != "-") {
		std::stringstream ds(dims);
		std::string tok;
		while (std::getline(ds, tok, ',')) {
			g.t.shape.push_back(std::stoi(tok));
		}
	}
	std::string hex;
	while (line >> hex) {
		if (g.dtype == "bool") {
			g.t.data.push_back(hex == "1" ? 1.0f : 0.0f);
		} else if (g.dtype == "i64") {
			g.t.data.push_back(static_cast<float>(std::stoll(hex)));
		} else {
			const uint32_t bits = static_cast<uint32_t>(std::stoul(hex, nullptr, 16));
			float f;
			std::memcpy(&f, &bits, 4);
			g.t.data.push_back(f);
		}
	}
	return g;
}

struct Case {
	std::string op;
	std::vector<std::pair<std::string, GoldenTensor>> inputs, outputs;
};

Case ReadCase(const std::string &stem) {
	Case c;
	std::ifstream f(stem + ".golden");
	REQUIRE(f.good());
	std::string text;
	while (std::getline(f, text)) {
		std::istringstream line(text);
		std::string kind, name;
		line >> kind;
		if (kind == "#") {
			std::string tag;
			line >> tag >> c.op;
			continue;
		}
		line >> name;
		(kind == "in" ? c.inputs : c.outputs).emplace_back(name, ParseTensor(line));
	}
	return c;
}

//! NaN equals NaN (any payload); everything else bit for bit, so +0 vs -0 is a failure.
bool SameBits(float got, float want) {
	if (std::isnan(want)) {
		return std::isnan(got);
	}
	uint32_t a, b;
	std::memcpy(&a, &got, 4);
	std::memcpy(&b, &want, 4);
	return a == b;
}

} // namespace

TEST_CASE("mlx goldens: Flatten and Split match ONNX Runtime bit for bit", "[adapter][mlx]") {
	const std::set<std::string> checked = {"Flatten", "Split"};
	std::map<std::string, int> ran;
	for (const auto &entry : fs::directory_iterator("test/fixtures/ggml_ops")) {
		if (entry.path().extension() != ".golden") {
			continue;
		}
		const std::string stem = (entry.path().parent_path() / entry.path().stem()).string();
		const Case c = ReadCase(stem);
		if (!checked.count(c.op)) {
			continue;
		}
		INFO("case " << entry.path().stem().string());
		Graph graph(stem + ".onnx", nullptr, 0, {});
		REQUIRE(graph.MissingOps().empty());
		std::vector<std::pair<std::string, Tensor>> feeds;
		for (const auto &in : c.inputs) {
			feeds.emplace_back(in.first, in.second.t);
		}
		std::vector<std::string> names;
		for (const auto &out : c.outputs) {
			names.push_back(out.first);
		}
		const auto got = graph.Run(feeds, names);
		for (const auto &out : c.outputs) {
			INFO("output " << out.first);
			const auto it = got.find(out.first);
			REQUIRE(it != got.end());
			REQUIRE(it->second.shape == out.second.t.shape);
			REQUIRE(it->second.data.size() == out.second.t.data.size());
			for (size_t i = 0; i < out.second.t.data.size(); i++) {
				INFO("element " << i << ": got " << it->second.data[i] << ", want " << out.second.t.data[i]);
				REQUIRE(SameBits(it->second.data[i], out.second.t.data[i]));
			}
		}
		ran[c.op]++;
	}
	// every case of both ops ran: the goldens hold 6 Flatten and 10 Split cases
	REQUIRE(ran["Flatten"] >= 6);
	REQUIRE(ran["Split"] >= 10);
}

TEST_CASE("mlx goldens: Split with num_outputs refuses a last part below zero, as ONNX and the ggml engine do", "[adapter][mlx]") {
	// x [5], num_outputs 4: ceil(5/4) = 2 per part leaves -1 for the last (tools/ggml_parity/make_mlx_fixtures.py)
	Graph graph("test/fixtures/mlx/split_num_outputs_negative_last.onnx", nullptr, 0, {});
	Tensor x;
	x.shape = {5};
	x.data = {1, 2, 3, 4, 5};
	REQUIRE_THROWS_WITH(graph.Run({{"x", x}}, {}), Catch::Contains("Split"));
}
