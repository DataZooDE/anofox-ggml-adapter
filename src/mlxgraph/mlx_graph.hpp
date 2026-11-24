/*===----------------------------------------------------------------------===
 *                         anofox-tabfm
 *
 * mlx_graph.hpp — execute a shipped ONNX graph on Apple MLX.
 *
 * docs/MLX_PLAN.md route 3, promoted to the plan of record when the goal became
 * "every registered model on MLX". The alternative was seven hand-ported
 * forwards — seven second implementations of models we already ship, each free
 * to drift from the graph every other backend runs. This runs the graph itself,
 * so a new model costs no new math and drift is impossible by construction.
 *
 * Scope was measured before any of it was written (docs/MLX_SPIKE_RESULTS.md):
 * 13 graphs, 4.5k–12.4k nodes, 33–48 distinct ops each, 64 in union. The Python
 * prototype in tools/mlx_spike settled every op's semantics against ORT CPU on
 * real weights first; this is that prototype in C++ over mlx-c.
 *
 * Two things about these graphs shape the whole design:
 *
 *   - SHAPES ARE DYNAMIC. They were produced by torch.export with dynamic rows
 *     and features, so they compute shapes at runtime (Shape/Slice/Concat) and
 *     feed them to Reshape/Expand. Integer tensors must therefore stay integers;
 *     a Shape result quietly promoted to fp32 breaks any dimension past 2^24.
 *   - INITIALIZERS LIVE OUTSIDE THE FILE. The ext graphs name model.safetensors
 *     and an absolute byte offset. They are mapped, never copied — which is what
 *     lets tabfm-v1's 6.6 GB load at all on a 16 GB machine.
 *===----------------------------------------------------------------------===*/

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace anofox {
namespace mlxgraph {

//! One input or output tensor, as the caller sees it. The values are held as float32; `dtype` (an ONNX element type) says what they
//! are: 1 float, 7 int64 (exact below 2^24: token ids, sizes), 9 bool (0/1). Outputs are always float32.
struct Tensor {
	std::vector<int> shape;
	std::vector<float> data;
	int dtype = 1;
};

//! A weight the host supplies for an external initializer, by its name: the host's memory, alive as long as the Graph.
struct Weight {
	const void *data = nullptr;
	int dtype = 1; //!< 1 float32, 10 float16, 16 bfloat16
	int64_t count = 0;
	std::vector<int64_t> shape; //!< optional: checked against the graph's declaration when given
};

//! A loaded graph. Construction parses the proto, maps the weights and builds
//! the node list; Run() executes it. Not thread-safe by itself — MLX streams
//! are thread-local, so Run() acquires the calling thread's stream each time.
class Graph {
public:
	//! The graph from `graph_path`, or from `graph_size` bytes at `graph_data` when that is set; its external initializers are served
	//! from `weights` by name (each must be there, with the element count and type the graph declares). Throws std::runtime_error.
	Graph(const std::string &graph_path, const void *graph_data, size_t graph_size, const std::map<std::string, Weight> &weights);
	~Graph();
	Graph(const Graph &) = delete;
	Graph &operator=(const Graph &) = delete;

	//! Named feeds in, the named outputs back (all graph outputs when `outputs` is empty).
	std::map<std::string, Tensor> Run(const std::vector<std::pair<std::string, Tensor>> &feeds, const std::vector<std::string> &outputs) const;

	//! Input names in graph order, so the caller can tell the two model
	//! families apart (train_size-scalar vs single_eval_pos) without guessing.
	const std::vector<std::string> &InputNames() const;

	//! Ops this graph needs that are not implemented. Empty means runnable.
	std::vector<std::string> MissingOps() const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

} // namespace mlxgraph
} // namespace anofox
