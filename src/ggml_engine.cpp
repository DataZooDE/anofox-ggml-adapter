#include "anofox_ggml_adapter/ggml_engine.hpp"

#include "anofox_ggml_adapter/ggml_ops.hpp"

namespace anofox {
namespace ggml_adapter {

std::string GgmlRefusal(const std::vector<std::string> &op_types, const std::string &remedy) {
	const auto missing = GgmlMissingOps(op_types);
	if (missing.empty()) {
		return std::string();
	}
	std::string list;
	for (const auto &op : missing) {
		list += (list.empty() ? "" : ", ") + op;
	}
	return "this graph needs ONNX ops the ggml CPU engine does not implement (" + list + "). " + remedy;
}

} // namespace ggml_adapter
} // namespace anofox
