#pragma once

//===----------------------------------------------------------------------===//
// ggml_engine.hpp — the ggml CPU engine (docs/GGML_PLAN.md).
//
// P0 is the skeleton: the refusal contract. The backend itself (GgmlBackend : TabFMBackend) arrives with P5.
//===----------------------------------------------------------------------===//

#include <string>
#include <vector>

namespace anofox {
namespace ggml_adapter {

//! Empty when every op in `op_types` is implemented; otherwise the error text naming the missing ops.
//! `remedy` is the host's advice appended to the message.
std::string GgmlRefusal(const std::vector<std::string> &op_types, const std::string &remedy = "Please report these op names.");

} // namespace ggml_adapter
} // namespace anofox
