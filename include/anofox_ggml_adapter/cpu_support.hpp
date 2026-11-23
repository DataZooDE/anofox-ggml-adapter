#pragma once

//===----------------------------------------------------------------------===//
// cpu_support.hpp — can this CPU run the ggml CPU backend this build contains?
//===----------------------------------------------------------------------===//

#include <string>

namespace anofox {
namespace ggml_adapter {

//! The ggml CPU backend is compiled for AVX2 + FMA + F16C + BMI2 on x86 (no runtime dispatch in a static build): on an older CPU its first
//! matmul would kill the host process with SIGILL. Returns the reason to refuse, or "" when the CPU is fine (always on other architectures).
//! The host turns it into its own error, naming its way out.
std::string UnsupportedCpuReason();

} // namespace ggml_adapter
} // namespace anofox
