#include "anofox_ggml_adapter/cpu_support.hpp"

#if defined(_MSC_VER) && (defined(__x86_64__) || defined(_M_X64))
#include <intrin.h>
#endif

namespace anofox {
namespace ggml_adapter {

std::string UnsupportedCpuReason() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
	int regs[4];
	__cpuid(regs, 0);
	if (regs[0] < 7) {
		return "its CPU predates the AVX2 instruction set";
	}
	__cpuidex(regs, 7, 0);
	const bool avx2 = (regs[1] & (1 << 5)) != 0, bmi2 = (regs[1] & (1 << 8)) != 0;
	__cpuid(regs, 1);
	const bool fma = (regs[2] & (1 << 12)) != 0, f16c = (regs[2] & (1 << 29)) != 0;
	if (!(avx2 && bmi2 && fma && f16c)) {
		return "its CPU lacks AVX2, FMA, F16C or BMI2";
	}
#else
	__builtin_cpu_init();
	if (!(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c") && __builtin_cpu_supports("bmi2"))) {
		return "its CPU lacks AVX2, FMA, F16C or BMI2";
	}
#endif
#endif
	return std::string();
}

} // namespace ggml_adapter
} // namespace anofox
