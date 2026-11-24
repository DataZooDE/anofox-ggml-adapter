#include "anofox_ggml_adapter/devices.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace anofox {
namespace ggml_adapter {

namespace {

bool StartsWith(const std::string &s, const std::string &prefix) {
	return s.compare(0, prefix.size(), prefix) == 0;
}

std::string CpuModelName() {
#if defined(__linux__)
	std::ifstream cpuinfo("/proc/cpuinfo");
	std::string line;
	while (std::getline(cpuinfo, line)) {
		if (StartsWith(line, "model name")) {
			const auto colon = line.find(':');
			if (colon != std::string::npos) {
				std::string name = line.substr(colon + 1);
				std::replace(name.begin(), name.end(), '\t', ' ');
				const auto first = name.find_first_not_of(' ');
				return first == std::string::npos ? std::string("cpu") : name.substr(first);
			}
		}
	}
#elif defined(__APPLE__)
	char brand[256];
	size_t size = sizeof(brand);
	if (sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0) == 0) {
		return std::string(brand);
	}
#endif
	return "cpu";
}

DeviceInfo CpuDevice() {
	DeviceInfo d;
	d.device_id = "cpu";
	d.backend = "ggml";
	d.name = CpuModelName();
	d.usable = true;
	return d;
}

//===----------------------------------------------------------------------===//
// CUDA: NVML, dlopened (it ships with every NVIDIA driver: libnvidia-ml.so.1, nvml.dll). No driver, no rows.
//===----------------------------------------------------------------------===//

struct NvmlMemory {
	unsigned long long total;
	unsigned long long free;
	unsigned long long used;
};

#ifdef _WIN32
using NvmlHandle = HMODULE;
const char *const kNvmlLibraries[] = {"nvml.dll"};
NvmlHandle NvmlOpen(const char *name) {
	return LoadLibraryA(name);
}
void *NvmlSymbol(NvmlHandle lib, const char *symbol) {
	return reinterpret_cast<void *>(GetProcAddress(lib, symbol));
}
void NvmlClose(NvmlHandle lib) {
	FreeLibrary(lib);
}
#else
using NvmlHandle = void *;
const char *const kNvmlLibraries[] = {"libnvidia-ml.so.1", "libnvidia-ml.so"};
NvmlHandle NvmlOpen(const char *name) {
	return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
}
void *NvmlSymbol(NvmlHandle lib, const char *symbol) {
	return dlsym(lib, symbol);
}
void NvmlClose(NvmlHandle lib) {
	dlclose(lib);
}
#endif

void ProbeCudaDevices(std::vector<DeviceInfo> &devices) {
	NvmlHandle nvml = nullptr;
	for (auto *candidate : kNvmlLibraries) {
		nvml = NvmlOpen(candidate);
		if (nvml) {
			break;
		}
	}
	if (!nvml) {
		return;
	}
	using InitFn = int (*)();
	using CountFn = int (*)(unsigned int *);
	using HandleFn = int (*)(unsigned int, void **);
	using NameFn = int (*)(void *, char *, unsigned int);
	using MemFn = int (*)(void *, NvmlMemory *);
	using CcFn = int (*)(void *, int *, int *);
	using DriverFn = int (*)(char *, unsigned int);
	auto nvml_init = reinterpret_cast<InitFn>(NvmlSymbol(nvml, "nvmlInit_v2"));
	auto nvml_shutdown = reinterpret_cast<InitFn>(NvmlSymbol(nvml, "nvmlShutdown"));
	auto nvml_count = reinterpret_cast<CountFn>(NvmlSymbol(nvml, "nvmlDeviceGetCount_v2"));
	auto nvml_handle = reinterpret_cast<HandleFn>(NvmlSymbol(nvml, "nvmlDeviceGetHandleByIndex_v2"));
	auto nvml_name = reinterpret_cast<NameFn>(NvmlSymbol(nvml, "nvmlDeviceGetName"));
	auto nvml_memory = reinterpret_cast<MemFn>(NvmlSymbol(nvml, "nvmlDeviceGetMemoryInfo"));
	auto nvml_cc = reinterpret_cast<CcFn>(NvmlSymbol(nvml, "nvmlDeviceGetCudaComputeCapability"));
	auto nvml_driver = reinterpret_cast<DriverFn>(NvmlSymbol(nvml, "nvmlSystemGetDriverVersion"));
	if (!nvml_init || !nvml_count || !nvml_handle || nvml_init() != 0) {
		NvmlClose(nvml);
		return;
	}
	char driver_version[96] = {0};
	if (nvml_driver) {
		nvml_driver(driver_version, sizeof(driver_version));
	}
	unsigned int count = 0;
	if (nvml_count(&count) == 0) {
		for (unsigned int i = 0; i < count; i++) {
			void *handle = nullptr;
			if (nvml_handle(i, &handle) != 0) {
				continue;
			}
			DeviceInfo d;
			d.device_id = "cuda:" + std::to_string(i);
			d.backend = "ggml-cuda";
			d.device_ordinal = static_cast<int>(i);
			char name[96] = {0};
			if (nvml_name && nvml_name(handle, name, sizeof(name)) == 0) {
				d.name = name;
			}
			int major = 0, minor = 0;
			if (nvml_cc && nvml_cc(handle, &major, &minor) == 0) {
				d.arch = "sm_" + std::to_string(major) + std::to_string(minor);
			}
			NvmlMemory memory {};
			if (nvml_memory && nvml_memory(handle, &memory) == 0) {
				d.vram_total = static_cast<int64_t>(memory.total);
				d.vram_free = static_cast<int64_t>(memory.free);
			}
			d.driver = driver_version;
			d.usable = CudaComputeCapabilityUsable(major, minor);
			devices.push_back(std::move(d));
		}
	}
	if (nvml_shutdown) {
		nvml_shutdown();
	}
	NvmlClose(nvml);
}

//===----------------------------------------------------------------------===//
// ROCm: the kernel's KFD topology (/sys/class/kfd/kfd/topology/nodes/*; /dev/kfd is the existence gate). gfx_target_version =
// major*10000 + minor*100 + step -> "gfx%d%x%x" (0: a CPU node); VRAM from the FB heaps of mem_banks (heap_type 1|2). The ordinal counts
// every GPU node, usable or not: it is the HIP device index.
//===----------------------------------------------------------------------===//

#ifndef _WIN32
bool ReadKfdProperty(const std::string &path, const std::string &key, uint64_t &value) {
	std::ifstream file(path);
	std::string line;
	while (std::getline(file, line)) {
		if (StartsWith(line, key + " ")) {
			value = std::strtoull(line.c_str() + key.size() + 1, nullptr, 10);
			return true;
		}
	}
	return false;
}

std::string GfxArchFromTargetVersion(uint64_t version) {
	char buffer[32];
	std::snprintf(buffer, sizeof(buffer), "gfx%llu%llx%llx", static_cast<unsigned long long>(version / 10000),
	              static_cast<unsigned long long>((version / 100) % 100), static_cast<unsigned long long>(version % 100));
	return buffer;
}

int64_t SumVramBanks(const std::string &node_path) {
	int64_t total = 0;
	const auto banks = node_path + "/mem_banks";
	DIR *dir = opendir(banks.c_str());
	if (!dir) {
		return -1;
	}
	while (auto *entry = readdir(dir)) {
		if (entry->d_name[0] == '.') {
			continue;
		}
		const auto properties = banks + "/" + entry->d_name + "/properties";
		uint64_t heap_type = 0, size = 0;
		if (ReadKfdProperty(properties, "heap_type", heap_type) && ReadKfdProperty(properties, "size_in_bytes", size) &&
		    (heap_type == 1 || heap_type == 2)) {
			total += static_cast<int64_t>(size);
		}
	}
	closedir(dir);
	return total > 0 ? total : -1;
}

std::string AmdgpuDriverVersion() {
	std::ifstream file("/sys/module/amdgpu/version");
	std::string version;
	return file && std::getline(file, version) ? version : std::string();
}
#endif

//===----------------------------------------------------------------------===//
// Apple Silicon: no driver to ask; one logical mlx:0 row. Under Rosetta an x86_64 process reads an Apple SoC name but cannot load the
// arm64 plugin, so usable follows the compiled architecture.
//===----------------------------------------------------------------------===//

#if defined(__APPLE__)
void ProbeMlxDevices(std::vector<DeviceInfo> &devices) {
	const std::string soc = CpuModelName();
	std::string lower = soc;
	for (auto &c : lower) {
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	}
	if (lower.find("apple") == std::string::npos) {
		return;
	}
	DeviceInfo d;
	d.device_id = "mlx:0";
	d.backend = "mlx";
	d.name = soc;
	d.arch = soc;
#if defined(__aarch64__) || defined(__arm64__)
	d.usable = true;
#else
	d.usable = false;
#endif
	devices.push_back(std::move(d));
}
#endif

} // namespace

bool CudaComputeCapabilityUsable(int major, int minor) {
	return major * 10 + minor >= 75;
}

const std::vector<std::string> &GgmlRocmPluginArchs() {
	static const std::vector<std::string> archs = {"gfx90a",  "gfx942",  "gfx1030", "gfx1100", "gfx1101",
	                                               "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201"};
	return archs;
}

bool RocmArchUsable(const std::string &arch) {
	const auto &archs = GgmlRocmPluginArchs();
	return std::find(archs.begin(), archs.end(), arch) != archs.end();
}

std::vector<DeviceInfo> DiscoverRocmDevicesFrom(const std::string &nodes_path, const std::string &driver) {
	std::vector<DeviceInfo> devices;
#ifndef _WIN32
	DIR *dir = opendir(nodes_path.c_str());
	if (!dir) {
		return devices;
	}
	std::vector<std::string> nodes;
	while (auto *entry = readdir(dir)) {
		if (entry->d_name[0] != '.') {
			nodes.push_back(entry->d_name);
		}
	}
	closedir(dir);
	std::sort(nodes.begin(), nodes.end(), [](const std::string &a, const std::string &b) { return std::stoll(a) < std::stoll(b); });
	int ordinal = 0;
	for (const auto &node : nodes) {
		const std::string path = nodes_path + "/" + node;
		uint64_t version = 0, simds = 0;
		if (!ReadKfdProperty(path + "/properties", "gfx_target_version", version) || version == 0) {
			continue; // a CPU node
		}
		ReadKfdProperty(path + "/properties", "simd_count", simds);
		if (simds == 0) {
			continue; // not compute-capable
		}
		DeviceInfo d;
		d.device_id = "rocm:" + std::to_string(ordinal);
		d.backend = "ggml-rocm";
		d.device_ordinal = ordinal++;
		d.arch = GfxArchFromTargetVersion(version);
		std::ifstream name_file(path + "/name");
		std::getline(name_file, d.name);
		if (d.name.empty()) {
			d.name = d.arch;
		}
		d.vram_total = SumVramBanks(path);
		d.driver = driver;
		d.usable = RocmArchUsable(d.arch);
		devices.push_back(std::move(d));
	}
#else
	(void)nodes_path;
	(void)driver;
#endif
	return devices;
}

std::vector<DeviceInfo> DiscoverDevices() {
	std::vector<DeviceInfo> devices;
	devices.push_back(CpuDevice());
	ProbeCudaDevices(devices);
#ifndef _WIN32
	struct stat kfd {};
	if (stat("/dev/kfd", &kfd) == 0) {
		for (auto &d : DiscoverRocmDevicesFrom("/sys/class/kfd/kfd/topology/nodes", AmdgpuDriverVersion())) {
			devices.push_back(std::move(d));
		}
	}
#endif
#if defined(__APPLE__)
	ProbeMlxDevices(devices);
#endif
	return devices;
}

} // namespace ggml_adapter
} // namespace anofox
