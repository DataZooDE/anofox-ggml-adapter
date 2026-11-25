#include "anofox_ggml_adapter/devices.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <thread>
#include <utility>

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
#if defined(__linux__)
#include <sched.h>
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
#else
using NvmlHandle = void *;
const char *const kNvmlLibraries[] = {"libnvidia-ml.so.1", "libnvidia-ml.so"};
NvmlHandle NvmlOpen(const char *name) {
	return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
}
void *NvmlSymbol(NvmlHandle lib, const char *symbol) {
	return dlsym(lib, symbol);
}
#endif

//! One NVML probe. Hosts discover on every query, so NVML is loaded and initialised once per process and never shut down (NVML
//! refcounts its init; opening and closing the library per query, next to a live CUDA context, is what a driver least expects),
//! and a probe that fails after one succeeded keeps the devices already seen: a GPU must not vanish mid-session because one
//! nvmlInit or nvmlDeviceGetCount failed under load (an explicit 'cuda' would then refuse a query the GPU had been serving).
void ProbeCudaDevicesOnce(std::vector<DeviceInfo> &devices);

void ProbeCudaDevices(std::vector<DeviceInfo> &devices) {
	static std::mutex lock;
	static std::vector<DeviceInfo> last_good;
	std::lock_guard<std::mutex> guard(lock);
	std::vector<DeviceInfo> found;
	ProbeCudaDevicesOnce(found);
	if (!found.empty()) {
		last_good = found;
	}
	const auto &use = found.empty() ? last_good : found;
	devices.insert(devices.end(), use.begin(), use.end());
}

void ProbeCudaDevicesOnce(std::vector<DeviceInfo> &devices) {
	static NvmlHandle nvml = [] {
		NvmlHandle handle = nullptr;
		for (auto *candidate : kNvmlLibraries) {
			handle = NvmlOpen(candidate);
			if (handle) {
				break;
			}
		}
		return handle;
	}();
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
	auto nvml_count = reinterpret_cast<CountFn>(NvmlSymbol(nvml, "nvmlDeviceGetCount_v2"));
	auto nvml_handle = reinterpret_cast<HandleFn>(NvmlSymbol(nvml, "nvmlDeviceGetHandleByIndex_v2"));
	auto nvml_name = reinterpret_cast<NameFn>(NvmlSymbol(nvml, "nvmlDeviceGetName"));
	auto nvml_memory = reinterpret_cast<MemFn>(NvmlSymbol(nvml, "nvmlDeviceGetMemoryInfo"));
	auto nvml_cc = reinterpret_cast<CcFn>(NvmlSymbol(nvml, "nvmlDeviceGetCudaComputeCapability"));
	auto nvml_driver = reinterpret_cast<DriverFn>(NvmlSymbol(nvml, "nvmlSystemGetDriverVersion"));
	static bool initialised = false; // under ProbeCudaDevices' lock
	if (!nvml_init || !nvml_count || !nvml_handle) {
		return;
	}
	if (!initialised) {
		if (nvml_init() != 0) {
			return; // retried on the next probe
		}
		initialised = true;
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
	// NVML stays loaded and initialised for the process: see ProbeCudaDevices
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

int PhysicalCpuCoresFrom(const std::string &cpu_dir, const std::vector<int> &allowed) {
	namespace fs = std::filesystem;
	const std::set<int> allow(allowed.begin(), allowed.end());
	std::set<std::pair<long, long>> cores; // (package, core): core ids repeat across packages
	std::set<std::string> core_lists;        // the cpus of each core, where the kernel writes them
	std::error_code ec;
	for (fs::directory_iterator it(cpu_dir, ec), end; !ec && it != end; it.increment(ec)) {
		const std::string name = it->path().filename().string();
		if (name.size() < 4 || name.compare(0, 3, "cpu") != 0 ||
		    !std::all_of(name.begin() + 3, name.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
			continue; // cpufreq, cpuidle, ...
		}
		const int index = std::atoi(name.c_str() + 3);
		if (!allow.empty() && !allow.count(index)) {
			continue;
		}
		std::string list;
		std::ifstream(it->path() / "topology" / "core_cpus_list") >> list;
		if (list.empty()) {
			std::ifstream(it->path() / "topology" / "thread_siblings_list") >> list;
		}
		if (!list.empty()) {
			core_lists.insert(list);
		}
		long package = -1, core = -1;
		std::ifstream(it->path() / "topology" / "physical_package_id") >> package;
		std::ifstream(it->path() / "topology" / "core_id") >> core;
		if (core >= 0) {
			cores.insert({package, core});
		}
	}
	// the sibling lists name each core exactly (core_id restarts per cluster on device-tree arm64); (package, core) otherwise
	return static_cast<int>(!core_lists.empty() ? core_lists.size() : cores.size());
}

int CgroupCpuLimitFrom(const std::string &cgroup_root) {
	auto ceil_div = [](long long quota, long long period) {
		return period > 0 && quota > 0 ? static_cast<int>((quota + period - 1) / period) : 0;
	};
	std::ifstream v2(cgroup_root + "/cpu.max"); // "<quota> <period>" or "max <period>"
	if (v2) {
		std::string quota;
		long long period = 0;
		v2 >> quota >> period;
		return quota == "max" || quota.empty() ? 0 : ceil_div(std::atoll(quota.c_str()), period);
	}
	long long quota = -1, period = 0;
	std::ifstream(cgroup_root + "/cpu/cpu.cfs_quota_us") >> quota;
	std::ifstream(cgroup_root + "/cpu/cpu.cfs_period_us") >> period;
	return ceil_div(quota, period);
}

int AppleCoresFrom(int perflevel0_physicalcpu, int physicalcpu) {
	return perflevel0_physicalcpu > 0 ? perflevel0_physicalcpu : physicalcpu;
}

int PhysicalCpuCores() {
	static const int cores = [] {
		int n = 0;
#if defined(__linux__)
		std::vector<int> allowed; // the cpus a taskset / cpuset leaves this process
		cpu_set_t set;
		if (sched_getaffinity(0, sizeof(set), &set) == 0) {
			for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
				if (CPU_ISSET(cpu, &set)) {
					allowed.push_back(cpu);
				}
			}
		}
		n = PhysicalCpuCoresFrom("/sys/devices/system/cpu", allowed);
		const int quota = CgroupCpuLimitFrom("/sys/fs/cgroup"); // a container's CPU limit (its own cgroup is mounted at the root)
		if (quota > 0 && (n <= 0 || quota < n)) {
			n = quota;
		}
#elif defined(__APPLE__)
		int value = 0, perf = 0;
		size_t size = sizeof(value), perf_size = sizeof(perf);
		if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &perf_size, nullptr, 0) != 0) {
			perf = 0;
		}
		if (sysctlbyname("hw.physicalcpu", &value, &size, nullptr, 0) == 0) {
			n = AppleCoresFrom(perf, value);
		}
#elif defined(_WIN32)
		DWORD bytes = 0;
		GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
		std::vector<char> buffer(bytes);
		if (bytes > 0 && GetLogicalProcessorInformationEx(RelationProcessorCore,
		                                                  reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &bytes)) {
			for (DWORD offset = 0; offset < bytes;) {
				auto *info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
				n++;
				offset += info->Size;
			}
		}
#endif
		const int logical = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
		return n > 0 ? std::min(n, logical) : logical;
	}();
	return cores;
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
