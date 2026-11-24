#pragma once

// devices — which GPUs this machine has, and whether the published plugins carry kernels for them. No vendor SDK is linked: NVML is
// dlopened (it ships with every NVIDIA driver), ROCm is read from the kernel's KFD topology in sysfs, Apple Silicon from the SoC name.
// How a host maps its device setting onto these rows (and words a refusal) stays in the host.

#include <cstdint>
#include <string>
#include <vector>

namespace anofox {
namespace ggml_adapter {

struct DeviceInfo {
	std::string device_id;  //!< "cpu", "cuda:<n>", "rocm:<n>", "mlx:0"
	std::string backend;    //!< what serves it: "ggml" (cpu), "ggml-cuda", "ggml-rocm" (the ggml GPU plugin) or "mlx"
	std::string name;       //!< human-readable
	std::string arch;       //!< "" (cpu), "sm_XY" (cuda), "gfxNNNN" (rocm), the SoC name (mlx)
	int64_t vram_total = -1; //!< bytes; -1 = unknown
	int64_t vram_free = -1;
	std::string driver;     //!< driver / runtime version, "" when unknown
	bool usable = false;    //!< the published plugin carries code for this device
	int device_ordinal = 0; //!< the vendor runtime's device index (what the plugin is handed)
};

//! The cpu row first, then every GPU found (CUDA via NVML, ROCm via KFD, Apple Silicon), whether or not its plugin is installed.
std::vector<DeviceInfo> DiscoverDevices();

//! The ROCm rows of a KFD topology directory (a test seam: /sys/class/kfd/kfd/topology/nodes on a real machine). Empty on Windows.
std::vector<DeviceInfo> DiscoverRocmDevicesFrom(const std::string &nodes_path, const std::string &driver);

//! The plugin is built for sm_75 (Turing) and newer: an older card has no kernel image.
bool CudaComputeCapabilityUsable(int major, int minor);

//! The number of physical CPU cores this process may run on (SMT siblings counted once), at least 1. The CPU engine is fastest with
//! one thread per physical core: on an SMT machine a second thread per core adds no arithmetic throughput, and every op's barrier
//! waits for the slowest thread (measured on a 12-core / 24-thread Ryzen AI: 25 s at 12 threads, 35 s at 24, the same answers).
//! Linux reads sysfs topology, macOS hw.physicalcpu, Windows the logical processor information; std::thread's count elsewhere.
int PhysicalCpuCores();

//! PhysicalCpuCores() over a sysfs cpu directory (a test seam: /sys/devices/system/cpu on a real machine), counting the distinct
//! cores among the cpus `allowed` says this process may use (empty: all): by topology/core_cpus_list (thread_siblings_list on
//! older kernels), else by (physical_package_id, core_id), which undercounts device-tree arm64 where core_id restarts per
//! cluster. 0 when the directory has no topology.
int PhysicalCpuCoresFrom(const std::string &cpu_dir, const std::vector<int> &allowed);

//! The CPUs a cgroup CPU quota allows, rounded up (a test seam over a cgroup root: cpu.max for v2, cpu/cpu.cfs_quota_us and
//! cpu/cpu.cfs_period_us for v1); 0 when there is no quota. PhysicalCpuCores() is capped by it on Linux: a container limited
//! to 2 CPUs on a 64-core host must not run 64 threads that hit barriers while throttled.
int CgroupCpuLimitFrom(const std::string &cgroup_root);

//! The gfx architectures the released ROCm plugin carries code objects for.
const std::vector<std::string> &GgmlRocmPluginArchs();
bool RocmArchUsable(const std::string &arch);

} // namespace ggml_adapter
} // namespace anofox
