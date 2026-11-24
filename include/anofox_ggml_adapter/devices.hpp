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

//! The gfx architectures the released ROCm plugin carries code objects for.
const std::vector<std::string> &GgmlRocmPluginArchs();
bool RocmArchUsable(const std::string &arch);

} // namespace ggml_adapter
} // namespace anofox
