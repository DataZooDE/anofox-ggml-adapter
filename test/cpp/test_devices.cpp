// Device discovery: what GPUs this machine has, and whether the published plugins carry kernels for them. No vendor SDK is needed:
// NVML is dlopened, ROCm is read from the KFD sysfs topology, Apple Silicon from the SoC name.

#include "catch.hpp"

#include "anofox_ggml_adapter/devices.hpp"

#include <filesystem>
#include <fstream>
#include <set>

using namespace anofox::ggml_adapter;

#ifndef _WIN32
namespace {

//! A fake /sys/class/kfd/kfd/topology/nodes: node 0 is the CPU (gfx_target_version 0), then the GPUs as (gfx_target_version, simd_count).
std::string FakeKfdTopology(const std::string &tag, const std::vector<std::pair<uint64_t, uint64_t>> &gpus) {
	namespace fs = std::filesystem;
	const fs::path root = fs::temp_directory_path() / ("anofox_ggml_fake_kfd_" + tag);
	fs::remove_all(root);
	auto node = [&](int index, uint64_t version, uint64_t simds, const std::string &name) {
		const fs::path dir = root / std::to_string(index);
		fs::create_directories(dir);
		std::ofstream(dir / "properties") << "cpu_cores_count 16\nsimd_count " << simds << "\ngfx_target_version " << version << "\n";
		std::ofstream(dir / "name") << name << "\n";
		if (simds > 0) {
			fs::create_directories(dir / "mem_banks" / "0");
			std::ofstream(dir / "mem_banks" / "0" / "properties") << "heap_type 1\nsize_in_bytes 17179869184\n";
		}
	};
	node(0, 0, 0, "");
	int index = 1;
	for (const auto &g : gpus) {
		node(index, g.first, g.second, "gpu" + std::to_string(index));
		index++;
	}
	return root.string();
}

} // namespace

TEST_CASE("devices: every compute-capable KFD node is a rocm device, usable when the plugin carries its architecture", "[adapter][devices]") {
	const auto root = FakeKfdTopology("two_gpus", {{110500, 64}, {120001, 2048}, {90000, 0}, {100306, 4}});
	const auto devices = DiscoverRocmDevicesFrom(root, "6.4.0");
	REQUIRE(devices.size() == 3); // the CPU node and the node without compute units never appear
	REQUIRE(devices[0].device_id == "rocm:0");
	REQUIRE(devices[0].backend == "ggml-rocm");
	REQUIRE(devices[0].arch == "gfx1150");
	REQUIRE(devices[0].device_ordinal == 0);
	REQUIRE(devices[0].usable);
	REQUIRE(devices[0].vram_total == 17179869184LL);
	REQUIRE(devices[0].vram_free == -1); // needs rocm_smi: unknown, not 0
	REQUIRE(devices[1].arch == "gfx1201");
	REQUIRE(devices[1].device_ordinal == 1);
	REQUIRE(devices[1].driver == "6.4.0");
	// an iGPU the plugin has no code objects for: listed (with its HIP ordinal) but not usable
	REQUIRE(devices[2].arch == "gfx1036");
	REQUIRE(devices[2].device_ordinal == 2);
	REQUIRE_FALSE(devices[2].usable);
	REQUIRE(DiscoverRocmDevicesFrom(root + "/does_not_exist", "").empty());
}
#endif

TEST_CASE("devices: which cards the published plugins carry kernels for", "[adapter][devices]") {
	REQUIRE(CudaComputeCapabilityUsable(7, 5));  // Turing, the oldest build target
	REQUIRE(CudaComputeCapabilityUsable(12, 0)); // Blackwell
	REQUIRE_FALSE(CudaComputeCapabilityUsable(6, 1));
	REQUIRE_FALSE(CudaComputeCapabilityUsable(7, 0));
	for (const auto &arch : GgmlRocmPluginArchs()) {
		REQUIRE(RocmArchUsable(arch));
	}
	REQUIRE_FALSE(RocmArchUsable("gfx906"));
	REQUIRE_FALSE(RocmArchUsable(""));
}

TEST_CASE("devices: discovery always yields the usable cpu row first, and unique ids", "[adapter][devices]") {
	const auto devices = DiscoverDevices();
	REQUIRE_FALSE(devices.empty());
	const auto &cpu = devices[0];
	REQUIRE(cpu.device_id == "cpu");
	REQUIRE(cpu.backend == "ggml");
	REQUIRE_FALSE(cpu.name.empty());
	REQUIRE(cpu.usable);
	// GPU rows depend on the machine: only their shape is asserted
	std::set<std::string> ids;
	for (const auto &d : devices) {
		REQUIRE(ids.insert(d.device_id).second);
		REQUIRE_FALSE(d.backend.empty());
		REQUIRE(d.device_ordinal >= 0);
	}
}
