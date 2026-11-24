// Device discovery: what GPUs this machine has, and whether the published plugins carry kernels for them. No vendor SDK is needed:
// NVML is dlopened, ROCm is read from the KFD sysfs topology, Apple Silicon from the SoC name.

#include "catch.hpp"

#include "anofox_ggml_adapter/devices.hpp"

#include <filesystem>
#include <fstream>
#include <set>
#include <thread>

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

//! A fake /sys/devices/system/cpu: cpuN/topology/{physical_package_id,core_id} for each (package, core) pair, cpu index = position,
//! and core_cpus_list (the cpus sharing that core) when `with_lists`, as current kernels write it.
std::string FakeCpuTopology(const std::string &tag, const std::vector<std::pair<int, int>> &cpus, bool with_lists = false,
                            const std::vector<std::string> &lists = {}) {
	namespace fs = std::filesystem;
	const fs::path root = fs::temp_directory_path() / ("anofox_ggml_fake_cpu_" + tag);
	fs::remove_all(root);
	for (size_t i = 0; i < cpus.size(); i++) {
		const fs::path dir = root / ("cpu" + std::to_string(i)) / "topology";
		fs::create_directories(dir);
		std::ofstream(dir / "physical_package_id") << cpus[i].first << "\n";
		std::ofstream(dir / "core_id") << cpus[i].second << "\n";
		if (with_lists) {
			std::ofstream(dir / "core_cpus_list") << lists[i] << "\n";
		}
	}
	fs::create_directories(root / "cpufreq"); // not a cpuN directory: ignored
	return root.string();
}

} // namespace

TEST_CASE("devices: physical cores count SMT siblings once, per package, within the allowed cpus", "[adapter][devices]") {
	// 4 cores with 2 threads each, siblings numbered apart as Linux does (cpu0/cpu4 share core 0)
	const auto smt = FakeCpuTopology("smt", {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 0}, {0, 1}, {0, 2}, {0, 3}});
	REQUIRE(PhysicalCpuCoresFrom(smt, {}) == 4);
	// a taskset/cpuset that allows both threads of one core and one thread of another: two cores
	REQUIRE(PhysicalCpuCoresFrom(smt, {0, 4, 1}) == 2);
	// core ids repeat across packages: two sockets of two cores are four cores
	const auto sockets = FakeCpuTopology("sockets", {{0, 0}, {0, 1}, {1, 0}, {1, 1}});
	REQUIRE(PhysicalCpuCoresFrom(sockets, {}) == 4);
	// no topology at all: 0, so the caller falls back
	REQUIRE(PhysicalCpuCoresFrom((std::filesystem::temp_directory_path() / "anofox_ggml_no_such_cpu_dir").string(), {}) == 0);
	// device-tree arm64 (RK3588: clusters of 4+2+2 cores): core_id restarts per cluster, so (package, core) undercounts; the
	// kernel's core_cpus_list tells the cores apart
	const auto clusters = FakeCpuTopology("clusters", {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 0}, {0, 1}, {0, 0}, {0, 1}}, true,
	                                      {"0", "1", "2", "3", "4", "5", "6", "7"});
	REQUIRE(PhysicalCpuCoresFrom(clusters, {}) == 8);
	// SMT with core_cpus_list: siblings share one list
	const auto smt_lists = FakeCpuTopology("smt_lists", {{0, 0}, {0, 1}, {0, 0}, {0, 1}}, true, {"0,2", "1,3", "0,2", "1,3"});
	REQUIRE(PhysicalCpuCoresFrom(smt_lists, {}) == 2);
	// on this machine: at least one core, never more than the hardware threads
	REQUIRE(PhysicalCpuCores() >= 1);
	REQUIRE(PhysicalCpuCores() <= static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
}

TEST_CASE("devices: a cgroup CPU quota caps the cores a container may use", "[adapter][devices]") {
	namespace fs = std::filesystem;
	const fs::path root = fs::temp_directory_path() / "anofox_ggml_fake_cgroup";
	fs::remove_all(root);
	fs::create_directories(root);
	REQUIRE(CgroupCpuLimitFrom(root.string()) == 0); // no controller files: no limit
	std::ofstream(root / "cpu.max") << "max 100000\n";
	REQUIRE(CgroupCpuLimitFrom(root.string()) == 0); // v2, unlimited
	std::ofstream(root / "cpu.max") << "250000 100000\n";
	REQUIRE(CgroupCpuLimitFrom(root.string()) == 3); // v2: 2.5 CPUs round up
	fs::remove(root / "cpu.max");
	fs::create_directories(root / "cpu");
	std::ofstream(root / "cpu" / "cpu.cfs_quota_us") << "200000\n";
	std::ofstream(root / "cpu" / "cpu.cfs_period_us") << "100000\n";
	REQUIRE(CgroupCpuLimitFrom(root.string()) == 2); // v1
	std::ofstream(root / "cpu" / "cpu.cfs_quota_us") << "-1\n";
	REQUIRE(CgroupCpuLimitFrom(root.string()) == 0); // v1, unlimited
}

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
