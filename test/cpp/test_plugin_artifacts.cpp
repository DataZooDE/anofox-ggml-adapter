// Where a plugin comes from: file names per platform, the pinned release, sha256 sidecars, and which (backend, os, arch) is published.

#include "catch.hpp"

#include "anofox_ggml_adapter/plugin_artifacts.hpp"

using namespace anofox::ggml_adapter;

TEST_CASE("plugin_artifacts: one file name per backend and platform, shared by every host", "[adapter][plugin]") {
	REQUIRE(PluginFileNameFor("cuda", "linux") == "libanofox_ggml-cuda_plugin.so");
	REQUIRE(PluginFileNameFor("rocm", "linux") == "libanofox_ggml-rocm_plugin.so");
	REQUIRE(PluginFileNameFor("migraphx", "linux") == "libanofox_ggml-rocm_plugin.so"); // the removed MIGraphX plugin's name
	REQUIRE(PluginFileNameFor("mlx", "osx") == "libanofox_mlx_plugin.dylib");
	REQUIRE(PluginFileNameFor("cuda", "windows") == "anofox_ggml-cuda_plugin.dll");
	REQUIRE(PluginSha256AssetName("cuda") == "ggml-cuda_plugin.sha256");
	REQUIRE(PluginSha256AssetName("mlx") == "mlx_plugin.sha256");
	REQUIRE(PluginReleaseAssetUrl("x.so", "v1") == "https://github.com/DataZooDE/anofox-ggml-adapter/releases/download/v1/x.so");
	REQUIRE(PluginReleaseAssetUrl("x.so", "").empty());
	REQUIRE(std::string(ANOFOX_GGML_PLUGIN_RELEASE_TAG).rfind("v", 0) == 0);
}

TEST_CASE("plugin_artifacts: only published platforms are offered, with the reason otherwise", "[adapter][plugin]") {
	REQUIRE(UnpublishedPluginReason("cuda", "linux", "amd64").empty());
	REQUIRE(UnpublishedPluginReason("rocm", "linux", "amd64").empty());
	REQUIRE(UnpublishedPluginReason("mlx", "osx", "arm64").empty());
	REQUIRE(UnpublishedPluginReason("cuda", "linux", "arm64").find("linux/amd64 only") != std::string::npos);
	REQUIRE(UnpublishedPluginReason("cuda", "osx", "arm64").find("MLX") != std::string::npos);
	REQUIRE(UnpublishedPluginReason("mlx", "linux", "amd64").find("Apple Silicon") != std::string::npos);
	REQUIRE(UnpublishedPluginReason("vulkan", "linux", "amd64").find("unknown backend 'vulkan'") != std::string::npos);
	REQUIRE(NormalizeBackend("MIGraphX") == "rocm");
	REQUIRE(NormalizeBackend("CUDA") == "cuda");
}

TEST_CASE("plugin_artifacts: a sidecar vouches only for the file it names", "[adapter][plugin]") {
	const std::string a(64, 'a'), b(64, 'B');
	const std::string sidecar = a + "  libanofox_ggml-cuda_plugin.so\n" + b + " *dir/ggml-rocm.so\r\nnot a line\n";
	REQUIRE(Sha256FromSidecar(sidecar, "libanofox_ggml-cuda_plugin.so") == a);
	REQUIRE(Sha256FromSidecar(sidecar, "ggml-rocm.so") == std::string(64, 'b')); // path and '*' stripped, lowercased
	REQUIRE(Sha256FromSidecar(sidecar, "missing.so").empty());
}
