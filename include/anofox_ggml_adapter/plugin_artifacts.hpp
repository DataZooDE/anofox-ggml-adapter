#pragma once

// plugin_artifacts — where a GPU backend plugin comes from, for every host: its file name on each platform, the adapter release that
// carries it (with sha256 sidecars), and whether one is published for a (backend, os, arch) at all. Pure functions taking the platform
// explicitly, so every combination is testable from any machine. The reasons name no host setting: each host appends its own remedy.

#include <cctype>
#include <string>

namespace anofox {
namespace ggml_adapter {

//! The adapter release whose assets carry the built plugins. Bumped BY HAND before tagging: the plugin workflow refuses to attach plugins
//! to a release whose tag is not this one, so a release always contains code that points at itself. A stale tag is safe: the loader
//! refuses a plugin built against another ABI version.
static constexpr const char *ANOFOX_GGML_PLUGIN_RELEASE_TAG = "v2026.11.24";

//! 'cuda', 'rocm', 'mlx' or an unknown name: lowercased; 'migraphx' (the removed MIGraphX plugin) is the ROCm plugin.
inline std::string NormalizeBackend(const std::string &requested) {
	std::string backend = requested;
	for (auto &c : backend) {
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	}
	if (backend == "migraphx" || backend == "ggml-rocm") {
		return "rocm";
	}
	return backend == "ggml-cuda" ? "cuda" : backend;
}

//! The plugin's base name: CUDA and ROCm are the ggml plugin built per vendor; Apple GPUs use the MLX plugin.
inline std::string PluginBaseName(const std::string &backend) {
	const auto b = NormalizeBackend(backend);
	if (b == "cuda") {
		return "ggml-cuda";
	}
	if (b == "rocm") {
		return "ggml-rocm";
	}
	return b;
}

inline std::string PluginLibrarySuffix(const std::string &os) {
	return os == "windows" ? ".dll" : os == "osx" ? ".dylib" : ".so";
}

//! The plugin's file name on `os` ("linux" | "osx" | "windows"); Windows has no 'lib' prefix (what CMake emits there).
inline std::string PluginFileNameFor(const std::string &backend, const std::string &os) {
	return std::string(os == "windows" ? "" : "lib") + "anofox_" + PluginBaseName(backend) + "_plugin" + PluginLibrarySuffix(os);
}

inline std::string HostPluginOs() {
#if defined(_WIN32)
	return "windows";
#elif defined(__APPLE__)
	return "osx";
#else
	return "linux";
#endif
}

//! "amd64" | "arm64" | "" when neither.
inline std::string HostPluginArch() {
#if defined(__aarch64__) || defined(_M_ARM64)
	return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
	return "amd64";
#else
	return "";
#endif
}

inline std::string PluginFileName(const std::string &backend) {
	return PluginFileNameFor(backend, HostPluginOs());
}

//! Empty when a plugin for `backend` is PUBLISHED (built and verified) for (os, arch); otherwise why not, and what is available there.
inline std::string UnpublishedPluginReason(const std::string &backend, const std::string &os, const std::string &arch) {
	const auto b = NormalizeBackend(backend);
	const std::string where = os + "/" + (arch.empty() ? std::string("unknown") : arch);
	if (b == "cuda") {
		if (os == "linux" && arch == "amd64") {
			return "";
		}
		return "no CUDA plugin is published for " + where + ": the ggml CUDA backend is built and verified for linux/amd64 only" +
		       (os == "osx" ? std::string(" (on Apple Silicon the MLX backend runs instead)") : std::string());
	}
	if (b == "rocm") {
		if (os == "linux" && arch == "amd64") {
			return "";
		}
		return "no ROCm plugin is published for " + where + ": the ggml HIP backend is built and verified for linux/amd64 only";
	}
	if (b == "mlx") {
		if (os == "osx" && arch == "arm64") {
			return "";
		}
		return "no MLX plugin is published for " + where + ": MLX is Apple Silicon only (macOS/arm64)";
	}
	return "unknown backend '" + backend + "' (expected 'cuda', 'rocm' or 'mlx')";
}

//! The release asset carrying a backend's sha256 sidecar (CI writes `sha256sum <file> | tee <base>_plugin.sha256`).
inline std::string PluginSha256AssetName(const std::string &backend) {
	return PluginBaseName(backend) + "_plugin.sha256";
}

//! One file's digest out of a `sha256sum` sidecar, matched by file name (a sidecar may list several). "" when the file is not listed,
//! which a caller must treat as a failure: a sidecar that does not name the file proves nothing about it.
inline std::string Sha256FromSidecar(const std::string &sidecar, const std::string &file_name) {
	size_t pos = 0;
	while (pos < sidecar.size()) {
		const auto eol = sidecar.find('\n', pos);
		const auto line = sidecar.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
		pos = eol == std::string::npos ? sidecar.size() : eol + 1;
		const auto space = line.find(' ');
		if (space != 64) { // "<64 hex>  <name>"
			continue;
		}
		auto named = line.substr(space);
		while (!named.empty() && (named.front() == ' ' || named.front() == '*')) {
			named.erase(named.begin());
		}
		while (!named.empty() && (named.back() == '\r' || named.back() == ' ')) {
			named.pop_back();
		}
		const auto slash = named.find_last_of("/\\");
		if (slash != std::string::npos) {
			named = named.substr(slash + 1);
		}
		if (named == file_name) {
			std::string digest = line.substr(0, 64);
			for (auto &c : digest) {
				c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
			}
			return digest;
		}
	}
	return "";
}

//! Download URL of one asset of an adapter release; "" without a tag.
inline std::string PluginReleaseAssetUrl(const std::string &asset, const std::string &release_tag) {
	if (release_tag.empty()) {
		return "";
	}
	return "https://github.com/DataZooDE/anofox-ggml-adapter/releases/download/" + release_tag + "/" + asset;
}

} // namespace ggml_adapter
} // namespace anofox
