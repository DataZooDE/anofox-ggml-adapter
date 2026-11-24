#include "anofox_ggml_adapter/plugin_host.hpp"

#include <cstring>

// Is this translation unit built with AddressSanitizer? Used only to drop RTLD_DEEPBIND, which the sanitizer runtime cannot tolerate
// (it aborts the process on the dlopen: google/sanitizers#611).
#if defined(__SANITIZE_ADDRESS__)
#define ANOFOX_GGML_SANITIZER_BUILD 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define ANOFOX_GGML_SANITIZER_BUILD 1
#endif
#endif

#ifndef _WIN32
#include <dlfcn.h>
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace anofox {
namespace ggml_adapter {

namespace {

std::string KindText(PluginError::Kind kind) {
	switch (kind) {
	case PluginError::Kind::NOT_LOADABLE:
		return "the backend plugin cannot be loaded";
	case PluginError::Kind::NOT_A_PLUGIN:
		return "the file is not an anofox backend plugin";
	case PluginError::Kind::ABI_MISMATCH:
		return "the backend plugin speaks another plugin ABI version";
	case PluginError::Kind::CREATE_FAILED:
		return "the backend could not be initialised";
	case PluginError::Kind::RUN_FAILED:
		return "the backend failed";
	case PluginError::Kind::OUT_OF_MEMORY:
		return "the backend ran out of device memory";
	}
	return "backend plugin error";
}

#ifndef _WIN32
using LibraryHandle = void *;
LibraryHandle OpenLibrary(const std::string &path) {
	// RTLD_DEEPBIND: the plugin prefers its OWN symbols over those the host already exports. Host and ggml GPU plugin both carry a
	// copy of ggml and of these engine sources (same symbol names, different builds: the plugin's has the CUDA or HIP backend);
	// wherever the host's copy is globally visible (a unittest binary linking the extension statically, a host loading the extension
	// RTLD_GLOBAL) the plugin's calls would otherwise bind to it. Safe for this ABI: no allocation crosses the boundary. glibc only
	// (macOS binds each image to the library it linked against). Dropped under a sanitizer, which aborts on it.
	int flags = RTLD_NOW | RTLD_LOCAL;
#if defined(RTLD_DEEPBIND) && !defined(ANOFOX_GGML_SANITIZER_BUILD)
	flags |= RTLD_DEEPBIND;
#endif
	return dlopen(path.c_str(), flags);
}
void *LibrarySymbol(LibraryHandle lib, const char *symbol) {
	return dlsym(lib, symbol);
}
void CloseLibrary(LibraryHandle lib) {
	dlclose(lib);
}
std::string LibraryError() {
	const char *err = dlerror();
	return err ? std::string(err) : std::string("unknown dynamic-loader error");
}
#else
using LibraryHandle = HMODULE;
LibraryHandle OpenLibrary(const std::string &path) {
	return LoadLibraryA(path.c_str());
}
void *LibrarySymbol(LibraryHandle lib, const char *symbol) {
	return reinterpret_cast<void *>(GetProcAddress(lib, symbol));
}
void CloseLibrary(LibraryHandle lib) {
	FreeLibrary(lib);
}
std::string LibraryError() {
	return "LoadLibrary/GetProcAddress failed (error " + std::to_string(GetLastError()) + ")";
}
#endif

//! Load and check a plugin library. Only `abi_version` is read before it is checked: in a library built against another layout every
//! later field sits at the wrong offset. On failure the library is closed (none of its code ran beyond the entry function).
const AnofoxGgmlPluginApi *LoadApi(const std::string &path, LibraryHandle &library) {
	library = OpenLibrary(path);
	if (!library) {
		// the loader's own text is the useful part: a plugin present but unloadable usually names the missing dependency
		throw PluginError(PluginError::Kind::NOT_LOADABLE, LibraryError());
	}
	auto entry = reinterpret_cast<AnofoxGgmlGetPluginApiFn>(LibrarySymbol(library, ANOFOX_GGML_PLUGIN_ENTRY_SYMBOL));
	if (!entry) {
		CloseLibrary(library);
		throw PluginError(PluginError::Kind::NOT_A_PLUGIN, std::string("it exports no ") + ANOFOX_GGML_PLUGIN_ENTRY_SYMBOL +
		                                                       " (a file with the right name but the wrong contents)");
	}
	const AnofoxGgmlPluginApi *api = entry();
	if (!api) {
		CloseLibrary(library);
		throw PluginError(PluginError::Kind::NOT_A_PLUGIN, "it returned no API table");
	}
	if (api->abi_version != ANOFOX_GGML_PLUGIN_ABI_VERSION) {
		const int plugin_abi = api->abi_version; // read BEFORE unloading: `api` points into the library's memory
		CloseLibrary(library);
		throw PluginError(PluginError::Kind::ABI_MISMATCH,
		                  "built against plugin ABI version " + std::to_string(plugin_abi) + ", this build speaks version " +
		                      std::to_string(ANOFOX_GGML_PLUGIN_ABI_VERSION),
		                  std::string(), plugin_abi);
	}
	return api;
}

void SetDetail(char *err, std::string &out) {
	err[511] = '\0';
	out = err[0] ? std::string(err) : std::string("(the plugin gave no reason)");
}

} // namespace

PluginError::PluginError(Kind kind_p, std::string detail_p, std::string backend_p, int plugin_abi_p)
    : std::runtime_error(KindText(kind_p) + (backend_p.empty() ? std::string() : " ('" + backend_p + "')") + ": " + detail_p),
      kind(kind_p), detail(std::move(detail_p)), backend(std::move(backend_p)), plugin_abi(plugin_abi_p) {
}

bool PluginLoadable(const std::string &path, std::string *error) {
	LibraryHandle library;
	try {
		LoadApi(path, library);
	} catch (const PluginError &e) {
		if (error) {
			*error = e.detail;
		}
		return false;
	}
	// Deliberately not unloaded: the host is about to open it for real, and a GPU plugin mapped once must not be unmapped under a
	// driver context it may have registered.
	return true;
}

PluginSession::PluginSession(const AnofoxGgmlPluginApi *api, void *handle) : api_(api), handle_(handle), name_(api->name()) {
	const char *const *names = api->graph_inputs ? api->graph_inputs(handle) : nullptr;
	for (auto *n = names; n && *n; n++) {
		graph_inputs_.emplace_back(*n);
	}
}

PluginSession::~PluginSession() {
	if (api_ && handle_) {
		api_->destroy(handle_);
	}
	// The library is intentionally NOT closed: CUDA/HIP/Metal contexts and thread-locals the plugin registered outlive this object.
}

std::unique_ptr<PluginSession> PluginSession::Open(const std::string &path, const PluginCreate &params) {
	LibraryHandle library;
	const AnofoxGgmlPluginApi *api = LoadApi(path, library);
	std::vector<AnofoxGgmlWeight> weights;
	weights.reserve(params.weights.size());
	for (const auto &w : params.weights) {
		weights.push_back({w.name.c_str(), w.dtype, w.data, w.count, w.shape.empty() ? -1 : static_cast<int32_t>(w.shape.size()),
		                   w.shape.empty() ? nullptr : w.shape.data()});
	}
	AnofoxGgmlCreateParams c {};
	c.graph_path = params.graph_path.c_str();
	c.graph_data = params.graph_data;
	c.graph_size = params.graph_size;
	c.weights = weights.empty() ? nullptr : weights.data();
	c.weight_count = weights.size();
	c.model = params.model.c_str();
	c.cache_dir = params.cache_dir.c_str();
	c.arch = params.arch.c_str();
	c.precision = params.precision.c_str();
	c.out_of_memory_remedy = params.out_of_memory_remedy.c_str();
	c.device_ordinal = params.device_ordinal;
	c.threads = params.threads;
	char err[512] = {0};
	void *handle = api->create(&c, err, sizeof(err));
	if (!handle) {
		// NOT closed: create executed plugin code (a GPU plugin initialises its runtime before it can fail), leaving TLS and atexit
		// registrations that point into the library.
		std::string detail;
		SetDetail(err, detail);
		throw PluginError(PluginError::Kind::CREATE_FAILED, detail, api->name());
	}
	return std::unique_ptr<PluginSession>(new PluginSession(api, handle));
}

HostTensors PluginSession::Run(const std::vector<PluginTensor> &inputs, const std::vector<std::string> &outputs) {
	std::vector<AnofoxGgmlTensorIn> in;
	in.reserve(inputs.size());
	for (const auto &t : inputs) {
		in.push_back({t.name.c_str(), t.dtype, static_cast<int32_t>(t.shape.size()), t.shape.data(), t.data, t.fold ? 1 : 0});
	}
	std::vector<const char *> out_names;
	out_names.reserve(outputs.size());
	for (const auto &o : outputs) {
		out_names.push_back(o.c_str());
	}
	AnofoxGgmlRunOutput result {};
	char err[512] = {0};
	const auto status = api_->run(handle_, in.data(), in.size(), out_names.data(), out_names.size(), &result, err, sizeof(err));
	if (status != ANOFOX_GGML_PLUGIN_OK) {
		api_->free_output(&result);
		std::string detail;
		SetDetail(err, detail);
		throw PluginError(status == ANOFOX_GGML_PLUGIN_OUT_OF_MEMORY ? PluginError::Kind::OUT_OF_MEMORY : PluginError::Kind::RUN_FAILED,
		                  detail, name_);
	}
	HostTensors host;
	for (size_t i = 0; i < result.count; i++) {
		const auto &t = result.tensors[i];
		HostTensor h;
		h.shape.assign(t.shape, t.shape + t.rank);
		h.data.assign(t.data, t.data + t.count);
		host[t.name ? t.name : ""] = std::move(h);
	}
	api_->free_output(&result);
	for (const auto &o : outputs) {
		if (!host.count(o)) {
			throw PluginError(PluginError::Kind::RUN_FAILED, "the plugin did not return the output '" + o + "'", name_);
		}
	}
	return host;
}

} // namespace ggml_adapter
} // namespace anofox
