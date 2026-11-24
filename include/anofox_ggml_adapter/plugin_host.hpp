#pragma once

// plugin_host — the host side of plugin_abi.h: load a GPU backend plugin, check its ABI, create a session and run it on named tensors.
//
// The loader is the trust boundary (a plugin built against another layout is undefined behaviour, not a wrong answer), so every
// failure is a PluginError with a kind: the host turns it into its own message with its own remedy (its SET / CALL), the adapter
// never names a host setting.

#include "anofox_ggml_adapter/ggml_lower.hpp"
#include "anofox_ggml_adapter/plugin_abi.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace anofox {
namespace ggml_adapter {

class PluginError : public std::runtime_error {
public:
	enum class Kind {
		NOT_LOADABLE, //!< the file is missing or the dynamic loader refused it (detail: the loader's text, e.g. a missing libcuda.so.1)
		NOT_A_PLUGIN, //!< no entry symbol, or no API table
		ABI_MISMATCH, //!< built against another ABI version (plugin_abi)
		CREATE_FAILED, //!< the plugin's create refused (detail: its message)
		RUN_FAILED, //!< a run failed (detail: its message)
		OUT_OF_MEMORY //!< a run ran out of device memory
	};
	PluginError(Kind kind_p, std::string detail_p, std::string backend_p = std::string(), int plugin_abi_p = 0);
	Kind kind;
	std::string detail;  //!< what the loader or the plugin said
	std::string backend; //!< the plugin's name(), when it got that far
	int plugin_abi;      //!< for ABI_MISMATCH: the version the plugin was built against
};

//! True when `path` loads and speaks this ABI; otherwise false and `*error` says why (without creating anything).
bool PluginLoadable(const std::string &path, std::string *error = nullptr);

struct PluginWeight {
	std::string name;
	int32_t dtype = ANOFOX_GGML_DTYPE_FLOAT; //!< FLOAT, FLOAT16 or BFLOAT16
	const void *data = nullptr;              //!< the host's memory: it must outlive the session
	int64_t count = 0;
	std::vector<int64_t> shape;              //!< optional (empty = unknown); checked against the graph when given
};

struct PluginCreate {
	std::string graph_path;            //!< the graph file, or:
	const void *graph_data = nullptr;  //!< the graph's bytes (read during Open only)
	size_t graph_size = 0;
	std::vector<PluginWeight> weights; //!< external initializers by name (their memory must outlive the session)
	std::string model;                 //!< the host's model name (a plugin may have a fast path for it)
	std::string cache_dir;
	std::string arch;
	std::string precision = "fp32";
	std::string out_of_memory_remedy;
	int device_ordinal = 0;
	int threads = 0;
};

struct PluginTensor {
	std::string name;
	int32_t dtype = ANOFOX_GGML_DTYPE_FLOAT; //!< FLOAT, INT64, BOOL (one uint8 per element)
	std::vector<int64_t> shape;
	const void *data = nullptr;
	bool fold = false; //!< an integer folded into the compiled graph (part of its compile signature)
};

class PluginSession {
public:
	//! Load the library at `path` (never unloaded once a plugin's code ran: GPU runtimes keep TLS and atexit hooks in it) and create a
	//! session. Throws PluginError.
	static std::unique_ptr<PluginSession> Open(const std::string &path, const PluginCreate &params);
	~PluginSession();
	PluginSession(const PluginSession &) = delete;
	PluginSession &operator=(const PluginSession &) = delete;

	const std::string &Name() const {
		return name_;
	}
	//! The graph's inputs (initializers excluded), as the plugin read them.
	const std::vector<std::string> &GraphInputs() const {
		return graph_inputs_;
	}
	//! Run on `inputs` and return `outputs` by name (as float32). Throws PluginError (RUN_FAILED, OUT_OF_MEMORY). Thread-safe only
	//! as far as the plugin is; the ggml plugins serialise runs per session.
	HostTensors Run(const std::vector<PluginTensor> &inputs, const std::vector<std::string> &outputs);

private:
	PluginSession(const AnofoxGgmlPluginApi *api, void *handle);
	const AnofoxGgmlPluginApi *api_;
	void *handle_;
	std::string name_;
	std::vector<std::string> graph_inputs_;
};

} // namespace ggml_adapter
} // namespace anofox
