/*
 * plugin_load_check — does a built backend plugin LOAD on a machine with no GPU, and refuse cleanly?
 *
 * CI builds the plugins on plain runners; nm proves the entry symbol is in the file, but only a real dlopen proves the artifact resolves
 * (a missing DT_NEEDED, a bad rpath and an SONAME regression all pass nm and fail here, the way they would on a user's machine).
 *
 *   PASS: dlopen resolves, AnofoxGgmlGetPluginApi answers, and the ABI version matches the header this harness was built against.
 *
 * It then calls create() with a graph that does not exist. Either outcome is fine; a crash is not: a user pointing a host at a plugin on a
 * machine without the hardware must get an error message, not a SIGSEGV. Reaching the last printf IS that assertion.
 *
 * Usage: plugin_load_check <plugin library>
 */
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include "../../include/anofox_ggml_adapter/plugin_abi.h"

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "usage: %s <plugin library>\n", argv[0]);
		return 2;
	}
	void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
	if (!lib) {
		printf("LOAD_CHECK_FAIL dlopen: %s\n", dlerror());
		return 1;
	}
	AnofoxGgmlGetPluginApiFn get_api = (AnofoxGgmlGetPluginApiFn)dlsym(lib, ANOFOX_GGML_PLUGIN_ENTRY_SYMBOL);
	if (!get_api) {
		printf("LOAD_CHECK_FAIL dlsym(%s): %s\n", ANOFOX_GGML_PLUGIN_ENTRY_SYMBOL, dlerror());
		return 1;
	}
	const AnofoxGgmlPluginApi *api = get_api();
	if (!api) {
		printf("LOAD_CHECK_FAIL %s returned NULL\n", ANOFOX_GGML_PLUGIN_ENTRY_SYMBOL);
		return 1;
	}
	if (api->abi_version != ANOFOX_GGML_PLUGIN_ABI_VERSION) {
		printf("LOAD_CHECK_FAIL abi_version %d, harness built against %d\n", api->abi_version, ANOFOX_GGML_PLUGIN_ABI_VERSION);
		return 1;
	}
	printf("LOAD_CHECK_OK name=%s abi=%d\n", api->name ? api->name() : "(null)", api->abi_version);
	AnofoxGgmlCreateParams params;
	memset(&params, 0, sizeof(params));
	params.graph_path = "/nonexistent/graph.onnx";
	params.cache_dir = "/tmp";
	params.arch = "load-check";
	params.precision = "fp32";
	params.model = "load-check";
	char err[2048];
	err[0] = '\0';
	void *handle = api->create(&params, err, sizeof(err));
	if (handle) {
		printf("CREATE_WITHOUT_GPU unexpectedly succeeded; destroying\n");
		api->destroy(handle);
	} else {
		printf("CREATE_WITHOUT_GPU graceful error: %s\n", err[0] ? err : "(empty message)");
	}
	return 0;
}
