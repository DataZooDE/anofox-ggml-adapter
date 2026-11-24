// A shared library that loads but exports no plugin entry point (a file with the right name but the wrong contents).
extern "C" int anofox_not_a_plugin(void) {
	return 42;
}
