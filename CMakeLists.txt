# anofox-ggml-adapter: ONNX graphs on ggml (the engine shared by anofox-tabfm and anofox-decide).
#
# Standalone build (this repo's own tests):
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ./build/adapter_tests
# Consumers include cmake/anofox_ggml_adapter.cmake instead (see that file).
cmake_minimum_required(VERSION 3.21)
project(anofox_ggml_adapter LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/anofox_ggml_adapter.cmake)

option(ANOFOX_GGML_ADAPTER_TESTS "Build the adapter's own tests" ON)
if(ANOFOX_GGML_ADAPTER_TESTS)
    add_executable(adapter_tests
        test/cpp/test_main.cpp
        test/cpp/test_ggml_ops.cpp
        test/cpp/test_ggml_engine.cpp
        test/cpp/test_rank_reduce.cpp
        test/cpp/test_shape_eval.cpp
        test/cpp/test_plugin_host.cpp
        test/cpp/test_plugin_ggml.cpp
        test/cpp/test_plugin_artifacts.cpp
        ${ANOFOX_GGML_ADAPTER_SOURCES})
    add_dependencies(adapter_tests ggml)
    target_include_directories(adapter_tests PRIVATE test/cpp)
    target_link_libraries(adapter_tests PRIVATE anofox_ggml)
    # The plugin loader is tested against real libraries: the fake plugin, the same source built against another ABI version, and a
    # shared library that is no plugin at all.
    foreach(_fake fake_plugin fake_plugin_bad_abi)
        add_library(${_fake} MODULE test/cpp/plugin_fixture/fake_plugin.cpp)
        target_include_directories(${_fake} PRIVATE "${ANOFOX_GGML_ADAPTER_INCLUDE_DIR}")
        set_target_properties(${_fake} PROPERTIES CXX_VISIBILITY_PRESET hidden PREFIX "lib")
    endforeach()
    target_compile_definitions(fake_plugin_bad_abi PRIVATE ANOFOX_FAKE_PLUGIN_BAD_ABI)
    add_library(not_a_plugin MODULE test/cpp/plugin_fixture/not_a_plugin.cpp)
    set_target_properties(not_a_plugin PROPERTIES PREFIX "lib")
    # The generic ggml plugin itself, built for the CPU device: CI runs the real plugin code end to end without a GPU.
    add_library(ggml_plugin_cpu MODULE src/plugin_ggml.cpp ${ANOFOX_GGML_ADAPTER_SOURCES})
    add_dependencies(ggml_plugin_cpu ggml)
    target_compile_definitions(ggml_plugin_cpu PRIVATE ANOFOX_GGML_PLUGIN_CPU_TEST)
    target_link_libraries(ggml_plugin_cpu PRIVATE anofox_ggml)
    set_target_properties(ggml_plugin_cpu PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON PREFIX "lib")
    add_dependencies(adapter_tests fake_plugin fake_plugin_bad_abi not_a_plugin ggml_plugin_cpu)
    target_compile_definitions(adapter_tests PRIVATE
        ANOFOX_FAKE_PLUGIN="$<TARGET_FILE:fake_plugin>"
        ANOFOX_FAKE_PLUGIN_BAD_ABI="$<TARGET_FILE:fake_plugin_bad_abi>"
        ANOFOX_NOT_A_PLUGIN="$<TARGET_FILE:not_a_plugin>"
        ANOFOX_GGML_CPU_PLUGIN="$<TARGET_FILE:ggml_plugin_cpu>")
    target_link_libraries(adapter_tests PRIVATE ${CMAKE_DL_LIBS})
    # the tests read their fixtures relative to the repository root
    set_target_properties(adapter_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
