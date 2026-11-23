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
        ${ANOFOX_GGML_ADAPTER_SOURCES})
    add_dependencies(adapter_tests ggml)
    target_include_directories(adapter_tests PRIVATE test/cpp)
    target_link_libraries(adapter_tests PRIVATE anofox_ggml)
    # the tests read their fixtures relative to the repository root
    set_target_properties(adapter_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
