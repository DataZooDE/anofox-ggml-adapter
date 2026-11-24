# The MLX backend plugin (src/plugin_mlx.cpp + src/mlxgraph, plugin ABI v3): Apple Silicon only, shared by every host extension.
#
#   brew install mlx mlx-c
#   cmake -S tools/mlx_plugin -B build/mlx -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build/mlx
#
# Output: libanofox_mlx_plugin.dylib. It links the user's MLX (libmlx, libmlxc) and carries its own minimal ONNX reader (the adapter's
# onnx_reader.cpp) rather than libonnx/libprotobuf: a separately distributed .dylib must not be coupled to protobuf's ABI.
cmake_minimum_required(VERSION 3.18)
project(anofox_mlx_plugin CXX)
set(_root "${CMAKE_CURRENT_LIST_DIR}/../..")
set(ANOFOX_MLX_DIR "/opt/homebrew" CACHE PATH "MLX + mlx-c install prefix (include/mlx/c, lib/libmlxc)")
if(NOT (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64"))
    message(FATAL_ERROR "the MLX plugin builds on Apple Silicon only")
endif()
find_library(ANOFOX_MLX_C_LIB mlxc PATHS "${ANOFOX_MLX_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(ANOFOX_MLX_LIB mlx PATHS "${ANOFOX_MLX_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
add_library(anofox_mlx_plugin SHARED
    "${_root}/src/plugin_mlx.cpp"
    "${_root}/src/mlxgraph/mlx_graph.cpp"
    "${_root}/src/onnx_reader.cpp")
target_include_directories(anofox_mlx_plugin PRIVATE "${_root}/src" "${_root}/include" "${ANOFOX_MLX_DIR}/include")
target_link_libraries(anofox_mlx_plugin PRIVATE "${ANOFOX_MLX_C_LIB}" "${ANOFOX_MLX_LIB}")
target_compile_features(anofox_mlx_plugin PRIVATE cxx_std_17)
set_target_properties(anofox_mlx_plugin PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
# mlx-c 0.7 changed two calls the plugin makes (cumsum over an axis became mlx_cumsum_axis with an output dtype; scaled dot-product
# attention gained force_fused) and has no version macro: probe for the new API.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${ANOFOX_MLX_DIR}/include")
set(CMAKE_REQUIRED_LIBRARIES "${ANOFOX_MLX_C_LIB}" "${ANOFOX_MLX_LIB}")
check_cxx_source_compiles("
#include \"mlx/c/ops.h\"
int main() {
    mlx_array r = mlx_array_new();
    mlx_optional_dtype none;
    none.has_value = false;
    return mlx_cumsum_axis(&r, r, 0, false, true, none, mlx_default_cpu_stream_new());
}" ANOFOX_MLX_C_HAS_CUMSUM_AXIS)
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_LIBRARIES)
if(ANOFOX_MLX_C_HAS_CUMSUM_AXIS)
    target_compile_definitions(anofox_mlx_plugin PRIVATE ANOFOX_MLX_C_07=1)
endif()
