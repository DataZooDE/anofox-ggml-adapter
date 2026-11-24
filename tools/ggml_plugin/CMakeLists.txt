# The generic ggml GPU plugin (src/plugin_ggml.cpp, plugin ABI v3): one library per GPU backend, shared by every host extension
# (anofox-tabfm, anofox-decide), running the same engine as their CPU builds on a ggml GPU device.
#
#   cmake -S tools/ggml_plugin -B build/ggml-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DANOFOX_GGML_GPU=cuda -DCMAKE_CUDA_ARCHITECTURES=89
#   cmake -S tools/ggml_plugin -B build/ggml-rocm -G Ninja -DCMAKE_BUILD_TYPE=Release -DANOFOX_GGML_GPU=hip  -DAMDGPU_TARGETS=gfx1201
#   cmake --build build/ggml-cuda
#
# Output: libanofox_ggml-cuda_plugin.so / libanofox_ggml-rocm_plugin.so (no 'lib' and a .dll on Windows). The CUDA build links the CUDA
# runtime and cuBLAS statically, so a user needs only the NVIDIA driver (scripts/check_plugin_deps.sh enforces it); the ROCm build links
# the system's HIP, hipBLAS and rocBLAS.
cmake_minimum_required(VERSION 3.18)
project(anofox_ggml_plugin CXX C)

set(ANOFOX_GGML_GPU "cuda" CACHE STRING "ggml GPU backend of this plugin: cuda | hip")
set_property(CACHE ANOFOX_GGML_GPU PROPERTY STRINGS cuda hip)
set(_root "${CMAKE_CURRENT_LIST_DIR}/../..")

if(ANOFOX_GGML_GPU STREQUAL "cuda")
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(_plugin_backend "ggml-cuda")
elseif(ANOFOX_GGML_GPU STREQUAL "hip")
    set(GGML_HIP ON CACHE BOOL "" FORCE)
    set(_plugin_backend "ggml-rocm")
else()
    message(FATAL_ERROR "ANOFOX_GGML_GPU must be cuda or hip")
endif()
set(ANOFOX_GGML_CUDA_STATIC ON CACHE BOOL "Link the CUDA runtime and cuBLAS statically (only the NVIDIA driver is then needed)")
# ggml configured with the GPU backend chosen above: static, no OpenMP (libgomp's spin-waiting starves ggml's own pool), llamafile on,
# GGML_NATIVE off (a distributable plugin must not depend on the build machine's CPU), hidden visibility, NDEBUG on ggml.
set(ANOFOX_GGML_CPU_ONLY OFF)
set(ANOFOX_GGML_NATIVE OFF CACHE BOOL "" FORCE)
include("${_root}/cmake/anofox_ggml_adapter.cmake")
if(ANOFOX_GGML_GPU STREQUAL "cuda" AND ANOFOX_GGML_CUDA_STATIC AND NOT WIN32)
    # ggml's own GGML_STATIC adds -static to every link, which a shared plugin cannot take; select the static CUDA targets for ggml-cuda
    # instead (Windows has no static cuBLAS). The libraries travel as $<LINK_ONLY:CUDA::...> in the static archive's interface, so both
    # properties are rewritten.
    find_package(CUDAToolkit REQUIRED)
    foreach(_prop LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
        get_target_property(_libs ggml-cuda ${_prop})
        if(_libs)
            string(REGEX REPLACE "CUDA::(cudart|cublasLt|cublas)([>;]|$)" "CUDA::\\1_static\\2" _libs "${_libs}")
            set_property(TARGET ggml-cuda PROPERTY ${_prop} "${_libs}")
        endif()
    endforeach()
endif()

add_library(anofox_ggml_plugin SHARED "${_root}/src/plugin_ggml.cpp" ${ANOFOX_GGML_ADAPTER_SOURCES})
target_include_directories(anofox_ggml_plugin PRIVATE "${ANOFOX_GGML_ADAPTER_INCLUDE_DIR}")
target_compile_features(anofox_ggml_plugin PRIVATE cxx_std_17)
set_target_properties(anofox_ggml_plugin PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
    OUTPUT_NAME "anofox_${_plugin_backend}_plugin")
if(WIN32)
    set_target_properties(anofox_ggml_plugin PROPERTIES PREFIX "")
endif()
target_link_libraries(anofox_ggml_plugin PRIVATE ggml)
