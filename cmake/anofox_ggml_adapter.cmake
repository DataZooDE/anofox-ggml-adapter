# anofox-ggml-adapter for a consuming project (a DuckDB extension, a GPU plugin, this repo's own tests):
#
#   include(<path>/anofox-ggml-adapter/cmake/anofox_ggml_adapter.cmake)
#   list(APPEND MY_SOURCES ${ANOFOX_GGML_ADAPTER_SOURCES})     # compile the engine into your target
#   add_dependencies(my_target ggml)                           # build order
#   target_link_libraries(my_target anofox_ggml)               # ggml archives + include dirs (adapter and ggml)
#
# The engine sources are compiled into the consumer (one copy of the engine per binary, built with the consumer's flags);
# ggml is built from vendor/ggml as static archives with hidden visibility.
#
# Variables a consumer may set BEFORE the include:
#   ANOFOX_GGML_CPU_ONLY  (default ON)  configure ggml with the CPU backend only. A GPU plugin sets it OFF and its own GGML_CUDA /
#                                       GGML_HIP options first.
#   ANOFOX_GGML_NATIVE    (option)      compile ggml for this machine's CPU (fast, may crash on older CPUs). Default OFF (ON on MSVC arm64,
#                                       where ggml-cpu requires it).
# Sets:
#   ANOFOX_GGML_ADAPTER_SOURCES, ANOFOX_GGML_ADAPTER_INCLUDE_DIR, target anofox_ggml (INTERFACE IMPORTED GLOBAL: the CPU archives; a GPU
#   consumer links the `ggml` target itself, which carries its GPU backend).

# The variables are set for every includer (they are directory-scoped); the targets are created once (they are global). One CMake
# tree that builds two consumers (e.g. two extensions in one DuckDB build) therefore shares the first includer's ggml and adapter; each
# consumer compiles the engine sources into its own binary.
get_filename_component(ANOFOX_GGML_ADAPTER_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(ANOFOX_GGML_ADAPTER_INCLUDE_DIR "${ANOFOX_GGML_ADAPTER_DIR}/include")
set(ANOFOX_GGML_ADAPTER_SOURCES
    "${ANOFOX_GGML_ADAPTER_DIR}/src/ggml_lower.cpp"
    "${ANOFOX_GGML_ADAPTER_DIR}/src/shape_eval.cpp"
    "${ANOFOX_GGML_ADAPTER_DIR}/src/onnx_reader.cpp"
    "${ANOFOX_GGML_ADAPTER_DIR}/src/ggml_engine.cpp"
    "${ANOFOX_GGML_ADAPTER_DIR}/src/cpu_support.cpp")
if(TARGET anofox_ggml)
    return()
endif()

if(NOT DEFINED ANOFOX_GGML_CPU_ONLY)
    set(ANOFOX_GGML_CPU_ONLY ON)
endif()
set(_anofox_ggml_native_default OFF)
if(MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64|aarch64")
    set(_anofox_ggml_native_default ON)
endif()
option(ANOFOX_GGML_NATIVE "Compile ggml for this machine's CPU (fast, but the binary may crash on older CPUs)" ${_anofox_ggml_native_default})

if(NOT EXISTS "${ANOFOX_GGML_ADAPTER_DIR}/vendor/ggml/CMakeLists.txt")
    message(FATAL_ERROR "anofox-ggml-adapter: vendor/ggml is empty. Run: git submodule update --init --recursive")
endif()

if(NOT TARGET ggml)
    # Static. No OpenMP: libgomp's spin-waiting starves ggml's own thread pool (measured 24 s vs 4 s on Nori 1.5), and the engine's threads
    # come from the host's setting, not OMP_NUM_THREADS.
    set(_anofox_saved_shared ${BUILD_SHARED_LIBS})
    set(_anofox_saved_pic ${CMAKE_POSITION_INDEPENDENT_CODE})
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    if(ANOFOX_GGML_CPU_ONLY)
        # ggml defaults Metal, Accelerate and BLAS to ON on Apple, which would add backends this wrapper does not link (undefined
        # ggml_backend_metal_reg, no frameworks, a Metal shader library nobody asked for).
        set(GGML_METAL OFF CACHE BOOL "" FORCE)
        set(GGML_BLAS OFF CACHE BOOL "" FORCE)
        set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
        set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
    endif()
    # Static ggml gets default visibility in a loadable extension; hide it so an interposing ggml elsewhere in the process (a llama.cpp
    # host, a preloaded library, the other anofox extension) cannot capture our calls.
    set(_anofox_saved_cvis ${CMAKE_C_VISIBILITY_PRESET})
    set(_anofox_saved_cxxvis ${CMAKE_CXX_VISIBILITY_PRESET})
    set(CMAKE_C_VISIBILITY_PRESET hidden)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(GGML_NATIVE ${ANOFOX_GGML_NATIVE} CACHE BOOL "" FORCE)
    # tinyBLAS: ggml's tuned float SGEMM for the CPU backend (off by default in a standalone ggml build).
    set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
    add_subdirectory("${ANOFOX_GGML_ADAPTER_DIR}/vendor/ggml" "${CMAKE_BINARY_DIR}/anofox_ggml_adapter_ggml" EXCLUDE_FROM_ALL)
    # ggml's CPU kernels carry plain assert()s such as !isnan(x) in softmax. They are live in a Debug build, and NaN is valid data (graphs mask
    # attention with NaN, the goldens feed NaN rows on purpose): a Debug build would abort() where a Release build returns NaN. GGML_ASSERT
    # (shape/type contracts) is a separate macro and stays on.
    foreach(_tgt ggml ggml-base ggml-cpu)
        if(TARGET ${_tgt})
            target_compile_definitions(${_tgt} PRIVATE NDEBUG)
        endif()
    endforeach()
    set(BUILD_SHARED_LIBS ${_anofox_saved_shared})
    set(CMAKE_POSITION_INDEPENDENT_CODE ${_anofox_saved_pic})
    set(CMAKE_C_VISIBILITY_PRESET ${_anofox_saved_cvis})
    set(CMAKE_CXX_VISIBILITY_PRESET ${_anofox_saved_cxxvis})
endif()

find_package(Threads REQUIRED)
# DuckDB export()s the extension targets and rejects non-imported build-tree dependencies (install(EXPORT) fails with 'requires target
# ggml that is not in any export set'), so the archives are wrapped in an IMPORTED INTERFACE target; consumers add_dependencies() on
# `ggml` for build order.
add_library(anofox_ggml INTERFACE IMPORTED GLOBAL)
set(_anofox_ggml_libs "$<TARGET_FILE:ggml>;$<TARGET_FILE:ggml-cpu>;$<TARGET_FILE:ggml-base>")
set_target_properties(anofox_ggml PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ANOFOX_GGML_ADAPTER_INCLUDE_DIR};${ANOFOX_GGML_ADAPTER_DIR}/vendor/ggml/include"
    INTERFACE_LINK_LIBRARIES "${_anofox_ggml_libs};Threads::Threads;${CMAKE_DL_LIBS}")
