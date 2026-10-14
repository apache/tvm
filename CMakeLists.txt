cmake_minimum_required(VERSION 3.18)
project(tvm C CXX)

# --- TVM version (no dependency on version.py) ---
# When built via scikit-build-core the version is resolved by setuptools_scm and
# passed in as SKBUILD_PROJECT_VERSION_FULL; bake it into the C++ TVM_VERSION macro.
# A bare `cmake` build (no scikit-build-core) leaves the checked-in default in
# include/tvm/runtime/base.h untouched. An explicit -DTVM_VERSION always wins.
if(NOT DEFINED TVM_VERSION AND DEFINED SKBUILD_PROJECT_VERSION_FULL)
  set(TVM_VERSION "${SKBUILD_PROJECT_VERSION_FULL}")
endif()
if(DEFINED TVM_VERSION)
  message(STATUS "TVM_VERSION=${TVM_VERSION}")
  add_compile_definitions(TVM_VERSION="${TVM_VERSION}")
endif()

# Utility functions
include(cmake/utils/Utils.cmake)
include(cmake/utils/Summary.cmake)
include(cmake/utils/Linker.cmake)
include(cmake/utils/Library.cmake)
include(cmake/utils/FindCUDA.cmake)
include(cmake/utils/FindNCCL.cmake)
include(cmake/utils/FindOpenCL.cmake)
include(cmake/utils/FindVulkan.cmake)
include(cmake/utils/FindLLVM.cmake)
include(cmake/utils/FindROCM.cmake)
include(cmake/utils/FindRCCL.cmake)
include(cmake/utils/FindNVSHMEM.cmake)

if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
  include(${CMAKE_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
    include(${CMAKE_SOURCE_DIR}/config.cmake)
  endif()
endif()

# NOTE: do not modify this file to change option values.
# You can create a config.cmake at build folder
# and add set(OPTION VALUE) to override these build options.
# Alernatively, use cmake -DOPTION=VALUE through command-line.
tvm_option(USE_CUDA "Build with CUDA" OFF)
tvm_option(USE_NCCL "Build with NCCL" OFF)
tvm_option(USE_OPENCL "Build with OpenCL" OFF)
tvm_option(USE_OPENCL_ENABLE_HOST_PTR "Enable OpenCL memory object access to host" OFF)
tvm_option(USE_OPENCL_GTEST "Path to OpenCL specific gtest version for runtime cpp tests." /path/to/opencl/gtest)
tvm_option(USE_VULKAN "Build with Vulkan" OFF)


# Whether to use spirv-tools and SPIRV-Headers from Khronos GitHub or GitLab.
#
# Possible values:
# - OFF: not to use
# - /path/to/install: path to your khronis spirv-tools and SPIRV-Headers installation directory
#
tvm_option(USE_KHRONOS_SPIRV "Whether to use spirv-tools and SPIRV-Headers from Khronos GitHub or GitLab" OFF)
tvm_option(USE_SPIRV_KHR_INTEGER_DOT_PRODUCT "whether enable SPIRV_KHR_DOT_PRODUCT" OFF)
tvm_option(USE_METAL "Build with Metal" OFF)
tvm_option(USE_ROCM "Build with ROCM" OFF)
tvm_option(USE_RCCL "Build with RCCL" OFF)
tvm_option(ROCM_PATH "The path to rocm" /opt/rocm)
tvm_option(USE_HEXAGON "Build with Hexagon support" OFF)
tvm_option(USE_HEXAGON_SDK "Path to the Hexagon SDK root (required for Hexagon support)" /path/to/sdk)
tvm_option(USE_HEXAGON_RPC "Enable Hexagon RPC using minRPC implementation over Android." OFF)
tvm_option(USE_HEXAGON_GTEST "Path to Hexagon specific gtest version for runtime cpp tests." /path/to/hexagon/gtest)
tvm_option(USE_HEXAGON_EXTERNAL_LIBS "Path to git repo containing external Hexagon runtime sources or libraries" OFF)

tvm_option(USE_RPC "Build with RPC" ON)
tvm_option(USE_THREADS "Build with thread support" ON)
tvm_option(USE_LLVM "Build with LLVM, can be set to specific llvm-config path" OFF)
tvm_option(USE_OPENMP "Build with OpenMP thread pool implementation" OFF)
tvm_option(TVM_DEBUG_WITH_ABI_CHANGE "Enable debug code that may cause ABI changes" OFF)
tvm_option(TVM_LOG_BEFORE_THROW "Whether log before throw, for debugging purposes" OFF)
tvm_option(USE_RTTI "Build with RTTI" ON)
tvm_option(USE_MSVC_MT "Build with MT" OFF)
tvm_option(INSTALL_DEV "Install compiler infrastructure" OFF)
tvm_option(HIDE_PRIVATE_SYMBOLS "Compile with -fvisibility=hidden." ON)
tvm_option(INDEX_DEFAULT_I64 "Defaults the index datatype to int64" ON)
tvm_option(USE_GTEST "Use GoogleTest for C++ sanity tests" AUTO)
tvm_option(USE_CUSTOM_LOGGING "Use user-defined custom logging, tvm::runtime::detail::LogFatalImpl and tvm::runtime::detail::LogMessageImpl must be implemented" OFF)
tvm_option(USE_ALTERNATIVE_LINKER "Use 'mold' or 'lld' if found when invoking compiler to link artifact" AUTO)
tvm_option(USE_CCACHE "Use ccache if found when invoking compiler" AUTO)

# 3rdparty libraries
tvm_option(COMPILER_RT_PATH "Path to COMPILER-RT" "3rdparty/compiler-rt")
# Contrib library options
tvm_option(USE_BLAS "The blas library to be linked" none)
tvm_option(USE_AMX "Enable Intel AMX" OFF)
tvm_option(USE_Z3 "Build with Z3 SMT solver support" AUTO)
tvm_option(USE_MKL "MKL root path when use MKL blas" OFF)
tvm_option(USE_DNNL "Enable DNNL codegen" OFF)
tvm_option(USE_CUDNN "Build with cuDNN" OFF)
tvm_option(USE_CUBLAS "Build with cuBLAS" OFF)
tvm_option(USE_NVTX "Build with NVTX" OFF)
tvm_option(USE_CUTLASS "Build with CUTLASS" OFF)
tvm_option(USE_THRUST "Build with Thrust" OFF)
tvm_option(USE_CURAND "Build with cuRAND" OFF)
tvm_option(USE_HIPBLAS "Build with ROCM:HIPBLAS" OFF)
tvm_option(USE_SORT "Build with sort support" ON)
tvm_option(USE_RANDOM "Build with random support" ON)
tvm_option(USE_CPP_RPC "Build CPP RPC" OFF)
tvm_option(USE_IOS_RPC "Build iOS RPC" OFF)
tvm_option(USE_COREML "Build with coreml support" OFF)
tvm_option(USE_TENSORRT_CODEGEN "Build with TensorRT Codegen support" OFF)
tvm_option(USE_TENSORRT_RUNTIME "Build with TensorRT runtime" OFF)
tvm_option(USE_NNAPI_CODEGEN "Build with NNAPI Codegen support" OFF)
tvm_option(USE_NNAPI_RUNTIME "Build with NNAPI runtime" OFF)
tvm_option(USE_EXAMPLE_NPU_CODEGEN "Build with Example NPU Codegen support" OFF)
tvm_option(USE_EXAMPLE_NPU_RUNTIME "Build with Example NPU runtime" OFF)
tvm_option(SUMMARIZE "Print CMake option summary after configuring" OFF)
tvm_option(USE_CLML "Build with CLML Codegen support" OFF)
tvm_option(USE_CLML_GRAPH_EXECUTOR "Build with CLML graph runtime" OFF)
tvm_option(USE_NVSHMEM "Build with NVSHMEM support" OFF)

# Python package options
tvm_option(TVM_BUILD_PYTHON_MODULE "Build Python module with scikit-build-core" OFF)

# include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories("include")
include_directories(SYSTEM ${COMPILER_RT_PATH})

# initial variables
set(TVM_LINKER_LIBS "")
set(TVM_RUNTIME_LINKER_LIBS "")
set(TVM_RUNTIME_BACKEND_LIBS "")
# Early target creation so contrib cmake files can call
# target_link_libraries(tvm_runtime_extra PRIVATE <object_lib>) directly.
add_library(tvm_runtime_extra SHARED)
list(APPEND TVM_RUNTIME_BACKEND_LIBS tvm_runtime_extra)
set_target_properties(tvm_runtime_extra PROPERTIES LINKER_LANGUAGE CXX)
# INTERFACE target carrying compile definitions for OBJECT libs that build
# into tvm_runtime_extra. On MSVC, TVM_RUNTIME_EXPORTS makes TVM_RUNTIME_DLL
# expand to __declspec(dllexport) so that functions defined in extra modules
# are properly exported from tvm_runtime_extra.dll.
add_library(tvm_runtime_extra_defs INTERFACE)
target_link_libraries(tvm_runtime_extra_defs INTERFACE tvm_ffi_header)
target_compile_definitions(tvm_runtime_extra_defs
  INTERFACE TVM_RUNTIME_EXPORTS TVM_FFI_EXPORTS)


# Check if this is being run on its own or as a subdirectory for another project
# If we update to CMake 2.21+, we can use PROJECT_IS_TOP_LEVEL instead
get_directory_property(IS_SUBPROJECT PARENT_DIRECTORY)

if(NOT IS_SUBPROJECT AND NOT DEFINED "${CMAKE_EXPORT_COMPILE_COMMANDS}")
  # If not set manually, change the default to ON
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(TVM_LOG_BEFORE_THROW)
  # log error before throw as
  # when system have issues with stack trace
  add_definitions(-DTVM_LOG_BEFORE_THROW=1)
endif()

# Generic compilation options
if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
  add_definitions(-DNOMINMAX)
  # regeneration does not work well with msbuild custom rules.
  set(CMAKE_SUPPRESS_REGENERATION ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  add_compile_options(/bigobj)

  # Use standard-conforming two-phase name resolution for templates.
  # This minimizes the differences between g++/clang builds on Linux,
  # and MSVC builds on Windows.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive-")

  # MSVC already errors on undefined symbols, no additional flag needed.
  set(TVM_NO_UNDEFINED_SYMBOLS "")

  if(USE_MSVC_MT)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
    # Static linking. CMake behavior changed in 3.15 making this necessary.
    add_compile_options(/MT)
  endif()
  # Disable common MSVC warnings
  # Integer conversion warnings(e.g. int64 to int)
  add_compile_options(/wd4244)
  add_compile_options(/wd4267)
  # Signed unsigned constant comparison
  add_compile_options(/wd4018)
  # Aligned alloc may not met(need c++17)
  add_compile_options(/wd4316)
  # unreferenced local variables(usually in exception catch)
  add_compile_options(/wd4101)
  # always inline keyword not necessary
  add_compile_options(/wd4180)
  # DLL interface warning in c++
  add_compile_options(/wd4251)
  # destructor was implicitly defined as deleted
  add_compile_options(/wd4624)
  # unary minus operator applied to unsigned type, result still unsigned
  add_compile_options(/wd4146)
  # 'inline': used more than once
  add_compile_options(/wd4141)
  # unknown pragma
  add_compile_options(/wd4068)
else(MSVC)
  set(WARNING_FLAG -Wall)
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Build in Debug mode")
    set(CMAKE_C_FLAGS "-O0 -g ${WARNING_FLAG} -fPIC ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O0 -g ${WARNING_FLAG} -fPIC ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "-O0 -g -Xcompiler=-Wall -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  else()
    set(CMAKE_C_FLAGS "-O2 ${WARNING_FLAG} -fPIC ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O2 ${WARNING_FLAG} -fPIC ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "-O2 -Xcompiler=-Wall -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
    set(TVM_VISIBILITY_FLAG "")
    if (HIDE_PRIVATE_SYMBOLS)
      message(STATUS "Hide private symbols...")
      set(TVM_VISIBILITY_FLAG "-fvisibility=hidden")
    endif(HIDE_PRIVATE_SYMBOLS)
  endif ()
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
  endif()

  # ld option to warn if symbols are undefined (e.g. libtvm_runtime.so
  # using symbols only present in libtvm.so).  Not needed for MSVC,
  # since this is already the default there.
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" OR ${CMAKE_SYSTEM_NAME} MATCHES "iOS")
    set(TVM_NO_UNDEFINED_SYMBOLS "-Wl,-undefined,error")
  else()
    set(TVM_NO_UNDEFINED_SYMBOLS "-Wl,--no-undefined")
  endif()
  message(STATUS "Forbidding undefined symbols in shared library, using ${TVM_NO_UNDEFINED_SYMBOLS} on platform ${CMAKE_SYSTEM_NAME}")

  # Detect if we're compiling for Hexagon.
  set(TEST_FOR_HEXAGON_CXX
      "#ifndef __hexagon__"
      "#error"
      "#endif"
      "int main() {}"
      # Define _start_main to avoid linking errors with -fPIC.
      "extern \"C\" void _start_main() {}")
  set(TEST_FOR_HEXAGON_DIR
      "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp")
  set(TEST_FOR_HEXAGON_FILE "${TEST_FOR_HEXAGON_DIR}/test_for_hexagon.cc")
  string(REPLACE ";" "\n" TEST_FOR_HEXAGON_CXX_TEXT "${TEST_FOR_HEXAGON_CXX}")
  file(WRITE "${TEST_FOR_HEXAGON_FILE}" "${TEST_FOR_HEXAGON_CXX_TEXT}")
  try_compile(BUILD_FOR_HEXAGON "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}"
              "${TEST_FOR_HEXAGON_FILE}")
  file(REMOVE "${TEST_FOR_HEXAGON_FILE}")
  if(BUILD_FOR_HEXAGON)
    message(STATUS "Building for Hexagon")
  endif()

  # Detect if we're compiling for Android.
  set(TEST_FOR_ANDROID_CXX
      "#ifndef __ANDROID__"
      "#error"
      "#endif"
      "int main() {}")
  set(TEST_FOR_ANDROID_DIR
      "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp")
  set(TEST_FOR_ANDROID_FILE "${TEST_FOR_ANDROID_DIR}/test_for_android.cc")
  string(REPLACE ";" "\n" TEST_FOR_ANDROID_CXX_TEXT "${TEST_FOR_ANDROID_CXX}")
  file(WRITE "${TEST_FOR_ANDROID_FILE}" "${TEST_FOR_ANDROID_CXX_TEXT}")
  try_compile(BUILD_FOR_ANDROID "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}"
              "${TEST_FOR_ANDROID_FILE}")
  file(REMOVE "${TEST_FOR_ANDROID_FILE}")
  if(BUILD_FOR_ANDROID)
    message(STATUS "Building for Android")
  endif()
endif(MSVC)

# Hexagon has dlopen built into QuRT (no need for static library).
if(NOT BUILD_FOR_HEXAGON)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CMAKE_DL_LIBS})
endif()

# add source group
tvm_file_glob(GLOB_RECURSE GROUP_SOURCE "src/*.cc")
tvm_file_glob(GLOB_RECURSE GROUP_INCLUDE "src/*.h" "include/*.h")
assign_source_group("Source" ${GROUP_SOURCE})
assign_source_group("Include" ${GROUP_INCLUDE})

# Source file lists
# Core compiler sources, extended by the backend modules.
tvm_file_glob(GLOB_RECURSE COMPILER_SRCS
    src/ir/*.cc
    src/sym/*.cc
    src/te/*.cc
    src/tirx/*.cc
    src/s_tir/*.cc
    src/topi/*.cc
    src/driver/*.cc
    src/support/*.cc
    # Shared printer and TVMScript IR builder. Per-dialect printer/builder pieces
    # live under src/<dialect>/script/. List shared files explicitly so a dialect
    # implementation cannot be silently included through a recursive glob.
    src/script/ir_builder/base.cc
    src/script/ir_builder/ir.cc
    src/script/ir_builder/frame.cc
    src/script/printer/config.cc
    src/script/printer/doc.cc
    src/script/printer/doc_translator.cc
    src/script/printer/ir/call.cc
    src/script/printer/ir/utils.cc
    src/script/printer/ir/ir.cc
    src/script/printer/ir/misc.cc
    src/script/printer/ir/prim_type.cc
    src/script/printer/ir/prim_expr.cc
    src/script/printer/ir/stmt.cc
    src/script/printer/printer.cc
    src/script/printer/doc_printer.cc
    src/script/printer/script_printer.cc
    src/relax/ir/*.cc
    src/relax/op/*.cc
    src/relax/analysis/*.cc
    src/relax/transform/*.cc
    src/relax/backend/vm/*.cc
    src/relax/backend/adreno/*.cc
    src/relax/backend/task_extraction.cc
    src/relax/backend/pattern_registry.cc
    src/relax/utils.cc
    src/relax/distributed/*.cc
    src/relax/distributed/transform/*.cc
    src/relax/op/distributed/*.cc
    src/relax/script/*.cc
    src/relax/testing/*.cc
    )

tvm_file_glob(GLOB CODEGEN_SRCS
  src/target/*.cc
  src/target/source/*.cc
  src/target/canonicalizer/*.cc
  src/target/llvm/canonicalizer/*.cc
  src/backend/cuda/codegen/*.cc
  src/backend/cuda/op/*.cc
  src/backend/cuda/transforms/*.cc
  src/backend/cuda/tile/*.cc
  src/backend/cuda/layout/*.cc
  src/backend/hexagon/codegen/*.cc
  src/backend/metal/codegen/*.cc
  src/backend/metal/op/*.cc
  src/backend/opencl/codegen/*.cc
  src/backend/opencl/op/*.cc
  src/backend/rocm/codegen/*.cc
  src/backend/trn/codegen/*.cc
  src/backend/trn/op/*.cc
  src/backend/trn/transform/*.cc
  src/backend/trn/layout/*.cc
  src/backend/vulkan/codegen/target_kind.cc
  src/backend/vulkan/codegen/vulkan_fallback_module.cc
  src/backend/webgpu/codegen/*.cc
    )

list(APPEND COMPILER_SRCS ${CODEGEN_SRCS})
# Core runtime sources, extended by the backend modules.
tvm_file_glob(GLOB RUNTIME_SRCS
  src/runtime/*.cc
  src/runtime/rpc/minrpc/*.cc
)
# VM-owned runtime implementations, including external-module constant loading.
tvm_file_glob(GLOB RUNTIME_VM_SRCS src/runtime/vm/*.cc)
list(APPEND RUNTIME_SRCS ${RUNTIME_VM_SRCS})
# Note: src/runtime/extra/disco/** moves to libtvm_runtime_extra.
# Note: src/backend/*/runtime sources move to per-backend DSOs.
set(TVM_RUNTIME_EXT_OBJS "")

if(USE_RPC)
  message(STATUS "Build with RPC support...")
  tvm_file_glob(GLOB RUNTIME_RPC_SRCS src/runtime/rpc/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_RPC_SRCS})
endif(USE_RPC)

if(BUILD_FOR_HEXAGON)
  # Allow undefined symbols in the shared runtime (there will be some from libc).
  set(TVM_NO_UNDEFINED_SYMBOLS "")

  add_definitions(-D_MACH_I32=int)
endif()

# Package runtime rules
if(NOT USE_RTTI)
endif()

if(INDEX_DEFAULT_I64)
  add_definitions(-DTVM_INDEX_DEFAULT_I64=1)
endif()

if(USE_KALLOC_ALIGNMENT)
  message(STATUS "Build Alloc alignment set to ${USE_KALLOC_ALIGNMENT}")
  add_definitions(-DTVM_KALLOC_ALIGNMENT=${USE_KALLOC_ALIGNMENT})
endif(USE_KALLOC_ALIGNMENT)

# Caches the build.
# Note that ccache-3.x doesn't support nvcc well, so CUDA kernels may never hit the cache and still
# need to be re-compiled every time. Using ccache 4.0+ can resolve this issue.
include(cmake/utils/CCache.cmake)

include(CheckCXXCompilerFlag)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

# Module rules
include(cmake/modules/CUDA.cmake)
include(cmake/modules/Hexagon.cmake) # This must come before logging.cmake
include(cmake/modules/contrib/CLML.cmake) # Must be before OpenCL.cmake
include(cmake/modules/OpenCL.cmake)
include(cmake/modules/OpenMP.cmake)
include(cmake/modules/Vulkan.cmake)
include(cmake/modules/Metal.cmake)
include(cmake/modules/ROCM.cmake)
include(cmake/modules/LLVM.cmake)
include(cmake/modules/contrib/BLAS.cmake)
include(cmake/modules/contrib/DNNL.cmake)
include(cmake/modules/contrib/AMX.cmake)
include(cmake/modules/contrib/CUTLASS.cmake)
include(cmake/modules/contrib/Random.cmake)
include(cmake/modules/contrib/Sort.cmake)
include(cmake/modules/contrib/Z3.cmake)
include(cmake/modules/contrib/CoreML.cmake)
include(cmake/modules/contrib/TensorRT.cmake)
include(cmake/modules/contrib/NNAPI.cmake)
include(cmake/modules/contrib/ExampleNPU.cmake)
include(cmake/modules/contrib/vllm.cmake)
include(cmake/modules/Git.cmake)

include(cmake/modules/RuntimeExtra.cmake)
add_subdirectory(3rdparty/tvm-ffi)
include(GNUInstallDirs)

if(USE_THREADS AND NOT BUILD_FOR_HEXAGON)
  message(STATUS "Build with thread support...")
  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)
endif()

set(TVM_IS_DEBUG_BUILD OFF)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR CMAKE_CXX_FLAGS MATCHES "-g")
  set(TVM_IS_DEBUG_BUILD ON)
  set(FILE_PREFIX_MAP_FLAG "-ffile-prefix-map=..=${CMAKE_CURRENT_SOURCE_DIR}")
  CHECK_CXX_COMPILER_FLAG("${FILE_PREFIX_MAP_FLAG}" FILE_PREFIX_MAP_SUPPORTED)
endif()
if(TVM_DEBUG_WITH_ABI_CHANGE)
  message(STATUS "Building with debug code that may cause ABI changes...")
endif()

# Core runtime objects are shared by the dynamic and static libraries.
add_library(tvm_runtime_objs OBJECT ${RUNTIME_SRCS})
target_link_libraries(tvm_runtime_objs PUBLIC tvm_ffi_header)
if(TVM_VISIBILITY_FLAG)
  target_compile_options(tvm_runtime_objs PRIVATE "${TVM_VISIBILITY_FLAG}")
endif()
if(TVM_DEBUG_WITH_ABI_CHANGE)
  target_compile_definitions(tvm_runtime_objs PRIVATE TVM_DEBUG_WITH_ABI_CHANGE)
endif()
if(MSVC)
  # Runtime objects export only runtime/FFI symbols, never compiler symbols.
  target_compile_definitions(tvm_runtime_objs PRIVATE TVM_RUNTIME_EXPORTS TVM_FFI_EXPORTS)
endif()

# Loaded with RTLD_GLOBAL so downstream runtime modules can resolve its symbols.
add_library(tvm_runtime SHARED
  $<TARGET_OBJECTS:tvm_runtime_objs>
  ${TVM_RUNTIME_EXT_OBJS}
)
target_link_libraries(tvm_runtime PUBLIC tvm_ffi_shared)
set_property(TARGET tvm_runtime APPEND PROPERTY LINK_OPTIONS "${TVM_NO_UNDEFINED_SYMBOLS}")
set_property(TARGET tvm_runtime APPEND PROPERTY LINK_OPTIONS "${TVM_VISIBILITY_FLAG}")

# Static consumers must retain registration objects from both archives, with
# tvm_ffi_static before tvm_runtime_static so FFI type attributes initialize first:
#   -Wl,--whole-archive tvm_ffi_static tvm_runtime_static -Wl,--no-whole-archive
# Use the corresponding archive-retention flags on other platforms.
add_library(tvm_runtime_static STATIC
  $<TARGET_OBJECTS:tvm_runtime_objs>
  ${TVM_RUNTIME_EXT_OBJS}
)
target_link_libraries(tvm_runtime_static PUBLIC tvm_ffi_static)
if(MSVC)
  # Static consumers reference ordinary symbols rather than DLL import stubs.
  target_compile_definitions(tvm_runtime_static INTERFACE
    TVM_RUNTIME_DLL= TVM_RUNTIME_DLL_EXPORT= TVM_FFI_DLL= TVM_FFI_DLL_EXPORT=)
endif()

foreach(runtime_target tvm_runtime tvm_runtime_static)
  target_include_directories(${runtime_target} PUBLIC "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  target_link_libraries(${runtime_target} PRIVATE ${TVM_RUNTIME_LINKER_LIBS})
  if(USE_THREADS AND NOT BUILD_FOR_HEXAGON)
    target_link_libraries(${runtime_target} PUBLIC Threads::Threads)
  endif()
endforeach()

set_target_properties(tvm_runtime tvm_runtime_static PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
if(MSVC)
  foreach(config_type Release RELEASE Debug DEBUG RelWithDebInfo RELWITHDEBINFO MinSizeRel MINSIZEREL)
    set_target_properties(tvm_runtime tvm_runtime_static PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
      LIBRARY_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
    )
  endforeach()
endif()

if(TVM_IS_DEBUG_BUILD AND FILE_PREFIX_MAP_SUPPORTED)
  target_compile_options(tvm_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FILE_PREFIX_MAP_FLAG}>)
  target_compile_options(tvm_runtime_objs PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FILE_PREFIX_MAP_FLAG}>)
endif()

tvm_ffi_add_apple_dsymutil(tvm_runtime)
tvm_configure_target_library(tvm_runtime)
add_custom_target(runtime DEPENDS tvm_runtime tvm_runtime_static)
install(TARGETS tvm_runtime DESTINATION lib${LIB_SUFFIX})
# Keep wheel contents unchanged; the archive is available to native consumers.
if(NOT TVM_BUILD_PYTHON_MODULE)
  install(TARGETS tvm_runtime_static DESTINATION lib${LIB_SUFFIX})
endif()
add_library(tvm_objs OBJECT ${COMPILER_SRCS})
target_link_libraries(tvm_objs PUBLIC tvm_ffi_header)
if(TARGET tvm_llvm_header)
  target_link_libraries(tvm_objs PUBLIC tvm_llvm_header)
endif()
if(TVM_VISIBILITY_FLAG)
  target_compile_options(tvm_objs PRIVATE "${TVM_VISIBILITY_FLAG}")
endif()
if(TVM_DEBUG_WITH_ABI_CHANGE)
  target_compile_definitions(tvm_objs PRIVATE TVM_DEBUG_WITH_ABI_CHANGE)
endif()
if(MSVC)
  # Compiler objects import runtime symbols from the separate runtime DLL.
  target_compile_definitions(tvm_objs PRIVATE TVM_EXPORTS TVM_FFI_EXPORTS)
endif()

# Compiler library: built from compiler-only sources.
# Links against tvm_runtime (and tvm_ffi_shared transitively). Loaded with
# RTLD_LOCAL so compiler-internal symbols don't leak into the global namespace.
add_library(tvm_compiler SHARED
  $<TARGET_OBJECTS:tvm_objs>
)
target_link_libraries(tvm_compiler PUBLIC tvm_runtime tvm_ffi_shared)
target_include_directories(tvm_compiler PUBLIC "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
set_property(TARGET tvm_compiler APPEND PROPERTY LINK_OPTIONS "${TVM_NO_UNDEFINED_SYMBOLS}")
set_property(TARGET tvm_compiler APPEND PROPERTY LINK_OPTIONS "${TVM_VISIBILITY_FLAG}")

# Work around a GNU ld (binutils) relaxation bug that miscompiles
# R_X86_64_GOTPCRELX relocations inside very large statically-linked archives.
# When the full LLVM static libraries are linked into libtvm_compiler.so, the
# library is large enough that ld can relax an indirect GOT call (LLVM built
# with -fno-plt emits these) into a direct call with an incorrect displacement.
# The call then targets read-only data instead of the intended function and
# crashes at runtime with a SIGSEGV inside llvm::X86Subtarget during code
# generation. Disabling linker relaxation keeps the GOT-indirect sequences and
# avoids the miscompilation; it is harmless when LLVM is linked dynamically.
# See binutils bug ld/25754.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT "${USE_LLVM}" MATCHES ${IS_FALSE_PATTERN})
  set_property(TARGET tvm_compiler APPEND PROPERTY LINK_OPTIONS "-Wl,--no-relax")
  # LLVM's --system-libs reports -lxml2, but TVM calls no libxml2 symbols. The
  # manylinux toolchain does not default to --as-needed, so set it explicitly:
  # the unused libxml2 is then not recorded as a NEEDED dependency, so auditwheel
  # does not vendor it (and its deps) into the wheel.
  set_property(TARGET tvm_compiler APPEND PROPERTY LINK_OPTIONS "-Wl,--as-needed")
endif()

set_target_properties(tvm_compiler PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
if(MSVC)
  foreach(config_type Release RELEASE Debug DEBUG RelWithDebInfo RELWITHDEBINFO MinSizeRel MINSIZEREL)
    set_target_properties(tvm_compiler PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
      LIBRARY_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
    )
  endforeach()
endif()

if(USE_THREADS AND NOT BUILD_FOR_HEXAGON)
  target_link_libraries(tvm_compiler PUBLIC Threads::Threads)
endif()
target_link_libraries(tvm_compiler PRIVATE ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS})

if(TVM_IS_DEBUG_BUILD)
  target_compile_options(tvm_compiler PRIVATE "${FILE_PREFIX_MAP_FLAG}")
  if(FILE_PREFIX_MAP_SUPPORTED)
    target_compile_options(tvm_compiler PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FILE_PREFIX_MAP_FLAG}>)
    target_compile_options(tvm_objs PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FILE_PREFIX_MAP_FLAG}>)
  endif()
endif()

tvm_ffi_add_apple_dsymutil(tvm_compiler)
tvm_configure_target_library(tvm_compiler)
install(TARGETS tvm_compiler DESTINATION lib${LIB_SUFFIX})

# logging option for libbacktrace
include(cmake/modules/Logging.cmake)

if(USE_CPP_RPC)
  add_subdirectory("apps/cpp_rpc")
endif()

if(USE_CPP_RTVM)
  add_subdirectory("apps/cpp_rtvm")
endif()

if(USE_IOS_RPC)
  add_subdirectory("apps/ios_rpc")
endif()

# Set flags for clang
include(cmake/modules/ClangFlags.cmake)
# Installation rules
if (INSTALL_DEV)
  install(
    DIRECTORY "include/" DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
  )
else(INSTALL_DEV)
  install(
    DIRECTORY "include/tvm/runtime" "include/tvm/support" DESTINATION "include/tvm"
    FILES_MATCHING
    PATTERN "*.h"
    )
endif(INSTALL_DEV)

include(CMakePackageConfigHelpers)
set(PROJECT_CONFIG_CONTENT "@PACKAGE_INIT@\n")
string(APPEND PROJECT_CONFIG_CONTENT "include(CMakeFindDependencyMacro)\n")
string(APPEND PROJECT_CONFIG_CONTENT "find_dependency(Threads REQUIRED)\n")
string(APPEND PROJECT_CONFIG_CONTENT
       "include(\"\${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}Targets.cmake\")")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/temp_config_file.cmake" ${PROJECT_CONFIG_CONTENT})

# install(EXPORT ${PROJECT_NAME}Targets
#   NAMESPACE ${PROJECT_NAME}::
#   DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

# Create config for find_package()
configure_package_config_file(
  "${CMAKE_CURRENT_BINARY_DIR}/temp_config_file.cmake" ${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

install(
  FILES
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

if(BUILD_FOR_HEXAGON)
  # Wrap pthread_create to allow setting custom stack size.
  set_property(TARGET tvm_runtime APPEND PROPERTY LINK_FLAGS
                        "-Wl,--wrap=pthread_create")
  # Link tvm_runtime into the RPC skel library. Make sure it's built
  # as a part of the "runtime" target.
  if(USE_HEXAGON_RPC)
    target_link_libraries(
      hexagon_rpc_skel -Wl,--whole-archive tvm_ffi_static tvm_runtime_static -Wl,--no-whole-archive)
    target_link_options(hexagon_rpc_skel PRIVATE "-Wl,--wrap=pthread_create")
    add_dependencies(runtime hexagon_rpc_skel)
  endif()
endif()

find_and_set_linker(${USE_ALTERNATIVE_LINKER})

if(${SUMMARIZE})
  print_summary()
endif()

dump_options_to_file("${TVM_ALL_OPTIONS}")

if(USE_CUDA AND USE_CUTLASS)
  install(TARGETS fpA_intB_gemm EXPORT ${PROJECT_NAME}Targets DESTINATION lib${LIB_SUFFIX})
  install(TARGETS flash_attn EXPORT ${PROJECT_NAME}Targets DESTINATION lib${LIB_SUFFIX})
  # fpA_intB_gemm, fpA_intB_gemm_tvm, and flash_attn are linked by
  # tvm_runtime_extra (see RuntimeExtra.cmake); no link needed here.
endif()

if(USE_CUDA AND USE_NVTX)
  add_compile_definitions(TVM_NVTX_ENABLED=1)
endif()

# Tests inherit the final project linker flags and compile definitions.
add_subdirectory(tests/cpp)
if(GTEST_FOUND)
  # CTest must also be enabled at the project root for subdirectory discovery.
  enable_testing()
  include(CTest)
endif()

# Python package installation configuration
# This section ensures that all necessary files are installed for the Python wheel
if(TVM_BUILD_PYTHON_MODULE)
  message(STATUS "Configuring Python package installation")

  # Install compiled shared libraries into <project>/lib so that
  # tvm_ffi.libinfo.load_lib_ctypes(package="tvm", target_name=...) can find
  # them via the package RECORD or the project's lib/ fallback dir.
  install(TARGETS tvm_compiler DESTINATION "lib")
  install(TARGETS tvm_runtime DESTINATION "lib")

  # Install third-party compiled dependencies into the same lib/ dir.
  if(TARGET fpA_intB_gemm)
    tvm_configure_target_library(fpA_intB_gemm)
    install(TARGETS fpA_intB_gemm DESTINATION "lib")
  endif()
  if(TARGET flash_attn)
    tvm_configure_target_library(flash_attn)
    install(TARGETS flash_attn DESTINATION "lib")
  endif()

  # Install prebuilt extra runtime libraries into the same lib/ dir. This is how
  # the separately-built CUDA runtime (libtvm_runtime_cuda.so) is bundled: the
  # publishing flow builds it in a CUDA-enabled environment and passes its path
  # via TVM_PACKAGE_EXTRA_LIBS, so it ships through the normal CMake install
  # rather than a post-build wheel rewrite.
  foreach(_extra_lib IN LISTS TVM_PACKAGE_EXTRA_LIBS)
    if(NOT EXISTS "${_extra_lib}")
      message(FATAL_ERROR "TVM_PACKAGE_EXTRA_LIBS entry does not exist: ${_extra_lib}")
    endif()
    install(FILES "${_extra_lib}" DESTINATION "lib")
  endforeach()

  # Install minimal header files needed by Python extensions
  install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/tvm/runtime/"
    DESTINATION "include/tvm/runtime/"
    FILES_MATCHING
    PATTERN "*.h"
  )

  # Install minimal CMake configuration
  install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/"
    DESTINATION "cmake/utils/"
    FILES_MATCHING
    PATTERN "*.cmake"
  )

  # Install CUTLASS headers only if available
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass/include")
    install(
      DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass/include/"
      DESTINATION "3rdparty/cutlass/include/"
      FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.hpp"
    )
  endif()

  # Install minimal source files
  install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/"
    DESTINATION "src/runtime/"
    FILES_MATCHING
    PATTERN "*.cc"
    PATTERN "*.h"
  )

  # Install web package
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/web/" DESTINATION "web/")

  # Install licenses (required for distribution)
  install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/licenses/"
    DESTINATION "licenses/"
  )

  # Install essential metadata files
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
    "${CMAKE_CURRENT_SOURCE_DIR}/NOTICE"
    DESTINATION "."
  )

  message(STATUS "Python package installation configured")
endif()
