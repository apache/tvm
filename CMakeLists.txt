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
include(cmake/modules/CompilerSources.cmake)
include(cmake/modules/RuntimeSources.cmake)

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

include(cmake/modules/Runtime.cmake)
include(cmake/modules/Compiler.cmake)

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
