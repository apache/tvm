cmake_minimum_required(VERSION 3.18)
project(tvm C CXX)

# Utility functions
include(cmake/utils/Utils.cmake)
include(cmake/utils/Summary.cmake)
include(cmake/utils/Linker.cmake)
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
tvm_option(USE_MSCCL "Build with MSCCL" OFF)
tvm_option(USE_OPENCL "Build with OpenCL" OFF)
tvm_option(USE_OPENCL_ENABLE_HOST_PTR "Enable OpenCL memory object access to host" OFF)
tvm_option(USE_OPENCL_GTEST "Path to OpenCL specific gtest version for runtime cpp tests." /path/to/opencl/gtest)
tvm_option(USE_VULKAN "Build with Vulkan" OFF)


# Whether to use spirv-tools.and SPIRV-Headers from Khronos github or gitlab.
#
# Possible values:
# - OFF: not to use
# - /path/to/install: path to your khronis spirv-tools and SPIRV-Headers installation directory
#
tvm_option(USE_KHRONOS_SPIRV "Whether to use spirv-tools.and SPIRV-Headers from Khronos github or gitlab" OFF)
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
tvm_option(USE_MLIR "Build with MLIR support" OFF)
tvm_option(USE_STACKVM_RUNTIME "Include stackvm into the runtime" OFF)
tvm_option(USE_GRAPH_EXECUTOR "Build with tiny graph executor" ON)
tvm_option(USE_GRAPH_EXECUTOR_CUDA_GRAPH "Build with tiny graph executor with CUDA Graph for GPUs" OFF)
tvm_option(USE_AOT_EXECUTOR "Build with AOT executor" ON)
tvm_option(USE_PROFILER "Build profiler for the VM and graph executor" ON)
tvm_option(USE_OPENMP "Build with OpenMP thread pool implementation" OFF)
tvm_option(USE_RELAY_DEBUG "Building Relay in debug mode..." OFF)
tvm_option(TVM_DEBUG_WITH_ABI_CHANGE "Enable debug code that may cause ABI changes" OFF)
tvm_option(TVM_LOG_BEFORE_THROW "Whether log before throw, for debugging purposes" OFF)
tvm_option(USE_RTTI "Build with RTTI" ON)
tvm_option(USE_MSVC_MT "Build with MT" OFF)
tvm_option(INSTALL_DEV "Install compiler infrastructure" OFF)
tvm_option(HIDE_PRIVATE_SYMBOLS "Compile with -fvisibility=hidden." OFF)
tvm_option(USE_TF_TVMDSOOP "Build with TensorFlow TVMDSOOp" OFF)
tvm_option(USE_PT_TVMDSOOP "Build with PyTorch TVMDSOOp" OFF)
tvm_option(USE_FALLBACK_STL_MAP "Use TVM's POD compatible Map" OFF)
tvm_option(INDEX_DEFAULT_I64 "Defaults the index datatype to int64" ON)
tvm_option(USE_LIBBACKTRACE "Use libbacktrace to supply linenumbers on stack traces" AUTO)
tvm_option(BACKTRACE_ON_SEGFAULT "Install a signal handler to print a backtrace on segfault" OFF)
tvm_option(BUILD_STATIC_RUNTIME "Build static version of libtvm_runtime" OFF)
tvm_option(BUILD_DUMMY_LIBTVM "Build a dummy version of libtvm" OFF)
tvm_option(USE_PAPI "Use Performance Application Programming Interface (PAPI) to read performance counters" OFF)
tvm_option(USE_GTEST "Use GoogleTest for C++ sanity tests" AUTO)
tvm_option(USE_CUSTOM_LOGGING "Use user-defined custom logging, tvm::runtime::detail::LogFatalImpl and tvm::runtime::detail::LogMessageImpl must be implemented" OFF)
tvm_option(USE_ALTERNATIVE_LINKER "Use 'mold' or 'lld' if found when invoking compiler to link artifact" AUTO)
tvm_option(USE_CCACHE "Use ccache if found when invoking compiler" AUTO)

# 3rdparty libraries
tvm_option(DLPACK_PATH "Path to DLPACK" "3rdparty/dlpack/include")
tvm_option(DMLC_PATH "Path to DMLC" "3rdparty/dmlc-core/include")
tvm_option(RANG_PATH "Path to RANG" "3rdparty/rang/include")
tvm_option(COMPILER_RT_PATH "Path to COMPILER-RT" "3rdparty/compiler-rt")
tvm_option(PICOJSON_PATH "Path to PicoJSON" "3rdparty/picojson")

# Contrib library options
tvm_option(USE_BYODT_POSIT "Build with BYODT software emulated posit custom datatype" OFF)
tvm_option(USE_BLAS "The blas library to be linked" none)
tvm_option(USE_AMX "Enable Intel AMX" OFF)
tvm_option(USE_MKL "MKL root path when use MKL blas" OFF)
tvm_option(USE_DNNL "Enable DNNL codegen" OFF)
tvm_option(USE_CUDNN "Build with cuDNN" OFF)
tvm_option(USE_CUBLAS "Build with cuBLAS" OFF)
tvm_option(USE_NVTX "Build with NVTX" OFF)
tvm_option(USE_CUTLASS "Build with CUTLASS" OFF)
tvm_option(USE_THRUST "Build with Thrust" OFF)
tvm_option(USE_CURAND "Build with cuRAND" OFF)
tvm_option(USE_MIOPEN "Build with ROCM:MIOpen" OFF)
tvm_option(USE_ROCBLAS "Build with ROCM:RoCBLAS" OFF)
tvm_option(USE_HIPBLAS "Build with ROCM:HIPBLAS" OFF)
tvm_option(USE_SORT "Build with sort support" ON)
tvm_option(USE_NNPACK "Build with nnpack support" OFF)
tvm_option(USE_LIBTORCH "Build with libtorch support" OFF)
tvm_option(USE_RANDOM "Build with random support" ON)
tvm_option(USE_CPP_RPC "Build CPP RPC" OFF)
tvm_option(USE_IOS_RPC "Build iOS RPC" OFF)
tvm_option(USE_TFLITE "Build with tflite support" OFF)
tvm_option(USE_TENSORFLOW_PATH "TensorFlow root path when use TFLite" none)
tvm_option(USE_COREML "Build with coreml support" OFF)
tvm_option(USE_BNNS "Build with BNNS support" OFF)
tvm_option(USE_TARGET_ONNX "Build with ONNX Codegen support" OFF)
tvm_option(USE_ARM_COMPUTE_LIB "Build with Arm Compute Library" OFF)
tvm_option(USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR "Build with Arm Compute Library graph executor" OFF)
tvm_option(USE_TENSORRT_CODEGEN "Build with TensorRT Codegen support" OFF)
tvm_option(USE_TENSORRT_RUNTIME "Build with TensorRT runtime" OFF)
tvm_option(USE_NNAPI_CODEGEN "Build with NNAPI Codegen support" OFF)
tvm_option(USE_NNAPI_RUNTIME "Build with NNAPI runtime" OFF)
tvm_option(USE_RUST_EXT "Build with Rust based compiler extensions, STATIC, DYNAMIC, or OFF" OFF)
tvm_option(USE_VITIS_AI "Build with VITIS-AI Codegen support" OFF)
tvm_option(SUMMARIZE "Print CMake option summary after configuring" OFF)
tvm_option(USE_CLML "Build with CLML Codegen support" OFF)
tvm_option(USE_CLML_GRAPH_EXECUTOR "Build with CLML graph runtime" OFF)
tvm_option(USE_UMA "Build with UMA support" OFF)
tvm_option(USE_VERILATOR "Build with Verilator support" OFF)
tvm_option(USE_MSC "Enable Multi-System Compiler" OFF)
tvm_option(USE_MRVL "Build with MRVL TVM support" OFF)
tvm_option(USE_NVSHMEM "Build with NVSHMEM support" OFF)

# include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories("include")
include_directories(SYSTEM ${DLPACK_PATH})
include_directories(SYSTEM ${DMLC_PATH})
include_directories(SYSTEM ${RANG_PATH})
include_directories(SYSTEM ${COMPILER_RT_PATH})
include_directories(SYSTEM ${PICOJSON_PATH})

# initial variables
set(TVM_LINKER_LIBS "")
set(TVM_RUNTIME_LINKER_LIBS "")


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
  add_definitions(-DDMLC_LOG_BEFORE_THROW=1)
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
    # Static linking. cmake behavior changed in 3.15 making this necessary.
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
tvm_file_glob(GLOB_RECURSE COMPILER_SRCS
    src/auto_scheduler/*.cc
    src/meta_schedule/*.cc
    src/node/*.cc
    src/ir/*.cc
    src/arith/*.cc
    src/te/*.cc
    src/autotvm/*.cc
    src/tir/*.cc
    src/topi/*.cc
    src/driver/*.cc
    src/support/*.cc
    src/script/*.cc
    src/relax/ir/*.cc
    src/relax/op/*.cc
    src/relax/analysis/*.cc
    src/relax/transform/*.cc
    src/relax/backend/vm/*.cc
    src/relax/backend/task_extraction.cc
    src/relax/backend/pattern_registry.cc
    src/relax/utils.cc
    src/relax/distributed/*.cc
    src/relax/distributed/transform/*.cc
    src/relax/op/distributed/*.cc
    src/relax/testing/*.cc
    )

tvm_file_glob(GLOB CODEGEN_SRCS
  src/target/*.cc
  src/target/source/*.cc
  src/target/parsers/*.cc
    )

list(APPEND COMPILER_SRCS ${CODEGEN_SRCS})

tvm_file_glob(GLOB_RECURSE RELAY_OP_SRCS
    src/relay/op/*.cc
    )
tvm_file_glob(GLOB_RECURSE RELAY_PASS_SRCS
    src/relay/analysis/*.cc
    src/relay/collage/*.cc
    src/relay/transforms/*.cc
    src/relay/quantize/*.cc
    )
tvm_file_glob(GLOB RELAY_BACKEND_SRCS
    src/relay/backend/*.cc
    src/relay/backend/vm/*.cc
    src/relay/backend/aot/*.cc
    )
tvm_file_glob(GLOB_RECURSE RELAY_IR_SRCS
    src/relay/ir/*.cc
    src/relay/printer/*.cc
    src/relay/parser/*.cc
    )
tvm_file_glob(GLOB_RECURSE RELAY_QNN_SRCS
    src/relay/qnn/*.cc
)
list(APPEND COMPILER_SRCS ${RELAY_OP_SRCS})
list(APPEND COMPILER_SRCS ${RELAY_PASS_SRCS})
list(APPEND COMPILER_SRCS ${RELAY_BACKEND_SRCS})
list(APPEND COMPILER_SRCS ${RELAY_IR_SRCS})
list(APPEND COMPILER_SRCS ${RELAY_QNN_SRCS})

tvm_file_glob(GLOB DATATYPE_SRCS src/target/datatype/*.cc)
list(APPEND COMPILER_SRCS ${DATATYPE_SRCS})
list(APPEND COMPILER_SRCS "src/target/datatype/myfloat/myfloat.cc")

tvm_file_glob(GLOB RUNTIME_SRCS
  src/runtime/*.cc
  src/runtime/vm/*.cc
  src/runtime/memory/*.cc
  src/runtime/disco/*.cc
  src/runtime/minrpc/*.cc
  src/runtime/relax_vm/*.cc
)
set(TVM_RUNTIME_EXT_OBJS "")

if(BUILD_FOR_HEXAGON)
  if(NOT BUILD_STATIC_RUNTIME)
    # Allow undefined symbols (there will be some from libc).
    set(TVM_NO_UNDEFINED_SYMBOLS "")
  endif()

  add_definitions(-D_MACH_I32=int)
  add_definitions(-DDMLC_CXX11_THREAD_LOCAL=0)
endif()

# distributed disco runtime are disabled for hexagon
if (NOT BUILD_FOR_HEXAGON)
  tvm_file_glob(GLOB RUNTIME_DISCO_DISTRIBUTED_SRCS src/runtime/disco/distributed/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_DISCO_DISTRIBUTED_SRCS})
endif()

# Package runtime rules
if(NOT USE_RTTI)
  add_definitions(-DDMLC_ENABLE_RTTI=0)
endif()

if (INDEX_DEFAULT_I64)
  add_definitions(-DTVM_INDEX_DEFAULT_I64=1)
endif()

if(USE_RPC)
  message(STATUS "Build with RPC support...")
  tvm_file_glob(GLOB RUNTIME_RPC_SRCS src/runtime/rpc/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_RPC_SRCS})
endif(USE_RPC)

tvm_file_glob(GLOB STACKVM_RUNTIME_SRCS src/runtime/stackvm/*.cc)
tvm_file_glob(GLOB STACKVM_CODEGEN_SRCS src/target/stackvm/*.cc)
list(APPEND COMPILER_SRCS ${STACKVM_CODEGEN_SRCS})
if(USE_STACKVM_RUNTIME)
  message(STATUS "Build with stackvm support in runtime...")
  list(APPEND RUNTIME_SRCS ${STACKVM_RUNTIME_SRCS})
else()
  list(APPEND COMPILER_SRCS ${STACKVM_RUNTIME_SRCS})
endif(USE_STACKVM_RUNTIME)

# NOTE(areusch): USE_GRAPH_RUNTIME will be deleted in a future release
if(USE_GRAPH_RUNTIME AND NOT DEFINED USE_GRAPH_EXECUTOR)
  message(WARNING "USE_GRAPH_RUNTIME renamed to USE_GRAPH_EXECUTOR. Please update your config.cmake")
  set(USE_GRAPH_EXECUTOR ${USE_GRAPH_RUNTIME})
  unset(USE_GRAPH_RUNTIME CACHE)
endif(USE_GRAPH_RUNTIME AND NOT DEFINED USE_GRAPH_EXECUTOR)

# NOTE(areusch): USE_GRAPH_RUNTIME_DEBUG will be deleted in a future release
if(USE_GRAPH_RUNTIME_DEBUG AND NOT DEFINED USE_PROFILER)
  message(WARNING "USE_GRAPH_RUNTIME_DEBUG renamed to USE_PROFILER. Please update your config.cmake")
  set(USE_PROFILER ${USE_GRAPH_RUNTIME_DEBUG})
  unset(USE_GRAPH_RUNTIME_DEBUG CACHE)
endif(USE_GRAPH_RUNTIME_DEBUG AND NOT DEFINED USE_PROFILER)

if(USE_GRAPH_EXECUTOR)
  message(STATUS "Build with Graph Executor support...")
  tvm_file_glob(GLOB RUNTIME_GRAPH_EXECUTOR_SRCS src/runtime/graph_executor/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_GRAPH_EXECUTOR_SRCS})

endif(USE_GRAPH_EXECUTOR)

# convert old options for profiler
if(USE_GRAPH_EXECUTOR_DEBUG)
  message(WARNING "USE_GRAPH_EXECUTOR_DEBUG renamed to USE_PROFILER. Please update your config.cmake")
  unset(USE_GRAPH_EXECUTOR_DEBUG CACHE)
  set(USE_PROFILER ON)
endif()
if(USE_VM_PROFILER)
  message(WARNING "USE_VM_PROFILER renamed to USE_PROFILER. Please update your config.cmake")
  unset(USE_VM_PROFILER CACHE)
  set(USE_PROFILER ON)
endif()

if(USE_PROFILER)
  message(STATUS "Build with profiler...")

  tvm_file_glob(GLOB RUNTIME_GRAPH_EXECUTOR_DEBUG_SRCS src/runtime/graph_executor/debug/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_GRAPH_EXECUTOR_DEBUG_SRCS})
  set_source_files_properties(${RUNTIME_GRAPH_EXECUTOR_SRCS}
    PROPERTIES COMPILE_DEFINITIONS "TVM_GRAPH_EXECUTOR_DEBUG")

  tvm_file_glob(GLOB RUNTIME_VM_PROFILER_SRCS src/runtime/vm/profiler/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_VM_PROFILER_SRCS})
endif(USE_PROFILER)

if(USE_CUDA AND USE_NCCL)
  message(STATUS "Build with NCCL...")
  find_nccl(${USE_NCCL})
  include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
  tvm_file_glob(GLOB RUNTIME_NCCL_SRC src/runtime/disco/nccl/*.cc src/runtime/disco/cuda_ipc/*.cc 3rdparty/tensorrt_llm/*.cu)
  set_source_files_properties(src/runtime/disco/nccl/nccl.cc PROPERTIES COMPILE_DEFINITIONS "TVM_NCCL_RCCL_SWITCH=0")
  list(APPEND RUNTIME_SRCS ${RUNTIME_NCCL_SRC})
endif()

if (USE_CUDA AND USE_NVSHMEM)
  message(STATUS "Build with NVSHMEM...")
  find_nvshmem(${USE_NVSHMEM})
  if (NOT NVSHMEM_FOUND)
    message(FATAL_ERROR "Cannot find NVSHMEM, USE_NVSHMEM=" ${USE_NVSHMEM})
  endif()
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  tvm_file_glob(GLOB RUNTIME_NVSHMEM_SRCS src/runtime/contrib/nvshmem/*.cc src/runtime/contrib/nvshmem/*.cu)
  list(APPEND RUNTIME_SRCS ${RUNTIME_NVSHMEM_SRCS})
endif()

if(USE_ROCM AND USE_RCCL)
  message(STATUS "Build with RCCL...")
  find_rccl(${USE_RCCL})
  include_directories(SYSTEM ${RCCL_INCLUDE_DIR})
  tvm_file_glob(GLOB RUNTIME_RCCL_SRC src/runtime/disco/nccl/*.cc)
  set_source_files_properties(src/runtime/disco/nccl/nccl.cc PROPERTIES COMPILE_DEFINITIONS "TVM_NCCL_RCCL_SWITCH=1")
  list(APPEND RUNTIME_SRCS ${RUNTIME_RCCL_SRC})
endif()

if(USE_AOT_EXECUTOR)
  message(STATUS "Build with AOT Executor support...")
  file(GLOB RUNTIME_AOT_EXECUTOR_SRCS src/runtime/aot_executor/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_AOT_EXECUTOR_SRCS})

endif(USE_AOT_EXECUTOR)

# Enable ctest if gtest is available
if(USE_GTEST)
  # Check env var for backward compatibility. A better way to specify package
  # locations is to use CMAKE_PREFIX_PATH or other standard cmake mechanism
  # (see cmake documentation for `find_package`).
  set(GTEST_ROOT "$ENV{GTEST_LIB}")
  if("${USE_GTEST}" STREQUAL "AUTO")
    # If USE_GTEST is AUTO, treat GTest as optional: enable if found.
    find_package(GTest)
  elseif("${USE_GTEST}" MATCHES ${IS_TRUE_PATTERN})
    # USE_GTEST is set to ON, TRUE, etc. Treat GTest as a required package.
    find_package(GTest REQUIRED)
  endif()
  if(GTEST_FOUND)
    if(NOT TARGET GTest::gmock)
      # GMock is formally supported in CMake 3.20; for now, expect libgmock.a in the same directory,
      # and require that folks compiling against GTest::gmock also link against GTest::GTest
      # (for the includes dir).
      add_library(GTest::gmock STATIC IMPORTED GLOBAL)
      get_target_property(GTEST_LIB_PATH GTest::GTest IMPORTED_LOCATION)
      if("${GTEST_LIB_PATH}" STREQUAL "GTEST_LIB_PATH-NOTFOUND")
        # CMake >= 3.20 makes GTest::GTest into a compatibility target. The real import location is in
        # GTest::gtest.
        get_target_property(GTEST_LIB_PATH GTest::gtest IMPORTED_LOCATION)
        if("${GTEST_LIB_PATH}" STREQUAL "GTEST_LIB_PATH-NOTFOUND")
          message(FATAL_ERROR "Neither GTest::GTest nor GTest::gtest targets defined IMPORTED_LOCATION")
        endif()
      endif()
      get_filename_component(GTEST_LIB_DIR "${GTEST_LIB_PATH}" DIRECTORY)
      set_target_properties(GTest::gmock PROPERTIES
          IMPORTED_LOCATION "${GTEST_LIB_DIR}/libgmock.a")
    endif()

    enable_testing()
    include(CTest)
  endif()
endif()

if(USE_PIPELINE_EXECUTOR)
  message(STATUS "Build with Pipeline Executor support...")
  tvm_file_glob(GLOB RUNTIME_PIPELINE_SRCS src/runtime/pipeline/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_PIPELINE_SRCS})
endif(USE_PIPELINE_EXECUTOR)

if(USE_KALLOC_ALIGNMENT)
  message(STATUS "Build Alloc alignment set to ${USE_KALLOC_ALIGNMENT}")
  add_definitions(-DTVM_KALLOC_ALIGNMENT=${USE_KALLOC_ALIGNMENT})
endif(USE_KALLOC_ALIGNMENT)

# Caches the build.
# Note that ccache-3.x doesn't support nvcc well, so CUDA kernels may never hit the cache and still
# need to be re-compiled every time. Using ccache 4.0+ can resolve this issue.
include(cmake/utils/CCache.cmake)

include(CheckCXXCompilerFlag)
if(NOT MSVC)
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
else()
  check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "/std:c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
endif()

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
include(cmake/modules/contrib/CODEGENC.cmake)
include(cmake/modules/contrib/DNNL.cmake)
include(cmake/modules/contrib/AMX.cmake)
include(cmake/modules/contrib/CUTLASS.cmake)
include(cmake/modules/contrib/ExampleTargetHooks.cmake)
include(cmake/modules/contrib/Random.cmake)
include(cmake/modules/contrib/Posit.cmake)
include(cmake/modules/contrib/MSCCLPP.cmake)
include(cmake/modules/contrib/Sort.cmake)
include(cmake/modules/contrib/NNPack.cmake)
include(cmake/modules/contrib/LibTorch.cmake)
include(cmake/modules/contrib/HybridDump.cmake)
include(cmake/modules/contrib/TFLite.cmake)
include(cmake/modules/contrib/TF_TVMDSOOP.cmake)
include(cmake/modules/contrib/PT_TVMDSOOP.cmake)
include(cmake/modules/contrib/CoreML.cmake)
include(cmake/modules/contrib/BNNS.cmake)
include(cmake/modules/contrib/ONNX.cmake)
include(cmake/modules/contrib/ArmComputeLib.cmake)
include(cmake/modules/contrib/TensorRT.cmake)
include(cmake/modules/contrib/NNAPI.cmake)
include(cmake/modules/contrib/VitisAI.cmake)
include(cmake/modules/contrib/Verilator.cmake)
include(cmake/modules/contrib/UMA.cmake)
include(cmake/modules/contrib/MSC.cmake)
include(cmake/modules/contrib/vllm.cmake)
include(cmake/modules/Git.cmake)
include(cmake/modules/LibInfo.cmake)
include(cmake/modules/RustExt.cmake)
include(cmake/modules/contrib/Mrvl.cmake)

set(LIBINFO_FILE ${CMAKE_CURRENT_LIST_DIR}/src/support/libinfo.cc)
add_lib_info(${LIBINFO_FILE})
list(REMOVE_ITEM COMPILER_SRCS ${LIBINFO_FILE})

add_library(tvm_objs OBJECT ${COMPILER_SRCS})
add_library(tvm_runtime_objs OBJECT ${RUNTIME_SRCS})
add_library(tvm_libinfo_objs OBJECT ${LIBINFO_FILE})

include(GNUInstallDirs)
if(NOT BUILD_DUMMY_LIBTVM)
  add_library(tvm SHARED
    $<TARGET_OBJECTS:tvm_objs>
    $<TARGET_OBJECTS:tvm_runtime_objs>
    $<TARGET_OBJECTS:tvm_libinfo_objs>
    ${TVM_RUNTIME_EXT_OBJS}
  )

else()
  # dummy version of libtvm that can be used by downstream to specify dependencies
  # the real runner still need a full version of libtvm
  add_library(tvm SHARED
    $<TARGET_OBJECTS:tvm_runtime_objs>
    $<TARGET_OBJECTS:tvm_libinfo_objs>
    ${TVM_RUNTIME_EXT_OBJS}
  )
endif()

target_include_directories(tvm PUBLIC "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
set_property(TARGET tvm APPEND PROPERTY LINK_OPTIONS "${TVM_NO_UNDEFINED_SYMBOLS}")
set_property(TARGET tvm APPEND PROPERTY LINK_OPTIONS "${TVM_VISIBILITY_FLAG}")
if(BUILD_STATIC_RUNTIME)
  add_library(tvm_runtime STATIC
    $<TARGET_OBJECTS:tvm_runtime_objs>
    $<TARGET_OBJECTS:tvm_libinfo_objs>
    ${TVM_RUNTIME_EXT_OBJS}
  )
  set(NOTICE_MULTILINE
    "You have build static version of the TVM runtime library. Make "
    "sure to use --whole-archive when linking it into your project.")
  string(CONCAT NOTICE ${NOTICE_MULTILINE})
  add_custom_command(TARGET tvm_runtime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --yellow --bold ${NOTICE})
else()
  add_library(tvm_runtime SHARED
    $<TARGET_OBJECTS:tvm_runtime_objs>
    $<TARGET_OBJECTS:tvm_libinfo_objs>
    ${TVM_RUNTIME_EXT_OBJS}
  )
  set_property(TARGET tvm_runtime APPEND PROPERTY LINK_OPTIONS "${TVM_NO_UNDEFINED_SYMBOLS}")
endif()


target_include_directories(tvm_runtime PUBLIC "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
set_property(TARGET tvm_runtime APPEND PROPERTY LINK_OPTIONS "${TVM_VISIBILITY_FLAG}")

target_compile_definitions(tvm_objs PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
target_compile_definitions(tvm_runtime_objs PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
target_compile_definitions(tvm_libinfo_objs PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
target_compile_definitions(tvm PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
target_compile_definitions(tvm_runtime PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

# logging option for libbacktrace
include(cmake/modules/Logging.cmake)

include(cmake/modules/contrib/PAPI.cmake)

if(USE_CPP_RPC)
  add_subdirectory("apps/cpp_rpc")
endif()

if(USE_CPP_RTVM)
  add_subdirectory("apps/cpp_rtvm")
endif()

if(USE_IOS_RPC)
  add_subdirectory("apps/ios_rpc")
endif()

if(USE_RELAY_DEBUG)
  message(STATUS "Building Relay in debug mode...")
  target_compile_definitions(tvm_objs PRIVATE "USE_RELAY_DEBUG")
  target_compile_definitions(tvm_objs PRIVATE "TVM_LOG_DEBUG")
  target_compile_definitions(tvm_runtime_objs PRIVATE "USE_RELAY_DEBUG")
  target_compile_definitions(tvm_runtime_objs PRIVATE "TVM_LOG_DEBUG")
  target_compile_definitions(tvm_libinfo_objs PRIVATE "USE_RELAY_DEBUG")
  target_compile_definitions(tvm_libinfo_objs PRIVATE "TVM_LOG_DEBUG")
else()
  target_compile_definitions(tvm_objs PRIVATE "NDEBUG")
  target_compile_definitions(tvm_runtime_objs PRIVATE "NDEBUG")
  target_compile_definitions(tvm_libinfo_objs PRIVATE "NDEBUG")
endif(USE_RELAY_DEBUG)

if(TVM_DEBUG_WITH_ABI_CHANGE)
  message(STATUS "Building with debug code that may cause ABI changes...")
  target_compile_definitions(tvm_objs PRIVATE "TVM_DEBUG_WITH_ABI_CHANGE")
  target_compile_definitions(tvm_runtime_objs PRIVATE "TVM_DEBUG_WITH_ABI_CHANGE")
  target_compile_definitions(tvm_libinfo_objs PRIVATE "TVM_DEBUG_WITH_ABI_CHANGE")
endif(TVM_DEBUG_WITH_ABI_CHANGE)

if(USE_FALLBACK_STL_MAP)
  message(STATUS "Building with STL Map...")
  target_compile_definitions(tvm_objs PRIVATE "USE_FALLBACK_STL_MAP=1")
  target_compile_definitions(tvm_runtime_objs PRIVATE "USE_FALLBACK_STL_MAP=1")
  target_compile_definitions(tvm_libinfo_objs PRIVATE "USE_FALLBACK_STL_MAP=1")
else()
  message(STATUS "Building with TVM Map...")
  target_compile_definitions(tvm_objs PRIVATE "USE_FALLBACK_STL_MAP=0")
  target_compile_definitions(tvm_runtime_objs PRIVATE "USE_FALLBACK_STL_MAP=0")
  target_compile_definitions(tvm_libinfo_objs PRIVATE "USE_FALLBACK_STL_MAP=0")
endif(USE_FALLBACK_STL_MAP)

if(USE_THREADS AND NOT BUILD_FOR_HEXAGON)
  message(STATUS "Build with thread support...")
  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)
  target_link_libraries(tvm PUBLIC Threads::Threads)
  target_link_libraries(tvm_runtime PUBLIC Threads::Threads)
endif()

if(NOT BUILD_DUMMY_LIBTVM)
  target_link_libraries(tvm PRIVATE ${TVM_LINKER_LIBS})
endif()

target_link_libraries(tvm PRIVATE ${TVM_RUNTIME_LINKER_LIBS})
target_link_libraries(tvm_runtime PRIVATE ${TVM_RUNTIME_LINKER_LIBS})

if(BUILD_FOR_HEXAGON AND DEFINED USE_HEXAGON_GTEST AND EXISTS ${USE_HEXAGON_GTEST})
  include(FetchContent)
  FetchContent_Declare(googletest SOURCE_DIR "${USE_HEXAGON_GTEST}")
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  target_link_libraries(tvm_runtime PUBLIC gtest)
  include_directories("${USE_HEXAGON_GTEST}/include")
endif()

# Set flags for clang
include(cmake/modules/ClangFlags.cmake)
set(CRC16_INCLUDE_PATH "3rdparty/libcrc/include")
target_include_directorieS(
  tvm_objs
  PRIVATE "${CRC16_INCLUDE_PATH}")
target_include_directorieS(
  tvm_libinfo_objs
  PRIVATE "${CRC16_INCLUDE_PATH}")
target_include_directorieS(
  tvm_runtime_objs
  PRIVATE "${CRC16_INCLUDE_PATH}")

set(TVM_TEST_LIBRARY_NAME tvm)
if (HIDE_PRIVATE_SYMBOLS AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  add_library(tvm_allvisible SHARED $<TARGET_OBJECTS:tvm_objs> $<TARGET_OBJECTS:tvm_runtime_objs> $<TARGET_OBJECTS:tvm_libinfo_objs>)
  target_include_directories(tvm_allvisible PUBLIC "$<TARGET_PROPERTY:tvm,INCLUDE_DIRECTORIES>")
  target_link_libraries(tvm_allvisible PRIVATE "$<TARGET_PROPERTY:tvm,LINK_LIBRARIES>")
  set(TVM_TEST_LIBRARY_NAME tvm_allvisible)

  set(HIDE_SYMBOLS_LINKER_FLAGS "-Wl,--exclude-libs,ALL")
  # Note: 'target_link_options' with 'PRIVATE' keyword would be cleaner
  # but it's not available until CMake 3.13. Switch to 'target_link_options'
  # once minimum CMake version is bumped up to 3.13 or above.
  target_link_libraries(tvm PRIVATE ${HIDE_SYMBOLS_LINKER_FLAGS})
  target_link_libraries(tvm_runtime PRIVATE ${HIDE_SYMBOLS_LINKER_FLAGS})
  target_compile_definitions(tvm_allvisible PUBLIC $<TARGET_PROPERTY:tvm,INTERFACE_COMPILE_DEFINITONS>)
  target_compile_definitions(tvm_allvisible PRIVATE $<TARGET_PROPERTY:tvm,COMPILE_DEFINITONS>)
endif()

# Create the `cpptest` target if we can find GTest.  If not, we create dummy
# targets that give the user an informative error message.
if(GTEST_FOUND)
  tvm_file_glob(GLOB_RECURSE TEST_SRCS tests/cpp/*.cc)
  add_executable(cpptest ${TEST_SRCS})
  # include runtime files for unit testing
  target_link_libraries(cpptest PRIVATE ${TVM_TEST_LIBRARY_NAME} GTest::GTest GTest::Main GTest::gmock pthread dl)
  if(DEFINED LLVM_LIBS)
    # The TVM library is linked with LLVM libraries. If the LLVM libraries are
    # static and the symbols are not hidden, then don't link them again into
    # cpptest since cpptest is itself linked against the TVM library. If static
    # LLVM libraries are linked in twice, it can cause issues with global
    # variable initialization (cl::opt).
    # If the LLVM libraries are dynamic, we have to link them again, since the
    # TVM library will not contain any LLVM definitions.
    unset(LLVM_SO)
    foreach(L IN LISTS LLVM_LIBS)
      if(L MATCHES "libLLVM.*\.so")
        set(LLVM_SO TRUE)
        break()
      endif()
    endforeach()
    if(DEFINED LLVM_SO OR HIDE_PRIVATE_SYMBOLS)
      target_link_libraries(cpptest PRIVATE ${LLVM_LIBS})
    endif()
  endif()
  set_target_properties(cpptest PROPERTIES EXCLUDE_FROM_ALL 1)
  set_target_properties(cpptest PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
  if(USE_RELAY_DEBUG)
    target_compile_definitions(cpptest PRIVATE "USE_RELAY_DEBUG")
    target_compile_definitions(cpptest PRIVATE "TVM_LOG_DEBUG")
  else()
    target_compile_definitions(cpptest PRIVATE "NDEBUG")
  endif()
  if(TVM_DEBUG_WITH_ABI_CHANGE)
    target_compile_definitions(cpptest PRIVATE "TVM_DEBUG_WITH_ABI_CHANGE")
  endif(TVM_DEBUG_WITH_ABI_CHANGE)

  # For some reason, compile definitions are not propagated correctly, so we manually add them here
  target_compile_definitions(cpptest PUBLIC $<TARGET_PROPERTY:tvm,INTERFACE_COMPILE_DEFINITIONS>)
  gtest_discover_tests(cpptest)
endif()

# Custom targets
add_custom_target(runtime DEPENDS tvm_runtime)

# Installation rules
install(TARGETS tvm EXPORT ${PROJECT_NAME}Targets DESTINATION lib${LIB_SUFFIX})
install(TARGETS tvm_runtime EXPORT ${PROJECT_NAME}Targets DESTINATION lib${LIB_SUFFIX})
if(BUILD_FOR_HEXAGON AND DEFINED USE_HEXAGON_GTEST AND EXISTS ${USE_HEXAGON_GTEST})
  install(TARGETS gtest EXPORT ${PROJECT_NAME}Targets DESTINATION lib${LIB_SUFFIX})
endif()

if (INSTALL_DEV)
  install(
    DIRECTORY "include/" DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
  )
  install(
    DIRECTORY "3rdparty/dlpack/include/" DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
    )
  install(
    DIRECTORY "3rdparty/dmlc-core/include/" DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
    )
else(INSTALL_DEV)
  install(
    DIRECTORY "include/tvm/runtime/" DESTINATION "include/tvm/runtime"
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

install(EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

# Create config for find_package()
configure_package_config_file(
  "${CMAKE_CURRENT_BINARY_DIR}/temp_config_file.cmake" ${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

install(
  FILES
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# More target definitions
if(MSVC)
  target_compile_definitions(tvm_objs PRIVATE -DTVM_EXPORTS)
  target_compile_definitions(tvm_libinfo_objs PRIVATE -DTVM_EXPORTS)
  target_compile_definitions(tvm_runtime_objs PRIVATE -DTVM_EXPORTS)
endif()

set(TVM_IS_DEBUG_BUILD OFF)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR CMAKE_CXX_FLAGS MATCHES "-g")
  set(TVM_IS_DEBUG_BUILD ON)
endif()

# Change relative paths in backtrace to absolute ones
if(TVM_IS_DEBUG_BUILD)
  set(FILE_PREFIX_MAP_FLAG "-ffile-prefix-map=..=${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_options(tvm PRIVATE "${FILE_PREFIX_MAP_FLAG}")
  CHECK_CXX_COMPILER_FLAG("${FILE_PREFIX_MAP_FLAG}" FILE_PREFIX_MAP_SUPPORTED)
  if(FILE_PREFIX_MAP_SUPPORTED)
    target_compile_options(tvm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FILE_PREFIX_MAP_FLAG}>)
    target_compile_options(tvm_objs PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FILE_PREFIX_MAP_FLAG}>)
    target_compile_options(tvm_libinfo_objs PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FILE_PREFIX_MAP_FLAG}>)
    target_compile_options(tvm_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FILE_PREFIX_MAP_FLAG}>)
    target_compile_options(tvm_runtime_objs PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FILE_PREFIX_MAP_FLAG}>)
  endif()
endif()

# Run dsymutil to generate debugging symbols for backtraces
if(APPLE AND TVM_IS_DEBUG_BUILD)
  find_program(DSYMUTIL dsymutil)
  mark_as_advanced(DSYMUTIL)
  add_custom_command(TARGET tvm
      POST_BUILD
      COMMAND ${DSYMUTIL} ARGS $<TARGET_FILE:tvm>
      COMMENT "Running dsymutil"
      VERBATIM
		  )
endif()

if(BUILD_FOR_HEXAGON)
  # Wrap pthread_create to allow setting custom stack size.
  set_property(TARGET tvm_runtime APPEND PROPERTY LINK_FLAGS
                        "-Wl,--wrap=pthread_create")
  # Link tvm_runtime into the RPC skel library. Make sure it's built
  # as a part of the "runtime" target.
  if(USE_HEXAGON_RPC)
    target_link_libraries(hexagon_rpc_skel -Wl,--whole-archive tvm_runtime -Wl,--no-whole-archive)
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
  target_link_libraries(tvm PRIVATE fpA_intB_gemm)
  target_link_libraries(tvm_runtime PRIVATE fpA_intB_gemm)
  target_link_libraries(tvm PRIVATE fpA_intB_gemm_tvm)
  target_link_libraries(tvm_runtime PRIVATE fpA_intB_gemm_tvm)

  install(TARGETS flash_attn EXPORT ${PROJECT_NAME}Targets DESTINATION lib${LIB_SUFFIX})
  target_link_libraries(tvm PRIVATE -Wl,--no-as-needed flash_attn)
  target_link_libraries(tvm_runtime PRIVATE -Wl,--no-as-needed flash_attn)
endif()

if(USE_CUDA AND USE_NVTX)
  set_source_files_properties(src/runtime/nvtx.cc PROPERTIES COMPILE_DEFINITIONS "TVM_NVTX_ENABLED=1")
endif()

if(USE_CUDA AND USE_NCCL)
  find_library(LIBRT rt)
  target_link_libraries(tvm PRIVATE nccl ${LIBRT})
  target_link_libraries(tvm_runtime PRIVATE nccl ${LIBRT})
endif()


if (USE_CUDA AND USE_NVSHMEM)
  include_directories(SYSTEM ${USE_NVSHMEM}/include)
  find_library(NVSHMEM_HOST nvshmem_host ${NVSHMEM_LIB_DIR})
  find_library(NVSHMEM_DEVICE nvshmem_device ${NVSHMEM_LIB_DIR})
  target_link_libraries(tvm PRIVATE ${NVSHMEM_HOST} ${NVSHMEM_DEVICE})
  target_link_libraries(tvm_runtime PRIVATE ${NVSHMEM_HOST} ${NVSHMEM_DEVICE})
  set_target_properties(tvm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(tvm_runtime PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(USE_ROCM AND USE_RCCL)
  target_link_libraries(tvm PRIVATE rccl)
  target_link_libraries(tvm_runtime PRIVATE rccl)
endif()


option(USE_FLASHINFER "Build TVM with FlashInfer" OFF)
if (USE_FLASHINFER STREQUAL "ON")
  message(STATUS "Build with FlashInfer")
  set(FLASHINFER_TVM_BINDING ON)
  set(FLASHINFER_TVM_SOURCE_DIR ${PROJECT_SOURCE_DIR})
  set(FLASHINFER_PREFILL OFF)
  set(FLASHINFER_DECODE OFF)
  set(FLASHINFER_PAGE OFF)
  set(FLASHINFER_CASCADE OFF)
  set(FLASHINFER_SAMPLING OFF)
  set(FLASHINFER_NORM OFF)
  add_subdirectory(3rdparty/flashinfer)
else ()
  message(STATUS "Build without FlashInfer")
endif ()


if (USE_FLASHINFER STREQUAL "ON")
  target_link_libraries(tvm PRIVATE flashinfer_tvm)
  target_link_libraries(tvm_runtime PRIVATE flashinfer_tvm)
endif ()
