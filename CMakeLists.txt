cmake_minimum_required(VERSION 3.2)
project(tvm C CXX)

# Utility functions
include(cmake/utils/Utils.cmake)
include(cmake/utils/FindCUDA.cmake)
include(cmake/utils/FindOpenCL.cmake)
include(cmake/utils/FindVulkan.cmake)
include(cmake/utils/FindLLVM.cmake)
include(cmake/utils/FindROCM.cmake)
include(cmake/utils/FindEthosN.cmake)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

# NOTE: do not modify this file to change option values.
# You can create a config.cmake at build folder
# and add set(OPTION VALUE) to override these build options.
# Alernatively, use cmake -DOPTION=VALUE through command-line.
tvm_option(USE_CUDA "Build with CUDA" OFF)
tvm_option(USE_OPENCL "Build with OpenCL" OFF)
tvm_option(USE_VULKAN "Build with Vulkan" OFF)
tvm_option(USE_METAL "Build with Metal" OFF)
tvm_option(USE_ROCM "Build with ROCM" OFF)
tvm_option(ROCM_PATH "The path to rocm" /opt/rocm)
tvm_option(USE_HEXAGON_DEVICE "Build with Hexagon device support in TVM runtime" OFF)
tvm_option(USE_HEXAGON_SDK "Path to the Hexagon SDK root (required for Hexagon support in TVM runtime or for building TVM runtime for Hexagon)" /path/to/sdk)
tvm_option(USE_RPC "Build with RPC" ON)
tvm_option(USE_THREADS "Build with thread support" ON)
tvm_option(USE_LLVM "Build with LLVM, can be set to specific llvm-config path" OFF)
tvm_option(USE_STACKVM_RUNTIME "Include stackvm into the runtime" OFF)
tvm_option(USE_GRAPH_EXECUTOR "Build with tiny graph executor" ON)
tvm_option(USE_GRAPH_EXECUTOR_CUDA_GRAPH "Build with tiny graph executor with CUDA Graph for GPUs" OFF)
tvm_option(USE_PROFILER "Build profiler for the VM and graph executor" ON)
tvm_option(USE_OPENMP "Build with OpenMP thread pool implementation" OFF)
tvm_option(USE_RELAY_DEBUG "Building Relay in debug mode..." OFF)
tvm_option(USE_RTTI "Build with RTTI" ON)
tvm_option(USE_MSVC_MT "Build with MT" OFF)
tvm_option(USE_MICRO "Build with Micro TVM support" OFF)
tvm_option(INSTALL_DEV "Install compiler infrastructure" OFF)
tvm_option(HIDE_PRIVATE_SYMBOLS "Compile with -fvisibility=hidden." OFF)
tvm_option(USE_TF_COMPILE_FLAGS "Build with TensorFlow's compile flags." OFF)
tvm_option(USE_TF_TVMDSOOP "Build with TensorFlow TVMDSOOp" OFF)
tvm_option(USE_FALLBACK_STL_MAP "Use TVM's POD compatible Map" OFF)
tvm_option(USE_ETHOSN "Build with Arm Ethos-N" OFF)
tvm_option(USE_CMSISNN "Build with Arm CMSIS-NN" OFF)
tvm_option(INDEX_DEFAULT_I64 "Defaults the index datatype to int64" ON)
tvm_option(USE_LIBBACKTRACE "Build libbacktrace to supply linenumbers on stack traces" AUTO)
tvm_option(BUILD_STATIC_RUNTIME "Build static version of libtvm_runtime" OFF)
tvm_option(USE_PAPI "Use Performance Application Programming Interface (PAPI) to read performance counters" OFF)

# 3rdparty libraries
tvm_option(DLPACK_PATH "Path to DLPACK" "3rdparty/dlpack/include")
tvm_option(DMLC_PATH "Path to DMLC" "3rdparty/dmlc-core/include")
tvm_option(RANG_PATH "Path to RANG" "3rdparty/rang/include")
tvm_option(COMPILER_RT_PATH "Path to COMPILER-RT" "3rdparty/compiler-rt")
tvm_option(PICOJSON_PATH "Path to PicoJSON" "3rdparty/picojson")

# Contrib library options
tvm_option(USE_BYODT_POSIT "Build with BYODT software emulated posit custom datatype" OFF)
tvm_option(USE_BLAS "The blas library to be linked" none)
tvm_option(USE_MKL "MKL root path when use MKL blas" OFF)
tvm_option(USE_MKLDNN "Build with MKLDNN" OFF)
tvm_option(USE_DNNL_CODEGEN "Enable MKLDNN (DNNL) codegen" OFF)
tvm_option(USE_CUDNN "Build with cuDNN" OFF)
tvm_option(USE_CUBLAS "Build with cuBLAS" OFF)
tvm_option(USE_THRUST "Build with Thrust" OFF)
tvm_option(USE_MIOPEN "Build with ROCM:MIOpen" OFF)
tvm_option(USE_ROCBLAS "Build with ROCM:RoCBLAS" OFF)
tvm_option(USE_SORT "Build with sort support" ON)
tvm_option(USE_NNPACK "Build with nnpack support" OFF)
tvm_option(USE_TENSORRT "Build with TensorRT, must have CUDA and CUDNN enabled" OFF)
tvm_option(USE_RANDOM "Build with random support" ON)
tvm_option(USE_MICRO_STANDALONE_RUNTIME "Build with micro.standalone_runtime support" OFF)
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
tvm_option(USE_RUST_EXT "Build with Rust based compiler extensions, STATIC, DYNAMIC, or OFF" OFF)
tvm_option(USE_VITIS_AI "Build with VITIS-AI Codegen support" OFF)
tvm_option(USE_MLAS "Build with MLAS library" OFF)

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
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")

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

  if(USE_TF_COMPILE_FLAGS)
    execute_process(COMMAND python3 -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
        OUTPUT_VARIABLE TF_COMPILE_FLAGS
        RESULT_VARIABLE TF_COMPILE_FLAGS_STATUS)
    if (NOT ${TF_COMPILE_FLAGS_STATUS} EQUAL 0)
        message(FATAL_ERROR "Fail to get TensorFlow compile flags")
    endif()
    string(REGEX REPLACE "\n" "" TF_COMPILE_FLAGS "${TF_COMPILE_FLAGS}")
    message("Use TensorFlow Compile flags=\"${TF_COMPILE_FLAGS}\"")
    set(CMAKE_CXX_FLAGS "${TF_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${TF_COMPILE_FLAGS} ${CMAKE_CUDA_FLAGS}")
  endif(USE_TF_COMPILE_FLAGS)

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
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
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

if(BUILD_FOR_ANDROID)
  # EmuTLS on Android is in libgcc. Without it linked in, libtvm_runtime.so
  # won't load on Android due to missing __emutls_XXX symbols.
  list(APPEND TVM_RUNTIME_LINKER_LIBS "gcc")
endif()

# add source group
FILE(GLOB_RECURSE GROUP_SOURCE "src/*.cc")
FILE(GLOB_RECURSE GROUP_INCLUDE "src/*.h" "include/*.h")
assign_source_group("Source" ${GROUP_SOURCE})
assign_source_group("Include" ${GROUP_INCLUDE})

# Source file lists
file(GLOB_RECURSE COMPILER_SRCS
    src/auto_scheduler/*.cc
    src/node/*.cc
    src/ir/*.cc
    src/arith/*.cc
    src/te/*.cc
    src/autotvm/*.cc
    src/tir/*.cc
    src/topi/*.cc
    src/driver/*.cc
    src/parser/*.cc
    src/printer/*.cc
    src/support/*.cc
    )

file(GLOB CODEGEN_SRCS
  src/target/*.cc
  src/target/source/*.cc
    )

list(APPEND COMPILER_SRCS ${CODEGEN_SRCS})

file(GLOB_RECURSE RELAY_OP_SRCS
    src/relay/op/*.cc
    )
file(GLOB_RECURSE RELAY_PASS_SRCS
    src/relay/analysis/*.cc
    src/relay/transforms/*.cc
    src/relay/quantize/*.cc
    )
file(GLOB RELAY_BACKEND_SRCS
    src/relay/backend/*.cc
    src/relay/backend/vm/*.cc
    )
file(GLOB_RECURSE RELAY_IR_SRCS
    src/relay/ir/*.cc
    )
file(GLOB_RECURSE RELAY_QNN_SRCS
    src/relay/qnn/*.cc
)
list(APPEND COMPILER_SRCS ${RELAY_OP_SRCS})
list(APPEND COMPILER_SRCS ${RELAY_PASS_SRCS})
list(APPEND COMPILER_SRCS ${RELAY_BACKEND_SRCS})
list(APPEND COMPILER_SRCS ${RELAY_IR_SRCS})
list(APPEND COMPILER_SRCS ${RELAY_QNN_SRCS})

file(GLOB DATATYPE_SRCS src/target/datatype/*.cc)
list(APPEND COMPILER_SRCS ${DATATYPE_SRCS})
list(APPEND COMPILER_SRCS "src/target/datatype/myfloat/myfloat.cc")

file(GLOB RUNTIME_SRCS
  src/runtime/*.cc
  src/runtime/vm/*.cc
)

if(BUILD_FOR_HEXAGON)
  # Add file implementing posix_memalign when building the runtime as
  # a shared library.
  # This function is actually defined in the static libc, but when linking
  # a shared library, libc is not linked into it. Some runtime systems
  # don't implement posix_runtime, which causes runtime failires.
  # To avoid this issue, Hexagon runtime contains an implementation of
  # posix_memalign, but it should only be used with the dynamic TVM
  # runtime, since it would cause multiple definition errors with the
  # static one.
  if(NOT BUILD_STATIC_RUNTIME)
    list(APPEND RUNTIME_SRCS src/runtime/hexagon/hexagon_posix.cc)
  endif()

  add_definitions(-D_MACH_I32=int)
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
  file(GLOB RUNTIME_RPC_SRCS src/runtime/rpc/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_RPC_SRCS})
endif(USE_RPC)

file(GLOB STACKVM_RUNTIME_SRCS src/runtime/stackvm/*.cc)
file(GLOB STACKVM_CODEGEN_SRCS src/target/stackvm/*.cc)
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
  file(GLOB RUNTIME_GRAPH_EXECUTOR_SRCS src/runtime/graph_executor/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_GRAPH_EXECUTOR_SRCS})

endif(USE_GRAPH_EXECUTOR)

# convert old options for profiler
if(USE_GRAPH_EXECUTOR_DEBUG)
  message(WARNING "USE_GRAPH_EXECUTOR renamed to USE_PROFILER. Please update your config.cmake")
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

  file(GLOB RUNTIME_GRAPH_EXECUTOR_DEBUG_SRCS src/runtime/graph_executor/debug/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_GRAPH_EXECUTOR_DEBUG_SRCS})
  set_source_files_properties(${RUNTIME_GRAPH_EXECUTOR_SRCS}
    PROPERTIES COMPILE_DEFINITIONS "TVM_GRAPH_EXECUTOR_DEBUG")

  file(GLOB RUNTIME_VM_PROFILER_SRCS src/runtime/vm/profiler/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_VM_PROFILER_SRCS})
endif(USE_PROFILER)

# Enable ctest if gtest is available
find_path(GTEST_INCLUDE_DIR gtest/gtest.h)
find_library(GTEST_LIB gtest "$ENV{GTEST_LIB}")
if(GTEST_INCLUDE_DIR AND GTEST_LIB)
  enable_testing()
  include(CTest)
  include(GoogleTest)
endif()

# Module rules
include(cmake/modules/VTA.cmake)
include(cmake/modules/StandaloneCrt.cmake)
include(cmake/modules/CUDA.cmake)
include(cmake/modules/Hexagon.cmake)
include(cmake/modules/OpenCL.cmake)
include(cmake/modules/OpenMP.cmake)
include(cmake/modules/Vulkan.cmake)
include(cmake/modules/Metal.cmake)
include(cmake/modules/ROCM.cmake)
include(cmake/modules/LLVM.cmake)
include(cmake/modules/Micro.cmake)
include(cmake/modules/contrib/EthosN.cmake)
include(cmake/modules/contrib/CMSISNN.cmake)
include(cmake/modules/contrib/EthosU.cmake)
include(cmake/modules/contrib/BLAS.cmake)
include(cmake/modules/contrib/CODEGENC.cmake)
include(cmake/modules/contrib/DNNL.cmake)
include(cmake/modules/contrib/Random.cmake)
include(cmake/modules/contrib/Posit.cmake)
include(cmake/modules/contrib/MicroStandaloneRuntime.cmake)
include(cmake/modules/contrib/Sort.cmake)
include(cmake/modules/contrib/NNPack.cmake)
include(cmake/modules/contrib/TensorRT.cmake)
include(cmake/modules/contrib/HybridDump.cmake)
include(cmake/modules/contrib/TFLite.cmake)
include(cmake/modules/contrib/TF_TVMDSOOP.cmake)
include(cmake/modules/contrib/CoreML.cmake)
include(cmake/modules/contrib/BNNS.cmake)
include(cmake/modules/contrib/ONNX.cmake)
include(cmake/modules/contrib/ArmComputeLib.cmake)
include(cmake/modules/contrib/TensorRT.cmake)
include(cmake/modules/contrib/VitisAI.cmake)
include(cmake/modules/contrib/Verilator.cmake)
include(cmake/modules/Git.cmake)
include(cmake/modules/LibInfo.cmake)
include(cmake/modules/RustExt.cmake)
include(cmake/modules/contrib/MLAS.cmake)

include(CheckCXXCompilerFlag)
if(NOT MSVC)
  check_cxx_compiler_flag("-std=c++14" SUPPORT_CXX14)
  set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
else()
  check_cxx_compiler_flag("/std:c++14" SUPPORT_CXX14)
  set(CMAKE_CXX_FLAGS "/std:c++14 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()

set(LIBINFO_FILE ${CMAKE_CURRENT_LIST_DIR}/src/support/libinfo.cc)
add_lib_info(${LIBINFO_FILE})
list(REMOVE_ITEM COMPILER_SRCS ${LIBINFO_FILE})

# TODO(trevmorr): Update tvm_runtime_static to match other targets
add_library(tvm_runtime_static STATIC ${RUNTIME_SRCS})
add_library(tvm_objs OBJECT ${COMPILER_SRCS})
add_library(tvm_runtime_objs OBJECT ${RUNTIME_SRCS})
add_library(tvm_libinfo_objs OBJECT ${LIBINFO_FILE})

add_library(tvm SHARED $<TARGET_OBJECTS:tvm_objs> $<TARGET_OBJECTS:tvm_runtime_objs> $<TARGET_OBJECTS:tvm_libinfo_objs>)
set_property(TARGET tvm APPEND PROPERTY LINK_OPTIONS "${TVM_NO_UNDEFINED_SYMBOLS}")
set_property(TARGET tvm APPEND PROPERTY LINK_OPTIONS "${TVM_VISIBILITY_FLAG}")
if(BUILD_STATIC_RUNTIME)
  add_library(tvm_runtime STATIC $<TARGET_OBJECTS:tvm_runtime_objs>)
  set(NOTICE_MULTILINE
    "You have build static version of the TVM runtime library. Make "
    "sure to use --whole-archive when linking it into your project.")
  string(CONCAT NOTICE ${NOTICE_MULTILINE})
  add_custom_command(TARGET tvm_runtime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --yellow --bold ${NOTICE})
else()
  add_library(tvm_runtime SHARED $<TARGET_OBJECTS:tvm_runtime_objs>)
  set_property(TARGET tvm_runtime APPEND PROPERTY LINK_OPTIONS "${TVM_NO_UNDEFINED_SYMBOLS}")
endif()
set_property(TARGET tvm_runtime APPEND PROPERTY LINK_OPTIONS "${TVM_VISIBILITY_FLAG}")

target_compile_definitions(tvm_objs PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
target_compile_definitions(tvm_runtime_objs PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
target_compile_definitions(tvm_libinfo_objs PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
target_compile_definitions(tvm PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
target_compile_definitions(tvm_runtime PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

# logging option for libbacktrace
include(cmake/modules/Logging.cmake)

include(cmake/modules/contrib/PAPI.cmake)

if(USE_MICRO)
  # NOTE: cmake doesn't track dependencies at the file level across subdirectories. For the
  # Unix Makefiles generator, need to add these explicit target-level dependency)
  add_dependencies(tvm host_standalone_crt)
  add_dependencies(tvm_runtime host_standalone_crt)
endif()

if(USE_CPP_RPC)
  add_subdirectory("apps/cpp_rpc")
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

if(BUILD_FOR_HEXAGON)
  # Wrap pthread_create to allow setting custom stack size.
  set_property(TARGET tvm_runtime APPEND PROPERTY LINK_FLAGS
                        "-Wl,--wrap=pthread_create")
endif()

if(USE_THREADS AND NOT BUILD_FOR_HEXAGON)
  message(STATUS "Build with thread support...")
  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)
  target_link_libraries(tvm PUBLIC Threads::Threads)
  target_link_libraries(tvm_runtime PUBLIC Threads::Threads)
endif()

target_link_libraries(tvm PRIVATE ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS})
target_link_libraries(tvm_runtime PRIVATE ${TVM_RUNTIME_LINKER_LIBS})

# Set flags for clang
include(cmake/modules/ClangFlags.cmake)

# Related headers
target_include_directories(
  tvm
  PUBLIC "topi/include")
target_include_directories(
  tvm_objs
  PUBLIC "topi/include")
target_include_directories(
  tvm_libinfo_objs
  PUBLIC "topi/include")
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
  target_compile_definitions(tvm_allvisible PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
endif()

# Create the `cpptest` target if we can find GTest.  If not, we create dummy
# targets that give the user an informative error message.
if(GTEST_INCLUDE_DIR AND GTEST_LIB)
  file(GLOB TEST_SRCS tests/cpp/*.cc)
  add_executable(cpptest ${TEST_SRCS})
  target_include_directories(cpptest SYSTEM PUBLIC ${GTEST_INCLUDE_DIR})
  target_link_libraries(cpptest PRIVATE ${TVM_TEST_LIBRARY_NAME} ${GTEST_LIB} gtest_main pthread dl)
  set_target_properties(cpptest PROPERTIES EXCLUDE_FROM_ALL 1)
  set_target_properties(cpptest PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
  gtest_discover_tests(cpptest)
elseif(NOT GTEST_INCLUDE_DIR)
  add_custom_target(cpptest
      COMMAND echo "Missing Google Test headers in include path"
      COMMAND exit 1)
elseif(NOT GTEST_LIB)
  add_custom_target(cpptest
      COMMAND echo "Missing Google Test library"
      COMMAND exit 1)
endif()

# Custom targets
add_custom_target(runtime DEPENDS tvm_runtime)

# Installation rules
install(TARGETS tvm DESTINATION lib${LIB_SUFFIX})
install(TARGETS tvm_runtime DESTINATION lib${LIB_SUFFIX})

if (INSTALL_DEV)
  install(
    DIRECTORY "include/." DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
  )
  install(
    DIRECTORY "3rdparty/dlpack/include/." DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
    )
  install(
    DIRECTORY "3rdparty/dmlc-core/include/." DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
    )
else(INSTALL_DEV)
  install(
    DIRECTORY "include/tvm/runtime/." DESTINATION "include/tvm/runtime"
    FILES_MATCHING
    PATTERN "*.h"
    )
endif(INSTALL_DEV)

# More target definitions
if(MSVC)
  target_compile_definitions(tvm_objs PRIVATE -DTVM_EXPORTS)
  target_compile_definitions(tvm_libinfo_objs PRIVATE -DTVM_EXPORTS)
  target_compile_definitions(tvm_runtime_objs PRIVATE -DTVM_EXPORTS)
  target_compile_definitions(tvm_runtime_static PRIVATE -DTVM_EXPORTS)
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

#Caches the build.
#Note that ccache-3.x doesn't support nvcc well, so CUDA kernels may never hit the cache and still
#need to be re-compiled every time. Using ccache 4.0+ can resolve this issue.

if(USE_CCACHE) # True for AUTO, ON, /path/to/ccache
  if("${USE_CCACHE}" STREQUAL "AUTO") # Auto mode
    find_program(CCACHE_FOUND ccache)
    if(CCACHE_FOUND)
      message(STATUS "Found the path to ccache, enabling ccache")
      set(PATH_TO_CCACHE ccache)
    else()
      message(STATUS "Didn't find the path to CCACHE, disabling ccache")
    endif(CCACHE_FOUND)
  elseif("${USE_CCACHE}" MATCHES ${IS_TRUE_PATTERN})
    find_program(CCACHE_FOUND ccache)
    if(CCACHE_FOUND)
      message(STATUS "Found the path to ccache, enabling ccache")
      set(PATH_TO_CCACHE ccache)
    else()
      message(FATAL_ERROR "Cannot find ccache. Set USE_CCACHE mode to AUTO or OFF to build without ccache. USE_CCACHE=" "${USE_CCACHE")
    endif(CCACHE_FOUND)
  else() # /path/to/ccache
    set(PATH_TO_CCACHE USE_CCACHE)
    message(STATUS "Setting ccache path to " "${PATH_TO_CCACHE}")
  endif()
  # Set the flag for ccache
  set(CXX_COMPILER_LAUNCHER PATH_TO_CCACHE)
endif(USE_CCACHE)
