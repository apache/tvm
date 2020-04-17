cmake_minimum_required(VERSION 3.2)
project(tvm C CXX)

# Utility functions
include(cmake/util/Util.cmake)
include(cmake/util/FindCUDA.cmake)
include(cmake/util/FindVulkan.cmake)
include(cmake/util/FindLLVM.cmake)
include(cmake/util/FindROCM.cmake)
include(cmake/util/FindANTLR.cmake)

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
tvm_option(USE_OPENGL "Build with OpenGL" OFF)
tvm_option(USE_METAL "Build with Metal" OFF)
tvm_option(USE_ROCM "Build with ROCM" OFF)
tvm_option(ROCM_PATH "The path to rocm" /opt/rocm)
tvm_option(USE_HEXAGON_DEVICE "Build with Hexagon device support in TVM runtime" OFF)
tvm_option(USE_HEXAGON_SDK "Path to the Hexagon SDK root (required for Hexagon support in TVM runtime or for building TVM runtime for Hexagon)" /path/to/sdk)
tvm_option(USE_RPC "Build with RPC" ON)
tvm_option(USE_THREADS "Build with thread support" ON)
tvm_option(USE_LLVM "Build with LLVM, can be set to specific llvm-config path" OFF)
tvm_option(USE_STACKVM_RUNTIME "Include stackvm into the runtime" OFF)
tvm_option(USE_GRAPH_RUNTIME "Build with tiny graph runtime" ON)
tvm_option(USE_GRAPH_RUNTIME_DEBUG "Build with tiny graph runtime debug mode" OFF)
tvm_option(USE_OPENMP "Build with OpenMP thread pool implementation" OFF)
tvm_option(USE_RELAY_DEBUG "Building Relay in debug mode..." OFF)
tvm_option(USE_RTTI "Build with RTTI" ON)
tvm_option(USE_MSVC_MT "Build with MT" OFF)
tvm_option(USE_MICRO "Build with Micro" OFF)
tvm_option(INSTALL_DEV "Install compiler infrastructure" OFF)
tvm_option(HIDE_PRIVATE_SYMBOLS "Compile with -fvisibility=hidden." OFF)
tvm_option(USE_TF_TVMDSOOP "Build with TensorFlow TVMDSOOp" OFF)

# 3rdparty libraries
tvm_option(DLPACK_PATH "Path to DLPACK" "3rdparty/dlpack/include")
tvm_option(DMLC_PATH "Path to DMLC" "3rdparty/dmlc-core/include")
tvm_option(RANG_PATH "Path to RANG" "3rdparty/rang/include")
tvm_option(COMPILER_RT_PATH "Path to COMPILER-RT" "3rdparty/compiler-rt")
tvm_option(PICOJSON_PATH "Path to PicoJSON" "3rdparty/picojson")

# Contrib library options
tvm_option(USE_BLAS "The blas library to be linked" none)
tvm_option(USE_MKL_PATH "MKL root path when use MKL blas" none)
tvm_option(USE_MKLDNN "Build with MKLDNN" OFF)
tvm_option(USE_CUDNN "Build with cuDNN" OFF)
tvm_option(USE_CUBLAS "Build with cuBLAS" OFF)
tvm_option(USE_THRUST "Build with Thrust" OFF)
tvm_option(USE_MIOPEN "Build with ROCM:MIOpen" OFF)
tvm_option(USE_ROCBLAS "Build with ROCM:RoCBLAS" OFF)
tvm_option(USE_SORT "Build with sort support" OFF)
tvm_option(USE_NNPACK "Build with nnpack support" OFF)
tvm_option(USE_RANDOM "Build with random support" OFF)
tvm_option(USE_MICRO_STANDALONE_RUNTIME "Build with micro.standalone_runtime support" OFF)
tvm_option(USE_ANTLR "Build with ANTLR for Relay parsing" OFF)
tvm_option(USE_CPP_RPC "Build CPP RPC" OFF)
tvm_option(USE_TFLITE "Build with tflite support" OFF)
tvm_option(USE_TENSORFLOW_PATH "TensorFlow root path when use TFLite" none)

if(USE_CPP_RPC AND UNIX)
  message(FATAL_ERROR "USE_CPP_RPC is only supported with WIN32. Use the Makefile for non-Windows.")
endif()

# include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories("include")
include_directories(${DLPACK_PATH})
include_directories(${DMLC_PATH})
include_directories(${RANG_PATH})
include_directories(${COMPILER_RT_PATH})
include_directories(${PICOJSON_PATH})

# initial variables
set(TVM_LINKER_LIBS "")
set(TVM_RUNTIME_LINKER_LIBS "")

# Generic compilation options
if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")
  if(USE_MSVC_MT)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
else(MSVC)
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message("Build in Debug mode")
    set(CMAKE_C_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "-O0 -g -Xcompiler=-Wall -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  else()
    set(CMAKE_C_FLAGS "-O2 -Wall -fPIC ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "-O2 -Xcompiler=-Wall -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
    if (HIDE_PRIVATE_SYMBOLS)
      message(STATUS "Hide private symbols...")
      set(CMAKE_C_FLAGS "-fvisibility=hidden ${CMAKE_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "-fvisibility=hidden ${CMAKE_CXX_FLAGS}")
    endif(HIDE_PRIVATE_SYMBOLS)
  endif ()
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
  endif()

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
endif(MSVC)

# Hexagon has dlopen built into QuRT (no need for static library).
if(NOT BUILD_FOR_HEXAGON)
  string(APPEND TVM_RUNTIME_LINKER_LIBS ${CMAKE_DL_LIBS})
endif()

# add source group
FILE(GLOB_RECURSE GROUP_SOURCE "src/*.cc")
FILE(GLOB_RECURSE GROUP_INCLUDE "src/*.h" "include/*.h")
assign_source_group("Source" ${GROUP_SOURCE})
assign_source_group("Include" ${GROUP_INCLUDE})

# Source file lists
file(GLOB_RECURSE COMPILER_SRCS
    src/node/*.cc
    src/ir/*.cc
    src/arith/*.cc
    src/te/*.cc
    src/autotvm/*.cc
    src/tir/*.cc
    src/driver/*.cc
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


if(USE_VM_PROFILER)
  message(STATUS "Build compiler with Relay VM profiler support...")
  file(GLOB BACKEND_VM_PROFILER_SRCS src/relay/backend/vm/profiler/*.cc)
  list(APPEND COMPILER_SRCS ${BACKEND_VM_PROFILER_SRCS})
endif(USE_VM_PROFILER)

file(GLOB DATATYPE_SRCS src/target/datatype/*.cc)
list(APPEND COMPILER_SRCS ${DATATYPE_SRCS})


file(GLOB TOPI_SRCS
    topi/src/*.cc
)

file(GLOB RUNTIME_SRCS
  src/runtime/*.cc
  src/runtime/vm/*.cc
)

if(BUILD_FOR_HEXAGON)
  # Add file implementing posix_memalign.
  list(APPEND RUNTIME_SRCS src/runtime/hexagon/hexagon_posix.cc)

  add_definitions(-D_MACH_I32=int)
endif()

# Package runtime rules
if(NOT USE_RTTI)
  add_definitions(-DDMLC_ENABLE_RTTI=0)
endif()

list(APPEND RUNTIME_SRCS 3rdparty/bfloat16/bfloat16.cc)

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

if(USE_GRAPH_RUNTIME)
  message(STATUS "Build with Graph runtime support...")
  file(GLOB RUNTIME_GRAPH_SRCS src/runtime/graph/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_GRAPH_SRCS})

  if(USE_GRAPH_RUNTIME_DEBUG)
    message(STATUS "Build with Graph runtime debug support...")
    file(GLOB RUNTIME_GRAPH_DEBUG_SRCS src/runtime/graph/debug/*.cc)
    list(APPEND RUNTIME_SRCS ${RUNTIME_GRAPH_DEBUG_SRCS})
    set_source_files_properties(${RUNTIME_GRAPH_SRCS}
      PROPERTIES COMPILE_DEFINITIONS "TVM_GRAPH_RUNTIME_DEBUG")
  endif(USE_GRAPH_RUNTIME_DEBUG)
endif(USE_GRAPH_RUNTIME)

if(USE_VM_PROFILER)
  message(STATUS "Build with Relay VM profiler support...")
  file(GLOB RUNTIME_VM_PROFILER_SRCS src/runtime/vm/profiler/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_VM_PROFILER_SRCS})
endif(USE_VM_PROFILER)

if(USE_EXAMPLE_EXT_RUNTIME)
  message(STATUS "Build with example external runtime...")
  file(GLOB RUNTIME_EXAMPLE_EXTERNAL_SRCS src/runtime/contrib/example_ext_runtime/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_EXAMPLE_EXTERNAL_SRCS})
endif(USE_EXAMPLE_EXT_RUNTIME)

# Module rules
include(cmake/modules/VTA.cmake)
include(cmake/modules/CUDA.cmake)
include(cmake/modules/Hexagon.cmake)
include(cmake/modules/OpenCL.cmake)
include(cmake/modules/OpenGL.cmake)
include(cmake/modules/OpenMP.cmake)
include(cmake/modules/Vulkan.cmake)
include(cmake/modules/Metal.cmake)
include(cmake/modules/ROCM.cmake)
include(cmake/modules/LLVM.cmake)
include(cmake/modules/Micro.cmake)
include(cmake/modules/ANTLR.cmake)
include(cmake/modules/contrib/BLAS.cmake)
include(cmake/modules/contrib/CODEGENC.cmake)
include(cmake/modules/contrib/DNNL.cmake)
include(cmake/modules/contrib/Random.cmake)
include(cmake/modules/contrib/MicroStandaloneRuntime.cmake)
include(cmake/modules/contrib/Sort.cmake)
include(cmake/modules/contrib/NNPack.cmake)
include(cmake/modules/contrib/HybridDump.cmake)
include(cmake/modules/contrib/TFLite.cmake)
include(cmake/modules/contrib/TF_TVMDSOOP.cmake)

if(NOT MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++14" SUPPORT_CXX14)
  message(STATUS "Build with c++14")
  set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()

add_library(tvm SHARED ${COMPILER_SRCS} ${RUNTIME_SRCS})
add_library(tvm_topi SHARED ${TOPI_SRCS})
add_library(tvm_runtime SHARED ${RUNTIME_SRCS})

if(USE_CPP_RPC)
  add_subdirectory("apps/cpp_rpc")
endif()

if(USE_RELAY_DEBUG)
  message(STATUS "Building Relay in debug mode...")
  set_target_properties(tvm PROPERTIES COMPILE_DEFINITIONS "USE_RELAY_DEBUG")
  set_target_properties(tvm PROPERTIES COMPILE_DEFINITIONS "DMLC_LOG_DEBUG")
else()
  set_target_properties(tvm PROPERTIES COMPILE_DEFINITIONS "NDEBUG")
endif(USE_RELAY_DEBUG)

if(BUILD_FOR_HEXAGON)
  # Wrap pthread_create to allow setting custom stack size.
  set_target_properties(tvm_runtime PROPERTIES LINK_FLAGS
                        "-Wl,--wrap=pthread_create")

  target_include_directories(tvm_runtime
    PUBLIC "${USE_HEXAGON_SDK}/libs/common/qurt/ADSPv62MP/include/posix"
    PUBLIC "${USE_HEXAGON_SDK}/libs/common/qurt/ADSPv62MP/include/qurt"
    PUBLIC "${USE_HEXAGON_SDK}/incs"
    PUBLIC "${USE_HEXAGON_SDK}/incs/stddef")
endif()

if(USE_THREADS AND NOT BUILD_FOR_HEXAGON)
  message(STATUS "Build with thread support...")
  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)
  target_link_libraries(tvm Threads::Threads)
  target_link_libraries(tvm_topi Threads::Threads)
  target_link_libraries(tvm_runtime Threads::Threads)
endif()

target_link_libraries(tvm ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS})
target_link_libraries(tvm_topi tvm ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS})
target_link_libraries(tvm_runtime ${TVM_RUNTIME_LINKER_LIBS})

if (HIDE_PRIVATE_SYMBOLS AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(HIDE_SYMBOLS_LINKER_FLAGS "-Wl,--exclude-libs,ALL")
  # Note: 'target_link_options' with 'PRIVATE' keyword would be cleaner
  # but it's not available until CMake 3.13. Switch to 'target_link_options'
  # once minimum CMake version is bumped up to 3.13 or above.
  target_link_libraries(tvm ${HIDE_SYMBOLS_LINKER_FLAGS})
  target_link_libraries(tvm_topi ${HIDE_SYMBOLS_LINKER_FLAGS})
  target_link_libraries(tvm_runtime ${HIDE_SYMBOLS_LINKER_FLAGS})
endif()

# Related headers
target_include_directories(
  tvm
  PUBLIC "topi/include")
target_include_directories(
  tvm_topi
  PUBLIC "topi/include")


# Tests
set(TEST_EXECS "")
file(GLOB TEST_SRCS tests/cpp/*.cc)
find_path(GTEST_INCLUDE_DIR gtest/gtest.h)
find_library(GTEST_LIB gtest "$ENV{GTEST_LIB}")

# Create the `cpptest` target if we can find GTest.  If not, we create dummy
# targets that give the user an informative error message.
if(GTEST_INCLUDE_DIR AND GTEST_LIB)
  foreach(__srcpath ${TEST_SRCS})
    get_filename_component(__srcname ${__srcpath} NAME)
    string(REPLACE ".cc" "" __execname ${__srcname})
    add_executable(${__execname} ${__srcpath})
    list(APPEND TEST_EXECS ${__execname})
    target_include_directories(${__execname} PUBLIC ${GTEST_INCLUDE_DIR})
    target_link_libraries(${__execname} tvm ${GTEST_LIB} pthread dl)
    set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_ALL 1)
    set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
  endforeach()
  add_custom_target(cpptest DEPENDS ${TEST_EXECS})
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
install(TARGETS tvm_topi DESTINATION lib${LIB_SUFFIX})
install(TARGETS tvm_runtime DESTINATION lib${LIB_SUFFIX})

if (INSTALL_DEV)
  install(
    DIRECTORY "include/." DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
  )
  install(
    DIRECTORY "topi/include/." DESTINATION "include"
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
  target_compile_definitions(tvm PRIVATE -DTVM_EXPORTS)
  target_compile_definitions(tvm_runtime PRIVATE -DTVM_EXPORTS)
endif()
