# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

cmake_minimum_required(VERSION 3.26)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR AND NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compilation commands")
endif()
project(tvm_tirx LANGUAGES C CXX)

# ######### Configuration ##########

set(TVM_TIRX_VERSION "0.1.0a0")
if(PROJECT_IS_TOP_LEVEL)
  option(TVM_TIRX_BUILD_TESTS "Build C++ tests" OFF)
  option(TVM_TIRX_BUILD_PYTHON_MODULE "Build the Python package with scikit-build-core" OFF)
endif()
option(TVM_TIRX_USE_THREADS "Build with thread support" ON)
option(TVM_TIRX_USE_METAL "Build with Metal" OFF)
option(TVM_TIRX_USE_OPENCL_ENABLE_HOST_PTR "Enable OpenCL memory object access to host" OFF)
option(TVM_TIRX_USE_OPENCL_EXTN_QCOM "Enable Qualcomm OpenCL extensions" OFF)
option(TVM_TIRX_USE_SPIRV_KHR_INTEGER_DOT_PRODUCT "Enable SPIR-V integer dot product" OFF)
option(TVM_TIRX_USE_CUSTOM_LOGGING "Use user-defined logging implementations" OFF)

# These features also accept SDK paths or a named implementation.
set(TVM_TIRX_USE_CUDA OFF CACHE STRING "Build with CUDA (OFF, ON, or SDK path)")
set(TVM_TIRX_USE_OPENCL OFF CACHE STRING "Build with OpenCL (OFF, ON, or SDK path)")
set(TVM_TIRX_USE_VULKAN OFF CACHE STRING "Build with Vulkan (OFF, ON, or SDK path)")
set(TVM_TIRX_USE_KHRONOS_SPIRV OFF CACHE STRING "Use Khronos SPIR-V tools (OFF or install path)")
set(TVM_TIRX_ROCM_PATH /opt/rocm CACHE PATH "Default ROCm SDK location")
set(TVM_TIRX_USE_ROCM OFF CACHE STRING "Build with ROCm (OFF, ON, or SDK path)")
set(TVM_TIRX_USE_LLVM OFF CACHE STRING "Build with LLVM (OFF, ON, or llvm-config command)")
set(TVM_TIRX_USE_OPENMP OFF CACHE STRING "OpenMP thread pool implementation (OFF, gnu, intel)")
set(TVM_TIRX_USE_Z3 AUTO CACHE STRING "Build with Z3 SMT solver support (OFF, ON, AUTO)")
set_property(CACHE TVM_TIRX_USE_OPENMP PROPERTY STRINGS OFF gnu intel)
set_property(CACHE TVM_TIRX_USE_Z3 PROPERTY STRINGS OFF ON AUTO)

# ######### Dependencies ##########

# Use the same source-built FFI targets throughout this build graph.
add_subdirectory(3rdparty/tvm-ffi)
include(cmake/utils/Library.cmake)
foreach(_dependency CUDA OpenCL Vulkan LLVM ROCM)
  include(cmake/utils/Find${_dependency}.cmake)
endforeach()
set(IS_FALSE_PATTERN "^[Oo][Ff][Ff]$|^0$|^[Ff][Aa][Ll][Ss][Ee]$|^[Nn][Oo]$|^[Nn][Oo][Tt][Ff][Oo][Uu][Nn][Dd]$|.*-[Nn][Oo][Tt][Ff][Oo][Uu][Nn][Dd]$|^$")
set(IS_TRUE_PATTERN "^[Oo][Nn]$|^[1-9][0-9]*$|^[Tt][Rr][Uu][Ee]$|^[Yy][Ee][Ss]$|^[Yy]$")
if(TVM_TIRX_USE_THREADS)
  find_package(Threads REQUIRED)
endif()

# ######### Target: `tvm_tirx_header` ##########

add_library(tvm_tirx_header INTERFACE)
set_target_properties(tvm_tirx_header PROPERTIES EXPORT_NAME header)
target_compile_features(tvm_tirx_header INTERFACE cxx_std_20)
target_compile_definitions(tvm_tirx_header INTERFACE TVM_INDEX_DEFAULT_I64=1)
target_include_directories(tvm_tirx_header INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(tvm_tirx_header INTERFACE tvm_ffi::header)

# ######### Sources and backends ##########

set(TVM_LINKER_LIBS "")
set(TVM_RUNTIME_LINKER_LIBS "${CMAKE_DL_LIBS}")
set(TVM_RUNTIME_BACKEND_LIBS "")
set(BUILD_FOR_ANDROID "${ANDROID}")
# Source file lists
# Core compiler sources, extended by the backend modules.
file(GLOB_RECURSE COMPILER_SRCS CONFIGURE_DEPENDS
    src/ir/*.cc
    src/sym/*.cc
    src/tirx/*.cc
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
    )

file(GLOB CODEGEN_SRCS CONFIGURE_DEPENDS
  src/target/*.cc
  src/codegen/c/*.cc
  src/codegen/*.cc
  src/codegen/canonicalizer/*.cc
  src/codegen/llvm/target_kind.cc
  src/codegen/llvm/canonicalizer/*.cc
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
file(GLOB RUNTIME_SRCS CONFIGURE_DEPENDS
  src/runtime/*.cc
)
set(TVM_RUNTIME_EXT_OBJS "")

# Backend modules add sources, dependencies, and separate runtime targets.
foreach(_module CUDA OpenCL OpenMP Vulkan Metal ROCM LLVM contrib/Z3 Git)
  include(cmake/modules/${_module}.cmake)
endforeach()

# ######### Targets: `tvm_runtime`, `tvm_runtime_static` ##########

add_library(tvm_runtime_objs OBJECT ${RUNTIME_SRCS})
target_link_libraries(tvm_runtime_objs PUBLIC tvm_tirx_header)
tvm_tirx_configure_target(tvm_runtime_objs)
if(MSVC)
  target_compile_definitions(tvm_runtime_objs PRIVATE TVM_RUNTIME_EXPORTS TVM_FFI_EXPORTS)
endif()

add_library(tvm_runtime SHARED $<TARGET_OBJECTS:tvm_runtime_objs> ${TVM_RUNTIME_EXT_OBJS})
target_link_libraries(tvm_runtime PUBLIC tvm_tirx_header tvm_ffi::shared)
add_library(tvm_runtime_static STATIC $<TARGET_OBJECTS:tvm_runtime_objs> ${TVM_RUNTIME_EXT_OBJS})
target_link_libraries(tvm_runtime_static PUBLIC tvm_tirx_header tvm_ffi::static)
# Static consumers retain registration objects from both archives, with
# tvm_ffi::static before tvm_runtime_static in their whole-archive link group.
if(MSVC)
  target_compile_definitions(tvm_runtime_static INTERFACE
    TVM_RUNTIME_DLL= TVM_RUNTIME_DLL_EXPORT= TVM_FFI_DLL= TVM_FFI_DLL_EXPORT=)
endif()
foreach(_target tvm_runtime tvm_runtime_static)
  target_link_libraries(${_target} PRIVATE ${TVM_RUNTIME_LINKER_LIBS})
  if(TVM_TIRX_USE_THREADS)
    target_link_libraries(${_target} PUBLIC Threads::Threads)
  endif()
  tvm_tirx_configure_target(${_target})
endforeach()

# ######### Target: `tvm_tirx` ##########

add_library(tvm_tirx_objs OBJECT ${COMPILER_SRCS})
target_link_libraries(tvm_tirx_objs PUBLIC tvm_tirx_header)
if(TARGET tvm_llvm_header)
  target_link_libraries(tvm_tirx_objs PRIVATE tvm_llvm_header)
endif()
if(MSVC)
  target_compile_definitions(tvm_tirx_objs PRIVATE TVM_TIRX_EXPORTS TVM_FFI_EXPORTS)
endif()
tvm_tirx_configure_target(tvm_tirx_objs)
add_library(tvm_tirx SHARED $<TARGET_OBJECTS:tvm_tirx_objs>)
target_link_libraries(tvm_tirx PUBLIC tvm_runtime tvm_tirx_header)
target_link_libraries(tvm_tirx PRIVATE ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS})
tvm_tirx_configure_target(tvm_tirx)

# Avoid GNU ld relaxation of LLVM's GOT calls into incorrect direct calls.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT "${TVM_TIRX_USE_LLVM}" MATCHES ${IS_FALSE_PATTERN})
  target_link_options(tvm_tirx PRIVATE "-Wl,--no-relax" "-Wl,--as-needed")
endif()
include(cmake/modules/Logging.cmake)
include(cmake/modules/ClangFlags.cmake)

set(_tirx_targets tvm_tirx_header tvm_tirx tvm_runtime tvm_runtime_static ${TVM_RUNTIME_BACKEND_LIBS})
foreach(_target IN LISTS _tirx_targets)
  get_target_property(_export_name ${_target} EXPORT_NAME)
  if(NOT _export_name)
    set(_export_name ${_target})
  endif()
  add_library(tvm_tirx::${_export_name} ALIAS ${_target})
endforeach()
set(TVM_TIRX_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
foreach(_variable _tirx_targets TVM_TIRX_VERSION TVM_TIRX_USE_CUDA TVM_TIRX_USE_ROCM
    TVM_TIRX_USE_THREADS TVM_FFI_USE_THREADS TVM_FFI_USE_LIBBACKTRACE BUILD_FOR_ANDROID)
  set_property(TARGET tvm_tirx PROPERTY TVM_TIRX_INSTALL_${_variable} "${${_variable}}")
endforeach()
include(cmake/Install.cmake)
if(NOT PROJECT_IS_TOP_LEVEL)
  foreach(_variable TVM_TIRX_VERSION TVM_TIRX_CMAKE_DIR TVM_TIRX_USE_CUDA TVM_TIRX_USE_ROCM)
    set(${_variable} "${${_variable}}" PARENT_SCOPE)
  endforeach()
  set(TVM_TIRX_BUILD_FOR_ANDROID "${BUILD_FOR_ANDROID}" PARENT_SCOPE)
  return()
endif()

# ----------------------------------------------------------------------------
# Standalone tests, Python packaging, and installation.
# ----------------------------------------------------------------------------

add_custom_target(runtime DEPENDS tvm_runtime tvm_runtime_static ${TVM_RUNTIME_BACKEND_LIBS})

# ######### C++ tests ##########

if(TVM_TIRX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

# ######### Installation ##########

tvm_tirx_install()
