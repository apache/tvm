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

# Declare native boolean switches and typed cache settings uniformly.
function(tvm_tirx_option variable type default description)
  if(type STREQUAL "BOOL")
    option(${variable} "${description}" "${default}")
  else()
    set(${variable} "${default}" CACHE ${type} "${description}")
  endif()
endfunction()

# ######### Common options ##########

set(TVM_TIRX_VERSION "0.1.0a0")
if(PROJECT_IS_TOP_LEVEL)
  tvm_tirx_option(TVM_TIRX_BUILD_TESTS BOOL OFF "Build C++ tests")
  tvm_tirx_option(TVM_TIRX_BUILD_PYTHON_MODULE BOOL OFF "Build the Python package with scikit-build-core")
endif()
tvm_tirx_option(TVM_TIRX_USE_THREADS BOOL ON "Build with thread support")
tvm_tirx_option(TVM_TIRX_USE_CUSTOM_LOGGING BOOL OFF "Use user-defined logging implementations")

# ######### LLVM ##########

tvm_tirx_option(TVM_TIRX_USE_LLVM STRING ON "Build with LLVM (OFF, ON, or llvm-config command)")

# ######### Z3 ##########

tvm_tirx_option(TVM_TIRX_USE_Z3 STRING AUTO "Build with Z3 SMT solver support (OFF, ON, AUTO)")

# ######### CUDA ##########

tvm_tirx_option(TVM_TIRX_USE_CUDA STRING AUTO "Build with CUDA (OFF, ON, AUTO, or SDK path)")

# ######### Other device runtimes ##########

# OpenCL
tvm_tirx_option(TVM_TIRX_USE_OPENCL STRING OFF "Build with OpenCL (OFF, ON, or SDK path)")
tvm_tirx_option(TVM_TIRX_USE_OPENCL_ENABLE_HOST_PTR BOOL OFF "Enable OpenCL memory object access to host")
tvm_tirx_option(TVM_TIRX_USE_OPENCL_EXTN_QCOM BOOL OFF "Enable Qualcomm OpenCL extensions")

# Vulkan
tvm_tirx_option(TVM_TIRX_USE_VULKAN STRING OFF "Build with Vulkan (OFF, ON, or SDK path)")
tvm_tirx_option(TVM_TIRX_USE_KHRONOS_SPIRV STRING OFF "Use Khronos SPIR-V tools (OFF or install path)")
tvm_tirx_option(TVM_TIRX_USE_SPIRV_KHR_INTEGER_DOT_PRODUCT BOOL OFF "Enable SPIR-V integer dot product")

# ROCm
tvm_tirx_option(TVM_TIRX_ROCM_PATH PATH /opt/rocm "Default ROCm SDK location")
tvm_tirx_option(TVM_TIRX_USE_ROCM STRING OFF "Build with ROCm (OFF, ON, or SDK path)")

# Metal
tvm_tirx_option(TVM_TIRX_USE_METAL BOOL OFF "Build with Metal")

# ######### OpenMP ##########

tvm_tirx_option(TVM_TIRX_USE_OPENMP STRING OFF "OpenMP thread pool implementation (OFF, gnu, intel)")
set_property(CACHE TVM_TIRX_USE_OPENMP PROPERTY STRINGS OFF gnu intel)

# ######### Dependencies ##########

# Use the same source-built FFI targets throughout this build graph.
add_subdirectory(3rdparty/tvm-ffi)
include(cmake/utils/Library.cmake)
include(cmake/utils/FindLLVM.cmake)
include(cmake/utils/FindCUDA.cmake)
include(cmake/utils/FindOpenCL.cmake)
include(cmake/utils/FindVulkan.cmake)
include(cmake/utils/FindROCM.cmake)
set(IS_FALSE_PATTERN "^[Oo][Ff][Ff]$|^0$|^[Ff][Aa][Ll][Ss][Ee]$|^[Nn][Oo]$|^[Nn][Oo][Tt][Ff][Oo][Uu][Nn][Dd]$|.*-[Nn][Oo][Tt][Ff][Oo][Uu][Nn][Dd]$|^$")
set(IS_TRUE_PATTERN "^[Oo][Nn]$|^[1-9][0-9]*$|^[Tt][Rr][Uu][Ee]$|^[Yy][Ee][Ss]$|^[Yy]$")
if(TVM_TIRX_USE_THREADS)
  find_package(Threads REQUIRED)
endif()

# ######### TIRx build settings ##########

include(GNUInstallDirs)
if(PROJECT_IS_TOP_LEVEL)
  if(TVM_TIRX_BUILD_PYTHON_MODULE)
    set(CMAKE_INSTALL_LIBDIR lib)
  endif()
endif()
add_compile_definitions(TVM_TIRX_VERSION="${TVM_TIRX_VERSION}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/compiler-rt")

if(MSVC)
  add_compile_definitions(WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS
    _SCL_SECURE_NO_WARNINGS _ENABLE_EXTENDED_ALIGNED_STORAGE NOMINMAX)
  add_compile_options(/EHsc /MP /bigobj /permissive-
    /wd4244 /wd4267 /wd4018 /wd4316 /wd4101 /wd4180 /wd4251 /wd4624 /wd4146 /wd4141 /wd4068)
else()
  add_compile_options(-Wall)
endif()

# Keep libraries together while honoring an embedding parent's output layout.
foreach(_kind ARCHIVE LIBRARY RUNTIME)
  if(NOT DEFINED CMAKE_${_kind}_OUTPUT_DIRECTORY)
    set(CMAKE_${_kind}_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
  endif()
  foreach(_config IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${_config}" _config)
    set(CMAKE_${_kind}_OUTPUT_DIRECTORY_${_config} "${CMAKE_${_kind}_OUTPUT_DIRECTORY}")
  endforeach()
endforeach()

# ######### Target: `tvm_tirx_header` ##########

add_library(tvm_tirx_header INTERFACE)
add_library(tvm_tirx::header ALIAS tvm_tirx_header)
target_compile_features(tvm_tirx_header INTERFACE cxx_std_20)
target_compile_definitions(tvm_tirx_header INTERFACE TVM_INDEX_DEFAULT_I64=1)
target_include_directories(tvm_tirx_header INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(tvm_tirx_header INTERFACE tvm_ffi::header)

# ######### Sources and backends ##########

set(TVM_LINKER_LIBS "")
set(TVM_RUNTIME_BACKEND_LIBS "")
set(BUILD_FOR_ANDROID "${ANDROID}")
# Source file lists
# Core compiler sources, extended by the backend modules.
file(GLOB_RECURSE COMPILER_SRCS CONFIGURE_DEPENDS
    src/ir/*.cc
    src/sym/*.cc
    src/tirx/*.cc
    src/support/*.cc
    src/script/*.cc
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

# ######### Targets: `tvm_runtime`, `tvm_runtime_static` ##########

add_library(tvm_runtime_objs OBJECT ${RUNTIME_SRCS})
target_link_libraries(tvm_runtime_objs PUBLIC tvm_tirx_header)
if(MSVC)
  target_compile_definitions(tvm_runtime_objs PRIVATE TVM_RUNTIME_EXPORTS TVM_FFI_EXPORTS)
endif()

add_library(tvm_runtime SHARED $<TARGET_OBJECTS:tvm_runtime_objs>)
add_library(tvm_runtime::shared ALIAS tvm_runtime)
target_link_libraries(tvm_runtime PUBLIC tvm_tirx_header tvm_ffi::shared)
add_library(tvm_runtime_static STATIC $<TARGET_OBJECTS:tvm_runtime_objs>)
target_link_libraries(tvm_runtime_static PUBLIC tvm_tirx_header tvm_ffi::static)
# Static consumers retain registration objects from both archives, with
# tvm_ffi::static before tvm_runtime_static in their whole-archive link group.
if(MSVC)
  target_compile_definitions(tvm_runtime_static INTERFACE
    TVM_RUNTIME_DLL= TVM_RUNTIME_DLL_EXPORT= TVM_FFI_DLL= TVM_FFI_DLL_EXPORT=)
endif()
foreach(_target tvm_runtime tvm_runtime_static)
  target_link_libraries(${_target} PRIVATE ${CMAKE_DL_LIBS})
  if(TVM_TIRX_USE_THREADS)
    target_link_libraries(${_target} PUBLIC Threads::Threads)
  endif()
endforeach()

# The core runtime owns the selected thread pool and its link dependencies.
include(cmake/modules/OpenMP.cmake)

# ######### Device runtimes and compiler backends ##########

# The core runtime exists before device runtimes link to it. Compiler backend
# contributions are collected before the compiler object target is created.
include(cmake/modules/CUDA.cmake)
include(cmake/modules/OpenCL.cmake)
include(cmake/modules/Vulkan.cmake)
include(cmake/modules/Metal.cmake)
include(cmake/modules/ROCM.cmake)
include(cmake/modules/LLVM.cmake)
include(cmake/modules/contrib/Z3.cmake)
include(cmake/modules/Git.cmake)
# ######### Target: `tvm_tirx` ##########

add_library(tvm_tirx_objs OBJECT ${COMPILER_SRCS})
target_link_libraries(tvm_tirx_objs PUBLIC tvm_tirx_header)
if(TARGET tvm_llvm_header)
  target_link_libraries(tvm_tirx_objs PRIVATE tvm_llvm_header)
endif()
if(MSVC)
  target_compile_definitions(tvm_tirx_objs PRIVATE TVM_TIRX_EXPORTS TVM_FFI_EXPORTS)
endif()
add_library(tvm_tirx SHARED $<TARGET_OBJECTS:tvm_tirx_objs>)
add_library(tvm_tirx::shared ALIAS tvm_tirx)
target_link_libraries(tvm_tirx PUBLIC tvm_runtime tvm_tirx_header)
target_link_libraries(tvm_tirx PRIVATE ${TVM_LINKER_LIBS})

# Avoid GNU ld relaxation of LLVM's GOT calls into incorrect direct calls.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT "${TVM_TIRX_USE_LLVM}" MATCHES ${IS_FALSE_PATTERN})
  target_link_options(tvm_tirx PRIVATE "-Wl,--no-relax" "-Wl,--as-needed")
endif()
include(cmake/modules/Logging.cmake)
include(cmake/modules/ClangFlags.cmake)

# ######### Library configuration ##########

foreach(_target tvm_runtime_objs tvm_tirx_objs tvm_runtime tvm_runtime_static tvm_tirx)
  tvm_tirx_configure_target(${_target})
  tvm_ffi_add_prefix_map(${_target} "${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()
foreach(_target IN LISTS TVM_RUNTIME_BACKEND_LIBS)
  tvm_ffi_add_prefix_map(${_target}_objs "${CMAKE_CURRENT_SOURCE_DIR}")
  tvm_ffi_add_prefix_map(${_target} "${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()
foreach(_target tvm_runtime tvm_tirx ${TVM_RUNTIME_BACKEND_LIBS})
  if(TVM_TIRX_BUILD_PYTHON_MODULE)
    if(APPLE)
      set_property(TARGET ${_target} APPEND PROPERTY INSTALL_RPATH "@loader_path/../../tvm_ffi/lib")
    elseif(UNIX)
      set_property(TARGET ${_target} APPEND PROPERTY INSTALL_RPATH "\$ORIGIN/../../tvm_ffi/lib")
    endif()
  endif()
  if(APPLE)
    target_link_options(${_target} PRIVATE "-Wl,-undefined,error")
  elseif(UNIX)
    target_link_options(${_target} PRIVATE "-Wl,--no-undefined")
  endif()
  tvm_ffi_hide_static_linked_lib_symbols(${_target})
  tvm_ffi_add_apple_dsymutil(${_target})
endforeach()

# ######### Installation ##########

# Public headers
file(GLOB_RECURSE _headers CONFIGURE_DEPENDS RELATIVE
  "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
foreach(_header IN LISTS _headers)
  get_filename_component(_directory "${_header}" DIRECTORY)
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/${_header}"
    DESTINATION "include/${_directory}" COMPONENT tvm_tirx)
endforeach()

# Libraries
install(TARGETS tvm_tirx_header tvm_tirx tvm_runtime tvm_runtime_static
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT tvm_tirx
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT tvm_tirx
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT tvm_tirx)

# CMake package and dependency helpers
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tvm_tirx-config.cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/python/tvm_tirx/config.py"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tvm_tirx" COMPONENT tvm_tirx)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tvm_tirx/utils"
  COMPONENT tvm_tirx FILES_MATCHING PATTERN "Find*.cmake")

# Python package artifacts and licenses
if(TVM_TIRX_BUILD_PYTHON_MODULE)
  foreach(_extra_lib IN LISTS TVM_TIRX_PACKAGE_EXTRA_LIBS)
    if(NOT EXISTS "${_extra_lib}")
      message(FATAL_ERROR "TVM_TIRX_PACKAGE_EXTRA_LIBS entry does not exist: ${_extra_lib}")
    endif()
    install(FILES "${_extra_lib}" DESTINATION lib COMPONENT tvm_tirx)
  endforeach()
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/licenses/" DESTINATION licenses COMPONENT tvm_tirx)
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE" "${CMAKE_CURRENT_SOURCE_DIR}/NOTICE"
    DESTINATION . COMPONENT tvm_tirx)
endif()

# ######### C++ tests ##########

if(TVM_TIRX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

set(TVM_TIRX_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if(NOT PROJECT_IS_TOP_LEVEL)
  # The parent checks the core version and reuses its dependency finders.
  set(TVM_TIRX_VERSION "${TVM_TIRX_VERSION}" PARENT_SCOPE)
  set(TVM_TIRX_CMAKE_DIR "${TVM_TIRX_CMAKE_DIR}" PARENT_SCOPE)
  # Parent runtime extensions must use the resolved AUTO selection, not the cache.
  set(TVM_TIRX_USE_CUDA "${TVM_TIRX_USE_CUDA}" PARENT_SCOPE)
  return()
endif()

# ######### Standalone build targets ##########

add_custom_target(runtime DEPENDS tvm_runtime tvm_runtime_static ${TVM_RUNTIME_BACKEND_LIBS})
