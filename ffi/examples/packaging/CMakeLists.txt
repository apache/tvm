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

cmake_minimum_required(VERSION 3.18)
project(my_ffi_extension)

option(TVM_FFI_EXT_FROM_SOURCE "Build tvm_ffi from source, useful for cross compilation." ON)
option(TVM_FFI_EXT_SHIP_DEBUG_SYMBOLS "Ship debug symbols" ON)

# There are two ways to include tvm_ffi
#
# 1. Build tvm_ffi from source, which is reasonably cheap since tvm ffi is small
# 2. Use the pre-built tvm_ffi shipped from the pip
#
# This example shows both options, you only need to pick a specific one.
#
# - For common build cases, using pre-built and link tvm_ffi_shared is sufficient.
# - For cases where you may want to cross-compile or bundle part of tvm_ffi_objects directly
#   into your project, opt for building tvm_ffi from source path.
#   Note that it is always safe to build from source and extra cost of building tvm_ffi is small.
#   So when in doubt, you can always choose to the building tvm_ffi from source route.
#
# In python or other cases when we dynamically load libtvm_ffi_shared. Even when you build
# from source, you do not need to ship libtvm_ffi.so built here as they are only
# used to supply the linking information.
# first find python related components
find_package(Python COMPONENTS Interpreter REQUIRED)
if (TVM_FFI_BUILD_FROM_SOURCE)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --sourcedir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE tvm_ffi_ROOT)
  message(STATUS "Building tvm_ffi from source: ${tvm_ffi_ROOT}")
  add_subdirectory(${tvm_ffi_ROOT} tvm_ffi)
else()
  # call tvm_ffi.config to get the cmake directory and set it to tvm_ffi_ROOT
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE tvm_ffi_ROOT)
  find_package(tvm_ffi CONFIG REQUIRED)
endif()

# use the projects as usual
add_library(my_ffi_extension SHARED src/extension.cc)
target_link_libraries(my_ffi_extension tvm_ffi_header)
target_link_libraries(my_ffi_extension tvm_ffi_shared)

# show as my_ffi_extension.so
set_target_properties(
  my_ffi_extension PROPERTIES PREFIX ""
)

if (TVM_FFI_EXT_SHIP_DEBUG_SYMBOLS)
  # ship debugging symbols for backtrace on macos
  tvm_ffi_add_prefix_map(my_ffi_extension ${CMAKE_CURRENT_SOURCE_DIR})
  tvm_ffi_add_apple_dsymutil(my_ffi_extension)
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ DESTINATION . FILES_MATCHING PATTERN "*.dSYM")
endif()

install(TARGETS my_ffi_extension DESTINATION .)
