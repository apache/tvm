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

find_package(Python COMPONENTS Interpreter REQUIRED)

# call tvm_ffi.config to get the cmake directory and set it to tvm_ffi_ROOT
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --includedir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE tvm_ffi_INCLUDE_DIR)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --dlpack-includedir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE tvm_ffi_DLPACK_INCLUDE_DIR)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --libfiles
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE tvm_ffi_LIB_FILES)

message(STATUS "Finding libfiles ${tvm_ffi_LIB_FILES}")

add_library(tvm_ffi_header INTERFACE)
target_compile_features(tvm_ffi_header INTERFACE cxx_std_17)
target_include_directories(tvm_ffi_header INTERFACE "${tvm_ffi_INCLUDE_DIR}")
target_include_directories(tvm_ffi_header INTERFACE "${tvm_ffi_DLPACK_INCLUDE_DIR}")

add_library(tvm_ffi_shared SHARED IMPORTED)
target_compile_features(tvm_ffi_shared INTERFACE cxx_std_17)

if(WIN32)
  set_target_properties(
    tvm_ffi_shared PROPERTIES IMPORTED_IMPLIB "${tvm_ffi_LIB_FILES}"
  )
else()
  set_target_properties(
    tvm_ffi_shared PROPERTIES IMPORTED_LOCATION "${tvm_ffi_LIB_FILES}"
  )
endif()

set_target_properties(
  tvm_ffi_shared PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
  "${tvm_ffi_INCLUDE_DIR};${tvm_ffi_DLPACK_INCLUDE_DIR}"
)
# extra cmake functions
include(${CMAKE_CURRENT_LIST_DIR}/Utils/Library.cmake)
