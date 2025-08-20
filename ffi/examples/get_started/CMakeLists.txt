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
project(tvm_ffi_example)


# first find python related components
find_package(Python COMPONENTS Interpreter REQUIRED)

# call tvm_ffi.config to get the cmake directory and set it to tvm_ffi_ROOT
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE tvm_ffi_ROOT)
# find package will automatically include the related projects
find_package(tvm_ffi CONFIG REQUIRED)

# use the projects as usual
add_library(add_one_cpu SHARED src/add_one_cpu.cc)
target_link_libraries(add_one_cpu tvm_ffi_header)
target_link_libraries(add_one_cpu tvm_ffi_shared)
# show as add_one_cpu.so
set_target_properties(
  add_one_cpu PROPERTIES
  PREFIX ""
  SUFFIX ".so"
)

# Check if CUDA is available
if(NOT WIN32)
  find_package(CUDA QUIET)
  if(CUDA_FOUND)
    enable_language(CUDA)
    add_library(add_one_cuda SHARED src/add_one_cuda.cu)
    target_link_libraries(add_one_cuda tvm_ffi_shared)

    # show as add_one_cuda.so
    set_target_properties(
      add_one_cuda PROPERTIES
      PREFIX ""
      SUFFIX ".so"
      )
  endif()
endif()

add_executable(run_example src/run_example.cc)
set_target_properties(
  run_example PROPERTIES
  CXX_STANDARD 17
)
target_link_libraries(run_example tvm_ffi_shared)
