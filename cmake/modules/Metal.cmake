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

if(USE_METAL)
  message(STATUS "Build metal device runtime")
  find_library(METAL_LIB Metal)
  find_library(FOUNDATION_LIB Foundation)
  tvm_file_glob(GLOB RUNTIME_METAL_SRCS src/runtime/metal/*.mm)

  add_library(tvm_runtime_metal_objs OBJECT ${RUNTIME_METAL_SRCS})
  target_link_libraries(tvm_runtime_metal_objs PUBLIC tvm_ffi_header)
  set_target_properties(tvm_runtime_metal_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(TVM_VISIBILITY_FLAG)
    target_compile_options(tvm_runtime_metal_objs PRIVATE "${TVM_VISIBILITY_FLAG}")
  endif()
  add_library(tvm_runtime_metal SHARED $<TARGET_OBJECTS:tvm_runtime_metal_objs>)
  list(APPEND TVM_RUNTIME_BACKEND_LIBS tvm_runtime_metal)
  target_link_libraries(tvm_runtime_metal PUBLIC tvm_runtime ${METAL_LIB} ${FOUNDATION_LIB})
  tvm_configure_runtime_module(tvm_runtime_metal)
endif(USE_METAL)
# When USE_METAL=OFF the codegen-side fallback in
# src/target/metal/metal_fallback_module.cc handles construction; no opt
# stub is needed (it is always compiled via CODEGEN_SRCS in CMakeLists.txt).
