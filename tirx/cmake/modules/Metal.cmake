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

if(TVM_TIRX_USE_METAL)
  message(STATUS "Build metal device runtime")
  find_library(METAL_LIB Metal)
  find_library(FOUNDATION_LIB Foundation)
  file(GLOB RUNTIME_METAL_SRCS CONFIGURE_DEPENDS src/backend/metal/runtime/*.mm)

  add_library(tvm_runtime_metal_objs OBJECT ${RUNTIME_METAL_SRCS})
  target_link_libraries(tvm_runtime_metal_objs PUBLIC tvm_tirx_header)
  set_target_properties(tvm_runtime_metal_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
  add_library(tvm_runtime_metal SHARED $<TARGET_OBJECTS:tvm_runtime_metal_objs>)
  if(TVM_TIRX_INSTALL)
    install(TARGETS tvm_runtime_metal DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
  list(APPEND TVM_RUNTIME_BACKEND_LIBS tvm_runtime_metal)
  target_link_libraries(tvm_runtime_metal PUBLIC tvm_runtime PRIVATE ${METAL_LIB} ${FOUNDATION_LIB})
  tvm_tirx_configure_target(tvm_runtime_metal_objs)
  tvm_tirx_configure_target(tvm_runtime_metal)
endif(TVM_TIRX_USE_METAL)
# When TVM_TIRX_USE_METAL=OFF the codegen-side fallback in
# src/backend/metal/codegen/metal_fallback_module.cc handles construction; no opt
# stub is needed (it is always compiled via CODEGEN_SRCS in CMakeLists.txt).
