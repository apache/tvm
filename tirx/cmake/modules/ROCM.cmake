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

# ROCM Module
if("${TVM_TIRX_USE_ROCM}" MATCHES ${IS_TRUE_PATTERN} AND IS_DIRECTORY "${TVM_TIRX_ROCM_PATH}")
  find_rocm("${TVM_TIRX_ROCM_PATH}")
else()
  find_rocm(${TVM_TIRX_USE_ROCM})
endif()

if(ROCM_FOUND)
  # always set the includedir
  # avoid global retrigger of CMake
  include_directories(SYSTEM ${ROCM_INCLUDE_DIRS})
  add_definitions(-D__HIP_PLATFORM_HCC__=1)
  add_definitions(-D__HIP_PLATFORM_AMD__=1)
endif(ROCM_FOUND)

if(TVM_TIRX_USE_ROCM)
  if(NOT ROCM_FOUND)
    message(FATAL_ERROR "Cannot find ROCM, TVM_TIRX_USE_ROCM=" ${TVM_TIRX_USE_ROCM})
  endif()
  message(STATUS "Build rocm device runtime")

  file(GLOB RUNTIME_ROCM_SRCS CONFIGURE_DEPENDS src/backend/rocm/runtime/*.cc)

  set(_rocm_libs ${ROCM_HIPHCC_LIBRARY})
  if(ROCM_HSA_LIBRARY)
    list(APPEND _rocm_libs ${ROCM_HSA_LIBRARY})
  endif()

  add_library(tvm_runtime_rocm_objs OBJECT ${RUNTIME_ROCM_SRCS})
  target_link_libraries(tvm_runtime_rocm_objs PUBLIC tvm_tirx_header)
  set_target_properties(tvm_runtime_rocm_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
  add_library(tvm_runtime_rocm SHARED $<TARGET_OBJECTS:tvm_runtime_rocm_objs>)
  install(TARGETS tvm_runtime_rocm DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT tvm_tirx)
  list(APPEND TVM_RUNTIME_BACKEND_LIBS tvm_runtime_rocm)
  target_link_libraries(tvm_runtime_rocm PUBLIC tvm_runtime PRIVATE ${_rocm_libs})
  tvm_tirx_configure_target(tvm_runtime_rocm_objs)
  tvm_tirx_configure_target(tvm_runtime_rocm)
endif(TVM_TIRX_USE_ROCM)
