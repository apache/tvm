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

if (USE_CSINN)
  find_csinn(${USE_CSINN} ${USE_CSINN_DEVICE_RUNTIME})
  if(NOT EXISTS ${CSINN_LIBRARIES})
    message(FATAL_ERROR "can not find CSI-NN2 lib at path: ${CSINN_LIBRARIES}")
  endif()

  if (USE_CSINN_DEVICE_RUNTIME)
    file(GLOB CSINN_RUNTIME_SRC src/runtime/contrib/csinn/csinn_json_runtime.cc)
    list(APPEND RUNTIME_SRCS ${CSINN_RUNTIME_SRC})
  endif()
  file(GLOB CSINN_RELAY_CONTRIB_SRC src/relay/backend/contrib/csinn/*.cc)

  include_directories(${CSINN_INCLUDE_DIRS})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CSINN_LIBRARIES})
  list(APPEND COMPILER_SRCS ${CSINN_RELAY_CONTRIB_SRC})

  if(${USE_CSINN_DEVICE_RUNTIME} STREQUAL "C906")
    add_definitions(-DUSE_CSINN_DEVICE_C906=1)
    set(__toolchain ${CSINN_INCLUDE_DIRS}/../tools/gcc-toolchain/bin)
    set(CMAKE_CXX_COMPILER ${__toolchain}/riscv64-unknown-linux-gnu-g++)
    list(APPEND TVM_RUNTIME_LINKER_LIBS atomic)
  endif()
endif(USE_CSINN)