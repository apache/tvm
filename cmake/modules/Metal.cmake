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
  message(STATUS "Build with Metal support")
  find_library(METAL_LIB Metal)
  find_library(FOUNDATION_LIB Foundation)
  tvm_file_glob(GLOB RUNTIME_METAL_SRCS src/runtime/metal/*.mm)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${METAL_LIB} ${FOUNDATION_LIB})
  list(APPEND RUNTIME_SRCS ${RUNTIME_METAL_SRCS})

  if(USE_MPS)
    tvm_file_glob(GLOB MPS_CONTRIB_SRC src/runtime/contrib/mps/*.mm)
    list(APPEND RUNTIME_SRCS ${MPS_CONTRIB_SRC})
    find_library(MPS_CONTRIB_LIB MetalPerformanceShaders)
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${MPS_CONTRIB_LIB})
  endif()
else(USE_METAL)
  list(APPEND COMPILER_SRCS src/target/opt/build_metal_off.cc)
endif(USE_METAL)
