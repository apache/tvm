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

if(USE_CUDA AND USE_CUTLASS)
  tvm_file_glob(GLOB CUTLASS_CONTRIB_SRC src/relay/backend/contrib/cutlass/*.cc src/relax/backend/contrib/cutlass/*.cc)
  list(APPEND COMPILER_SRCS ${CUTLASS_CONTRIB_SRC})

  set(FPA_INTB_GEMM_TVM_BINDING ON)
  set(FPA_INTB_GEMM_TVM_HOME ${PROJECT_SOURCE_DIR})

  set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cutlass)
  add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/cutlass_fpA_intB_gemm)
  add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/libflash_attn)
  list(APPEND RUNTIME_SRCS src/runtime/contrib/cutlass/weight_preprocess.cc)

  message(STATUS "Build with CUTLASS")
endif()
