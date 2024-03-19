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
  set(CUTLASS_GEN_COND "$<AND:$<BOOL:${USE_CUDA}>,$<BOOL:${USE_CUTLASS}>>")
  set(CUTLASS_RUNTIME_OBJS "")

  tvm_file_glob(GLOB CUTLASS_CONTRIB_SRC
    src/relay/backend/contrib/cutlass/*.cc
    src/relax/backend/contrib/cutlass/*.cc
  )
  list(APPEND COMPILER_SRCS ${CUTLASS_CONTRIB_SRC})

  set(FPA_INTB_GEMM_TVM_BINDING ON)
  set(FPA_INTB_GEMM_TVM_HOME ${PROJECT_SOURCE_DIR})

  ### Build cutlass runtime objects for fpA_intB_gemm using its cutlass submodule
  add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/cutlass_fpA_intB_gemm)
  target_include_directories(fpA_intB_gemm PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/cutlass_fpA_intB_gemm
    ${PROJECT_SOURCE_DIR}/3rdparty/cutlass_fpA_intB_gemm/cutlass/include
  )
  set(CUTLASS_FPA_INTB_RUNTIME_SRCS "")
  list(APPEND CUTLASS_FPA_INTB_RUNTIME_SRCS src/runtime/contrib/cutlass/moe_gemm.cc)
  list(APPEND CUTLASS_FPA_INTB_RUNTIME_SRCS src/runtime/contrib/cutlass/weight_preprocess.cc)
  add_library(fpA_intB_cutlass_objs OBJECT ${CUTLASS_FPA_INTB_RUNTIME_SRCS})
  target_include_directories(fpA_intB_cutlass_objs PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/cutlass_fpA_intB_gemm/cutlass/include
  )
  list(APPEND CUTLASS_RUNTIME_OBJS "$<${CUTLASS_GEN_COND}:$<TARGET_OBJECTS:fpA_intB_cutlass_objs>>")

  ### Build cutlass runtime objects for flash attention using its cutlass submodule
  add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/libflash_attn)
  target_include_directories(flash_attn PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/libflash_attn
    ${PROJECT_SOURCE_DIR}/3rdparty/libflash_attn/cutlass/include
  )
  set(CUTLASS_FLASH_ATTN_RUNTIME_SRCS "")
  list(APPEND CUTLASS_FLASH_ATTN_RUNTIME_SRCS src/runtime/contrib/cutlass/flash_decoding.cu)
  add_library(flash_attn_cutlass_objs OBJECT ${CUTLASS_FLASH_ATTN_RUNTIME_SRCS})
  target_include_directories(flash_attn_cutlass_objs PRIVATE
    ${PROJECT_SOURCE_DIR}/3rdparty/libflash_attn/cutlass/include
  )
  list(APPEND CUTLASS_RUNTIME_OBJS "$<${CUTLASS_GEN_COND}:$<TARGET_OBJECTS:flash_attn_cutlass_objs>>")

  ### Build cutlass runtime objects using TVM's 3rdparty/cutlass submodule
  set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cutlass)
  set(TVM_CUTLASS_RUNTIME_SRCS "")
  if (CMAKE_CUDA_ARCHITECTURES MATCHES "90")
    list(APPEND TVM_CUTLASS_RUNTIME_SRCS src/runtime/contrib/cutlass/fp16_fp8_gemm.cu)
  endif()
  if(TVM_CUTLASS_RUNTIME_SRCS)
    add_library(tvm_cutlass_objs OBJECT ${TVM_CUTLASS_RUNTIME_SRCS})
    target_include_directories(tvm_cutlass_objs PRIVATE ${CUTLASS_DIR}/include)
    list(APPEND CUTLASS_RUNTIME_OBJS "$<${CUTLASS_GEN_COND}:$<TARGET_OBJECTS:tvm_cutlass_objs>>")
  endif()

  ### Add cutlass objects to list of TVM runtime extension objs
  list(APPEND TVM_RUNTIME_EXT_OBJS "${CUTLASS_RUNTIME_OBJS}")

  message(STATUS "Build with CUTLASS")
endif()