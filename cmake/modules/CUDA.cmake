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

# CUDA Module
find_cuda(${USE_CUDA})

if(CUDA_FOUND)
  # always set the includedir when cuda is available
  # avoid global retrigger of cmake
	include_directories(${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)

if(USE_CUDA)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
  endif()
  message(STATUS "Build with CUDA support")
  file(GLOB RUNTIME_CUDA_SRCS src/runtime/cuda/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_CUDA_SRCS})
  list(APPEND COMPILER_SRCS src/target/opt/build_cuda_on.cc)

  list(APPEND TVM_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDA_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})

  if(USE_CUDNN)
    message(STATUS "Build with cuDNN support")
    file(GLOB CONTRIB_CUDNN_SRCS src/runtime/contrib/cudnn/*.cc)
    list(APPEND RUNTIME_SRCS ${CONTRIB_CUDNN_SRCS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDNN_LIBRARY})
    include_directories(${USE_CUDNN}/include)
  endif(USE_CUDNN)

  if(USE_CUBLAS)
    message(STATUS "Build with cuBLAS support")
    file(GLOB CONTRIB_CUBLAS_SRCS src/runtime/contrib/cublas/*.cc)
    list(APPEND RUNTIME_SRCS ${CONTRIB_CUBLAS_SRCS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUBLAS_LIBRARY})
    if(NOT CUDA_CUBLASLT_LIBRARY STREQUAL "CUDA_CUBLASLT_LIBRARY-NOTFOUND")
      list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUBLASLT_LIBRARY})
    endif()
  endif(USE_CUBLAS)

  if(USE_THRUST)
    message(STATUS "Build with Thrust support")
    cmake_minimum_required(VERSION 3.13) # to compile CUDA code
    enable_language(CUDA)
    file(GLOB CONTRIB_THRUST_SRC src/runtime/contrib/thrust/*.cu)
    list(APPEND RUNTIME_SRCS ${CONTRIB_THRUST_SRC})
  endif(USE_THRUST)

else(USE_CUDA)
  list(APPEND COMPILER_SRCS src/target/opt/build_cuda_off.cc)
endif(USE_CUDA)
