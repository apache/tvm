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
find_cuda(${USE_CUDA} ${USE_CUDNN})

if(CUDA_FOUND)
  # always set the includedir when cuda is available
  # avoid global retrigger of cmake
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)

if(USE_CUDA)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
  endif()
  message(STATUS "Build with CUDA ${CUDA_VERSION} support")
  enable_language(CUDA)
  tvm_file_glob(GLOB RUNTIME_CUDA_SRCS src/runtime/cuda/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_CUDA_SRCS})
  list(APPEND COMPILER_SRCS src/target/opt/build_cuda_on.cc)

  list(APPEND TVM_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDA_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_LESS "3.24")
      message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES not set. Please upgrade cmake to 3.24 to use native, or set CMAKE_CUDA_ARCHITECTURES manually")
    endif()
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not set, using native")
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()

  if(USE_CUDNN)
    message(STATUS "Build with cuDNN support")
    include_directories(SYSTEM ${CUDA_CUDNN_INCLUDE_DIRS})
    tvm_file_glob(GLOB CUDNN_RELAY_CONTRIB_SRC src/relay/backend/contrib/cudnn/*.cc src/relax/backend/contrib/cudnn/*.cc)
    list(APPEND COMPILER_SRCS ${CUDNN_RELAY_CONTRIB_SRC})
    tvm_file_glob(GLOB CONTRIB_CUDNN_SRCS src/runtime/contrib/cudnn/*.cc)
    list(APPEND RUNTIME_SRCS ${CONTRIB_CUDNN_SRCS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDNN_LIBRARY})
  endif(USE_CUDNN)

  if(USE_CUBLAS)
    message(STATUS "Build with cuBLAS support")
    tvm_file_glob(GLOB CUBLAS_CONTRIB_SRC src/relay/backend/contrib/cublas/*.cc src/relax/backend/contrib/cublas/*.cc)
    list(APPEND COMPILER_SRCS ${CUBLAS_CONTRIB_SRC})
    tvm_file_glob(GLOB CONTRIB_CUBLAS_SRCS src/runtime/contrib/cublas/*.cc)
    list(APPEND RUNTIME_SRCS ${CONTRIB_CUBLAS_SRCS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUBLAS_LIBRARY})
    if(NOT CUDA_CUBLASLT_LIBRARY STREQUAL "CUDA_CUBLASLT_LIBRARY-NOTFOUND")
      list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUBLASLT_LIBRARY})
    endif()
  endif(USE_CUBLAS)

  if(USE_THRUST)
    message(STATUS "Build with Thrust support")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
    tvm_file_glob(GLOB CONTRIB_THRUST_SRC src/runtime/contrib/thrust/*.cu)
    list(APPEND RUNTIME_SRCS ${CONTRIB_THRUST_SRC})
  endif(USE_THRUST)

  if(USE_CURAND)
    message(STATUS "Build with cuRAND support")
    message(STATUS "${CUDA_CURAND_LIBRARY}")
    tvm_file_glob(GLOB CONTRIB_CURAND_SRC_CC src/runtime/contrib/curand/*.cc)
    tvm_file_glob(GLOB CONTRIB_CURAND_SRC_CU src/runtime/contrib/curand/*.cu)
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CURAND_LIBRARY})
    list(APPEND RUNTIME_SRCS ${CONTRIB_CURAND_SRC_CC})
    list(APPEND RUNTIME_SRCS ${CONTRIB_CURAND_SRC_CU})
  endif(USE_CURAND)

  if(USE_NVTX)
    message(STATUS "Build with NVTX support")
    message(STATUS "${CUDA_NVTX_LIBRARY}")
    cmake_minimum_required(VERSION 3.13) # to compile CUDA code
    enable_language(CUDA)
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_NVTX_LIBRARY})
  endif(USE_NVTX)

  if(USE_GRAPH_EXECUTOR_CUDA_GRAPH)
    if(NOT USE_GRAPH_EXECUTOR)
      message(FATAL_ERROR "CUDA Graph is only supported by graph executor, please set USE_GRAPH_EXECUTOR=ON")
    endif()
    if(CUDAToolkit_VERSION_MAJOR LESS "10")
      message(FATAL_ERROR "CUDA Graph requires CUDA 10 or above, got=" ${CUDAToolkit_VERSION})
    endif()
    message(STATUS "Build with Graph executor with CUDA Graph support...")
    tvm_file_glob(GLOB RUNTIME_CUDA_GRAPH_SRCS src/runtime/graph_executor/cuda_graph/*.cc)
    list(APPEND RUNTIME_SRCS ${RUNTIME_CUDA_GRAPH_SRCS})
  endif()

  # Add CUDA builtins to RelaxVM
  tvm_file_glob(GLOB RELAX_VM_CUDA_BUILTIN_SRC_CC src/runtime/relax_vm/cuda/*.cc)
  list(APPEND RUNTIME_SRCS ${RELAX_VM_CUDA_BUILTIN_SRC_CC})
else(USE_CUDA)
  list(APPEND COMPILER_SRCS src/target/opt/build_cuda_off.cc)
endif(USE_CUDA)
