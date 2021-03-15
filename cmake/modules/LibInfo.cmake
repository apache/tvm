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

# This script provides
#   - add_lib_info - A function to add definition flags to a specific file

function(add_lib_info src_file)
  if (NOT DEFINED TVM_INFO_LLVM_VERSION)
    set(TVM_INFO_LLVM_VERSION "NOT-FOUND")
  else()
    string(STRIP ${TVM_INFO_LLVM_VERSION} TVM_INFO_LLVM_VERSION)
  endif()
  set_property(
    SOURCE ${src_file}
    APPEND
    PROPERTY COMPILE_DEFINITIONS
    TVM_INFO_GIT_COMMIT_HASH="${TVM_GIT_COMMIT_HASH}"
    TVM_INFO_USE_CUDA="${USE_CUDA}"
    TVM_INFO_USE_OPENCL="${USE_OPENCL}"
    TVM_INFO_USE_VULKAN="${USE_VULKAN}"
    TVM_INFO_USE_METAL="${USE_METAL}"
    TVM_INFO_USE_ROCM="${USE_ROCM}"
    TVM_INFO_ROCM_PATH="${ROCM_PATH}"
    TVM_INFO_USE_HEXAGON_DEVICE="${USE_HEXAGON_DEVICE}"
    TVM_INFO_USE_HEXAGON_SDK="${USE_HEXAGON_SDK}"
    TVM_INFO_USE_RPC="${USE_RPC}"
    TVM_INFO_USE_THREADS="${USE_THREADS}"
    TVM_INFO_USE_LLVM="${USE_LLVM}"
    TVM_INFO_LLVM_VERSION="${TVM_INFO_LLVM_VERSION}"
    TVM_INFO_USE_STACKVM_RUNTIME="${USE_STACKVM_RUNTIME}"
    TVM_INFO_USE_GRAPH_RUNTIME="${USE_GRAPH_RUNTIME}"
    TVM_INFO_USE_GRAPH_RUNTIME_DEBUG="${USE_GRAPH_RUNTIME_DEBUG}"
    TVM_INFO_USE_OPENMP="${USE_OPENMP}"
    TVM_INFO_USE_RELAY_DEBUG="${USE_RELAY_DEBUG}"
    TVM_INFO_USE_RTTI="${USE_RTTI}"
    TVM_INFO_USE_MSVC_MT="${USE_MSVC_MT}"
    TVM_INFO_USE_MICRO="${USE_MICRO}"
    TVM_INFO_INSTALL_DEV="${INSTALL_DEV}"
    TVM_INFO_HIDE_PRIVATE_SYMBOLS="${HIDE_PRIVATE_SYMBOLS}"
    TVM_INFO_USE_TF_TVMDSOOP="${USE_TF_TVMDSOOP}"
    TVM_INFO_USE_FALLBACK_STL_MAP="${USE_FALLBACK_STL_MAP}"
    TVM_INFO_USE_BYODT_POSIT="${USE_BYODT_POSIT}"
    TVM_INFO_USE_BLAS="${USE_BLAS}"
    TVM_INFO_USE_MKL="${USE_MKL}"
    TVM_INFO_USE_MKLDNN="${USE_MKLDNN}"
    TVM_INFO_USE_DNNL_CODEGEN="${USE_DNNL_CODEGEN}"
    TVM_INFO_USE_CUDNN="${USE_CUDNN}"
    TVM_INFO_USE_CUBLAS="${USE_CUBLAS}"
    TVM_INFO_USE_THRUST="${USE_THRUST}"
    TVM_INFO_USE_MIOPEN="${USE_MIOPEN}"
    TVM_INFO_USE_ROCBLAS="${USE_ROCBLAS}"
    TVM_INFO_USE_SORT="${USE_SORT}"
    TVM_INFO_USE_NNPACK="${USE_NNPACK}"
    TVM_INFO_USE_RANDOM="${USE_RANDOM}"
    TVM_INFO_USE_MICRO_STANDALONE_RUNTIME="${USE_MICRO_STANDALONE_RUNTIME}"
    TVM_INFO_USE_CPP_RPC="${USE_CPP_RPC}"
    TVM_INFO_USE_TFLITE="${USE_TFLITE}"
    TVM_INFO_USE_TENSORFLOW_PATH="${USE_TENSORFLOW_PATH}"
    TVM_INFO_USE_COREML="${USE_COREML}"
    TVM_INFO_USE_TARGET_ONNX="${USE_TARGET_ONNX}"
    TVM_INFO_USE_ARM_COMPUTE_LIB="${USE_ARM_COMPUTE_LIB}"
    TVM_INFO_USE_ARM_COMPUTE_LIB_GRAPH_RUNTIME="${USE_ARM_COMPUTE_LIB_GRAPH_RUNTIME}"
    TVM_INFO_INDEX_DEFAULT_I64="${INDEX_DEFAULT_I64}"
    TVM_CXX_COMPILER_PATH="${CMAKE_CXX_COMPILER}"
  )

endfunction()
