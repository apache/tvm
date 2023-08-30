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
  if (NOT DEFINED CUDA_VERSION)
    set(TVM_INFO_CUDA_VERSION "NOT-FOUND")
  else()
    string(STRIP ${CUDA_VERSION} TVM_INFO_CUDA_VERSION)
  endif()

  set_property(
    SOURCE ${src_file}
    APPEND
    PROPERTY COMPILE_DEFINITIONS
    TVM_CXX_COMPILER_PATH="${CMAKE_CXX_COMPILER}"
    TVM_INFO_BUILD_STATIC_RUNTIME="${BUILD_STATIC_RUNTIME}"
    TVM_INFO_BUILD_DUMMY_LIBTVM="${BUILD_DUMMY_LIBTVM}"
    TVM_INFO_COMPILER_RT_PATH="${COMPILER_RT_PATH}"
    TVM_INFO_CUDA_VERSION="${TVM_INFO_CUDA_VERSION}"
    TVM_INFO_DLPACK_PATH="${DLPACK_PATH}"
    TVM_INFO_DMLC_PATH="${DMLC_PATH}"
    TVM_INFO_GIT_COMMIT_HASH="${TVM_GIT_COMMIT_HASH}"
    TVM_INFO_GIT_COMMIT_TIME="${TVM_GIT_COMMIT_TIME}"
    TVM_INFO_HIDE_PRIVATE_SYMBOLS="${HIDE_PRIVATE_SYMBOLS}"
    TVM_INFO_INDEX_DEFAULT_I64="${INDEX_DEFAULT_I64}"
    TVM_INFO_INSTALL_DEV="${INSTALL_DEV}"
    TVM_INFO_LLVM_VERSION="${TVM_INFO_LLVM_VERSION}"
    TVM_INFO_PICOJSON_PATH="${PICOJSON_PATH}"
    TVM_INFO_RANG_PATH="${RANG_PATH}"
    TVM_INFO_ROCM_PATH="${ROCM_PATH}"
    TVM_INFO_SUMMARIZE="${SUMMARIZE}"
    TVM_INFO_USE_ALTERNATIVE_LINKER="${USE_ALTERNATIVE_LINKER}"
    TVM_INFO_USE_AOT_EXECUTOR="${USE_AOT_EXECUTOR}"
    TVM_INFO_USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR="${USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR}"
    TVM_INFO_USE_ARM_COMPUTE_LIB="${USE_ARM_COMPUTE_LIB}"
    TVM_INFO_USE_BLAS="${USE_BLAS}"
    TVM_INFO_USE_BNNS="${USE_BNNS}"
    TVM_INFO_USE_BYODT_POSIT="${USE_BYODT_POSIT}"
    TVM_INFO_USE_CMSISNN="${USE_CMSISNN}"
    TVM_INFO_USE_COREML="${USE_COREML}"
    TVM_INFO_USE_CPP_RPC="${USE_CPP_RPC}"
    TVM_INFO_USE_CPP_RTVM="${USE_CPP_RTVM}"
    TVM_INFO_USE_CUBLAS="${USE_CUBLAS}"
    TVM_INFO_USE_CUDA="${USE_CUDA}"
    TVM_INFO_USE_NCCL="${USE_NCCL}"
    TVM_INFO_USE_CUDNN="${USE_CUDNN}"
    TVM_INFO_USE_CUSTOM_LOGGING="${USE_CUSTOM_LOGGING}"
    TVM_INFO_USE_CUTLASS="${USE_CUTLASS}"
    TVM_INFO_USE_AMX="${USE_AMX}"
    TVM_INFO_USE_DNNL="${USE_DNNL}"
    TVM_INFO_USE_ETHOSN="${USE_ETHOSN}"
    TVM_INFO_USE_ETHOSU="${USE_ETHOSU}"
    TVM_INFO_USE_FALLBACK_STL_MAP="${USE_FALLBACK_STL_MAP}"
    TVM_INFO_USE_GRAPH_EXECUTOR_CUDA_GRAPH="${USE_GRAPH_EXECUTOR_CUDA_GRAPH}"
    TVM_INFO_USE_GRAPH_EXECUTOR="${USE_GRAPH_EXECUTOR}"
    TVM_INFO_USE_GTEST="${USE_GTEST}"
    TVM_INFO_USE_HEXAGON="${USE_HEXAGON}"
    TVM_INFO_USE_HEXAGON_RPC="${USE_HEXAGON_RPC}"
    TVM_INFO_USE_HEXAGON_SDK="${USE_HEXAGON_SDK}"
    TVM_INFO_USE_HEXAGON_GTEST="${USE_HEXAGON_GTEST}"
    TVM_INFO_USE_HEXAGON_EXTERNAL_LIBS="${USE_HEXAGON_EXTERNAL_LIBS}"
    TVM_INFO_USE_IOS_RPC="${USE_IOS_RPC}"
    TVM_INFO_USE_KHRONOS_SPIRV="${USE_KHRONOS_SPIRV}"
    TVM_INFO_USE_LIBBACKTRACE="${USE_LIBBACKTRACE}"
    TVM_INFO_USE_LIBTORCH="${USE_LIBTORCH}"
    TVM_INFO_USE_LLVM="${USE_LLVM}"
    TVM_INFO_USE_METAL="${USE_METAL}"
    TVM_INFO_USE_MICRO_STANDALONE_RUNTIME="${USE_MICRO_STANDALONE_RUNTIME}"
    TVM_INFO_USE_MICRO="${USE_MICRO}"
    TVM_INFO_USE_MIOPEN="${USE_MIOPEN}"
    TVM_INFO_USE_MKL="${USE_MKL}"
    TVM_INFO_USE_MSVC_MT="${USE_MSVC_MT}"
    TVM_INFO_USE_NNPACK="${USE_NNPACK}"
    TVM_INFO_USE_OPENCL="${USE_OPENCL}"
    TVM_INFO_USE_OPENCL_ENABLE_HOST_PTR="${USE_OPENCL_ENABLE_HOST_PTR}"
    TVM_INFO_USE_OPENCL_GTEST="${USE_OPENCL_GTEST}"
    TVM_INFO_USE_OPENMP="${USE_OPENMP}"
    TVM_INFO_USE_PAPI="${USE_PAPI}"
    TVM_INFO_USE_PROFILER="${USE_PROFILER}"
    TVM_INFO_USE_PT_TVMDSOOP="${USE_PT_TVMDSOOP}"
    TVM_INFO_USE_RANDOM="${USE_RANDOM}"
    TVM_INFO_USE_RELAY_DEBUG="${USE_RELAY_DEBUG}"
    TVM_INFO_USE_ROCBLAS="${USE_ROCBLAS}"
    TVM_INFO_USE_ROCM="${USE_ROCM}"
    TVM_INFO_USE_RCCL="${USE_RCCL}"
    TVM_INFO_USE_RPC="${USE_RPC}"
    TVM_INFO_USE_RTTI="${USE_RTTI}"
    TVM_INFO_USE_RUST_EXT="${USE_RUST_EXT}"
    TVM_INFO_USE_SORT="${USE_SORT}"
    TVM_INFO_USE_SPIRV_KHR_INTEGER_DOT_PRODUCT="${USE_SPIRV_KHR_INTEGER_DOT_PRODUCT}"
    TVM_INFO_USE_STACKVM_RUNTIME="${USE_STACKVM_RUNTIME}"
    TVM_INFO_USE_TARGET_ONNX="${USE_TARGET_ONNX}"
    TVM_INFO_USE_TENSORFLOW_PATH="${USE_TENSORFLOW_PATH}"
    TVM_INFO_USE_TENSORRT_CODEGEN="${USE_TENSORRT_CODEGEN}"
    TVM_INFO_USE_TENSORRT_RUNTIME="${USE_TENSORRT_RUNTIME}"
    TVM_INFO_USE_TF_TVMDSOOP="${USE_TF_TVMDSOOP}"
    TVM_INFO_USE_TFLITE="${USE_TFLITE}"
    TVM_INFO_USE_THREADS="${USE_THREADS}"
    TVM_INFO_USE_THRUST="${USE_THRUST}"
    TVM_INFO_USE_CURAND="${USE_CURAND}"
    TVM_INFO_USE_VITIS_AI="${USE_VITIS_AI}"
    TVM_INFO_USE_VULKAN="${USE_VULKAN}"
    TVM_INFO_USE_CLML="${USE_CLML}"
    TVM_INFO_USE_CLML_GRAPH_EXECUTOR="${USE_CLML_GRAPH_EXECUTOR}"
    TVM_INFO_USE_TVM_CLML_VERSION="${CLML_VERSION_MAJOR}"
    TVM_INFO_USE_UMA="${USE_UMA}"
    TVM_INFO_USE_VERILATOR="${USE_VERILATOR}"
    TVM_INFO_USE_CCACHE="${USE_CCACHE}"
    TVM_INFO_BACKTRACE_ON_SEGFAULT="${BACKTRACE_ON_SEGFAULT}"
  )

endfunction()
