/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/container/map.h>


#ifndef TVM_INFO_GIT_COMMIT_HASH
#define TVM_INFO_GIT_COMMIT_HASH "NOT-FOUND"
#endif

#ifndef TVM_INFO_GIT_COMMIT_TIME
#define TVM_INFO_GIT_COMMIT_TIME "NOT-FOUND"
#endif

#ifndef TVM_INFO_LLVM_VERSION
#define TVM_INFO_LLVM_VERSION "NOT-FOUND"
#endif

#ifndef TVM_INFO_MLIR_VERSION
#define TVM_INFO_MLIR_VERSION "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_CUDA
#define TVM_INFO_USE_CUDA "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_NVTX
#define TVM_INFO_USE_NVTX "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_NCCL
#define TVM_INFO_USE_NCCL "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_MSCCLPP
#define TVM_INFO_USE_MSCCLPP "NOT-FOUND"
#endif

#ifndef TVM_INFO_CUDA_VERSION
#define TVM_INFO_CUDA_VERSION "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_OPENCL
#define TVM_INFO_USE_OPENCL "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_OPENCL_ENABLE_HOST_PTR
#define TVM_INFO_USE_OPENCL_ENABLE_HOST_PTR "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_OPENCL_EXTN_QCOM
#define TVM_INFO_USE_OPENCL_EXTN_QCOM "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_OPENCL_GTEST
#define TVM_INFO_USE_OPENCL_GTEST "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_VULKAN
#define TVM_INFO_USE_VULKAN "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_METAL
#define TVM_INFO_USE_METAL "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_ROCM
#define TVM_INFO_USE_ROCM "NOT-FOUND"
#endif

#ifndef TVM_INFO_ROCM_PATH
#define TVM_INFO_ROCM_PATH "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_RCCL
#define TVM_INFO_USE_RCCL "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_HEXAGON
#define TVM_INFO_USE_HEXAGON "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_HEXAGON_SDK
#define TVM_INFO_USE_HEXAGON_SDK "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_HEXAGON_GTEST
#define TVM_INFO_USE_HEXAGON_GTEST "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_RPC
#define TVM_INFO_USE_RPC "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_THREADS
#define TVM_INFO_USE_THREADS "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_LLVM
#define TVM_INFO_USE_LLVM "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_OPENMP
#define TVM_INFO_USE_OPENMP "NOT-FOUND"
#endif

#ifndef TVM_INFO_DEBUG_WITH_ABI_CHANGE
#define TVM_INFO_DEBUG_WITH_ABI_CHANGE "NOT-FOUND"
#endif

#ifndef TVM_INFO_LOG_BEFORE_THROW
#define TVM_INFO_LOG_BEFORE_THROW "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_RTTI
#define TVM_INFO_USE_RTTI "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_MSVC_MT
#define TVM_INFO_USE_MSVC_MT "NOT-FOUND"
#endif

#ifndef TVM_INFO_INSTALL_DEV
#define TVM_INFO_INSTALL_DEV "NOT-FOUND"
#endif

#ifndef TVM_INFO_HIDE_PRIVATE_SYMBOLS
#define TVM_INFO_HIDE_PRIVATE_SYMBOLS "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_FALLBACK_STL_MAP
#define TVM_INFO_USE_FALLBACK_STL_MAP "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_BYODT_POSIT
#define TVM_INFO_USE_BYODT_POSIT "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_BLAS
#define TVM_INFO_USE_BLAS "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_MKL
#define TVM_INFO_USE_MKL "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_MRVL
#define TVM_INFO_USE_MRVL "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_AMX
#define TVM_INFO_USE_AMX "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_DNNL
#define TVM_INFO_USE_DNNL "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_CUDNN
#define TVM_INFO_USE_CUDNN "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_CUBLAS
#define TVM_INFO_USE_CUBLAS "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_THRUST
#define TVM_INFO_USE_THRUST "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_CURAND
#define TVM_INFO_USE_CURAND "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_MIOPEN
#define TVM_INFO_USE_MIOPEN "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_ROCBLAS
#define TVM_INFO_USE_ROCBLAS "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_SORT
#define TVM_INFO_USE_SORT "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_NNPACK
#define TVM_INFO_USE_NNPACK "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_RANDOM
#define TVM_INFO_USE_RANDOM "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_CPP_RPC
#define TVM_INFO_USE_CPP_RPC "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_CPP_RTVM
#define TVM_INFO_USE_CPP_RTVM "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_TFLITE
#define TVM_INFO_USE_TFLITE "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_TENSORFLOW_PATH
#define TVM_INFO_USE_TENSORFLOW_PATH "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_COREML
#define TVM_INFO_USE_COREML "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_ARM_COMPUTE_LIB
#define TVM_INFO_USE_ARM_COMPUTE_LIB "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR
#define TVM_INFO_USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR "NOT-FOUND"
#endif

#ifndef TVM_INFO_INDEX_DEFAULT_I64
#define TVM_INFO_INDEX_DEFAULT_I64 "NOT-FOUND"
#endif

#ifndef TVM_CXX_COMPILER_PATH
#define TVM_CXX_COMPILER_PATH ""
#endif

#ifndef TVM_INFO_USE_CCACHE
#define TVM_INFO_USE_CCACHE "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_NVSHMEM
#define TVM_INFO_USE_NVSHMEM "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_NNAPI_CODEGEN
#define TVM_INFO_USE_NNAPI_CODEGEN "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_NNAPI_RUNTIME
#define TVM_INFO_USE_NNAPI_RUNTIME "NOT-FOUND"
#endif

namespace tvm {

/*!
 * \brief Get a dictionary containing compile-time info, including cmake flags and git commit hash
 * \return The compile-time info
 */
TVM_DLL ffi::Map<ffi::String, ffi::String> GetLibInfo() {
  ffi::Map<ffi::String, ffi::String> result = {
      {"BUILD_STATIC_RUNTIME", TVM_INFO_BUILD_STATIC_RUNTIME},
      {"BUILD_DUMMY_LIBTVM", TVM_INFO_BUILD_DUMMY_LIBTVM},
      {"COMPILER_RT_PATH", TVM_INFO_COMPILER_RT_PATH},
      {"CUDA_VERSION", TVM_INFO_CUDA_VERSION},
      {"DLPACK_PATH", TVM_INFO_DLPACK_PATH},
      {"DMLC_PATH", TVM_INFO_DMLC_PATH},
      {"GIT_COMMIT_HASH", TVM_INFO_GIT_COMMIT_HASH},
      {"GIT_COMMIT_TIME", TVM_INFO_GIT_COMMIT_TIME},
      {"HIDE_PRIVATE_SYMBOLS", TVM_INFO_HIDE_PRIVATE_SYMBOLS},
      {"INDEX_DEFAULT_I64", TVM_INFO_INDEX_DEFAULT_I64},
      {"INSTALL_DEV", TVM_INFO_INSTALL_DEV},
      {"LLVM_VERSION", TVM_INFO_LLVM_VERSION},
      {"MLIR_VERSION", TVM_INFO_MLIR_VERSION},
      {"PICOJSON_PATH", TVM_INFO_PICOJSON_PATH},
      {"RANG_PATH", TVM_INFO_RANG_PATH},
      {"ROCM_PATH", TVM_INFO_ROCM_PATH},
      {"SUMMARIZE", TVM_INFO_SUMMARIZE},
      {"TVM_CXX_COMPILER_PATH", TVM_CXX_COMPILER_PATH},
      {"USE_ALTERNATIVE_LINKER", TVM_INFO_USE_ALTERNATIVE_LINKER},
      {"USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR", TVM_INFO_USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR},
      {"USE_ARM_COMPUTE_LIB", TVM_INFO_USE_ARM_COMPUTE_LIB},
      {"USE_BLAS", TVM_INFO_USE_BLAS},
      {"USE_BNNS", TVM_INFO_USE_BNNS},
      {"USE_BYODT_POSIT", TVM_INFO_USE_BYODT_POSIT},
      {"USE_COREML", TVM_INFO_USE_COREML},
      {"USE_CPP_RPC", TVM_INFO_USE_CPP_RPC},
      {"USE_CPP_RTVM", TVM_INFO_USE_CPP_RTVM},
      {"USE_CUBLAS", TVM_INFO_USE_CUBLAS},
      {"USE_CUDA", TVM_INFO_USE_CUDA},
      {"USE_NVTX", TVM_INFO_USE_NVTX},
      {"USE_NCCL", TVM_INFO_USE_NCCL},
      {"USE_MSCCL", TVM_INFO_USE_MSCCL},
      {"USE_CUDNN", TVM_INFO_USE_CUDNN},
      {"USE_CUSTOM_LOGGING", TVM_INFO_USE_CUSTOM_LOGGING},
      {"USE_CUTLASS", TVM_INFO_USE_CUTLASS},
      {"USE_FLASHINFER", TVM_INFO_USE_FLASHINFER},
      {"USE_AMX", TVM_INFO_USE_AMX},
      {"USE_DNNL", TVM_INFO_USE_DNNL},
      {"USE_FALLBACK_STL_MAP", TVM_INFO_USE_FALLBACK_STL_MAP},
      {"USE_GTEST", TVM_INFO_USE_GTEST},
      {"USE_HEXAGON", TVM_INFO_USE_HEXAGON},
      {"USE_HEXAGON_RPC", TVM_INFO_USE_HEXAGON_RPC},
      {"USE_HEXAGON_SDK", TVM_INFO_USE_HEXAGON_SDK},
      {"USE_HEXAGON_GTEST", TVM_INFO_USE_HEXAGON_GTEST},
      {"USE_HEXAGON_EXTERNAL_LIBS", TVM_INFO_USE_HEXAGON_EXTERNAL_LIBS},
      {"USE_IOS_RPC", TVM_INFO_USE_IOS_RPC},
      {"USE_KHRONOS_SPIRV", TVM_INFO_USE_KHRONOS_SPIRV},
      {"USE_LIBBACKTRACE", TVM_INFO_USE_LIBBACKTRACE},
      {"USE_LIBTORCH", TVM_INFO_USE_LIBTORCH},
      {"USE_LLVM", TVM_INFO_USE_LLVM},
      {"USE_MLIR", TVM_INFO_USE_MLIR},
      {"USE_METAL", TVM_INFO_USE_METAL},
      {"USE_MIOPEN", TVM_INFO_USE_MIOPEN},
      {"USE_MKL", TVM_INFO_USE_MKL},
      {"USE_MRVL", TVM_INFO_USE_MRVL},
      {"USE_MSVC_MT", TVM_INFO_USE_MSVC_MT},
      {"USE_NNPACK", TVM_INFO_USE_NNPACK},
      {"USE_OPENCL", TVM_INFO_USE_OPENCL},
      {"USE_OPENCL_ENABLE_HOST_PTR", TVM_INFO_USE_OPENCL_ENABLE_HOST_PTR},
      {"USE_OPENCL_EXTN_QCOM", TVM_INFO_USE_OPENCL_EXTN_QCOM},
      {"USE_OPENCL_GTEST", TVM_INFO_USE_OPENCL_GTEST},
      {"USE_OPENMP", TVM_INFO_USE_OPENMP},
      {"USE_PAPI", TVM_INFO_USE_PAPI},
      {"USE_RANDOM", TVM_INFO_USE_RANDOM},
      {"TVM_DEBUG_WITH_ABI_CHANGE", TVM_INFO_TVM_DEBUG_WITH_ABI_CHANGE},
      {"TVM_LOG_BEFORE_THROW", TVM_INFO_TVM_LOG_BEFORE_THROW},
      {"USE_ROCBLAS", TVM_INFO_USE_ROCBLAS},
      {"USE_HIPBLAS", TVM_INFO_USE_HIPBLAS},
      {"USE_ROCM", TVM_INFO_USE_ROCM},
      {"USE_RCCL", TVM_INFO_USE_RCCL},
      {"USE_RPC", TVM_INFO_USE_RPC},
      {"USE_RTTI", TVM_INFO_USE_RTTI},
      {"USE_RUST_EXT", TVM_INFO_USE_RUST_EXT},
      {"USE_SORT", TVM_INFO_USE_SORT},
      {"USE_SPIRV_KHR_INTEGER_DOT_PRODUCT", TVM_INFO_USE_SPIRV_KHR_INTEGER_DOT_PRODUCT},
      {"USE_STACKVM_RUNTIME", TVM_INFO_USE_STACKVM_RUNTIME},
      {"USE_TENSORFLOW_PATH", TVM_INFO_USE_TENSORFLOW_PATH},
      {"USE_TENSORRT_CODEGEN", TVM_INFO_USE_TENSORRT_CODEGEN},
      {"USE_TENSORRT_RUNTIME", TVM_INFO_USE_TENSORRT_RUNTIME},
      {"USE_TFLITE", TVM_INFO_USE_TFLITE},
      {"USE_THREADS", TVM_INFO_USE_THREADS},
      {"USE_THRUST", TVM_INFO_USE_THRUST},
      {"USE_CURAND", TVM_INFO_USE_CURAND},
      {"USE_VULKAN", TVM_INFO_USE_VULKAN},
      {"USE_CLML", TVM_INFO_USE_CLML},
      {"TVM_CLML_VERSION", TVM_INFO_USE_TVM_CLML_VERSION},
      {"USE_CLML_GRAPH_EXECUTOR", TVM_INFO_USE_CLML_GRAPH_EXECUTOR},
      {"USE_UMA", TVM_INFO_USE_UMA},
      {"USE_MSC", TVM_INFO_USE_MSC},
      {"USE_CCACHE", TVM_INFO_USE_CCACHE},
      {"USE_NVSHMEM", TVM_INFO_USE_NVSHMEM},
      {"USE_NNAPI_CODEGEN", TVM_INFO_USE_NNAPI_CODEGEN},
      {"USE_NNAPI_RUNTIME", TVM_INFO_USE_NNAPI_RUNTIME},
      {"BACKTRACE_ON_SEGFAULT", TVM_INFO_BACKTRACE_ON_SEGFAULT},
  };
  return result;
}

TVM_REGISTER_GLOBAL("support.GetLibInfo").set_body_typed(GetLibInfo);

}  // namespace tvm
