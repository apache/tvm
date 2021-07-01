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

#ifndef TVM_INFO_GIT_COMMIT_HASH
#define TVM_INFO_GIT_COMMIT_HASH "NOT-FOUND"
#endif

#ifndef TVM_INFO_GIT_COMMIT_TIME
#define TVM_INFO_GIT_COMMIT_TIME "NOT-FOUND"
#endif

#ifndef TVM_INFO_LLVM_VERSION
#define TVM_INFO_LLVM_VERSION "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_CUDA
#define TVM_INFO_USE_CUDA "NOT-FOUND"
#endif

#ifndef TVM_INFO_CUDA_VERSION
#define TVM_INFO_CUDA_VERSION "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_OPENCL
#define TVM_INFO_USE_OPENCL "NOT-FOUND"
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

#ifndef TVM_INFO_USE_HEXAGON_DEVICE
#define TVM_INFO_USE_HEXAGON_DEVICE "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_HEXAGON_SDK
#define TVM_INFO_USE_HEXAGON_SDK "NOT-FOUND"
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

#ifndef TVM_INFO_USE_STACKVM_RUNTIME
#define TVM_INFO_USE_STACKVM_RUNTIME "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_GRAPH_EXECUTOR
#define TVM_INFO_USE_GRAPH_EXECUTOR "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_GRAPH_EXECUTOR_DEBUG
#define TVM_INFO_USE_GRAPH_EXECUTOR_DEBUG "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_OPENMP
#define TVM_INFO_USE_OPENMP "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_RELAY_DEBUG
#define TVM_INFO_USE_RELAY_DEBUG "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_RTTI
#define TVM_INFO_USE_RTTI "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_MSVC_MT
#define TVM_INFO_USE_MSVC_MT "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_MICRO
#define TVM_INFO_USE_MICRO "NOT-FOUND"
#endif

#ifndef TVM_INFO_INSTALL_DEV
#define TVM_INFO_INSTALL_DEV "NOT-FOUND"
#endif

#ifndef TVM_INFO_HIDE_PRIVATE_SYMBOLS
#define TVM_INFO_HIDE_PRIVATE_SYMBOLS "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_TF_TVMDSOOP
#define TVM_INFO_USE_TF_TVMDSOOP "NOT-FOUND"
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

#ifndef TVM_INFO_USE_MKLDNN
#define TVM_INFO_USE_MKLDNN "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_DNNL_CODEGEN
#define TVM_INFO_USE_DNNL_CODEGEN "NOT-FOUND"
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

#ifndef TVM_INFO_USE_MICRO_STANDALONE_RUNTIME
#define TVM_INFO_USE_MICRO_STANDALONE_RUNTIME "NOT-FOUND"
#endif

#ifndef TVM_INFO_USE_CPP_RPC
#define TVM_INFO_USE_CPP_RPC "NOT-FOUND"
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

#ifndef TVM_INFO_USE_TARGET_ONNX
#define TVM_INFO_USE_TARGET_ONNX "NOT-FOUND"
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

namespace tvm {

/*!
 * \brief Get a dictionary containing compile-time info, including cmake flags and git commit hash
 * \return The compile-time info
 */
TVM_DLL Map<String, String> GetLibInfo() {
  Map<String, String> result = {
      {"GIT_COMMIT_HASH", TVM_INFO_GIT_COMMIT_HASH},
      {"GIT_COMMIT_TIME", TVM_INFO_GIT_COMMIT_TIME},
      {"USE_CUDA", TVM_INFO_USE_CUDA},
      {"USE_OPENCL", TVM_INFO_USE_OPENCL},
      {"USE_VULKAN", TVM_INFO_USE_VULKAN},
      {"USE_METAL", TVM_INFO_USE_METAL},
      {"USE_ROCM", TVM_INFO_USE_ROCM},
      {"ROCM_PATH", TVM_INFO_ROCM_PATH},
      {"USE_HEXAGON_DEVICE", TVM_INFO_USE_HEXAGON_DEVICE},
      {"USE_HEXAGON_SDK", TVM_INFO_USE_HEXAGON_SDK},
      {"USE_RPC", TVM_INFO_USE_RPC},
      {"USE_THREADS", TVM_INFO_USE_THREADS},
      {"USE_LLVM", TVM_INFO_USE_LLVM},
      {"LLVM_VERSION", TVM_INFO_LLVM_VERSION},
      {"CUDA_VERSION", TVM_INFO_CUDA_VERSION},
      {"USE_STACKVM_RUNTIME", TVM_INFO_USE_STACKVM_RUNTIME},
      {"USE_GRAPH_EXECUTOR", TVM_INFO_USE_GRAPH_EXECUTOR},
      {"USE_GRAPH_EXECUTOR_DEBUG", TVM_INFO_USE_GRAPH_EXECUTOR_DEBUG},
      {"USE_OPENMP", TVM_INFO_USE_OPENMP},
      {"USE_RELAY_DEBUG", TVM_INFO_USE_RELAY_DEBUG},
      {"USE_RTTI", TVM_INFO_USE_RTTI},
      {"USE_MSVC_MT", TVM_INFO_USE_MSVC_MT},
      {"USE_MICRO", TVM_INFO_USE_MICRO},
      {"INSTALL_DEV", TVM_INFO_INSTALL_DEV},
      {"HIDE_PRIVATE_SYMBOLS", TVM_INFO_HIDE_PRIVATE_SYMBOLS},
      {"USE_TF_TVMDSOOP", TVM_INFO_USE_TF_TVMDSOOP},
      {"USE_FALLBACK_STL_MAP", TVM_INFO_USE_FALLBACK_STL_MAP},
      {"USE_BYODT_POSIT", TVM_INFO_USE_BYODT_POSIT},
      {"USE_BLAS", TVM_INFO_USE_BLAS},
      {"USE_MKL", TVM_INFO_USE_MKL},
      {"USE_MKLDNN", TVM_INFO_USE_MKLDNN},
      {"USE_DNNL_CODEGEN", TVM_INFO_USE_DNNL_CODEGEN},
      {"USE_CUDNN", TVM_INFO_USE_CUDNN},
      {"USE_CUBLAS", TVM_INFO_USE_CUBLAS},
      {"USE_THRUST", TVM_INFO_USE_THRUST},
      {"USE_MIOPEN", TVM_INFO_USE_MIOPEN},
      {"USE_ROCBLAS", TVM_INFO_USE_ROCBLAS},
      {"USE_SORT", TVM_INFO_USE_SORT},
      {"USE_NNPACK", TVM_INFO_USE_NNPACK},
      {"USE_RANDOM", TVM_INFO_USE_RANDOM},
      {"USE_MICRO_STANDALONE_RUNTIME", TVM_INFO_USE_MICRO_STANDALONE_RUNTIME},
      {"USE_CPP_RPC", TVM_INFO_USE_CPP_RPC},
      {"USE_TFLITE", TVM_INFO_USE_TFLITE},
      {"USE_TENSORFLOW_PATH", TVM_INFO_USE_TENSORFLOW_PATH},
      {"USE_COREML", TVM_INFO_USE_COREML},
      {"USE_TARGET_ONNX", TVM_INFO_USE_TARGET_ONNX},
      {"USE_ARM_COMPUTE_LIB", TVM_INFO_USE_ARM_COMPUTE_LIB},
      {"USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR", TVM_INFO_USE_ARM_COMPUTE_LIB_GRAPH_EXECUTOR},
      {"INDEX_DEFAULT_I64", TVM_INFO_INDEX_DEFAULT_I64},
      {"TVM_CXX_COMPILER_PATH", TVM_CXX_COMPILER_PATH}};
  return result;
}

TVM_REGISTER_GLOBAL("support.GetLibInfo").set_body_typed(GetLibInfo);

}  // namespace tvm
