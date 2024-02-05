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

/*!
 * \file Use external cudnn utils function
 */

#ifndef TVM_RUNTIME_CONTRIB_CUBLAS_CUBLAS_UTILS_H_
#define TVM_RUNTIME_CONTRIB_CUBLAS_CUBLAS_UTILS_H_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/logging.h>

#include <cstdint>
#if CUDART_VERSION >= 10010
#include <cublasLt.h>
#endif  // CUDART_VERSION >= 10010

namespace tvm {
namespace contrib {

inline const char* GetCublasErrorString(int error) {
  switch (error) {
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "Unrecognized error";
}

#ifndef CHECK_CUBLAS_ERROR
#define CHECK_CUBLAS_ERROR(fn)                                                            \
  do {                                                                                    \
    int error = static_cast<int>(fn);                                                     \
    ICHECK_EQ(error, CUBLAS_STATUS_SUCCESS) << "CUBLAS: " << GetCublasErrorString(error); \
  } while (0)  // ; intentionally left off.
#endif         // CHECK_CUBLAS_ERROR

struct CuBlasThreadEntry {
  CuBlasThreadEntry();
  ~CuBlasThreadEntry();
  cublasHandle_t handle{nullptr};
  static CuBlasThreadEntry* ThreadLocal();
};  // CuBlasThreadEntry

struct CuBlasLtThreadEntry {
  CuBlasLtThreadEntry();
  ~CuBlasLtThreadEntry();

  cublasLtHandle_t handle{nullptr};
  cublasLtMatmulPreference_t matmul_pref_desc{nullptr};
  void* workspace_ptr{nullptr};
  // 32MB workspace as suggested by NVIDIA
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace.
  static constexpr const size_t workspace_size = 33554432;

  static CuBlasLtThreadEntry* ThreadLocal();
};  // CuBlasLtThreadEntry

inline cudaDataType_t GetCudaDataType(DLDataType type) {
  if (type.code == kDLInt) {
    switch (type.bits) {
      case 8:
        return CUDA_R_8I;
      case 32:
        return CUDA_R_32I;
    }
  } else if (type.code == kDLUInt) {
    switch (type.bits) {
      case 8:
        return CUDA_R_8U;
      case 32:
        return CUDA_R_32U;
    }
  } else if (type.code == kDLFloat) {
    switch (type.bits) {
      case 16:
        return CUDA_R_16F;
      case 32:
        return CUDA_R_32F;
      case 64:
        return CUDA_R_64F;
    }
  }
  LOG(FATAL) << "Unsupported cuda type";
}

/*! \brief Execute matrix multiply followed by the specified epilogue, using cuBLASLt. */
void CallCublasLt(cublasLtHandle_t hdl, cudaStream_t stream,
                  cublasLtMatmulPreference_t matmul_pref_desc, const DLTensor* A, const DLTensor* B,
                  const DLTensor* bias, const DLTensor* C, bool transa, bool transb,
                  void* workspace_ptr, size_t workspace_size,
                  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT);

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_CUBLAS_CUBLAS_UTILS_H_
