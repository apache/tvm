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
 * \file Use external hipblas utils function
 */
#ifndef TVM_RUNTIME_CONTRIB_HIPBLAS_HIPBLAS_UTILS_H_
#define TVM_RUNTIME_CONTRIB_HIPBLAS_HIPBLAS_UTILS_H_

#include <dlpack/dlpack.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <tvm/runtime/logging.h>

#include <cstdint>

namespace tvm {
namespace contrib {
inline const char* GetHipblasErrorString(int error) {
  switch (error) {
    case HIPBLAS_STATUS_NOT_INITIALIZED:
      return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:
      return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:
      return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_ARCH_MISMATCH:
      return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_MAPPING_ERROR:
      return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED:
      return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:
      return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED:
      return "HIPBLAS_STATUS_NOT_SUPPORTED";
  }
  return "Unrecognized error";
}

#ifndef CHECK_HIPBLAS_ERROR
#define CHECK_HIPBLAS_ERROR(fn)                                                              \
  do {                                                                                       \
    int error = static_cast<int>(fn);                                                        \
    ICHECK_EQ(error, HIPBLAS_STATUS_SUCCESS) << "HIPBLAS: " << GetHipblasErrorString(error); \
  } while (0)  // ; intentionally left off.
#endif         // CHECK_HIPBLAS_ERROR

struct HipBlasThreadEntry {
  HipBlasThreadEntry();
  ~HipBlasThreadEntry();
  hipblasHandle_t handle{nullptr};
  static HipBlasThreadEntry* ThreadLocal();
};  // HipBlasThreadEntry

struct HipBlasLtThreadEntry {
  HipBlasLtThreadEntry();
  ~HipBlasLtThreadEntry();

  hipblasLtHandle_t handle{nullptr};
  hipblasLtMatmulPreference_t matmul_pref_desc{nullptr};
  void* workspace_ptr{nullptr};
  // 32MB workspace as suggested by NVIDIA
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace.
  static constexpr const size_t workspace_size = 33554432;

  static HipBlasLtThreadEntry* ThreadLocal();
};  // HipBlasLtThreadEntry

inline hipDataType GetHipDataType(DLDataType type) {
  if (type.code == kDLInt) {
    switch (type.bits) {
      case 8:
        return HIP_R_8I;
      case 32:
        return HIP_R_32I;
    }
  } else if (type.code == kDLUInt) {
    switch (type.bits) {
      case 8:
        return HIP_R_8U;
      case 32:
        return HIP_R_32U;
    }
  } else if (type.code == kDLFloat) {
    switch (type.bits) {
      case 16:
        return HIP_R_16F;
      case 32:
        return HIP_R_32F;
      case 64:
        return HIP_R_64F;
    }
  }
  LOG(FATAL) << "Unsupported hip type";
}

inline hipblasDatatype_t GetHipBlasDataType(DLDataType type) {
  if (type.code == kDLInt) {
    switch (type.bits) {
      case 8:
        return HIPBLAS_R_8I;
      case 32:
        return HIPBLAS_R_32I;
    }
  } else if (type.code == kDLUInt) {
    switch (type.bits) {
      case 8:
        return HIPBLAS_R_8U;
      case 32:
        return HIPBLAS_R_32U;
    }
  } else if (type.code == kDLFloat) {
    switch (type.bits) {
      case 16:
        return HIPBLAS_R_16F;
      case 32:
        return HIPBLAS_R_32F;
      case 64:
        return HIPBLAS_R_64F;
    }
  }
  LOG(FATAL) << "Unsupported hip type";
}

/*! \brief Execute matrix multiply followed by the specified epilogue, using hipBLASLt. */
void CallHipblasLt(hipblasLtHandle_t hdl, hipStream_t stream,
                   hipblasLtMatmulPreference_t matmul_pref_desc, const DLTensor* A,
                   const DLTensor* B, const DLTensor* bias, const DLTensor* C, bool transa,
                   bool transb, void* workspace_ptr, size_t workspace_size,
                   hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT);

}  // namespace contrib

}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_HIPBLAS_HIPBLAS_UTILS_H_
