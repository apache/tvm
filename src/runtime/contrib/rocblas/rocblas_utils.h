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

#ifndef TVM_RUNTIME_CONTRIB_ROCBLAS_ROCBLAS_UTILS_H_
#define TVM_RUNTIME_CONTRIB_ROCBLAS_ROCBLAS_UTILS_H_

#include <rocblas/rocblas.h>
#include <rocblas/internal/rocblas-beta.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <dlpack/dlpack.h>
#include <tvm/runtime/logging.h>


namespace tvm {
namespace contrib {

typedef struct rocblas_half
{
    uint16_t data;
} rocblas_half;

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_ROCBLAS_ERROR
#define CHECK_ROCBLAS_ERROR(error)                                                                \
  if (error != rocblas_status_success) {                                                          \
    fprintf(stderr, "rocBLAS error: ");                                                           \
    if (error == rocblas_status_invalid_handle) fprintf(stderr, "rocblas_status_invalid_handle"); \
    if (error == rocblas_status_not_implemented)                                                  \
      fprintf(stderr, " rocblas_status_not_implemented");                                         \
    if (error == rocblas_status_invalid_pointer)                                                  \
      fprintf(stderr, "rocblas_status_invalid_pointer");                                          \
    if (error == rocblas_status_invalid_size) fprintf(stderr, "rocblas_status_invalid_size");     \
    if (error == rocblas_status_memory_error) fprintf(stderr, "rocblas_status_memory_error");     \
    if (error == rocblas_status_internal_error) fprintf(stderr, "rocblas_status_internal_error"); \
    fprintf(stderr, "\n");                                                                        \
    exit(EXIT_FAILURE);                                                                           \
  }
#endif

#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                                      \
    if(error != HIPBLAS_STATUS_SUCCESS)                                                   \
    {                                                                                     \
        fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__); \
        fprintf(stderr, "\n");                                                            \
        exit(EXIT_FAILURE);                                                               \
    }
#endif

struct RocBlasThreadEntry {
  RocBlasThreadEntry() { CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle)); }

  ~RocBlasThreadEntry() {
    if (handle) {
      CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));
      handle = nullptr;
    }
  }

  rocblas_handle handle;
  static RocBlasThreadEntry* ThreadLocal();
};  // RocBlasThreadEntry

struct HipBlasLtThreadEntry {
  HipBlasLtThreadEntry() { CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle)); }

  ~HipBlasLtThreadEntry() {
    if (handle) {
      CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
      handle = nullptr;
    }
  }

  hipblasLtHandle_t handle;
  static HipBlasLtThreadEntry* ThreadLocal();
};  // HipBlasLtThreadEntry

typedef dmlc::ThreadLocalStore<RocBlasThreadEntry> RocBlasThreadStore;

typedef dmlc::ThreadLocalStore<HipBlasLtThreadEntry> HipBlasLtThreadStore;


void CallHipblasLt(hipblasLtHandle_t hdl, hipStream_t stream, const DLTensor* A, const DLTensor* B,
                  const DLTensor* bias, const DLTensor* C, bool transa, bool transb,
                  hipblasLtEpilogue_t epilogue);

void CallRocblas(rocblas_handle hdl, hipStream_t stream, const DLTensor* A, const DLTensor* B,
                  const DLTensor* bias, const DLTensor* C, bool transa, bool transb,
                  hipblasLtEpilogue_t epilogue, size_t algo);

void CallRocblasBatch(rocblas_handle hdl, hipStream_t stream, const DLTensor* A, const DLTensor* B,
                  const DLTensor* bias, const DLTensor* C, bool transa, bool transb,
                  hipblasLtEpilogue_t epilogue, size_t algo_batch);

int TuneRocblas(rocblas_handle hdl, hipStream_t stream, const DLTensor* A, const DLTensor* B,
                  const DLTensor* bias, const DLTensor* C, bool transa, bool transb,
                  hipblasLtEpilogue_t epilogue);

int TuneRocblasBatch(rocblas_handle hdl, hipStream_t stream, const DLTensor* A, const DLTensor* B,
                  const DLTensor* bias, const DLTensor* C, bool transa, bool transb,
                  hipblasLtEpilogue_t epilogue);

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_ROCBLAS_ROCBLAS_UTILS_H_
