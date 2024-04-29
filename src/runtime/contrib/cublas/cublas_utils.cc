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
#include "cublas_utils.h"

#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>

#include "../../cuda/cuda_common.h"

namespace tvm {
namespace contrib {

CuBlasThreadEntry::CuBlasThreadEntry() { CHECK_CUBLAS_ERROR(cublasCreate(&handle)); }

CuBlasThreadEntry::~CuBlasThreadEntry() {
  if (handle) {
    cublasDestroy(handle);
    handle = nullptr;
  }
}

typedef dmlc::ThreadLocalStore<CuBlasThreadEntry> CuBlasThreadStore;

CuBlasThreadEntry* CuBlasThreadEntry::ThreadLocal() {
  auto stream = runtime::CUDAThreadEntry::ThreadLocal()->stream;
  CuBlasThreadEntry* retval = CuBlasThreadStore::Get();
  CHECK_CUBLAS_ERROR(cublasSetStream(retval->handle, static_cast<cudaStream_t>(stream)));
  return retval;
}

CuBlasLtThreadEntry::CuBlasLtThreadEntry() {
  CHECK_CUBLAS_ERROR(cublasLtCreate(&handle));
  CHECK_CUBLAS_ERROR(cublasLtMatmulPreferenceCreate(&matmul_pref_desc));
  CUDA_CALL(cudaMalloc(&workspace_ptr, workspace_size));
}

CuBlasLtThreadEntry::~CuBlasLtThreadEntry() {
  if (handle) {
    cublasLtDestroy(handle);
    handle = nullptr;
  }
  if (matmul_pref_desc) {
    cublasLtMatmulPreferenceDestroy(matmul_pref_desc);
    matmul_pref_desc = nullptr;
  }
  if (workspace_ptr != nullptr) {
    cudaFree(workspace_ptr);
    workspace_ptr = nullptr;
  }
}

typedef dmlc::ThreadLocalStore<CuBlasLtThreadEntry> CuBlasLtThreadStore;

CuBlasLtThreadEntry* CuBlasLtThreadEntry::ThreadLocal() { return CuBlasLtThreadStore::Get(); }

}  // namespace contrib
}  // namespace tvm
