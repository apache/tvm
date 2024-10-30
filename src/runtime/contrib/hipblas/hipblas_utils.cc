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
#include "hipblas_utils.h"

#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>

#include "../../rocm/rocm_common.h"

namespace tvm {
namespace contrib {

HipBlasThreadEntry::HipBlasThreadEntry() { CHECK_HIPBLAS_ERROR(hipblasCreate(&handle)); }

HipBlasThreadEntry::~HipBlasThreadEntry() {
  if (handle) {
    hipblasDestroy(handle);
    handle = nullptr;
  }
}

typedef dmlc::ThreadLocalStore<HipBlasThreadEntry> HipBlasThreadStore;

HipBlasThreadEntry* HipBlasThreadEntry::ThreadLocal() {
  auto stream = runtime::ROCMThreadEntry::ThreadLocal()->stream;
  HipBlasThreadEntry* retval = HipBlasThreadStore::Get();
  CHECK_HIPBLAS_ERROR(hipblasSetStream(retval->handle, static_cast<hipStream_t>(stream)));
  return retval;
}

HipBlasLtThreadEntry::HipBlasLtThreadEntry() {
  CHECK_HIPBLAS_ERROR(hipblasLtCreate(&handle));
  CHECK_HIPBLAS_ERROR(hipblasLtMatmulPreferenceCreate(&matmul_pref_desc));
  ROCM_CALL(hipMalloc(&workspace_ptr, workspace_size));
}

HipBlasLtThreadEntry::~HipBlasLtThreadEntry() {
  if (handle) {
    hipblasLtDestroy(handle);
    handle = nullptr;
  }
  if (matmul_pref_desc) {
    hipblasLtMatmulPreferenceDestroy(matmul_pref_desc);
    matmul_pref_desc = nullptr;
  }
  if (workspace_ptr != nullptr) {
    hipFree(workspace_ptr);
    workspace_ptr = nullptr;
  }
}

typedef dmlc::ThreadLocalStore<HipBlasLtThreadEntry> HipBlasLtThreadStore;

HipBlasLtThreadEntry* HipBlasLtThreadEntry::ThreadLocal() { return HipBlasLtThreadStore::Get(); }

}  // namespace contrib

}  // namespace tvm
