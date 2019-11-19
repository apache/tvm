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
#include "cusparse_utils.h"
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include "../../cuda/cuda_common.h"

namespace tvm {
namespace contrib {


CuSparseThreadEntry::CuSparseThreadEntry() {
  CHECK_CUSPARSE_ERROR(cusparseCreate(&handle));
  CHECK_CUSPARSE_ERROR(cusparseCreateMatDescr(&descr));
  CHECK_CUSPARSE_ERROR(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  CHECK_CUSPARSE_ERROR(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
}


CuSparseThreadEntry::~CuSparseThreadEntry() {
  if (handle) {
    cusparseDestroy(handle);
    handle = 0;
  }
  if (descr) {
    cusparseDestroyMatDescr(descr);
    descr = 0;
  }
}


typedef dmlc::ThreadLocalStore<CuSparseThreadEntry> CuSparseThreadStore;

CuSparseThreadEntry* CuSparseThreadEntry::ThreadLocal() {
  auto stream = runtime::CUDAThreadEntry::ThreadLocal()->stream;
  CuSparseThreadEntry* retval = CuSparseThreadStore::Get();
  CHECK_CUSPARSE_ERROR(cusparseSetStream(retval->handle, static_cast<cudaStream_t>(stream)));
  return retval;
}


}  // namespace contrib
}  // namespace tvm
