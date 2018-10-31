/*!
 *  Copyright (c) 2018 by Contributors
 * \file Use external cudnn utils function
 */
#include "cublas_utils.h"
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include "../../runtime/cuda/cuda_common.h"

namespace tvm {
namespace contrib {


CuBlasThreadEntry::CuBlasThreadEntry() {
  CHECK_CUBLAS_ERROR(cublasCreate(&handle));
}


CuBlasThreadEntry::~CuBlasThreadEntry() {
  if (handle) {
    cublasDestroy(handle);
    handle = 0;
  }
}


typedef dmlc::ThreadLocalStore<CuBlasThreadEntry> CuBlasThreadStore;


CuBlasThreadEntry* CuBlasThreadEntry::ThreadLocal() {
  auto stream = runtime::CUDAThreadEntry::ThreadLocal()->stream;
  CuBlasThreadEntry* retval = CuBlasThreadStore::Get();
  CHECK_CUBLAS_ERROR(cublasSetStream(retval->handle, static_cast<cudaStream_t>(stream)));
  return retval;
}


}  // namespace contrib
}  // namespace tvm
