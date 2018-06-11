/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_common.h
 * \brief Common utilities for CUDA
 */
#ifndef TVM_RUNTIME_CUDA_CUDA_COMMON_H_
#define TVM_RUNTIME_CUDA_CUDA_COMMON_H_

#include <cuda_runtime.h>
#include <tvm/runtime/packed_func.h>
#include <string>
#include "../workspace_pool.h"

namespace tvm {
namespace runtime {

#define CUDA_DRIVER_CALL(x)                                             \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
      const char *msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      LOG(FATAL)                                                        \
          << "CUDAError: " #x " failed with error: " << msg;            \
    }                                                                   \
  }

#define CUDA_CALL(func)                                            \
  {                                                                \
    cudaError_t e = (func);                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

/*! \brief Thread local workspace */
class CUDAThreadEntry {
 public:
  /*! \brief The cuda stream */
  cudaStream_t stream{nullptr};
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  CUDAThreadEntry();
  // get the threadlocal workspace
  static CUDAThreadEntry* ThreadLocal();
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CUDA_CUDA_COMMON_H_
