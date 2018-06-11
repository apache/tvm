/*!
 *  Copyright (c) 2017 by Contributors
 * \file rocm_common.h
 * \brief Common utilities for ROCM
 */
#ifndef TVM_RUNTIME_ROCM_ROCM_COMMON_H_
#define TVM_RUNTIME_ROCM_ROCM_COMMON_H_

#include <tvm/runtime/packed_func.h>
#include <hip/hip_runtime_api.h>
#include <string>
#include "../workspace_pool.h"

namespace tvm {
namespace runtime {

#define ROCM_DRIVER_CALL(x)                                             \
  {                                                                     \
    hipError_t result = x;                                              \
    if (result != hipSuccess && result != hipErrorDeinitialized) {      \
      LOG(FATAL)                                                        \
          << "ROCM HIP Error: " #x " failed with error: " << hipGetErrorString(result); \
    }                                                                   \
  }

#define ROCM_CALL(func)                                               \
  {                                                                   \
    hipError_t e = (func);                                            \
    CHECK(e == hipSuccess)                                            \
        << "ROCM HIP: " << hipGetErrorString(e);                      \
  }

/*! \brief Thread local workspace */
class ROCMThreadEntry {
 public:
  /*! \brief The hip stream */
  hipStream_t stream{nullptr};
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  ROCMThreadEntry();
  // get the threadlocal workspace
  static ROCMThreadEntry* ThreadLocal();
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_ROCM_ROCM_COMMON_H_
