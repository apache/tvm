/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_common.h
 * \brief Common utilities for CUDA
 */
#ifndef TVM_RUNTIME_CUDA_CUDA_COMMON_H_
#define TVM_RUNTIME_CUDA_CUDA_COMMON_H_

#include <tvm/runtime/config.h>
#include <tvm/runtime/packed_func.h>
#include <string>

#if TVM_CUDA_RUNTIME
#include <cuda_runtime.h>

namespace tvm {
namespace runtime {

#define CUDA_DRIVER_CALL(x)                                             \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS) {                                       \
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


/*!
 * \brief Compile code into ptx using NVRTC
 * \param code The cuda code.
 * \return The PTX code.
 */
std::string NVRTCCompile(const std::string& code);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_CUDA_RUNTIME
#endif  // TVM_RUNTIME_CUDA_CUDA_COMMON_H_
