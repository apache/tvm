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

/*!
 * \brief Automatically detect and set cuda device.
 * \param args The arguments.
 */
inline void AutoSetCUDADevice(const TVMArgs& args) {
  int dev_id = -1;
  for (int i = 0; i < args.size(); ++i) {
    if (args.type_codes[i] == kArrayHandle) {
      TVMContext ctx = static_cast<TVMArray*>(
          args.values[i].v_handle)->ctx;
      CHECK_EQ(ctx.dev_mask, kGPU)
          << "All operands need to be GPU";
      if (dev_id == -1) {
        dev_id = ctx.dev_id;
      } else {
        CHECK_EQ(dev_id, ctx.dev_id)
            << "Operands comes from different devices ";
      }
    }
  }
  CUDA_CALL(cudaSetDevice(dev_id));
}
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_CUDA_RUNTIME
#endif  // TVM_RUNTIME_CUDA_CUDA_COMMON_H_
