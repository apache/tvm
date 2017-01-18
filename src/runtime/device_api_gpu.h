/*!
 *  Copyright (c) 2016 by Contributors
 * \file device_api_gpu.h
 * \brief GPU specific API
 */
#ifndef TVM_RUNTIME_DEVICE_API_GPU_H_
#define TVM_RUNTIME_DEVICE_API_GPU_H_

#include <dmlc/logging.h>
#include "./device_api.h"

#if TVM_CUDA_RUNTIME
#include <cuda_runtime.h>

namespace tvm {
namespace runtime {

/*!
 * \brief Protected CUDA call.
 * \param func Expression to call.
 *
 * It checks for CUDA errors after invocation of the expression.
 */
#define CUDA_CALL(func)                                            \
  {                                                                \
    cudaError_t e = (func);                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

template<>
inline void* AllocDataSpace<kGPU>(TVMContext ctx, size_t size, size_t alignment) {
  CUDA_CALL(cudaSetDevice(ctx.dev_id));
  CHECK_EQ(256 % alignment, 0U)
      << "CUDA space is aligned at 256 bytes";
  void *ret;
  CUDA_CALL(cudaMalloc(&ret, size));
  return ret;
}

template<>
inline void FreeDataSpace<kGPU>(TVMContext ctx, void* ptr) {
  CUDA_CALL(cudaSetDevice(ctx.dev_id));
  CUDA_CALL(cudaFree(ptr));
}

inline void GPUCopy(const void* from,
                    void* to,
                    size_t size,
                    cudaMemcpyKind kind,
                    cudaStream_t stream) {
  if (stream != 0) {
    CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
  } else {
    CUDA_CALL(cudaMemcpy(to, from, size, kind));
  }
}

template<>
inline void CopyDataFromTo<kGPU>(const void* from,
                                 void* to,
                                 size_t size,
                                 TVMContext ctx_from,
                                 TVMContext ctx_to,
                                 TVMStreamHandle stream) {
  cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
  if (ctx_from.dev_mask == kGPU && ctx_to.dev_mask == kGPU) {
    CUDA_CALL(cudaSetDevice(ctx_from.dev_id));
    if (ctx_from.dev_id == ctx_to.dev_id) {
      GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
    } else {
      cudaMemcpyPeerAsync(to, ctx_to.dev_id,
                          from, ctx_from.dev_id,
                          size, cu_stream);
    }
  } else if (ctx_from.dev_mask == kGPU && ctx_to.dev_mask == kCPU) {
    CUDA_CALL(cudaSetDevice(ctx_from.dev_id));
    GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
  } else if (ctx_from.dev_mask == kCPU && ctx_to.dev_mask == kGPU) {
    CUDA_CALL(cudaSetDevice(ctx_to.dev_id));
    GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
  } else {
    LOG(FATAL) << "expect copy from/to GPU or between GPU";
  }
}

template<>
inline void StreamSync<kGPU>(TVMContext ctx, TVMStreamHandle stream) {
  CUDA_CALL(cudaSetDevice(ctx.dev_id));
  CUDA_CALL(cudaStreamSynchronize(
      static_cast<cudaStream_t>(stream)));
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_CUDA_RUNTIME
#endif  // TVM_RUNTIME_DEVICE_API_GPU_H_
