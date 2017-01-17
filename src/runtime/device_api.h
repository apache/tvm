/*!
 *  Copyright (c) 2016 by Contributors
 * \file device_api.hx
 * \brief Device specific API
 */
#ifndef TVM_RUNTIME_DEVICE_API_H_
#define TVM_RUNTIME_DEVICE_API_H_

#include <tvm/base.h>
#include <tvm/c_runtime_api.h>

namespace tvm {
namespace runtime {
/*!
 * \brief Whether ctx is enabled.
 * \param ctx The device context to perform operation.
 * \tparam xpu The device mask.
 */
template<TVMDeviceMask xpu>
inline bool CheckEnabled(TVMContext ctx) {
  return true;
}

/*!
 * \brief Allocate a data space on device.
 * \param ctx The device context to perform operation.
 * \param size The size of the memory
 * \param alignment The alignment of the memory.
 * \return The allocated device pointer
 * \tparam xpu The device mask.
 */
template<TVMDeviceMask xpu>
inline void* AllocDataSpace(TVMContext ctx, size_t size, size_t alignment);

/*!
 * \brief Free a data space on device.
 * \param ctx The device context to perform operation.
 * \param ptr The data space.
 * \tparam xpu The device mask.
 */
template<TVMDeviceMask xpu>
inline void FreeDataSpace(TVMContext ctx, void* ptr);

/*!
 * \brief copy data from one place to another
 * \param dev The device to perform operation.
 * \param from The source array.
 * \param to The target array.
 * \param size The size of the memory
 * \param ctx_from The source context
 * \param ctx_to The target context
 * \tparam xpu The device mask.
 */
template<TVMDeviceMask xpu>
inline void CopyDataFromTo(const void* from,
                           void* to,
                           size_t size,
                           TVMContext ctx_from,
                           TVMContext ctx_to,
                           TVMStreamHandle stream);
/*!
 * \brief Synchronize the stream
 * \param ctx The context to perform operation.
 * \param stream The stream to be sync.
 * \tparam xpu The device mask.
 */
template<TVMDeviceMask xpu>
inline void StreamSync(TVMContext ctx, TVMStreamHandle stream);

// macro to run cuda related code
#if TVM_CUDA_RUNTIME
#define TVM_RUN_CUDA(OP) { const TVMDeviceMask xpu = kGPU; OP; }
#else
#define TVM_RUN_CUDA(OP) LOG(FATAL) << "CUDA is not enabled";
#endif

// macro to run opencl related code
#if TVM_OPENCL_RUNTIME
#define TVM_RUN_OPENCL(OP) { const TVMDeviceMask xpu = kOpenCL; OP; }
#else
#define TVM_RUN_OPENCL(OP) LOG(FATAL) << "OpenCL is not enabled";
#endif

// macro to switch options between devices
#define TVM_DEVICE_SWITCH(ctx, OP)                                  \
  switch (ctx.dev_mask) {                                           \
    case kCPU: { const TVMDeviceMask xpu = kCPU; OP; break; }       \
    case kGPU: TVM_RUN_CUDA(OP); break;                             \
    case kOpenCL: TVM_RUN_OPENCL(OP); break;                        \
    default: LOG(FATAL) << "unknown device_mask " << ctx.dev_mask;  \
  }

}  // namespace runtime
}  // namespace tvm

#include "./device_api_gpu.h"
#include "./device_api_cpu.h"

#endif  // TVM_RUNTIME_DEVICE_API_H_
