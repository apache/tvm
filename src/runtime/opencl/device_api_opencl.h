/*!
 *  Copyright (c) 2017 by Contributors
 * \file device_api_opencl.h
 * \brief OpenCL specific API
 */
#ifndef TVM_RUNTIME_OPENCL_DEVICE_API_OPENCL_H_
#define TVM_RUNTIME_OPENCL_DEVICE_API_OPENCL_H_

#include <tvm/runtime/config.h>

#if TVM_OPENCL_RUNTIME
#include <string>
#include <vector>
#include "./opencl_common.h"

namespace tvm {
namespace runtime {

template<>
inline void* AllocDataSpace<kOpenCL>(TVMContext ctx, size_t size, size_t alignment) {
  cl::OpenCLWorkspace* w = cl::OpenCLWorkspace::Global();
  cl_int err_code;
  cl_mem mptr = clCreateBuffer(
      w->context, CL_MEM_READ_WRITE, size, nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  return mptr;
}

template<>
inline void FreeDataSpace<kOpenCL>(TVMContext ctx, void* ptr) {
  cl_mem mptr = static_cast<cl_mem>(ptr);
  OPENCL_CALL(clReleaseMemObject(mptr));
}

template<>
inline void CopyDataFromTo<kOpenCL>(const void* from,
                                    void* to,
                                    size_t size,
                                    TVMContext ctx_from,
                                    TVMContext ctx_to,
                                    TVMStreamHandle stream) {
  CHECK(stream == nullptr);
  cl::OpenCLWorkspace* w = cl::OpenCLWorkspace::Global();
  if (ctx_from.dev_mask == kOpenCL && ctx_to.dev_mask == kOpenCL) {
    OPENCL_CALL(clEnqueueCopyBuffer(
        w->GetQueue(ctx_to),
        static_cast<cl_mem>((void*)from),  // NOLINT(*)
        static_cast<cl_mem>(to),
        0, 0, size, 0, nullptr, nullptr));
  } else if (ctx_from.dev_mask == kOpenCL && ctx_to.dev_mask == kCPU) {
    OPENCL_CALL(clEnqueueReadBuffer(
        w->GetQueue(ctx_from),
        static_cast<cl_mem>((void*)from),  // NOLINT(*)
        CL_FALSE, 0, size, to,
        0, nullptr, nullptr));
    OPENCL_CALL(clFinish(w->GetQueue(ctx_from)));
  } else if (ctx_from.dev_mask == kCPU && ctx_to.dev_mask == kOpenCL) {
    OPENCL_CALL(clEnqueueWriteBuffer(
        w->GetQueue(ctx_to),
        static_cast<cl_mem>(to),
        CL_FALSE, 0, size, from,
        0, nullptr, nullptr));
    OPENCL_CALL(clFinish(w->GetQueue(ctx_to)));
  } else {
    LOG(FATAL) << "Expect copy from/to GPU or between GPU";
  }
}

template<>
inline void StreamSync<kOpenCL>(TVMContext ctx, TVMStreamHandle stream) {
  CHECK(stream == nullptr);
  cl::OpenCLWorkspace* w = cl::OpenCLWorkspace::Global();
  OPENCL_CALL(clFinish(w->GetQueue(ctx)));
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_OPENCL_RUNTIME
#endif  // TVM_RUNTIME_OPENCL_DEVICE_API_OPENCL_H_
