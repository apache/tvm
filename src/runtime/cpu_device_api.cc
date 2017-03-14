/*!
 *  Copyright (c) 2016 by Contributors
 * \file device_api_gpu.h
 * \brief GPU specific API
 */
#ifndef TVM_RUNTIME_DEVICE_API_CPU_H_
#define TVM_RUNTIME_DEVICE_API_CPU_H_

#include <dmlc/logging.h>
#include <tvm/runtime/registry.h>
#include <cstdlib>
#include <cstring>
#include "./device_api.h"

namespace tvm {
namespace runtime {

class CPUDeviceAPI : public DeviceAPI {
 public:
  void* AllocDataSpace(TVMContext ctx, size_t size, size_t alignment) final {
    void* ptr;
#if _MSC_VER
    ptr = _aligned_malloc(size, alignment);
    if (ptr == nullptr) throw std::bad_alloc();
#else
    int ret = posix_memalign(&ptr, alignment, size);
    if (ret != 0) throw std::bad_alloc();
#endif
    return ptr;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

  void CopyDataFromTo(const void* from,
                      void* to,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMStreamHandle stream) final {
    memcpy(to, from, size);
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
  }
};

TVM_REGISTER_GLOBAL(_device_api_cpu)
.set_body([](TVMArgs args, TVMRetValue* rv) {
    static CPUDeviceAPI inst;
    DeviceAPI* ptr = &inst;
    *rv = static_cast<void*>(ptr);
  });
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DEVICE_API_CPU_H_
