/*!
 *  Copyright (c) 2016 by Contributors
 * \file device_api_gpu.h
 * \brief GPU specific API
 */
#ifndef TVM_RUNTIME_DEVICE_API_CPU_H_
#define TVM_RUNTIME_DEVICE_API_CPU_H_

#include <dmlc/logging.h>
#include <cstdlib>
#include <cstring>
#include "./device_api.h"

namespace tvm {
namespace runtime {

template<>
void* AllocDataSpace<kCPU>(TVMContext ctx, size_t size, size_t alignment) {
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

template<>
void FreeDataSpace<kCPU>(TVMContext ctx, void* ptr) {
#if _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

template<>
void CopyDataFromTo<kCPU>(const void* from,
                          void* to,
                          size_t size,
                          TVMContext ctx_from,
                          TVMContext ctx_to,
                          TVMStreamHandle stream) {
  memcpy(to, from, size);
}

template<>
void StreamSync<kCPU>(TVMContext ctx, TVMStreamHandle stream) {
}
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DEVICE_API_CPU_H_
