/*!
 *  Copyright (c) 2017 by Contributors
 * \file External random functions for tensor.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <algorithm>
#ifndef _LIBCPP_SGX_CONFIG
#include "mt_random_engine.cc"
#else
#include "sgx_random_engine.cc"
#endif

#define DLPACK_INTEGER_TYPE_SWITCH(type, DType, ...)    \
  if (type.code == kDLInt && type.bits == 32) {         \
    typedef int32_t DType;                              \
    {__VA_ARGS__}                                       \
  } else if (type.code == kDLInt && type.bits == 16) {  \
    typedef int16_t DType;                              \
    {__VA_ARGS__}                                       \
  } else if (type.code == kDLInt && type.bits == 8) {   \
    typedef int8_t DType;                               \
    {__VA_ARGS__}                                       \
  } else if (type.code == kDLUInt && type.bits == 32) { \
    typedef uint32_t DType;                             \
    {__VA_ARGS__}                                       \
  } else if (type.code == kDLUInt && type.bits == 16) { \
    typedef uint16_t DType;                             \
    {__VA_ARGS__}                                       \
  } else if (type.code == kDLUInt && type.bits == 8) {  \
    typedef uint8_t DType;                              \
    {__VA_ARGS__}                                       \
  } else {                                              \
    LOG(FATAL) << "unknown data type";                  \
  }

namespace tvm {
namespace contrib {

using namespace runtime;

struct RandomThreadLocalEntry {
  RandomEngine random_engine;
  static RandomThreadLocalEntry* ThreadLocal();
};

typedef dmlc::ThreadLocalStore<RandomThreadLocalEntry> RandomThreadLocalStore;

RandomThreadLocalEntry* RandomThreadLocalEntry::ThreadLocal() {
  return RandomThreadLocalStore::Get();
}


TVM_REGISTER_GLOBAL("tvm.contrib.random.randint")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    RandomThreadLocalEntry *entry = RandomThreadLocalEntry::ThreadLocal();
    int64_t low = args[0];
    int64_t high = args[1];
    DLTensor* out = args[2];
    CHECK_GT(high, low) << "high must be bigger than low";
    CHECK(out->strides == nullptr);

    DLDataType dtype = out->dtype;
    int64_t size = 1;
    for (int i = 0; i < out->ndim; ++i) {
      size *= out->shape[i];
    }

    DLPACK_INTEGER_TYPE_SWITCH(dtype, DType, {
      int64_t numeric_low = std::numeric_limits<DType>::min();
      int64_t numeric_high = std::numeric_limits<DType>::max();
      numeric_high += 1;  // exclusive upper bound
      low = std::max(low, numeric_low);
      high = std::min(high, numeric_high);

      if (out->ctx.device_type == kDLCPU) {
          // file the data with random byte
          std::generate_n(static_cast<DType*>(out->data), size, [&] () {
            unsigned rint = entry->random_engine.GetRandInt();
            return low + rint % (high - low);
          });
      } else {
        LOG(FATAL) << "Do not support random.randint on this device yet";
      }
    })
  });


TVM_REGISTER_GLOBAL("tvm.contrib.random.uniform")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    RandomThreadLocalEntry *entry = RandomThreadLocalEntry::ThreadLocal();
    double low = args[0];
    double high = args[1];
    DLTensor* out = args[2];
    entry->random_engine.SampleUniform(out, low, high);
  });


TVM_REGISTER_GLOBAL("tvm.contrib.random.normal")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    RandomThreadLocalEntry *entry = RandomThreadLocalEntry::ThreadLocal();
    double loc = args[0];
    double scale = args[1];
    DLTensor* out = args[2];
    entry->random_engine.SampleNormal(out, loc, scale);
  });


}  // namespace contrib
}  // namespace tvm
