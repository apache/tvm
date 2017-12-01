/*!
 *  Copyright (c) 2017 by Contributors
 * \file External random functions for tensor.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <algorithm>
#include <random>
#include <ctime>

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

class RandomEngine {
 public:
  RandomEngine() {
    this->Seed(time(0));
  }
  explicit RandomEngine(int seed) {
    this->Seed(seed);
  }

  ~RandomEngine() {}

  inline void Seed(int seed) {
    rnd_engine_.seed(seed);
    this->rseed_ = static_cast<unsigned>(seed);
  }

  inline unsigned GetSeed() const {
    return rseed_;
  }

  inline unsigned GetRandInt() {
    return rnd_engine_();
  }

  void SampleUniform(DLTensor* data, float low, float high) {
    CHECK_GT(high, low) << "high must be bigger than low";
    CHECK(data->strides == nullptr);

    DLDataType dtype = data->dtype;
    int64_t size = 1;
    for (int i = 0; i < data->ndim; ++i) {
      size *= data->shape[i];
    }

    CHECK(dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1);

    if (data->ctx.device_type == kDLCPU) {
      std::uniform_real_distribution<float> uniform_dist(low, high);
      std::generate_n(static_cast<float*>(data->data), size, [&] () {
        return uniform_dist(rnd_engine_);
      });
    } else {
      LOG(FATAL) << "Do not support random.randint on this device yet";
    }
  }

 private:
  std::mt19937 rnd_engine_;
  unsigned rseed_;
};

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


}  // namespace contrib
}  // namespace tvm
