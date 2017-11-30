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

 private:
  std::mt19937 rnd_engine_;
  unsigned rseed_;

};

struct RandomThreadLocalEntry {
  RandomEngine rnd_engine;
  static RandomThreadLocalEntry* ThreadLocal();
};

typedef dmlc::ThreadLocalStore<RandomThreadLocalEntry> RandomThreadLocalStore;

RandomThreadLocalEntry* RandomThreadLocalEntry::ThreadLocal() {
  return RandomThreadLocalStore::Get();
}


// matrix multiplication for row major
TVM_REGISTER_GLOBAL("tvm.contrib.random.randint")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    RandomThreadLocalEntry *entry = RandomThreadLocalEntry::ThreadLocal();
    int low = args[0];
    int high = args[1];
    DLTensor* out = args[2];
    CHECK_GT(high, low) << "high must be bigger than low in randint";
    CHECK(out->strides == nullptr);

    int ndim = out->ndim;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
      size *= out->shape[i];
    }

    if (out->ctx.device_type == kDLCPU) {
      DLDataType dtype = out->dtype;
      CHECK(dtype.code == kDLInt && dtype.bits == 32 && dtype.lanes == 1)
        << "only support int32 for now";

      // file the data with random byte
      std::generate_n(static_cast<int32_t*>(out->data), size, [&] () {
        unsigned rint = entry->rnd_engine.GetRandInt();
        return low + rint % (high - low);
      });
    } else {
      LOG(FATAL) << "Do not support random.randint on this device yet";
    }
  });


}  // namespace contrib
}  // namespace tvm
