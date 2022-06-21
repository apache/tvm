/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file External random functions for tensor.
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/threading_backend.h>

#include <algorithm>

#include "mt_random_engine.cc"

#define DLPACK_INTEGER_TYPE_SWITCH(type, DType, ...)    \
  if (type.code == kDLInt && type.bits == 32) {         \
    typedef int32_t DType;                              \
    { __VA_ARGS__ }                                     \
  } else if (type.code == kDLInt && type.bits == 16) {  \
    typedef int16_t DType;                              \
    { __VA_ARGS__ }                                     \
  } else if (type.code == kDLInt && type.bits == 8) {   \
    typedef int8_t DType;                               \
    { __VA_ARGS__ }                                     \
  } else if (type.code == kDLUInt && type.bits == 32) { \
    typedef uint32_t DType;                             \
    { __VA_ARGS__ }                                     \
  } else if (type.code == kDLUInt && type.bits == 16) { \
    typedef uint16_t DType;                             \
    { __VA_ARGS__ }                                     \
  } else if (type.code == kDLUInt && type.bits == 8) {  \
    typedef uint8_t DType;                              \
    { __VA_ARGS__ }                                     \
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

TVM_REGISTER_GLOBAL("tvm.contrib.random.randint").set_body([](TVMArgs args, TVMRetValue* ret) {
  RandomThreadLocalEntry* entry = RandomThreadLocalEntry::ThreadLocal();
  int64_t low = args[0];
  int64_t high = args[1];
  DLTensor* out = args[2];
  ICHECK_GT(high, low) << "high must be bigger than low";
  ICHECK(out->strides == nullptr);

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

    if (out->device.device_type == kDLCPU) {
      // file the data with random byte
      std::generate_n(static_cast<DType*>(out->data), size, [&]() {
        unsigned rint = entry->random_engine.GetRandInt();
        return low + rint % (high - low);
      });
    } else {
      LOG(FATAL) << "Do not support random.randint on this device yet";
    }
  })
});

TVM_REGISTER_GLOBAL("tvm.contrib.random.uniform").set_body([](TVMArgs args, TVMRetValue* ret) {
  RandomThreadLocalEntry* entry = RandomThreadLocalEntry::ThreadLocal();
  double low = args[0];
  double high = args[1];
  DLTensor* out = args[2];
  entry->random_engine.SampleUniform(out, low, high);
});

TVM_REGISTER_GLOBAL("tvm.contrib.random.normal").set_body([](TVMArgs args, TVMRetValue* ret) {
  RandomThreadLocalEntry* entry = RandomThreadLocalEntry::ThreadLocal();
  double loc = args[0];
  double scale = args[1];
  DLTensor* out = args[2];
  entry->random_engine.SampleNormal(out, loc, scale);
});

TVM_REGISTER_GLOBAL("tvm.contrib.random.random_fill").set_body([](TVMArgs args, TVMRetValue* ret) {
  RandomThreadLocalEntry* entry = RandomThreadLocalEntry::ThreadLocal();
  DLTensor* out = args[0];
  entry->random_engine.RandomFill(out);
});

TVM_REGISTER_GLOBAL("tvm.contrib.random.random_fill_for_measure")
    .set_body([](TVMArgs args, TVMRetValue* ret) -> void {
      static const PackedFunc* curand = Registry::Get("runtime.contrib.curand.RandomFill");
      DLTensor* out = args[0];
      if (curand && out->device.device_type == DLDeviceType::kDLCUDA) {
        if (out->dtype.code == DLDataTypeCode::kDLFloat) {
          (*curand)(out);
          return;
        }
      }
      RandomThreadLocalEntry* entry = RandomThreadLocalEntry::ThreadLocal();
      entry->random_engine.RandomFillForMeasure(out);
    });

}  // namespace contrib
}  // namespace tvm
