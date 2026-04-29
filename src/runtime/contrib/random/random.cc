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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>

#include <algorithm>
#include <cstdint>

#include "mt_random_engine.cc"

#define DLPACK_INTEGER_TYPE_SWITCH(type, DType, ...)     \
  if (type.code == kDLInt && type.bits == 32) {          \
    typedef int32_t DType;                               \
    {                                                    \
      __VA_ARGS__                                        \
    }                                                    \
  } else if (type.code == kDLInt && type.bits == 16) {   \
    typedef int16_t DType;                               \
    {                                                    \
      __VA_ARGS__                                        \
    }                                                    \
  } else if (type.code == kDLInt && type.bits == 8) {    \
    typedef int8_t DType;                                \
    {                                                    \
      __VA_ARGS__                                        \
    }                                                    \
  } else if (type.code == kDLUInt && type.bits == 32) {  \
    typedef uint32_t DType;                              \
    {                                                    \
      __VA_ARGS__                                        \
    }                                                    \
  } else if (type.code == kDLUInt && type.bits == 16) {  \
    typedef uint16_t DType;                              \
    {                                                    \
      __VA_ARGS__                                        \
    }                                                    \
  } else if (type.code == kDLUInt && type.bits == 8) {   \
    typedef uint8_t DType;                               \
    {                                                    \
      __VA_ARGS__                                        \
    }                                                    \
  } else {                                               \
    TVM_FFI_THROW(InternalError) << "unknown data type"; \
  }

namespace tvm {
namespace contrib {

using namespace runtime;

struct RandomThreadLocalEntry {
  RandomEngine random_engine;
  static RandomThreadLocalEntry* ThreadLocal();
};

RandomThreadLocalEntry* RandomThreadLocalEntry::ThreadLocal() {
  static thread_local RandomThreadLocalEntry inst;
  return &inst;
}

namespace {

unsigned CombineSeeds(int64_t seed, int64_t seed2) {
  uint64_t combined = (static_cast<uint64_t>(static_cast<uint32_t>(seed)) << 32) |
                      static_cast<uint32_t>(seed2);
  combined ^= combined >> 32;
  return static_cast<unsigned>(combined & 0xFFFFFFFFu);
}

RandomEngine* GetRandomEngineForArgs(const ffi::PackedArgs& args, int seed_idx, int seed2_idx) {
  if (args.size() <= seed2_idx) {
    return &RandomThreadLocalEntry::ThreadLocal()->random_engine;
  }

  int64_t seed = args[seed_idx].cast<int64_t>();
  int64_t seed2 = args[seed2_idx].cast<int64_t>();
  if (seed == 0 && seed2 == 0) {
    return &RandomThreadLocalEntry::ThreadLocal()->random_engine;
  }

  static thread_local RandomEngine seeded_engine;
  seeded_engine.Seed(CombineSeeds(seed, seed2));
  return &seeded_engine;
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("tvm.contrib.random.randint",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    RandomThreadLocalEntry* entry = RandomThreadLocalEntry::ThreadLocal();
                    int64_t low = args[0].cast<int64_t>();
                    int64_t high = args[1].cast<int64_t>();
                    auto out = args[2].cast<DLTensor*>();
                    TVM_FFI_ICHECK_GT(high, low) << "high must be bigger than low";
                    TVM_FFI_ICHECK(ffi::IsContiguous(*out));

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
                        TVM_FFI_THROW(InternalError)
                            << "Do not support random.randint on this device yet";
                      }
                    })
                  })
      .def_packed("tvm.contrib.random.uniform",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    RandomEngine* engine = nullptr;
                    double low = 0.0;
                    double high = 0.0;
                    DLTensor* out = nullptr;

                    if (args.size() == 3) {
                      if (auto opt_out = args[0].try_cast<DLTensor*>()) {
                        engine = &RandomThreadLocalEntry::ThreadLocal()->random_engine;
                        out = opt_out.value();
                        low = args[1].cast<double>();
                        high = args[2].cast<double>();
                      } else {
                        engine = &RandomThreadLocalEntry::ThreadLocal()->random_engine;
                        low = args[0].cast<double>();
                        high = args[1].cast<double>();
                        out = args[2].cast<DLTensor*>();
                      }
                    } else if (args.size() == 5) {
                      if (auto opt_out = args[0].try_cast<DLTensor*>()) {
                        out = opt_out.value();
                        engine = GetRandomEngineForArgs(args, 1, 2);
                        low = args[3].cast<double>();
                        high = args[4].cast<double>();
                      } else {
                        engine = GetRandomEngineForArgs(args, 0, 1);
                        low = args[2].cast<double>();
                        high = args[3].cast<double>();
                        out = args[4].cast<DLTensor*>();
                      }
                    } else {
                      TVM_FFI_THROW(InternalError)
                          << "tvm.contrib.random.uniform expects either 3 or 5 arguments, but got "
                          << args.size();
                    }

                    engine->SampleUniform(out, low, high);
                  })
      .def_packed("tvm.contrib.random.normal",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    RandomEngine* engine = nullptr;
                    double loc = 0.0;
                    double scale = 0.0;
                    DLTensor* out = nullptr;

                    if (args.size() == 3) {
                      if (auto opt_out = args[0].try_cast<DLTensor*>()) {
                        engine = &RandomThreadLocalEntry::ThreadLocal()->random_engine;
                        out = opt_out.value();
                        loc = args[1].cast<double>();
                        scale = args[2].cast<double>();
                      } else {
                        engine = &RandomThreadLocalEntry::ThreadLocal()->random_engine;
                        loc = args[0].cast<double>();
                        scale = args[1].cast<double>();
                        out = args[2].cast<DLTensor*>();
                      }
                    } else if (args.size() == 5) {
                      if (auto opt_out = args[0].try_cast<DLTensor*>()) {
                        out = opt_out.value();
                        engine = GetRandomEngineForArgs(args, 1, 2);
                        loc = args[3].cast<double>();
                        scale = args[4].cast<double>();
                      } else {
                        engine = GetRandomEngineForArgs(args, 0, 1);
                        loc = args[2].cast<double>();
                        scale = args[3].cast<double>();
                        out = args[4].cast<DLTensor*>();
                      }
                    } else {
                      TVM_FFI_THROW(InternalError)
                          << "tvm.contrib.random.normal expects either 3 or 5 arguments, but got "
                          << args.size();
                    }

                    engine->SampleNormal(out, loc, scale);
                  })
      .def_packed("tvm.contrib.random.random_fill",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    RandomThreadLocalEntry* entry = RandomThreadLocalEntry::ThreadLocal();
                    auto out = args[0].cast<DLTensor*>();
                    entry->random_engine.RandomFill(out);
                  })
      .def_packed("tvm.contrib.random.random_fill_for_measure",
                  [](ffi::PackedArgs args, ffi::Any* ret) -> void {
                    const auto curand =
                        tvm::ffi::Function::GetGlobal("runtime.contrib.curand.RandomFill");
                    auto out = args[0].cast<DLTensor*>();
                    if (curand.has_value() && out->device.device_type == DLDeviceType::kDLCUDA) {
                      if (out->dtype.code == DLDataTypeCode::kDLFloat) {
                        (*curand)(out);
                        return;
                      }
                    }
                    RandomThreadLocalEntry* entry = RandomThreadLocalEntry::ThreadLocal();
                    entry->random_engine.RandomFillForMeasure(out);
                  });
}

}  // namespace contrib
}  // namespace tvm
