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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <string>

#include "../../src/backend/cuda/launch/launch_plan.h"
#include "../../src/runtime/metadata.h"

namespace {
using namespace tvm_cuda_launch;

TEST(CudaLaunch, LegacyTagsAreDecodedOnlyAtTheBoundary) {
  PackedLaunchPlan plan({},
                        {"blockIdx.x", "threadIdx.x", "clusterCtaIdx.x", "preferredClusterCtaIdx.x",
                         "tirx.use_programtic_dependent_launch", "tirx.use_dyn_shared_memory"},
                        {});
  EXPECT_EQ(plan.num_values, 5);
  EXPECT_EQ(plan.operands[4].field, Field::kProgrammaticStreamSerialization);
  EXPECT_EQ(plan.operands[4].index, -1);
  LaunchValues legacy;
  legacy.SetInt(Field::kGridX, 0);
  legacy.SetInt(Field::kClusterX, 1);
  legacy.SetInt(Field::kPreferredClusterX, 1);
  plan.NormalizeLegacyPresence(&legacy);
  EXPECT_EQ(legacy.Int(Field::kGridX), 1);
  EXPECT_TRUE(legacy.Has(Option::kCluster));
  EXPECT_FALSE(legacy.Has(Option::kPreferredCluster));
  EXPECT_THROW(PackedLaunchPlan({}, {"tirx.use_dyn_shared_memory", "blockIdx.x"}, {}),
               std::invalid_argument);
  EXPECT_THROW(PackedLaunchPlan({"grid.x"}, {"blockIdx.x"}, {}), std::invalid_argument);
  EXPECT_THROW(PackedLaunchPlan({"grid.x", "grid.x"}, {}, {}), std::invalid_argument);
}

class MetadataStream : public tvm::support::Stream {
 public:
  using Stream::Read;
  using Stream::Write;
  size_t Read(void* ptr, size_t size) override {
    size = std::min(size, data.size() - offset);
    std::memcpy(ptr, data.data() + offset, size);
    offset += size;
    return size;
  }
  size_t Write(const void* ptr, size_t size) override {
    data.append(static_cast<const char*>(ptr), size);
    return size;
  }
  std::string data;
  size_t offset = 0;
};

TEST(CudaLaunch, NativeOperandsRetainTheirTypes) {
  PackedLaunchPlan plan({"grid.x", "grid.y", "grid.z", "block.x", "block.y", "block.z", "stream",
                         "access_policy_window.hit_ratio"},
                        {}, {});
  void* stream = reinterpret_cast<void*>(0x20000);
  auto decode = tvm::ffi::Function::FromPacked([&](tvm::ffi::PackedArgs args, tvm::ffi::Any*) {
    LaunchValues values = plan.Extract(args, 1);
    EXPECT_EQ(values.Int(Field::kGridX), 8);
    EXPECT_EQ(values.Int(Field::kBlockX), 32);
    EXPECT_EQ(values.Get(Field::kStream).handle, stream);
    EXPECT_DOUBLE_EQ(values.Get(Field::kAccessPolicyWindowHitRatio).real, 0.25);
    EXPECT_FALSE(values.Has(Option::kCluster));
  });
  decode(999, 8, 1, 1, 32, 1, 1, stream, 0.25);
  EXPECT_THROW(decode(999, 8), std::invalid_argument);
}

TEST(CudaLaunch, MetadataReadsLegacyAndRoundTripsNative) {
  using namespace tvm;
  runtime::FunctionInfo old("kernel", {{kDLInt, 32, 1}}, {"blockIdx.x", "threadIdx.x"}, {});
  MetadataStream bytes;
  bytes.Write(old->name);
  bytes.Write(old->arg_types);
  bytes.Write(old->launch_param_tags);
  bytes.Write(old->arg_extra_tags);
  runtime::FunctionInfo decoded(ffi::UnsafeInit{});
  ASSERT_TRUE(bytes.Read(&decoded));
  EXPECT_EQ(decoded->name, "kernel");
  EXPECT_EQ(decoded->launch_param_tags.size(), 2);
  EXPECT_TRUE(decoded->cuda_launch_fields.empty());

  runtime::FunctionInfo native("native", {{kDLInt, 32, 1}}, {}, {},
                               {"grid.x", "grid.y", "grid.z", "block.x", "block.y", "block.z"},
                               {{"min_blocks_per_sm", 2}});
  MetadataStream native_bytes;
  native_bytes.Write(native);
  ASSERT_TRUE(native_bytes.Read(&decoded));
  EXPECT_EQ(decoded->cuda_launch_fields.size(), 6);
  EXPECT_EQ(decoded->cuda_kernel_attrs.at("min_blocks_per_sm"), 2);
  auto json_decoded = ffi::make_object<runtime::FunctionInfoObj>();
  json_decoded->LoadFromJSON(native->SaveToJSON().cast<ffi::json::Object>());
  EXPECT_EQ(json_decoded->cuda_launch_fields.size(), 6);
  EXPECT_EQ(json_decoded->cuda_kernel_attrs.at("min_blocks_per_sm"), 2);
}
}  // namespace

#if __has_include(<cuda.h>) && __has_include(<cuda_runtime.h>)
#include <cuda.h>
#include <cuda_runtime.h>
#if CUDA_VERSION >= 12000 && CUDART_VERSION >= 12000
#include "../../src/backend/cuda/launch/launch_config.h"

namespace {
TEST(CudaLaunch, DriverAndRuntimeEncodeTheSameLaunch) {
  LaunchValues values;
  values.SetInt(Field::kGridX, 8);
  values.SetInt(Field::kBlockX, 128);
  values.SetInt(Field::kClusterX, 2);
  values.SetInt(Field::kCooperative, 1);
  values.SetInt(Field::kProgrammaticStreamSerialization, 1);
  values.SetInt(Field::kPriority, -2);
  values.SetInt(Field::kDynamicSmemBytes, 65536);
  values.SetFloat(Field::kAccessPolicyWindowHitRatio, 0.5);
  values.SetInt(Field::kAccessPolicyWindowNumBytes, 4096);
  values.SetHandle(Field::kAccessPolicyWindowBasePtr, reinterpret_cast<void*>(0x10000));
  values.SetHandle(Field::kStream, reinterpret_cast<void*>(0x20000));
  values = Normalize(values, {});
  CUlaunchConfig driver{};
  cudaLaunchConfig_t runtime{};
  std::array<CUlaunchAttribute, kMaxAttributes> driver_attrs{};
  std::array<cudaLaunchAttribute, kMaxAttributes> runtime_attrs{};
  EncodeDriver(values, {}, nullptr, &driver, &driver_attrs);
  EncodeRuntime(values, {}, nullptr, &runtime, &runtime_attrs);
  EXPECT_EQ(driver.gridDimX, runtime.gridDim.x);
  EXPECT_EQ(driver.blockDimX, runtime.blockDim.x);
  EXPECT_EQ(driver.sharedMemBytes, runtime.dynamicSmemBytes);
  EXPECT_EQ(driver.hStream, runtime.stream);
  ASSERT_EQ(driver.numAttrs, runtime.numAttrs);
  for (unsigned i = 0; i < driver.numAttrs; ++i) {
    EXPECT_EQ(static_cast<int>(driver_attrs[i].id), static_cast<int>(runtime_attrs[i].id));
    EXPECT_EQ(
        std::memcmp(&driver_attrs[i].value, &runtime_attrs[i].val, sizeof(driver_attrs[i].value)),
        0);
  }
  values.SetInt(Field::kGridX, 7);
  EXPECT_THROW(Normalize(values, {}), std::invalid_argument);
  values.SetInt(Field::kGridX, 0);
  EXPECT_THROW(Normalize(values, {}), std::invalid_argument);
}

struct FakeResourceAPI {
  using Kernel = int;
  static inline int initializations = 0;
  static void Initialize(Kernel, int, ResourceState* state) {
    ++initializations;
    state->portable_bytes = 49152;
    state->optin_bytes = 229376;
  }
  static void AllowCluster(Kernel) {}
  static void SetDynamicSmem(Kernel, size_t) {}
};

TEST(CudaLaunch, ResourceCachingHandlesIncreasingSizesAndExplicitModes) {
  ResourceState state;
  FakeResourceAPI::initializations = 0;
  for (int64_t bytes : {1024, 65536, 131072, 1024}) {
    LaunchValues values;
    values.SetInt(Field::kDynamicSmemBytes, bytes);
    PrepareResources<FakeResourceAPI>(0, 0, &values, &state);
#if CUDA_VERSION >= 13020 && CUDART_VERSION >= 13020
    if (bytes > 49152) {
      EXPECT_EQ(values.Int(Field::kSharedMemoryMode), 2);
    }
#else
    if (bytes > 49152) {
      EXPECT_GE(state.configured_bytes, bytes);
    }
#endif
  }
  EXPECT_EQ(FakeResourceAPI::initializations, 1);
  LaunchValues explicit_mode;
  explicit_mode.SetInt(Field::kSharedMemoryMode, 1);
  PrepareResources<FakeResourceAPI>(0, 0, &explicit_mode, &state);
  EXPECT_EQ(explicit_mode.Int(Field::kSharedMemoryMode), 1);
}
}  // namespace
#endif
#endif
