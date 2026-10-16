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

// This header is also embedded in exported cuda_host translation units. Keep
// it independent of TVM headers, registries, runtime services, and Python.
#ifndef TVM_BACKEND_CUDA_LAUNCH_LAUNCH_CONFIG_H_
#define TVM_BACKEND_CUDA_LAUNCH_LAUNCH_CONFIG_H_

#include <cmath>
#include <limits>
#include <mutex>
#include <utility>

#include "launch_attributes.h"

namespace tvm_cuda_launch {

inline void Require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("CUDA launch: " + message);
}

inline void Check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

inline void Check(CUresult error) {
  if (error != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(error, &message);
    throw std::runtime_error(message ? message : "CUDA driver error");
  }
}

inline Field Axis(Field x, int axis) { return static_cast<Field>(static_cast<int>(x) + axis); }

inline LaunchValues Normalize(LaunchValues config, const KernelRequirements& kernel) {
  for (Field x : {Field::kGridX, Field::kBlockX, Field::kClusterX, Field::kPreferredClusterX}) {
    for (int axis = 0; axis != 3; ++axis) {
      auto& value = config.values[static_cast<size_t>(Axis(x, axis))].integer;
      Require(value > 0 && static_cast<uint64_t>(value) <= UINT32_MAX,
              std::string(kFields[static_cast<size_t>(Axis(x, axis))].name) +
                  " must be a positive uint32 dimension");
    }
  }
  int64_t threads = 1;
  for (int axis = 0; axis != 3; ++axis) {
    int64_t extent = config.Int(Axis(Field::kBlockX, axis));
    Require(extent <= 1024 / threads, "block must contain at most 1024 threads");
    threads *= extent;
  }
  Require(config.Int(Field::kDynamicSmemBytes) >= 0 &&
              static_cast<uint64_t>(config.Int(Field::kDynamicSmemBytes)) <= UINT32_MAX,
          "dynamic_smem_bytes must be a nonnegative uint32 byte count");
  Require(config.Int(Field::kPriority) >= INT32_MIN && config.Int(Field::kPriority) <= INT32_MAX,
          "priority must fit in int32");
  Require(!config.Has(Option::kPreferredCluster) || config.Has(Option::kCluster),
          "preferred_cluster requires an explicit cluster");
  for (int axis = 0; axis != 3; ++axis) {
    int64_t grid = config.Int(Axis(Field::kGridX, axis));
    int64_t cluster = config.Int(Axis(Field::kClusterX, axis));
    int64_t preferred = config.Int(Axis(Field::kPreferredClusterX, axis));
    if (config.Has(Option::kCluster)) {
      Require(grid % cluster == 0, "grid must be divisible by cluster in every dimension");
    }
    if (config.Has(Option::kPreferredCluster)) {
      Require(grid % preferred == 0 && preferred % cluster == 0,
              "preferred_cluster must divide grid and be a multiple of cluster");
      Require(axis != 2 || preferred == cluster, "preferred_cluster.z must equal cluster.z");
    }
    if (kernel.required_block_size) {
      Require(!kernel.block[axis] || kernel.block[axis] == config.Int(Axis(Field::kBlockX, axis)),
              "block does not match the kernel's required block size");
      Require(!kernel.cluster[axis] || kernel.cluster[axis] == cluster,
              "cluster does not match the kernel's required cluster size");
      Require(!config.Has(Option::kPreferredCluster) || preferred == cluster,
              "required_block_size cannot use a different preferred_cluster");
    }
  }
  for (Field flag :
       {Field::kCooperative, Field::kProgrammaticStreamSerialization,
        Field::kNvlinkUtilCentricScheduling, Field::kProgrammaticEventTriggerAtBlockStart}) {
    Require(config.Int(flag) == 0 || config.Int(flag) == 1,
            std::string(kFields[static_cast<size_t>(flag)].name) + " must be a boolean");
  }
  for (auto entry :
       {std::pair{Field::kClusterSchedulingPolicy, 2}, std::pair{Field::kMemSyncDomain, 1},
        std::pair{Field::kPortableClusterSizeMode, 2}, std::pair{Field::kSharedMemoryMode, 4},
        std::pair{Field::kAccessPolicyWindowHitProp, 2},
        std::pair{Field::kAccessPolicyWindowMissProp, 2}}) {
    Require(
        config.Int(entry.first) >= 0 && config.Int(entry.first) <= entry.second,
        std::string(kFields[static_cast<size_t>(entry.first)].name) + " has an invalid enum value");
  }
  if (config.Has(Option::kPreferredSharedMemoryCarveout)) {
    Require(config.Int(Field::kPreferredSharedMemoryCarveout) >= 0 &&
                config.Int(Field::kPreferredSharedMemoryCarveout) <= 100,
            "preferred_shared_memory_carveout must be between 0 and 100");
  }
  if (config.Has(Option::kMemSyncDomainMap)) {
    for (Field field : {Field::kMemSyncDomainMapDefault, Field::kMemSyncDomainMapRemote}) {
      Require(config.Int(field) >= 0 && config.Int(field) <= UINT8_MAX,
              "memory synchronization domain IDs must fit in uint8");
    }
  }
  if (config.Has(Option::kAccessPolicyWindow)) {
    double ratio = config.Get(Field::kAccessPolicyWindowHitRatio).real;
    Require(std::isfinite(ratio) && ratio >= 0.0 && ratio <= 1.0,
            "access_policy_window.hit_ratio must be between 0 and 1");
    Require(config.Int(Field::kAccessPolicyWindowNumBytes) >= 0,
            "access_policy_window.num_bytes must be nonnegative");
    Require(config.Int(Field::kAccessPolicyWindowNumBytes) == 0 ||
                config.Get(Field::kAccessPolicyWindowBasePtr).handle != nullptr,
            "access_policy_window.base_ptr is null for a nonempty window");
  }
  for (auto entry :
       {std::pair{Option::kProgrammaticEvent, Field::kProgrammaticEventEvent},
        std::pair{Option::kLaunchCompletionEvent, Field::kLaunchCompletionEventEvent}}) {
    if (config.Has(entry.first)) {
      Require(config.Get(entry.second).handle != nullptr, "launch event must not be null");
      Require(config.Int(Axis(entry.second, 1)) == 0,
              "launch event flags must be zero; external recording is unsupported");
    }
  }
  return config;
}

// Owned by the kernel wrapper and indexed by device, so unloading a module also
// destroys its state. Opt-in byte limits only increase between invocations.
struct ResourceState {
  std::mutex mutex;
  bool initialized = false;
  size_t static_bytes = 0;
  size_t portable_bytes = 0;
  size_t optin_bytes = 0;
  size_t configured_bytes = 0;
  bool nonportable_cluster = false;
};

struct DriverAPI {
  using Kernel = CUfunction;
  static void Initialize(Kernel kernel, int device, ResourceState* state) {
    int value;
    Check(cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel));
    state->static_bytes = value;
    Check(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));
    state->portable_bytes = value;
    Check(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                               device));
    state->optin_bytes = value;
  }
  static void AllowCluster(Kernel kernel) {
    Check(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
  }
  static void SetDynamicSmem(Kernel kernel, size_t bytes) {
    Check(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes));
  }
};

struct RuntimeAPI {
  using Kernel = const void*;
  static void Initialize(Kernel kernel, int device, ResourceState* state) {
    cudaFuncAttributes attributes{};
    Check(cudaFuncGetAttributes(&attributes, kernel));
    state->static_bytes = attributes.sharedSizeBytes;
    int value;
    Check(cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock, device));
    state->portable_bytes = value;
    Check(cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    state->optin_bytes = value;
  }
  static void AllowCluster(Kernel kernel) {
    Check(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  }
  static void SetDynamicSmem(Kernel kernel, size_t bytes) {
    Check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  }
};

template <typename API>
inline void PrepareResources(typename API::Kernel kernel, int device, LaunchValues* config,
                             ResourceState* state) {
  std::lock_guard<std::mutex> lock(state->mutex);
  if (!state->initialized) {
    API::Initialize(kernel, device, state);
    state->initialized = true;
  }
  if (config->Has(Option::kCluster) && !config->Has(Option::kPortableClusterSizeMode)) {
#if CUDA_VERSION >= 13020 && CUDART_VERSION >= 13020
    config->SetInt(Field::kPortableClusterSizeMode, 2);
#else
    if (!state->nonportable_cluster) {
      API::AllowCluster(kernel);
      state->nonportable_cluster = true;
    }
#endif
  }
  if (!config->Has(Option::kSharedMemoryMode)) {
    size_t bytes = config->Int(Field::kDynamicSmemBytes);
    if (bytes + state->static_bytes > state->optin_bytes) {
#if CUDA_VERSION >= 13040 && CUDART_VERSION >= 13040
      Require(config->Has(Option::kCluster), "oversized shared memory needs a cluster launch");
      config->SetInt(Field::kSharedMemoryMode, 3);
#else
      throw std::invalid_argument("CUDA oversized shared memory requires CUDA 13.4 or newer");
#endif
    } else if (bytes + state->static_bytes > state->portable_bytes) {
#if CUDA_VERSION >= 13020 && CUDART_VERSION >= 13020
      config->SetInt(Field::kSharedMemoryMode, 2);
#else
      if (bytes > state->configured_bytes) {
        API::SetDynamicSmem(kernel, bytes);
        state->configured_bytes = bytes;
      }
#endif
    }
  }
}

inline void EncodeDriver(const LaunchValues& values, const KernelRequirements& kernel,
                         CUstream default_stream, CUlaunchConfig* config,
                         std::array<CUlaunchAttribute, kMaxAttributes>* attrs) {
  unsigned int* grid[] = {&config->gridDimX, &config->gridDimY, &config->gridDimZ};
  unsigned int* block[] = {&config->blockDimX, &config->blockDimY, &config->blockDimZ};
  for (int axis = 0; axis != 3; ++axis) {
    *grid[axis] = values.Int(Axis(Field::kGridX, axis));
    *block[axis] = values.Int(Axis(Field::kBlockX, axis));
    if (kernel.required_block_size) {
#if CUDA_VERSION >= 13000
      *grid[axis] /= values.Int(Axis(Field::kClusterX, axis));
      *block[axis] = 1;
#else
      throw std::invalid_argument("CUDA required block size requires CUDA 13 or newer");
#endif
    }
  }
  config->sharedMemBytes = values.Int(Field::kDynamicSmemBytes);
  config->hStream = values.Has(Option::kStream)
                        ? static_cast<CUstream>(values.Get(Field::kStream).handle)
                        : default_stream;
  config->numAttrs = EncodeDriverAttributes(values, attrs, kernel.required_block_size);
  config->attrs = config->numAttrs ? attrs->data() : nullptr;
}

inline void EncodeRuntime(const LaunchValues& values, const KernelRequirements& kernel,
                          cudaStream_t default_stream, cudaLaunchConfig_t* config,
                          std::array<cudaLaunchAttribute, kMaxAttributes>* attrs) {
  unsigned int* grid[] = {&config->gridDim.x, &config->gridDim.y, &config->gridDim.z};
  unsigned int* block[] = {&config->blockDim.x, &config->blockDim.y, &config->blockDim.z};
  for (int axis = 0; axis != 3; ++axis) {
    *grid[axis] = values.Int(Axis(Field::kGridX, axis));
    *block[axis] = values.Int(Axis(Field::kBlockX, axis));
    if (kernel.required_block_size) {
#if CUDART_VERSION >= 13000
      *grid[axis] /= values.Int(Axis(Field::kClusterX, axis));
      *block[axis] = 1;
#else
      throw std::invalid_argument("CUDA required block size requires CUDA 13 or newer");
#endif
    }
  }
  config->dynamicSmemBytes = values.Int(Field::kDynamicSmemBytes);
  config->stream = values.Has(Option::kStream)
                       ? static_cast<cudaStream_t>(values.Get(Field::kStream).handle)
                       : default_stream;
  config->numAttrs = EncodeRuntimeAttributes(values, attrs, kernel.required_block_size);
  config->attrs = config->numAttrs ? attrs->data() : nullptr;
}

}  // namespace tvm_cuda_launch
#endif  // TVM_BACKEND_CUDA_LAUNCH_LAUNCH_CONFIG_H_
