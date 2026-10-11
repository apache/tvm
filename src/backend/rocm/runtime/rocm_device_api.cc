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
 * \file rocm_device_api.cc
 * \brief GPU specific API
 */
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/timer.h>

#include <algorithm>

#include "rocm_common.h"

namespace tvm {
namespace runtime {

class ROCMDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final { ROCM_CALL(hipSetDevice(dev.device_id)); }
  ffi::Any GetAttr(Device device, const ffi::String& name) final {
    ffi::Any rv;
    int value = 0;
    if (name == "exist") {
      if (hsa_init() == HSA_STATUS_SUCCESS) {
        int count = 0;
        auto err = hipGetDeviceCount(&count);
        value = err == hipSuccess && device.device_id >= 0 && device.device_id < count;
        hsa_shut_down();
      } else {
        value = 0;
      }
      return bool(value);
    }
    if (name == "max_threads_per_block") {
      ROCM_CALL(
          hipDeviceGetAttribute(&value, hipDeviceAttributeMaxThreadsPerBlock, device.device_id));
      return value;
    }
    if (name == "warp_size") {
      ROCM_CALL(hipDeviceGetAttribute(&value, hipDeviceAttributeWarpSize, device.device_id));
      return value;
    }
    if (name == "max_shared_memory_per_block") {
      ROCM_CALL(hipDeviceGetAttribute(&value, hipDeviceAttributeMaxSharedMemoryPerBlock,
                                      device.device_id));
      return value;
    }
    if (name == "compute_version") {
      std::ostringstream os;
      ROCM_CALL(hipDeviceGetAttribute(&value, hipDeviceAttributeComputeCapabilityMajor,
                                      device.device_id));
      os << value << ".";
      ROCM_CALL(hipDeviceGetAttribute(&value, hipDeviceAttributeComputeCapabilityMinor,
                                      device.device_id));
      os << value;
      rv = os.str();
      return rv;
    }
    if (name == "device_name") {
      std::string name(256, 0);
      ROCM_CALL(hipDeviceGetName(&name[0], name.size(), device.device_id));
      name.resize(strlen(name.c_str()));
      rv = std::move(name);
      return rv;
    }
    if (name == "max_clock_rate") {
      ROCM_CALL(hipDeviceGetAttribute(&value, hipDeviceAttributeClockRate, device.device_id));
      return value;
    }
    if (name == "multi_processor_count") {
      ROCM_CALL(
          hipDeviceGetAttribute(&value, hipDeviceAttributeMultiprocessorCount, device.device_id));
      return value;
    }
    if (name == "max_thread_dimensions") {
      int dims[3];
      ROCM_CALL(hipDeviceGetAttribute(&dims[0], hipDeviceAttributeMaxBlockDimX, device.device_id));
      ROCM_CALL(hipDeviceGetAttribute(&dims[1], hipDeviceAttributeMaxBlockDimY, device.device_id));
      ROCM_CALL(hipDeviceGetAttribute(&dims[2], hipDeviceAttributeMaxBlockDimZ, device.device_id));

      std::stringstream ss;
      ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
      rv = ss.str();
      return rv;
    }
    if (name == "max_registers_per_block") {
      ROCM_CALL(
          hipDeviceGetAttribute(&value, hipDeviceAttributeMaxRegistersPerBlock, device.device_id));
      return value;
    }
    if (name == "gcn_arch") {
      hipDeviceProp_t prop;
      ROCM_CALL(hipGetDeviceProperties(&prop, device.device_id));
      rv = prop.gcnArchName;
      return rv;
    }
    if (name == "api_version") {
      rv = HIP_VERSION;
      return rv;
    }
    if (name == "l2_cache_size_bytes") {
      // Get size of device l2 cache size in bytes.
      int l2_size;
      ROCM_CALL(hipDeviceGetAttribute(&l2_size, hipDeviceAttributeL2CacheSize, device.device_id));
      rv = l2_size;
      return rv;
    }
    if (name == "total_global_memory") {
      hipDeviceProp_t prop;
      ROCM_CALL(hipGetDeviceProperties(&prop, device.device_id));
      int64_t total_global_memory = prop.totalGlobalMem;
      rv = total_global_memory;
      return rv;
    }
    if (name == "available_global_memory") {
      // Not currently implemented.
      rv = nullptr;
      return rv;
    }

    return rv;
  }
  ffi::Tensor Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                    ffi::Optional<ffi::String> mem_scope = std::nullopt) final {
    TVM_FFI_CHECK(
        !mem_scope.has_value() || mem_scope.value().empty() || mem_scope.value() == "global",
        ValueError)
        << "Unsupported memory scope: " << mem_scope.value_or("");
    struct Allocator {
      ROCMDeviceAPI* api;
      void AllocData(DLTensor* tensor) {
        size_t alignment =
            std::max(static_cast<size_t>(kAllocAlignment),
                     static_cast<size_t>((tensor->dtype.bits / 8) * tensor->dtype.lanes));
        tensor->data = api->AllocDataSpace(tensor->device, ffi::GetDataSize(*tensor), alignment,
                                           tensor->dtype);
      }
      void FreeData(DLTensor* tensor) { api->FreeDataSpace(tensor->device, tensor->data); }
    };
    return ffi::Tensor::FromNDAlloc(Allocator{this}, shape, dtype, device);
  }

 private:
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) {
    TVM_FFI_ICHECK_EQ(256 % alignment, 0U) << "ROCM space is aligned at 256 bytes";
    void* ret;
    if (dev.device_type == kDLROCMHost) {
      VLOG(1) << "allocating " << nbytes << "bytes on host";
      ROCM_CALL(hipHostMalloc(&ret, nbytes));
    } else {
      ROCM_CALL(hipSetDevice(dev.device_id));
      VLOG(1) << "allocating " << nbytes << " bytes on device";
      ROCM_CALL(hipMalloc(&ret, nbytes));
    }
    return ret;
  }

  void FreeDataSpace(Device dev, void* ptr) {
    if (dev.device_type == kDLROCMHost) {
      ROCM_CALL(hipHostFree(ptr));
    } else {
      ROCM_CALL(hipSetDevice(dev.device_id));
      ROCM_CALL(hipFree(ptr));
    }
  }

 public:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;

    if (dev_from.device_type == kDLROCMHost) {
      dev_from.device_type = kDLCPU;
    }

    if (dev_to.device_type == kDLROCMHost) {
      dev_to.device_type = kDLCPU;
    }

    // In case there is a copy from host mem to host mem */
    if (dev_to.device_type == kDLCPU && dev_from.device_type == kDLCPU) {
      memcpy(to, from, size);
      return;
    }

    if (dev_from.device_type == kDLROCM && dev_to.device_type == kDLROCM) {
      ROCM_CALL(hipSetDevice(dev_from.device_id));
      if (dev_from.device_id == dev_to.device_id) {
        GPUCopy(from, to, size, hipMemcpyDeviceToDevice, hip_stream);
      } else {
        ROCM_CALL(
            hipMemcpyPeerAsync(to, dev_to.device_id, from, dev_from.device_id, size, hip_stream));
      }
    } else if (dev_from.device_type == kDLROCM && dev_to.device_type == kDLCPU) {
      ROCM_CALL(hipSetDevice(dev_from.device_id));
      GPUCopy(from, to, size, hipMemcpyDeviceToHost, hip_stream);
    } else if (dev_from.device_type == kDLCPU && dev_to.device_type == kDLROCM) {
      ROCM_CALL(hipSetDevice(dev_to.device_id));
      GPUCopy(from, to, size, hipMemcpyHostToDevice, hip_stream);
    } else {
      TVM_FFI_THROW(InternalError) << "expect copy from/to GPU or between GPU";
    }
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    ROCM_CALL(hipSetDevice(dev.device_id));
    ROCM_CALL(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
  }

  static ROCMDeviceAPI* Global() {
    static ROCMDeviceAPI* inst = new ROCMDeviceAPI();
    return inst;
  }

 private:
  static void GPUCopy(const void* from, void* to, size_t size, hipMemcpyKind kind,
                      hipStream_t stream) {
    if (stream != nullptr) {
      ROCM_CALL(hipMemcpyAsync(to, from, size, kind, stream));
    } else {
      ROCM_CALL(hipMemcpy(to, from, size, kind));
    }
  }
};

ROCMThreadEntry::ROCMThreadEntry() {}

ROCMThreadEntry* ROCMThreadEntry::ThreadLocal() {
  static thread_local ROCMThreadEntry inst;
  return &inst;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("device_api.rocm",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    DeviceAPI* ptr = ROCMDeviceAPI::Global();
                    *rv = static_cast<void*>(ptr);
                  })
      .def_packed("device_api.rocm_host", [](ffi::PackedArgs args, ffi::Any* rv) {
        DeviceAPI* ptr = ROCMDeviceAPI::Global();
        *rv = static_cast<void*>(ptr);
      });
}

class ROCMTimerNode : public TimerNode {
 public:
  virtual void Start() {
    int device_id;
    ROCM_CALL(hipGetDevice(&device_id));
    stream_ = TVMFFIEnvGetStream(kDLROCM, device_id);
    ROCM_CALL(hipEventRecord(start_, static_cast<hipStream_t>(stream_)));
  }
  virtual void Stop() {
    int device_id;
    ROCM_CALL(hipGetDevice(&device_id));
    ROCM_CALL(hipEventRecord(stop_, static_cast<hipStream_t>(stream_)));
  }
  virtual int64_t SyncAndGetElapsedNanos() {
    ROCM_CALL(hipEventSynchronize(stop_));
    float milliseconds = 0;
    ROCM_CALL(hipEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds * 1e6;
  }
  virtual ~ROCMTimerNode() {
    ROCM_CALL(hipEventDestroy(start_));
    ROCM_CALL(hipEventDestroy(stop_));
  }
  ROCMTimerNode() {
    ROCM_CALL(hipEventCreate(&start_));
    ROCM_CALL(hipEventCreate(&stop_));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.rocm.ROCMTimerNode", ROCMTimerNode, TimerNode);

 private:
  hipEvent_t start_;
  hipEvent_t stop_;
  TVMStreamHandle stream_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.timer.rocm",
           [](Device dev) { return Timer(ffi::UnsafeInit{}, ffi::make_object<ROCMTimerNode>()); })
      .def("runtime.get_rocm_stream", []() {
        int device_id;
        ROCM_CALL(hipGetDevice(&device_id));
        return static_cast<void*>(TVMFFIEnvGetStream(kDLROCM, device_id));
      });
}

}  // namespace runtime
}  // namespace tvm
