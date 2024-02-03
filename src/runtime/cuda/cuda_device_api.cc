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
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include <cstring>

#include "cuda_common.h"

namespace tvm {
namespace runtime {

class CUDADeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final { CUDA_CALL(cudaSetDevice(dev.device_id)); }
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final {
    int value = 0;
    switch (kind) {
      case kExist:
        int count;
        CUDA_CALL(cudaGetDeviceCount(&count));
        value = static_cast<int>(dev.device_id < count);
        break;
      case kMaxThreadsPerBlock: {
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, dev.device_id));
        break;
      }
      case kWarpSize: {
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, dev.device_id));
        break;
      }
      case kMaxSharedMemoryPerBlock: {
        CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock, dev.device_id));
        break;
      }
      case kComputeVersion: {
        std::ostringstream os;
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMajor, dev.device_id));
        os << value << ".";
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMinor, dev.device_id));
        os << value;
        *rv = os.str();
        return;
      }
      case kDeviceName: {
        std::string name(256, 0);
        CUDA_DRIVER_CALL(cuDeviceGetName(&name[0], name.size(), dev.device_id));
        name.resize(strlen(name.c_str()));
        *rv = std::move(name);
        return;
      }
      case kMaxClockRate: {
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrClockRate, dev.device_id));
        break;
      }
      case kMultiProcessorCount: {
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, dev.device_id));
        break;
      }
      case kMaxThreadDimensions: {
        int dims[3];
        CUDA_CALL(cudaDeviceGetAttribute(&dims[0], cudaDevAttrMaxBlockDimX, dev.device_id));
        CUDA_CALL(cudaDeviceGetAttribute(&dims[1], cudaDevAttrMaxBlockDimY, dev.device_id));
        CUDA_CALL(cudaDeviceGetAttribute(&dims[2], cudaDevAttrMaxBlockDimZ, dev.device_id));

        std::stringstream ss;  // use json string to return multiple int values;
        ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
        *rv = ss.str();
        return;
      }
      case kMaxRegistersPerBlock: {
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrMaxRegistersPerBlock, dev.device_id));
        break;
      }
      case kGcnArch:
        return;
      case kApiVersion: {
        *rv = CUDA_VERSION;
        return;
      }
      case kDriverVersion:
        return;
      case kL2CacheSizeBytes: {
        // Get size of device l2 cache size in bytes.
        int l2_size = 0;
        CUDA_CALL(cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, dev.device_id));
        *rv = l2_size;
        return;
      }
      case kTotalGlobalMemory: {
        cudaDeviceProp prop;
        CUDA_CALL(cudaGetDeviceProperties(&prop, dev.device_id));
        int64_t total_global_memory = prop.totalGlobalMem;
        *rv = total_global_memory;
        return;
      }
    }
    *rv = value;
  }
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    ICHECK_EQ(256 % alignment, 0U) << "CUDA space is aligned at 256 bytes";
    void* ret;
    if (dev.device_type == kDLCUDAHost) {
      VLOG(1) << "allocating " << nbytes << "bytes on host";
      CUDA_CALL(cudaMallocHost(&ret, nbytes));
    } else {
      CUDA_CALL(cudaSetDevice(dev.device_id));
      size_t free_mem, total_mem;
      CUDA_CALL(cudaMemGetInfo(&free_mem, &total_mem));
      VLOG(1) << "allocating " << nbytes << " bytes on device, with " << free_mem
              << " bytes currently free out of " << total_mem << " bytes available";
      CUDA_CALL(cudaMalloc(&ret, nbytes));
    }
    return ret;
  }

  void FreeDataSpace(Device dev, void* ptr) final {
    if (dev.device_type == kDLCUDAHost) {
      VLOG(1) << "freeing host memory";
      CUDA_CALL(cudaFreeHost(ptr));
    } else {
      CUDA_CALL(cudaSetDevice(dev.device_id));
      VLOG(1) << "freeing device memory";
      CUDA_CALL(cudaFree(ptr));
    }
  }

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;

    if (dev_from.device_type == kDLCUDAHost) {
      dev_from.device_type = kDLCPU;
    }

    if (dev_to.device_type == kDLCUDAHost) {
      dev_to.device_type = kDLCPU;
    }

    // In case there is a copy from host mem to host mem */
    if (dev_to.device_type == kDLCPU && dev_from.device_type == kDLCPU) {
      memcpy(to, from, size);
      return;
    }

    if (dev_from.device_type == kDLCUDA && dev_to.device_type == kDLCUDA) {
      CUDA_CALL(cudaSetDevice(dev_from.device_id));
      if (dev_from.device_id == dev_to.device_id) {
        GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
      } else {
        cudaMemcpyPeerAsync(to, dev_to.device_id, from, dev_from.device_id, size, cu_stream);
      }
    } else if (dev_from.device_type == kDLCUDA && dev_to.device_type == kDLCPU) {
      CUDA_CALL(cudaSetDevice(dev_from.device_id));
      GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
    } else if (dev_from.device_type == kDLCPU && dev_to.device_type == kDLCUDA) {
      CUDA_CALL(cudaSetDevice(dev_to.device_id));
      GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
    } else {
      LOG(FATAL) << "expect copy from/to GPU or between GPU";
    }
  }

 public:
  TVMStreamHandle CreateStream(Device dev) {
    CUDA_CALL(cudaSetDevice(dev.device_id));
    cudaStream_t retval;
    CUDA_CALL(cudaStreamCreate(&retval));
    return static_cast<TVMStreamHandle>(retval);
  }

  void FreeStream(Device dev, TVMStreamHandle stream) {
    CUDA_CALL(cudaSetDevice(dev.device_id));
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    CUDA_CALL(cudaStreamDestroy(cu_stream));
  }

  void SyncStreamFromTo(Device dev, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    CUDA_CALL(cudaSetDevice(dev.device_id));
    cudaStream_t src_stream = static_cast<cudaStream_t>(event_src);
    cudaStream_t dst_stream = static_cast<cudaStream_t>(event_dst);
    cudaEvent_t evt;
    CUDA_CALL(cudaEventCreate(&evt));
    CUDA_CALL(cudaEventRecord(evt, src_stream));
    CUDA_CALL(cudaStreamWaitEvent(dst_stream, evt, 0));
    CUDA_CALL(cudaEventDestroy(evt));
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {
    CUDA_CALL(cudaSetDevice(dev.device_id));
    CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

  void SetStream(Device dev, TVMStreamHandle stream) final {
    CUDAThreadEntry::ThreadLocal()->stream = static_cast<cudaStream_t>(stream);
  }

  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final {
    return CUDAThreadEntry::ThreadLocal()->pool.AllocWorkspace(dev, size);
  }

  void FreeWorkspace(Device dev, void* data) final {
    CUDAThreadEntry::ThreadLocal()->pool.FreeWorkspace(dev, data);
  }

  static CUDADeviceAPI* Global() {
    // NOTE: explicitly use new to avoid exit-time destruction of global state
    // Global state will be recycled by OS as the process exits.
    static auto* inst = new CUDADeviceAPI();
    return inst;
  }

 private:
  static void GPUCopy(const void* from, void* to, size_t size, cudaMemcpyKind kind,
                      cudaStream_t stream) {
    if (stream != nullptr) {
      CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
    } else {
      CUDA_CALL(cudaMemcpy(to, from, size, kind));
    }
  }
};

typedef dmlc::ThreadLocalStore<CUDAThreadEntry> CUDAThreadStore;

CUDAThreadEntry::CUDAThreadEntry() : pool(kDLCUDA, CUDADeviceAPI::Global()) {}

CUDAThreadEntry* CUDAThreadEntry::ThreadLocal() { return CUDAThreadStore::Get(); }

TVM_REGISTER_GLOBAL("device_api.cuda").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = CUDADeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

TVM_REGISTER_GLOBAL("device_api.cuda_host").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = CUDADeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

class CUDATimerNode : public TimerNode {
 public:
  virtual void Start() {
    // This initial cudaEventRecord is sometimes pretty slow (~100us). Does
    // cudaEventRecord do some stream synchronization?
    CUDA_CALL(cudaEventRecord(start_, CUDAThreadEntry::ThreadLocal()->stream));
  }
  virtual void Stop() { CUDA_CALL(cudaEventRecord(stop_, CUDAThreadEntry::ThreadLocal()->stream)); }
  virtual int64_t SyncAndGetElapsedNanos() {
    CUDA_CALL(cudaEventSynchronize(stop_));
    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds * 1e6;
  }
  virtual ~CUDATimerNode() {
    CUDA_CALL(cudaEventDestroy(start_));
    CUDA_CALL(cudaEventDestroy(stop_));
  }
  CUDATimerNode() {
    CUDA_CALL(cudaEventCreate(&start_));
    CUDA_CALL(cudaEventCreate(&stop_));
  }

  static constexpr const char* _type_key = "CUDATimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(CUDATimerNode, TimerNode);

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

TVM_REGISTER_OBJECT_TYPE(CUDATimerNode);

TVM_REGISTER_GLOBAL("profiling.timer.cuda").set_body_typed([](Device dev) {
  return Timer(make_object<CUDATimerNode>());
});

TVM_DLL String GetCudaFreeMemory() {
  size_t free_mem, total_mem;
  CUDA_CALL(cudaMemGetInfo(&free_mem, &total_mem));
  std::stringstream ss;
  ss << "Current CUDA memory is " << free_mem << " bytes free out of " << total_mem
     << " bytes on device";
  return ss.str();
}

TVM_REGISTER_GLOBAL("runtime.GetCudaFreeMemory").set_body_typed(GetCudaFreeMemory);

TVM_REGISTER_GLOBAL("runtime.get_cuda_stream").set_body_typed([]() {
  return static_cast<void*>(CUDAThreadEntry::ThreadLocal()->stream);
});

TVM_DLL int GetCudaDeviceCount() {
  int count;
  CUDA_CALL(cudaGetDeviceCount(&count));
  return count;
}

TVM_REGISTER_GLOBAL("runtime.GetCudaDeviceCount").set_body_typed(GetCudaDeviceCount);

}  // namespace runtime
}  // namespace tvm
