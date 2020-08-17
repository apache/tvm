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
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include "rocm_common.h"

namespace tvm {
namespace runtime {

class ROCMDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final { ROCM_CALL(hipSetDevice(ctx.device_id)); }
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    int value = 0;
    switch (kind) {
      case kExist: {
        if (hsa_init() == HSA_STATUS_SUCCESS) {
          int dev;
          ROCM_CALL(hipGetDeviceCount(&dev));
          value = dev > ctx.device_id ? 1 : 0;
          hsa_shut_down();
        } else {
          value = 0;
        }
        break;
      }
      case kMaxThreadsPerBlock: {
        ROCM_CALL(
            hipDeviceGetAttribute(&value, hipDeviceAttributeMaxThreadsPerBlock, ctx.device_id));
        break;
      }
      case kWarpSize: {
        ROCM_CALL(hipDeviceGetAttribute(&value, hipDeviceAttributeWarpSize, ctx.device_id));
        break;
      }
      case kMaxSharedMemoryPerBlock: {
        ROCM_CALL(hipDeviceGetAttribute(&value, hipDeviceAttributeMaxSharedMemoryPerBlock,
                                        ctx.device_id));
        break;
      }
      case kComputeVersion: {
        std::ostringstream os;
        ROCM_CALL(
            hipDeviceGetAttribute(&value, hipDeviceAttributeComputeCapabilityMajor, ctx.device_id));
        os << value << ".";
        ROCM_CALL(
            hipDeviceGetAttribute(&value, hipDeviceAttributeComputeCapabilityMinor, ctx.device_id));
        os << value;
        *rv = os.str();
        return;
      }
      case kDeviceName: {
        std::string name(256, 0);
        ROCM_CALL(hipDeviceGetName(&name[0], name.size(), ctx.device_id));
        name.resize(strlen(name.c_str()));
        *rv = std::move(name);
        return;
      }
      case kMaxClockRate: {
        ROCM_CALL(hipDeviceGetAttribute(&value, hipDeviceAttributeClockRate, ctx.device_id));
        break;
      }
      case kMultiProcessorCount: {
        ROCM_CALL(
            hipDeviceGetAttribute(&value, hipDeviceAttributeMultiprocessorCount, ctx.device_id));
        break;
      }
      case kMaxThreadDimensions: {
        int dims[3];
        ROCM_CALL(hipDeviceGetAttribute(&dims[0], hipDeviceAttributeMaxBlockDimX, ctx.device_id));
        ROCM_CALL(hipDeviceGetAttribute(&dims[1], hipDeviceAttributeMaxBlockDimY, ctx.device_id));
        ROCM_CALL(hipDeviceGetAttribute(&dims[2], hipDeviceAttributeMaxBlockDimZ, ctx.device_id));

        std::stringstream ss;
        ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
        *rv = ss.str();
        return;
      }
      case kMaxRegistersPerBlock:
        ROCM_CALL(
            hipDeviceGetAttribute(&value, hipDeviceAttributeMaxRegistersPerBlock, ctx.device_id));
        break;
      case kGcnArch: {
        hipDeviceProp_t prop;
        ROCM_CALL(hipGetDeviceProperties(&prop, ctx.device_id));
        *rv = prop.gcnArch;
        return;
      }
      case kApiVersion: {
        *rv = HIP_VERSION;
        return;
      }
    }
    *rv = value;
  }
  void* AllocDataSpace(TVMContext ctx, size_t nbytes, size_t alignment,
                       DLDataType type_hint) final {
    ROCM_CALL(hipSetDevice(ctx.device_id));
    CHECK_EQ(256 % alignment, 0U) << "ROCM space is aligned at 256 bytes";
    void* ret;
    ROCM_CALL(hipMalloc(&ret, nbytes));
    return ret;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    ROCM_CALL(hipSetDevice(ctx.device_id));
    ROCM_CALL(hipFree(ptr));
  }

  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      TVMContext ctx_from, TVMContext ctx_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;
    if (ctx_from.device_type == kDLROCM && ctx_to.device_type == kDLROCM) {
      ROCM_CALL(hipSetDevice(ctx_from.device_id));
      if (ctx_from.device_id == ctx_to.device_id) {
        GPUCopy(from, to, size, hipMemcpyDeviceToDevice, hip_stream);
      } else {
        hipMemcpyPeerAsync(to, ctx_to.device_id, from, ctx_from.device_id, size, hip_stream);
      }
    } else if (ctx_from.device_type == kDLROCM && ctx_to.device_type == kDLCPU) {
      ROCM_CALL(hipSetDevice(ctx_from.device_id));
      GPUCopy(from, to, size, hipMemcpyDeviceToHost, hip_stream);
    } else if (ctx_from.device_type == kDLCPU && ctx_to.device_type == kDLROCM) {
      ROCM_CALL(hipSetDevice(ctx_to.device_id));
      GPUCopy(from, to, size, hipMemcpyHostToDevice, hip_stream);
    } else {
      LOG(FATAL) << "expect copy from/to GPU or between GPU";
    }
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
    ROCM_CALL(hipSetDevice(ctx.device_id));
    ROCM_CALL(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
  }

  void SetStream(TVMContext ctx, TVMStreamHandle stream) final {
    ROCMThreadEntry::ThreadLocal()->stream = static_cast<hipStream_t>(stream);
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, DLDataType type_hint) final {
    return ROCMThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
    ROCMThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
  }

  static ROCMDeviceAPI* Global() {
    static ROCMDeviceAPI* inst = new ROCMDeviceAPI();
    return inst;
  }

 private:
  static void GPUCopy(const void* from, void* to, size_t size, hipMemcpyKind kind,
                      hipStream_t stream) {
    if (stream != 0) {
      ROCM_CALL(hipMemcpyAsync(to, from, size, kind, stream));
    } else {
      ROCM_CALL(hipMemcpy(to, from, size, kind));
    }
  }
};

typedef dmlc::ThreadLocalStore<ROCMThreadEntry> ROCMThreadStore;

ROCMThreadEntry::ROCMThreadEntry() : pool(kDLROCM, ROCMDeviceAPI::Global()) {}

ROCMThreadEntry* ROCMThreadEntry::ThreadLocal() { return ROCMThreadStore::Get(); }

TVM_REGISTER_GLOBAL("device_api.rocm").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = ROCMDeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});
}  // namespace runtime
}  // namespace tvm
