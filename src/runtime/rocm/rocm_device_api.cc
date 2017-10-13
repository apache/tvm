/*!
 *  Copyright (c) 2017 by Contributors
 * \file rocm_device_api.cc
 * \brief GPU specific API
 */
#include <tvm/runtime/config.h>
#include <tvm/runtime/device_api.h>

#if TVM_ROCM_RUNTIME
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include "./rocm_common.h"

namespace tvm {
namespace runtime {

class ROCMDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final {
    ROCM_CALL(hipSetDevice(ctx.device_id));
  }
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
        value = 1024;
        break;
      }
      case kWarpSize: {
        value = 64;
        break;
      }
      case kComputeVersion:
        hipDeviceProp_t prop;
        ROCM_CALL(hipGetDeviceProperties(&prop, ctx.device_id));
        *rv = prop.gcnArch;
        return;
    }
    *rv = value;
  }
  void* AllocDataSpace(TVMContext ctx, size_t size, size_t alignment) final {
    ROCM_CALL(hipSetDevice(ctx.device_id));
    CHECK_EQ(256 % alignment, 0U)
        << "ROCM space is aligned at 256 bytes";
    void *ret;
    ROCM_CALL(hipMalloc(&ret, size));
    return ret;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    ROCM_CALL(hipSetDevice(ctx.device_id));
    ROCM_CALL(hipFree(ptr));
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMStreamHandle stream) final {
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;
    if (ctx_from.device_type == kROCM && ctx_to.device_type == kROCM) {
      ROCM_CALL(hipSetDevice(ctx_from.device_id));
      if (ctx_from.device_id == ctx_to.device_id) {
        GPUCopy(from, to, size, hipMemcpyDeviceToDevice, hip_stream);
      } else {
        hipMemcpyPeerAsync(to, ctx_to.device_id,
                            from, ctx_from.device_id,
                            size, hip_stream);
      }
    } else if (ctx_from.device_type == kROCM && ctx_to.device_type == kCPU) {
      ROCM_CALL(hipSetDevice(ctx_from.device_id));
      GPUCopy(from, to, size, hipMemcpyDeviceToHost, hip_stream);
    } else if (ctx_from.device_type == kCPU && ctx_to.device_type == kROCM) {
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
    ROCMThreadEntry::ThreadLocal()
        ->stream = static_cast<hipStream_t>(stream);
  }

  void* AllocWorkspace(TVMContext ctx, size_t size) final {
    return ROCMThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
    ROCMThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
  }

  static const std::shared_ptr<ROCMDeviceAPI>& Global() {
    static std::shared_ptr<ROCMDeviceAPI> inst =
        std::make_shared<ROCMDeviceAPI>();
    return inst;
  }

 private:
  static void GPUCopy(const void* from,
                      void* to,
                      size_t size,
                      hipMemcpyKind kind,
                      hipStream_t stream) {
    if (stream != 0) {
      ROCM_CALL(hipMemcpyAsync(to, from, size, kind, stream));
    } else {
      ROCM_CALL(hipMemcpy(to, from, size, kind));
    }
  }
};

typedef dmlc::ThreadLocalStore<ROCMThreadEntry> ROCMThreadStore;

ROCMThreadEntry::ROCMThreadEntry()
    : pool(kROCM, ROCMDeviceAPI::Global()) {
}

ROCMThreadEntry* ROCMThreadEntry::ThreadLocal() {
  return ROCMThreadStore::Get();
}

TVM_REGISTER_GLOBAL("device_api.rocm")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = ROCMDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_ROCM_RUNTIME
