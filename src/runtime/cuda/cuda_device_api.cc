/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */
#include <tvm/runtime/config.h>
#include <tvm/runtime/device_api.h>

#if TVM_CUDA_RUNTIME
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <cuda_runtime.h>
#include "./cuda_common.h"

namespace tvm {
namespace runtime {

class CUDADeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
  }
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    int value = 0;
    switch (kind) {
      case kExist:
        value = (
            cudaDeviceGetAttribute(
                &value, cudaDevAttrMaxThreadsPerBlock, ctx.device_id)
            == cudaSuccess);
        break;
      case kMaxThreadsPerBlock: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrMaxThreadsPerBlock, ctx.device_id));
        break;
      }
      case kWarpSize: {
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrWarpSize, ctx.device_id));
        break;
      }
      case kComputeVersion: {
        std::ostringstream os;
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrComputeCapabilityMajor, ctx.device_id));
        os << value << ".";
        CUDA_CALL(cudaDeviceGetAttribute(
            &value, cudaDevAttrComputeCapabilityMinor, ctx.device_id));
        os << value;
        *rv = os.str();
        return;
      }
    }
    *rv = value;
  }
  void* AllocDataSpace(TVMContext ctx, size_t size, size_t alignment) final {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    CHECK_EQ(256 % alignment, 0U)
        << "CUDA space is aligned at 256 bytes";
    void *ret;
    CUDA_CALL(cudaMalloc(&ret, size));
    return ret;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    CUDA_CALL(cudaFree(ptr));
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMStreamHandle stream) final {
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;
    if (ctx_from.device_type == kGPU && ctx_to.device_type == kGPU) {
      CUDA_CALL(cudaSetDevice(ctx_from.device_id));
      if (ctx_from.device_id == ctx_to.device_id) {
        GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
      } else {
        cudaMemcpyPeerAsync(to, ctx_to.device_id,
                            from, ctx_from.device_id,
                            size, cu_stream);
      }
    } else if (ctx_from.device_type == kGPU && ctx_to.device_type == kCPU) {
      CUDA_CALL(cudaSetDevice(ctx_from.device_id));
      GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
    } else if (ctx_from.device_type == kCPU && ctx_to.device_type == kGPU) {
      CUDA_CALL(cudaSetDevice(ctx_to.device_id));
      GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
    } else {
      LOG(FATAL) << "expect copy from/to GPU or between GPU";
    }
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

  void SetStream(TVMContext ctx, TVMStreamHandle stream) final {
    CUDAThreadEntry::ThreadLocal()
        ->stream = static_cast<cudaStream_t>(stream);
  }

  void* AllocWorkspace(TVMContext ctx, size_t size) final {
    return CUDAThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
    CUDAThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
  }

  static const std::shared_ptr<CUDADeviceAPI>& Global() {
    static std::shared_ptr<CUDADeviceAPI> inst =
        std::make_shared<CUDADeviceAPI>();
    return inst;
  }

 private:
  static void GPUCopy(const void* from,
                      void* to,
                      size_t size,
                      cudaMemcpyKind kind,
                      cudaStream_t stream) {
    if (stream != 0) {
      CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
    } else {
      CUDA_CALL(cudaMemcpy(to, from, size, kind));
    }
  }
};

typedef dmlc::ThreadLocalStore<CUDAThreadEntry> CUDAThreadStore;

CUDAThreadEntry::CUDAThreadEntry()
    : pool(kGPU, CUDADeviceAPI::Global()) {
}

CUDAThreadEntry* CUDAThreadEntry::ThreadLocal() {
  return CUDAThreadStore::Get();
}

TVM_REGISTER_GLOBAL("device_api.gpu")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = CUDADeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_CUDA_RUNTIME
