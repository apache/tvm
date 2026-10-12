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

#include <cuda_runtime.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/disco/cuda_ipc_memory.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "../../../../../3rdparty/tensorrt_llm/custom_allreduce_kernels.h"
#include "../nccl/nccl_context.h"

namespace tvm {
namespace runtime {
namespace cuda_ipc {

using tensorrt_llm::MAX_ALL_REDUCE_BLOCKS;
using tensorrt_llm::MAX_RANKS_PER_NODE;

namespace {

// Weak entries allow lookup without retaining either the pool or its backing tensors.
struct IPCMemoryRegistry {
  std::mutex mutex;
  std::unordered_map<void*, std::weak_ptr<CUDAIPCMemory>> allocations;

  static IPCMemoryRegistry* Global() {
    static auto* registry = new IPCMemoryRegistry();
    return registry;
  }
};

void CheckCleanup(cudaError_t error) {
  if (error != cudaSuccess) {
    LOG(WARNING) << "CUDA IPC cleanup: " << cudaGetErrorString(error);
  }
}

struct CUDAAllocation {
  void* data = nullptr;
  ~CUDAAllocation() {
    if (data) CheckCleanup(cudaFree(data));
  }
};

/*! \brief All-gather IPC handles, retaining temporary buffers on failure. */
std::vector<cudaIpcMemHandle_t> AllGatherIPCHandles(nccl::CCLThreadLocalContext* ctx,
                                                    cudaIpcMemHandle_t local_handle) {
  CUDAAllocation src, dst;
  size_t bytes = CUDA_IPC_HANDLE_SIZE * ctx->worker->num_workers;
  TVM_FFI_CHECK_CUDA_ERROR(cudaMalloc(&src.data, CUDA_IPC_HANDLE_SIZE));
  TVM_FFI_CHECK_CUDA_ERROR(cudaMalloc(&dst.data, bytes));
  TVM_FFI_CHECK_CUDA_ERROR(
      cudaMemcpy(src.data, &local_handle, CUDA_IPC_HANDLE_SIZE, cudaMemcpyHostToDevice));
  NCCL_CALL(ncclAllGather(src.data, dst.data, CUDA_IPC_HANDLE_SIZE, ncclChar, ctx->global_comm,
                          /*stream=*/nullptr));
  std::vector<cudaIpcMemHandle_t> handles(ctx->worker->num_workers);
  TVM_FFI_CHECK_CUDA_ERROR(cudaMemcpy(handles.data(), dst.data, bytes, cudaMemcpyDeviceToHost));
  return handles;
}

struct IPCBuffer {
  int device_id;
  int worker_id;
  void* local = nullptr;
  std::vector<void*> peers;

  IPCBuffer(int device_id, int worker_id, int num_workers)
      : device_id(device_id), worker_id(worker_id), peers(num_workers, nullptr) {}

  ~IPCBuffer() {
    // Destruction may run on another thread, or after its NCCL context has been cleared.
    int previous = 0;
    cudaError_t queried = cudaGetDevice(&previous);
    CheckCleanup(queried);
    CheckCleanup(cudaSetDevice(device_id));
    for (size_t i = 0; i < peers.size(); ++i) {
      if (static_cast<int>(i) != worker_id && peers[i]) {
        CheckCleanup(cudaIpcCloseMemHandle(peers[i]));
      }
    }
    if (local) CheckCleanup(cudaFree(local));
    if (queried == cudaSuccess) CheckCleanup(cudaSetDevice(previous));
  }

  void Allocate(size_t size, bool reset, nccl::CCLThreadLocalContext* ctx) {
    TVM_FFI_CHECK_CUDA_ERROR(cudaMalloc(&local, size));
    peers[worker_id] = local;
    if (reset) {
      TVM_FFI_CHECK_CUDA_ERROR(cudaMemset(local, 0, size));
      // Peers must not observe a barrier before its initialization finishes.
      TVM_FFI_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    cudaIpcMemHandle_t local_handle;
    TVM_FFI_CHECK_CUDA_ERROR(cudaIpcGetMemHandle(&local_handle, local));
    auto handles = AllGatherIPCHandles(ctx, local_handle);
    for (size_t i = 0; i < handles.size(); ++i) {
      if (static_cast<int>(i) != worker_id) {
        TVM_FFI_CHECK_CUDA_ERROR(
            cudaIpcOpenMemHandle(&peers[i], handles[i], cudaIpcMemLazyEnablePeerAccess));
      }
    }
  }
};

}  // namespace

/*! \brief Resource owner independent of the allocating thread and its pool. */
class CUDAIPCAllocation {
 public:
  IPCBuffer data;
  IPCBuffer barrier_in;
  IPCBuffer barrier_out;

  CUDAIPCAllocation(int device_id, int worker_id, int num_workers)
      : data(device_id, worker_id, num_workers),
        barrier_in(device_id, worker_id, num_workers),
        barrier_out(device_id, worker_id, num_workers) {}

  ~CUDAIPCAllocation() {
    auto* registry = IPCMemoryRegistry::Global();
    {
      std::lock_guard<std::mutex> lock(registry->mutex);
      registry->allocations.erase(data.local);
    }
    // Complete local kernels before closing any peer mappings. The original
    // stream or NCCL context need no longer exist on this destruction thread.
    int previous = 0;
    cudaError_t queried = cudaGetDevice(&previous);
    CheckCleanup(queried);
    CheckCleanup(cudaSetDevice(data.device_id));
    CheckCleanup(cudaDeviceSynchronize());
    if (queried == cudaSuccess) CheckCleanup(cudaSetDevice(previous));
  }
};

CUDAIPCMemoryObj::CUDAIPCMemoryObj(std::shared_ptr<CUDAIPCAllocation> allocation)
    : num_workers(static_cast<int>(allocation->data.peers.size())),
      worker_id(allocation->data.worker_id),
      remote_data(allocation->data.peers),
      barrier_in(allocation->barrier_in.peers),
      barrier_out(allocation->barrier_out.peers),
      allocation_(std::move(allocation)) {}

/*! \brief A pool belongs to one worker and one NCCL communicator lifetime. */
class CUDAIPCMemoryPool final : public TensorMemoryPool {
 public:
  explicit CUDAIPCMemoryPool(nccl::CCLThreadLocalContext* ctx, cudaStream_t stream)
      : generation_(ctx->generation),
        communicator_(ctx->global_comm),
        device_id_(ctx->device_id),
        worker_id_(ctx->worker->worker_id),
        num_workers_(ctx->worker->num_workers),
        stream_(stream) {}

  bool Matches(nccl::CCLThreadLocalContext* ctx) const {
    return generation_ == ctx->generation && communicator_ == ctx->global_comm &&
           device_id_ == ctx->device_id && worker_id_ == ctx->worker->worker_id &&
           num_workers_ == ctx->worker->num_workers;
  }

  ffi::Tensor Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                    ffi::Optional<ffi::String> mem_scope = std::nullopt) final {
    TVM_FFI_CHECK(!mem_scope.has_value() || mem_scope.value() == "ipc_memory", ValueError)
        << "CUDA IPC memory requires the ipc_memory scope";
    TVM_FFI_CHECK(device.device_type == kDLCUDA && device.device_id == device_id_, ValueError)
        << "CUDA IPC memory must use the communicator's device";
    return TensorMemoryPool::Empty(shape, dtype, device);
  }

  static CUDAIPCMemoryPool* ThreadLocal() {
    auto* ctx = nccl::CCLThreadLocalContext::Get();
    TVM_FFI_CHECK(ctx->worker && ctx->global_comm, ValueError)
        << "CUDA IPC allocation requires an initialized NCCL worker";
    auto& pools = ThreadLocalPools();
    if (!pools.empty() && !pools.front()->Matches(ctx)) {
      for (auto& pool : pools) pool->Clear();
      pools.clear();
    }
    cudaStream_t stream = ctx->GetDefaultStream();
    for (auto& pool : pools) {
      if (pool->stream_ == stream) return pool.get();
    }
    pools.push_back(ffi::make_object<CUDAIPCMemoryPool>(ctx, stream));
    return pools.back().get();
  }

  static void ClearThreadLocal() {
    for (auto& pool : ThreadLocalPools()) pool->Clear();
  }

 private:
  static std::vector<ffi::ObjectPtr<CUDAIPCMemoryPool>>& ThreadLocalPools() {
    static thread_local std::vector<ffi::ObjectPtr<CUDAIPCMemoryPool>> pools;
    return pools;
  }

  ffi::Tensor AllocBacking(size_t nbytes, size_t alignment, DLDevice device) final {
    TVM_FFI_CHECK(alignment != 0 && 256 % alignment == 0, ValueError)
        << "CUDA IPC space is aligned at 256 bytes";
    ffi::CUDADeviceGuard guard(device.device_id);
    auto* ctx = nccl::CCLThreadLocalContext::Get();
    TVM_FFI_ICHECK(Matches(ctx));
    auto allocation = std::make_shared<CUDAIPCAllocation>(device_id_, worker_id_, num_workers_);
    allocation->data.Allocate(nbytes, false, ctx);
    size_t barrier_bytes = sizeof(uint32_t) * (MAX_ALL_REDUCE_BLOCKS + 2) * MAX_RANKS_PER_NODE;
    allocation->barrier_in.Allocate(barrier_bytes, true, ctx);
    allocation->barrier_out.Allocate(barrier_bytes, true, ctx);
    auto memory = std::make_shared<CUDAIPCMemory>(
        ffi::UnsafeInit{}, ffi::make_object<CUDAIPCMemoryObj>(std::move(allocation)));
    struct Backing {
      std::shared_ptr<CUDAIPCMemory> memory;
      void AllocData(DLTensor* tensor) {
        tensor->data = (*memory)->remote_data[(*memory)->worker_id];
      }
      void FreeData(DLTensor*) { memory.reset(); }
    };
    int64_t extent = static_cast<int64_t>(nbytes);
    ffi::Tensor backing = ffi::Tensor::FromNDAlloc(Backing{memory}, ffi::ShapeView(&extent, 1),
                                                   DLDataType{kDLUInt, 8, 1}, device);
    auto* registry = IPCMemoryRegistry::Global();
    std::lock_guard<std::mutex> lock(registry->mutex);
    registry->allocations.emplace(backing.data_ptr(), memory);
    return backing;
  }

  uint64_t generation_;
  ncclComm_t communicator_;
  int device_id_;
  int worker_id_;
  int num_workers_;
  cudaStream_t stream_;
};

ffi::Tensor IPCEmpty(ffi::Shape shape, DLDataType dtype) {
  auto* ctx = nccl::CCLThreadLocalContext::Get();
  auto* pool = CUDAIPCMemoryPool::ThreadLocal();
  return pool->Empty(shape, dtype, Device{kDLCUDA, ctx->device_id}, ffi::String("ipc_memory"));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.disco.cuda_ipc.empty", IPCEmpty)
      .def("runtime.disco.cuda_ipc.cuda_ipc_memory_allocator_clear",
           CUDAIPCMemoryPool::ClearThreadLocal);
}

CUDAIPCMemory CUDAIPCMemory::GetIPCMemoryFromDevicePtr(void* ptr) {
  auto* registry = IPCMemoryRegistry::Global();
  std::shared_ptr<CUDAIPCMemory> memory;
  {
    std::lock_guard<std::mutex> lock(registry->mutex);
    auto it = registry->allocations.find(ptr);
    TVM_FFI_CHECK(it != registry->allocations.end(), ValueError)
        << "The pointer was not allocated by runtime.disco.cuda_ipc.empty";
    memory = it->second.lock();
  }
  TVM_FFI_CHECK(memory != nullptr, ValueError) << "The CUDA IPC allocation has expired";
  return *memory;
}

}  // namespace cuda_ipc
}  // namespace runtime
}  // namespace tvm
