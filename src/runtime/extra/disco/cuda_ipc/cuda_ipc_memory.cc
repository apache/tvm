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

#include <cstdio>
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

struct IPCMemoryRegistry {
  std::mutex mutex;
  std::unordered_map<void*, std::weak_ptr<CUDAIPCMemory>> allocations;

  static IPCMemoryRegistry* Global() {
    static auto* registry = new IPCMemoryRegistry();
    return registry;
  }
};

bool CheckCleanup(cudaError_t error) noexcept {
  if (error == cudaSuccess) return true;
  std::fprintf(stderr, "CUDA IPC cleanup: %s\n", cudaGetErrorString(error));
  return false;
}

bool CheckCleanup(ncclResult_t error) noexcept {
  if (error == ncclSuccess) return true;
  std::fprintf(stderr, "CUDA IPC collective cleanup: %s\n", ncclGetErrorString(error));
  return false;
}

/*!
 * \brief Independently retained collective resources for IPC handle lifetime.
 *
 * Backings own this session. The pool retains only a weak reference, and neither
 * destruction nor handle exchange depends on a thread-local NCCL context. All
 * workers must issue corresponding allocations and releases in the same order.
 */
class IPCResourceSession {
 public:
  struct Message {
    uint64_t ordinal{0};
    uint64_t phase{0};
    uint64_t success{0};
    char payload[sizeof(ncclUniqueId)]{};
  };

  explicit IPCResourceSession(nccl::CCLThreadLocalContext* ctx)
      : device_id(ctx->device_id),
        worker_id(ctx->worker->worker_id),
        num_workers(ctx->worker->num_workers),
        received(num_workers) {}

  void Init(nccl::CCLThreadLocalContext* ctx) {
    TVM_FFI_CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    TVM_FFI_CHECK_CUDA_ERROR(cudaMalloc(&send, sizeof(Message)));
    TVM_FFI_CHECK_CUDA_ERROR(cudaMalloc(&receive, sizeof(Message) * num_workers));
    if (worker_id == 0) NCCL_CALL(ncclGetUniqueId(&unique_id));
    TVM_FFI_CHECK_CUDA_ERROR(
        cudaMemcpyAsync(send, &unique_id, sizeof(unique_id), cudaMemcpyHostToDevice, stream));
    NCCL_CALL(
        ncclBroadcast(send, receive, sizeof(unique_id), ncclChar, 0, ctx->global_comm, stream));
    TVM_FFI_CHECK_CUDA_ERROR(
        cudaMemcpyAsync(&unique_id, receive, sizeof(unique_id), cudaMemcpyDeviceToHost, stream));
    TVM_FFI_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    // Do not require ncclCommSplit: older supported NCCL versions have InitRank.
    NCCL_CALL(ncclCommInitRank(&communicator, num_workers, unique_id, worker_id));
    healthy = true;
  }

  ~IPCResourceSession() noexcept {
    int previous = 0;
    bool queried = CheckCleanup(cudaGetDevice(&previous));
    if (!CheckCleanup(cudaSetDevice(device_id))) return;
    if (communicator) {
      CheckCleanup(healthy ? ncclCommDestroy(communicator) : ncclCommAbort(communicator));
    }
    if (stream) CheckCleanup(cudaStreamSynchronize(stream));
    if (send) CheckCleanup(cudaFree(send));
    if (receive) CheckCleanup(cudaFree(receive));
    if (stream) CheckCleanup(cudaStreamDestroy(stream));
    if (queried) CheckCleanup(cudaSetDevice(previous));
  }

  // The caller holds mutex. A fixed envelope size lets mismatched allocation and
  // teardown phases fail closed instead of interpreting one another's payloads.
  bool Exchange(uint64_t ordinal, uint64_t phase, bool success,
                const cudaIpcMemHandle_t* handle = nullptr) noexcept {
    if (!healthy) return false;
    message = Message{};
    message.ordinal = ordinal;
    message.phase = phase;
    message.success = success;
    if (handle) std::memcpy(message.payload, handle, sizeof(*handle));
    if (!CheckCleanup(
            cudaMemcpyAsync(send, &message, sizeof(message), cudaMemcpyHostToDevice, stream)) ||
        !CheckCleanup(
            ncclAllGather(send, receive, sizeof(message), ncclChar, communicator, stream)) ||
        !CheckCleanup(cudaMemcpyAsync(received.data(), receive, sizeof(message) * num_workers,
                                      cudaMemcpyDeviceToHost, stream)) ||
        !CheckCleanup(cudaStreamSynchronize(stream))) {
      healthy = false;
      return false;
    }
    bool agreed = true;
    for (const auto& peer : received) {
      if (peer.ordinal != ordinal || peer.phase != phase) {
        std::fprintf(stderr,
                     "CUDA IPC collective allocation/release order differs across workers\n");
        healthy = false;
        return false;
      }
      agreed &= peer.success != 0;
    }
    return agreed;
  }

  std::mutex mutex;
  const int device_id;
  const int worker_id;
  const int num_workers;
  uint64_t next_ordinal{0};
  std::vector<Message> received;

 private:
  ncclUniqueId unique_id{};
  Message message;
  ncclComm_t communicator{nullptr};
  cudaStream_t stream{nullptr};
  void* send{nullptr};
  void* receive{nullptr};
  bool healthy{false};
};

struct IPCBuffer {
  void* local = nullptr;
  std::vector<void*> peers;

  explicit IPCBuffer(int num_workers) : peers(num_workers, nullptr) {}

  // Resources are released only by CUDAIPCAllocation's acknowledged protocol.
  bool CloseImports(int worker_id) noexcept {
    bool success = true;
    for (size_t i = 0; i < peers.size(); ++i) {
      if (static_cast<int>(i) != worker_id && peers[i]) {
        bool closed = CheckCleanup(cudaIpcCloseMemHandle(peers[i]));
        success &= closed;
        if (closed) peers[i] = nullptr;
      }
    }
    return success;
  }

  void FreeExport() noexcept {
    if (local) CheckCleanup(cudaFree(local));
    local = nullptr;
  }

  void Allocate(size_t size, bool reset, IPCResourceSession* session, uint64_t ordinal,
                uint64_t phase) {
    bool success = CheckCleanup(cudaMalloc(&local, size));
    peers[session->worker_id] = local;
    if (success && reset) {
      success = CheckCleanup(cudaMemset(local, 0, size));
      // Peers cannot access barriers until their initialization completes.
      if (success) success = CheckCleanup(cudaDeviceSynchronize());
    }
    cudaIpcMemHandle_t handle{};
    if (success) success = CheckCleanup(cudaIpcGetMemHandle(&handle, local));
    TVM_FFI_CHECK(session->Exchange(ordinal, phase, success, &handle), InternalError)
        << "CUDA IPC allocation/handle exchange failed on a worker";
    for (int i = 0; i < session->num_workers; ++i) {
      if (i != session->worker_id) {
        std::memcpy(&handle, session->received[i].payload, sizeof(handle));
        success &=
            CheckCleanup(cudaIpcOpenMemHandle(&peers[i], handle, cudaIpcMemLazyEnablePeerAccess));
      }
    }
    TVM_FFI_CHECK(session->Exchange(ordinal, phase + 1, success), InternalError)
        << "CUDA IPC peer mapping failed on a worker";
  }
};

}  // namespace

/*! \brief Resource owner independent of the allocating thread and its pool. */
class CUDAIPCAllocation {
 public:
  const std::shared_ptr<IPCResourceSession> session;
  const uint64_t ordinal;
  const uint64_t generation;
  void* const communicator;
  void* const stream;
  IPCBuffer data;
  IPCBuffer barrier_in;
  IPCBuffer barrier_out;

  CUDAIPCAllocation(std::shared_ptr<IPCResourceSession> session, uint64_t ordinal,
                    uint64_t generation, void* communicator, void* stream)
      : session(std::move(session)),
        ordinal(ordinal),
        generation(generation),
        communicator(communicator),
        stream(stream),
        data(this->session->num_workers),
        barrier_in(this->session->num_workers),
        barrier_out(this->session->num_workers) {}

  ~CUDAIPCAllocation() noexcept {
    // Never release an export until every importer confirms all its mappings closed.
    // In a failed runtime/collective, retaining exports is safer than dangling peer memory.
    int previous = 0;
    bool queried = CheckCleanup(cudaGetDevice(&previous));
    try {
      auto* registry = IPCMemoryRegistry::Global();
      {
        std::lock_guard<std::mutex> lock(registry->mutex);
        registry->allocations.erase(data.local);
      }
      std::lock_guard<std::mutex> lock(session->mutex);
      bool ready = CheckCleanup(cudaSetDevice(session->device_id));
      if (ready) ready = CheckCleanup(cudaDeviceSynchronize());
      if (ready) {
        ready = data.CloseImports(session->worker_id);
        ready &= barrier_in.CloseImports(session->worker_id);
        ready &= barrier_out.CloseImports(session->worker_id);
      }
      if (session->Exchange(ordinal, /*teardown=*/7, ready)) {
        data.FreeExport();
        barrier_in.FreeExport();
        barrier_out.FreeExport();
      } else {
        std::fprintf(stderr,
                     "CUDA IPC retaining exports: peer mapping closure was not confirmed\n");
      }
    } catch (...) {
      std::fprintf(stderr, "CUDA IPC retaining exports after unexpected cleanup failure\n");
    }
    if (queried) CheckCleanup(cudaSetDevice(previous));
  }
};

CUDAIPCMemoryObj::CUDAIPCMemoryObj(std::shared_ptr<CUDAIPCAllocation> allocation)
    : num_workers(allocation->session->num_workers),
      worker_id(allocation->session->worker_id),
      device_id(allocation->session->device_id),
      generation(allocation->generation),
      communicator(allocation->communicator),
      stream(allocation->stream),
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
    auto session = resource_session_.lock();
    if (!session) {
      session = std::make_shared<IPCResourceSession>(ctx);
      session->Init(ctx);
      resource_session_ = session;
    }
    // Keep the allocation outside the lock's scope: an error unwinds the lock
    // before the allocation's destructor enters its collective cleanup protocol.
    std::shared_ptr<CUDAIPCAllocation> allocation;
    {
      std::lock_guard<std::mutex> lock(session->mutex);
      uint64_t ordinal = session->next_ordinal++;
      allocation = std::make_shared<CUDAIPCAllocation>(session, ordinal, generation_, communicator_,
                                                       stream_);
      allocation->data.Allocate(nbytes, false, session.get(), ordinal, 1);
      size_t barrier_bytes = sizeof(uint32_t) * (MAX_ALL_REDUCE_BLOCKS + 2) * MAX_RANKS_PER_NODE;
      allocation->barrier_in.Allocate(barrier_bytes, true, session.get(), ordinal, 3);
      allocation->barrier_out.Allocate(barrier_bytes, true, session.get(), ordinal, 5);
    }
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

  std::weak_ptr<IPCResourceSession> resource_session_;
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
