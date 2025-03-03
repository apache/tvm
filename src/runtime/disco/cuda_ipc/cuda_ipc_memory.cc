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
#include <tvm/runtime/disco/cuda_ipc_memory.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/registry.h>

#include "../../../../3rdparty/tensorrt_llm/custom_allreduce_kernels.h"
#include "../../cuda/cuda_common.h"
#include "../../memory/pooled_allocator.h"
#include "../nccl/nccl_context.h"

namespace tvm {
namespace runtime {
namespace cuda_ipc {

using tensorrt_llm::MAX_ALL_REDUCE_BLOCKS;
using tensorrt_llm::MAX_RANKS_PER_NODE;
using tvm::runtime::memory::Buffer;

/*!
 * \brief All-gather the IPC memory handles across all distributed workers.
 * On each worker, we copy the IPC handle to GPU memory. And nccl AllGather
 * is reused to all-gather the handles. Finally the all-gathered handles
 * on each worker are copied from GPU to CPU.
 */
std::vector<cudaIpcMemHandle_t> AllGatherIPCHandles(nccl::CCLThreadLocalContext* ctx,
                                                    cudaIpcMemHandle_t local_handle) {
  void *d_src, *d_dst;
  CUDA_CALL(cudaMalloc(&d_src, CUDA_IPC_HANDLE_SIZE));
  CUDA_CALL(cudaMalloc(&d_dst, CUDA_IPC_HANDLE_SIZE * ctx->worker->num_workers));
  CUDA_CALL(cudaMemcpy(d_src, &local_handle, CUDA_IPC_HANDLE_SIZE, cudaMemcpyHostToDevice));
  NCCL_CALL(ncclAllGather(d_src, d_dst, CUDA_IPC_HANDLE_SIZE, ncclChar, ctx->global_comm,
                          /*stream=*/nullptr));
  std::vector<char> serial_handles(CUDA_IPC_HANDLE_SIZE * ctx->worker->num_workers, 0);
  CUDA_CALL(cudaMemcpy(serial_handles.data(), d_dst,
                       CUDA_IPC_HANDLE_SIZE * ctx->worker->num_workers, cudaMemcpyDefault));
  std::vector<cudaIpcMemHandle_t> handles(ctx->worker->num_workers);
  for (int i = 0; i < ctx->worker->num_workers; ++i) {
    memcpy(handles[i].reserved, &serial_handles[i * CUDA_IPC_HANDLE_SIZE], CUDA_IPC_HANDLE_SIZE);
  }
  CUDA_CALL(cudaFree(d_src));
  CUDA_CALL(cudaFree(d_dst));
  return handles;
}

/*!
 * \brief The memory allocator of CUDAIPCMemory.
 * Overriding PooledAllocator for efficient memory management.
 */
class CUDAIPCMemoryAllocator final : public memory::PooledAllocator {
 public:
  explicit CUDAIPCMemoryAllocator() : PooledAllocator() {}

  bool AllowMemoryScope(const std::string& mem_scope) const final {
    // The allowed memory scope of CUDAIPCMemory is "ipc_memory";
    return mem_scope == "ipc_memory";
  }

  CUDAIPCMemory GetIPCMemoryFromDevicePtr(void* ptr) const {
    auto it = ipc_memory_map_.find(ptr);
    CHECK(it != ipc_memory_map_.end())
        << "The given pointer's CUDAIPCMemory object does not exist. Please use global function "
           "\"cuda_ipc.alloc_storage\" to allocate the CUDAIPCMemory object first.";
    return it->second;
  }

  /*! \brief Return the global CUDAIPCMemory singleton allocator. */
  static CUDAIPCMemoryAllocator* Global() {
    static CUDAIPCMemoryAllocator* allocator = new CUDAIPCMemoryAllocator();
    return allocator;
  }

 private:
  void* DeviceAllocDataSpace(Device dev, size_t size, size_t alignment,
                             DLDataType type_hint) final {
    auto [data_ptr, data_comm_ptrs] =
        AllocIPCMemory(dev, size, alignment, type_hint, /*reset_memory_to_zero=*/false);
    int barrier_ptr_size = sizeof(uint32_t) * (MAX_ALL_REDUCE_BLOCKS + 2) * MAX_RANKS_PER_NODE;
    auto [barrier_in_ptr, barrier_in_comm_ptrs] = AllocIPCMemory(
        dev, barrier_ptr_size, alignment, DataType::UInt(32), /*reset_memory_to_zero=*/true);
    auto [barrier_out_ptr, barrier_out_comm_ptrs] = AllocIPCMemory(
        dev, barrier_ptr_size, alignment, DataType::UInt(32), /*reset_memory_to_zero=*/true);

    // Create the CUDAIPCMemory object.
    ObjectPtr<CUDAIPCMemoryObj> ipc_memory = make_object<CUDAIPCMemoryObj>();
    nccl::CCLThreadLocalContext* nccl_ctx = nccl::CCLThreadLocalContext::Get();
    ipc_memory->remote_data = data_comm_ptrs;
    ipc_memory->barrier_in = barrier_in_comm_ptrs;
    ipc_memory->barrier_out = barrier_out_comm_ptrs;
    ipc_memory->barrier_flag = 1;
    ipc_memory->num_workers = nccl_ctx->worker->num_workers;
    ipc_memory->worker_id = nccl_ctx->worker->worker_id;
    ipc_memory_map_[data_ptr] = CUDAIPCMemory(std::move(ipc_memory));
    return data_ptr;
  }

  void DeviceFreeDataSpace(Device dev, void* ptr) final {
    ICHECK(dev.device_type == kDLCUDA);
    CUDA_CALL(cudaSetDevice(dev.device_id));
    nccl::CCLThreadLocalContext* ctx = nccl::CCLThreadLocalContext::Get();
    auto it = ipc_memory_map_.find(ptr);
    ICHECK(it != ipc_memory_map_.end());
    FreeIPCMemory(it->second->remote_data, ctx->worker->worker_id);
    FreeIPCMemory(it->second->barrier_in, ctx->worker->worker_id);
    FreeIPCMemory(it->second->barrier_out, ctx->worker->worker_id);
    ipc_memory_map_.erase(it);
  }

  /*!
   * \brief Allocate CUDA memory with the required size, alignment and dtype,
   * and return the IPC memory data pointers.
   * \returns The local data pointer of the allocated CUDA memory,
   * and a list of pointers that contains the CUDA IPC memory pointer
   * of the allocated memory on each worker.
   * For the i-th pointer, if i is the worker id of the given device,
   * then the returned i-th pointer points to the local CUDA memory,
   * or otherwise it is an IPC memory pointer.
   * \details This function first allocates local memory on every worker,
   * and creates an IPC memory pointer for the local memory.
   * Then it uses nccl all-gather to synchronize the IPC memory pointers
   * across all workers, so that every worker know each other's IPC memory
   * pointer.
   */
  std::pair<void*, std::vector<void*>> AllocIPCMemory(Device dev, size_t size, size_t alignment,
                                                      DLDataType type_hint,
                                                      bool reset_memory_to_zero) {
    // Alloc local buffer
    ICHECK(dev.device_type == kDLCUDA);
    void* ptr;
    CUDA_CALL(cudaSetDevice(dev.device_id));
    CUDA_CALL(cudaMalloc(&ptr, size));
    // Reset allocated memory to zero when required.
    // We explicitly synchronize after memset, to make sure memset finishes
    // before using all-gather to exchange IPC handles.
    // This is important to ensure the memory reset get ordered
    // before any other peers read the memory.
    if (reset_memory_to_zero) {
      CUDA_CALL(cudaMemset(ptr, 0, size));
      CUDA_CALL(cudaDeviceSynchronize());
    }
    // Create ipc handle
    cudaIpcMemHandle_t local_handle;
    CUDA_CALL(cudaIpcGetMemHandle(&local_handle, ptr));
    // All-gather IPC handles.
    nccl::CCLThreadLocalContext* ctx = nccl::CCLThreadLocalContext::Get();
    std::vector<cudaIpcMemHandle_t> handles = AllGatherIPCHandles(ctx, local_handle);
    // Collect the all-gather results.
    std::vector<void*> comm_ptrs(ctx->worker->num_workers);
    for (size_t node_id = 0; node_id < handles.size(); ++node_id) {
      if (static_cast<int>(node_id) == ctx->worker->worker_id) {
        comm_ptrs[node_id] = ptr;
      } else {
        uint8_t* foreign_buffer;
        CUDA_CALL(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&foreign_buffer), handles[node_id],
                                       cudaIpcMemLazyEnablePeerAccess));
        comm_ptrs[node_id] = foreign_buffer;
      }
    }
    return std::make_pair(ptr, comm_ptrs);
  }

  /*! \brief Free the IPC memory pointers. */
  void FreeIPCMemory(std::vector<void*> comm_ptrs, int worker_id) {
    for (int i = 0; i < static_cast<int>(comm_ptrs.size()); ++i) {
      if (i != worker_id) {
        // Free ipc handle.
        CUDA_CALL(cudaIpcCloseMemHandle(comm_ptrs[i]));
      } else {
        // Free local buffer.
        CUDA_CALL(cudaFree(comm_ptrs[i]));
      }
    }
  }

  /*! \brief The mapping from local CUDA memory pointer to its allocated CUDAIPCMemory object. */
  std::unordered_map<void*, CUDAIPCMemory> ipc_memory_map_;
};

/*!
 * \brief Allocate a storage object with CUDA IPC memory.
 * \param buffer_shape The shape of the storage to allocate.
 * \param dtype_hint The dtype of the storage to allocate.
 * \return The allocated storage object with internal CUDA IPC memory buffer.
 */
memory::Storage IPCAllocStorage(ShapeTuple buffer_shape, DLDataType dtype_hint) {
  auto storage_obj = runtime::SimpleObjAllocator().make_object<memory::StorageObj>();
  nccl::CCLThreadLocalContext* nccl_ctx = nccl::CCLThreadLocalContext::Get();
  Device device{DLDeviceType::kDLCUDA, nccl_ctx->device_id};
  CUDAIPCMemoryAllocator* allocator = CUDAIPCMemoryAllocator::Global();
  storage_obj->buffer = CUDAIPCMemoryAllocator::Global()->Alloc(
      device, std::move(buffer_shape), dtype_hint, /*mem_scope=*/"ipc_memory");
  storage_obj->allocator = allocator;
  memory::Storage storage(storage_obj);
  return storage;
}

TVM_REGISTER_GLOBAL("runtime.disco.cuda_ipc.alloc_storage").set_body_typed(IPCAllocStorage);

TVM_REGISTER_GLOBAL("runtime.disco.cuda_ipc.cuda_ipc_memory_allocator_clear").set_body_typed([]() {
  CUDAIPCMemoryAllocator::Global()->Clear();
});

/******************** CUDAIPCMemoryObj ********************/

TVM_REGISTER_OBJECT_TYPE(CUDAIPCMemoryObj);

// Direct to CUDAIPCMemoryAllocator::Global.
memory::Allocator* CUDAIPCMemory::GlobalAllocator() { return CUDAIPCMemoryAllocator::Global(); }

// Direct to CUDAIPCMemoryAllocator::GlobalGetIPCMemoryFromDevicePtr.
CUDAIPCMemory CUDAIPCMemory::GetIPCMemoryFromDevicePtr(void* ptr) {
  return CUDAIPCMemoryAllocator::Global()->GetIPCMemoryFromDevicePtr(ptr);
}

}  // namespace cuda_ipc
}  // namespace runtime
}  // namespace tvm
