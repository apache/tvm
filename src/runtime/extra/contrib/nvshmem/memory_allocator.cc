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
#include <nvshmem.h>
#include <nvshmemx.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>

#include <memory>
#include <mutex>
#include <vector>

#include "../../disco/utils.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Tensor pool for the process-wide NVSHMEM symmetric heap.
 *
 * Allocations, tensor lifetimes and finalization must be collective and ordered
 * consistently across PEs, as required by nvshmem_align/nvshmem_free. In particular,
 * allocation failure must not trigger a unilateral cache eviction and retry.
 * Corresponding calls on every PE must use matching thread/stream pool histories.
 */
class NVSHMEMPool final : public TensorMemoryPool {
 public:
  NVSHMEMPool(uint64_t thread, TVMStreamHandle stream) : thread_(thread), stream_(stream) {}

  static ffi::Tensor EmptyForSession(ffi::ShapeView shape, DLDataType dtype, DLDevice device) {
    auto* heap = Heap::Global();
    std::lock_guard<std::mutex> lock(heap->mutex);
    CheckInitialized();
    TVM_FFI_CHECK(device.device_type == kDLCUDA, ValueError)
        << "NVSHMEM can only allocate CUDA device memory";
    int current_device;
    TVM_FFI_CHECK_CUDA_ERROR(cudaGetDevice(&current_device));
    TVM_FFI_CHECK_EQ(device.device_id, current_device, ValueError)
        << "NVSHMEM tensors must use the current device of the symmetric heap";
    TVM_FFI_CHECK(heap->device_id == -1 || heap->device_id == device.device_id, ValueError)
        << "NVSHMEM tensors cannot share a symmetric heap across devices";
    heap->device_id = device.device_id;
    // Native thread identifiers may be reused after a worker exits.
    static thread_local uint64_t thread = heap->next_thread_id++;
    auto stream = TVMFFIEnvGetStream(device.device_type, device.device_id);
    for (auto& pool : heap->pools) {
      if (pool->thread_ == thread && pool->stream_ == stream) {
        return pool->Empty(shape, dtype, device);
      }
    }
    auto pool = ffi::make_object<NVSHMEMPool>(thread, stream);
    heap->pools.push_back(pool);
    return pool->Empty(shape, dtype, device);
  }

  ffi::Tensor Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                    ffi::Optional<ffi::String> mem_scope = std::nullopt) final {
    TVM_FFI_CHECK(device.device_type == kDLCUDA, ValueError)
        << "NVSHMEM can only allocate CUDA device memory";
    TVM_FFI_CHECK(!mem_scope.has_value() || mem_scope.value() == "nvshmem", ValueError)
        << "NVSHMEM memory requires the nvshmem scope";
    TVM_FFI_CHECK(dtype.code == kDLInt || dtype.code == kDLUInt || dtype.code == kDLFloat ||
                      dtype.code == kDLBfloat,
                  ValueError)
        << "NVSHMEM supports int, unsigned int, float, and bfloat tensors";
    return TensorMemoryPool::Empty(shape, dtype, device);
  }

  static void Finalize() {
    auto* heap = Heap::Global();
    std::lock_guard<std::mutex> lock(heap->mutex);
    CheckInitialized();
    for (auto& pool : heap->pools) {
      TVM_FFI_CHECK(!pool->HasLiveAllocations(), ValueError)
          << "Release all NVSHMEM tensors and views before finalizing NVSHMEM";
    }
    // Finalization is collective. Complete kernels before releasing the symmetric
    // heap, including work enqueued on a stream other than the finalizing thread's.
    int device_id = heap->device_id;
    if (device_id == -1) TVM_FFI_CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
    ffi::CUDADeviceGuard guard(device_id);
    TVM_FFI_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // Pools follow first-use order; each pool releases backing in allocation order.
    // The application must issue the same collective allocation sequence on all PEs.
    for (auto& pool : heap->pools) pool->Clear();
    heap->pools.clear();
    nvshmem_finalize();
    heap->device_id = -1;
  }

 private:
  struct Heap {
    std::mutex mutex;
    int device_id = -1;
    uint64_t next_thread_id = 0;
    std::vector<ffi::ObjectPtr<NVSHMEMPool>> pools;
    static Heap* Global() {
      // Collective teardown is explicit, never part of process static destruction.
      static auto* heap = new Heap();
      return heap;
    }
  };

  static void CheckInitialized() {
    int status = nvshmemx_init_status();
    TVM_FFI_CHECK(status == NVSHMEM_STATUS_IS_INITIALIZED || status == NVSHMEM_STATUS_LIMITED_MPG ||
                      status == NVSHMEM_STATUS_FULL_MPG,
                  ValueError)
        << "NVSHMEM must be initialized before allocating or finalizing tensors";
  }

  ffi::Tensor AllocBacking(size_t nbytes, size_t alignment, DLDevice device) final {
    struct Allocation {
      void* data;
      int device_id;
      ~Allocation() {
        // The owner is independent of the pool; finalize forbids outstanding leases.
        if (data) {
          ffi::CUDADeviceGuard guard(device_id);
          nvshmem_free(data);
        }
      }
    };
    ffi::CUDADeviceGuard guard(device.device_id);
    // Construct the independent owner before allocating the collective resource.
    auto owner = std::make_shared<Allocation>();
    owner->device_id = device.device_id;
    owner->data = nvshmem_align(alignment, nbytes);
    TVM_FFI_CHECK(owner->data != nullptr, InternalError) << "NVSHMEM allocation failed";
    struct Backing {
      std::shared_ptr<Allocation> owner;
      void AllocData(DLTensor* tensor) { tensor->data = owner->data; }
      void FreeData(DLTensor*) { owner.reset(); }
    };
    int64_t extent = static_cast<int64_t>(nbytes);
    return ffi::Tensor::FromNDAlloc(Backing{std::move(owner)}, ffi::ShapeView(&extent, 1),
                                    DLDataType{kDLUInt, 8, 1}, device);
  }

  uint64_t thread_;
  TVMStreamHandle stream_;
};

ffi::Tensor NVSHMEMEmpty(ffi::Shape shape, DLDataType dtype, ffi::Optional<Device> device) {
  Device resolved_device =
      device.has_value() ? device.value() : UseDefaultDeviceIfNone(std::nullopt);
  return NVSHMEMPool::EmptyForSession(shape, dtype, resolved_device);
}

void NVSHMEMFinalize() { NVSHMEMPool::Finalize(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.disco.nvshmem.empty", NVSHMEMEmpty)
      .def("runtime.disco.nvshmem.finalize_nvshmem", NVSHMEMFinalize);
}

}  // namespace runtime
}  // namespace tvm
