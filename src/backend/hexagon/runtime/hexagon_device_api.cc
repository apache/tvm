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
 * \file hexagon_device_api.cc
 */

#include "hexagon_device_api.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/runtime/tensor.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "../../../runtime/workspace_pool.h"
#include "hexagon_common.h"

namespace tvm {
namespace runtime {
namespace hexagon {

HexagonDeviceAPI* HexagonDeviceAPI::Global() {
  static auto* inst = new HexagonDeviceAPI();
  return inst;
}

ffi::Any HexagonDeviceAPI::GetAttr(Device dev, const ffi::String& name) {
  if (name == "exist") {
    return IsValidDevice(dev) && dev.device_id == 0;
  }
  return {};
}

ffi::Tensor HexagonDeviceAPI::Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                                    ffi::Optional<runtime::MemoryScope> mem_scope) {
  TVM_FFI_ICHECK(runtime_hexbuffs) << "Call AcquireResources before allocating Hexagon tensors";
  struct NDAlloc {
    HexagonDeviceAPI* api;
    // Each VTCM allocation retains its device resources; DDR tensors need only
    // their original buffer manager, even after the creating session ends.
    std::shared_ptr<HexagonBufferManager> buffers;
    ffi::Optional<runtime::MemoryScope> scope;
    void AllocData(DLTensor* tensor) {
      tensor->data =
          api->AllocDataSpace(tensor->device, tensor->ndim, tensor->shape, tensor->dtype, scope);
    }
    void FreeData(DLTensor* tensor) { buffers->FreeHexagonBuffer(tensor->data); }
  };
  return ffi::Tensor::FromNDAlloc(NDAlloc{this, runtime_hexbuffs, mem_scope}, shape, dtype, device);
}

// DataSpace: static allocations for Hexagon
void* HexagonDeviceAPI::AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                                       ffi::Optional<runtime::MemoryScope> mem_scope) {
  TVM_FFI_ICHECK(shape || ndim == 0)
      << "shape array is null for a non-scalar tensor, ndim = " << ndim;
  TVM_FFI_ICHECK(IsValidDevice(dev)) << "dev.device_type: " << dev.device_type;

  // IMPORTANT NOTE!
  // Hexagon treats "global" memory scope VERY DIFFERENTLY from all the others.
  //
  // With "global":
  //    - As with "global.ddr", this uses the target device's DDR memory.
  //    - The memory allocation must be a single, contiguous region of
  //      (virtual) memory addresses.
  //    - 'ndim' and 'shape' give the dimensions of the tensor to be stored
  //      in this allocation.  There's no (practical) limit on the maximum
  //      rank (ndim) of the tensor.
  //
  // All other supported memory-scope names:
  //   - 'ndim' must be exactly 1 or 2:
  //      1: A single, contiguous region of memory is requested.
  //      2: A two-level memory allocation is required, suitable for storing a tensor
  //         in Hexagon's "indirect tensor" format:
  //         - shape[0] indicates the number of tensor-content memory allocations.
  //         - shape[1] indicates the size of each tensor-content memory allocation.
  if (!mem_scope.has_value() || mem_scope.value() == runtime::MemoryScope::Global()) {
    DLTensor tensor{};
    tensor.device = dev;
    tensor.ndim = ndim;
    tensor.shape = const_cast<int64_t*>(shape);
    tensor.dtype = dtype;
    size_t alignment = std::max(static_cast<size_t>(kAllocAlignment),
                                static_cast<size_t>((dtype.bits * dtype.lanes + 7) / 8));
    return AllocDataSpace(dev, std::max(size_t{1}, ffi::GetDataSize(tensor)), alignment, dtype);
  }

  // NOTE: This check should be superfluous, but it's probably a good idea to leave it in
  // until the AoT executor's multi-device dispatch code is mature. --cconvey 2022-08-26
  TVM_FFI_ICHECK(dev.device_type == kDLHexagon)
      << "dev.device_type: " << dev.device_type << " DeviceName(" << dev.device_type
      << "): " << DLDeviceType2Str(dev.device_type) << "";

  TVM_FFI_ICHECK(ndim >= 0 && ndim <= 2)
      << "Hexagon Device API supports only 1d and 2d allocations, but received ndim = " << ndim;

  const size_t typesize = (dtype.bits * dtype.lanes + 7) / 8;

  TVM_FFI_ICHECK(runtime_hexbuffs)
      << "Attempted to allocate Hexagon data with "
      << "HexagonDeviceAPI::AllocDataSpace before initializing resources.  "
      << "Please call HexagonDeviceAPI::AcquireResources";

  if (ndim == 0) {
    // Allocate storage for a single scalar value.
    return runtime_hexbuffs->AllocateHexagonBuffer(typesize, kHexagonAllocAlignment, mem_scope);
  } else if (ndim == 1) {
    // Allocate a single, contiguous memory region.
    size_t nbytes = std::max(size_t{1}, shape[0] * typesize);
    return runtime_hexbuffs->AllocateHexagonBuffer(nbytes, kHexagonAllocAlignment, mem_scope);
  } else if (ndim == 2) {
    // Allocate the region(s) needed for Hexagon's indirect-tensor format.
    size_t nallocs = std::max(int64_t{1}, shape[0]);
    size_t nbytes = std::max(size_t{1}, shape[1] * typesize);
    return runtime_hexbuffs->AllocateHexagonBuffer(nallocs, nbytes, kHexagonAllocAlignment,
                                                   mem_scope);
  } else {
    return nullptr;  // unreachable
  }
}

void* HexagonDeviceAPI::AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                                       DLDataType type_hint) {
  TVM_FFI_ICHECK(nbytes) << "number of bytes is zero";
  TVM_FFI_ICHECK(alignment) << "alignment is zero";
  TVM_FFI_ICHECK(IsValidDevice(dev)) << "dev.device_type: " << dev.device_type;
  if (alignment < kHexagonAllocAlignment) {
    alignment = kHexagonAllocAlignment;
  }
  TVM_FFI_ICHECK(runtime_hexbuffs)
      << "Attempted to allocate Hexagon data with "
      << "HexagonDeviceAPI::AllocDataSpace before initializing resources.  "
      << "Please call HexagonDeviceAPI::AcquireResources";
  return runtime_hexbuffs->AllocateHexagonBuffer(nbytes, alignment, runtime::MemoryScope::Global());
}

void HexagonDeviceAPI::FreeDataSpace(Device dev, void* ptr) {
  TVM_FFI_ICHECK(ptr) << "buffer pointer is null";
  TVM_FFI_ICHECK(IsValidDevice(dev)) << "dev.device_type: " << dev.device_type;
  if (runtime_hexbuffs) {
    runtime_hexbuffs->FreeHexagonBuffer(ptr);
  } else {
    // Either AcquireResources was never called, or ReleaseResources was called.  Since this can
    // occur in the normal course of shutdown, log a message and continue.
    DLOG(INFO) << "FreeDataSpace called outside a session for " << ptr;
  }
}

void* HexagonDeviceAPI::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  TVM_FFI_ICHECK(IsValidDevice(dev)) << "dev.device_type: " << dev.device_type;
  TVM_FFI_ICHECK(runtime_hexbuffs) << "Call AcquireResources before allocating Hexagon workspace";
  std::lock_guard<std::mutex> lock(workspace_mutex_);
  auto& pool = workspace_pools_[std::this_thread::get_id()];
  if (!pool) pool = std::make_unique<WorkspacePool>(kDLHexagon, this);
  return pool->AllocWorkspace(dev, size);
}

void HexagonDeviceAPI::FreeWorkspace(Device dev, void* data) {
  TVM_FFI_ICHECK(IsValidDevice(dev)) << "dev.device_type: " << dev.device_type;
  std::lock_guard<std::mutex> lock(workspace_mutex_);
  auto it = workspace_pools_.find(std::this_thread::get_id());
  TVM_FFI_ICHECK(it != workspace_pools_.end())
      << "Attempt made to free unknown workspace allocation";
  it->second->FreeWorkspace(dev, data);
}

void HexagonDeviceAPI::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  TVM_FFI_ICHECK_EQ(from->byte_offset, 0);
  TVM_FFI_ICHECK_EQ(to->byte_offset, 0);
  TVM_FFI_ICHECK_EQ(ffi::GetDataSize(*from), ffi::GetDataSize(*to));
  TVM_FFI_ICHECK(runtime_hexbuffs)
      << "Attempted to copy Hexagon data with "
      << "HexagonDeviceAPI::CopyDataFromTo before initializing resources.  "
      << "Please call HexagonDeviceAPI::AcquireResources";

  auto lookup_hexagon_buffer = [this](void* ptr) -> HexagonBuffer* {
    return runtime_hexbuffs->FindHexagonBuffer(ptr);
  };

  HexagonBuffer* hex_from_buf = lookup_hexagon_buffer(from->data);
  HexagonBuffer* hex_to_buf = lookup_hexagon_buffer(to->data);

  if (hex_from_buf && hex_to_buf) {
    hex_to_buf->CopyFrom(*hex_from_buf, ffi::GetDataSize(*from));
  } else if (hex_to_buf) {
    hex_to_buf->CopyFrom(from->data, ffi::GetDataSize(*from));
  } else if (hex_from_buf) {
    hex_from_buf->CopyTo(to->data, ffi::GetDataSize(*to));
  } else {
    TVM_FFI_ICHECK(false)
        << "CopyDataFromTo requested between src and dst which are not managed by the "
           "hexagon device api.";
  }
}

void HexagonDeviceAPI::CopyDataFromTo(const void* from, size_t from_offset, void* to,
                                      size_t to_offset, size_t size, Device dev_from, Device dev_to,
                                      DLDataType type_hint, TVMStreamHandle stream) {
  memcpy(static_cast<char*>(to) + to_offset, static_cast<const char*>(from) + from_offset, size);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("device_api.hexagon.dma_copy_dltensor",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    auto dst = args[0].cast<DLTensor*>();
                    auto src = args[1].cast<DLTensor*>();
                    int size = args[2].cast<int>();
                    TVM_FFI_ICHECK(size > 0);
                    bool bypass_cache = args[3].cast<bool>();

                    int ret = DMA_RETRY;
                    do {
                      ret = HexagonDeviceAPI::Global()->UserDMA()->Copy(
                          SYNC_DMA_QUEUE, dst->data, src->data, size, bypass_cache);
                    } while (ret == DMA_RETRY);
                    TVM_FFI_ICHECK(ret == DMA_SUCCESS);
                    HexagonDeviceAPI::Global()->UserDMA()->Wait(SYNC_DMA_QUEUE, 0);

                    *rv = static_cast<int32_t>(0);
                  })
      .def_packed("device_api.hexagon.dma_copy",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    uint32_t queue_id = args[0].cast<uint32_t>();
                    void* dst = args[1].cast<void*>();
                    void* src = args[2].cast<void*>();
                    uint32_t size = args[3].cast<uint32_t>();
                    TVM_FFI_ICHECK(size > 0);
                    bool bypass_cache = args[4].cast<bool>();

                    int ret = DMA_RETRY;
                    do {
                      ret = HexagonDeviceAPI::Global()->UserDMA()->Copy(queue_id, dst, src, size,
                                                                        bypass_cache);
                    } while (ret == DMA_RETRY);
                    TVM_FFI_ICHECK(ret == DMA_SUCCESS);
                    *rv = static_cast<int32_t>(ret);
                  })
      .def_packed("device_api.hexagon.dma_wait",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    uint32_t queue_id = args[0].cast<uint32_t>();
                    int inflight = args[1].cast<int>();
                    TVM_FFI_ICHECK(inflight >= 0);
                    HexagonDeviceAPI::Global()->UserDMA()->Wait(queue_id, inflight);
                    *rv = static_cast<int32_t>(0);
                  })
      .def_packed("device_api.hexagon.dma_start_group",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    uint32_t queue_id = args[0].cast<uint32_t>();
                    HexagonDeviceAPI::Global()->UserDMA()->StartGroup(queue_id);
                    *rv = static_cast<int32_t>(0);
                  })
      .def_packed("device_api.hexagon.dma_end_group",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    uint32_t queue_id = args[0].cast<uint32_t>();
                    HexagonDeviceAPI::Global()->UserDMA()->EndGroup(queue_id);
                    *rv = static_cast<int32_t>(0);
                  })
      .def_packed("device_api.hexagon.alloc_nd",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    static const auto scope_global_vtcm = runtime::MemoryScope::Get("global.vtcm");
                    int32_t device_type = args[0].cast<int32_t>();
                    int32_t device_id = args[1].cast<int32_t>();
                    int32_t dtype_code_hint = args[2].cast<int32_t>();
                    int32_t dtype_bits_hint = args[3].cast<int32_t>();
                    auto scope = args[4].cast<runtime::MemoryScope>();
                    TVM_FFI_ICHECK(scope == scope_global_vtcm);
                    int64_t ndim = args[5].cast<int64_t>();
                    TVM_FFI_ICHECK((ndim == 1 || ndim == 2) &&
                                   "Hexagon Device API supports only 1d and 2d allocations");
                    int64_t* shape = static_cast<int64_t*>(args[6].cast<void*>());

                    Device dev;
                    dev.device_type = static_cast<DLDeviceType>(device_type);
                    dev.device_id = device_id;

                    DLDataType type_hint;
                    type_hint.code = static_cast<decltype(type_hint.code)>(dtype_code_hint);
                    type_hint.bits = static_cast<decltype(type_hint.bits)>(dtype_bits_hint);
                    type_hint.lanes = 1;

                    HexagonDeviceAPI* hexapi = HexagonDeviceAPI::Global();
                    *rv = hexapi->AllocDataSpace(dev, ndim, shape, type_hint, scope);
                  })
      .def_packed("device_api.hexagon.free_nd",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    static const auto scope_global_vtcm = runtime::MemoryScope::Get("global.vtcm");
                    int32_t device_type = args[0].cast<int32_t>();
                    int32_t device_id = args[1].cast<int32_t>();
                    auto scope = args[2].cast<runtime::MemoryScope>();
                    TVM_FFI_ICHECK(scope == scope_global_vtcm);
                    void* ptr = args[3].cast<void*>();

                    Device dev;
                    dev.device_type = static_cast<DLDeviceType>(device_type);
                    dev.device_id = device_id;

                    HexagonDeviceAPI* hexapi = HexagonDeviceAPI::Global();
                    hexapi->FreeDataSpace(dev, ptr);
                    *rv = static_cast<int32_t>(0);
                  })
      .def_packed("device_api.hexagon.acquire_resources",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    HexagonDeviceAPI* api = HexagonDeviceAPI::Global();
                    api->AcquireResources();
                  })
      .def_packed("device_api.hexagon.release_resources",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    HexagonDeviceAPI* api = HexagonDeviceAPI::Global();
                    api->ReleaseResources();
                  })
      .def_packed("device_api.hexagon.vtcm_device_bytes",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    HexagonDeviceAPI* api = HexagonDeviceAPI::Global();
                    *rv = static_cast<int32_t>(api->VtcmPool()->VtcmDeviceBytes());
                  })
      .def_packed("device_api.hexagon", [](ffi::PackedArgs args, ffi::Any* rv) {
        DeviceAPI* ptr = HexagonDeviceAPI::Global();
        *rv = static_cast<void*>(ptr);
      });
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
