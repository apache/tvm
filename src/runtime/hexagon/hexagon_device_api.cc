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

#include <dmlc/thread_local.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstdlib>
#include <cstring>

#include "../workspace_pool.h"
#include "hexagon_buffer.h"
#include "hexagon_common.h"
#include "qurt_memory.h"

namespace tvm {
namespace runtime {
namespace hexagon {

HexagonDeviceAPI* HexagonDeviceAPI::Global() {
  static auto* inst = new HexagonDeviceAPI();
  return inst;
}

void HexagonDeviceAPI::GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) {
  if (kind == kExist) {
    *rv = 1;
  }
}

// DataSpace: static allocations for Hexagon
void* HexagonDeviceAPI::AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                                       Optional<String> mem_scope) {
  CHECK(shape || ndim == 0) << "shape array is null for a non-scalar tensor, ndim = " << ndim;
  CHECK(IsValidDevice(dev)) << "dev.device_type: " << dev.device_type;

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
  if (!mem_scope.defined() || mem_scope.value() == "global") {
    return DeviceAPI::AllocDataSpace(dev, ndim, shape, dtype, mem_scope);
  }

  // NOTE: This check should be superfluous, but it's probably a good idea to leave it in
  // until the AoT executor's multi-device dispatch code is mature. --cconvey 2022-08-26
  CHECK(dev.device_type == kDLHexagon)
      << "dev.device_type: " << dev.device_type << " DeviceName(" << dev.device_type
      << "): " << DLDeviceType2Str(dev.device_type) << "";

  CHECK(ndim >= 0 && ndim <= 2)
      << "Hexagon Device API supports only 1d and 2d allocations, but received ndim = " << ndim;

  const size_t typesize = (dtype.bits / 8) * dtype.lanes;

  CHECK(runtime_hexbuffs) << "Attempted to allocate Hexagon data with "
                          << "HexagonDeviceAPI::AllocDataSpace before initializing resources.  "
                          << "Please call HexagonDeviceAPI::AcquireResources";
  void* base_ptr;
  PhysicalShape physical_shape;
  if (ndim == 0) {
    // Allocate storage for a single scalar value.
    base_ptr = runtime_hexbuffs->AllocateHexagonBuffer(typesize, kHexagonAllocAlignment, mem_scope);
    physical_shape = {1, 1, typesize};
  } else if (ndim == 1) {
    // Allocate a single, contiguous memory region.
    size_t nbytes = shape[0] * typesize;
    base_ptr = runtime_hexbuffs->AllocateHexagonBuffer(nbytes, kHexagonAllocAlignment, mem_scope);
    physical_shape = {1, 1, nbytes};
  } else if (ndim == 2) {
    // Allocate the region(s) needed for Hexagon's indirect-tensor format.
    size_t nallocs = shape[0];
    size_t nbytes = shape[1] * typesize;
    base_ptr =
        runtime_hexbuffs->AllocateHexagonBuffer(nallocs, nbytes, kHexagonAllocAlignment, mem_scope);
    physical_shape = {2, nallocs, nbytes};
  } else {
    return nullptr;  // unreachable
  }
  SetPhysicalShape(base_ptr, physical_shape);
  return base_ptr;
}

void* HexagonDeviceAPI::AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                                       DLDataType type_hint) {
  CHECK(nbytes) << "number of bytes is zero";
  CHECK(alignment) << "alignment is zero";
  CHECK(IsValidDevice(dev)) << "dev.device_type: " << dev.device_type;
  if (alignment < kHexagonAllocAlignment) {
    alignment = kHexagonAllocAlignment;
  }
  CHECK(runtime_hexbuffs) << "Attempted to allocate Hexagon data with "
                          << "HexagonDeviceAPI::AllocDataSpace before initializing resources.  "
                          << "Please call HexagonDeviceAPI::AcquireResources";
  void* base_ptr = runtime_hexbuffs->AllocateHexagonBuffer(nbytes, alignment, String("global"));
  PhysicalShape physical_shape = {1, 1, nbytes};
  LOG(INFO) << "Setting physical shape to 1D\n";
  SetPhysicalShape(base_ptr, physical_shape);
  return base_ptr;
}

void HexagonDeviceAPI::FreeDataSpace(Device dev, void* ptr) {
  CHECK(ptr) << "buffer pointer is null";
  CHECK(IsValidDevice(dev)) << "dev.device_type: " << dev.device_type;
  if (runtime_hexbuffs) {
    runtime_hexbuffs->FreeHexagonBuffer(ptr);
  } else {
    // Either AcquireResources was never called, or ReleaseResources was called.  Since this can
    // occur in the normal course of shutdown, log a message and continue.
    DLOG(INFO) << "FreeDataSpace called outside a session for " << ptr;
  }
  ndarray_physical_shape.erase(ptr);
}

// WorkSpace: runtime allocations for Hexagon
struct HexagonWorkspacePool : public WorkspacePool {
  HexagonWorkspacePool()
      : WorkspacePool(static_cast<DLDeviceType>(kDLHexagon), HexagonDeviceAPI::Global()) {}
};

void* HexagonDeviceAPI::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  CHECK(IsValidDevice(dev)) << "dev.device_type: " << dev.device_type;
  return dmlc::ThreadLocalStore<HexagonWorkspacePool>::Get()->AllocWorkspace(dev, size);
}

void HexagonDeviceAPI::FreeWorkspace(Device dev, void* data) {
  CHECK(IsValidDevice(dev)) << "dev.device_type: " << dev.device_type;
  CHECK(runtime_hexbuffs) << "Attempted to free Hexagon workspace with "
                          << "HexagonDeviceAPI::FreeWorkspace outside of a session.  "
                          << "Please call HexagonDeviceAPI::AcquireResources";
  CHECK(runtime_hexbuffs->FindHexagonBuffer(data) != nullptr)
      << "Attempt made to free unknown or already freed workspace allocation";
  dmlc::ThreadLocalStore<HexagonWorkspacePool>::Get()->FreeWorkspace(dev, data);
}

void* get_data_start(DLTensor* tensor) { return (reinterpret_cast<uint8_t*>(tensor->data)); }

void HexagonDeviceAPI::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  CHECK_EQ(from->byte_offset, 0);
  CHECK_EQ(to->byte_offset, 0);
  CHECK_EQ(GetDataSize(*from), GetDataSize(*to));
  CHECK(runtime_hexbuffs) << "Attempted to copy Hexagon data with "
                          << "HexagonDeviceAPI::CopyDataFromTo before initializing resources.  "
                          << "Please call HexagonDeviceAPI::AcquireResources";

  auto numBytes = GetDataSize(*from);

  size_t FlatShape = 1;
  for (auto i = 0; i < from->ndim; ++i) FlatShape *= from->shape[i];

  PhysicalShape source_shape = {1, 1, FlatShape};
  PhysicalShape dest_shape = {1, 1, FlatShape};
  auto it1 = ndarray_physical_shape.find(from->data);
  if (it1 != ndarray_physical_shape.end()) source_shape = it1->second;
  size_t src_rank = source_shape.ndim;
  void* src_start = get_data_start(from);
  void* dst_start = get_data_start(to);
  BufferSet src((src_rank == 1) ? &(src_start) : static_cast<void**>(src_start),
                source_shape.nblocks, numBytes / source_shape.nblocks);
  auto it2 = ndarray_physical_shape.find(to->data);
  if (it2 != ndarray_physical_shape.end()) dest_shape = it2->second;
  size_t dest_rank = dest_shape.ndim;
  BufferSet dest((dest_rank == 1) ? &(dst_start) : static_cast<void**>(dst_start),
                 dest_shape.nblocks, numBytes / dest_shape.nblocks);
  HexagonBufferCopyAcrossRegions(dest, src, numBytes, (it1 != ndarray_physical_shape.end()),
                                 (it2 != ndarray_physical_shape.end()));
  return;
}

void HexagonDeviceAPI::SetPhysicalShape(const DLTensor* tensor, const int64_t ndim,
                                        const int64_t* shape) {
  PhysicalShape physical_shape = {static_cast<size_t>(ndim), static_cast<size_t>(shape[0]),
                                  static_cast<size_t>(shape[1])};
  SetPhysicalShape(tensor->data, physical_shape);
}

void HexagonDeviceAPI::SetPhysicalShape(const void* data, const PhysicalShape& physical_shape) {
  auto it = ndarray_physical_shape.find(const_cast<void*>(data));
  if (it != ndarray_physical_shape.end()) {
    ndarray_physical_shape[const_cast<void*>(data)] = physical_shape;
  } else {
    ndarray_physical_shape.insert(
        std::pair<void*, PhysicalShape>(const_cast<void*>(data), physical_shape));
  }
}

void HexagonDeviceAPI::CopyDataFromTo(const void* from, size_t from_offset, void* to,
                                      size_t to_offset, size_t size, Device dev_from, Device dev_to,
                                      DLDataType type_hint, TVMStreamHandle stream) {
  memcpy(static_cast<char*>(to) + to_offset, static_cast<const char*>(from) + from_offset, size);
}

TVM_REGISTER_GLOBAL("device_api.hexagon.dma_copy_dltensor")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      DLTensor* dst = args[0];
      DLTensor* src = args[1];
      int size = args[2];
      ICHECK(size > 0);
      bool bypass_cache = args[3];

      int ret = DMA_RETRY;
      do {
        ret = HexagonDeviceAPI::Global()->UserDMA()->Copy(SYNC_DMA_QUEUE, dst->data, src->data,
                                                          size, bypass_cache);
      } while (ret == DMA_RETRY);
      CHECK(ret == DMA_SUCCESS);
      HexagonDeviceAPI::Global()->UserDMA()->Wait(SYNC_DMA_QUEUE, 0);

      *rv = static_cast<int32_t>(0);
    });

TVM_REGISTER_GLOBAL("device_api.hexagon.dma_copy").set_body([](TVMArgs args, TVMRetValue* rv) {
  uint32_t queue_id = static_cast<int>(args[0]);
  void* dst = args[1];
  void* src = args[2];
  uint32_t size = static_cast<int>(args[3]);
  ICHECK(size > 0);
  bool bypass_cache = args[4];

  int ret = DMA_RETRY;
  do {
    ret = HexagonDeviceAPI::Global()->UserDMA()->Copy(queue_id, dst, src, size, bypass_cache);
  } while (ret == DMA_RETRY);
  CHECK(ret == DMA_SUCCESS);
  *rv = static_cast<int32_t>(ret);
});

TVM_REGISTER_GLOBAL("device_api.hexagon.dma_wait").set_body([](TVMArgs args, TVMRetValue* rv) {
  uint32_t queue_id = static_cast<int>(args[0]);
  int inflight = args[1];
  ICHECK(inflight >= 0);
  HexagonDeviceAPI::Global()->UserDMA()->Wait(queue_id, inflight);
  *rv = static_cast<int32_t>(0);
});

TVM_REGISTER_GLOBAL("device_api.hexagon.dma_start_group")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      uint32_t queue_id = static_cast<int>(args[0]);
      HexagonDeviceAPI::Global()->UserDMA()->StartGroup(queue_id);
      *rv = static_cast<int32_t>(0);
    });

TVM_REGISTER_GLOBAL("device_api.hexagon.dma_end_group").set_body([](TVMArgs args, TVMRetValue* rv) {
  uint32_t queue_id = static_cast<int>(args[0]);
  HexagonDeviceAPI::Global()->UserDMA()->EndGroup(queue_id);
  *rv = static_cast<int32_t>(0);
});

TVM_REGISTER_GLOBAL("device_api.hexagon.alloc_nd").set_body([](TVMArgs args, TVMRetValue* rv) {
  int32_t device_type = args[0];
  int32_t device_id = args[1];
  int32_t dtype_code_hint = args[2];
  int32_t dtype_bits_hint = args[3];
  std::string scope = args[4];
  CHECK(scope.find("global.vtcm") != std::string::npos);
  int64_t ndim = args[5];
  CHECK((ndim == 1 || ndim == 2) && "Hexagon Device API supports only 1d and 2d allocations");
  int64_t* shape = static_cast<int64_t*>(static_cast<void*>(args[6]));

  Device dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;

  DLDataType type_hint;
  type_hint.code = static_cast<decltype(type_hint.code)>(dtype_code_hint);
  type_hint.bits = static_cast<decltype(type_hint.bits)>(dtype_bits_hint);
  type_hint.lanes = 1;

  HexagonDeviceAPI* hexapi = HexagonDeviceAPI::Global();
  *rv = hexapi->AllocDataSpace(dev, ndim, shape, type_hint, String(scope));
});

TVM_REGISTER_GLOBAL("device_api.hexagon.free_nd").set_body([](TVMArgs args, TVMRetValue* rv) {
  int32_t device_type = args[0];
  int32_t device_id = args[1];
  std::string scope = args[2];
  CHECK(scope.find("global.vtcm") != std::string::npos);
  void* ptr = args[3];

  Device dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;

  HexagonDeviceAPI* hexapi = HexagonDeviceAPI::Global();
  hexapi->FreeDataSpace(dev, ptr);
  *rv = static_cast<int32_t>(0);
});

TVM_REGISTER_GLOBAL("device_api.hexagon.acquire_resources")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      HexagonDeviceAPI* api = HexagonDeviceAPI::Global();
      api->AcquireResources();
    });

TVM_REGISTER_GLOBAL("device_api.hexagon.release_resources")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      HexagonDeviceAPI* api = HexagonDeviceAPI::Global();
      api->ReleaseResources();
    });

TVM_REGISTER_GLOBAL("device_api.hexagon.vtcm_device_bytes")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      HexagonDeviceAPI* api = HexagonDeviceAPI::Global();
      *rv = static_cast<int32_t>(api->VtcmPool()->VtcmDeviceBytes());
    });

TVM_REGISTER_GLOBAL("device_api.hexagon").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = HexagonDeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
