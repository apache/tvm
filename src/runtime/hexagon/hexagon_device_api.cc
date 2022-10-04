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
#include "hexagon_common.h"

namespace tvm {
namespace runtime {
namespace hexagon {

int hexagon_user_dma_1d_sync(void* dst, void* src, uint32_t length);

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
  CHECK(shape) << "shape array is null";
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
  CHECK(TVMDeviceExtType(dev.device_type) == kDLHexagon)
      << "dev.device_type: " << dev.device_type << " DeviceName(" << dev.device_type
      << "): " << DeviceName(dev.device_type) << "";

  CHECK(ndim >= 0 && ndim <= 2)
      << "Hexagon Device API supports only 1d and 2d allocations, but received ndim = " << ndim;

  const size_t typesize = (dtype.bits / 8) * dtype.lanes;

  if (ndim == 0) {
    // Allocate storage for a single scalar value.
    return mgr->AllocateHexagonBuffer(typesize, kHexagonAllocAlignment, mem_scope);
  } else if (ndim == 1) {
    // Allocate a single, contiguous memory region.
    size_t nbytes = shape[0] * typesize;
    return mgr->AllocateHexagonBuffer(nbytes, kHexagonAllocAlignment, mem_scope);
  } else if (ndim == 2) {
    // Allocate the region(s) needed for Hexagon's indirect-tensor format.
    size_t nallocs = shape[0];
    size_t nbytes = shape[1] * typesize;
    return mgr->AllocateHexagonBuffer(nallocs, nbytes, kHexagonAllocAlignment, mem_scope);
  } else {
    return nullptr;  // unreachable
  }
}

void* HexagonDeviceAPI::AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                                       DLDataType type_hint) {
  CHECK(nbytes) << "number of bytes is zero";
  CHECK(alignment) << "alignment is zero";
  CHECK(IsValidDevice(dev)) << "dev.device_type: " << dev.device_type;
  if (alignment < kHexagonAllocAlignment) {
    alignment = kHexagonAllocAlignment;
  }
  return mgr->AllocateHexagonBuffer(nbytes, alignment, String("global"));
}

void HexagonDeviceAPI::FreeDataSpace(Device dev, void* ptr) {
  CHECK(ptr) << "buffer pointer is null";
  CHECK(IsValidDevice(dev)) << "dev.device_type: " << dev.device_type;
  mgr->FreeHexagonBuffer(ptr);
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
  CHECK(mgr->count(data) != 0)
      << "Attempt made to free unknown or already freed workspace allocation";
  dmlc::ThreadLocalStore<HexagonWorkspacePool>::Get()->FreeWorkspace(dev, data);
}

void* HexagonDeviceAPI::AllocVtcmWorkspace(Device dev, int ndim, const int64_t* shape,
                                           DLDataType dtype, Optional<String> mem_scope) {
  // must be Hexagon device (not CPU)
  CHECK(TVMDeviceExtType(dev.device_type) == kDLHexagon) << "dev.device_type: " << dev.device_type;
  CHECK((ndim == 1 || ndim == 2) && "Hexagon Device API supports only 1d and 2d allocations");
  return AllocDataSpace(dev, ndim, shape, dtype, mem_scope);
}

void HexagonDeviceAPI::FreeVtcmWorkspace(Device dev, void* ptr) {
  // must be Hexagon device (not CPU)
  CHECK(TVMDeviceExtType(dev.device_type) == kDLHexagon) << "dev.device_type: " << dev.device_type;
  FreeDataSpace(dev, ptr);
}

void HexagonDeviceAPI::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  CHECK_EQ(from->byte_offset, 0);
  CHECK_EQ(to->byte_offset, 0);
  CHECK_EQ(GetDataSize(*from), GetDataSize(*to));

  auto lookup_hexagon_buffer = [this](void* ptr) -> HexagonBuffer* { return mgr->find(ptr); };

  HexagonBuffer* hex_from_buf = lookup_hexagon_buffer(from->data);
  HexagonBuffer* hex_to_buf = lookup_hexagon_buffer(to->data);

  if (hex_from_buf && hex_to_buf) {
    hex_to_buf->CopyFrom(*hex_from_buf, GetDataSize(*from));
  } else if (hex_to_buf) {
    hex_to_buf->CopyFrom(from->data, GetDataSize(*from));
  } else if (hex_from_buf) {
    hex_from_buf->CopyTo(to->data, GetDataSize(*to));
  } else {
    CHECK(false) << "CopyDataFromTo requested between src and dst which are not managed by the "
                    "hexagon device api.";
  }
}

void HexagonDeviceAPI::CopyDataFromTo(const void* from, size_t from_offset, void* to,
                                      size_t to_offset, size_t size, Device dev_from, Device dev_to,
                                      DLDataType type_hint, TVMStreamHandle stream) {
  memcpy(static_cast<char*>(to) + to_offset, static_cast<const char*>(from) + from_offset, size);
}

TVM_REGISTER_GLOBAL("device_api.hexagon.mem_copy_DLTensor")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      DLTensor* dst = args[0];
      DLTensor* src = args[1];
      int size = args[2];

      hexagon_user_dma_1d_sync(dst->data, src->data, size);

      *rv = static_cast<int32_t>(0);
    });

TVM_REGISTER_GLOBAL("device_api.hexagon.mem_copy").set_body([](TVMArgs args, TVMRetValue* rv) {
  void* dst = args[0];
  void* src = args[1];
  int size = args[2];

  int error_code = hexagon_user_dma_1d_sync(dst, src, size);
  CHECK_EQ(error_code, 0);

  *rv = static_cast<int32_t>(0);
});

TVM_REGISTER_GLOBAL("device_api.hexagon.dma_copy").set_body([](TVMArgs args, TVMRetValue* rv) {
  int queue_id = args[0];
  void* dst = args[1];
  void* src = args[2];
  int size = args[3];
  ICHECK(size > 0);

  int ret = DMA_RETRY;
  do {
    ret = HexagonDeviceAPI::Global()->UserDMA()->Copy(queue_id, dst, src, size);
  } while (ret == DMA_RETRY);
  *rv = static_cast<int32_t>(ret);
});

TVM_REGISTER_GLOBAL("device_api.hexagon.dma_wait").set_body([](TVMArgs args, TVMRetValue* rv) {
  int queue_id = args[0];
  int inflight = args[1];
  ICHECK(inflight >= 0);
  HexagonDeviceAPI::Global()->UserDMA()->Wait(queue_id, inflight);
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
  *rv = hexapi->AllocVtcmWorkspace(dev, ndim, shape, type_hint, String(scope));
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
  hexapi->FreeVtcmWorkspace(dev, ptr);
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

TVM_REGISTER_GLOBAL("device_api.hexagon").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = HexagonDeviceAPI::Global();
  *rv = static_cast<void*>(ptr);
});

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
