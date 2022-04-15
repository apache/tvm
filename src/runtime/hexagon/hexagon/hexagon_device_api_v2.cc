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
 * \file hexagon_device_api_v2.cc
 */

#include "hexagon_device_api_v2.h"

#include <dmlc/thread_local.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstdlib>
#include <cstring>

#include "../../workspace_pool.h"
#include "hexagon_buffer.h"
#include "hexagon_common.h"

namespace tvm {
namespace runtime {
namespace hexagon {

int hexagon_user_dma_1d_sync(void* dst, void* src, uint32_t length);

HexagonDeviceAPIv2* HexagonDeviceAPIv2::Global() {
  static auto* inst = new HexagonDeviceAPIv2();
  return inst;
}

void HexagonDeviceAPIv2::GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) {
  if (kind == kExist) {
    *rv = 1;
  }
}

// DataSpace: static allocations for Hexagon
void* HexagonDeviceAPIv2::AllocDataSpace(Device dev, int ndim, const int64_t* shape,
                                         DLDataType dtype, Optional<String> mem_scope) {
  if (!mem_scope.defined() || mem_scope.value() == "global") {
    return DeviceAPI::AllocDataSpace(dev, ndim, shape, dtype, mem_scope);
  }

  CHECK(TVMDeviceExtType(dev.device_type) == kDLHexagon) << "dev.device_type: " << dev.device_type;

  size_t typesize = (dtype.bits / 8) * dtype.lanes;

  size_t alignment = shape[ndim - 1] * typesize;
  if (alignment < kHexagonAllocAlignment) {
    alignment = kHexagonAllocAlignment;
  }

  if (ndim == 1) {
    size_t nbytes = shape[0] * typesize;
    return AllocateHexagonBuffer(nbytes, alignment, mem_scope);
  } else if (ndim == 2) {
    size_t nallocs = shape[0];
    size_t nbytes = shape[1] * typesize;
    return AllocateHexagonBuffer(nallocs, nbytes, alignment, mem_scope);
  } else {
    LOG(FATAL) << "Hexagon Device API supports only 1d and 2d allocations, but received ndim = "
               << ndim;
    return nullptr;
  }
}

void* HexagonDeviceAPIv2::AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                                         DLDataType type_hint) {
  bool is_valid_device = (TVMDeviceExtType(dev.device_type) == kDLHexagon) ||
                         (DLDeviceType(dev.device_type) == kDLCPU);
  CHECK(is_valid_device) << "dev.device_type: " << dev.device_type;
  if (alignment < kHexagonAllocAlignment) {
    alignment = kHexagonAllocAlignment;
  }
  return AllocateHexagonBuffer(nbytes, alignment, String("global"));
}

void HexagonDeviceAPIv2::FreeDataSpace(Device dev, void* ptr) {
  bool is_valid_device = (TVMDeviceExtType(dev.device_type) == kDLHexagon) ||
                         (DLDeviceType(dev.device_type) == kDLCPU);
  CHECK(is_valid_device) << "dev.device_type: " << dev.device_type;
  FreeHexagonBuffer(ptr);
}

// WorkSpace: runtime allocations for Hexagon
struct HexagonWorkspacePool : public WorkspacePool {
  HexagonWorkspacePool()
      : WorkspacePool(static_cast<DLDeviceType>(kDLHexagon), HexagonDeviceAPIv2::Global()) {}
};

void* HexagonDeviceAPIv2::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  CHECK(TVMDeviceExtType(dev.device_type) == kDLHexagon) << "dev.device_type: " << dev.device_type;
  return dmlc::ThreadLocalStore<HexagonWorkspacePool>::Get()->AllocWorkspace(dev, size);
}

void HexagonDeviceAPIv2::FreeWorkspace(Device dev, void* data) {
  CHECK(TVMDeviceExtType(dev.device_type) == kDLHexagon) << "dev.device_type: " << dev.device_type;
  CHECK(hexagon_buffer_map_.count(data) != 0)
      << "Attempt made to free unknown or already freed workspace allocation";
  dmlc::ThreadLocalStore<HexagonWorkspacePool>::Get()->FreeWorkspace(dev, data);
}

void* HexagonDeviceAPIv2::AllocVtcmWorkspace(Device dev, int ndim, const int64_t* shape,
                                             DLDataType dtype, Optional<String> mem_scope) {
  CHECK(TVMDeviceExtType(dev.device_type) == kDLHexagon) << "dev.device_type: " << dev.device_type;
  CHECK((ndim == 1 || ndim == 2) && "Hexagon Device API supports only 1d and 2d allocations");
  return AllocDataSpace(dev, ndim, shape, dtype, mem_scope);
}

void HexagonDeviceAPIv2::FreeVtcmWorkspace(Device dev, void* ptr) {
  CHECK(TVMDeviceExtType(dev.device_type) == kDLHexagon) << "dev.device_type: " << dev.device_type;
  FreeDataSpace(dev, ptr);
}

void HexagonDeviceAPIv2::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  CHECK_EQ(from->byte_offset, 0);
  CHECK_EQ(to->byte_offset, 0);
  CHECK_EQ(GetDataSize(*from), GetDataSize(*to));

  auto lookup_hexagon_buffer = [this](void* ptr) -> HexagonBuffer* {
    auto it = this->hexagon_buffer_map_.find(ptr);
    CHECK(it != this->hexagon_buffer_map_.end())
        << "Lookup failed for non-HexagonBuffer allocation, CopyDataFromTo can only copy data "
           "from, to or between HexagonBuffers";
    return it->second.get();
  };

  if (TVMDeviceExtType(from->device.device_type) == kDLHexagon &&
      TVMDeviceExtType(to->device.device_type) == kDLHexagon) {
    HexagonBuffer* hex_from_buf = lookup_hexagon_buffer(from->data);
    HexagonBuffer* hex_to_buf = lookup_hexagon_buffer(to->data);
    hex_to_buf->CopyFrom(*hex_from_buf, GetDataSize(*from));
  } else if (from->device.device_type == kDLCPU &&
             TVMDeviceExtType(to->device.device_type) == kDLHexagon) {
    HexagonBuffer* hex_to_buf = lookup_hexagon_buffer(to->data);
    hex_to_buf->CopyFrom(from->data, GetDataSize(*from));
  } else if (TVMDeviceExtType(from->device.device_type) == kDLHexagon &&
             to->device.device_type == kDLCPU) {
    HexagonBuffer* hex_from_buf = lookup_hexagon_buffer(from->data);
    hex_from_buf->CopyTo(to->data, GetDataSize(*to));
  } else {
    CHECK(false)
        << "Expect copy between DLTensor devices of types kDLHexagon and kDLCPU (external) only.";
  }
}

void HexagonDeviceAPIv2::CopyDataFromTo(const void* from, size_t from_offset, void* to,
                                        size_t to_offset, size_t size, Device dev_from,
                                        Device dev_to, DLDataType type_hint,
                                        TVMStreamHandle stream) {
  memcpy(static_cast<char*>(to) + to_offset, static_cast<const char*>(from) + from_offset, size);
}

void HexagonDeviceAPIv2::FreeHexagonBuffer(void* ptr) {
  auto it = hexagon_buffer_map_.find(ptr);
  CHECK(it != hexagon_buffer_map_.end())
      << "Attempt made to free unknown or already freed dataspace allocation";
  CHECK(it->second != nullptr);
  hexagon_buffer_map_.erase(it);
}

TVM_REGISTER_GLOBAL("device_api.hexagon.mem_copy").set_body([](TVMArgs args, TVMRetValue* rv) {
  void* dst = args[0];
  void* src = args[1];
  int size = args[2];

  hexagon_user_dma_1d_sync(dst, src, size);

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

  HexagonDeviceAPIv2* hexapi = HexagonDeviceAPIv2::Global();
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

  HexagonDeviceAPIv2* hexapi = HexagonDeviceAPIv2::Global();
  hexapi->FreeVtcmWorkspace(dev, ptr);
  *rv = static_cast<int32_t>(0);
});

TVM_REGISTER_GLOBAL("device_api.hexagon.v2").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = HexagonDeviceAPIv2::Global();
  *rv = static_cast<void*>(ptr);
});

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
