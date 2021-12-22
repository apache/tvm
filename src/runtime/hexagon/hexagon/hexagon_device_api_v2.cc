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
#define TVM_LOG_CUSTOMIZE 1

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
  CHECK(TVMDeviceExtType(dev.device_type) == kDLHexagon);

  // Forcing contiguous allocation, for now
  // TODO(Straw): Enable discontiguous allocation after RFC 39 lands
  size_t nallocs = 1;
  size_t nbytes = 1;
  for (int i = 0; i < ndim; ++i) {
    nbytes *= shape[i];
  }
  size_t typesize = (dtype.bits / 8) * dtype.lanes;
  nbytes *= typesize;

  size_t alignment = typesize;
  if (alignment < kHexagonAllocAlignment) {
    alignment = kHexagonAllocAlignment;
  }
  return new HexagonBuffer(nallocs, nbytes, alignment, mem_scope);
}

void* HexagonDeviceAPIv2::AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                                         DLDataType type_hint) {
  if (alignment < kHexagonAllocAlignment) {
    alignment = kHexagonAllocAlignment;
  }
  return new HexagonBuffer(nbytes, alignment, String("global"));
}

void HexagonDeviceAPIv2::FreeDataSpace(Device dev, void* ptr) {
  CHECK(TVMDeviceExtType(dev.device_type) == kDLHexagon);
  auto* hexbuf = static_cast<HexagonBuffer*>(ptr);
  CHECK(hexbuf != nullptr);
  delete hexbuf;
}

// WorkSpace: runtime allocations for Hexagon
struct HexagonWorkspacePool : public WorkspacePool {
  HexagonWorkspacePool() : WorkspacePool(kDLCPU, HexagonDeviceAPIv2::Global()) {}
};

void* HexagonDeviceAPIv2::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  CHECK(TVMDeviceExtType(dev.device_type) == kDLHexagon);
  auto* hexbuf = static_cast<HexagonBuffer*>(
      dmlc::ThreadLocalStore<HexagonWorkspacePool>::Get()->AllocWorkspace(dev, size));

  // Assumes a single contiguous allocation
  // TODO(Straw): Enable discontiguous allocation after RFC 39 lands
  void* ptr = hexbuf->GetPointer()[0];
  workspace_allocations_.insert({ptr, hexbuf});
  return ptr;
}

void HexagonDeviceAPIv2::FreeWorkspace(Device dev, void* data) {
  CHECK(TVMDeviceExtType(dev.device_type) == kDLHexagon);
  auto it = workspace_allocations_.find(data);
  CHECK(it != workspace_allocations_.end())
      << "Attempt made to free unknown or already freed workspace allocation";
  dmlc::ThreadLocalStore<HexagonWorkspacePool>::Get()->FreeWorkspace(dev, it->second);
  workspace_allocations_.erase(it);
}

void HexagonDeviceAPIv2::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  CHECK_EQ(from->byte_offset, 0);
  CHECK_EQ(to->byte_offset, 0);
  CHECK_EQ(GetDataSize(*from), GetDataSize(*to));

  HexagonBuffer* hex_from_buf = static_cast<HexagonBuffer*>(from->data);
  HexagonBuffer* hex_to_buf = static_cast<HexagonBuffer*>(to->data);

  if (TVMDeviceExtType(from->device.device_type) == kDLHexagon &&
      TVMDeviceExtType(to->device.device_type) == kDLHexagon) {
    CHECK(hex_from_buf != nullptr);
    CHECK(hex_to_buf != nullptr);
    hex_to_buf->CopyFrom(*hex_from_buf, GetDataSize(*from));
  } else if (from->device.device_type == kDLCPU &&
             TVMDeviceExtType(to->device.device_type) == kDLHexagon) {
    CHECK(hex_to_buf != nullptr);
    hex_to_buf->CopyFrom(from->data, GetDataSize(*from));
  } else if (TVMDeviceExtType(from->device.device_type) == kDLHexagon &&
             to->device.device_type == kDLCPU) {
    CHECK(hex_from_buf != nullptr);
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

TVM_REGISTER_GLOBAL("device_api.hexagon.v2").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = HexagonDeviceAPIv2::Global();
  *rv = static_cast<void*>(ptr);
});

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
