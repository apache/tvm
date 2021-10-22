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

HexagonDeviceAPIv2* HexagonDeviceAPIv2::Global() {
  static auto* inst = new HexagonDeviceAPIv2();
  return inst;
}

void HexagonDeviceAPIv2::GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) {
  if (kind == kExist) {
    *rv = 1;
  }
}

void* HexagonDeviceAPIv2::AllocDataSpace(Device dev, int ndim, const int64_t* shape,
                                         DLDataType dtype, Optional<String> mem_scope) {
  return new HexagonBuffer(ndim, shape, dtype, mem_scope.defined() ? mem_scope : String("global"));
}

void* HexagonDeviceAPIv2::AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                                         DLDataType type_hint) {
  return new HexagonBuffer(nbytes, alignment, String("global"));
}

void HexagonDeviceAPIv2::FreeDataSpace(Device dev, void* ptr) {
  auto* pbuf = static_cast<HexagonBuffer*>(ptr);
  delete pbuf;
}

struct HexagonWorkspacePool : public WorkspacePool {
  HexagonWorkspacePool() : WorkspacePool(kDLCPU, HexagonDeviceAPIv2::Global()) {}
};

void* HexagonDeviceAPIv2::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  auto* buffer = static_cast<HexagonBuffer*>(
      dmlc::ThreadLocalStore<HexagonWorkspacePool>::Get()->AllocWorkspace(dev, size));
  void* ptr = buffer->GetPointer();
  workspace_allocations_.insert({ptr, buffer});
  return ptr;
}

void HexagonDeviceAPIv2::FreeWorkspace(Device dev, void* data) {
  auto it = workspace_allocations_.find(data);
  ICHECK(it != workspace_allocations_.end())
      << "Attempt made to free unknown or already freed workspace allocation";
  dmlc::ThreadLocalStore<HexagonWorkspacePool>::Get()->FreeWorkspace(dev, it->second);
  workspace_allocations_.erase(it);
}

void HexagonDeviceAPIv2::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  if (IsHexagonDevice(from->device) && IsHexagonDevice(to->device)) {
    HexagonBuffer* buffer_src = static_cast<HexagonBuffer*>(from->data);
    HexagonBuffer* buffer_dst = static_cast<HexagonBuffer*>(to->data);
    // Check storage scopes
    if (buffer_src->GetStorageScope() == HexagonBuffer::StorageScope::kDDR &&
        buffer_dst->GetStorageScope() == HexagonBuffer::StorageScope::kDDR) {
      memcpy(static_cast<char*>(buffer_dst->GetPointer()) + to->byte_offset,
             static_cast<const char*>(buffer_src->GetPointer()) + from->byte_offset,
             GetDataSize(*from));
    } else {
      ICHECK(false) << "Currently only copying between DDR storage is supported.";
    }
  } else if (IsHexagonDevice(from->device) && to->device.device_type == kDLCPU) {
    HexagonBuffer* buffer_src = static_cast<HexagonBuffer*>(from->data);
    memcpy(static_cast<char*>(to->data) + to->byte_offset,
           static_cast<const char*>(buffer_src->GetPointer()) + from->byte_offset,
           GetDataSize(*from));
  } else if (from->device.device_type == kDLCPU && IsHexagonDevice(to->device)) {
    HexagonBuffer* buffer_dst = static_cast<HexagonBuffer*>(to->data);
    memcpy(static_cast<char*>(buffer_dst->GetPointer()) + to->byte_offset,
           static_cast<const char*>(from->data) + from->byte_offset, GetDataSize(*from));
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
