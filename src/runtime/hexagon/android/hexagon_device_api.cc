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

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <cstring>

#include "hexagon_device.h"

namespace tvm {
namespace runtime {

class HexagonDeviceAPI : public DeviceAPI {
 public:
  void SetDevice(Device dev) final;
  void GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final;
  void FreeDataSpace(Device dev, void* ptr) final;
  void StreamSync(Device dev, TVMStreamHandle stream) final;
  void* AllocWorkspace(Device dev, size_t nbytes, DLDataType type_hint = {}) final;
  void FreeWorkspace(Device dev, void* ptr) final;

  static HexagonDeviceAPI* Global() {
    // NOTE: explicitly use new to avoid destruction of global state
    // Global state will be recycled by OS as the process exits.
    static HexagonDeviceAPI* inst = new HexagonDeviceAPI();
    return inst;
  }

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset,
                      size_t num_bytes, Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final;
};

// HexagonDeviceAPI.

inline void HexagonDeviceAPI::SetDevice(Device dev) {}

inline void HexagonDeviceAPI::GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) {
  if (kind == kExist) *rv = 1;
}

inline void* HexagonDeviceAPI::AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                                              DLDataType type_hint) {
  ICHECK(hexagon::Device::ValidateDeviceId(dev.device_id));
  return hexagon::Device::Global()->Alloc(nbytes, alignment);
}

inline void HexagonDeviceAPI::FreeDataSpace(Device dev, void* ptr) {
  ICHECK(hexagon::Device::ValidateDeviceId(dev.device_id));
  hexagon::Device::Global()->Free(ptr);
}

inline void HexagonDeviceAPI::CopyDataFromTo(const void* from, size_t from_offset, void* to,
                                             size_t to_offset, size_t num_bytes, Device dev_from,
                                             Device dev_to, DLDataType type_hint,
                                             TVMStreamHandle stream) {
  const char* src = static_cast<const char*>(from) + from_offset;
  char* dst = static_cast<char*>(to) + to_offset;

  auto Is32bit = [](const void* p) {
    return p == reinterpret_cast<const void*>(uint32_t(uintptr_t(p)));
  };
  (void)Is32bit;

  if (dev_from.device_type == dev_to.device_type) {
    if (dev_from.device_type == kDLCPU) {
      memmove(dst, src, num_bytes);
    } else if (static_cast<int>(dev_from.device_type) == kDLHexagon) {
      ICHECK(hexagon::Device::ValidateDeviceId(dev_from.device_id));
      ICHECK_EQ(dev_from.device_id, dev_to.device_id);
      ICHECK(Is32bit(dst) && Is32bit(src));
      hexagon::Device::Global()->CopyDeviceToDevice(dst, src, num_bytes);
    }
  } else {
    if (dev_from.device_type == kDLCPU) {
      ICHECK_EQ(static_cast<int>(dev_to.device_type), kDLHexagon);
      ICHECK(Is32bit(dst));
      ICHECK(hexagon::Device::ValidateDeviceId(dev_to.device_id));
      hexagon::Device::Global()->CopyHostToDevice(dst, src, num_bytes);
    } else {
      ICHECK_EQ(static_cast<int>(dev_from.device_type), kDLHexagon);
      ICHECK_EQ(dev_to.device_type, kDLCPU);
      ICHECK(Is32bit(src));
      ICHECK(hexagon::Device::ValidateDeviceId(dev_from.device_id));
      hexagon::Device::Global()->CopyDeviceToHost(dst, src, num_bytes);
    }
  }
}

inline void HexagonDeviceAPI::StreamSync(Device dev, TVMStreamHandle stream) {}

inline void* HexagonDeviceAPI::AllocWorkspace(Device dev, size_t nbytes, DLDataType type_hint) {
  ICHECK(hexagon::Device::ValidateDeviceId(dev.device_id));
  if (type_hint.code == 100) {
    size_t align = std::min(nbytes, 2048lu);
    return hexagon::Device::Global()->AllocVtcm(nbytes, align);
  }
  return DeviceAPI::AllocWorkspace(dev, nbytes, type_hint);
}

inline void HexagonDeviceAPI::FreeWorkspace(Device dev, void* ptr) {
  ICHECK(hexagon::Device::ValidateDeviceId(dev.device_id));
  DeviceAPI::FreeWorkspace(dev, ptr);
}

TVM_REGISTER_GLOBAL("device_api.hexagon").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = HexagonDeviceAPI::Global();
  *rv = ptr;
});
}  // namespace runtime
}  // namespace tvm

// Hexagon-specific runtime functions to allocate/deallocate workspaces
// in VTCM.
extern "C" {
void* HexagonBackendAllocateVTCM(uint32_t nbytes, uint32_t align) {
  align = std::max(align, 2048u);
  return tvm::runtime::hexagon::Device::Global()->AllocVtcm(nbytes, align);
}
void HexagonBackendFreeVTCM(void* ptr) {
  return tvm::runtime::hexagon::Device::Global()->FreeVtcm(ptr);
}
}
