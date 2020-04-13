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

#include <dmlc/logging.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <cstring>

#include "hexagon_module.h"

namespace tvm {
namespace runtime {

class HexagonDeviceAPI : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final;
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(TVMContext ctx, size_t nbytes, size_t alignment,
                       DLDataType type_hint) final;
  void FreeDataSpace(TVMContext ctx, void* ptr) final;
  void CopyDataFromTo(const void* from, size_t from_offset, void* to,
                      size_t to_offset, size_t num_bytes, TVMContext ctx_from,
                      TVMContext ctx_to, DLDataType type_hint,
                      TVMStreamHandle stream) final;
  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final;
  void* AllocWorkspace(TVMContext ctx, size_t nbytes,
                       DLDataType type_hint = {}) final;
  void FreeWorkspace(TVMContext ctx, void* ptr) final;

  static const std::shared_ptr<HexagonDeviceAPI>& Global() {
    static std::shared_ptr<HexagonDeviceAPI> inst =
        std::make_shared<HexagonDeviceAPI>();
    return inst;
  }
};

// HexagonDeviceAPI.

inline void HexagonDeviceAPI::SetDevice(TVMContext ctx) {}

inline void HexagonDeviceAPI::GetAttr(TVMContext ctx, DeviceAttrKind kind,
                                      TVMRetValue* rv) {
  if (kind == kExist) *rv = 1;
}

inline void* HexagonDeviceAPI::AllocDataSpace(TVMContext ctx, size_t nbytes,
                                              size_t alignment,
                                              DLDataType type_hint) {
  CHECK(hexagon::Device::ValidateDeviceId(ctx.device_id));
  return hexagon::Device::Global()->Alloc(nbytes, alignment);
}

inline void HexagonDeviceAPI::FreeDataSpace(TVMContext ctx, void* ptr) {
  CHECK(hexagon::Device::ValidateDeviceId(ctx.device_id));
  hexagon::Device::Global()->Free(ptr);
}

inline void HexagonDeviceAPI::CopyDataFromTo(
    const void* from, size_t from_offset, void* to, size_t to_offset,
    size_t num_bytes, TVMContext ctx_from, TVMContext ctx_to,
    DLDataType type_hint, TVMStreamHandle stream) {
  const char* src = static_cast<const char*>(from) + from_offset;
  char* dst = static_cast<char*>(to) + to_offset;

  auto Is32bit = [](const void* p) {
    return p == reinterpret_cast<const void*>(uint32_t(uintptr_t(p)));
  };
  (void)Is32bit;

  if (ctx_from.device_type == ctx_to.device_type) {
    if (ctx_from.device_type == kDLCPU) {
      memmove(dst, src, num_bytes);
    } else if (static_cast<int>(ctx_from.device_type) == kDLHexagon) {
      CHECK(hexagon::Device::ValidateDeviceId(ctx_from.device_id));
      CHECK_EQ(ctx_from.device_id, ctx_to.device_id);
      CHECK(Is32bit(dst) && Is32bit(src));
      hexagon::Device::Global()->CopyDeviceToDevice(dst, src, num_bytes);
    }
  } else {
    if (ctx_from.device_type == kDLCPU) {
      CHECK_EQ(static_cast<int>(ctx_to.device_type), kDLHexagon);
      CHECK(Is32bit(dst));
      CHECK(hexagon::Device::ValidateDeviceId(ctx_to.device_id));
      hexagon::Device::Global()->CopyHostToDevice(dst, src, num_bytes);
    } else {
      CHECK_EQ(static_cast<int>(ctx_from.device_type), kDLHexagon);
      CHECK_EQ(ctx_to.device_type, kDLCPU);
      CHECK(Is32bit(src));
      CHECK(hexagon::Device::ValidateDeviceId(ctx_from.device_id));
      hexagon::Device::Global()->CopyDeviceToHost(dst, src, num_bytes);
    }
  }
}

inline void HexagonDeviceAPI::StreamSync(TVMContext ctx,
                                         TVMStreamHandle stream) {}

inline void* HexagonDeviceAPI::AllocWorkspace(TVMContext ctx, size_t nbytes,
                                              DLDataType type_hint) {
  CHECK(hexagon::Device::ValidateDeviceId(ctx.device_id));
  if (type_hint.code == 100) {
    size_t align = std::min(nbytes, 2048lu);
    return hexagon::Device::Global()->AllocVtcm(nbytes, align);
  }
  return DeviceAPI::AllocWorkspace(ctx, nbytes, type_hint);
}

inline void HexagonDeviceAPI::FreeWorkspace(TVMContext ctx, void* ptr) {
  CHECK(hexagon::Device::ValidateDeviceId(ctx.device_id));
  DeviceAPI::FreeWorkspace(ctx, ptr);
}

TVM_REGISTER_GLOBAL("device_api.hexagon")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      DeviceAPI* ptr = HexagonDeviceAPI::Global().get();
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
