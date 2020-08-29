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
 * \file device_api.cc
 * \brief TVM device API for VTA
 */

#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>

#include "../../src/runtime/workspace_pool.h"
#include "runtime.h"

namespace tvm {
namespace runtime {

class VTADeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final {}

  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }

  void* AllocDataSpace(TVMContext ctx, size_t size, size_t alignment, DLDataType type_hint) final {
    return VTABufferAlloc(size);
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final { VTABufferFree(ptr); }

  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      TVMContext ctx_from, TVMContext ctx_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    int kind_mask = 0;
    if (ctx_from.device_type != kDLCPU) {
      kind_mask |= 2;
    }
    if (ctx_to.device_type != kDLCPU) {
      kind_mask |= 1;
    }
    VTABufferCopy(from, from_offset, to, to_offset, size, kind_mask);
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {}

  void* AllocWorkspace(TVMContext ctx, size_t size, DLDataType type_hint) final;

  void FreeWorkspace(TVMContext ctx, void* data) final;

  static VTADeviceAPI* Global() {
    static VTADeviceAPI* inst = new VTADeviceAPI();
    return inst;
  }
};

struct VTAWorkspacePool : public WorkspacePool {
  VTAWorkspacePool() : WorkspacePool(kDLExtDev, VTADeviceAPI::Global()) {}
};

void* VTADeviceAPI::AllocWorkspace(TVMContext ctx, size_t size, DLDataType type_hint) {
  return dmlc::ThreadLocalStore<VTAWorkspacePool>::Get()->AllocWorkspace(ctx, size);
}

void VTADeviceAPI::FreeWorkspace(TVMContext ctx, void* data) {
  dmlc::ThreadLocalStore<VTAWorkspacePool>::Get()->FreeWorkspace(ctx, data);
}

// Register device api with override.
static TVM_ATTRIBUTE_UNUSED auto& __register_dev__ =
    ::tvm::runtime::Registry::Register("device_api.ext_dev", true)
        .set_body([](TVMArgs args, TVMRetValue* rv) {
          DeviceAPI* ptr = VTADeviceAPI::Global();
          *rv = static_cast<void*>(ptr);
        });
}  // namespace runtime
}  // namespace tvm
