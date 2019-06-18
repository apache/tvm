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
 *  Copyright (c) 2019 by Contributors
 * \file micro_device_api.cc
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include "../workspace_pool.h"
#include "micro_session.h"

namespace tvm {
namespace runtime {
/*!
 * \brief device API for uTVM micro devices
 */
class MicroDeviceAPI final : public DeviceAPI {
 public:
  /*! \brief constructor */
  MicroDeviceAPI() { }

  void SetDevice(TVMContext ctx) final {}

  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }

  void* AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       TVMType type_hint) final {
    auto session_ = MicroSession::Global();
    // If there is an allocation for a reference to an invalid session, then
    // something has gone very wrong. All allocations should be contained within
    // the `with` block for the corresponding `MicroSession`.
    CHECK(session_->valid()) << "data space alloc on invalid session";

    void* data = session_->AllocateInSection(SectionKind::kHeap, nbytes).cast_to<void*>();
    DeviceSpace* dev_space = new DeviceSpace();
    dev_space->data = data;
    dev_space->session = session_;
    return static_cast<void*>(dev_space);
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    auto session_ = MicroSession::Global();
    // It is possible (and usually the case) to have dangling references to a
    // session after the session has ended (due to Python scoping). In this
    // case, freeing is a no-op.
    if (!session_->valid()) return;

    DeviceSpace* dev_space = static_cast<DeviceSpace*>(ptr);
    session_->FreeInSection(SectionKind::kHeap,
                            DevBaseOffset(reinterpret_cast<std::uintptr_t>(dev_space->data)));
    delete dev_space;
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMType type_hint,
                      TVMStreamHandle stream) final {
    auto session_ = MicroSession::Global();
    if (!session_->valid()) return;

    std::tuple<int, int> type_from_to(ctx_from.device_type, ctx_to.device_type);
    const std::shared_ptr<LowLevelDevice>& lld = session_->low_level_device();

    if (type_from_to == std::make_tuple(kDLMicroDev, kDLMicroDev)) {
      // Copying from the device to the device.
      CHECK(ctx_from.device_id == ctx_to.device_id)
        << "can only copy between the same micro device";

      DevBaseOffset from_dev_offset = GetDevLoc(from, from_offset);
      DevBaseOffset to_dev_offset = GetDevLoc(to, to_offset);

      std::vector<uint8_t> buffer(size);
      lld->Read(from_dev_offset, static_cast<void*>(buffer.data()), size);
      lld->Write(to_dev_offset, static_cast<void*>(buffer.data()), size);
    } else if (type_from_to == std::make_tuple(kDLMicroDev, kDLCPU)) {
      // Reading from the device.
      DevBaseOffset from_dev_offset = GetDevLoc(from, from_offset);
      void* to_host_ptr = GetHostLoc(to, to_offset);
      lld->Read(from_dev_offset, to_host_ptr, size);
    } else if (type_from_to == std::make_tuple(kDLCPU, kDLMicroDev)) {
      // Writing to the device.
      void* from_host_ptr = GetHostLoc(from, from_offset);
      DevBaseOffset to_dev_offset = GetDevLoc(to, to_offset);
      lld->Write(to_dev_offset, from_host_ptr, size);
    } else {
      LOG(FATAL) << "Expect copy from/to micro_dev or between micro_dev\n";
    }
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) final {
    auto session_ = MicroSession::Global();
    CHECK(session_->valid()) << "workspace alloc on invalid session";

    void* data = session_->AllocateInSection(SectionKind::kWorkspace, size).cast_to<void*>();
    DeviceSpace* dev_space = new DeviceSpace();
    dev_space->data = data;
    dev_space->session = session_;
    return static_cast<void*>(dev_space);
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
    auto session_ = MicroSession::Global();
    if (!session_->valid()) return;

    DeviceSpace* dev_space = static_cast<DeviceSpace*>(data);
    session_->FreeInSection(SectionKind::kWorkspace,
                            DevBaseOffset(reinterpret_cast<std::uintptr_t>(dev_space->data)));
    delete dev_space;
  }

  /*!
   * \brief obtain a global singleton of MicroDeviceAPI
   * \return global shared pointer to MicroDeviceAPI
   */
  static const std::shared_ptr<MicroDeviceAPI>& Global() {
    static std::shared_ptr<MicroDeviceAPI> inst =
        std::make_shared<MicroDeviceAPI>();
    return inst;
  }

 private:
  DevBaseOffset GetDevLoc(const void* ptr, size_t offset) {
    auto session_ = MicroSession::Global();
    DeviceSpace* dev_space = static_cast<DeviceSpace*>(const_cast<void*>(ptr));
    CHECK(dev_space->session == session_) << "session mismatch";
    DevBaseOffset dev_offset =
        DevBaseOffset(reinterpret_cast<std::uintptr_t>(dev_space->data) + offset);
    return dev_offset;
  }

  void* GetHostLoc(const void* ptr, size_t offset) {
    return reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(ptr) + offset);
  }
};

// register device that can be obtained from Python frontend
TVM_REGISTER_GLOBAL("device_api.micro_dev")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = MicroDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
    });
}  // namespace runtime
}  // namespace tvm
