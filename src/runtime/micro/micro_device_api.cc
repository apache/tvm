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
    std::shared_ptr<MicroSession>& session = MicroSession::Current();
    void* data = session->AllocateInSection(SectionKind::kHeap, nbytes).cast_to<void*>();
    CHECK(data != nullptr) << "unable to allocate " << nbytes << " bytes on device heap";
    MicroDevSpace* dev_space = new MicroDevSpace();
    dev_space->data = data;
    dev_space->session = session;
    return static_cast<void*>(dev_space);
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    MicroDevSpace* dev_space = static_cast<MicroDevSpace*>(ptr);
    dev_space->session->FreeInSection(
      SectionKind::kHeap, DevBaseOffset(reinterpret_cast<std::uintptr_t>(dev_space->data)));
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
    std::tuple<int, int> type_from_to(ctx_from.device_type, ctx_to.device_type);
    if (type_from_to == std::make_tuple(kDLMicroDev, kDLMicroDev)) {
      // Copying from the device to the device.

      MicroDevSpace* from_space = static_cast<MicroDevSpace*>(const_cast<void*>(from));
      MicroDevSpace* to_space = static_cast<MicroDevSpace*>(const_cast<void*>(to));
      CHECK(from_space->session == to_space->session)
          << "attempt to copy data between different micro sessions (" << from_space->session
          << " != " << to_space->session << ")";
      CHECK(ctx_from.device_id == ctx_to.device_id)
        << "can only copy between the same micro device";
      std::shared_ptr<MicroSession>& session = from_space->session;
      const std::shared_ptr<LowLevelDevice>& lld = session->low_level_device();

      DevBaseOffset from_dev_offset = GetDevLoc(from_space, from_offset);
      DevBaseOffset to_dev_offset = GetDevLoc(to_space, to_offset);

      std::vector<uint8_t> buffer(size);
      lld->Read(from_dev_offset, static_cast<void*>(buffer.data()), size);
      lld->Write(to_dev_offset, static_cast<void*>(buffer.data()), size);
    } else if (type_from_to == std::make_tuple(kDLMicroDev, kDLCPU)) {
      // Reading from the device.

      MicroDevSpace* from_space = static_cast<MicroDevSpace*>(const_cast<void*>(from));
      std::shared_ptr<MicroSession>& session = from_space->session;
      const std::shared_ptr<LowLevelDevice>& lld = session->low_level_device();

      DevBaseOffset from_dev_offset = GetDevLoc(from_space, from_offset);
      void* to_host_ptr = GetHostLoc(to, to_offset);
      lld->Read(from_dev_offset, to_host_ptr, size);
    } else if (type_from_to == std::make_tuple(kDLCPU, kDLMicroDev)) {
      // Writing to the device.

      MicroDevSpace* to_space = static_cast<MicroDevSpace*>(const_cast<void*>(to));
      std::shared_ptr<MicroSession>& session = to_space->session;
      const std::shared_ptr<LowLevelDevice>& lld = session->low_level_device();

      void* from_host_ptr = GetHostLoc(from, from_offset);
      DevBaseOffset to_dev_offset = GetDevLoc(to_space, to_offset);
      lld->Write(to_dev_offset, from_host_ptr, size);
    } else {
      LOG(FATAL) << "Expect copy from/to micro device or between micro device\n";
    }
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) final {
    std::shared_ptr<MicroSession>& session = MicroSession::Current();

    void* data = session->AllocateInSection(SectionKind::kWorkspace, size).cast_to<void*>();
    CHECK(data != nullptr) << "unable to allocate " << size << " bytes on device workspace";
    MicroDevSpace* dev_space = new MicroDevSpace();
    dev_space->data = data;
    dev_space->session = session;
    return static_cast<void*>(dev_space);
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
    MicroDevSpace* dev_space = static_cast<MicroDevSpace*>(data);
    std::shared_ptr<MicroSession>& session = dev_space->session;
    session->FreeInSection(SectionKind::kWorkspace,
                           DevBaseOffset(reinterpret_cast<std::uintptr_t>(dev_space->data)));
    delete dev_space;
  }

  /*!
   * \brief obtain a global singleton of MicroDeviceAPI
   * \return global shared pointer to MicroDeviceAPI
   */
  static const std::shared_ptr<MicroDeviceAPI>& Global() {
    static std::shared_ptr<MicroDeviceAPI> inst = std::make_shared<MicroDeviceAPI>();
    return inst;
  }

 private:
  DevBaseOffset GetDevLoc(MicroDevSpace* dev_space, size_t offset) {
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
