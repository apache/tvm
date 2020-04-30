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
                       DLDataType type_hint) final {
    ObjectPtr<MicroSession>& session = MicroSession::Current();
    TargetPtr data = session->AllocateInSection(SectionKind::kHeap, nbytes);
    CHECK(data != nullptr) << "unable to allocate " << nbytes << " bytes on device heap";
    return reinterpret_cast<void*>(new MicroDevSpace{data, session});
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    MicroDevSpace* dev_space = static_cast<MicroDevSpace*>(ptr);
    dev_space->session->FreeInSection(SectionKind::kHeap, dev_space->data);
    delete dev_space;
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      DLDataType type_hint,
                      TVMStreamHandle stream) final {
    std::tuple<int, int> type_from_to(ctx_from.device_type, ctx_to.device_type);
    if (type_from_to == std::make_tuple(kDLMicroDev, kDLMicroDev)) {
      // Copying from the device to the device.
      MicroDevSpace* from_space = static_cast<MicroDevSpace*>(const_cast<void*>(from));
      MicroDevSpace* to_space = static_cast<MicroDevSpace*>(const_cast<void*>(to));
      CHECK(from_space->session == to_space->session)
          << "attempt to copy data between different micro sessions ("
          << from_space->session.get()
          << " != " << to_space->session.get() << ")";
      CHECK(ctx_from.device_id == ctx_to.device_id)
        << "can only copy between the same micro device";
      ObjectPtr<MicroSession>& session = from_space->session;
      // flush all pending tasks to ensure data is consistent
      session->FlushTaskQueue();
      const std::shared_ptr<LowLevelDevice>& lld = session->low_level_device();

      TargetPtr from_dev_addr = GetDevLoc(from_space, from_offset);
      TargetPtr to_dev_addr = GetDevLoc(to_space, to_offset);

      std::vector<uint8_t> buffer(size);
      lld->Read(from_dev_addr, static_cast<void*>(buffer.data()), size);
      lld->Write(to_dev_addr, static_cast<void*>(buffer.data()), size);

    } else if (type_from_to == std::make_tuple(kDLMicroDev, kDLCPU)) {
      // Reading from the device.
      MicroDevSpace* from_space = static_cast<MicroDevSpace*>(const_cast<void*>(from));
      ObjectPtr<MicroSession>& session = from_space->session;
      // flush all pending tasks to ensure data is consistent
      session->FlushTaskQueue();
      const std::shared_ptr<LowLevelDevice>& lld = session->low_level_device();

      TargetPtr from_dev_addr = GetDevLoc(from_space, from_offset);
      void* to_host_ptr = GetHostLoc(to, to_offset);
      lld->Read(from_dev_addr, to_host_ptr, size);

    } else if (type_from_to == std::make_tuple(kDLCPU, kDLMicroDev)) {
      // Writing to the device.
      MicroDevSpace* to_space = static_cast<MicroDevSpace*>(const_cast<void*>(to));
      ObjectPtr<MicroSession>& session = to_space->session;
      // flush all pending tasks to ensure data is consistent
      session->FlushTaskQueue();
      const std::shared_ptr<LowLevelDevice>& lld = session->low_level_device();

      void* from_host_ptr = GetHostLoc(from, from_offset);
      TargetPtr to_dev_addr = GetDevLoc(to_space, to_offset);
      lld->Write(to_dev_addr, from_host_ptr, size);

    } else {
      LOG(FATAL) << "Expect copy from/to micro device or between micro device\n";
    }
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
    MicroSession::Current()->FlushTaskQueue();
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, DLDataType type_hint) final {
    CHECK(false) << "the on-device workspace allocator isn't aware of this function";
    ObjectPtr<MicroSession>& session = MicroSession::Current();

    TargetPtr data = session->AllocateInSection(SectionKind::kWorkspace, size);
    CHECK(data.value().uint64() != 0)
      << "unable to allocate " << size << " bytes on device workspace";
    return static_cast<void*>(new MicroDevSpace{data, session});
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
    CHECK(false) << "the on-device workspace allocator isn't aware of this function";
    MicroDevSpace* dev_space = static_cast<MicroDevSpace*>(data);
    ObjectPtr<MicroSession>& session = dev_space->session;
    session->FreeInSection(SectionKind::kWorkspace, dev_space->data);
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
  TargetPtr GetDevLoc(MicroDevSpace* dev_space, size_t offset) {
    return dev_space->data + offset;
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
