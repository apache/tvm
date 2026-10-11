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

#include <gtest/gtest.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/device_api.h>

#include <unordered_set>

namespace tvm {
namespace runtime {
namespace {

class StreamTestDevice final : public DeviceAPI {
 public:
  std::unordered_set<void*> live;
  void* current{nullptr};
  void SetDevice(Device) final {}
  void GetAttr(Device, DeviceAttrKind, ffi::Any* result) final { *result = 1; }
  void* AllocDataSpace(Device, size_t, size_t, DLDataType) final { return nullptr; }
  void FreeDataSpace(Device, void*) final {}
  void* CreateStream(Device) final {
    auto handle = new int(0);
    live.insert(handle);
    return handle;
  }
  void FreeStream(Device, void* handle) final {
    EXPECT_EQ(live.erase(handle), 1u);
    delete static_cast<int*>(handle);
  }
  void SetStream(Device, void* handle) final {
    EXPECT_TRUE(handle == nullptr || live.count(handle));
    current = handle;
  }
  void* GetCurrentStream(Device) final { return current; }
  void StreamSync(Device, void* handle) final {
    EXPECT_TRUE(handle == nullptr || live.count(handle));
  }

 protected:
  void CopyDataFromTo(const void*, size_t, void*, size_t, size_t, Device, Device, DLDataType,
                      TVMStreamHandle) final {}
};

TEST(RPC, StreamOwnershipAndAliases) {
  auto make_session = ffi::Function::GetGlobal("rpc.LocalSession");
  if (!make_session.has_value()) GTEST_SKIP() << "RPC is disabled";
  if (ffi::Function::GetGlobal("device_api.ext_dev").has_value()) {
    GTEST_SKIP() << "External device is already registered";
  }
  static StreamTestDevice backend;
  ffi::Function::SetGlobal("device_api.ext_dev",
                           ffi::Function::FromTyped([]() { return static_cast<void*>(&backend); }));
  auto get = [](const char* name) { return ffi::Function::GetGlobalRequired(name); };
  ffi::Module session = (*make_session)().cast<ffi::Module>();
  auto device = get("rpc.RemoteDevice")(session, Device{kDLExtDev, 0});
  // The selected stream survives destruction of its temporary wrapper.
  get("rpc.DeviceSetStream")(device, get("rpc.DeviceCreateStream")(device));
  EXPECT_EQ(backend.live.size(), 1u);
  EXPECT_NE(backend.current, nullptr);
  auto alias = get("rpc.DeviceCurrentStream")(device);
  auto another_alias = get("rpc.DeviceCurrentStream")(device);
  get("rpc.DeviceSync")(device, alias);
  get("rpc.DeviceFreeStream")(device, alias);
  EXPECT_TRUE(backend.live.empty());
  EXPECT_EQ(backend.current, nullptr);
  EXPECT_THROW(get("rpc.DeviceSync")(device, another_alias), ffi::Error);
  // Explicit session closure releases remaining handles and invalidates aliases.
  auto stream = get("rpc.DeviceCreateStream")(device);
  EXPECT_EQ(backend.live.size(), 1u);
  session->GetFunction("CloseRPCConnection").value()();
  EXPECT_TRUE(backend.live.empty());
  EXPECT_THROW(get("rpc.DeviceSync")(device, stream), ffi::Error);
}

}  // namespace
}  // namespace runtime
}  // namespace tvm
