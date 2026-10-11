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
#include <tvm/ffi/function.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/device_api.h>

#include <limits>
#include <string>
#include <thread>

namespace tvm {
namespace runtime {
namespace {

class SelectionDeviceAPI : public DeviceAPI {
 public:
  void SetDevice(Device dev) final {
    TVM_FFI_CHECK_EQ(dev.device_type, kDLExtDev, ValueError);
    TVM_FFI_CHECK_LT(dev.device_id, 4, ValueError) << "Invalid selection";
    selected_index = dev.device_id;
  }
  int GetDeviceIndex() final {
    if (selected_index == 3) {
      TVM_FFI_THROW(RuntimeError) << "Selection query failed";
    }
    return selected_index;
  }
  void GetAttr(Device, DeviceAttrKind, ffi::Any*) final {}
  void* AllocDataSpace(Device, size_t, size_t, DLDataType) final { return nullptr; }
  void FreeDataSpace(Device, void*) final {}
  void StreamSync(Device, TVMStreamHandle) final {}

  static thread_local int selected_index;
};
thread_local int SelectionDeviceAPI::selected_index = 0;

class BackendDeviceSelection : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    static SelectionDeviceAPI api;
    ffi::Function::SetGlobal("device_api.ext_dev",
                             ffi::Function::FromTyped([]() -> void* { return &api; }));
  }

  void ExpectError(int status, const char* kind) {
    ASSERT_EQ(status, -1);
    auto error = ffi::details::MoveFromSafeCallRaised();
    EXPECT_EQ(std::string(error->kind.data, error->kind.size), kind);
  }
};

TEST_F(BackendDeviceSelection, ThreadLocalDispatch) {
  ASSERT_EQ(TVMBackendSetDevice(kDLExtDev, 1), 0);
  int selected = -1;
  ASSERT_EQ(TVMBackendGetDeviceIndex(kDLExtDev, &selected), 0);
  EXPECT_EQ(selected, 1);
  std::thread other([] {
    int selected = -1;
    ASSERT_EQ(TVMBackendSetDevice(kDLExtDev, 2), 0);
    ASSERT_EQ(TVMBackendGetDeviceIndex(kDLExtDev, &selected), 0);
    EXPECT_EQ(selected, 2);
  });
  other.join();
  ASSERT_EQ(TVMBackendGetDeviceIndex(kDLExtDev, &selected), 0);
  EXPECT_EQ(selected, 1);
}

TEST_F(BackendDeviceSelection, DefaultGetterIsUnsupported) {
  ASSERT_EQ(TVMBackendSetDevice(kDLCPU, 0), 0);
  int selected = 123;
  ExpectError(TVMBackendGetDeviceIndex(kDLCPU, &selected), "NotImplementedError");
  EXPECT_EQ(selected, 123);
}

TEST_F(BackendDeviceSelection, InvalidArguments) {
  for (int type : {-1, 0, 36, 127, 128, 130, std::numeric_limits<int>::max()}) {
    int selected = 123;
    ExpectError(TVMBackendSetDevice(type, 0), "ValueError");
    ExpectError(TVMBackendGetDeviceIndex(type, &selected), "ValueError");
    EXPECT_EQ(selected, 123);
  }
  ExpectError(TVMBackendSetDevice(kDLExtDev, -1), "ValueError");
  ExpectError(TVMBackendGetDeviceIndex(kDLExtDev, nullptr), "ValueError");
}

TEST_F(BackendDeviceSelection, BackendFailures) {
  ASSERT_EQ(TVMBackendSetDevice(kDLExtDev, 1), 0);
  ExpectError(TVMBackendSetDevice(kDLExtDev, 4), "ValueError");
  int selected = -1;
  ASSERT_EQ(TVMBackendGetDeviceIndex(kDLExtDev, &selected), 0);
  EXPECT_EQ(selected, 1);
  ASSERT_EQ(TVMBackendSetDevice(kDLExtDev, 3), 0);
  ExpectError(TVMBackendGetDeviceIndex(kDLExtDev, &selected), "RuntimeError");
  EXPECT_EQ(selected, 1);
}

TEST_F(BackendDeviceSelection, UnavailableBackend) {
  if (ffi::Function::GetGlobal("device_api.vpi").has_value()) {
    GTEST_SKIP() << "VPI backend is available";
  }
  int selected = 123;
  ExpectError(TVMBackendSetDevice(kDLVPI, 0), "InternalError");
  ExpectError(TVMBackendGetDeviceIndex(kDLVPI, &selected), "InternalError");
  EXPECT_EQ(selected, 123);
}

}  // namespace
}  // namespace runtime
}  // namespace tvm
