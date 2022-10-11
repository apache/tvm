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

#include "../src/runtime/hexagon/hexagon_device_api.h"

using namespace tvm::runtime;
using namespace tvm::runtime::hexagon;

class HexagonDeviceAPITest : public ::testing::Test {
 protected:
  void SetUp() override {
    hexapi = HexagonDeviceAPI::Global();
    cpu_dev.device_type = DLDeviceType(kDLCPU);
    hex_dev.device_type = DLDeviceType(kDLHexagon);
    invalid_dev.device_type = DLDeviceType(kDLExtDev);
    int8.bits = 8;
    int8.code = 0;
    int8.lanes = 1;
  }
  DLDevice cpu_dev;
  DLDevice hex_dev;
  DLDevice invalid_dev;
  DLDataType int8;
  HexagonDeviceAPI* hexapi;
  size_t nbytes{256};
  size_t alignment{64};
  int64_t shape1d[1]{256};
  int64_t shape2d[2]{256, 256};
  int64_t shape3d[3]{256, 256, 256};
  Optional<String> default_scope;
  Optional<String> invalid_scope{"invalid"};
  Optional<String> global_scope{"global"};
  Optional<String> global_vtcm_scope{"global.vtcm"};
};

TEST_F(HexagonDeviceAPITest, global) { CHECK(hexapi != nullptr); }

TEST_F(HexagonDeviceAPITest, alloc_free_cpu) {
  void* buf = hexapi->AllocDataSpace(cpu_dev, nbytes, alignment, int8);
  CHECK(buf != nullptr);
  hexapi->FreeDataSpace(cpu_dev, buf);
}

TEST_F(HexagonDeviceAPITest, alloc_free_hex) {
  void* buf = hexapi->AllocDataSpace(hex_dev, nbytes, alignment, int8);
  CHECK(buf != nullptr);
  hexapi->FreeDataSpace(hex_dev, buf);
}

TEST_F(HexagonDeviceAPITest, alloc_errors) {
  // invalid device
  EXPECT_THROW(hexapi->AllocDataSpace(invalid_dev, nbytes, alignment, int8), InternalError);
  // 0 size
  EXPECT_THROW(hexapi->AllocDataSpace(hex_dev, 0, alignment, int8), InternalError);
  // 0 alignment
  EXPECT_THROW(hexapi->AllocDataSpace(hex_dev, nbytes, 0, int8), InternalError);
}

TEST_F(HexagonDeviceAPITest, free_errors) {
  void* buf = hexapi->AllocDataSpace(hex_dev, nbytes, alignment, int8);

  // invalid device
  EXPECT_THROW(hexapi->FreeDataSpace(invalid_dev, buf), InternalError);
  // invalid pointer
  EXPECT_THROW(hexapi->FreeDataSpace(hex_dev, &buf), InternalError);
  // nullptr
  EXPECT_THROW(hexapi->FreeDataSpace(hex_dev, nullptr), InternalError);
  // double free
  hexapi->FreeDataSpace(hex_dev, buf);
  EXPECT_THROW(hexapi->FreeDataSpace(hex_dev, buf), InternalError);
}

TEST_F(HexagonDeviceAPITest, allocnd_free_cpu) {
  void* buf = hexapi->AllocDataSpace(cpu_dev, 3, shape3d, int8, global_scope);
  CHECK(buf != nullptr);
  hexapi->FreeDataSpace(cpu_dev, buf);
}

TEST_F(HexagonDeviceAPITest, allocnd_free_hex) {
  void* buf = hexapi->AllocDataSpace(hex_dev, 3, shape3d, int8, global_scope);
  CHECK(buf != nullptr);
  hexapi->FreeDataSpace(hex_dev, buf);
}

TEST_F(HexagonDeviceAPITest, allocnd_free_hex_vtcm) {
  void* buf1d = hexapi->AllocDataSpace(hex_dev, 1, shape1d, int8, global_vtcm_scope);
  CHECK(buf1d != nullptr);
  hexapi->FreeDataSpace(hex_dev, buf1d);

  void* buf2d = hexapi->AllocDataSpace(hex_dev, 2, shape2d, int8, global_vtcm_scope);
  CHECK(buf2d != nullptr);
  hexapi->FreeDataSpace(hex_dev, buf2d);
}

TEST_F(HexagonDeviceAPITest, allocnd_erros) {
  // invalid device
  EXPECT_THROW(hexapi->AllocDataSpace(invalid_dev, 2, shape2d, int8, global_vtcm_scope),
               InternalError);

  // Hexagon VTCM allocations must have 0 (scalar) 1 or 2 dimensions
  EXPECT_THROW(hexapi->AllocDataSpace(hex_dev, 3, shape3d, int8, global_vtcm_scope), InternalError);

  // null shape
  EXPECT_THROW(hexapi->AllocDataSpace(hex_dev, 2, nullptr, int8, global_vtcm_scope), InternalError);

  // null shape
  EXPECT_THROW(hexapi->AllocDataSpace(hex_dev, 2, shape2d, int8, invalid_scope), InternalError);

  // cpu & global.vtcm scope
  EXPECT_THROW(hexapi->AllocDataSpace(cpu_dev, 2, shape2d, int8, global_vtcm_scope), InternalError);
}

TEST_F(HexagonDeviceAPITest, alloc_scalar) {
  void* cpuscalar = hexapi->AllocDataSpace(cpu_dev, 0, new int64_t, int8, global_scope);
  CHECK(cpuscalar != nullptr);

  void* hexscalar = hexapi->AllocDataSpace(hex_dev, 0, new int64_t, int8, global_vtcm_scope);
  CHECK(hexscalar != nullptr);
}

// alloc and free of the same buffer on different devices should throw
// but it currently works with no error
// hexagon and cpu device types may merge long term which would make this test case moot
// disabling this test case, for now
// TODO(HWE): Re-enable or delete this test case once we land on device type strategy
TEST_F(HexagonDeviceAPITest, DISABLED_alloc_free_diff_dev) {
  void* buf = hexapi->AllocDataSpace(hex_dev, nbytes, alignment, int8);
  CHECK(buf != nullptr);
  EXPECT_THROW(hexapi->FreeDataSpace(cpu_dev, buf), InternalError);
}

// Alloc a non-runtime buffer
// Alloc a runtime buffer
// "Release" resources for runtime
// Verify the runtime buffer cannot be freed, but the non-runtime buffer can
// This test should be run last
TEST_F(HexagonDeviceAPITest, leak_resources) {
  hexapi->ReleaseResources();
  void* pre_runtime_buf = hexapi->AllocDataSpace(hex_dev, nbytes, alignment, int8);
  CHECK(pre_runtime_buf != nullptr);
  hexapi->AcquireResources();
  void* runtime_buf = hexapi->AllocDataSpace(hex_dev, nbytes, alignment, int8);
  CHECK(runtime_buf != nullptr);
  hexapi->ReleaseResources();
  EXPECT_THROW(hexapi->FreeDataSpace(hex_dev, runtime_buf), InternalError);
  hexapi->FreeDataSpace(hex_dev, pre_runtime_buf);
  hexapi->AcquireResources();
}

// Ensure thread manager is properly configured and destroyed
// in Acquire/Release
TEST_F(HexagonDeviceAPITest, thread_manager) {
  HexagonThreadManager* threads = hexapi->ThreadManager();
  CHECK(threads != nullptr);
  hexapi->ReleaseResources();
  EXPECT_THROW(hexapi->ThreadManager(), InternalError);
  hexapi->AcquireResources();
}

// Ensure user DMA manager is properly configured and destroyed
// in Acquire/Release
TEST_F(HexagonDeviceAPITest, user_dma) {
  HexagonUserDMA* user_dma = hexapi->UserDMA();
  CHECK(user_dma != nullptr);
  hexapi->ReleaseResources();
  EXPECT_THROW(hexapi->UserDMA(), InternalError);
  hexapi->AcquireResources();
}

// Ensure VTCM pool is properly configured and destroyed
// in Acquire/Release
TEST_F(HexagonDeviceAPITest, vtcm_pool) {
  HexagonVtcmPool* vtcm_pool = hexapi->VtcmPool();
  CHECK(vtcm_pool != nullptr);
  hexapi->ReleaseResources();
  EXPECT_THROW(hexapi->VtcmPool(), InternalError);
  hexapi->AcquireResources();
}
