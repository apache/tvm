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
#include <tvm/ffi/error.h>
#include <tvm/runtime/c_backend_api.h>

namespace tvm {
namespace runtime {
namespace {

// A device type with no registered device API in this test binary.
constexpr int kUnavailableDeviceType = kDLHexagon;

}  // namespace

TEST(BackendWorkspace, AllocAndFreeOnCPU) {
  void* ptr = TVMBackendAllocWorkspace(kDLCPU, 0, 64, kDLFloat, 32);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(TVMBackendFreeWorkspace(kDLCPU, 0, ptr), 0);
}

TEST(BackendWorkspace, AllocFailureIsReportedThroughCAPI) {
  void* ptr = TVMBackendAllocWorkspace(kUnavailableDeviceType, 0, 64, kDLFloat, 32);
  EXPECT_EQ(ptr, nullptr);
  ffi::Error error = ffi::details::MoveFromSafeCallRaised();
  EXPECT_FALSE(error.message().empty());
}

TEST(BackendWorkspace, FreeFailureIsReportedThroughCAPI) {
  int dummy = 0;
  EXPECT_EQ(TVMBackendFreeWorkspace(kUnavailableDeviceType, 0, &dummy), -1);
  ffi::Error error = ffi::details::MoveFromSafeCallRaised();
  EXPECT_FALSE(error.message().empty());
}

}  // namespace runtime
}  // namespace tvm
