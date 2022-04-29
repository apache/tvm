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

#include "../../../../../src/relay/op/memory/on_device.h"

#include <gtest/gtest.h>

#include <string>

namespace tvm {
namespace relay {

TEST(OnDeviceOp, Name) { EXPECT_EQ(OnDeviceOp()->name, "on_device"); }

TEST(OnDevice, Default) {
  Var body("x", {});
  VirtualDevice virtual_device = VirtualDevice::ForDeviceType(kDLCPU, 3);
  Call call = OnDevice(body, virtual_device);
  EXPECT_EQ(call->op, OnDeviceOp());
  EXPECT_EQ(call->args.size(), 1);
  EXPECT_EQ(call->args[0], body);
  const auto* attrs = call->attrs.as<OnDeviceAttrs>();
  ASSERT_TRUE(attrs != nullptr);
  EXPECT_EQ(attrs->virtual_device, virtual_device);
  EXPECT_FALSE(attrs->constrain_result);
  EXPECT_TRUE(attrs->constrain_body);
}

TEST(OnDevice, Fixed) {
  Var body("x", {});
  VirtualDevice virtual_device = VirtualDevice::ForDeviceType(kDLCPU, 3);
  Call call = OnDevice(body, virtual_device, /*constrain_result=*/true);
  const auto* attrs = call->attrs.as<OnDeviceAttrs>();
  ASSERT_TRUE(attrs != nullptr);
  EXPECT_TRUE(attrs->constrain_result);
  EXPECT_TRUE(attrs->constrain_body);
}

TEST(OnDevice, Free) {
  Var body("x", {});
  VirtualDevice virtual_device = VirtualDevice::ForDeviceType(kDLCPU, 3);
  Call call = OnDevice(body, virtual_device, /*constrain_result=*/false, /*constrain_body=*/false);
  const auto* attrs = call->attrs.as<OnDeviceAttrs>();
  ASSERT_TRUE(attrs != nullptr);
  EXPECT_FALSE(attrs->constrain_result);
  EXPECT_FALSE(attrs->constrain_body);
}

TEST(GetOnDeviceProps, Correct) {
  Var body("x", {});
  VirtualDevice virtual_device = VirtualDevice::ForDeviceType(kDLCPU, 3);
  Call call = OnDevice(body, virtual_device, /*constrain_result=*/true, /*constrain_body=*/false);
  OnDeviceProps props = GetOnDeviceProps(call);
  ASSERT_TRUE(props.body.defined());
  ASSERT_EQ(props.virtual_device, virtual_device);
  ASSERT_TRUE(props.constrain_result);
  ASSERT_FALSE(props.constrain_body);
}

TEST(MaybeOnDevice, Wrapped) {
  VirtualDevice virtual_device = VirtualDevice::ForDeviceType(kDLCPU, 3);
  Var body("x", {});
  Call inner = OnDevice(body, virtual_device);
  Call outer = OnDevice(inner, virtual_device);
  OnDeviceProps props = GetOnDeviceProps(outer);
  ASSERT_TRUE(props.body.defined());
  ASSERT_EQ(props.virtual_device, virtual_device);
  ASSERT_FALSE(props.constrain_result);
  ASSERT_TRUE(props.constrain_body);
}

}  // namespace relay
}  // namespace tvm
