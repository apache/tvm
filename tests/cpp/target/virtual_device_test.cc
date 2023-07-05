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
#include <tvm/target/target.h>
#include <tvm/target/virtual_device.h>

namespace tvm {
namespace {

TEST(VirtualDevice, Join_Defined) {
  {
    Target target_a = Target("cuda");
    VirtualDevice lhs = VirtualDevice(kDLCUDA, 3);
    VirtualDevice rhs = VirtualDevice(kDLCUDA, -1, target_a, "global");
    Optional<VirtualDevice> actual = VirtualDevice::Join(lhs, rhs);
    EXPECT_TRUE(actual.operator bool());
    VirtualDevice expected = VirtualDevice(kDLCUDA, 3, target_a, "global");
    EXPECT_TRUE(StructuralEqual()(actual.value(), expected));
  }
  {
    Target target_a = Target("cuda");
    VirtualDevice lhs = VirtualDevice(kDLCUDA, -1, target_a, "global");
    VirtualDevice rhs = VirtualDevice(kDLCUDA, 3);
    Optional<VirtualDevice> actual = VirtualDevice::Join(lhs, rhs);
    EXPECT_TRUE(actual.operator bool());
    VirtualDevice expected = VirtualDevice(kDLCUDA, 3, target_a, "global");
    EXPECT_TRUE(StructuralEqual()(actual.value(), expected));
  }
  {
    Target target_a = Target("cuda");
    VirtualDevice lhs = VirtualDevice(kDLCUDA);
    VirtualDevice rhs = VirtualDevice(kDLCUDA, 2, target_a);
    Optional<VirtualDevice> actual = VirtualDevice::Join(lhs, rhs);
    EXPECT_TRUE(actual.operator bool());
    VirtualDevice expected = VirtualDevice(kDLCUDA, 2, target_a);
    EXPECT_TRUE(StructuralEqual()(actual.value(), expected));
  }
  {
    Target target_a = Target("cuda");
    VirtualDevice lhs = VirtualDevice();
    VirtualDevice rhs = VirtualDevice(kDLCUDA, 3, target_a, "global");
    Optional<VirtualDevice> actual = VirtualDevice::Join(lhs, rhs);
    EXPECT_TRUE(actual.operator bool());
    VirtualDevice expected = rhs;
    EXPECT_TRUE(StructuralEqual()(actual.value(), expected));
  }
}

TEST(VirtualDevice, Join_Undefined) {
  {
    VirtualDevice lhs = VirtualDevice(kDLCUDA);
    VirtualDevice rhs = VirtualDevice(kDLCPU);
    Optional<VirtualDevice> actual = VirtualDevice::Join(lhs, rhs);
    EXPECT_FALSE(actual);
  }
  {
    VirtualDevice lhs = VirtualDevice(kDLCUDA, 3);
    VirtualDevice rhs = VirtualDevice(kDLCUDA, 4);
    Optional<VirtualDevice> actual = VirtualDevice::Join(lhs, rhs);
    EXPECT_FALSE(actual);
  }
  {
    VirtualDevice lhs = VirtualDevice(kDLCUDA, 3, Target("cuda"));
    VirtualDevice rhs = VirtualDevice(kDLCUDA, 3, Target("cuda"));
    Optional<VirtualDevice> actual = VirtualDevice::Join(lhs, rhs);
    EXPECT_FALSE(actual);
  }
  {
    VirtualDevice lhs = VirtualDevice(kDLCUDA, 3, Target("cuda"), "local");
    VirtualDevice rhs = VirtualDevice(kDLCUDA, 3, Target("cuda"), "global");
    Optional<VirtualDevice> actual = VirtualDevice::Join(lhs, rhs);
    EXPECT_FALSE(actual);
  }
}

TEST(VirtualDevice, Default) {
  Target target_a = Target("cuda");
  VirtualDevice lhs = VirtualDevice(kDLCUDA, -1, Target(), "global");
  VirtualDevice rhs = VirtualDevice(kDLCUDA, 3, target_a, "local");
  VirtualDevice actual = VirtualDevice::Default(lhs, rhs);
  VirtualDevice expected = VirtualDevice(kDLCUDA, 3, target_a, "global");
  EXPECT_TRUE(StructuralEqual()(actual, expected));
}

TEST(VirtualDevice, Constructor_Invalid) {
  EXPECT_ANY_THROW(VirtualDevice(kDLCPU, -1, Target("cuda")));
}

TEST(VirtualDeviceCache, Memoized) {
  VirtualDeviceCache cache;
  Target target_a = Target("cuda");
  Target target_b = Target("llvm");
  Target target_c = Target("cuda");
  VirtualDevice virtual_device_a = cache.Make(kDLCUDA, 3, target_a, "local");
  VirtualDevice virtual_device_b = cache.Make(kDLCPU, 1, target_b, "global");

  EXPECT_EQ(cache.Make(kDLCUDA, 3, target_a, "local"), virtual_device_a);
  EXPECT_EQ(cache.Make(kDLCPU, 1, target_b, "global"), virtual_device_b);
  EXPECT_NE(cache.Make(kDLCUDA, 2, target_a, "local"), virtual_device_a);
  EXPECT_NE(cache.Make(kDLCPU, 3, target_b, "local"), virtual_device_a);
  EXPECT_NE(cache.Make(kDLCUDA, 3, target_a, "global"), virtual_device_a);
  EXPECT_EQ(cache.Make(kDLCUDA, 3, Target("cuda"), "local"), virtual_device_a);
  EXPECT_NE(cache.Make(kDLCUDA, 3, Target("cuda -max_threads_per_block=4096"), "local"),
            virtual_device_a);
}

}  // namespace
}  // namespace tvm
