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
#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/runtime/tensor.h>

namespace tvm {
namespace runtime {

TEST(MemoryScope, CanonicalDefaultAndTypedBoundaries) {
  MemoryScope global;
  EXPECT_TRUE(global.same_as(MemoryScope::Global()));
  EXPECT_TRUE(global.same_as(MemoryScope::Get("global")));
  EXPECT_EQ(PtrType(PrimType::Int(32))->storage_scope, global);
  EXPECT_EQ(PtrType(PrimType::Int(32), "")->storage_scope, global);
  EXPECT_EQ(ffi::Any(ffi::String("shared")).cast<MemoryScope>(), MemoryScope::Get("shared"));
  EXPECT_THROW(MemoryScope::Get("memory_scope_test.missing"), ffi::Error);
  EXPECT_THROW(ffi::Any(ffi::String("memory_scope_test.missing")).cast<MemoryScope>(), ffi::Error);
  EXPECT_THROW(MemoryScope::Register("global"), ffi::Error);
  EXPECT_THROW(MemoryScope::Register(""), ffi::Error);
}

TEST(MemoryScope, ExtensionRegistrationAndAttributes) {
  ffi::reflection::EnumDef<MemoryScopeNode>("memory_scope_test.scratch")
      .set_attr("memory_scope_test.capacity", 1024);
  auto scope = MemoryScope::Get("memory_scope_test.scratch");
  EXPECT_EQ(scope.name(), "memory_scope_test.scratch");
  EXPECT_EQ(scope, ffi::Any(scope).cast<MemoryScope>());
  EXPECT_EQ(scope, PtrType(PrimType::Float(32), scope)->storage_scope);
  EXPECT_THROW(MemoryScope::Register(scope.name()), ffi::Error);
  auto registered = MemoryScope::Register("memory_scope_test.registered");
  EXPECT_EQ(registered, MemoryScope::Get(registered.name()));
}

TEST(MemoryScope, GlobalTensorAllocation) {
  auto tensor = Empty(ffi::Shape({8}), DLDataType{kDLFloat, 32, 1}, DLDevice{kDLCPU, 0},
                      MemoryScope::Global());
  EXPECT_EQ(tensor.shape()[0], 8);
  EXPECT_THROW(Empty(ffi::Shape({8}), DLDataType{kDLFloat, 32, 1}, DLDevice{kDLCPU, 0},
                     MemoryScope::Get("shared")),
               ffi::Error);
}

}  // namespace runtime
}  // namespace tvm
