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
namespace {
std::optional<MemoryAllocProperty> AllocProperty(const MemoryScope& scope) {
  auto state = ffi::reflection::TypeAttrColumn(
                   ffi::type_attr::kEnumState)[MemoryScopeObj::RuntimeTypeIndex()]
                   .cast<ffi::EnumState>();
  auto column = state->attrs.Get(attr::kMemoryAllocProperty.name);
  return attr::kMemoryAllocProperty.ValueFrom(column ? column->Get(scope).value_or(ffi::Any())
                                                     : ffi::Any());
}
}  // namespace

TEST(MemoryScope, CanonicalDefaultAndTypedBoundaries) {
  MemoryScope global;
  EXPECT_TRUE(global.same_as(MemoryScope::Global()));
  EXPECT_TRUE(global.same_as(MemoryScope::Get("global")));
  EXPECT_EQ(PtrType(PrimType::Int(32))->scope, global);
  EXPECT_THROW(ffi::Any(ffi::String("")).cast<MemoryScope>(), ffi::Error);
  EXPECT_EQ(&MemoryScope::Global(), &MemoryScope::Global());
  EXPECT_EQ(ffi::Any(ffi::String("shared")).cast<MemoryScope>(), MemoryScope::Get("shared"));
  EXPECT_THROW(MemoryScope::Get("memory_scope_test.missing"), ffi::Error);
  EXPECT_THROW(ffi::Any(ffi::String("memory_scope_test.missing")).cast<MemoryScope>(), ffi::Error);
  EXPECT_THROW(MemoryScope::Register("global"), ffi::Error);
  EXPECT_THROW(MemoryScope::Register(""), ffi::Error);
}

TEST(MemoryScope, ExtensionRegistrationAndAttributes) {
  ffi::reflection::EnumDef<MemoryScopeObj>("memory_scope_test.scratch")
      .set_attr("memory_scope_test.capacity", 1024);
  auto scope = MemoryScope::Get("memory_scope_test.scratch");
  EXPECT_EQ(scope.name(), "memory_scope_test.scratch");
  EXPECT_FALSE(AllocProperty(scope).has_value());
  EXPECT_EQ(scope, ffi::Any(scope).cast<MemoryScope>());
  EXPECT_EQ(scope, PtrType(PrimType::Float(32), scope)->scope);
  EXPECT_THROW(MemoryScope::Register(scope.name()), ffi::Error);
  auto registered = MemoryScope::Register("memory_scope_test.registered");
  EXPECT_EQ(registered, MemoryScope::Get(registered.name()));
}

TEST(MemoryScope, MembershipQueryPreservesRegistry) {
  auto scope = MemoryScope::Register("memory_scope_test.membership");
  auto state = ffi::reflection::TypeAttrColumn(
                   ffi::type_attr::kEnumState)[MemoryScopeObj::RuntimeTypeIndex()]
                   .cast<ffi::EnumState>();
  auto num_entries = state->entries.size();
  auto num_indexes = state->indexes.size();
  auto num_attrs = state->attrs.size();
  auto global = MemoryScope::Global();
  auto property = AllocProperty(global);
  EXPECT_TRUE(MemoryScope::IsDefined("global"));
  EXPECT_TRUE(MemoryScope::IsDefined(scope.name()));
  EXPECT_FALSE(MemoryScope::IsDefined("memory_scope_test.unknown"));
  EXPECT_FALSE(MemoryScope::IsDefined(""));
  EXPECT_FALSE(MemoryScope::IsDefined(ffi::String("global\0suffix", 13)));
  EXPECT_EQ(state->entries.size(), num_entries);
  EXPECT_EQ(state->indexes.size(), num_indexes);
  EXPECT_EQ(state->attrs.size(), num_attrs);
  EXPECT_TRUE(MemoryScope::Get(scope.name()).same_as(scope));
  EXPECT_TRUE(MemoryScope::Global().same_as(global));
  EXPECT_EQ(AllocProperty(global), property);
  EXPECT_THROW(MemoryScope::Get("memory_scope_test.unknown"), ffi::Error);
}

TEST(MemoryScope, AllocationProperties) {
  EXPECT_EQ(AllocProperty(MemoryScope::Global()), MemoryAllocProperty::kNormal);
  EXPECT_EQ(AllocProperty(MemoryScope::Get("shared")), MemoryAllocProperty::kNormal);
  EXPECT_EQ(AllocProperty(MemoryScope::Get("shared.dyn")), MemoryAllocProperty::kSingleAllocation);
  EXPECT_EQ(AllocProperty(MemoryScope::Get("local")), MemoryAllocProperty::kSeparateAllocations);
  ffi::reflection::EnumDef<MemoryScopeObj>("memory_scope_test.single")
      .set_attr(attr::kMemoryAllocProperty,
                static_cast<int>(MemoryAllocProperty::kSingleAllocation));
  EXPECT_EQ(AllocProperty(MemoryScope::Get("memory_scope_test.single")),
            MemoryAllocProperty::kSingleAllocation);
  ffi::reflection::EnumDef<MemoryScopeObj>("memory_scope_test.invalid_property")
      .set_attr(attr::kMemoryAllocProperty, 99);
  EXPECT_EQ(AllocProperty(MemoryScope::Get("memory_scope_test.invalid_property")),
            static_cast<MemoryAllocProperty>(99));
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
