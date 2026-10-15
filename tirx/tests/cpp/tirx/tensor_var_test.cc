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
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/expr.h>

#include <type_traits>
#include <utility>

namespace tvm::tirx {
namespace {

static_assert(std::is_same_v<decltype(std::declval<TensorVar>().operator->()), const VarNode*>);
static_assert(std::is_same_v<decltype(std::declval<TensorVar>().ty()), TensorType>);
static_assert(std::is_same_v<decltype(DefaultIndexType()), PrimType>);
static_assert(
    std::is_same_v<decltype(std::declval<TensorTypeNode>().DefaultIndexType()), PrimType>);

TEST(TensorVar, CheckedViewPreservesOrdinaryVariableIdentity) {
  TensorType type({IntImm(PrimType::Int(32), 16)}, PrimType::Float(16));
  Location loc = SourceLoc(SourceName::Get("tensor.cc"), 1, 2, 1, 8);
  Var var("A", type, loc);
  TensorVar tensor = ffi::GetRef<Var>(var.get()).as_or_throw<TensorVar>();
  EXPECT_EQ(tensor.get(), var.get());
  EXPECT_EQ(tensor.operator->(), var.get());
  EXPECT_EQ(tensor->name, "A");
  EXPECT_TRUE(tensor->loc.same_as(loc));
  EXPECT_TRUE(tensor.ty().same_as(type));
  EXPECT_TRUE(tensor.shape().same_as(type->shape));
  EXPECT_TRUE(tensor.dtype().same_as(type->dtype));
  EXPECT_TRUE(tensor.strides().empty());
  EXPECT_TRUE(prim::IsZero(tensor.elem_offset()));
  EXPECT_EQ(tensor.scope(), type->scope);
  EXPECT_TRUE(tensor == TensorVar(var));
  Var scalar("scalar", PrimType::Int(32));
  EXPECT_THROW((void)scalar.as_or_throw<TensorVar>(), ffi::Error);
  EXPECT_THROW((void)TensorVar(scalar), ffi::Error);
  auto scalar_copy = ffi::make_object<VarNode>(*scalar.get());
  EXPECT_THROW((void)TensorVar(std::move(scalar_copy)), ffi::Error);
}

TEST(TensorVar, RebuiltTypeRetainsMetadataWithFreshIdentity) {
  TensorVar original("A", TensorType({16}));
  auto copied_type = ffi::make_object<TensorTypeNode>(*original.ty().get());
  copied_type->shape = {8};
  TensorVar rebuilt(original->name, TensorType(std::move(copied_type)), original->loc);
  EXPECT_FALSE(original.same_as(rebuilt));
  EXPECT_FALSE(original.ty().same_as(rebuilt.ty()));
  EXPECT_TRUE(original != rebuilt);
  EXPECT_EQ(original.shape()[0].as<IntImmNode>()->value, 16);
  EXPECT_EQ(rebuilt.shape()[0].as<IntImmNode>()->value, 8);
  EXPECT_EQ(rebuilt->name, original->name);
  EXPECT_TRUE(rebuilt->loc.same_as(original->loc));
  EXPECT_EQ(rebuilt.ty()->data_alignment, runtime::kAllocAlignment);
  EXPECT_EQ(rebuilt.ty()->offset_factor, 1);
  EXPECT_FALSE(rebuilt.ty()->layout.has_value());
}

TEST(TensorVar, PreferredIndexTypeUsesShapeOrConfiguredDefault) {
  TensorType scalar(ffi::Array<PrimExpr>{});
#if TVM_INDEX_DEFAULT_I64
  EXPECT_EQ(DefaultIndexType().bits(), 64);
#else
  EXPECT_EQ(DefaultIndexType().bits(), 32);
#endif
  EXPECT_TRUE(scalar->DefaultIndexType().same_as(DefaultIndexType()));
  EXPECT_TRUE(scalar->elem_offset.ty().same_as(DefaultIndexType()));
  for (int bits : {32, 64}) {
    PrimType index_type = PrimType::Int(bits);
    TensorType tensor({IntImm(index_type, 16)});
    EXPECT_TRUE(tensor->DefaultIndexType().same_as(index_type));
  }
}

}  // namespace
}  // namespace tvm::tirx
