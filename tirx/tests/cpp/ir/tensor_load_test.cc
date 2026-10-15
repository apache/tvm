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
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ir/expr_functor.h>

namespace tvm {
namespace {

// This source type deliberately has no dependency on a tensor dialect.
class LoadSourceTypeNode : public TypeNode {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.LoadSourceType", LoadSourceTypeNode, TypeNode);
};

class InvalidLoadSourceTypeNode : public TypeNode {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.InvalidLoadSourceType", InvalidLoadSourceTypeNode,
                                    TypeNode);
};

PrimType InferTestLoad(const TensorLoadNode* load) {
  EXPECT_TRUE(load->source->ty.as<LoadSourceTypeNode>());
  EXPECT_TRUE(load->loc.same_as(load->source->loc));
  TVM_FFI_CHECK(!load->indices.empty(), ValueError) << "test load requires an index";
  int64_t bits = load->indices[0].as<IntImmNode>()->value.as<int64_t>().value();
  TVM_FFI_CHECK(bits > 0, ValueError) << "test load inference failed";
  return PrimType::Int(bits);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<LoadSourceTypeNode>();
  refl::ObjectDef<InvalidLoadSourceTypeNode>();
  // A packed hook can return an arbitrary value; native hooks must obey their
  // non-nullable PrimType return contract before entering the native ABI.
  refl::TypeAttrDef<InvalidLoadSourceTypeNode>().attr(
      type_attr::kTensorLoadInferType,
      ffi::Function::FromTyped([](const TensorLoadNode* load) -> ffi::Any {
        if (load->indices.empty()) return Type::Missing();
        return nullptr;
      }));
  refl::TypeAttrDef<LoadSourceTypeNode>().attr(type_attr::kTensorLoadInferType,
                                               FTensorLoadInferType::FromNative<&InferTestLoad>());
}

Var LoadSource() {
  return Var("source", ffi::GetRef<Type>(ffi::make_object<LoadSourceTypeNode>().get()),
             SourceLoc(SourceName::Get("tensor_load"), 1, 2, 1, 3));
}

TEST(TensorLoad, SourceTypeInference) {
  Var source = LoadSource();
  TensorLoad load(source, {IntImm::Int32(16)}, source->loc);
  EXPECT_TRUE(load->source.same_as(source));
  EXPECT_TRUE(load->loc.same_as(source->loc));
  EXPECT_EQ(load.ty(), PrimType::Int(16));
  // The same VarNode class with another type cannot use this hook.
  EXPECT_THROW(TensorLoad(Var("scalar", PrimType::Int(32)), {}), ffi::Error);
}

TEST(TensorLoad, InferenceFailures) {
  Var source = LoadSource();
  EXPECT_THROW(TensorLoad(source, {}, source->loc), ffi::Error);
  EXPECT_THROW(TensorLoad(source, {IntImm::Int32(-1)}, source->loc), ffi::Error);
  EXPECT_THROW(TensorLoad(Expr(ffi::UnsafeInit{}), {}), ffi::Error);
  EXPECT_THROW(TensorLoad(Var("missing", Type::Missing()), {}), ffi::Error);
  Var invalid_source("invalid",
                     ffi::GetRef<Type>(ffi::make_object<InvalidLoadSourceTypeNode>().get()));
  EXPECT_THROW(TensorLoad(invalid_source, {}), ffi::Error);
  EXPECT_THROW(TensorLoad(invalid_source, {IntImm::Int32(0)}), ffi::Error);
}

TEST(TensorLoad, ExplicitTypeBypassesInference) {
  Var source = LoadSource();
  // This index would fail inference, but Unchecked retains the supplied full type.
  TensorLoad load =
      TensorLoad::Unchecked(PrimType::Float(32, 4), source, {IntImm::Int32(-1)}, source->loc);
  EXPECT_EQ(load.ty(), PrimType::Float(32, 4));
  EXPECT_TRUE(load->source.same_as(source));
  EXPECT_TRUE(load->loc.same_as(source->loc));
  EXPECT_THROW(TensorLoad::Unchecked(PrimType(ffi::UnsafeInit{}), source, {}), ffi::Error);
  auto untyped = ffi::make_object<TensorLoadNode>(source);
  EXPECT_THROW(TensorLoad(std::move(untyped)), ffi::Error);
}

class ReplaceLoadType : public ExprMutator {
 public:
  using ExprMutator::Mutate;
  explicit ReplaceLoadType(Type replacement) : replacement_(std::move(replacement)) {}
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<PrimTypeNode>()) return ffi::Any(replacement_);
    return ExprMutator::Mutate(value, inplace_mode);
  }

 private:
  Type replacement_;
};

TEST(TensorLoad, ExprMutationRequiresPrimitiveType) {
  for (auto mode : {InplaceMode::kDisallow, InplaceMode::kAllow}) {
    auto mutator = ffi::make_object<ReplaceLoadType>(Type::Missing());
    auto load = TensorLoad::Unchecked(PrimType::Float(32), LoadSource(), {});
    EXPECT_THROW(mutator->Mutate_(load.get(), mode), ffi::Error);
    EXPECT_EQ(load.ty(), PrimType::Float(32));

    auto valid_mutator = ffi::make_object<ReplaceLoadType>(PrimType::Float(64));
    auto result = valid_mutator->Mutate_(load.get(), mode);
    auto mapped = std::move(result).ValueOrUnchanged(load);
    EXPECT_EQ(mapped.ty(), PrimType::Float(64));
  }
}

TEST(TensorLoad, StructuralMutationRequiresPrimitiveType) {
  auto replace_type = [](PrimType) -> ffi::Any { return Type::Missing(); };
  using Mutator = ffi::StructuralMapEngine<ffi::StructuralMapEngineBase, ffi::WalkOrder::kPostOrder,
                                           decltype(replace_type)>;
  for (auto mode : {ffi::InplaceMode::kDisallow, ffi::InplaceMode::kAllow}) {
    auto mutator = ffi::make_object<Mutator>(replace_type);
    auto load = TensorLoad::Unchecked(PrimType::Float(32), LoadSource(), {});
    auto result = mutator->MutateExpected(ffi::AnyView(load), mode);
    EXPECT_TRUE(result.is_err());
    EXPECT_EQ(load.ty(), PrimType::Float(32));
  }
}

}  // namespace
}  // namespace tvm
