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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr_functor.h>

#include <vector>

namespace tvm {
namespace {

class PairExprNode : public ExprNode {
 public:
  Expr left;
  Expr right;
  static void RegisterReflection() {
    ffi::reflection::ObjectDef<PairExprNode>()
        .def_ro("left", &PairExprNode::left)
        .def_ro("right", &PairExprNode::right);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.NativePairExpr", PairExprNode, ExprNode);
};
class PairExpr : public Expr {
 public:
  PairExpr(Expr left, Expr right) {
    auto node = ffi::make_object<PairExprNode>();
    node->left = std::move(left);
    node->right = std::move(right);
    data_ = std::move(node);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PairExpr, Expr, PairExprNode);
};
TVM_FFI_STATIC_INIT_BLOCK() { PairExprNode::RegisterReflection(); }

class Collect : public ExprVisitor {
 public:
  using ExprVisitor::Visit_;
  std::vector<int64_t> values;
  ffi::Optional<VisitInterrupt> Visit_(const IntImmNode* node) override {
    values.push_back(node->value);
    return std::nullopt;
  }
};

TEST(ExprVisitor, StructuralFallback) {
  auto visitor = ffi::make_object<Collect>();
  PairExpr pair(IntImm::Int32(1), IntImm::Int32(2));
  auto result = visitor->VisitExpected(pair);
  ASSERT_TRUE(result.is_ok());
  EXPECT_FALSE(result.value().has_value());
  EXPECT_EQ(visitor->values, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(pair->left.as<IntImmNode>()->value, 1);
  EXPECT_EQ(pair->right.as<IntImmNode>()->value, 2);
}

class Rewrite : public ExprMutator {
 public:
  using ExprMutator::Mutate_;
  UnchangedOr<PrimExpr> Mutate_(const IntImmNode* node, InplaceMode inplace_mode) override {
    return IntImm::Int32(node->value + 1);
  }
};

TEST(ExprMutator, StructuralFallback) {
  auto mutator = ffi::make_object<Rewrite>();
  PairExpr pair(IntImm::Int32(1), IntImm::Int32(2));
  auto result = mutator->MutateExpected(pair).value();
  ASSERT_FALSE(result.IsUnchanged());
  auto replacement = std::move(result).ValueUnchecked();
  const auto* changed = replacement.as<PairExprNode>();
  ASSERT_NE(changed, nullptr);
  EXPECT_EQ(changed->left.as<IntImmNode>()->value, 2);
  EXPECT_EQ(changed->right.as<IntImmNode>()->value, 3);
  EXPECT_EQ(pair->left.as<IntImmNode>()->value, 1);
  EXPECT_EQ(pair->right.as<IntImmNode>()->value, 2);
}

class ThrowNativeError : public ExprMutator {
 public:
  using ExprMutator::Mutate_;
  ffi::Error error{"ValueError", "native mutation error", ""};
  UnchangedOr<PrimExpr> Mutate_(const IntImmNode*, InplaceMode) override { throw error; }
};

TEST(ExprMutator, NativeErrorContextThroughStructuralFallback) {
  auto mutator = ffi::make_object<ThrowNativeError>();
  PrimExpr sum = prim::Add(IntImm::Int32(1), IntImm::Int32(2));
  PairExpr pair(sum, IntImm::Int32(3));
  auto result = mutator->MutateExpected(pair);
  ASSERT_TRUE(result.is_err());
  EXPECT_TRUE(result.error().same_as(mutator->error));
  auto context = ffi::VisitErrorContext::TryGetFromError(result.error());
  ASSERT_TRUE(context.has_value());
  const auto& pattern = context.value()->reverse_visit_pattern;
  ASSERT_EQ(pattern.size(), 3);
  EXPECT_TRUE(pattern[0].same_as(sum.as<prim::AddNode>()->a));
  EXPECT_TRUE(pattern[1].same_as(sum));
  EXPECT_TRUE(pattern[2].same_as(pair));
}

TEST(TensorRegion, GenericSourceTraversalAndCopyOnWrite) {
  Expr source = Tuple({IntImm::Int32(3)});
  TensorRegion region(source, {Range::FromMinExtent(2, 5)}, PrimType::Int(32));
  auto visitor = ffi::make_object<Collect>();
  visitor->Visit(region);
  EXPECT_EQ(visitor->values, (std::vector<int64_t>{3, 2, 5}));

  auto identity = ffi::make_object<ExprMutator>();
  EXPECT_TRUE(identity->Mutate(region).IsUnchanged());
  auto mutator = ffi::make_object<Rewrite>();
  auto changed = mutator->Mutate(region).ValueUnchecked().as_or_throw<TensorRegion>();
  EXPECT_FALSE(changed.same_as(region));
  EXPECT_EQ(changed->source.as<TupleNode>()->fields[0].as<IntImmNode>()->value, 4);
  EXPECT_EQ(changed->region[0]->min.as<IntImmNode>()->value, 3);
  EXPECT_EQ(changed->region[0]->extent.as<IntImmNode>()->value, 6);
  EXPECT_EQ(region->source.as<TupleNode>()->fields[0].as<IntImmNode>()->value, 3);
  EXPECT_EQ(region->region[0]->min.as<IntImmNode>()->value, 2);
  EXPECT_TRUE(changed->ty.same_as(region->ty));
}

TEST(TensorRegion, NativeAndStructuralInplaceMutation) {
  auto make_region = []() {
    return TensorRegion(Tuple({IntImm::Int32(3)}), {Range::FromMinExtent(2, 5)}, PrimType::Int(32));
  };
  TensorRegion native = make_region();
  const auto* native_identity = native.get();
  auto mutator = ffi::make_object<Rewrite>();
  EXPECT_TRUE(mutator->Mutate(native, InplaceMode::kAllow).IsUnchanged());
  EXPECT_EQ(native.get(), native_identity);
  EXPECT_EQ(native->region[0]->extent.as<IntImmNode>()->value, 6);

  TensorRegion structural = make_region();
  const auto* structural_identity = structural.get();
  auto changed =
      ffi::StructuralMutate(std::move(structural),
                            [](const IntImmNode* node, ffi::StructuralMutatorObj*) -> ffi::Any {
                              return IntImm::Int32(node->value + 1);
                            })
          .as_or_throw<TensorRegion>();
  EXPECT_EQ(changed.get(), structural_identity);
  EXPECT_EQ(changed->source.as<TupleNode>()->fields[0].as<IntImmNode>()->value, 4);
  EXPECT_EQ(changed->region[0]->min.as<IntImmNode>()->value, 3);
}

TEST(TensorRegion, RemapsSourceAndInheritedType) {
  PrimVar source("source");
  PrimVar replacement("replacement");
  TensorRegion region(source, {Range::FromMinExtent(source, 1)}, PrimType::Int(32));
  auto mutator = ffi::make_object<ExprMutator>();
  mutator->VarRemapSet(source, replacement);
  auto changed = mutator->Mutate(region).ValueUnchecked().as_or_throw<TensorRegion>();
  EXPECT_TRUE(changed->source.same_as(replacement));
  EXPECT_TRUE(changed->region[0]->min.same_as(replacement));
  EXPECT_TRUE(region->source.same_as(source));

  auto retyped =
      ffi::StructuralMutate(region,
                            [](const PrimTypeNode*, ffi::StructuralMutatorObj*) -> ffi::Any {
                              return PrimType::Int(64);
                            })
          .as_or_throw<TensorRegion>();
  EXPECT_EQ(retyped->ty.as_or_throw<PrimType>().bits(), 64);
  EXPECT_EQ(region->ty.as_or_throw<PrimType>().bits(), 32);
}

}  // namespace
}  // namespace tvm
