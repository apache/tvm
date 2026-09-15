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
  UnchangedOr<ffi::Any> Mutate_(const IntImmNode* node, InplaceMode inplace_mode) override {
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
  UnchangedOr<ffi::Any> Mutate_(const IntImmNode*, InplaceMode) override { throw error; }
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

}  // namespace
}  // namespace tvm
