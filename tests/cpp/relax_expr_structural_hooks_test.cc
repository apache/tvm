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
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

namespace {

template <typename TNode>
void ExpectStructuralHooks() {
  namespace refl = tvm::ffi::reflection;
  for (const char* attr_name :
       {refl::type_attr::kStructuralVisit, refl::type_attr::kStructuralMutate,
        refl::type_attr::kStructuralMaybeInplaceMutate}) {
    refl::TypeAttrColumn column(attr_name);
    EXPECT_EQ(column[TNode::RuntimeTypeIndex()].type_index(), tvm::ffi::TypeIndex::kTVMFFIOpaquePtr)
        << TNode::_type_key << " is missing " << attr_name;
  }
}

TEST(RelaxExprStructuralHooks, EveryConcreteExprHasExplicitHooks) {
  using namespace tvm::relax;
  ExpectStructuralHooks<ShapeExprNode>();
  ExpectStructuralHooks<DataflowVarNode>();
  ExpectStructuralHooks<ConstantNode>();
  ExpectStructuralHooks<StringImmNode>();
  ExpectStructuralHooks<DataTypeImmNode>();
  ExpectStructuralHooks<SeqExprNode>();
  ExpectStructuralHooks<IfNode>();
  ExpectStructuralHooks<FunctionNode>();
  ExpectStructuralHooks<ExternFuncNode>();
}

TEST(RelaxExprStructuralHooks, ReviewedCoreExprNodesHaveExplicitHooks) {
  ExpectStructuralHooks<tvm::GlobalVarNode>();
  ExpectStructuralHooks<tvm::OpNode>();
  ExpectStructuralHooks<tvm::tirx::PrimFuncNode>();
}

TEST(RelaxExprStructuralHooks, OpHookPreservesIdentityWithoutDescendingIntoMetadata) {
  using namespace tvm;
  Op input = Op::Get("ir.prim.likely");
  int op_callbacks = 0;
  int metadata_callbacks = 0;
  auto observe_op = [&](const Op&) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    ++op_callbacks;
    return ffi::Unchanged();
  };
  auto observe_metadata = [&](const ffi::String&) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    ++metadata_callbacks;
    return ffi::Unchanged();
  };

  Op mapped = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(input, observe_op, observe_metadata)
                  .cast<Op>();

  EXPECT_TRUE(mapped.same_as(input));
  EXPECT_EQ(op_callbacks, 1);
  EXPECT_EQ(metadata_callbacks, 0);
}

TEST(RelaxExprStructuralHooks, GlobalVarCallbackReplacementIsMemoized) {
  using namespace tvm;
  GlobalVar symbol("f");
  Expr input = Tuple({symbol, symbol});
  int callbacks = 0;
  auto replace_symbol = [&](const GlobalVar&) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    ++callbacks;
    return ffi::Any(GlobalVar("g"));
  };

  Expr mapped = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(input, replace_symbol).cast<Expr>();
  const auto* tuple = mapped.as<TupleNode>();
  ASSERT_NE(tuple, nullptr);
  ASSERT_EQ(tuple->fields.size(), 2U);
  EXPECT_EQ(callbacks, 1);
  EXPECT_TRUE(tuple->fields[0].same_as(tuple->fields[1]));
  EXPECT_FALSE(tuple->fields[0].same_as(symbol));
}

TEST(RelaxExprStructuralHooks, PrimFuncDescendsIntoBody) {
  using namespace tvm;
  tirx::PrimFunc input({}, tirx::Evaluate(IntImm(PrimType::Int(32), 1)));
  auto replace_one = [](const IntImm& value) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (value->value != 1) return ffi::Unchanged();
    return ffi::Any(IntImm(value.ty().as_or_throw<PrimType>(), 2));
  };

  tirx::PrimFunc mapped =
      ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(input, replace_one).cast<tirx::PrimFunc>();

  const auto* evaluate = mapped->body.as<tirx::EvaluateNode>();
  ASSERT_NE(evaluate, nullptr);
  EXPECT_EQ(evaluate->value.as<IntImmNode>()->value, 2);
}

TEST(RelaxExprStructuralHooks, UnchangedCallbackPreservesAncestorIdentity) {
  using namespace tvm;
  using namespace tvm::relax;
  DataflowVar var("x", AnyType());
  Expr input = SeqExpr({}, var);
  const auto* original = input.get();

  auto miss = [](const DataflowVar&) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    return ffi::Unchanged();
  };
  Expr mapped = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(input, miss).cast<Expr>();
  EXPECT_EQ(mapped.get(), original);
}

TEST(RelaxExprStructuralHooks, DataflowVarMapsItsInheritedTypeField) {
  using namespace tvm;
  using namespace tvm::relax;
  DataflowVar input_var("x", AnyType());
  Function input({input_var}, SeqExpr({}, input_var), AnyType());
  auto replace_any_type = [](const AnyType&) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    return ffi::Any(TensorMapType());
  };
  Function mapped =
      ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(input, replace_any_type).cast<Function>();

  const auto* var = mapped->params[0].as<DataflowVarNode>();
  ASSERT_NE(var, nullptr);
  EXPECT_NE(var->ty.as<TensorMapTypeNode>(), nullptr);
}

TEST(RelaxExprStructuralHooks, StructuralEqualAndHashStillUseReflectedConstants) {
  using namespace tvm;
  using namespace tvm::relax;
  ffi::StructuralEqual equal;
  ffi::StructuralHash hash;

  StringImm lhs("lhs");
  StringImm rhs("rhs");
  EXPECT_FALSE(equal(lhs, rhs));
  EXPECT_NE(hash(lhs), hash(rhs));

  ExternFunc first("first");
  ExternFunc second("second");
  EXPECT_FALSE(equal(first, second));
  EXPECT_NE(hash(first), hash(second));

  Function pure({}, SeqExpr({}, Tuple(ffi::Array<Expr>{})), TupleType(ffi::Array<Type>{}), true);
  Function impure({}, SeqExpr({}, Tuple(ffi::Array<Expr>{})), TupleType(ffi::Array<Type>{}), false);
  EXPECT_FALSE(equal(pure, impure));
  EXPECT_NE(hash(pure), hash(impure));
}

}  // namespace
