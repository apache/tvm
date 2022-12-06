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
#include <tvm/tir/builtin.h>
#include <tvm/tir/data_type_rewriter.h>
#include <tvm/tir/op.h>

using namespace tvm;
using namespace tvm::tir;
using namespace tvm::runtime;

using BinaryOpTypes =
    ::testing::Types<Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod, Min, Max, EQ, NE, LT, LE, GT, GE>;

template <typename T>
class DataTypeLegalizerBinaryOp : public ::testing::Test {};

TYPED_TEST_SUITE(DataTypeLegalizerBinaryOp, BinaryOpTypes);

TYPED_TEST(DataTypeLegalizerBinaryOp, Basic) {
  using RefType = TypeParam;
  using NodeType = typename RefType::ContainerType;
  auto node = make_object<NodeType>();
  node->a = Var("a", DataType::Int(32));
  node->b = IntImm(DataType::Int(64), 2);
  DataTypeLegalizer legalizer;
  auto new_expr = Downcast<RefType>(legalizer(RefType(node)));
  auto target_dtype = DataType::Int(64);
  ASSERT_EQ(new_expr->a.dtype(), target_dtype);
  ASSERT_EQ(new_expr->b.dtype(), target_dtype);
}

TEST(DataTypeLegalizer, Select) {
  auto node = make_object<SelectNode>();
  node->condition = Var("cond", DataType::Bool());
  node->true_value = Var("a", DataType::Int(64));
  node->false_value = IntImm(DataType::Int(32), 2);
  DataTypeLegalizer legalizer;
  Select new_select = Downcast<Select>(legalizer(Select(node)));
  auto target_dtype = DataType::Int(64);
  ASSERT_EQ(new_select->true_value.dtype(), target_dtype);
  ASSERT_EQ(new_select->false_value.dtype(), target_dtype);
  ASSERT_EQ(new_select.dtype(), target_dtype);
  ASSERT_EQ(new_select->condition.dtype(), node->condition.dtype());
}
TEST(DataTypeLegalizer, IfThenElse) {
  auto cond = Var("cond", DataType::Bool());
  PrimExpr call = Call(DataType::Int(32), builtin::if_then_else(),
                       {cond, Var("a", DataType::Int(64)), IntImm(DataType::Int(32), 2)});
  DataTypeLegalizer legalizer;
  Call new_call = Downcast<Call>(legalizer(call));
  auto target_dtype = DataType::Int(64);
  ASSERT_EQ(new_call->args[1].dtype(), target_dtype);
  ASSERT_EQ(new_call->args[2].dtype(), target_dtype);
  ASSERT_EQ(new_call->dtype, target_dtype);
}

TEST(DataTypeLegalizer, Block) {
  auto block_node = make_object<BlockNode>();
  auto iter_var_node = make_object<IterVarNode>();
  iter_var_node->var = Var("i", DataType::Int(32));
  iter_var_node->dom =
      Range::FromMinExtent(IntImm(DataType::Int(64), 0), IntImm(DataType::Int(64), 10));
  iter_var_node->iter_type = IterVarType::kDataPar;
  block_node->iter_vars = {IterVar(iter_var_node)};
  block_node->reads = {};
  block_node->writes = {};
  block_node->name_hint = "block";
  block_node->body = Evaluate(Integer(0));
  auto block_realize_node = make_object<BlockRealizeNode>();
  auto loop_var = Var("i", DataType::Int(32));
  block_realize_node->iter_values = {loop_var};
  block_realize_node->predicate = const_true();
  block_realize_node->block = Block(block_node);
  auto for_node = make_object<ForNode>();
  for_node->loop_var = loop_var;
  for_node->min = IntImm(DataType::Int(64), 0);
  for_node->extent = IntImm(DataType::Int(64), 10);
  for_node->kind = ForKind::kSerial;
  for_node->body = BlockRealize(block_realize_node);
  Stmt stmt = For(for_node);

  DataTypeLegalizer legalizer;
  DataType target_dtype = loop_var->dtype;
  Stmt new_stmt = legalizer(stmt);
  const ForNode* new_for = new_stmt.as<ForNode>();
  ASSERT_EQ(new_for->loop_var.dtype(), target_dtype);
  ASSERT_EQ(new_for->min.dtype(), target_dtype);
  ASSERT_EQ(new_for->extent.dtype(), target_dtype);
  const BlockRealizeNode* new_block_realize = new_for->body.as<BlockRealizeNode>();
  ASSERT_EQ(new_block_realize->iter_values[0].dtype(), target_dtype);
  const BlockNode* new_block = new_block_realize->block.as<BlockNode>();
  ASSERT_EQ(new_block->iter_vars[0]->dom->min.dtype(), target_dtype);
  ASSERT_EQ(new_block->iter_vars[0]->dom->extent.dtype(), target_dtype);
  ASSERT_EQ(new_block->iter_vars[0]->var.dtype(), target_dtype);
}

TEST(DataTypeLegalizer, For) {
  auto node = make_object<ForNode>();
  node->body = Evaluate(Integer(0));
  node->loop_var = Var("i", DataType::Int(32));
  node->min = IntImm(DataType::Int(64), 0);
  node->extent = IntImm(DataType::Int(64), 10);
  DataTypeLegalizer legalizer;
  For new_for = Downcast<For>(legalizer(For(node)));
  ASSERT_EQ(new_for->min.dtype(), DataType::Int(32));
  ASSERT_EQ(new_for->extent.dtype(), DataType::Int(32));
  ASSERT_EQ(new_for->loop_var.dtype(), DataType::Int(32));
}

TEST(DataTypeLegalizer, Ramp) {
  auto node = make_object<RampNode>();
  node->base = IntImm(DataType::Int(64), 0);
  node->stride = IntImm(DataType::Int(32), 1);
  int lanes = 4;
  node->lanes = lanes;
  DataTypeLegalizer legalizer;
  Ramp new_ramp = Downcast<Ramp>(legalizer(Ramp(node)));
  DataType target_dtype = DataType::Int(64);
  ASSERT_EQ(new_ramp->base.dtype(), target_dtype);
  ASSERT_EQ(new_ramp->stride.dtype(), target_dtype);
  ASSERT_EQ(new_ramp->dtype, target_dtype.with_lanes(lanes));
}
