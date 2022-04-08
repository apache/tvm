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
#include <tvm/relay/dataflow_pattern.h>
#include <tvm/tir/analysis.h>

TEST(DFPattern, IsVar) {
  using namespace tvm;
  using namespace tvm::relay;
  auto pattern = IsVar("add");
  auto* node = pattern.as<VarPatternNode>();
  ICHECK(node);
  ICHECK(node->name == String("add"));
}

TEST(DFPattern, IsConstant) {
  using namespace tvm;
  using namespace tvm::relay;
  auto pattern = IsConstant();
  auto* node = pattern.as<ConstantPatternNode>();
  ICHECK(node);
}

TEST(DFPattern, IsOp) {
  using namespace tvm;
  using namespace tvm::relay;
  auto pattern = IsOp("add");
  auto* node = pattern.as<ExprPatternNode>();
  ICHECK(node);
  ICHECK(node->expr == Op::Get("add"));
}

TEST(DFPattern, IsTuple) {
  using namespace tvm;
  using namespace tvm::relay;
  auto a = WildcardPattern();
  auto b = WildcardPattern();
  auto pattern = IsTuple({a, b});
  auto* node = pattern.as<TuplePatternNode>();
  ICHECK(node);
  ICHECK(node->fields[0] == a);
  ICHECK(node->fields[1] == b);
}

TEST(DFPattern, IsTupleGetItem) {
  using namespace tvm;
  using namespace tvm::relay;
  auto a = WildcardPattern();
  auto b = WildcardPattern();
  auto tuple = IsTuple({a, b});
  auto pattern = IsTupleGetItem(tuple, 1);
  auto* node = pattern.as<TupleGetItemPatternNode>();
  ICHECK(node);
  ICHECK(node->tuple == tuple);
  ICHECK(node->index == 1);
}

TEST(DFPattern, ADD) {
  using namespace tvm;
  using namespace tvm::relay;
  auto a = WildcardPattern();
  auto b = WildcardPattern();
  auto pattern = a + b;
  auto* node = pattern.as<CallPatternNode>();
  ICHECK(node);
  ICHECK(node->args[0] == a);
  ICHECK(node->args[1] == b);
  auto* expr_pattern = node->op.as<ExprPatternNode>();
  ICHECK(expr_pattern);
  ICHECK(expr_pattern->expr == Op::Get("add"));
}

TEST(DFPattern, SUB) {
  using namespace tvm;
  using namespace tvm::relay;
  auto a = WildcardPattern();
  auto b = WildcardPattern();
  auto pattern = a - b;
  auto* node = pattern.as<CallPatternNode>();
  ICHECK(node);
  ICHECK(node->args[0] == a);
  ICHECK(node->args[1] == b);
  auto* expr_pattern = node->op.as<ExprPatternNode>();
  ICHECK(expr_pattern);
  ICHECK(expr_pattern->expr == Op::Get("subtract"));
}

TEST(DFPattern, MUL) {
  using namespace tvm;
  using namespace tvm::relay;
  auto a = WildcardPattern();
  auto b = WildcardPattern();
  auto pattern = a * b;
  auto* node = pattern.as<CallPatternNode>();
  ICHECK(node);
  ICHECK(node->args[0] == a);
  ICHECK(node->args[1] == b);
  auto* expr_pattern = node->op.as<ExprPatternNode>();
  ICHECK(expr_pattern);
  ICHECK(expr_pattern->expr == Op::Get("multiply"));
}

TEST(DFPattern, DIV) {
  using namespace tvm;
  using namespace tvm::relay;
  auto a = WildcardPattern();
  auto b = WildcardPattern();
  auto pattern = a / b;
  auto* node = pattern.as<CallPatternNode>();
  ICHECK(node);
  ICHECK(node->args[0] == a);
  ICHECK(node->args[1] == b);
  auto* expr_pattern = node->op.as<ExprPatternNode>();
  ICHECK(expr_pattern);
  ICHECK(expr_pattern->expr == Op::Get("divide"));
}

TEST(DFPattern, OR) {
  using namespace tvm;
  using namespace tvm::relay;
  auto a = WildcardPattern();
  auto b = WildcardPattern();
  auto pattern = a || b;
  auto* node = pattern.as<AltPatternNode>();
  ICHECK(node);
  ICHECK(node->left == a);
  ICHECK(node->right == b);
}

TEST(DFPattern, Optional) {
  using namespace tvm;
  using namespace tvm::relay;
  DFPattern a = WildcardPattern();
  DFPattern b = WildcardPattern();
  auto pattern = a.Optional([b](const DFPattern& other) { return other + b; });
  auto* node = pattern.as<AltPatternNode>();
  ICHECK(node);
  ICHECK(node->left == a);
  auto* right_node = node->right.as<CallPatternNode>();
  ICHECK(right_node);
  ICHECK(right_node->args.size() == 2);
  ICHECK(right_node->args[0] == a);
  ICHECK(right_node->args[1] == b);
  auto* expr_pattern = right_node->op.as<ExprPatternNode>();
  ICHECK(expr_pattern);
  ICHECK(expr_pattern->expr == Op::Get("add"));
}

TEST(DFPattern, HasAttr) {
  using namespace tvm;
  using namespace tvm::relay;
  auto a = WildcardPattern();
  Map<String, ObjectRef> attrs;
  auto b = String("b");
  attrs.Set("a", b);
  auto pattern = a.HasAttr(attrs);
  auto* node = pattern.as<AttrPatternNode>();
  ICHECK(node);
  ICHECK(node->pattern == a);
  ICHECK(node->attrs->dict.at("a") == b);
}

TEST(DFPattern, HasType) {
  using namespace tvm;
  using namespace tvm::relay;
  auto a = WildcardPattern();
  TensorType type({1, 2, 3}, DataType(runtime::String2DLDataType("float32")));
  auto pattern = a.HasType(type);
  auto* node = pattern.as<TypePatternNode>();
  ICHECK(node);
  ICHECK(node->pattern == a);
  ICHECK(node->type == type);
}

TEST(DFPattern, HasDtype) {
  using namespace tvm;
  using namespace tvm::relay;
  auto a = WildcardPattern();
  auto pattern = a.HasDtype("float32");
  auto* node = pattern.as<DataTypePatternNode>();
  ICHECK(node);
  ICHECK(node->pattern == a);
  ICHECK(runtime::DLDataType2String(node->dtype.operator DLDataType()) == "float32");
}

TEST(DFPattern, HasShape) {
  using namespace tvm;
  using namespace tvm::relay;
  auto a = WildcardPattern();
  Array<PrimExpr> shape{1, 2, 3};
  auto pattern = a.HasShape(shape);
  auto* node = pattern.as<ShapePatternNode>();
  ICHECK(node);
  ICHECK(node->pattern == a);
  ICHECK(node->shape == shape);
}
