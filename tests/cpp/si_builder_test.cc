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

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/ir/si_builder.h>
#include <tvm/ir/source_map.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

tvm::Span _CreateSpan(std::string text) {
  return tvm::Span(tvm::SourceName::Get(text), 0, 0, 0, 0);
}

class RelayCheckSpan : public tvm::relay::ExprVisitor {
 public:
  std::vector<tvm::Span> tmp_result_;
  std::vector<tvm::Span> lhs_spans_;
  std::vector<tvm::Span> rhs_spans_;

  std::vector<tvm::Span> CollectSpan(tvm::relay::Expr expr) {
    tmp_result_.clear();
    VisitExpr(expr);
    return tmp_result_;
  }

  void Check(tvm::relay::Expr lhs, tvm::relay::Expr rhs) {
    tvm::relay::Function lhs_f =
        tvm::relay::Function(tvm::relay::FreeVars(lhs), lhs, tvm::relay::Type(), {});
    tvm::relay::Function rhs_f =
        tvm::relay::Function(tvm::relay::FreeVars(rhs), rhs, tvm::relay::Type(), {});
    EXPECT_TRUE(tvm::StructuralEqual()(lhs_f, rhs_f));
    lhs_spans_ = CollectSpan(lhs);
    rhs_spans_ = CollectSpan(rhs);

    EXPECT_EQ(lhs_spans_.size(), rhs_spans_.size());
    for (std::size_t i = 0; i != lhs_spans_.size(); i++) {
      EXPECT_TRUE(tvm::StructuralEqual()(lhs_spans_[i], rhs_spans_[i]));
    }
  }

  void VisitExpr(const tvm::relay::Expr& expr) {
    if (expr->span.defined()) {
      tmp_result_.push_back(expr->span);
    }
    using TParent = ExprFunctor<void(const tvm::relay::Expr&)>;
    TParent::VisitExpr(expr);
    visit_counter_.emplace(expr.get(), 1);
  }
};

TEST(SIBuilder, SequentialSpan) {
  using namespace tvm;
  Array<Span> ingredients = {_CreateSpan("first"), _CreateSpan("second"), _CreateSpan("third")};

  SequentialSpan seq_span_1{ingredients[0], ingredients[1]};
  EXPECT_EQ(seq_span_1->spans.size(), 2);
  for (std::size_t i = 0; i != seq_span_1->spans.size(); i++) {
    EXPECT_EQ(seq_span_1->spans[i], ingredients[i]);
  }

  // nested SequentialSpan test
  SequentialSpan seq_span_2{seq_span_1, ingredients[2]};
  EXPECT_EQ(seq_span_2->spans.size(), 3);
  for (std::size_t i = 0; i != seq_span_2->spans.size(); i++) {
    EXPECT_EQ(seq_span_2->spans[i], ingredients[i]);
  }

  // Array constructor test
  Array<Span> tvm_array(ingredients);
  SequentialSpan seq_span_3(tvm_array);
  EXPECT_EQ(seq_span_3->spans.size(), 3);
  for (std::size_t i = 0; i != seq_span_3->spans.size(); i++) {
    EXPECT_EQ(seq_span_3->spans[i], ingredients[i]);
  }
}

TEST(SIBuilder, CreateSapn) {
  using namespace tvm;
  auto pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("ir.enable_si_builder", Bool(true));
  tvm::With<transform::PassContext> ctx_scope(pass_ctx);
  Span span_1 = _CreateSpan("first");
  {
    SIBuilder si_builder(span_1);
    EXPECT_EQ(span_1, si_builder.Build());
  }

  Span span_2 = _CreateSpan("second");
  Array<Span> ingredients = {span_1, span_2};
  SequentialSpan seq_span_1{ingredients[0], ingredients[1]};
  {
    SIBuilder si_builder_1(seq_span_1);
    SIBuilder si_builder_2({span_1, span_2});
    SIBuilder si_builder_3{span_1, span_2};

    Span created_span_1 = si_builder_1.Build();
    Span created_span_2 = si_builder_2.Build();
    Span created_span_3 = si_builder_3.Build();

    auto created_seq_span_1 = created_span_1.as<SequentialSpanNode>();
    auto created_seq_span_2 = created_span_2.as<SequentialSpanNode>();
    auto created_seq_span_3 = created_span_3.as<SequentialSpanNode>();
    EXPECT_EQ(created_seq_span_1->spans.size(), 2);
    EXPECT_EQ(created_seq_span_2->spans.size(), 2);
    EXPECT_EQ(created_seq_span_3->spans.size(), 2);
    for (std::size_t i = 0; i != 2; i++) {
      EXPECT_EQ(created_seq_span_1->spans[i], ingredients[i]);
      EXPECT_EQ(created_seq_span_2->spans[i], ingredients[i]);
      EXPECT_EQ(created_seq_span_3->spans[i], ingredients[i]);
    }
  }
}

TEST(SIBuilder, DisableSIBuilder) {
  using namespace tvm;
  auto pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("ir.enable_si_builder", Bool(false));
  tvm::With<transform::PassContext> ctx_scope(pass_ctx);
  Span span_1 = _CreateSpan("first");
  {
    SIBuilder si_builder(span_1);
    EXPECT_NE(span_1, si_builder.Build());
  }
}

TEST(SIBuilder, RelayRecursivelyFill) {
  using namespace tvm;
  auto pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("ir.enable_si_builder", Bool(true));
  tvm::With<transform::PassContext> ctx_scope(pass_ctx);
  Span test_span = _CreateSpan("test_span");
  Span a_node_span = _CreateSpan("a_node");

  auto tensor_type = relay::TensorType({2, 3}, tvm::DataType::Float(32));
  relay::Expr add_op = relay::Op::Get("add");
  relay::Expr relu_op = relay::Op::Get("nn.relu");
  relay::Expr leaky_relu_op = relay::Op::Get("nn.leaky_relu");
  // Reset span of OpNode. Because a relay Op Node is a static reference, any change on it will
  // be assigned the original object.
  add_op->span = Span();
  relu_op->span = Span();
  leaky_relu_op->span = Span();

  relay::Expr a = relay::Var("a", tensor_type, a_node_span);
  relay::Expr x = relay::Call(relu_op, {a}, tvm::Attrs(), {});
  relay::Expr y = relay::Call(leaky_relu_op, {x}, tvm::Attrs(), {});
  relay::Expr z = relay::Call(add_op, {y, x}, tvm::Attrs(), {});

  relay::Expr expected_a = relay::Var("a", tensor_type, a_node_span);
  relay::Expr expected_x = relay::Call(relu_op, {expected_a}, tvm::Attrs(), {}, test_span);
  relay::Expr expected_y = relay::Call(leaky_relu_op, {expected_x}, tvm::Attrs(), {}, test_span);
  relay::Expr expected_z =
      relay::Call(add_op, {expected_y, expected_x}, tvm::Attrs(), {}, test_span);

  SIBuilder si_builder(test_span);
  si_builder.RecursivelyFillSpan(z, {a});
  RelayCheckSpan checker;
  checker.Check(z, expected_z);
}

TEST(SIBuilder, RelayCollectSpans) {
  using namespace tvm;
  auto pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("ir.enable_si_builder", Bool(true));
  tvm::With<transform::PassContext> ctx_scope(pass_ctx);
  Span a_node_span = _CreateSpan("a_node");
  Span x_node_span = _CreateSpan("x_node");
  Span y_node_span = _CreateSpan("y_node");
  Span z_node_span = _CreateSpan("z_node");
  std::vector<Span> target = {z_node_span, y_node_span, x_node_span, a_node_span};

  auto tensor_type = relay::TensorType({2, 3}, tvm::DataType::Float(32));
  relay::Expr add_op = relay::Op::Get("add");
  relay::Expr relu_op = relay::Op::Get("nn.relu");
  relay::Expr leaky_relu_op = relay::Op::Get("nn.leaky_relu");
  // Reset span of OpNode. Because a relay Op Node is a static reference, any change on it will
  // be assigned the original object.
  add_op->span = Span();
  relu_op->span = Span();
  leaky_relu_op->span = Span();

  relay::Expr a = relay::Var("a", tensor_type, a_node_span);
  relay::Expr x = relay::Call(relu_op, {a}, tvm::Attrs(), {}, x_node_span);
  relay::Expr y = relay::Call(leaky_relu_op, {x}, tvm::Attrs(), {}, y_node_span);
  relay::Expr z = relay::Call(add_op, {y, x}, tvm::Attrs(), {}, z_node_span);

  SIBuilder si_builder(z, {a});
  Span created_span = si_builder.Build();
  auto created_seq_span = created_span.as<SequentialSpanNode>();
  EXPECT_EQ(created_seq_span->spans.size(), 4);
  for (std::size_t i = 0; i != created_seq_span->spans.size(); i++) {
    EXPECT_TRUE(StructuralEqual()(created_seq_span->spans[i], target[i]));
  }
}

TEST(SIBuilder, TirCollectSpansPrimExpr) {
  using namespace tvm;
  auto pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("ir.enable_si_builder", Bool(true));
  tvm::With<transform::PassContext> ctx_scope(pass_ctx);
  Span a_node_span = _CreateSpan("a_node");
  Span b_node_span = _CreateSpan("b_node");
  Span x_node_span = _CreateSpan("x_node");
  Span add_1_node_span = _CreateSpan("add_1_node");
  Span add_2_node_span = _CreateSpan("add_2_node");
  Span z_node_span = _CreateSpan("z_node");
  std::vector<Span> target = {z_node_span, add_2_node_span, add_1_node_span, x_node_span,
                              a_node_span};
  tir::Var a("a");
  tir::Var b("b");
  auto x = a + b;
  auto add_1 = x + 1;
  auto add_2 = add_1 + 2;
  auto z = max(add_2, 100);
  x->span = x_node_span;
  a->span = a_node_span;
  b->span = b_node_span;
  add_1->span = add_1_node_span;
  add_2->span = add_2_node_span;
  z->span = z_node_span;

  SIBuilder si_builder(z, {x});
  Span created_span = si_builder.Build();
  auto created_seq_span = created_span.as<SequentialSpanNode>();

  EXPECT_EQ(created_seq_span->spans.size(), 4);
  for (std::size_t i = 0; i != created_seq_span->spans.size(); i++) {
    EXPECT_TRUE(StructuralEqual()(created_seq_span->spans[i], target[i]));
  }
}

TEST(SIBuilder, TirCollectSpansStmtWithPrimInput) {
  using namespace tvm;
  auto pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("ir.enable_si_builder", Bool(true));
  tvm::With<transform::PassContext> ctx_scope(pass_ctx);
  Span a_node_span = _CreateSpan("a_node");
  Span b_node_span = _CreateSpan("b_node");
  Span x_node_span = _CreateSpan("x_node");
  Span z_node_span = _CreateSpan("z_plus_1");
  Span stmt_node_span = _CreateSpan("stmt_node");
  std::vector<Span> target = {stmt_node_span, z_node_span, x_node_span};
  tir::Var a("a");
  tir::Var b("b");
  auto x = a + b;
  x->span = x_node_span;
  auto fmaketest = [&]() {
    auto z = x + 1;
    z->span = z_node_span;
    tir::Stmt ret = te::Evaluate(z);
    return ret;
  };
  auto stmt = fmaketest();
  stmt->span = stmt_node_span;
  SIBuilder si_builder(stmt, {x});
  Span created_span = si_builder.Build();
  auto created_seq_span = created_span.as<SequentialSpanNode>();

  EXPECT_EQ(created_seq_span->spans.size(), 3);
  for (std::size_t i = 0; i != created_seq_span->spans.size(); i++) {
    EXPECT_TRUE(StructuralEqual()(created_seq_span->spans[i], target[i]));
  }
}

TEST(SIBuilder, TirCollectSpansStmtWithStmtInput) {
  using namespace tvm;
  auto pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("ir.enable_si_builder", Bool(true));
  tvm::With<transform::PassContext> ctx_scope(pass_ctx);
  Span zero_node_span = _CreateSpan("zero_node");
  Span body_node_span = _CreateSpan("body_node");
  Span init_node_span = _CreateSpan("init_node");
  Span block_node_span = _CreateSpan("block_node");
  std::vector<Span> target = {block_node_span, init_node_span, body_node_span};

  tir::Stmt zero = tir::Evaluate(Integer(0), zero_node_span);
  tir::Stmt body = tir::Evaluate(Integer(1), body_node_span);
  tir::Stmt init = tir::IfThenElse(tir::const_true(), zero, zero, init_node_span);
  tir::Block block({}, {}, {}, "block", body, init, Array<tir::Buffer>(),
                   Array<tir::MatchBufferRegion>(), Map<String, ObjectRef>(), block_node_span);
  SIBuilder si_builder(block, {init});
  Span created_span = si_builder.Build();
  auto created_seq_span = created_span.as<SequentialSpanNode>();

  EXPECT_EQ(created_seq_span->spans.size(), 3);
  for (std::size_t i = 0; i != created_seq_span->spans.size(); i++) {
    EXPECT_TRUE(StructuralEqual()(created_seq_span->spans[i], target[i]));
  }
}

TEST(SIBuilder, TirRecursivelyFillPrimExpr) {
  using namespace tvm;
  auto pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("ir.enable_si_builder", Bool(true));
  tvm::With<transform::PassContext> ctx_scope(pass_ctx);
  Span test_span = _CreateSpan("test_span");
  tir::Var a("a");
  tir::Var b("b");
  auto x = a + b;
  auto add_1 = x + 1;
  auto add_2 = add_1 + 2;
  auto z = max(add_2, 100);

  SIBuilder si_builder(test_span);
  si_builder.RecursivelyFillSpan(z, {a, b});
  EXPECT_TRUE(!a->span.defined());
  EXPECT_TRUE(!b->span.defined());
  EXPECT_TRUE(StructuralEqual()(x->span, test_span));
  EXPECT_TRUE(StructuralEqual()(add_1->span, test_span));
  EXPECT_TRUE(StructuralEqual()(add_2->span, test_span));
  EXPECT_TRUE(StructuralEqual()(z->span, test_span));

  ObjectRef tmp = z;
  PrimExpr zz = Downcast<PrimExpr>(tmp);
  std::ostringstream os;
  os << z;
  EXPECT_TRUE(zz.same_as(z));
  EXPECT_EQ(os.str(), "T.max(a + b + 1 + 2, 100)");
}

TEST(SIBuilder, TirRecursivelyFillStmtWithPrimInput) {
  using namespace tvm;
  auto pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("ir.enable_si_builder", Bool(true));
  tvm::With<transform::PassContext> ctx_scope(pass_ctx);
  Span test_span = _CreateSpan("test_span");
  tir::Var a("a");
  tir::Var b("b");
  auto x = a + b;
  auto z = x + 1;
  tir::Stmt stmt = te::Evaluate(z);
  SIBuilder si_builder(test_span);
  const std::unordered_set<PrimExpr, ObjectPtrHash, ObjectPtrEqual> inputs = {a, b};
  si_builder.RecursivelyFillSpan(stmt, inputs);

  EXPECT_TRUE(!a->span.defined());
  EXPECT_TRUE(!b->span.defined());
  EXPECT_TRUE(StructuralEqual()(x->span, test_span));
  EXPECT_TRUE(StructuralEqual()(z->span, test_span));
  EXPECT_TRUE(StructuralEqual()(stmt->span, test_span));

  ObjectRef tmp = z;
  PrimExpr zz = Downcast<PrimExpr>(tmp);
  std::ostringstream os;
  os << z;
  EXPECT_TRUE(zz.same_as(z));
  EXPECT_EQ(os.str(), "a + b + 1");
}

TEST(SIBuilder, TirRecursivelyFillStmtWithStmtInput) {
  using namespace tvm;
  auto pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("ir.enable_si_builder", Bool(true));
  tvm::With<transform::PassContext> ctx_scope(pass_ctx);
  tir::Stmt zero = tir::Evaluate(Integer(0));
  tir::Stmt init = tir::IfThenElse(tir::const_true(), zero, zero);
  tir::Stmt body = tir::Evaluate(Integer(1));
  tir::Block block(/*iter_vars=*/{}, /*reads=*/{},
                   /*writes=*/{}, /*name_hint=*/"block", /*body=*/body,
                   /*init=*/init);

  Span test_span = _CreateSpan("test_span");
  const std::unordered_set<tir::Stmt, ObjectPtrHash, ObjectPtrEqual> inputs = {init};
  SIBuilder si_builder(test_span);
  si_builder.RecursivelyFillSpan(block, {init});
  EXPECT_TRUE(!zero->span.defined());
  EXPECT_TRUE(!init->span.defined());
  EXPECT_TRUE(StructuralEqual()(body->span, test_span));
  EXPECT_TRUE(StructuralEqual()(block->span, test_span));

  tir::Stmt expected_zero = tir::Evaluate(Integer(0));
  tir::Stmt expected_init = tir::IfThenElse(tir::const_true(), zero, zero);
  tir::Stmt expected_body = tir::Evaluate(Integer(1));
  tir::Block expected_block(/*iter_vars=*/{}, /*reads=*/{},
                            /*writes=*/{}, /*name_hint=*/"block", /*body=*/expected_body,
                            /*init=*/expected_init);
  EXPECT_TRUE(tvm::StructuralEqual()(block, expected_block));
}
