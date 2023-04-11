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
#include <tvm/ir/module.h>
#include <tvm/node/functor.h>
#include <tvm/relay/function.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

TEST(IRF, Basic) {
  using namespace tvm;
  using namespace tvm::tir;
  Var x("x");
  auto z = x + 1;

  NodeFunctor<int(const ObjectRef& n, int b)> f;
  f.set_dispatch<VarNode>([](const ObjectRef& n, int b) { return b; });
  f.set_dispatch<AddNode>([](const ObjectRef& n, int b) { return b + 2; });
  ICHECK_EQ(f(x, 2), 2);
  ICHECK_EQ(f(z, 2), 4);
}

TEST(IRF, CountVar) {
  using namespace tvm;
  using namespace tvm::tir;
  int n_var = 0;
  Var x("x"), y;

  auto z = x + 1 + y + y;
  tir::PostOrderVisit(z, [&n_var](const ObjectRef& n) {
    if (n.as<VarNode>()) ++n_var;
  });
  ICHECK_EQ(n_var, 2);
}

TEST(IRF, VisitPrimFuncs) {
  using namespace tvm;
  using namespace tvm::tir;
  PrimFunc prim_func(/*params=*/{}, /*body=*/Evaluate(Integer(0)));
  auto c_data = tvm::runtime::NDArray::Empty({1, 2, 3}, {kDLFloat, 32, 1}, {kDLCPU, 0});
  relay::Function relay_func(/*params=*/{}, /*body=*/relay::Expr(relay::Constant(c_data)),
                             /*ret_type=*/relay::Type(), /*ty_params=*/{});
  IRModule mod({
      {GlobalVar("main"), prim_func},
      {GlobalVar("main2"), relay_func},
  });
  int n_visited = 0;
  VisitPrimFuncs(mod, [&](const PrimFuncNode* func) { ++n_visited; });
  ASSERT_EQ(n_visited, 1);
}

TEST(IRF, PreOrderVisit) {
  using namespace tvm;
  using namespace tvm::tir;
  Stmt init = IfThenElse(const_true(), Evaluate(Integer(0)), Evaluate(Integer(0)));
  Stmt body = Evaluate(Integer(1));
  Block block(/*iter_vars=*/{}, /*reads=*/{},
              /*writes=*/{}, /*name_hint=*/"block", /*body=*/body,
              /*init=*/init);
  bool init_visited = false;
  bool stopped_at_if = true;
  bool body_visited = false;
  PreOrderVisit(block, [&](const ObjectRef& n) -> bool {
    if (n->IsInstance<IfThenElseNode>()) {
      init_visited = true;
      return false;
    }
    if (const auto* eval = n.as<EvaluateNode>()) {
      if (const auto* int_imm = eval->value.as<IntImmNode>()) {
        if (int_imm->value == 0) {
          stopped_at_if = false;
        } else if (int_imm->value == 1) {
          body_visited = true;
        } else {
          LOG(FATAL) << "Unreachable";
        }
      }
    }
    return true;
  });
  ASSERT_EQ(init_visited, true);
  ASSERT_EQ(stopped_at_if, true);
  ASSERT_EQ(body_visited, true);
}

TEST(IRF, ExprTransform) {
  using namespace tvm;
  using namespace tvm::tir;
  Var x("x");
  auto z = x + 1;

  class MyExprFunctor : public tir::ExprFunctor<int(const PrimExpr&, int)> {
   public:
    int VisitExpr_(const VarNode* op, int b) final { return b; }
    int VisitExpr_(const IntImmNode* op, int b) final { return op->value; }
    int VisitExpr_(const AddNode* op, int b) final {
      return VisitExpr(op->a, b) + VisitExpr(op->b, b);
    }
  };
  MyExprFunctor f;
  ICHECK_EQ(f(x, 2), 2);
  ICHECK_EQ(f(z, 2), 3);
  try {
    f(z - 1, 2);
    LOG(FATAL) << "should fail";
  } catch (Error&) {
  }
}

TEST(IRF, ExprVisit) {
  using namespace tvm;
  using namespace tvm::tir;
  Var x("x");
  auto z = x + 1;

  class MyVisitor : public tir::ExprFunctor<void(const PrimExpr&)>,
                    public tir::StmtFunctor<void(const Stmt&)> {
   public:
    int count = 0;
    // implementation
    void VisitExpr_(const VarNode* op) final { ++count; }
    void VisitExpr_(const IntImmNode* op) final {}
    void VisitExpr_(const AddNode* op) final {
      VisitExpr(op->a);
      VisitExpr(op->b);
    }
    void VisitStmt_(const EvaluateNode* op) final { VisitExpr(op->value); }
  };
  MyVisitor v;
  v.VisitStmt(Evaluate(z));
  ICHECK_EQ(v.count, 1);
}

TEST(IRF, StmtVisitor) {
  using namespace tvm;
  using namespace tvm::tir;
  Var x("x");
  class MyVisitor : public StmtExprVisitor {
   public:
    int count = 0;
    // implementation
    void VisitExpr_(const VarNode* op) final { ++count; }
  };
  MyVisitor v;
  auto fmaketest = [&]() {
    auto z = x + 1;
    Stmt body = Evaluate(z);
    DataType dtype = DataType::Float(32);
    Var buffer("b", PointerType(PrimType(dtype)));
    return Allocate(buffer, dtype, {z, z}, const_true(), body);
  };
  v(fmaketest());
  ICHECK_EQ(v.count, 3);

  {
    // tests for block and block_realize
    Stmt body = fmaketest();
    DataType dtype = DataType::Float(32);
    Var buf_var("b", PointerType(PrimType(dtype)));
    Buffer buffer = decl_buffer({16});
    body = DeclBuffer(buffer, std::move(body));
    BufferRegion buffer_region(buffer, {Range::FromMinExtent(x + 1, 1)});
    MatchBufferRegion match_buffer_region(decl_buffer({1}), buffer_region);

    // construct block and block_realize
    Block block =
        Block({}, {buffer_region}, {buffer_region}, "block", body, body, {}, {match_buffer_region});
    Stmt block_realize = BlockRealize({}, const_true(), block);

    v.count = 0;
    v(block_realize);
    ICHECK_EQ(v.count, 9);
  }
}

TEST(IRF, StmtMutator) {
  using namespace tvm;
  using namespace tvm::tir;
  Var x("x");

  class MyVisitor : public tir::StmtMutator, public tir::ExprMutator {
   public:
    using StmtMutator::operator();
    using ExprMutator::operator();

   protected:
    // implementation
    PrimExpr VisitExpr_(const AddNode* op) final { return op->a; }
    Stmt VisitStmt_(const SeqStmtNode* op) final { return StmtMutator::VisitSeqStmt_(op, true); }
    PrimExpr VisitExpr(const PrimExpr& expr) final { return ExprMutator::VisitExpr(expr); }
  };
  auto fmakealloc = [&]() {
    auto z = x + 1;
    Stmt body = Evaluate(z);
    DataType dtype = DataType::Float(32);
    Var buffer("b", PointerType(PrimType(dtype)));
    return Allocate(buffer, dtype, {1, z}, const_true(), body);
  };

  auto fmakeif = [&]() {
    auto z = x + 1;
    Stmt body = Evaluate(z);
    return IfThenElse(x, Evaluate(0), body);
  };

  MyVisitor v;
  {
    auto body = fmakealloc();
    Stmt body2 = Evaluate(1);
    Stmt bref = body.as<AllocateNode>()->body;
    auto* extentptr = body.as<AllocateNode>()->extents.get();
    Array<Stmt> arr{std::move(body), body2, body2};
    auto* arrptr = arr.get();
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    ICHECK(arr.get() == arrptr);
    // inplace update body
    ICHECK(arr[0].as<AllocateNode>()->extents[1].same_as(x));
    ICHECK(arr[0].as<AllocateNode>()->extents.get() == extentptr);
    // copy because there is additional refs
    ICHECK(!arr[0].as<AllocateNode>()->body.same_as(bref));
    ICHECK(arr[0].as<AllocateNode>()->body.as<EvaluateNode>()->value.same_as(x));
    ICHECK(bref.as<EvaluateNode>()->value.as<AddNode>());
  }
  {
    Array<Stmt> arr{fmakealloc()};
    // mutate array get reference by another one, triiger copy.
    Array<Stmt> arr2 = arr;
    auto* arrptr = arr.get();
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    ICHECK(arr.get() != arrptr);
    ICHECK(arr[0].as<AllocateNode>()->extents[1].same_as(x));
    ICHECK(!arr2[0].as<AllocateNode>()->extents[1].same_as(x));
    // mutate but no content change.
    arr2 = arr;
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    ICHECK(arr2.get() == arr.get());
  }
  {
    Array<Stmt> arr{fmakeif()};
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    ICHECK(arr[0].as<IfThenElseNode>()->else_case.as<EvaluateNode>()->value.same_as(x));
    // mutate but no content change.
    auto arr2 = arr;
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    ICHECK(arr2.get() == arr.get());
  }

  {
    auto body =
        Evaluate(Call(DataType::Int(32), builtin::call_extern(), {StringImm("xyz"), x + 1}));
    auto res = v(std::move(body));
    ICHECK(res.as<EvaluateNode>()->value.as<CallNode>()->args[1].same_as(x));
  }
  {
    Stmt body = fmakealloc();
    Stmt body2 = Evaluate(1);
    auto* ref2 = body2.get();
    auto* extentptr = body.as<AllocateNode>()->extents.get();
    // construct a recursive SeqStmt.
    body = SeqStmt({body, body2});
    body = SeqStmt({body, body2});
    body = v(std::move(body));
    // the seq get flattened
    ICHECK(body.as<SeqStmtNode>()->size() == 3);
    ICHECK(body.as<SeqStmtNode>()->seq[0].as<AllocateNode>()->extents.get() == extentptr);
    ICHECK(body.as<SeqStmtNode>()->seq[1].get() == ref2);
  }

  {
    // Cannot cow because of bref
    Stmt body = fmakealloc();
    Stmt body2 = Evaluate(1);
    auto* extentptr = body.as<AllocateNode>()->extents.get();
    // construct a recursive SeqStmt.
    body = SeqStmt({body, body2});
    auto bref = body;
    body = SeqStmt({body, body2});
    body = v(std::move(body));
    // the seq get flattened
    ICHECK(body.as<SeqStmtNode>()->seq[0].as<AllocateNode>()->extents.get() != extentptr);
  }

  {
    // tests for block and block_realize
    Stmt body = fmakealloc();
    DataType dtype = DataType::Float(32);
    Var buf_var("b", PointerType(PrimType(dtype)));
    Buffer buffer = decl_buffer({16});
    body = DeclBuffer(buffer, std::move(body));
    BufferRegion buffer_region(buffer, {Range::FromMinExtent(x + 1, 1)});
    MatchBufferRegion match_buffer_region(decl_buffer({1}), buffer_region);
    // construct block and block_realize
    Block block =
        Block({}, {buffer_region}, {buffer_region}, "block", body, body, {}, {match_buffer_region});
    Stmt block_realize = BlockRealize({}, const_true(), block);
    body = v(std::move(block_realize));
    // the body should be changed
    Block new_block = body.as<BlockRealizeNode>()->block;
    ICHECK(new_block->body.as<DeclBufferNode>()->body.as<AllocateNode>()->extents[1].same_as(x));
    ICHECK(new_block->init.as<DeclBufferNode>()->body.as<AllocateNode>()->extents[1].same_as(x));
    ICHECK(new_block->reads[0]->region[0]->min.same_as(x));
    ICHECK(new_block->writes[0]->region[0]->min.same_as(x));
    ICHECK(new_block->match_buffers[0]->source->region[0]->min.same_as(x));
  }
}

TEST(IRF, Substitute) {
  using namespace tvm;
  using namespace tvm::tir;
  DataType dtype = DataType::Float(32);
  Var x("x", PointerType(PrimType(dtype), ""));
  auto fmaketest = [&]() {
    Buffer buffer{/*data=*/x,
                  /*dtype=*/DataType::Float(32),
                  /*shape=*/{},
                  /*strides=*/{},
                  /*elem_offset=*/NullValue<PrimExpr>(),
                  /*name=*/"buf",
                  /*data_alignment=*/1,
                  /*offset_factor=*/1,
                  /*buffer_type=*/BufferType::kDefault};
    return BufferLoad(buffer, {});
  };

  {
    // test substitute buffer var
    Var y = x.copy_with_suffix("subst");
    BufferLoad buffer_load = fmaketest();
    auto f_subst = [&](const Var& var) -> Optional<PrimExpr> {
      if (var.same_as(x)) {
        return y;
      }
      return NullOpt;
    };
    BufferLoad new_buffer_load = Downcast<BufferLoad>(Substitute(buffer_load, f_subst));
    ICHECK(new_buffer_load->buffer->data.same_as(y));
  }

  {
    // test identity substitution
    PrimExpr expr = fmaketest();
    auto f_subst = [&](const Var& var) -> Optional<PrimExpr> { return var; };
    PrimExpr new_expr = Substitute(expr, f_subst);
    // the expression is not changed
    ICHECK(new_expr.same_as(expr));
  }
}
