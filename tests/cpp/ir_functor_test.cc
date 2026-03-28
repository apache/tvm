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
#include <tvm/ir/module.h>
#include <tvm/node/functor.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

TEST(IRF, Basic) {
  using namespace tvm;
  using namespace tvm::tirx;
  Var x("x");
  auto z = x + 1;

  NodeFunctor<int(const ObjectRef& n, int b)> f;
  f.set_dispatch<VarNode>([](const ObjectRef& n, int b) { return b; });
  f.set_dispatch<AddNode>([](const ObjectRef& n, int b) { return b + 2; });
  TVM_FFI_ICHECK_EQ(f(x, 2), 2);
  TVM_FFI_ICHECK_EQ(f(z, 2), 4);
}

TEST(IRF, CountVar) {
  using namespace tvm;
  using namespace tvm::tirx;
  int n_var = 0;
  Var x("x"), y;

  auto z = x + 1 + y + y;
  tirx::PostOrderVisit(z, [&n_var](const ObjectRef& n) {
    if (n.as<VarNode>()) ++n_var;
  });
  TVM_FFI_ICHECK_EQ(n_var, 2);
}

TEST(IRF, PreOrderVisit) {
  using namespace tvm;
  using namespace tvm::tirx;
  Stmt init = IfThenElse(const_true(), Evaluate(Integer(0)), Evaluate(Integer(0)));
  Stmt body = Evaluate(Integer(1));
  SBlock block(/*iter_vars=*/{}, /*reads=*/{},
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
          TVM_FFI_THROW(InternalError) << "Unreachable";
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
  using namespace tvm::tirx;
  Var x("x");
  auto z = x + 1;

  class MyExprFunctor : public tirx::ExprFunctor<int(const PrimExpr&, int)> {
   public:
    int VisitExpr_(const VarNode* op, int b) final { return b; }
    int VisitExpr_(const IntImmNode* op, int b) final { return op->value; }
    int VisitExpr_(const AddNode* op, int b) final {
      return VisitExpr(op->a, b) + VisitExpr(op->b, b);
    }
  };
  MyExprFunctor f;
  TVM_FFI_ICHECK_EQ(f(x, 2), 2);
  TVM_FFI_ICHECK_EQ(f(z, 2), 3);
  try {
    f(z - 1, 2);
    TVM_FFI_THROW(InternalError) << "should fail";
  } catch (Error&) {
  }
}

TEST(IRF, ExprVisit) {
  using namespace tvm;
  using namespace tvm::tirx;
  Var x("x");
  auto z = x + 1;

  class MyVisitor : public tirx::ExprFunctor<void(const PrimExpr&)>,
                    public tirx::StmtFunctor<void(const Stmt&)> {
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
  TVM_FFI_ICHECK_EQ(v.count, 1);
}

TEST(IRF, StmtVisitor) {
  using namespace tvm;
  using namespace tvm::tirx;
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
    Stmt eval_body = Evaluate(z);
    DataType dtype = DataType::Float(32);
    Var data_var("b", PointerType(PrimType(dtype)));
    Buffer buf(data_var, dtype, {z, z}, {}, PrimExpr(), "b", 0, 0, BufferType::kDefault);
    // AllocBuffer is flat (no body). Return as SeqStmt with eval.
    return SeqStmt({AllocBuffer(buf), eval_body});
  };
  v(fmaketest());
  // AllocBuffer visits buffer shape via VisitBufferDef.
  // shape = {z, z} where z = x + 1, so x is visited twice from shape + once from eval = 3
  TVM_FFI_ICHECK_EQ(v.count, 3);

  {
    // tests for block and block_realize
    Stmt body = fmaketest();
    DataType dtype = DataType::Float(32);
    Var buf_var("b", PointerType(PrimType(dtype)));
    Buffer buffer = decl_buffer({16});
    body = SeqStmt({DeclBuffer(buffer), std::move(body)});
    BufferRegion buffer_region(buffer, {Range::FromMinExtent(x + 1, 1)});
    MatchBufferRegion match_buffer_region(decl_buffer({1}), buffer_region);

    // construct block and block_realize
    SBlock block = SBlock({}, {buffer_region}, {buffer_region}, "block", body, body, {},
                          {match_buffer_region});
    Stmt block_realize = SBlockRealize({}, const_true(), block);

    v.count = 0;
    v(block_realize);
    // x visited in: reads range (1), writes range (1), match_buffers range (1),
    // init DeclBuffer(0) + AllocBuffer shape(2) + Evaluate(1) = 3,
    // body DeclBuffer(0) + AllocBuffer shape(2) + Evaluate(1) = 3.
    // Total: 1 + 1 + 1 + 3 + 3 = 9.
    TVM_FFI_ICHECK_EQ(v.count, 9);
  }
}

TEST(IRF, StmtMutator) {
  using namespace tvm;
  using namespace tvm::tirx;
  Var x("x");

  class MyVisitor : public tirx::StmtMutator, public tirx::ExprMutator {
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
    DataType dtype = DataType::Float(32);
    Var data_var("b", PointerType(PrimType(dtype)));
    Buffer buf(data_var, dtype, {1, z}, {}, PrimExpr(), "b", 0, 0, BufferType::kDefault);
    return AllocBuffer(buf);
  };

  auto fmakeif = [&]() {
    auto z = x + 1;
    Stmt body = Evaluate(z);
    return IfThenElse(x, Evaluate(0), body);
  };

  MyVisitor v;
  {
    auto alloc = fmakealloc();
    Stmt body2 = Evaluate(1);
    auto* bufptr = alloc.as<AllocBufferNode>()->buffer.get();
    ffi::Array<Stmt> arr{std::move(alloc), body2, body2};
    auto* arrptr = arr.get();
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    TVM_FFI_ICHECK(arr.get() == arrptr);
    // buffer IS mutated now (AllocBuffer mutator visits buffer shape via VisitBufferDef)
    // shape was {1, x+1}, mutator transforms x+1 -> x, so buffer changes
    TVM_FFI_ICHECK(arr[0].as<AllocBufferNode>()->buffer.get() != bufptr);
  }
  {
    ffi::Array<Stmt> arr{fmakealloc()};
    // mutate array get reference by another one, trigger copy.
    ffi::Array<Stmt> arr2 = arr;
    auto* arrptr = arr.get();
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    TVM_FFI_ICHECK(arr.get() != arrptr);
    // buffer is mutated in arr but not in arr2
    TVM_FFI_ICHECK(arr[0].as<AllocBufferNode>()->buffer.get() !=
                   arr2[0].as<AllocBufferNode>()->buffer.get());
    // mutate but no content change.
    arr2 = arr;
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    TVM_FFI_ICHECK(arr2.get() == arr.get());
  }
  {
    ffi::Array<Stmt> arr{fmakeif()};
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    TVM_FFI_ICHECK(arr[0].as<IfThenElseNode>()->else_case.as<EvaluateNode>()->value.same_as(x));
    // mutate but no content change.
    auto arr2 = arr;
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    TVM_FFI_ICHECK(arr2.get() == arr.get());
  }

  {
    auto body =
        Evaluate(Call(DataType::Int(32), builtin::call_extern(), {StringImm("xyz"), x + 1}));
    auto res = v(std::move(body));
    TVM_FFI_ICHECK(res.as<EvaluateNode>()->value.as<CallNode>()->args[1].same_as(x));
  }
  {
    Stmt body = fmakealloc();
    Stmt body2 = Evaluate(1);
    auto* ref2 = body2.get();
    auto* bufptr = body.as<AllocBufferNode>()->buffer.get();
    // construct a recursive SeqStmt.
    body = SeqStmt({body, body2});
    body = SeqStmt({body, body2});
    body = v(std::move(body));
    // the seq get flattened
    TVM_FFI_ICHECK(body.as<SeqStmtNode>()->size() == 3);
    // buffer is now mutated (shape x+1 -> x via VisitBufferDef)
    TVM_FFI_ICHECK(body.as<SeqStmtNode>()->seq[0].as<AllocBufferNode>()->buffer.get() != bufptr);
    TVM_FFI_ICHECK(body.as<SeqStmtNode>()->seq[1].get() == ref2);
  }

  {
    // Cannot cow because of bref
    Stmt body = fmakealloc();
    Stmt body2 = Evaluate(1);
    // construct a recursive SeqStmt.
    body = SeqStmt({body, body2});
    auto bref = body;
    body = SeqStmt({body, body2});
    body = v(std::move(body));
    // the seq get flattened
    TVM_FFI_ICHECK(body.as<SeqStmtNode>()->size() == 3);
    // buffer is mutated (shape x+1 -> x via VisitBufferDef)
    TVM_FFI_ICHECK(body.as<SeqStmtNode>()->seq[0].as<AllocBufferNode>() != nullptr);
    // bref still holds the old SeqStmt (not shared with new one due to copy)
    TVM_FFI_ICHECK(!bref.same_as(body));
  }

  {
    // tests for block and block_realize
    // AllocBuffer and DeclBuffer are flat (no body), placed as siblings in SeqStmt
    Stmt eval_body = Evaluate(x + 1);
    Buffer buffer = decl_buffer({16});
    Stmt decl = DeclBuffer(buffer);
    Stmt alloc = fmakealloc();
    // body is: DeclBuffer, AllocBuffer, Evaluate
    Stmt body = SeqStmt({decl, alloc, eval_body});
    BufferRegion buffer_region(buffer, {Range::FromMinExtent(x + 1, 1)});
    MatchBufferRegion match_buffer_region(decl_buffer({1}), buffer_region);
    // construct block and block_realize
    SBlock block = SBlock({}, {buffer_region}, {buffer_region}, "block", body, body, {},
                          {match_buffer_region});
    Stmt block_realize = SBlockRealize({}, const_true(), block);
    body = v(std::move(block_realize));
    // the body should be changed
    SBlock new_block = body.as<SBlockRealizeNode>()->block;
    // body is a SeqStmt; the Evaluate(x+1) -> Evaluate(x)
    auto* seq = new_block->body.as<SeqStmtNode>();
    TVM_FFI_ICHECK(seq != nullptr);
    TVM_FFI_ICHECK(seq->seq[2].as<EvaluateNode>()->value.same_as(x));
    auto* init_seq = new_block->init.value().as<SeqStmtNode>();
    TVM_FFI_ICHECK(init_seq != nullptr);
    TVM_FFI_ICHECK(init_seq->seq[2].as<EvaluateNode>()->value.same_as(x));
    // buffer region min is mutated: x+1 -> x
    TVM_FFI_ICHECK(new_block->reads[0]->region[0]->min.same_as(x));
    TVM_FFI_ICHECK(new_block->writes[0]->region[0]->min.same_as(x));
    TVM_FFI_ICHECK(new_block->match_buffers[0]->source->region[0]->min.same_as(x));
  }
}

TEST(IRF, Substitute) {
  using namespace tvm;
  using namespace tvm::tirx;
  DataType dtype = DataType::Float(32);
  Var x("x", PointerType(PrimType(dtype), ""));
  Var n("n", DataType::Int(32));

  auto fmakebuffer = [&]() {
    return Buffer{/*data=*/x,
                  /*dtype=*/DataType::Float(32),
                  /*shape=*/{n},
                  /*strides=*/{},
                  /*elem_offset=*/NullValue<PrimExpr>(),
                  /*name=*/"buf",
                  /*data_alignment=*/1,
                  /*offset_factor=*/1,
                  /*buffer_type=*/BufferType::kDefault};
  };

  {
    // test substitute buffer data var and shape var via DeclBuffer
    Var y = x.copy_with_suffix("subst");
    Var m("m", DataType::Int(32));
    Buffer buffer = fmakebuffer();
    Stmt store = BufferStore(buffer, FloatImm(dtype, 0), {IntImm(DataType::Int(32), 0)});
    Stmt decl = SeqStmt({DeclBuffer(buffer), store});
    auto f_subst = [&](const Var& var) -> ffi::Optional<PrimExpr> {
      if (var.same_as(x)) return y;
      if (var.same_as(n)) return m;
      return std::nullopt;
    };
    Stmt new_decl = Substitute(decl, f_subst);
    auto* seq_node = new_decl.as<SeqStmtNode>();
    TVM_FFI_ICHECK(seq_node != nullptr);
    auto* decl_node = seq_node->seq[0].as<DeclBufferNode>();
    TVM_FFI_ICHECK(decl_node != nullptr);
    TVM_FFI_ICHECK(decl_node->buffer->data.same_as(y));
    TVM_FFI_ICHECK(decl_node->buffer->shape[0].same_as(m));
  }

  {
    // test identity substitution on expression
    Buffer buffer = fmakebuffer();
    PrimExpr expr = BufferLoad(buffer, {IntImm(DataType::Int(32), 0)});
    auto f_subst = [&](const Var& var) -> ffi::Optional<PrimExpr> { return var; };
    PrimExpr new_expr = Substitute(expr, f_subst);
    // the expression is not changed
    TVM_FFI_ICHECK(new_expr.same_as(expr));
  }
}
