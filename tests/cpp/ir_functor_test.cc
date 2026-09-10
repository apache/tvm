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
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/module.h>
#include <tvm/ir/node_functor.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <initializer_list>

TEST(IRF, Basic) {
  using namespace tvm;
  using namespace tvm::tirx;
  PrimVar x("x");
  auto z = x + 1;

  NodeFunctor<int(const ffi::ObjectRef& n, int b)> f;
  f.set_dispatch<VarNode>([](const ffi::ObjectRef& n, int b) { return b; });
  f.set_dispatch<prim::AddNode>([](const ffi::ObjectRef& n, int b) { return b + 2; });
  TVM_FFI_ICHECK_EQ(f(x, 2), 2);
  TVM_FFI_ICHECK_EQ(f(z, 2), 4);
}

TEST(IRF, CountVar) {
  using namespace tvm;
  using namespace tvm::tirx;
  int n_var = 0;
  PrimVar x("x"), y("y");

  auto z = x + 1 + y + y;
  tirx::PostOrderVisit(z, [&n_var](const ffi::ObjectRef& n) {
    if (n.as<VarNode>()) ++n_var;
  });
  TVM_FFI_ICHECK_EQ(n_var, 2);
}

TEST(IRF, PreOrderVisit) {
  using namespace tvm;
  using namespace tvm::tirx;
  Stmt init =
      IfThenElse(IntImm::Bool(true), Evaluate(IntImm::Int32(0)), Evaluate(IntImm::Int32(0)));
  Stmt body = Evaluate(IntImm::Int32(1));
  SBlock block(/*iter_vars=*/{}, /*reads=*/{},
               /*writes=*/{}, /*name_hint=*/"block", /*body=*/body,
               /*init=*/init);
  bool init_visited = false;
  bool stopped_at_if = true;
  bool body_visited = false;
  PreOrderVisit(block, [&](const ffi::ObjectRef& n) -> bool {
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
  PrimVar x("x");
  auto z = x + 1;

  class MyExprFunctor : public tirx::ExprFunctor<int(const Expr&, int)> {
   public:
    int VisitExpr_(const VarNode* op, int b) final { return b; }
    int VisitExpr_(const IntImmNode* op, int b) final { return op->value; }
    int VisitExpr_(const prim::AddNode* op, int b) final {
      return VisitExpr(op->a, b) + VisitExpr(op->b, b);
    }
  };
  MyExprFunctor f;
  TVM_FFI_ICHECK_EQ(f(x, 2), 2);
  TVM_FFI_ICHECK_EQ(f(z, 2), 3);
  try {
    f(z - 1, 2);
    TVM_FFI_THROW(InternalError) << "should fail";
  } catch (tvm::ffi::Error&) {
  }
}

TEST(IRF, ExprVisit) {
  using namespace tvm;
  using namespace tvm::tirx;
  PrimVar x("x");
  auto z = x + 1;

  class MyVisitor : public tirx::ExprFunctor<void(const Expr&)>,
                    public tirx::StmtFunctor<void(const Stmt&)> {
   public:
    int count = 0;
    // implementation
    void VisitExpr_(const VarNode* op) final { ++count; }
    void VisitExpr_(const IntImmNode* op) final {}
    void VisitExpr_(const prim::AddNode* op) final {
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
  PrimVar x("x");
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
    PrimType dtype = PrimType::Float(32);
    BufferVar buf("b", BufferType("global", dtype, {z, z}, {}, PrimExpr(), 0, 0));
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
    PrimType dtype = PrimType::Float(32);
    tirx::Var buf_var("b", PointerType(dtype));
    BufferVar buffer = decl_buffer({16});
    body = SeqStmt({DeclBuffer(buffer, buf_var), std::move(body)});
    BufferRegion buffer_region(buffer, {Range::FromMinExtent(x + 1, 1)});
    MatchBufferRegion match_buffer_region(decl_buffer({1}), buffer_region);

    // construct block and block_realize
    SBlock block = SBlock({}, {buffer_region}, {buffer_region}, "block", body, body, {},
                          {match_buffer_region});
    Stmt block_realize = SBlockRealize({}, IntImm::Bool(true), block);

    v.count = 0;
    v(block_realize);
    // x visited in: reads range (1), writes range (1), match_buffers range (1),
    // init DeclBuffer(0) + AllocBuffer shape(2) + Evaluate(1) = 3,
    // body DeclBuffer(0) + AllocBuffer shape(2) + Evaluate(1) = 3.
    // The block's read/write BufferTypes each visit their dependent shape once,
    // in addition to the ranges, match buffer, init, and body.
    // Total: 2 + 2 + 1 + 3 + 3 = 11.
    TVM_FFI_ICHECK_EQ(v.count, 11);
  }
}

TEST(IRF, StmtMutator) {
  using namespace tvm;
  using namespace tvm::tirx;
  PrimVar x("x");

  class MyVisitor : public tirx::StmtMutator, public tirx::ExprMutator {
   public:
    using StmtMutator::operator();
    using ExprMutator::operator();

   protected:
    // implementation
    Expr VisitExpr_(const prim::AddNode* op) final { return op->a; }
    Stmt VisitStmt_(const SeqStmtNode* op) final { return StmtMutator::VisitSeqStmt_(op, true); }
    Expr VisitExpr(const Expr& expr) final { return ExprMutator::VisitExpr(expr); }
  };
  auto fmakealloc = [&]() {
    auto z = x + 1;
    PrimType dtype = PrimType::Float(32);
    BufferVar buf("b", BufferType("global", dtype, {1, z}, {}, PrimExpr(), 0, 0));
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
        Evaluate(Call(PrimType::Int(32), builtin::call_extern(), {prim::StringImm("xyz"), x + 1}));
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
    BufferVar buffer = decl_buffer({16});
    tirx::Var buffer_data("buffer_data", buffer.DataPointerType());
    Stmt decl = DeclBuffer(buffer, buffer_data);
    Stmt alloc = fmakealloc();
    // body is: DeclBuffer, AllocBuffer, Evaluate
    Stmt body = SeqStmt({decl, alloc, eval_body});
    BufferRegion buffer_region(buffer, {Range::FromMinExtent(x + 1, 1)});
    MatchBufferRegion match_buffer_region(decl_buffer({1}), buffer_region);
    // construct block and block_realize
    SBlock block = SBlock({}, {buffer_region}, {buffer_region}, "block", body, body, {},
                          {match_buffer_region});
    Stmt block_realize = SBlockRealize({}, IntImm::Bool(true), block);
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

TEST(IRF, StructuralMapSplicesMappedSeqStmtChild) {
  using namespace tvm;
  using namespace tvm::tirx;

  auto make_input = []() -> Stmt {
    return SeqStmt(
        {Evaluate(IntImm::Int32(5)), Evaluate(IntImm::Int32(1)), Evaluate(IntImm::Int32(4))});
  };
  auto expand_one = [](const Evaluate& evaluate) -> Stmt {
    const auto* value = evaluate->value.as<IntImmNode>();
    if (value != nullptr && value->value == 1) {
      return SeqStmt({Evaluate(IntImm::Int32(2)), Evaluate(IntImm::Int32(3))});
    }
    return evaluate;
  };
  auto check_values = [](const Stmt& stmt, std::initializer_list<int64_t> expected) {
    const auto* seq = stmt.as<SeqStmtNode>();
    ASSERT_NE(seq, nullptr);
    ASSERT_EQ(seq->seq.size(), expected.size());
    size_t i = 0;
    for (int64_t expected_value : expected) {
      const auto* evaluate = seq->seq[i].as<EvaluateNode>();
      ASSERT_NE(evaluate, nullptr);
      const auto* value = evaluate->value.as<IntImmNode>();
      ASSERT_NE(value, nullptr);
      EXPECT_EQ(value->value, expected_value);
      ++i;
    }
  };
  {
    Stmt input = make_input();
    Stmt shared = input;
    Stmt mapped =
        ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(input, expand_one).as_or_throw<Stmt>();
    EXPECT_FALSE(mapped.same_as(input));
    EXPECT_EQ(shared.as<SeqStmtNode>()->seq.size(), 3);
    check_values(mapped, {5, 2, 3, 4});
  }

  {
    Stmt input = make_input();
    const auto* original = input.get();
    const auto* original_array = input.as<SeqStmtNode>()->seq.GetArrayObj();
    Stmt mapped =
        ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(input), expand_one)
            .as_or_throw<Stmt>();
    EXPECT_EQ(mapped.get(), original);
    EXPECT_NE(mapped.as<SeqStmtNode>()->seq.GetArrayObj(), original_array);
    check_values(mapped, {5, 2, 3, 4});
  }

  auto make_boundary_input = []() -> Stmt {
    return SeqStmt({Evaluate(IntImm::Int32(1)), Evaluate(IntImm::Int32(2)),
                    Evaluate(IntImm::Int32(3)), Evaluate(IntImm::Int32(4))});
  };
  auto check_differential = [&](const auto& transform, std::initializer_list<int64_t> expected,
                                bool expect_array_reuse) {
    Stmt ordinary_input = make_boundary_input();
    Stmt ordinary = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(ordinary_input, transform)
                        .template as_or_throw<Stmt>();

    Stmt inplace_input = make_boundary_input();
    const auto* original_root = inplace_input.get();
    const auto* original_array = inplace_input.as<SeqStmtNode>()->seq.GetArrayObj();
    Stmt inplace =
        ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(inplace_input), transform)
            .template as_or_throw<Stmt>();

    EXPECT_EQ(inplace.get(), original_root);
    if (expect_array_reuse) {
      EXPECT_EQ(inplace.as<SeqStmtNode>()->seq.GetArrayObj(), original_array);
    } else {
      EXPECT_NE(inplace.as<SeqStmtNode>()->seq.GetArrayObj(), original_array);
    }
    EXPECT_TRUE(ffi::StructuralEqual()(ordinary, inplace));
    check_values(inplace, expected);
  };

  auto shrink_first_grow_last = [](const Evaluate& evaluate) -> Stmt {
    const auto* value = evaluate->value.as<IntImmNode>();
    if (value != nullptr && value->value == 1) {
      return Evaluate(0);
    }
    if (value != nullptr && value->value == 4) {
      return SeqStmt({Evaluate(IntImm::Int32(30)), Evaluate(IntImm::Int32(31))});
    }
    return evaluate;
  };
  check_differential(shrink_first_grow_last, {2, 3, 30, 31}, true);

  auto empty_first_grow_last = [](const Evaluate& evaluate) -> Stmt {
    const auto* value = evaluate->value.as<IntImmNode>();
    if (value != nullptr && value->value == 1) {
      auto empty = ffi::make_object<SeqStmtNode>();
      empty->seq = {};
      return Stmt(std::move(empty));
    }
    if (value != nullptr && value->value == 4) {
      return SeqStmt({Evaluate(IntImm::Int32(30)), Evaluate(IntImm::Int32(31))});
    }
    return evaluate;
  };
  check_differential(empty_first_grow_last, {2, 3, 30, 31}, true);

  int overflow_callback_count = 0;
  auto grow_first_shrink_last = [&overflow_callback_count](const Evaluate& evaluate) -> Stmt {
    ++overflow_callback_count;
    const auto* value = evaluate->value.as<IntImmNode>();
    if (value != nullptr && value->value == 1) {
      return SeqStmt({Evaluate(IntImm::Int32(10)), Evaluate(IntImm::Int32(11))});
    }
    if (value != nullptr && value->value == 4) {
      return Evaluate(0);
    }
    return evaluate;
  };
  check_differential(grow_first_shrink_last, {10, 11, 2, 3}, true);
  EXPECT_EQ(overflow_callback_count, 8);

  auto grow_last_over_capacity = [](const Evaluate& evaluate) -> Stmt {
    const auto* value = evaluate->value.as<IntImmNode>();
    if (value != nullptr && value->value == 4) {
      return SeqStmt({Evaluate(IntImm::Int32(40)), Evaluate(IntImm::Int32(41))});
    }
    return evaluate;
  };
  check_differential(grow_last_over_capacity, {1, 2, 3, 40, 41}, false);

  auto replace_middle_with_one = [](const Evaluate& evaluate) -> Stmt {
    const auto* value = evaluate->value.as<IntImmNode>();
    if (value != nullptr && value->value == 2) {
      return Evaluate(IntImm::Int32(10));
    }
    return evaluate;
  };
  check_differential(replace_middle_with_one, {1, 10, 3, 4}, true);

  auto remove_first = [](const Evaluate& evaluate) -> Stmt {
    const auto* value = evaluate->value.as<IntImmNode>();
    return value != nullptr && value->value == 1 ? Evaluate(0) : Stmt(evaluate);
  };
  check_differential(remove_first, {2, 3, 4}, true);

  auto remove_all = [](const Evaluate&) -> Stmt { return Evaluate(0); };
  Stmt ordinary_input = make_boundary_input();
  Stmt ordinary =
      ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(ordinary_input, remove_all)
          .as_or_throw<Stmt>();
  Stmt inplace_input = make_boundary_input();
  Stmt inplace =
      ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(inplace_input), remove_all)
          .as_or_throw<Stmt>();
  EXPECT_TRUE(ffi::StructuralEqual()(ordinary, inplace));
  for (const Stmt& result : {ordinary, inplace}) {
    const auto* evaluate = result.as<EvaluateNode>();
    ASSERT_NE(evaluate, nullptr);
    const auto* value = evaluate->value.as<IntImmNode>();
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(value->value, 0);
  }

  auto keep_last = [](const Evaluate& evaluate) -> Stmt {
    const auto* value = evaluate->value.as<IntImmNode>();
    return value != nullptr && value->value == 4 ? Stmt(evaluate) : Stmt(Evaluate(0));
  };
  ordinary_input = make_boundary_input();
  ordinary = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(ordinary_input, keep_last)
                 .as_or_throw<Stmt>();
  inplace_input = make_boundary_input();
  inplace = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(inplace_input), keep_last)
                .as_or_throw<Stmt>();
  EXPECT_TRUE(ffi::StructuralEqual()(ordinary, inplace));
  for (const Stmt& result : {ordinary, inplace}) {
    const auto* evaluate = result.as<EvaluateNode>();
    ASSERT_NE(evaluate, nullptr);
    const auto* value = evaluate->value.as<IntImmNode>();
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(value->value, 4);
  }
}

TEST(IRF, StructuralMapPreservesSeqStmtElementUniqueness) {
  using namespace tvm;
  using namespace tvm::tirx;

  auto replace_one = [](const IntImm& value) -> PrimExpr {
    return value->value == 1 ? IntImm::Int32(2) : PrimExpr(value);
  };

  {
    Stmt input = SeqStmt({Evaluate(IntImm::Int32(1)), Evaluate(IntImm::Int32(3))});
    const auto* original_root = input.get();
    const auto* original_first = input.as<SeqStmtNode>()->seq[0].as<EvaluateNode>();
    Stmt mapped =
        ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(input), replace_one)
            .as_or_throw<Stmt>();

    const auto* mapped_seq = mapped.as<SeqStmtNode>();
    ASSERT_NE(mapped_seq, nullptr);
    EXPECT_EQ(mapped.get(), original_root);
    EXPECT_EQ(mapped_seq->seq[0].get(), original_first);
    EXPECT_EQ(mapped_seq->seq[0].as<EvaluateNode>()->value.as<IntImmNode>()->value, 2);
  }

  {
    ffi::Array<Stmt> shared_seq = {Evaluate(IntImm::Int32(1)), Evaluate(IntImm::Int32(3))};
    const auto* shared_first = shared_seq[0].as<EvaluateNode>();
    Stmt input = SeqStmt(shared_seq);
    Stmt mapped =
        ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(input), replace_one)
            .as_or_throw<Stmt>();

    const auto* mapped_seq = mapped.as<SeqStmtNode>();
    ASSERT_NE(mapped_seq, nullptr);
    EXPECT_FALSE(mapped_seq->seq.same_as(shared_seq));
    EXPECT_EQ(shared_seq[0].get(), shared_first);
    EXPECT_EQ(shared_seq[0].as<EvaluateNode>()->value.as<IntImmNode>()->value, 1);
    EXPECT_EQ(mapped_seq->seq[0].as<EvaluateNode>()->value.as<IntImmNode>()->value, 2);

    Stmt unchanged_input = SeqStmt(shared_seq);
    const auto* unchanged_root = unchanged_input.get();
    auto no_float_match = [](const FloatImm& value) -> PrimExpr { return value; };
    Stmt unchanged =
        ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(std::move(unchanged_input), no_float_match)
            .as_or_throw<Stmt>();
    EXPECT_EQ(unchanged.get(), unchanged_root);
    EXPECT_TRUE(unchanged.as<SeqStmtNode>()->seq.same_as(shared_seq));
  }
}

TEST(IRF, StructuralHooksPreserveScopeIdDefRegions) {
  using namespace tvm;
  using namespace tvm::tirx;

  auto make_input = []() -> Stmt {
    return ScopeIdDefStmt(ScopeIdDef({PrimVar("binder")}, ffi::Array<PrimExpr>{PrimVar("extent")},
                                     ScopeBinding::kCtaThread,
                                     ffi::Array<PrimExpr>{PrimVar("preferred")}));
  };
  auto check_kinds = [](int binder, int extent, int preferred) {
    EXPECT_EQ(binder, kTVMFFIDefRegionKindSimple);
    EXPECT_EQ(extent, kTVMFFIDefRegionKindNone);
    EXPECT_EQ(preferred, kTVMFFIDefRegionKindNone);
  };

  {
    int binder = -1;
    int extent = -1;
    int preferred = -1;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        make_input(),
        [&](const Var& var, TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::WalkResult> {
          if (var->name == "binder") binder = kind;
          if (var->name == "extent") extent = kind;
          if (var->name == "preferred") preferred = kind;
          return ffi::WalkResult::Advance();
        });
    check_kinds(binder, extent, preferred);
  }

  auto check_map = [&](Stmt input) {
    int binder = -1;
    int extent = -1;
    int preferred = -1;
    ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
        std::move(input), [&](const Var& var, TVMFFIDefRegionKind kind) -> Var {
          if (var->name == "binder") binder = kind;
          if (var->name == "extent") extent = kind;
          if (var->name == "preferred") preferred = kind;
          return var;
        });
    check_kinds(binder, extent, preferred);
  };

  {
    Stmt input = make_input();
    Stmt shared = input;
    check_map(input);
  }
  check_map(make_input());
}

TEST(IRF, Substitute) {
  using namespace tvm;
  using namespace tvm::tirx;
  PrimType dtype = PrimType::Float(32);
  tirx::Var x("x", PointerType(dtype, ""));
  PrimVar n("n", PrimType::Int(32));

  auto fmakebuffer = [&]() {
    return BufferVar("buf", BufferType(/*storage_scope=*/"global",
                                       /*dtype=*/PrimType::Float(32),
                                       /*shape=*/{n},
                                       /*strides=*/{},
                                       /*elem_offset=*/PrimExpr(),
                                       /*data_alignment=*/1,
                                       /*offset_factor=*/1));
  };

  {
    // Test substitution of an explicit DeclBuffer source and a dependent
    // BufferType shape.  Changing the type creates one fresh Var identity
    // that is shared by the declaration and every use.
    tirx::Var y = x.CopyWithSuffix("subst");
    PrimVar m("m", PrimType::Int(32));
    BufferVar buffer = fmakebuffer();
    Stmt store = BufferStore(buffer, FloatImm(dtype, 0), {IntImm::Int32(0)});
    Stmt decl = SeqStmt({DeclBuffer(buffer, x), store});
    auto f_subst = [&](const tirx::Var& var) -> ffi::Optional<Expr> {
      if (var.same_as(x)) return Expr(y);
      if (var.same_as(n)) return Expr(m);
      return std::nullopt;
    };
    Stmt new_decl = Substitute(decl, f_subst);
    auto* seq_node = new_decl.as<SeqStmtNode>();
    TVM_FFI_ICHECK(seq_node != nullptr);
    auto* decl_node = seq_node->seq[0].as<DeclBufferNode>();
    TVM_FFI_ICHECK(decl_node != nullptr);
    TVM_FFI_ICHECK(decl_node->data.same_as(y));
    TVM_FFI_ICHECK(decl_node->buffer->shape[0].same_as(m));
    TVM_FFI_ICHECK(!decl_node->buffer.same_as(buffer));
    auto* store_node = seq_node->seq[1].as<BufferStoreNode>();
    TVM_FFI_ICHECK(store_node != nullptr);
    TVM_FFI_ICHECK(store_node->buffer.same_as(decl_node->buffer));
  }

  {
    // test identity substitution on expression
    BufferVar buffer = fmakebuffer();
    PrimExpr expr = BufferLoad(buffer, {IntImm::Int32(0)});
    auto f_subst = [&](const tirx::Var& var) -> ffi::Optional<Expr> { return Expr(var); };
    PrimExpr new_expr = Substitute(expr, f_subst);
    // the expression is not changed
    TVM_FFI_ICHECK(new_expr.same_as(expr));
  }
}

TEST(IRF, SubstituteWithDataTypeLegalizationPreservesShiftAmounts) {
  using namespace tvm;
  using namespace tvm::tirx;

  PrimVar x("x", PrimType::Int(64));
  PrimVar y("y", PrimType::Int(32));
  auto f_subst = [&](const tirx::Var& var) -> ffi::Optional<PrimExpr> {
    if (var.same_as(x)) return PrimExpr(y);
    return std::nullopt;
  };

  PrimExpr shift_amount = IntImm::Int64(40);
  PrimExpr widened_y = cast(PrimType::Int(64), y);
  PrimExpr actual_left = SubstituteWithDataTypeLegalization(x << shift_amount, f_subst);
  PrimExpr actual_right = SubstituteWithDataTypeLegalization(x >> shift_amount, f_subst);

  ffi::StructuralEqual structural_equal;
  EXPECT_TRUE(structural_equal(actual_left, widened_y << shift_amount));
  EXPECT_TRUE(structural_equal(actual_right, widened_y >> shift_amount));
}
