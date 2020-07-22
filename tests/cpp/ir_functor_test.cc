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
#include <tvm/node/functor.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
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
  CHECK_EQ(f(x, 2), 2);
  CHECK_EQ(f(z, 2), 4);
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
  CHECK_EQ(n_var, 2);
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
  CHECK_EQ(f(x, 2), 2);
  CHECK_EQ(f(z, 2), 3);
  try {
    f(z - 1, 2);
    LOG(FATAL) << "should fail";
  } catch (dmlc::Error) {
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
  CHECK_EQ(v.count, 1);
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
    Var buffer("b", DataType::Handle());
    return Allocate(buffer, DataType::Float(32), {z, z}, const_true(), body);
  };
  v(fmaketest());
  CHECK_EQ(v.count, 3);
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
    Var buffer("b", DataType::Handle());
    return Allocate(buffer, DataType::Float(32), {1, z}, const_true(), body);
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
    CHECK(arr.get() == arrptr);
    // inplace update body
    CHECK(arr[0].as<AllocateNode>()->extents[1].same_as(x));
    CHECK(arr[0].as<AllocateNode>()->extents.get() == extentptr);
    // copy because there is additional refs
    CHECK(!arr[0].as<AllocateNode>()->body.same_as(bref));
    CHECK(arr[0].as<AllocateNode>()->body.as<EvaluateNode>()->value.same_as(x));
    CHECK(bref.as<EvaluateNode>()->value.as<AddNode>());
  }
  {
    Array<Stmt> arr{fmakealloc()};
    // mutate array get reference by another one, triiger copy.
    Array<Stmt> arr2 = arr;
    auto* arrptr = arr.get();
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    CHECK(arr.get() != arrptr);
    CHECK(arr[0].as<AllocateNode>()->extents[1].same_as(x));
    CHECK(!arr2[0].as<AllocateNode>()->extents[1].same_as(x));
    // mutate but no content change.
    arr2 = arr;
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    CHECK(arr2.get() == arr.get());
  }
  {
    Array<Stmt> arr{fmakeif()};
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    CHECK(arr[0].as<IfThenElseNode>()->else_case.as<EvaluateNode>()->value.same_as(x));
    // mutate but no content change.
    auto arr2 = arr;
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    CHECK(arr2.get() == arr.get());
  }

  {
    auto body =
        Evaluate(Call(DataType::Int(32), builtin::call_extern(), {StringImm("xyz"), x + 1}));
    auto res = v(std::move(body));
    CHECK(res.as<EvaluateNode>()->value.as<CallNode>()->args[1].same_as(x));
  }
  {
    Stmt body = fmakealloc();
    Stmt body2 = Evaluate(1);
    auto* ref2 = body2.get();
    auto* extentptr = body.as<AllocateNode>()->extents.get();
    // construct a recursive SeqStmt.
    body = SeqStmt({body});
    body = SeqStmt({body, body2});
    body = SeqStmt({body, body2});
    body = v(std::move(body));
    // the seq get flattened
    CHECK(body.as<SeqStmtNode>()->size() == 3);
    CHECK(body.as<SeqStmtNode>()->seq[0].as<AllocateNode>()->extents.get() == extentptr);
    CHECK(body.as<SeqStmtNode>()->seq[1].get() == ref2);
  }

  {
    // Cannot cow because of bref
    Stmt body = fmakealloc();
    Stmt body2 = Evaluate(1);
    auto* extentptr = body.as<AllocateNode>()->extents.get();
    // construct a recursive SeqStmt.
    body = SeqStmt({body});
    auto bref = body;
    body = SeqStmt({body, body2});
    body = v(std::move(body));
    // the seq get flattened
    CHECK(body.as<SeqStmtNode>()->seq[0].as<AllocateNode>()->extents.get() != extentptr);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
