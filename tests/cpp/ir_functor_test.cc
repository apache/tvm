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
#include <tvm/ir.h>
#include <tvm/expr_operator.h>
#include <tvm/node/functor.h>
#include <tvm/ir_functor_ext.h>

TEST(IRF, Basic) {
  using namespace tvm;
  using namespace tvm::ir;
  Var x("x");
  auto z = x + 1;

  NodeFunctor<int(const ObjectRef& n, int b)> f;
  f.set_dispatch<Variable>([](const ObjectRef& n, int b) {
      return b;
    });
  f.set_dispatch<Add>([](const ObjectRef& n, int b) {
      return b + 2;
    });
  CHECK_EQ(f(x, 2),  2);
  CHECK_EQ(f(z, 2),  4);
}

TEST(IRF, CountVar) {
  using namespace tvm;
  int n_var = 0;
  Var x("x"), y;

  auto z = x + 1 + y + y;
  ir::PostOrderVisit(z, [&n_var](const ObjectRef& n) {
      if (n.as<Variable>()) ++n_var;
    });
  CHECK_EQ(n_var, 2);
}


TEST(IRF, ExprTransform) {
  using namespace tvm;
  using namespace tvm::ir;
  Var x("x");
  auto z = x + 1;

  class MyExprFunctor
      : public ir::ExprFunctor<int(const Expr&, int)> {
   public:
    int VisitExpr_(const Variable* op, int b) final {
      return b;
    }
    int VisitExpr_(const IntImm* op, int b) final {
      return op->value;
    }
    int VisitExpr_(const Add* op, int b) final {
      return VisitExpr(op->a, b) + VisitExpr(op->b, b);
    }
  };
  MyExprFunctor f;
  CHECK_EQ(f(x, 2),  2);
  CHECK_EQ(f(z, 2),  3);
  try {
    f(z - 1, 2);
    LOG(FATAL) << "should fail";
  } catch(dmlc::Error) {
  }
}

TEST(IRF, ExprVisit) {
  using namespace tvm;
  using namespace tvm::ir;
  Var x("x");
  auto z = x + 1;

  class MyVisitor
      : public ir::ExprFunctor<void(const Expr&)>,
        public ir::StmtFunctor<void(const Stmt&)> {
   public:
    int count = 0;
    // implementation
    void VisitExpr_(const Variable* op) final {
      ++count;
    }
    void VisitExpr_(const IntImm* op) final {
    }
    void VisitExpr_(const Add* op) final {
      VisitExpr(op->a);
      VisitExpr(op->b);
    }
    void VisitStmt_(const Evaluate* op) final {
      VisitExpr(op->value);
    }
  };
  MyVisitor v;
  v.VisitStmt(Evaluate::make(z));
  CHECK_EQ(v.count, 1);
}


TEST(IRF, StmtVisitor) {
  using namespace tvm;
  using namespace tvm::ir;
  Var x("x");
  class MyVisitor
      : public StmtExprVisitor {
   public:
    int count = 0;
    // implementation
    void VisitExpr_(const Variable* op) final {
      ++count;
    }
  };
  MyVisitor v;
  auto fmaketest = [&]() {
    auto z = x + 1;
    Stmt body = Evaluate::make(z);
    Var buffer("b", DataType::Handle());
    return Allocate::make(buffer, DataType::Float(32), {z, z}, const_true(), body);
  };
  v(fmaketest());
  CHECK_EQ(v.count, 3);
}

TEST(IRF, StmtMutator) {
  using namespace tvm;
  using namespace tvm::ir;
  Var x("x");

  class MyVisitor
      : public ir::StmtMutator,
        public ir::ExprMutator {
   public:
    using StmtMutator::operator();
    using ExprMutator::operator();

   protected:
    // implementation
    Expr VisitExpr_(const Add* op) final {
      return op->a;
    }
    Expr VisitExpr(const Expr& expr) final {
      return ExprMutator::VisitExpr(expr);
    }
  };
  auto fmakealloc = [&]() {
    auto z = x + 1;
    Stmt body = Evaluate::make(z);
    Var buffer("b", DataType::Handle());
    return Allocate::make(buffer, DataType::Float(32), {1, z}, const_true(), body);
  };

  auto fmakeif = [&]() {
    auto z = x + 1;
    Stmt body = Evaluate::make(z);
    return IfThenElse::make(x < 0, Evaluate::make(0), body);
  };

  MyVisitor v;
  {
    auto body = fmakealloc();
    Stmt body2 = Evaluate::make(1);
    Stmt bref = body.as<Allocate>()->body;
    auto* extentptr = body.as<Allocate>()->extents.get();
    Array<Stmt> arr{std::move(body), body2, body2};
    auto* arrptr = arr.get();
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    CHECK(arr.get() == arrptr);
    // inplace update body
    CHECK(arr[0].as<Allocate>()->extents[1].same_as(x));
    CHECK(arr[0].as<Allocate>()->extents.get() == extentptr);
    // copy because there is additional refs
    CHECK(!arr[0].as<Allocate>()->body.same_as(bref));
    CHECK(arr[0].as<Allocate>()->body.as<Evaluate>()->value.same_as(x));
    CHECK(bref.as<Evaluate>()->value.as<Add>());
  }
  {
    Array<Stmt> arr{fmakealloc()};
    // mutate array get reference by another one, triiger copy.
    Array<Stmt> arr2 = arr;
    auto* arrptr = arr.get();
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    CHECK(arr.get() != arrptr);
    CHECK(arr[0].as<Allocate>()->extents[1].same_as(x));
    CHECK(!arr2[0].as<Allocate>()->extents[1].same_as(x));
    // mutate but no content change.
    arr2 = arr;
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    CHECK(arr2.get() == arr.get());
  }
  {
    Array<Stmt> arr{fmakeif()};
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    CHECK(arr[0].as<IfThenElse>()->else_case.as<Evaluate>()->value.same_as(x));
    // mutate but no content change.
    auto arr2 = arr;
    arr.MutateByApply([&](Stmt s) { return v(std::move(s)); });
    CHECK(arr2.get() == arr.get());
  }

  {
    auto body = Evaluate::make(Call::make(DataType::Int(32), "xyz", {x + 1}, Call::Extern));
    auto res = v(std::move(body));
    CHECK(res.as<Evaluate>()->value.as<Call>()->args[0].same_as(x));
  }
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
