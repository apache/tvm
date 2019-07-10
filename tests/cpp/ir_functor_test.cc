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
#include <tvm/node/ir_functor.h>
#include <tvm/ir_functor_ext.h>

TEST(IRF, Basic) {
  using namespace tvm;
  using namespace tvm::ir;
  Var x("x");
  auto z = x + 1;

  IRFunctor<int(const NodeRef& n, int b)> f;
  LOG(INFO) << "x";
  f.set_dispatch<Variable>([](const Variable* n, int b) {
      return b;
    });
  f.set_dispatch<Add>([](const Add* n, int b) {
      return b + 2;
    });
  CHECK_EQ(f(x, 2),  2);
  CHECK_EQ(f(z, 2),  4);
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

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
