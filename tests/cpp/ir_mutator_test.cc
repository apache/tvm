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
#include <tvm/ir_mutator.h>
#include <tvm/expr_operator.h>

namespace {
using namespace tvm;
using namespace tvm::ir;

// replace variable to constant
class IRVar2Const : public IRMutator {
 public:
  Var var;
  int int_val;
  Expr Mutate(Expr expr) final {
    static const FMutateExpr& f = IRVar2Const::vtable_expr();
    return (f.can_dispatch(expr) ?
            f(expr, expr, this) : IRMutator::Mutate(expr));
  }
  static FMutateExpr &vtable_expr();
};

// implement vtable
IRMutator::FMutateExpr &IRVar2Const::vtable_expr() {  // NOLINT(*)
  static FMutateExpr inst; return inst;
}

TVM_STATIC_IR_FUNCTOR(IRVar2Const, vtable_expr)
.set_dispatch<Variable>([](const Variable* op, const Expr &e, IRMutator* m) {
    IRVar2Const* vm = static_cast<IRVar2Const*>(m);
    if (e.same_as(vm->var)) {
      return Expr(IntImm::make(Int(32), vm->int_val));
    } else {
      return e;
    }
  });

}  // namespace

TEST(IRMutator, Basic) {
  using namespace tvm::ir;
  using namespace tvm;
  Var x("x"), y;
  auto z = x + y;
  IRVar2Const mu;
  mu.var = y;
  mu.int_val = 10;
  auto zz = mu.Mutate(z);
  std::ostringstream os;
  os << zz;
  CHECK(os.str() == "(x + 10)");
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
