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

/*!
 *  Remove intrinsic calls when possible.
 * \file remove_intrin.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/api_registry.h>
#include <tvm/expr_operator.h>

namespace tvm {
namespace ir {

class IntrinRemover : public IRMutator {
 public:
  Expr Mutate_(const Call* op, const Expr& e) final {
    if (op->is_intrinsic(intrinsic::tvm_assert_bound)) {
      return op->args[0];  // simply return the value
    }
    return IRMutator::Mutate_(op, e);
  }
};

Stmt RemoveIntrinStmt(Stmt stmt) {
  return IntrinRemover().Mutate(stmt);
}

Expr RemoveIntrinExpr(Expr expr) {
  return IntrinRemover().Mutate(expr);
}

LoweredFunc RemoveIntrin(LoweredFunc f) {
  auto n = make_node<LoweredFuncNode>(*f.operator->());
  n->body = RemoveIntrinStmt(n->body);
  return LoweredFunc(n);
}

// Register the api only for test purposes
TVM_REGISTER_API("ir_pass._RemoveIntrinStmt")
.set_body_typed(RemoveIntrinStmt);

TVM_REGISTER_API("ir_pass._RemoveIntrinExpr")
.set_body_typed(RemoveIntrinExpr);

}  // namespace ir
}  // namespace tvm

