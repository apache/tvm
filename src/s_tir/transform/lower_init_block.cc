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
 * Lower block init stmt into branch stmt
 * \file lower_reduction.cc
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../../tir/transform/ir_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tir;

class InitBlockLower : public StmtMutator {
 private:
  Stmt VisitStmt_(const SBlockNode* block) final {
    if (!block->init.defined()) {
      return StmtMutator::VisitStmt_(block);
    }
    Stmt init = DoLowering(block->init.value(), block->iter_vars);
    Stmt body = VisitStmt(block->body);
    auto n = CopyOnWrite(block);
    n->init = std::nullopt;
    n->body = SeqStmt::Flatten(init, body);
    return SBlock(n);
  }

  static Stmt DoLowering(const Stmt& init, const ffi::Array<IterVar>& iter_vars) {
    std::vector<PrimExpr> conditions;
    for (const IterVar& var : iter_vars) {
      if (var->iter_type == IterVarType::kCommReduce) {
        conditions.push_back(equal(var->var, var->dom->min));
      }
    }
    // Handle the case where there is no condition
    if (conditions.empty()) {
      return init;
    }
    // Concat the conditions with logical and (&&)
    PrimExpr cond = conditions[0];
    for (size_t i = 1; i < conditions.size(); ++i) {
      cond = logical_and(cond, conditions[i]);
    }
    return IfThenElse(cond, init);
  }
};

namespace transform {

Pass LowerInitBlock() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto fptr = f.CopyOnWrite();
    fptr->body = InitBlockLower()(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "s_tir.LowerInitBlock", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.LowerInitBlock", LowerInitBlock);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
