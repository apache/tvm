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
 *  Combine calls into context related function into one.
 *
 * \file combine_context_call.cc
 */
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

#include "ir_utils.h"

namespace tvm {
namespace tir {

// Calculate the statistics of packed function.
// These information are needed during codegen.
class ContextCallCombiner final : public StmtExprMutator {
 public:
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::tvm_thread_context())) {
      ICHECK_EQ(op->args.size(), 1U);
      PrimExpr ctx = op->args[0];
      auto it = ctx_map_.find(ctx);
      if (it != ctx_map_.end()) {
        return it->second;
      } else {
        ICHECK(ctx.dtype().is_handle());
        Var ctx_var("ctx_cache_", ctx.dtype());
        ctx_map_[ctx] = ctx_var;
        return std::move(ctx_var);
      }
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::coproc_uop_scope) {
      // Map of comparison expression to variable
      std::unordered_map<PrimExpr, Var, StructuralHash, StructuralEqual> temp;
      std::swap(temp, ctx_map_);
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      std::swap(temp, ctx_map_);
      return BuildContext(temp, stmt);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kParallel) {
      // Map of comparison expression to variable
      std::unordered_map<PrimExpr, Var, StructuralHash, StructuralEqual> temp;
      std::swap(temp, ctx_map_);
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      std::swap(temp, ctx_map_);
      return BuildContext(temp, stmt);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt Combine(Stmt stmt) { return BuildContext(ctx_map_, this->VisitStmt(stmt)); }

 private:
  static Stmt BuildContext(
      const std::unordered_map<PrimExpr, Var, StructuralHash, StructuralEqual>& cmap, Stmt body) {
    for (const auto& kv : cmap) {
      body = LetStmt(kv.second, kv.first, body);
    }
    return body;
  }
  // Map of comparison expression to variable
  std::unordered_map<PrimExpr, Var, StructuralHash, StructuralEqual> ctx_map_;
};

namespace transform {

Pass CombineContextCall() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    if (IsHostFunc(f).value_or(false)) {
      f.CopyOnWrite()->body = ContextCallCombiner().Combine(f->body);
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.CombineContextCall", {});
}

TVM_REGISTER_GLOBAL("tir.transform.CombineContextCall").set_body_typed(CombineContextCall);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
