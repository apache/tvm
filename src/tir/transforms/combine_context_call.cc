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
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/registry.h>


#include <unordered_map>

namespace tvm {
namespace tir {

// Calculate the statistics of packed function.
// These information are needed during codegen.
class ContextCallCombiner final : public StmtExprMutator {
 public:
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->is_intrinsic(intrinsic::tvm_thread_context)) {
      CHECK_EQ(op->args.size(), 1U);
      PrimExpr ctx = op->args[0];
      auto it  = ctx_map_.find(ctx);
      if (it != ctx_map_.end()) {
        return it->second;
      } else {
        CHECK(ctx.dtype().is_handle());
        std::string name;
        if (const CallNode* call = ctx.as<CallNode>()) {
          name = call->name + "_cache";
        } else {
          name = "ctx_cache_";
        }
        Var ctx_var(name, ctx.dtype());
        ctx_map_[ctx] = ctx_var;
        return std::move(ctx_var);
      }
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent ||
        op->attr_key == attr::coproc_uop_scope) {
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
    if (op->for_type == ForType::Parallel) {
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

  Stmt Combine(Stmt stmt) {
    return BuildContext(ctx_map_, this->VisitStmt(stmt));
  }

 private:
  static Stmt BuildContext(
      const std::unordered_map<PrimExpr, Var, StructuralHash, StructuralEqual>& cmap,
      Stmt body) {
    for (const auto& kv : cmap) {
      body = LetStmtNode::make(kv.second, kv.first, body);
    }
    return body;
  }
  // Map of comparison expression to variable
  std::unordered_map<PrimExpr, Var, StructuralHash, StructuralEqual> ctx_map_;
};


namespace transform {

Pass CombineContextCall() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = ContextCallCombiner().Combine(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.CombineContextCall", {});
}

TVM_REGISTER_GLOBAL("tir.transform.CombineContextCall")
.set_body_typed(CombineContextCall);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
