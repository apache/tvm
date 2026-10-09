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
 * \file hoist_if.cc
 * \brief Hoist loop-invariant If statements.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "ir_utils.h"

namespace tvm {
namespace tirx {

class IfHoister : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input, InplaceMode inplace_mode) final {
    // Expressions are inspected only as whole If predicates, never split or hoisted.
    if (input.as<ExprNode>()) return ffi::Unchanged();
    if (input.as<StmtNode>()) {
      const auto* seq = input.as<SeqStmtNode>();
      if (!input.as<ForNode>() && !input.as<IfNode>() && !(seq && seq->seq.size() == 1)) {
        // Bindings and sibling statements stay ordered.  Regions and unknown
        // statements have no code-motion contract; process their loops locally.
        auto outer_loops = std::move(loops_);
        loops_.clear();
        auto result = StmtExprMutator::Mutate(input, inplace_mode);
        loops_ = std::move(outer_loops);
        return result;
      }
    }
    return StmtExprMutator::Mutate(input, inplace_mode);
  }

 private:
  struct Loop {
    Var var;
    std::vector<std::pair<PrimExpr, bool>> conditions;
  };

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    Loop loop{op->loop_var, {}};
    loops_.push_back(&loop);
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    loops_.pop_back();
    for (auto it = loop.conditions.rbegin(); it != loop.conditions.rend(); ++it) {
      stmt = it->second ? If(it->first, stmt, SeqStmt(stmt)) : If(it->first, stmt);
    }
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const IfNode* op, InplaceMode inplace_mode) final {
    if (!loops_.empty() && SideEffect(op->condition) <= CallEffectKind::kPure) {
      auto vars = UndefinedVars(op->condition);
      size_t destination = loops_.size();
      while (destination > 0) {
        const Var& loop_var = loops_[destination - 1]->var;
        if (std::any_of(vars.begin(), vars.end(),
                        [&](const Var& var) { return var.same_as(loop_var); })) {
          break;
        }
        --destination;
      }
      if (destination < loops_.size()) {
        loops_[destination]->conditions.emplace_back(op->condition, op->else_case.has_value());
      }
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  std::vector<Loop*> loops_;
};

namespace transform {

Pass HoistIf() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto* n = f.CopyOnWrite();
    Stmt body = ffi::make_object<IfHoister>()
                    ->Mutate(n->body.value(), InplaceMode::kAllow)
                    .ValueOrUnchanged(n->body.value());
    n->body = tirx::ConvertSSA(std::move(body));
    return f;
  };
  return tvm::transform::Sequential(
      {CreateFunctionPass(pass_func, 0, "tirx.InsertHoistedIf", {}), StmtSimplify(), RemoveNoOp()},
      "tirx.HoistIf");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.HoistIf", HoistIf);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
