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
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "ir_utils.h"

namespace tvm {
namespace tirx {

// Purity alone does not permit evaluating a predicate before its guards or
// before a zero-trip loop.  Keep potentially trapping operations in place.
static bool CanEvaluateEarly(const PrimExpr& condition) {
  if (SideEffect(condition) > CallEffectKind::kPure) return false;
  auto unsafe = ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      condition, [](const PrimExpr& expr) -> ffi::Expected<ffi::WalkResult> {
        ffi::Optional<PrimExpr> divisor;
        if (const auto* op = expr.as<prim::DivNode>()) divisor = op->b;
        if (const auto* op = expr.as<prim::ModNode>()) divisor = op->b;
        if (const auto* op = expr.as<prim::FloorDivNode>()) divisor = op->b;
        if (const auto* op = expr.as<prim::FloorModNode>()) divisor = op->b;
        bool safe = true;
        if (divisor.has_value()) {
          const auto* value = divisor.value().as<IntImmNode>();
          // Positive constant divisors exclude both zero and signed min / -1.
          safe = value && value->value > 0;
        }
        if (expr.as<prim::LShiftNode>() || expr.as<prim::RShiftNode>()) safe = false;
        if (const auto* op = expr.as<prim::CastNode>()) {
          if (expr.ty().MatchesCode(kDLInt, kDLUInt) &&
              !op->value.ty().MatchesCode(kDLInt, kDLUInt))
            safe = false;
        }
        if (const auto* op = expr.as<CallNode>()) {
          safe = op->op.same_as(prim::likely_op());
        }
        return safe ? ffi::WalkResult::Advance()
                    : ffi::WalkResult::Interrupt(ffi::VisitInterrupt(expr));
      });
  return !unsafe.has_value();
}

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
    int else_depth;
    std::vector<std::pair<PrimExpr, bool>> conditions;
  };

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    Loop loop{op->loop_var, else_depth_, {}};
    loops_.push_back(&loop);
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    loops_.pop_back();
    for (auto it = loop.conditions.rbegin(); it != loop.conditions.rend(); ++it) {
      // Duplicate only when an alternate branch must remain reachable.
      stmt = it->second ? If(it->first, stmt, SeqStmt(stmt)) : If(it->first, stmt);
    }
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const IfNode* op, InplaceMode inplace_mode) final {
    bool has_else = op->else_case.has_value();
    if (!loops_.empty() && CanEvaluateEarly(op->condition)) {
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
        // A no-else If can still have siblings in an enclosing alternate
        // branch.  Branches outside the destination loop do not require a copy.
        bool preserve_else = has_else || else_depth_ > loops_[destination]->else_depth;
        loops_[destination]->conditions.emplace_back(op->condition, preserve_else);
      }
    }
    else_depth_ += has_else;
    auto result = StmtExprMutator::Mutate_(op, inplace_mode);
    else_depth_ -= has_else;
    return result;
  }

  int else_depth_{0};
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
