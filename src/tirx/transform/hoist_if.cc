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
#include <tvm/ir/prim/op.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
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
    if (input.as<StmtNode>() && !input.as<ForNode>() && !input.as<IfNode>() &&
        !input.as<SeqStmtNode>()) {
      // Bindings, regions and unknown statements have no code-motion contract.
      // Hide outer loops and their variable depths while visiting this subtree;
      // active_loops_/split_in_nest_ still bound duplication of the enclosing nest.
      return scopes_.WithNewScope([&] { return StmtExprMutator::Mutate(input, inplace_mode); });
    }
    return StmtExprMutator::Mutate(input, inplace_mode);
  }

 private:
  struct LoopState {
    // Enclosing alternate-branch depth at loop entry, before visiting its body.
    int else_depth_at_entry;
    // Predicates lifted to this loop, in traversal order; bool retains a false-path copy.
    std::vector<std::pair<PrimExpr, bool>> conditions;
  };

  struct ScopeState {
    // Active destination loops, outermost first, within the current motion boundary.
    std::vector<LoopState*> loops;
    // Identity-based indices into loops for variables defined by those loops.
    std::unordered_map<const VarNode*, size_t> loop_depths;
  };

  UnchangedOr<Stmt> Mutate_(const SeqStmtNode* op, InplaceMode inplace_mode) final {
    // Sibling statements isolate loop placement; a singleton stays transparent.
    // An If inside one sibling must not move outside the loop executing all siblings.
    if (op->seq.size() == 1) return StmtExprMutator::Mutate_(op, inplace_mode);
    return scopes_.WithNewScope([&] { return StmtExprMutator::Mutate_(op, inplace_mode); });
  }

  // Purity alone does not permit evaluating a predicate before its guards or
  // before a zero-trip loop.  Check safety and loop dependencies in one walk.
  size_t FindLiftDestination(const PrimExpr& condition) const {
    const auto& scope = scopes_.Current();
    // First loop the predicate can cross: loops [i, j, k] and i + j select index 2 (k).
    size_t destination = 0;
    std::unordered_set<const ExprNode*> visited;
    auto advance = [&](const ExprNode* op) -> ffi::Expected<ffi::WalkResult> {
      // StructuralWalk visits occurrences. For Add(e, e), safety and dependencies
      // need checking only once per shared e, within this predicate's fixed scope.
      return visited.insert(op).second ? ffi::WalkResult::Advance() : ffi::WalkResult::Skip();
    };
    auto division = [&](const ExprNode* op,
                        const PrimExpr& divisor) -> ffi::Expected<ffi::WalkResult> {
      // Positive constant divisors exclude both zero and signed min / -1.
      const auto* value = divisor.as<IntImmNode>();
      if (!value || value->value <= 0) return ffi::WalkResult::Interrupt();
      return advance(op);
    };
    auto blocked = ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        condition,
        [&](const VarNode* op) -> ffi::Expected<ffi::WalkResult> {
          // Stay inside every referenced loop variable's definition. Other
          // variables remain in scope at every candidate destination.
          auto it = scope.loop_depths.find(op);
          if (it != scope.loop_depths.end()) destination = std::max(destination, it->second + 1);
          // An innermost-loop dependency already rules out every destination.
          if (destination == scope.loops.size()) return ffi::WalkResult::Interrupt();
          // Variable types describe metadata, not evaluated predicate dependencies.
          return ffi::WalkResult::Skip();
        },
        [](const TypeNode*) -> ffi::Expected<ffi::WalkResult> { return ffi::WalkResult::Skip(); },
        [](const TensorLoadNode*) -> ffi::Expected<ffi::WalkResult> {
          // Memory reads can change across iterations or depend on an execution guard.
          return ffi::WalkResult::Interrupt();
        },
        [&](const prim::DivNode* op) { return division(op, op->b); },
        [&](const prim::ModNode* op) { return division(op, op->b); },
        [&](const prim::FloorDivNode* op) { return division(op, op->b); },
        [&](const prim::FloorModNode* op) { return division(op, op->b); },
        [](const prim::LShiftNode*) -> ffi::Expected<ffi::WalkResult> {
          // A shift count may be outside the defined range.
          return ffi::WalkResult::Interrupt();
        },
        [](const prim::RShiftNode*) -> ffi::Expected<ffi::WalkResult> {
          // A shift count may be outside the defined range.
          return ffi::WalkResult::Interrupt();
        },
        [&](const prim::CastNode* op) -> ffi::Expected<ffi::WalkResult> {
          // Floating-to-integer conversion may be undefined outside the integer range.
          if (ffi::GetRef<PrimExpr>(op).ty().MatchesCode(kDLInt, kDLUInt) &&
              !op->value.ty().MatchesCode(kDLInt, kDLUInt))
            return ffi::WalkResult::Interrupt();
          return advance(op);
        },
        [&](const CallNode* op) -> ffi::Expected<ffi::WalkResult> {
          // Even pure calls may trap; only the transparent likely hint is known safe.
          if (!op->op.same_as(prim::likely_op())) return ffi::WalkResult::Interrupt();
          return advance(op);
        },
        advance);
    return blocked.has_value() ? scope.loops.size() : destination;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) final {
    // Moving guards inside a thread launch can prevent later synchronization
    // insertion, even when the guards themselves are loop-invariant.
    if (op->op.same_as(launch_thread_op())) return ffi::Unchanged();
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    // Preserve the same synchronization boundary before thread-binding lowering.
    if (GetThreadBinding(op).has_value()) return ffi::Unchanged();
    // Only ordinary serial loops permit code motion across their execution scope.
    // Hoisting must not suppress evaluation of an effectful loop header either.
    if (op->kind != ForKind::kDefault || SideEffect(op->min) > CallEffectKind::kReadState ||
        SideEffect(op->extent) > CallEffectKind::kReadState ||
        (op->step && SideEffect(op->step.value()) > CallEffectKind::kReadState)) {
      return scopes_.WithNewScope([&] { return StmtExprMutator::Mutate_(op, inplace_mode); });
    }
    auto& scope = scopes_.Current();
    // A new outermost eligible loop gets one two-sided split. Do not reset at
    // scope barriers: an outer i / SeqStmt / inner j still shares this budget.
    if (active_loops_++ == 0) split_in_nest_ = false;
    // Snapshot ancestors outside this loop so they do not require a false-path copy.
    LoopState loop{else_depth_, {}};
    const VarNode* var = op->loop_var.get();
    // Register this destination and its variable together. An index in loops
    // identifies both where to wrap a lifted If and which variables it must avoid.
    scope.loop_depths[var] = scope.loops.size();
    scope.loops.push_back(&loop);
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    // Only descendants can lift to this loop; remove both entries before siblings.
    // The local LoopState remains alive below to emit its collected conditions.
    scope.loops.pop_back();
    scope.loop_depths.erase(var);
    // Reaching zero lets the next independent nest start with a fresh split budget.
    --active_loops_;
    // Collected p then q must become if p: if q: loop, so wrap q before p.
    for (auto it = loop.conditions.rbegin(); it != loop.conditions.rend(); ++it) {
      // Duplicate only when an alternate branch must remain reachable.
      const auto& [condition, preserve_else] = *it;
      stmt = preserve_else ? If(condition, stmt, SeqStmt(stmt)) : If(condition, stmt);
    }
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const IfNode* op, InplaceMode inplace_mode) final {
    auto& scope = scopes_.Current();
    // Moving a one-sided inner If could otherwise skip this predicate's effects.
    if (!scope.loops.empty() && SideEffect(op->condition) > CallEffectKind::kReadState) {
      return scopes_.WithNewScope([&] { return StmtExprMutator::Mutate_(op, inplace_mode); });
    }
    bool has_else = op->else_case.has_value();
    if (!scope.loops.empty()) {
      size_t destination = FindLiftDestination(op->condition);
      if (destination < scope.loops.size()) {
        // A false predicate must still execute an enclosing alternate branch
        // inside the destination loop, even when this If has no else of its own.
        // In for i: if p(i) { if q { A } } else { B }, q=false still needs B.
        // The entry snapshot excludes Ifs outside this loop,
        // whose alternate branches remain reachable without duplicating it.
        bool preserve_else =
            has_else || else_depth_ > scope.loops[destination]->else_depth_at_entry;
        auto* loop = scope.loops[destination];
        // Each two-sided predicate doubles the enclosing loop subtree. Limit
        // splitting to once per loop nest, even across code-motion barriers.
        if (!preserve_else || !split_in_nest_) {
          // Record at the destination now; its For unwind wraps the complete
          // rewritten body. Preorder collection preserves enclosing predicate order.
          loop->conditions.emplace_back(op->condition, preserve_else);
          // Hoists needing no false-path copy spend no budget. A copy consumes
          // it for the rest of this nest, including loops behind scope barriers.
          split_in_nest_ |= preserve_else;
        }
      }
    }
    // Descendant hoists must preserve this If's alternate arm, in either branch.
    else_depth_ += has_else;
    auto result = StmtExprMutator::Mutate_(op, inplace_mode);
    // Restore the ancestor count so following siblings do not inherit this If.
    else_depth_ -= has_else;
    return result;
  }

  // Number of enclosing Ifs with an else; each loop saves its entry baseline.
  int else_depth_{0};
  // Active eligible loops across motion boundaries; zero starts a new split budget.
  // Unlike scoped destinations, this stays nonzero inside a nested SeqStmt barrier.
  int active_loops_{0};
  // Whether this loop nest has used its one two-sided split, bounding subtree growth.
  bool split_in_nest_{false};
  // Scoped destinations/dependencies; barriers hide outer loops without resetting the budget.
  ScopeStack<ScopeState> scopes_;
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
      {CreateFunctionPass(pass_func, 0, "tirx.InsertHoistedIf"), StmtSimplify(), RemoveNoOp()},
      "tirx.HoistIf");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.HoistIf", HoistIf);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
