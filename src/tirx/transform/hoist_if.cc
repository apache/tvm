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
#include <tvm/tirx/analysis.h>
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
      return MutateLocally([&] { return StmtExprMutator::Mutate(input, inplace_mode); });
    }
    return StmtExprMutator::Mutate(input, inplace_mode);
  }

 private:
  struct Loop {
    int else_depth;
    std::vector<std::pair<PrimExpr, bool>> conditions;
  };

  // Sibling statements and region boundaries isolate both placement contexts.
  template <typename F>
  auto MutateLocally(F mutate) -> decltype(mutate()) {
    auto outer_loops = std::move(loops_);
    auto outer_depths = std::move(loop_depths_);
    loops_.clear();
    loop_depths_.clear();
    auto result = mutate();
    loops_ = std::move(outer_loops);
    loop_depths_ = std::move(outer_depths);
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtNode* op, InplaceMode inplace_mode) final {
    if (op->seq.size() == 1) return StmtExprMutator::Mutate_(op, inplace_mode);
    return MutateLocally([&] { return StmtExprMutator::Mutate_(op, inplace_mode); });
  }

  // Purity alone does not permit evaluating a predicate before its guards or
  // before a zero-trip loop.  Check safety and loop dependencies in one walk.
  size_t FindDestination(const PrimExpr& condition) const {
    size_t destination = 0;
    std::unordered_set<const ExprNode*> visited;
    auto advance = [&](const ExprNode* op) -> ffi::Expected<ffi::WalkResult> {
      // StructuralWalk visits occurrences; prune shared expression subtrees.
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
          auto it = loop_depths_.find(op);
          if (it != loop_depths_.end()) destination = std::max(destination, it->second + 1);
          // An innermost-loop dependency already rules out every destination.
          if (destination == loops_.size()) return ffi::WalkResult::Interrupt();
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
    return blocked.has_value() ? loops_.size() : destination;
  }

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    Loop loop{else_depth_, {}};
    const VarNode* var = op->loop_var.get();
    auto it = loop_depths_.find(var);
    size_t previous_depth = it == loop_depths_.end() ? loops_.size() : it->second;
    loop_depths_[var] = loops_.size();
    loops_.push_back(&loop);
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    loops_.pop_back();
    if (previous_depth == loops_.size()) {
      loop_depths_.erase(var);
    } else {
      loop_depths_[var] = previous_depth;
    }
    for (auto it = loop.conditions.rbegin(); it != loop.conditions.rend(); ++it) {
      // Duplicate only when an alternate branch must remain reachable.
      stmt = it->second ? If(it->first, stmt, SeqStmt(stmt)) : If(it->first, stmt);
    }
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const IfNode* op, InplaceMode inplace_mode) final {
    bool has_else = op->else_case.has_value();
    if (!loops_.empty()) {
      size_t destination = FindDestination(op->condition);
      if (destination < loops_.size()) {
        // A false predicate must still execute an enclosing alternate branch
        // inside the destination loop, even when this If has no else of its own.
        // Branches outside that loop remain guarded and need no extra loop copy.
        bool preserve_else = has_else || else_depth_ > loops_[destination]->else_depth;
        loops_[destination]->conditions.emplace_back(op->condition, preserve_else);
      }
    }
    else_depth_ += has_else;
    auto result = StmtExprMutator::Mutate_(op, inplace_mode);
    else_depth_ -= has_else;
    return result;
  }

  // Number of enclosing Ifs with an else; each loop saves its entry baseline.
  int else_depth_{0};
  std::vector<Loop*> loops_;
  std::unordered_map<const VarNode*, size_t> loop_depths_;
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
