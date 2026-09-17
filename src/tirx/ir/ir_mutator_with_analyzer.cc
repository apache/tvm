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
 * \file tirx/ir/ir_mutator_with_analyzer.cc
 */
#include "ir_mutator_with_analyzer.h"

#include <tvm/arith/iter_affine_map.h>
#include <tvm/ffi/cast.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>

#include "../../arith/constraint_helpers.h"

namespace tvm {
namespace tirx {
void IRMutatorWithAnalyzer::InitVTable(VTable* vtable) { StmtExprMutator::InitVTable(vtable); }

const IRMutatorWithAnalyzer::VTable* IRMutatorWithAnalyzer::GlobalVTable() {
  static const VTable table = [] {
    VTable table;
    InitVTable(&table);
    table.Finalize();
    return table;
  }();
  return &table;
}

using namespace tvm::prim;

using arith::detail::EnterConstraintFacts;

void IRMutatorWithAnalyzer::MarkBufferParamShapes(const tirx::PrimFunc& func) {
  // Mark all symbolic buffer-parameter shape values as positive.
  for (const tirx::Var& param : func->params) {
    if (!param->ty.as<tirx::BufferTypeNode>()) {
      continue;
    }
    tirx::BufferVar buffer(param);
    for (PrimExpr shape : buffer->shape) {
      analyzer_->MarkGlobalNonNegValue(shape);
    }
  }
}

ffi::Array<PrimExpr> IRMutatorWithAnalyzer::IterMapSimplifyWithContext(
    const ffi::Array<PrimExpr>& indices, bool non_trivial_only) {
  PrimExpr pred = IntImm::Bool(true);
  for (PrimExpr val : iter_predicates_) {
    pred = pred && val;
  }
  int n = indices.size();
  arith::Analyzer analyzer_ref = ffi::GetRef<arith::Analyzer>(this->analyzer_);
  ffi::Array<PrimExpr> simplified = arith::IterMapSimplify(
      indices, this->iter_vars_, pred, arith::IterMapLevel::Surjective, analyzer_ref);
  if (non_trivial_only) {
    for (int i = 0; i < n; ++i) {
      if (simplified[i]->IsInstance<IntImmNode>() && indices[i].as<PrimVar>()) {
        simplified.Set(i, indices[i]);
      }
    }
  }
  return simplified;
}

UnchangedOr<Stmt> IRMutatorWithAnalyzer::Mutate_(const ForNode* op, InplaceMode inplace_mode) {
  return constraint_scope_.WithNewScope([&]() -> UnchangedOr<Stmt> {
    // record the loop variable as iterators
    Range dom = Range::FromMinExtent(op->min, op->extent);
    analyzer_->Bind(op->loop_var, dom);
    iter_vars_.Set(op->loop_var, dom);
    auto min_result = this->Mutate(op->min, inplace_mode);
    bool min_unchanged = min_result.UnchangedOrSameAs(op->min);
    PrimExpr min = std::move(min_result).ValueOrUnchanged(op->min);
    auto extent_result = this->Mutate(op->extent, inplace_mode);
    bool extent_unchanged = extent_result.UnchangedOrSameAs(op->extent);
    PrimExpr extent = std::move(extent_result).ValueOrUnchanged(op->extent);
    ffi::Optional<PrimExpr> step{std::nullopt};
    if (op->step.has_value()) {
      step = this->Mutate(*op->step, inplace_mode).ValueOrUnchanged(*op->step);
    }
    Stmt body = constraint_scope_.WithNewScope([&]() -> Stmt {
      EnterConstraintFacts(&constraint_scope_.Current(), analyzer_,
                           extent > IntImm(extent.ty(), 0));
      return this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    });
    if (min_unchanged && extent_unchanged && body.same_as(op->body) && step.same_as(op->step)) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        auto* n = const_cast<ForNode*>(op);
        n->min = std::move(min);
        n->extent = std::move(extent);
        n->step = std::move(step);
        n->body = std::move(body);
        return ffi::Unchanged();
      }
      auto n = ffi::make_object<ForNode>(*op);
      n->min = std::move(min);
      n->extent = std::move(extent);
      n->step = std::move(step);
      n->body = std::move(body);
      return Stmt(n);
    }
  });
}

UnchangedOr<Stmt> IRMutatorWithAnalyzer::Mutate_(const BindNode* op, InplaceMode inplace_mode) {
  auto value_result = this->Mutate(op->value, inplace_mode);
  bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
  Expr value = std::move(value_result).ValueOrUnchanged(op->value);
  if (auto prim_value = value.as<PrimExpr>()) {
    if (SideEffect(prim_value.value()) <= CallEffectKind::kPure) {
      analyzer_->Bind(op->var, prim_value.value());
    }
  }
  if (value_unchanged) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* n = const_cast<BindNode*>(op);
    n->value = std::move(value);
    return ffi::Unchanged();
  }
  auto n = ffi::make_object<BindNode>(*op);
  n->value = std::move(value);
  return Stmt(n);
}

UnchangedOr<Stmt> IRMutatorWithAnalyzer::Mutate_(const IfThenElseNode* op,
                                                 InplaceMode inplace_mode) {
  return constraint_scope_.WithNewScope([&]() -> UnchangedOr<Stmt> {
    auto condition_result = this->Mutate(op->condition, inplace_mode);
    bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
    PrimExpr condition = std::move(condition_result).ValueOrUnchanged(op->condition);
    PrimExpr real_condition = condition;

    if (auto call = condition.as<CallNode>()) {
      static const Op& likely_op = Op::Get("prim.likely");
      if (call->op.same_as(likely_op)) {
        real_condition = call->args[0].as_or_throw<PrimExpr>();
      }
    }

    Stmt then_case;
    ffi::Optional<Stmt> else_case;
    constraint_scope_.WithNewScope([&]() {
      EnterConstraintFacts(&constraint_scope_.Current(), analyzer_, real_condition);
      WithRecordIterPredicate(real_condition, [&] {
        then_case = this->Mutate(op->then_case, inplace_mode).ValueOrUnchanged(op->then_case);
      });
    });
    if (op->else_case) {
      PrimExpr neg_condition = analyzer_->rewrite_simplify(prim::Not(real_condition));
      constraint_scope_.WithNewScope([&]() {
        constraint_scope_.Current().Emplace(analyzer_, neg_condition);
        else_case = this->Mutate(op->else_case.value(), inplace_mode)
                        .ValueOrUnchanged(op->else_case.value());
      });
    }
    if (is_one(real_condition)) return then_case;
    if (is_zero(real_condition)) {
      return else_case.value_or(Evaluate(0));
    }

    if (condition_unchanged && then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        auto* n = const_cast<IfThenElseNode*>(op);
        n->condition = std::move(condition);
        n->then_case = std::move(then_case);
        n->else_case = std::move(else_case);
        return ffi::Unchanged();
      }
      auto n = ffi::make_object<IfThenElseNode>(*op);
      n->condition = std::move(condition);
      n->then_case = std::move(then_case);
      n->else_case = std::move(else_case);
      return Stmt(n);
    }
  });
}

UnchangedOr<Stmt> IRMutatorWithAnalyzer::Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) {
  return constraint_scope_.WithNewScope([&]() -> UnchangedOr<Stmt> {
    if (op->attr_key == tirx::attr::thread_extent || op->attr_key == "virtual_thread") {
      IterVar iv = op->node.as_or_throw<IterVar>();
      TVM_FFI_ICHECK_NE(iv->thread_tag.length(), 0U);
      Range dom = Range::FromMinExtent(IntImm(op->value.ty(), 0), op->value);
      analyzer_->Bind(iv->var, dom);
      iter_vars_.Set(iv->var, dom);
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  });
}

UnchangedOr<Stmt> IRMutatorWithAnalyzer::Mutate_(const AssertStmtNode* op,
                                                 InplaceMode inplace_mode) {
  auto condition_result = this->Mutate(op->condition, inplace_mode);
  bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
  PrimExpr condition = std::move(condition_result).ValueOrUnchanged(op->condition);
  constraint_scope_.Current().Emplace(analyzer_, condition);

  if (condition_unchanged) {
    return ffi::Unchanged();
  } else {
    if (inplace_mode == InplaceMode::kAllow) {
      auto* n = const_cast<AssertStmtNode*>(op);
      n->condition = std::move(condition);
      return ffi::Unchanged();
    }
    auto n = ffi::make_object<AssertStmtNode>(*op);
    n->condition = std::move(condition);
    return Stmt(n);
  }
}

UnchangedOr<Expr> IRMutatorWithAnalyzer::Mutate_(const CallNode* op, InplaceMode inplace_mode) {
  // add condition context to if_then_else
  static const Op& if_then_else_op = Op::Get("prim.if_then_else");
  if (op->op.same_as(if_then_else_op)) {
    PrimExpr cond = this->Mutate(op->args[0]).ValueOrUnchanged(op->args[0]).as_or_throw<PrimExpr>();
    Expr true_value, false_value;
    constraint_scope_.WithNewScope([&]() {
      EnterConstraintFacts(&constraint_scope_.Current(), analyzer_, cond);
      WithRecordIterPredicate(cond, [&] {
        true_value = this->Mutate(op->args[1]).ValueOrUnchanged(op->args[1]).as_or_throw<Expr>();
      });
    });
    {
      PrimExpr not_cond = prim::Not(cond);
      constraint_scope_.WithNewScope([&]() {
        constraint_scope_.Current().Emplace(analyzer_, not_cond);
        WithRecordIterPredicate(not_cond, [&] {
          false_value = this->Mutate(op->args[2]).ValueOrUnchanged(op->args[2]).as_or_throw<Expr>();
        });
      });
    }
    if (is_zero(cond)) {
      return false_value;
    }
    if (is_one(cond)) {
      return true_value;
    }
    if (cond.same_as(op->args[0]) && true_value.same_as(op->args[1]) &&
        false_value.same_as(op->args[2])) {
      return ffi::Unchanged();
    } else {
      return Call(op->ty, op->op, {cond, true_value, false_value}, op->attrs, {}, op->span);
    }
  }
  return StmtExprMutator::Mutate_(op, inplace_mode);
}

UnchangedOr<PrimExpr> IRMutatorWithAnalyzer::Mutate_(const prim::LetNode* op,
                                                     InplaceMode inplace_mode) {
  auto value_result = this->Mutate(op->value, inplace_mode);
  bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
  PrimExpr value = std::move(value_result).ValueOrUnchanged(op->value);
  if (SideEffect(value) <= CallEffectKind::kPure) {
    analyzer_->Bind(op->var, value);
  }
  // We keep the let-binding here
  // as sub-class may or maynot choose to replace it.
  auto body_result = this->Mutate(op->body, inplace_mode);
  bool body_unchanged = body_result.UnchangedOrSameAs(op->body);
  PrimExpr body = std::move(body_result).ValueOrUnchanged(op->body);
  if (value_unchanged && body_unchanged) {
    return ffi::Unchanged();
  } else {
    return prim::Let(op->var, value, body);
  }
}

UnchangedOr<PrimExpr> IRMutatorWithAnalyzer::Mutate_(const prim::SelectNode* op,
                                                     InplaceMode inplace_mode) {
  auto cond_result = this->Mutate(op->condition, inplace_mode);
  bool cond_unchanged = cond_result.UnchangedOrSameAs(op->condition);
  PrimExpr cond = std::move(cond_result).ValueOrUnchanged(op->condition);
  PrimExpr true_value, false_value;
  constraint_scope_.WithNewScope([&]() {
    EnterConstraintFacts(&constraint_scope_.Current(), analyzer_, cond);
    true_value = Mutate(op->true_value, inplace_mode).ValueOrUnchanged(op->true_value);
  });
  {
    PrimExpr neg_cond = analyzer_->rewrite_simplify(prim::Not(cond));
    constraint_scope_.WithNewScope([&]() {
      constraint_scope_.Current().Emplace(analyzer_, neg_cond);
      false_value = Mutate(op->false_value, inplace_mode).ValueOrUnchanged(op->false_value);
    });
  }
  if (is_zero(cond)) {
    return false_value;
  }
  if (is_one(cond)) {
    return true_value;
  }
  // normal path
  if (cond_unchanged && true_value.same_as(op->true_value) &&
      false_value.same_as(op->false_value)) {
    return ffi::Unchanged();
  } else {
    return prim::Select(cond, true_value, false_value);
  }
}

}  // namespace tirx
}  // namespace tvm
