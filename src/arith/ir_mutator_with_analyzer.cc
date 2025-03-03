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
 * \file tvm/arith/ir_mutator_with_analyzer.cc
 */
#include "ir_mutator_with_analyzer.h"

#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace arith {

using namespace tir;

void IRMutatorWithAnalyzer::MarkBufferMapShapes(const tir::PrimFunc& func) {
  // Mark the all the symbolic buffer shape values in the buffer map as positive value.
  for (auto kv : func->buffer_map) {
    for (PrimExpr shape : kv.second->shape) {
      analyzer_->MarkGlobalNonNegValue(shape);
    }
  }
}

Array<PrimExpr> IRMutatorWithAnalyzer::IterMapSimplifyWithContext(const Array<PrimExpr>& indices,
                                                                  bool non_trivial_only) {
  PrimExpr pred = const_true();
  for (PrimExpr val : iter_predicates_) {
    pred = pred && val;
  }
  int n = indices.size();
  Array<PrimExpr> simplified = arith::IterMapSimplify(
      indices, this->iter_vars_, pred, arith::IterMapLevel::Surjective, this->analyzer_);
  if (non_trivial_only) {
    for (int i = 0; i < n; ++i) {
      if (simplified[i]->IsInstance<IntImmNode>() && indices[i]->IsInstance<VarNode>()) {
        simplified.Set(i, indices[i]);
      }
    }
  }
  return simplified;
}

Stmt IRMutatorWithAnalyzer::VisitStmt_(const ForNode* op) {
  // record the loop variable as iterators
  Range dom = Range::FromMinExtent(op->min, op->extent);
  analyzer_->Bind(op->loop_var, dom);
  iter_vars_.Set(op->loop_var, dom);
  return StmtExprMutator::VisitStmt_(op);
}

Stmt IRMutatorWithAnalyzer::VisitStmt_(const BlockNode* op) {
  for (const auto& iter_var : op->iter_vars) {
    analyzer_->Bind(iter_var->var, iter_var->dom);
    iter_vars_.Set(iter_var->var, iter_var->dom);
  }
  return StmtExprMutator::VisitStmt_(op);
}

Stmt IRMutatorWithAnalyzer::VisitStmt_(const LetStmtNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  if (SideEffect(value) <= CallEffectKind::kPure) {
    analyzer_->Bind(op->var, value);
  }
  // We keep the let-binding here
  // as sub-class may or maynot choose to replace it.
  Stmt body = this->VisitStmt(op->body);
  if (value.same_as(op->value) && body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = this->CopyOnWrite(op);
    n->value = std::move(value);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt IRMutatorWithAnalyzer::VisitStmt_(const IfThenElseNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  PrimExpr real_condition = condition;
  static auto op_likely = Op::Get("tir.likely");

  if (auto call = condition.as<CallNode>()) {
    if (call->op.same_as(op_likely)) {
      real_condition = call->args[0];
    }
  }

  Stmt then_case;
  Optional<Stmt> else_case;
  {
    With<ConstraintContext> ctx(analyzer_, real_condition);
    WithRecordIterPredicate(real_condition, [&] { then_case = this->VisitStmt(op->then_case); });
  }
  if (op->else_case) {
    With<ConstraintContext> ctx(analyzer_, analyzer_->rewrite_simplify(Not(real_condition)));
    else_case = this->VisitStmt(op->else_case.value());
  }
  if (is_one(real_condition)) return then_case;
  if (is_zero(real_condition)) {
    return else_case.value_or(Evaluate(0));
  }

  if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
      else_case.same_as(op->else_case)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = this->CopyOnWrite(op);
    n->condition = std::move(condition);
    n->then_case = std::move(then_case);
    n->else_case = std::move(else_case);
    return Stmt(n);
  }
}

Stmt IRMutatorWithAnalyzer::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::thread_extent || op->attr_key == tir::attr::virtual_thread) {
    IterVar iv = Downcast<IterVar>(op->node);
    ICHECK_NE(iv->thread_tag.length(), 0U);
    Range dom = Range::FromMinExtent(make_zero(op->value.dtype()), op->value);
    analyzer_->Bind(iv->var, dom);
    iter_vars_.Set(iv->var, dom);
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    return stmt;
  } else {
    return StmtExprMutator::VisitStmt_(op);
  }
}

Stmt IRMutatorWithAnalyzer::VisitStmt_(const AssertStmtNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  PrimExpr message = this->VisitExpr(op->message);
  With<ConstraintContext> ctx(analyzer_, condition);
  Stmt body = this->VisitStmt(op->body);

  if (condition.same_as(op->condition) && message.same_as(op->message) && body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = this->CopyOnWrite(op);
    n->condition = std::move(condition);
    n->message = std::move(message);
    n->body = std::move(body);
    return Stmt(n);
  }
}

PrimExpr IRMutatorWithAnalyzer::VisitExpr_(const CallNode* op) {
  // add condition context to if_then_else
  static auto op_if_then_else = Op::Get("tir.if_then_else");
  if (op->op.same_as(op_if_then_else)) {
    PrimExpr cond = this->VisitExpr(op->args[0]);
    PrimExpr true_value, false_value;
    {
      With<ConstraintContext> constraint(analyzer_, cond);
      WithRecordIterPredicate(cond, [&] { true_value = this->VisitExpr(op->args[1]); });
    }
    {
      PrimExpr not_cond = Not(cond);
      With<ConstraintContext> constraint(analyzer_, not_cond);
      WithRecordIterPredicate(not_cond, [&] { false_value = this->VisitExpr(op->args[2]); });
    }
    if (is_zero(cond)) {
      return false_value;
    }
    if (is_one(cond)) {
      return true_value;
    }
    if (cond.same_as(op->args[0]) && true_value.same_as(op->args[1]) &&
        false_value.same_as(op->args[2])) {
      return GetRef<PrimExpr>(op);
    } else {
      return Call(op->dtype, op->op, {cond, true_value, false_value});
    }
  }
  return StmtExprMutator::VisitExpr_(op);
}

PrimExpr IRMutatorWithAnalyzer::VisitExpr_(const LetNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  if (SideEffect(value) <= CallEffectKind::kPure) {
    analyzer_->Bind(op->var, value);
  }
  // We keep the let-binding here
  // as sub-class may or maynot choose to replace it.
  PrimExpr body = this->VisitExpr(op->body);
  if (value.same_as(op->value) && body.same_as(op->body)) {
    return GetRef<PrimExpr>(op);
  } else {
    return Let(op->var, value, body);
  }
}

PrimExpr IRMutatorWithAnalyzer::VisitExpr_(const SelectNode* op) {
  PrimExpr cond = this->VisitExpr(op->condition);
  PrimExpr true_value, false_value;
  {
    With<ConstraintContext> constraint(analyzer_, cond);
    true_value = VisitExpr(op->true_value);
  }
  {
    With<ConstraintContext> constraint(analyzer_, analyzer_->rewrite_simplify(Not(cond)));
    false_value = VisitExpr(op->false_value);
  }
  if (is_zero(cond)) {
    return false_value;
  }
  if (is_one(cond)) {
    return true_value;
  }
  // normal path
  if (cond.same_as(op->condition) && true_value.same_as(op->true_value) &&
      false_value.same_as(op->false_value)) {
    return GetRef<PrimExpr>(op);
  } else {
    return Select(cond, true_value, false_value);
  }
}

PrimExpr IRMutatorWithAnalyzer::VisitExpr_(const ReduceNode* op) {
  // Setup the domain information before simplification.
  for (const IterVar& iv : op->axis) {
    analyzer_->Bind(iv->var, iv->dom);
  }
  // Recursively call simplification when necessary.
  return StmtExprMutator::VisitExpr_(op);
}

}  // namespace arith
}  // namespace tvm
