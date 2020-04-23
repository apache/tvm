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
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include "ir_mutator_with_analyzer.h"

namespace tvm {
namespace arith {

using namespace tir;

Stmt IRMutatorWithAnalyzer::
VisitStmt_(const ForNode* op) {
  analyzer_->Bind(op->loop_var,
                  Range::make_by_min_extent(op->min, op->extent));
  return StmtExprMutator::VisitStmt_(op);
}

Stmt IRMutatorWithAnalyzer::
VisitStmt_(const LetStmtNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  if (!tir::HasSideEffect(value)) {
    analyzer_->Bind(op->var, value);
  }
  // We keep the let-binding here
  // as sub-class may or maynot choose to replace it.
  Stmt body = this->VisitStmt(op->body);
  if (value.same_as(op->value) &&
      body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = this->CopyOnWrite(op);
    n->value = std::move(value);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt IRMutatorWithAnalyzer::
VisitStmt_(const IfThenElseNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  Stmt then_case, else_case;
  {
    With<ConstraintContext> ctx(analyzer_, condition);
    then_case = this->VisitStmt(op->then_case);
  }
  if (op->else_case.defined()) {
      With<ConstraintContext> ctx(analyzer_,
                                  analyzer_->rewrite_simplify(NotNode::make(condition)));
      else_case = this->VisitStmt(op->else_case);
  }
  if (is_one(condition)) return then_case;
  if (is_zero(condition)) {
    if (else_case.defined()) {
      return else_case;
    }
    return EvaluateNode::make(0);
  }

  if (condition.same_as(op->condition) &&
      then_case.same_as(op->then_case) &&
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

Stmt IRMutatorWithAnalyzer::
VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::thread_extent ||
      op->attr_key == tir::attr::virtual_thread) {
    IterVar iv = Downcast<IterVar>(op->node);
    CHECK_NE(iv->thread_tag.length(), 0U);
    analyzer_->Bind(iv->var,
                    Range::make_by_min_extent(0, op->value));
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    return stmt;
  } else {
    return StmtExprMutator::VisitStmt_(op);
  }
}

Stmt IRMutatorWithAnalyzer::
VisitStmt_(const AssertStmtNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  PrimExpr message = this->VisitExpr(op->message);
  With<ConstraintContext> ctx(analyzer_, condition);
  Stmt body = this->VisitStmt(op->body);

  if (condition.same_as(op->condition) &&
      message.same_as(op->message) &&
      body.same_as(op->body)) {
    return GetRef<Stmt>(op);
  } else {
    auto n = this->CopyOnWrite(op);
    n->condition = std::move(condition);
    n->message = std::move(message);
    n->body = std::move(body);
    return Stmt(n);
  }
}

PrimExpr IRMutatorWithAnalyzer::
VisitExpr_(const CallNode* op) {
  // add condition context to if_then_else
  if (op->is_intrinsic(tir::intrinsic::tvm_if_then_else)) {
    PrimExpr cond = this->VisitExpr(op->args[0]);
    PrimExpr true_value, false_value;
    {
      With<ConstraintContext> constraint(analyzer_, cond);
      true_value = this->VisitExpr(op->args[1]);
    }
    {
      With<ConstraintContext> constraint(analyzer_,
                                         analyzer_->rewrite_simplify(NotNode::make(cond)));
      false_value = this->VisitExpr(op->args[2]);
    }
    if (is_zero(cond)) {
      return false_value;
    }
    if (is_one(cond)) {
      return true_value;
    }
    if (cond.same_as(op->args[0]) &&
        true_value.same_as(op->args[1]) &&
        false_value.same_as(op->args[2])) {
      return GetRef<PrimExpr>(op);
    } else {
      return CallNode::make(op->dtype, op->name,
                        {cond, true_value, false_value},
                        op->call_type);
    }
  }
  return StmtExprMutator::VisitExpr_(op);
}

PrimExpr IRMutatorWithAnalyzer::
VisitExpr_(const LetNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  if (!tir::HasSideEffect(value)) {
    analyzer_->Bind(op->var, value);
  }
  // We keep the let-binding here
  // as sub-class may or maynot choose to replace it.
  PrimExpr body = this->VisitExpr(op->body);
  if (value.same_as(op->value) &&
      body.same_as(op->body)) {
    return GetRef<PrimExpr>(op);
  } else {
    return LetNode::make(op->var, value, body);
  }
}

PrimExpr IRMutatorWithAnalyzer::
VisitExpr_(const SelectNode* op) {
  PrimExpr cond = this->VisitExpr(op->condition);
  PrimExpr true_value, false_value;
  {
    With<ConstraintContext> constraint(analyzer_, cond);
    true_value = VisitExpr(op->true_value);
  }
  {
    With<ConstraintContext> constraint(analyzer_,
                                       analyzer_->rewrite_simplify(NotNode::make(cond)));
    false_value = VisitExpr(op->false_value);
  }
  if (is_zero(cond)) {
    return false_value;
  }
  if (is_one(cond)) {
    return true_value;
  }
  // normal path
  if (cond.same_as(op->condition) &&
      true_value.same_as(op->true_value) &&
      false_value.same_as(op->false_value)) {
    return GetRef<PrimExpr>(op);
  } else {
    return SelectNode::make(cond, true_value, false_value);
  }
}

PrimExpr IRMutatorWithAnalyzer::
VisitExpr_(const ReduceNode* op) {
  // Setup the domain information before simplification.
  for (const IterVar& iv : op->axis) {
    analyzer_->Bind(iv->var, iv->dom);
  }
  // Recursively call simplification when necessary.
  return StmtExprMutator::VisitExpr_(op);
}

}  // namespace arith
}  // namespace tvm
