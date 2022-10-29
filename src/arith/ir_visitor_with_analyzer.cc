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
 * \file tvm/arith/ir_visitor_with_analyzer.cc
 */
#include "ir_visitor_with_analyzer.h"

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace arith {

using namespace tir;

void IRVisitorWithAnalyzer::VisitStmt_(const ForNode* op) {
  analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
  StmtExprVisitor::VisitStmt_(op);
}

void IRVisitorWithAnalyzer::VisitStmt_(const BlockNode* op) {
  for (const auto& iter_var : op->iter_vars) {
    analyzer_.Bind(iter_var->var, iter_var->dom);
  }
  StmtExprVisitor::VisitStmt_(op);
}

void IRVisitorWithAnalyzer::VisitStmt_(const LetStmtNode* op) {
  this->VisitExpr(op->value);
  analyzer_.Bind(op->var, op->value);
  this->VisitStmt(op->body);
}

void IRVisitorWithAnalyzer::VisitStmt_(const IfThenElseNode* op) {
  this->VisitExpr(op->condition);

  PrimExpr real_condition = ExtractRealCondition(op->condition);

  {
    With<ConstraintContext> constraint(&analyzer_, real_condition);
    this->VisitStmt(op->then_case);
  }
  if (op->else_case) {
    With<ConstraintContext> constraint(&analyzer_, analyzer_.rewrite_simplify(Not(real_condition)));
    this->VisitStmt(op->else_case.value());
  }
}

void IRVisitorWithAnalyzer::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::thread_extent || op->attr_key == tir::attr::virtual_thread) {
    IterVar iv = Downcast<IterVar>(op->node);
    ICHECK_NE(iv->thread_tag.length(), 0U);
    analyzer_.Bind(iv->var, Range::FromMinExtent(0, op->value));
  }
  StmtExprVisitor::VisitStmt_(op);
}

void IRVisitorWithAnalyzer::VisitStmt_(const AssertStmtNode* op) {
  this->VisitExpr(op->condition);
  this->VisitExpr(op->message);
  With<ConstraintContext> constraint(&analyzer_, op->condition);
  this->VisitStmt(op->body);
}

void IRVisitorWithAnalyzer::VisitExpr_(const CallNode* op) {
  // add condition context to if_then_else
  static auto op_if_then_else = Op::Get("tir.if_then_else");
  if (op->op.same_as(op_if_then_else)) {
    PrimExpr cond = op->args[0];
    this->VisitExpr(op->args[0]);
    {
      With<ConstraintContext> constraint(&analyzer_, cond);
      this->VisitExpr(op->args[1]);
    }
    {
      With<ConstraintContext> constraint(&analyzer_, analyzer_.rewrite_simplify(Not(cond)));
      this->VisitExpr(op->args[2]);
    }
  } else {
    StmtExprVisitor::VisitExpr_(op);
  }
}

void IRVisitorWithAnalyzer::VisitExpr_(const LetNode* op) {
  this->VisitExpr(op->value);
  analyzer_.Bind(op->var, op->value);
  this->VisitExpr(op->body);
}

void IRVisitorWithAnalyzer::VisitExpr_(const ReduceNode* op) {
  for (const IterVar& iv : op->axis) {
    analyzer_.Bind(iv->var, iv->dom);
  }
  StmtExprVisitor::VisitExpr_(op);
}

PrimExpr IRVisitorWithAnalyzer::ExtractRealCondition(PrimExpr condition) const {
  if (auto call = condition.as<CallNode>()) {
    if (call->op.same_as(builtin::likely())) {
      return call->args[0];
    }
  }

  return condition;
}

}  // namespace arith
}  // namespace tvm
