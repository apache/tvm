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
 * \file tvm/arithmetic/ir_mutator_with_analyzer.cc
 */
#include <tvm/ir_pass.h>
#include <tvm/expr_operator.h>
#include "ir_mutator_with_analyzer.h"

namespace tvm {
namespace arith {

using namespace ir;

Stmt IRMutatorWithAnalyzer::
Mutate_(const For* op, const Stmt& s) {
  analyzer_->Bind(op->loop_var,
                 Range::make_by_min_extent(op->min, op->extent));
  return IRMutator::Mutate_(op, s);
}

Stmt IRMutatorWithAnalyzer::
Mutate_(const LetStmt* op, const Stmt& s) {
  Expr value = this->Mutate(op->value);
  if (!ir::HasSideEffect(value)) {
    analyzer_->Bind(op->var, value);
  }
  // We keep the let-binding here
  // as sub-class may or maynot choose to replace it.
  Stmt body = this->Mutate(op->body);
  if (value.same_as(op->value) &&
      body.same_as(op->body)) {
    return s;
  } else {
    return LetStmt::make(op->var, value, body);
  }
}

Stmt IRMutatorWithAnalyzer::
Mutate_(const IfThenElse* op, const Stmt& s) {
  Expr condition = this->Mutate(op->condition);
  Stmt then_case, else_case;
  {
    With<ConstraintContext> ctx(analyzer_, condition);
    then_case = this->Mutate(op->then_case);
  }
  if (op->else_case.defined()) {
      With<ConstraintContext> ctx(analyzer_,
                                  analyzer_->rewrite_simplify(Not::make(condition)));
      else_case = this->Mutate(op->else_case);
  }
  if (is_one(condition)) return then_case;
  if (is_zero(condition)) {
    if (else_case.defined()) {
      return else_case;
    }
    return Evaluate::make(0);
  }

  if (condition.same_as(op->condition) &&
      then_case.same_as(op->then_case) &&
      else_case.same_as(op->else_case)) {
    return s;
  } else {
    return IfThenElse::make(condition, then_case, else_case);
  }
}

Stmt IRMutatorWithAnalyzer::
Mutate_(const AttrStmt* op, const Stmt& s) {
  if (op->attr_key == attr::thread_extent ||
      op->attr_key == attr::virtual_thread) {
    IterVar iv = Downcast<IterVar>(op->node);
    CHECK_NE(iv->thread_tag.length(), 0U);
    analyzer_->Bind(iv->var,
                    Range::make_by_min_extent(0, op->value));
    Stmt stmt = IRMutator::Mutate_(op, s);
    return stmt;
  } else {
    return IRMutator::Mutate_(op, s);
  }
}

Stmt IRMutatorWithAnalyzer::
Mutate_(const AssertStmt* op, const Stmt& s) {
  Expr condition = this->Mutate(op->condition);
  Expr message = this->Mutate(op->message);
  With<ConstraintContext> ctx(analyzer_, condition);
  Stmt body = this->Mutate(op->body);

  if (condition.same_as(op->condition) &&
      message.same_as(op->message) &&
      body.same_as(op->body)) {
    return s;
  } else {
    return AssertStmt::make(condition, message, body);
  }
}

Expr IRMutatorWithAnalyzer::
Mutate_(const Call* op, const Expr& self) {
  // add condition context to if_then_else
  if (op->is_intrinsic(ir::intrinsic::tvm_if_then_else)) {
    Expr cond = Mutate(op->args[0]);
    Expr true_value, false_value;
    {
      With<ConstraintContext> constraint(analyzer_, cond);
      true_value = Mutate(op->args[1]);
    }
    {
      With<ConstraintContext> constraint(analyzer_,
                                         analyzer_->rewrite_simplify(Not::make(cond)));
      false_value = Mutate(op->args[2]);
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
      return self;
    } else {
      return Call::make(op->dtype, op->name,
                        {cond, true_value, false_value},
                        op->call_type);
    }
  }
  return IRMutator::Mutate_(op, self);
}

Expr IRMutatorWithAnalyzer::
Mutate_(const Let* op, const Expr& self) {
  Expr value = this->Mutate(op->value);
  if (!ir::HasSideEffect(value)) {
    analyzer_->Bind(op->var, value);
  }
  // We keep the let-binding here
  // as sub-class may or maynot choose to replace it.
  Expr body = this->Mutate(op->body);
  if (value.same_as(op->value) &&
      body.same_as(op->body)) {
    return self;
  } else {
    return Let::make(op->var, value, body);
  }
}

Expr IRMutatorWithAnalyzer::
Mutate_(const Select* op, const Expr& self) {
  Expr cond = Mutate(op->condition);
  Expr true_value, false_value;
  {
    With<ConstraintContext> constraint(analyzer_, cond);
    true_value = Mutate(op->true_value);
  }
  {
    With<ConstraintContext> constraint(analyzer_,
                                       analyzer_->rewrite_simplify(Not::make(cond)));
    false_value = Mutate(op->false_value);
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
    return self;
  } else {
    return Select::make(cond, true_value, false_value);
  }
}

Expr IRMutatorWithAnalyzer::
Mutate_(const Reduce* op, const Expr& self) {
  // Setup the domain information before simplification.
  for (const IterVar& iv : op->axis) {
    analyzer_->Bind(iv->var, iv->dom);
  }
  // Recursively call simplification when necessary.
  return IRMutator::Mutate_(op, self);
}

}  // namespace arith
}  // namespace tvm
