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
 *  Copyright (c) 2016 by Contributors
 *  SSA related checks and pass.
 *
 *  SSA requires each varaible to be only defined once.
 * \file ssa.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace ir {
namespace {
class IRVerifySSA final : public IRVisitor {
 public:
  bool is_ssa{true};

  void Visit(const NodeRef& n) final {
    if (!is_ssa) return;
    IRVisitor::Visit(n);
  }
  void Visit_(const Let* op) final {
    MarkDef(op->var.get());
    IRVisitor::Visit_(op);
  }
  void Visit_(const LetStmt* op) final {
    MarkDef(op->var.get());
    IRVisitor::Visit_(op);
  }
  void Visit_(const For* op) final {
    MarkDef(op->loop_var.get());
    IRVisitor::Visit_(op);
  }
  void Visit_(const Allocate* op) final {
    MarkDef(op->buffer_var.get());
    IRVisitor::Visit_(op);
  }

 private:
  void MarkDef(const Variable* v) {
    if (defined_.count(v) != 0) {
      is_ssa = false; return;
    } else {
      defined_[v] = 1;
    }
  }
  std::unordered_map<const Variable*, int> defined_;
};

class IRConvertSSA final : public IRMutator {
 public:
  Expr Mutate_(const Variable* op, const Expr& e) final {
    if (scope_.count(op)) {
      return scope_[op].back();
    } else {
      return e;
    }
  }
  Expr Mutate_(const Let* op, const Expr& e) final {
    const VarExpr& v = op->var;
    if (defined_.count(v.get())) {
      Expr value = IRMutator::Mutate(op->value);
      VarExpr new_var = Variable::make(v.type(), v->name_hint);
      scope_[v.get()].push_back(new_var);
      Expr body = IRMutator::Mutate(op->body);
      scope_[v.get()].pop_back();
      return Let::make(new_var, value, body);
    } else {
      defined_.insert(v.get());
      return IRMutator::Mutate_(op, e);
    }
  }
  Expr Mutate_(const Load* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Load>();
    if (scope_.count(op->buffer_var.get())) {
      return Load::make(
          op->type, scope_[op->buffer_var.get()].back(),
          op->index, op->predicate);
    } else {
      return expr;
    }
  }
  Stmt Mutate_(const Store* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Store>();
    if (scope_.count(op->buffer_var.get())) {
      return Store::make(
          scope_[op->buffer_var.get()].back(), op->value,
          op->index, op->predicate);
    } else {
      return stmt;
    }
  }
  Stmt Mutate_(const LetStmt* op, const Stmt& s) final {
    const VarExpr& v = op->var;
    if (defined_.count(v.get())) {
      Expr value = IRMutator::Mutate(op->value);
      VarExpr new_var = Variable::make(v.type(), v->name_hint);
      scope_[v.get()].push_back(new_var);
      Stmt body = IRMutator::Mutate(op->body);
      scope_[v.get()].pop_back();
      return LetStmt::make(new_var, value, body);
    } else {
      defined_.insert(v.get());
      return IRMutator::Mutate_(op, s);
    }
  }
  Stmt Mutate_(const For* op, const Stmt& s) final {
    const VarExpr& v = op->loop_var;
    if (defined_.count(v.get())) {
      VarExpr new_var = Variable::make(v.type(), v->name_hint);
      scope_[v.get()].push_back(new_var);
      Stmt stmt = IRMutator::Mutate_(op, s);
      scope_[v.get()].pop_back();
      op = stmt.as<For>();
      return For::make(
          new_var, op->min, op->extent, op->for_type, op->device_api, op->body);
    } else {
      defined_.insert(v.get());
      return IRMutator::Mutate_(op, s);
    }
  }
  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    const VarExpr& v = op->buffer_var;
    if (defined_.count(v.get())) {
      VarExpr new_var = Variable::make(v.type(), v->name_hint);
      scope_[v.get()].push_back(new_var);
      Stmt stmt = IRMutator::Mutate_(op, s);
      scope_[v.get()].pop_back();
      op = stmt.as<Allocate>();
      return Allocate::make(
          new_var, op->type, op->extents, op->condition,
          op->body, op->new_expr, op->free_function);
    } else {
      defined_.insert(v.get());
      return IRMutator::Mutate_(op, s);
    }
  }
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (const Variable* v = op->node.as<Variable>()) {
      if (op->attr_key == attr::storage_scope) {
        const Allocate* alloc = op->body.as<Allocate>();
        if (alloc && op->node.same_as(alloc->buffer_var)) {
          Stmt new_alloc = Mutate(op->body);
          if (new_alloc.same_as(op->body)) return s;
          alloc = new_alloc.as<Allocate>();
          CHECK(alloc);
          return AttrStmt::make(
              alloc->buffer_var, op->attr_key, op->value, new_alloc);
        }
      }
      Stmt stmt = IRMutator::Mutate_(op, s);
      op = stmt.as<AttrStmt>();
      if (scope_.count(v) && scope_[v].size() != 0) {
        return AttrStmt::make(
            scope_[v].back(), op->attr_key, op->value, op->body);
      } else {
        return stmt;
      }
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

 private:
  std::unordered_map<const Variable*, std::vector<VarExpr> > scope_;
  std::unordered_set<const Variable*> defined_;
};

}  // namespace

bool VerifySSA(const Stmt& ir) {
  IRVerifySSA v;
  v.Visit(ir);
  return v.is_ssa;
}

Stmt ConvertSSA(Stmt stmt) {
  return IRConvertSSA().Mutate(stmt);
}

}  // namespace ir
}  // namespace tvm
