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
 *  SSA related checks and pass.
 *
 *  SSA requires each varaible to be only defined once.
 * \file ssa.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/ir_pass.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace ir {
namespace {
class IRVerifySSA final : public StmtExprVisitor {
 public:
  bool is_ssa{true};

  void VisitExpr(const Expr& n) final {
    if (!is_ssa) return;
    StmtExprVisitor::VisitExpr(n);
  }
  void VisitStmt(const Stmt& n) final {
    if (!is_ssa) return;
    StmtExprVisitor::VisitStmt(n);
  }
  void VisitExpr_(const Let* op) final {
    MarkDef(op->var.get());
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const LetStmt* op) final {
    MarkDef(op->var.get());
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const For* op) final {
    MarkDef(op->loop_var.get());
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const Allocate* op) final {
    MarkDef(op->buffer_var.get());
    StmtExprVisitor::VisitStmt_(op);
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


class IRConvertSSA final : public StmtExprMutator {
 public:
  Expr VisitExpr_(const Variable* op) final {
    if (scope_.count(op)) {
      return scope_[op].back();
    } else {
      return GetRef<Expr>(op);
    }
  }
  Expr VisitExpr_(const Let* op) final {
    const VarExpr& v = op->var;
    if (defined_.count(v.get())) {
      Expr value = this->VisitExpr(op->value);
      VarExpr new_var = Variable::make(v.dtype(), v->name_hint);
      scope_[v.get()].push_back(new_var);
      Expr body = this->VisitExpr(op->body);
      scope_[v.get()].pop_back();
      return Let::make(new_var, value, body);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitExpr_(op);
    }
  }
  Expr VisitExpr_(const Load* op) final {
    Expr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<Load>();
    if (scope_.count(op->buffer_var.get())) {
      return Load::make(
          op->dtype, scope_[op->buffer_var.get()].back(),
          op->index, op->predicate);
    } else {
      return expr;
    }
  }
  Stmt VisitStmt_(const Store* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<Store>();
    if (scope_.count(op->buffer_var.get())) {
      return Store::make(
          scope_[op->buffer_var.get()].back(), op->value,
          op->index, op->predicate);
    } else {
      return stmt;
    }
  }
  Stmt VisitStmt_(const LetStmt* op) final {
    const VarExpr& v = op->var;
    if (defined_.count(v.get())) {
      Expr value = this->VisitExpr(op->value);
      VarExpr new_var = Variable::make(v.dtype(), v->name_hint);
      scope_[v.get()].push_back(new_var);
      Stmt body = this->VisitStmt(op->body);
      scope_[v.get()].pop_back();
      return LetStmt::make(new_var, value, body);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const For* op) final {
    const VarExpr& v = op->loop_var;
    if (defined_.count(v.get())) {
      VarExpr new_var = Variable::make(v.dtype(), v->name_hint);
      scope_[v.get()].push_back(new_var);
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      scope_[v.get()].pop_back();
      op = stmt.as<For>();
      return For::make(
          new_var, op->min, op->extent, op->for_type, op->device_api, op->body);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const Allocate* op) final {
    const VarExpr& v = op->buffer_var;
    if (defined_.count(v.get())) {
      VarExpr new_var = Variable::make(v.dtype(), v->name_hint);
      scope_[v.get()].push_back(new_var);
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      scope_[v.get()].pop_back();
      op = stmt.as<Allocate>();
      return Allocate::make(
          new_var, op->dtype, op->extents, op->condition,
          op->body, op->new_expr, op->free_function);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const AttrStmt* op) final {
    if (const Variable* v = op->node.as<Variable>()) {
      if (op->attr_key == attr::storage_scope) {
        const Allocate* alloc = op->body.as<Allocate>();
        if (alloc && op->node.same_as(alloc->buffer_var)) {
          Stmt new_alloc = this->VisitStmt(op->body);
          if (new_alloc.same_as(op->body)) return GetRef<Stmt>(op);
          alloc = new_alloc.as<Allocate>();
          CHECK(alloc);
          return AttrStmt::make(
              alloc->buffer_var, op->attr_key, op->value, new_alloc);
        }
      }
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      op = stmt.as<AttrStmt>();
      if (scope_.count(v) && scope_[v].size() != 0) {
        return AttrStmt::make(
            scope_[v].back(), op->attr_key, op->value, op->body);
      } else {
        return stmt;
      }
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

 private:
  std::unordered_map<const Variable*, std::vector<VarExpr> > scope_;
  std::unordered_set<const Variable*> defined_;
};

}  // namespace

bool VerifySSA(const Stmt& ir) {
  IRVerifySSA visitor;
  visitor(ir);
  return visitor.is_ssa;
}

Stmt ConvertSSA(Stmt stmt) {
  return IRConvertSSA()(std::move(stmt));
}

}  // namespace ir
}  // namespace tvm
