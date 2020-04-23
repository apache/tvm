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
 * \file ir_util.cc
 * \brief Helper functions to construct and compose IR nodes.
 */
#include <tvm/tir/stmt_functor.h>
#include <utility>
#include <unordered_set>
#include <unordered_map>
#include "ir_util.h"

namespace tvm {
namespace tir {

Stmt MergeNest(const std::vector<Stmt>& nest, Stmt body) {
  // use reverse iteration
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    Stmt s = *ri;
    if (const auto* for_ = s.as<ForNode>()) {
      auto n = make_object<ForNode>(*for_);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* let = s.as<LetStmtNode>()) {
      auto n = make_object<LetStmtNode>(*let);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* attr = s.as<AttrStmtNode>()) {
      auto n = make_object<AttrStmtNode>(*attr);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* ite = s.as<IfThenElseNode>()) {
      auto n = make_object<IfThenElseNode>(*ite);
      CHECK(is_no_op(n->then_case));
      CHECK(!n->else_case.defined());
      n->then_case = body;
      body = Stmt(n);
    } else if (const auto* seq = s.as<SeqStmtNode>()) {
      auto n = make_object<SeqStmtNode>(*seq);
      CHECK(n->size() != 0 && is_no_op(n->seq[n->size() - 1]));
      n->seq.Set(n->size() - 1, body);
      body = Stmt(n);
    } else if (const auto* assert_ = s.as<AssertStmtNode>()) {
      auto n = make_object<AssertStmtNode>(*assert_);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* alloc = s.as<AllocateNode>()) {
      auto n = make_object<AllocateNode>(*alloc);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else {
      LOG(FATAL) << "not supported nest type";
    }
  }
  return body;
}

Stmt MergeNest(const std::vector<std::vector<Stmt>>& nest, Stmt body) {
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    body = MergeNest(*ri, body);
  }
  return body;
}


class IRConvertSSA final : public StmtExprMutator {
 public:
  PrimExpr VisitExpr_(const VarNode* op) final {
    if (scope_.count(op)) {
      return scope_[op].back();
    } else {
      return GetRef<PrimExpr>(op);
    }
  }
  PrimExpr VisitExpr_(const LetNode* op) final {
    const Var& v = op->var;
    if (defined_.count(v.get())) {
      PrimExpr value = this->VisitExpr(op->value);
      Var new_var(v->name_hint, v.dtype());
      scope_[v.get()].push_back(new_var);
      PrimExpr body = this->VisitExpr(op->body);
      scope_[v.get()].pop_back();
      return LetNode::make(new_var, value, body);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitExpr_(op);
    }
  }
  PrimExpr VisitExpr_(const LoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<LoadNode>();
    if (scope_.count(op->buffer_var.get())) {
      return LoadNode::make(
          op->dtype, scope_[op->buffer_var.get()].back(),
          op->index, op->predicate);
    } else {
      return expr;
    }
  }
  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<StoreNode>();
    if (scope_.count(op->buffer_var.get())) {
      return StoreNode::make(
          scope_[op->buffer_var.get()].back(), op->value,
          op->index, op->predicate);
    } else {
      return stmt;
    }
  }
  Stmt VisitStmt_(const LetStmtNode* op) final {
    const Var& v = op->var;
    if (defined_.count(v.get())) {
      PrimExpr value = this->VisitExpr(op->value);
      Var new_var(v->name_hint, v.dtype());
      scope_[v.get()].push_back(new_var);
      Stmt body = this->VisitStmt(op->body);
      scope_[v.get()].pop_back();
      return LetStmtNode::make(new_var, value, body);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const ForNode* op) final {
    const Var& v = op->loop_var;
    if (defined_.count(v.get())) {
      Var new_var(v->name_hint, v.dtype());
      scope_[v.get()].push_back(new_var);
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      scope_[v.get()].pop_back();
      op = stmt.as<ForNode>();
      return ForNode::make(
          new_var, op->min, op->extent, op->for_type, op->device_api, op->body);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const AllocateNode* op) final {
    const Var& v = op->buffer_var;
    if (defined_.count(v.get())) {
      Var new_var(v->name_hint, v.dtype());
      scope_[v.get()].push_back(new_var);
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      scope_[v.get()].pop_back();
      op = stmt.as<AllocateNode>();
      return AllocateNode::make(
          new_var, op->dtype, op->extents, op->condition,
          op->body);
    } else {
      defined_.insert(v.get());
      return StmtExprMutator::VisitStmt_(op);
    }
  }
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (const VarNode* v = op->node.as<VarNode>()) {
      if (op->attr_key == attr::storage_scope) {
        const AllocateNode* alloc = op->body.as<AllocateNode>();
        if (alloc && op->node.same_as(alloc->buffer_var)) {
          Stmt new_alloc = this->VisitStmt(op->body);
          if (new_alloc.same_as(op->body)) return GetRef<Stmt>(op);
          alloc = new_alloc.as<AllocateNode>();
          CHECK(alloc);
          return AttrStmtNode::make(
              alloc->buffer_var, op->attr_key, op->value, new_alloc);
        }
      }
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      op = stmt.as<AttrStmtNode>();
      if (scope_.count(v) && scope_[v].size() != 0) {
        return AttrStmtNode::make(
            scope_[v].back(), op->attr_key, op->value, op->body);
      } else {
        return stmt;
      }
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

 private:
  std::unordered_map<const VarNode*, std::vector<Var> > scope_;
  std::unordered_set<const VarNode*> defined_;
};

Stmt ConvertSSA(Stmt stmt) {
  return IRConvertSSA()(std::move(stmt));
}

}  // namespace tir
}  // namespace tvm
