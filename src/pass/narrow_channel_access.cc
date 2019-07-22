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
 *  Copyright (c) 2017 by Contributors
 * \file narrow_channel_access.cc
 * \brief Narrow channel access to a smaller range
 *  when possible by bringing it to the internal loop.
 */
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/arithmetic.h>
#include <tvm/channel.h>
#include "ir_util.h"

namespace tvm {
namespace ir {
using namespace arith;

// Bound deducer for channel access.
class ChannelAccessBound : public IRVisitor {
 public:
  ChannelAccessBound(const Variable* buf_var, bool read_access)
      : buf_var_(buf_var), read_access_(read_access) {}

  void Visit_(const Store* op) final {
    if (!read_access_ && buf_var_ == op->buffer_var.get()) {
      ret_.emplace_back(EvalSet(op->index, dom_map_));
    }
    IRVisitor::Visit_(op);
  }
  void Visit_(const For* op) final {
    CHECK(is_zero(op->min));
    // We know that the extent of the loop won't depend on relaxed scope.
    // TODO(tqchen) have a verification pass.
    dom_map_[op->loop_var.get()] = IntSet::interval(op->min, op->extent - 1);
    IRVisitor::Visit_(op);
  }
  void Visit_(const Load* op) final {
    if (read_access_ && buf_var_ == op->buffer_var.get()) {
      ret_.emplace_back(EvalSet(op->index, dom_map_));
    }
    IRVisitor::Visit_(op);
  }
  void Visit_(const Let* op) final {
    LOG(FATAL) << "cannot pass through let";
  }
  void Visit_(const LetStmt* op) final {
    LOG(FATAL) << "cannot pass through let";
  }
  IntSet Eval(const Stmt& stmt) {
    Visit(stmt);
    return Union(ret_);
  }

 private:
  // The buffer variable.
  const Variable* buf_var_;
  // read or write
  bool read_access_{true};
  // Box
  std::vector<IntSet> ret_;
  // Domain map.
  std::unordered_map<const Variable*, IntSet> dom_map_;
};

class ChannelAccessIndexRewriter : public IRMutator {
 public:
  ChannelAccessIndexRewriter(const Variable* buf_var,
                             Expr min,
                             bool read_access)
      : buf_var_(buf_var), min_(min), read_access_(read_access) {}
  Expr Mutate_(const Load* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Load>();
    if (read_access_ && buf_var_ == op->buffer_var.get()) {
      return Load::make(
          op->type, op->buffer_var, ir::Simplify(op->index - min_),
          op->predicate);
    } else {
      return expr;
    }
  }
  Stmt Mutate_(const Store* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Store>();
    if (!read_access_ && buf_var_ == op->buffer_var.get()) {
      return Store::make(
          op->buffer_var, op->value, ir::Simplify(op->index - min_),
          op->predicate);
    } else {
      return stmt;
    }
  }

 private:
  // The buffer variable.
  const Variable* buf_var_;
  // The min bound.
  Expr min_;
  // read or write
  bool read_access_{true};
};


// Rewrite channel access pattern.
class ChannelAccessRewriter : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    Stmt ret;
    const AttrStmt* adv = op->body.as<AttrStmt>();
    if ((op->attr_key == ir::attr::channel_read_scope &&
         adv && adv->attr_key == ir::attr::channel_read_advance) ||
        (op->attr_key == ir::attr::channel_write_scope &&
         adv && adv->attr_key == ir::attr::channel_write_advance)) {
      RewriteEntry e;
      e.window = op;
      e.advance = adv;
      e.read_access = op->attr_key == ir::attr::channel_read_scope;
      tasks_.push_back(e);
      ret = IRMutator::Mutate_(op, s);
      if (tasks_.back().rewrite_success) {
        ret = ret.as<AttrStmt>()->body.as<AttrStmt>()->body;
      }
      tasks_.pop_back();
      return ret;
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const For* op, const Stmt& s) final {
    std::vector<RewriteEntry> tasks;
    std::swap(tasks_, tasks);
    Stmt body = op->body;
    std::vector<Stmt> nest;
    for (RewriteEntry& e : tasks) {
      body = RewriteAccess(op, body, &e, &nest);
    }

    if (!body.same_as(op->body)) {
      body = Mutate(body);
      body = For::make(
          op->loop_var, op->min, op->extent,
          op->for_type, op->device_api, body);
      body = MergeNest(nest, body);
    } else {
      CHECK_EQ(nest.size(), 0U);
      body = IRMutator::Mutate_(op, s);
    }
    std::swap(tasks_, tasks);
    return body;
  }

 private:
  struct RewriteEntry {
    bool read_access;
    const AttrStmt* window;
    const AttrStmt* advance;
    bool rewrite_success{false};
  };

  Stmt RewriteAccess(const For* for_op,
                     Stmt body,
                     RewriteEntry* e,
                     std::vector<Stmt>* outer_nest) {
    const AttrStmt* adv_op = e->advance;
    const Expr& window = e->window->value;
    bool read_access = e->read_access;
    Var var(for_op->loop_var);
    Channel ch(adv_op->node.node_);
    ChannelAccessBound acc(ch->handle_var.get(), read_access);
    IntSet iset = acc.Eval(for_op->body);
    Range r = iset.cover_range(Range::make_by_min_extent(0, window));
    r = Range::make_by_min_extent(
        ir::Simplify(r->min), ir::Simplify(r->extent));
    if (ExprUseVar(r->extent, var)) return body;
    Array<Expr> linear_eq = DetectLinearEquation(r->min, {var});
    if (linear_eq.size() == 0) return body;
    Expr coeff = linear_eq[0];
    Expr base = linear_eq[1];
    if (!is_zero(base)) return body;
    Expr left = ir::Simplify(adv_op->value - coeff * for_op->extent);
    if (!analyzer_.CanProve(left >= 0)) return body;
    // rewrite access index.
    ChannelAccessIndexRewriter rw(
        ch->handle_var.get(), var * coeff, read_access);
    body = rw.Mutate(body);

    if (read_access) {
      body = AttrStmt::make(
          ch, ir::attr::channel_read_scope, r->extent,
          AttrStmt::make(ch, ir::attr::channel_read_advance, coeff,
                         body));
    } else {
      body = AttrStmt::make(
          ch, ir::attr::channel_write_scope, r->extent,
          AttrStmt::make(ch, ir::attr::channel_write_advance, coeff,
                         body));
    }

    if (!is_zero(left)) {
      Stmt no_op = Evaluate::make(0);
      if (read_access) {
        outer_nest->emplace_back(
            AttrStmt::make(ch, ir::attr::channel_read_advance, left, no_op));
      } else {
        outer_nest->emplace_back(
            AttrStmt::make(ch, ir::attr::channel_write_advance, left, no_op));
      }
    }

    e->rewrite_success = true;
    return body;
  }

  arith::Analyzer analyzer_;
  std::vector<RewriteEntry> tasks_;
};

Stmt NarrowChannelAccess(Stmt stmt) {
  return ChannelAccessRewriter().Mutate(stmt);
}

}  // namespace ir
}  // namespace tvm
