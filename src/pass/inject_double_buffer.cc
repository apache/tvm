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
 *
 * \brief Inject double buffering optimization for data fetch.
 * \file inject_double_buffer.cc
 */
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/expr_operator.h>
#include "ir_util.h"
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace ir {

// Detect double buffer variables.
class DoubleBufferDetector : public IRVisitor {
 public:
  void Visit_(const AttrStmt* op) final {
    if (op->attr_key == attr::double_buffer_scope) {
      touched_.insert(op->node.as<Variable>());
      IRVisitor::Visit_(op);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Variable* op) final {
    if (touched_.count(op)) {
      touched_.erase(op);
    }
  }
  // The set of touched variable.
  std::unordered_set<const Variable*> touched_;
};


class StripDoubleBufferWrite : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::double_buffer_write) {
      return Mutate(op->body);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }
};

class DoubleBufferInjector : public IRMutator {
 public:
  explicit DoubleBufferInjector(int split_loop)
      : split_loop_(split_loop) {}

  Stmt Inject(const Stmt& stmt) {
    DoubleBufferDetector detector;
    detector.Visit(stmt);
    if (detector.touched_.empty()) return stmt;
    for (const Variable* v : detector.touched_) {
      dbuffer_info_[v] = StorageEntry();
    }
    return ConvertSSA(this->Mutate(stmt));
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::storage_scope) {
      const Variable* buf = op->node.as<Variable>();
      auto it = dbuffer_info_.find(buf);
      if (it != dbuffer_info_.end()) {
        it->second.scope = op->value.as<StringImm>()->value;
        return Mutate(op->body);
      } else {
        return IRMutator::Mutate_(op, s);
      }
    } else if (op->attr_key == attr::double_buffer_scope) {
      return MakeProducer(op, s);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    auto it = dbuffer_info_.find(op->buffer_var.get());
    if (it != dbuffer_info_.end()) {
      it->second.stride = arith::ComputeReduce<Mul>(
          op->extents, Expr()) * op->type.lanes();
      Stmt stmt = IRMutator::Mutate_(op, s);
      op = stmt.as<Allocate>();
      Array<Expr> new_extents{make_const(op->extents[0].type(), 2)};
      for (Expr e : op->extents) {
        new_extents.push_back(e);
      }
      CHECK(it->second.loop != nullptr);
      auto& alloc_nest = loop_allocs_[it->second.loop];
      alloc_nest.emplace_back(AttrStmt::make(
          op->buffer_var, attr::storage_scope,
          StringImm::make(it->second.scope),
          Evaluate::make(0)));
      alloc_nest.emplace_back(Allocate::make(
          op->buffer_var, op->type, new_extents, op->condition,
          Evaluate::make(0)));
      return op->body;
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const For* op, const Stmt& s) final {
    loop_nest_.push_back(op);
    Stmt stmt = IRMutator::Mutate_(op, s);
    auto it = loop_pre_.find(op);
    if (it != loop_pre_.end()) {
      const For* old_loop = stmt.as<For>();
      if (split_loop_ != 0) {
        // Explicitly unroll the loop
        CHECK(split_loop_ % 2 == 0 || split_loop_ == 1)
            << "It is better to split with multiple of 2";
        CHECK(is_zero(old_loop->min));
        Expr zero = old_loop->min;
        Expr new_ext =
            old_loop->extent - make_const(old_loop->loop_var.type(), 1);
        Expr factor = make_const(new_ext.type(), split_loop_);
        Expr outer_ext = new_ext / factor;
        Expr tail_base = outer_ext * factor;
        Var outer_var(old_loop->loop_var->name_hint + ".outer", old_loop->loop_var.type());
        std::unordered_map<const Variable*, Expr> vmap;
        std::vector<Stmt> loop_seq;
        for (int32_t i = 0; i < split_loop_; ++i) {
          vmap[old_loop->loop_var.get()] = outer_var * factor + make_const(factor.type(), i);
          loop_seq.emplace_back(Substitute(old_loop->body, vmap));
        }
        Stmt loop = For::make(
            outer_var, zero, outer_ext, old_loop->for_type, old_loop->device_api,
            MergeSeq(loop_seq));
        // tail
        std::vector<Stmt> tail_seq;
        Stmt tail_body = StripDoubleBufferWrite().Mutate(old_loop->body);
        for (int32_t i = 0; i < split_loop_; ++i) {
          Expr idx = tail_base + make_const(tail_base.type(), i);
          vmap[old_loop->loop_var.get()] = idx;
          tail_seq.emplace_back(
              IfThenElse::make(idx < old_loop->extent,
                               Substitute(tail_body, vmap)));
        }
        stmt = Block::make(loop, MergeSeq(tail_seq));
      }
      stmt = Block::make(MergeSeq(it->second), stmt);
    }
    it = loop_allocs_.find(op);
    if (it != loop_allocs_.end()) {
      stmt = MergeNest(it->second, stmt);
    }
    loop_nest_.pop_back();
    return stmt;
  }

  Stmt Mutate_(const Store* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Store>();
    auto it = dbuffer_info_.find(op->buffer_var.get());
    if (it != dbuffer_info_.end()) {
      const StorageEntry& e = it->second;
      CHECK(in_double_buffer_scope_);
      CHECK(e.stride.defined());
      return Store::make(op->buffer_var,
                         op->value,
                         e.switch_write_var * e.stride + op->index,
                         op->predicate);
    } else {
      return stmt;
    }
  }

  Expr Mutate_(const Load* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Load>();
    auto it = dbuffer_info_.find(op->buffer_var.get());
    if (it != dbuffer_info_.end()) {
      const StorageEntry& e = it->second;
      CHECK(e.stride.defined());
      CHECK(e.switch_read_var.defined());
      return Load::make(op->type,
                        op->buffer_var,
                        e.switch_read_var * e.stride + op->index,
                        op->predicate);
    } else {
      return expr;
    }
  }

  Expr Mutate_(const Variable* op, const Expr& e) final {
    CHECK(!dbuffer_info_.count(op));
    return e;
  }

 private:
  Stmt MakeProducer(const AttrStmt* op, const Stmt& s) {
    const VarExpr buffer(op->node.node_);
    CHECK_NE(loop_nest_.size(), 0U)
        << "Double buffer scope must be inside a loop";
    auto it = dbuffer_info_.find(buffer.get());
    if (it == dbuffer_info_.end()) {
      LOG(WARNING) << "Skip double buffer scope " << op->node;
      return Mutate(op->body);
    }
    StorageEntry& e = it->second;
    e.loop = loop_nest_.back();
    Expr zero = make_const(e.loop->loop_var.type(), 0);
    Expr one = make_const(e.loop->loop_var.type(), 1);
    Expr two = make_const(e.loop->loop_var.type(), 2);
    Expr loop_shift = e.loop->loop_var + one;
    e.switch_write_var = Var(e.loop->loop_var->name_hint + ".db",
                             e.loop->loop_var.type());
    e.switch_read_var = e.loop->loop_var % two;
    in_double_buffer_scope_ = true;
    Stmt body = Mutate(op->body);
    in_double_buffer_scope_ = false;
    std::unordered_map<const Variable*, Expr> vmap;
    vmap[e.switch_write_var.get()] = zero;
    vmap[e.loop->loop_var.get()] = zero;
    loop_pre_[e.loop].emplace_back(Substitute(body, vmap));
    vmap[e.loop->loop_var.get()] = loop_shift;
    vmap[e.switch_write_var.get()] = loop_shift % two;
    body = Substitute(body, vmap);
    body = AttrStmt::make(buffer, attr::double_buffer_write, 1, body);
    body = IfThenElse::make(loop_shift < e.loop->extent, body);
    return body;
  }
  // Storage entry for those who need double buffering.
  struct StorageEntry {
    // The size of the buffer
    Expr stride;
    // The loop we need
    const For* loop{nullptr};
    // The switch variable.
    VarExpr switch_write_var;
    // The switch variable for reading.
    Expr switch_read_var;
    // The storage scope.
    std::string scope;
  };
  // Whether split loop
  int32_t split_loop_;
  // Whether we are inside double buffer scope.
  bool in_double_buffer_scope_{false};
  // The current loop next
  std::vector<const For*> loop_nest_;
  // The allocs to be appended before the loop
  std::unordered_map<const For*, std::vector<Stmt> > loop_allocs_;
  // The stmt to be appended before the loop
  std::unordered_map<const For*, std::vector<Stmt> > loop_pre_;
  // The allocation size of the buffer
  std::unordered_map<const Variable*, StorageEntry> dbuffer_info_;
};


Stmt InjectDoubleBuffer(Stmt stmt, int split_loop) {
  return DoubleBufferInjector(split_loop).Inject(stmt);
}
}  // namespace ir
}  // namespace tvm
