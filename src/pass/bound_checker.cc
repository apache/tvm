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
 *  Copyright (c) 2018 by Contributors
 * \file bounds_checker.cc
 */
// Instrument checkers for out of the bounds access.

#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <vector>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace ir {

class BoundCollector : public IRVisitor {
 public:
  BoundCollector() {}

  void Visit_(const AttrStmt *op) {
    if (op->attr_key == ir::attr::buffer_bound) {
      if (const Variable *key = op->node.as<Variable>()) {
        mem_to_shape[key] = op->value;
      }
    }
    IRVisitor::Visit_(op);
  }
  // Hashtable which maps buffer_var to shape.
  std::unordered_map<const Variable *, Expr> mem_to_shape;
};

class BoundChecker : public IRMutator {
 public:
  explicit BoundChecker(
      const std::unordered_map<const Variable *, Expr> &mem_to_shape)
      : mem_to_shape_(mem_to_shape) {}

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    // If the shape was updated we should update the hashtable.
    if (UpdateIsNeeded(op->buffer_var)) {
      Update(op->buffer_var, op->extents, op->type);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &ex) final {
    if (process_store_ && op->is_intrinsic(intrinsic::tvm_if_then_else)) {
      unsafe_rewritten_ = true;
    }
    return IRMutator::Mutate_(op, ex);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    store_scope_bound_collector_.clear();
    process_store_ = true;
    unsafe_rewritten_ = false;
    IRMutator::Mutate_(op, s);
    process_store_ = false;
    if (CanInstrument(op->index, op->buffer_var)) {
      Collect(op->index, op->buffer_var);
    }
    // The collector should has at least one item.
    if (store_scope_bound_collector_.size()) {
      Expr condition = MakeCondition();
      if (!condition.as<StringImm>()) {
        Stmt nop = Evaluate::make(1);
        Stmt then_case =
            Store::make(op->buffer_var, op->value, op->index, op->predicate);
        Stmt else_case =
            AssertStmt::make(condition, StringImm::make(error_message_), nop);
        Stmt body = IfThenElse::make(condition, then_case, else_case);
        return body;
      }
    }
    return s;
  }

  Expr Mutate_(const Load *op, const Expr &ex) final {
    if (CanInstrument(op->index, op->buffer_var)) {
      Collect(op->index, op->buffer_var);
    }
    return IRMutator::Mutate_(op, ex);
  }

 private:
  bool UpdateIsNeeded(const VarExpr &buffer_var) const {
    return (buffer_var.defined() && mem_to_shape_.count(buffer_var.get()));
  }

  void Update(const VarExpr &buffer_var, const Array<Expr> &new_shape,
              const Type &type) {
    // Sanity check at first.
    if (!new_shape.size()) {
      return;
    }

    for (size_t i = 0; i < new_shape.size(); ++i) {
      if (!new_shape[0].defined() || !new_shape[i].type().is_scalar() ||
          is_negative_const(new_shape[i])) {
        return;
      }
    }

    // Scalarize the shape.
    Expr shape = Mul::make(make_const(UInt(64), type.lanes()),
                           Cast::make(UInt(64), new_shape[0]));
    for (size_t i = 1; i < new_shape.size(); ++i) {
      // Cast to unsigned to avoid integer overlow at frist.
      shape = Mul::make(shape, Mul::make(make_const(UInt(64), type.lanes()),
                                         Cast::make(UInt(64), new_shape[i])));
    }
    mem_to_shape_[buffer_var.get()] = shape;
  }

  bool IndexIsValid(const Expr &index) const {
    if (!index.defined()) {
      return false;
    }

    if (const Ramp *ramp_index = index.as<Ramp>()) {
      return ramp_index->base.defined() &&
             ramp_index->base.type().is_scalar() &&
             ramp_index->stride.defined() &&
             ramp_index->stride.type().is_scalar() && (ramp_index->lanes > 0);
    }
    return true;
  }

  bool CanInstrument(const Expr &index, const VarExpr &buffer_var) const {
    return buffer_var.defined() && mem_to_shape_.count(buffer_var.get()) &&
           IndexIsValid(index) && !unsafe_rewritten_;
  }

  void Collect(Expr index, VarExpr buffer_var) {
    store_scope_bound_collector_.push_back(
        std::make_pair(index, mem_to_shape_[buffer_var.get()]));
  }

  Expr MakeCondition() {
    Expr condition;
    for (size_t i = 0; i < store_scope_bound_collector_.size(); ++i) {
      std::pair<Expr, Expr> buffer_to_mem = store_scope_bound_collector_[i];
      Expr index = buffer_to_mem.first;
      Expr upper_bound = buffer_to_mem.second;

      if (const Ramp *ramp_index = index.as<Ramp>()) {
        // In case index is base + stride * i.
        // Non inclusive range.
        index = Add::make(
            ramp_index->base,
            Mul::make(ramp_index->stride, make_const(ramp_index->stride.type(),
                                                     ramp_index->lanes - 1)));
      }

      // Try to simplify index and bound.
      index = ir::Simplify(index);
      upper_bound = ir::Simplify(upper_bound);

      // Cast to the same type - signed, to be able to check lower bound.
      index = Cast::make(Int(64), index);
      upper_bound = Cast::make(Int(64), upper_bound);

      // Looks like a lower bound should always be zero after normalization.
      Expr lower_bound = make_zero(Int(64));

      Expr current_condition =
          And::make(GE::make(index, lower_bound), LT::make(index, upper_bound));
      condition =
          !i ? current_condition : And::make(condition, current_condition);
    }
    return condition;
  }

  // Whether we process store value recursively.
  bool process_store_{false};
  // Whether we face tvm_if_then_else intrinsic.
  bool unsafe_rewritten_{false};
  // Pool which collects the pair of index and shape for specific store/load.
  std::vector<std::pair<Expr, Expr>> store_scope_bound_collector_;
  // Error message.
  const char *const error_message_ = "OUT OF THE BOUNDS";
  // Hashtable which maps buffer_var to shape.
  std::unordered_map<const Variable *, Expr> mem_to_shape_;
};

Stmt InstrumentBoundCheckers(Stmt stmt) {
  BoundCollector bound_collector;
  // At first walk recursively and collect bound attributes.
  bound_collector.Visit(stmt);
  return BoundChecker(bound_collector.mem_to_shape).Mutate(stmt);
}
}  // namespace ir
}  // namespace tvm
