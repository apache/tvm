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
 * \file bounds_checker.cc
 */
// Instrument checkers for out of the bounds access.

#include <tvm/runtime/registry.h>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include <vector>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace tir {

class BoundCollector : public StmtVisitor {
 public:
  BoundCollector() {}

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::buffer_bound) {
      if (const VarNode *key = op->node.as<VarNode>()) {
        mem_to_shape[key] = op->value;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }
  // Hashtable which maps buffer_var to shape.
  std::unordered_map<const VarNode *, PrimExpr> mem_to_shape;
};

class BoundChecker : public StmtExprMutator {
 public:
  explicit BoundChecker(
      const std::unordered_map<const VarNode *, PrimExpr> &mem_to_shape)
      : mem_to_shape_(mem_to_shape) {}

  Stmt VisitStmt_(const AllocateNode* op) final {
    // If the shape was updated we should update the hashtable.
    if (UpdateIsNeeded(op->buffer_var)) {
      Update(op->buffer_var, op->extents, op->dtype);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (process_store_ && op->is_intrinsic(intrinsic::tvm_if_then_else)) {
      unsafe_rewritten_ = true;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    store_scope_bound_collector_.clear();
    process_store_ = true;
    unsafe_rewritten_ = false;
    StmtExprMutator::VisitStmt_(op);
    process_store_ = false;
    if (CanInstrument(op->index, op->buffer_var)) {
      Collect(op->index, op->buffer_var);
    }
    // The collector should has at least one item.
    if (store_scope_bound_collector_.size()) {
      PrimExpr condition = MakeCondition();
      if (!condition.as<StringImmNode>()) {
        Stmt nop = EvaluateNode::make(1);
        Stmt then_case =
            StoreNode::make(op->buffer_var, op->value, op->index, op->predicate);
        Stmt else_case =
            AssertStmtNode::make(condition, StringImmNode::make(error_message_), nop);
        Stmt body = IfThenElseNode::make(condition, then_case, else_case);
        return body;
      }
    }
    return GetRef<Stmt>(op);
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    if (CanInstrument(op->index, op->buffer_var)) {
      Collect(op->index, op->buffer_var);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

 private:
  bool UpdateIsNeeded(const Var& buffer_var) const {
    return (buffer_var.defined() && mem_to_shape_.count(buffer_var.get()));
  }

  void Update(const Var& buffer_var,
              const Array<PrimExpr>& new_shape,
              const DataType& type) {
    // Sanity check at first.
    if (!new_shape.size()) {
      return;
    }

    for (size_t i = 0; i < new_shape.size(); ++i) {
      if (!new_shape[0].defined() || !new_shape[i].dtype().is_scalar() ||
          is_negative_const(new_shape[i])) {
        return;
      }
    }

    // Scalarize the shape.
    PrimExpr shape = MulNode::make(make_const(DataType::UInt(64), type.lanes()),
                           CastNode::make(DataType::UInt(64), new_shape[0]));
    for (size_t i = 1; i < new_shape.size(); ++i) {
      // Cast to unsigned to avoid integer overlow at frist.
      shape = MulNode::make(shape, MulNode::make(make_const(DataType::UInt(64), type.lanes()),
                                         CastNode::make(DataType::UInt(64), new_shape[i])));
    }
    mem_to_shape_[buffer_var.get()] = shape;
  }

  bool IndexIsValid(const PrimExpr& index) const {
    if (!index.defined()) {
      return false;
    }

    if (const RampNode *ramp_index = index.as<RampNode>()) {
      return ramp_index->base.defined() &&
             ramp_index->base.dtype().is_scalar() &&
             ramp_index->stride.defined() &&
             ramp_index->stride.dtype().is_scalar() && (ramp_index->lanes > 0);
    }
    return true;
  }

  bool CanInstrument(const PrimExpr& index, const Var& buffer_var) const {
    return buffer_var.defined() && mem_to_shape_.count(buffer_var.get()) &&
           IndexIsValid(index) && !unsafe_rewritten_;
  }

  void Collect(PrimExpr index, Var buffer_var) {
    store_scope_bound_collector_.push_back(
        std::make_pair(index, mem_to_shape_[buffer_var.get()]));
  }

  PrimExpr MakeCondition() {
    PrimExpr condition;
    for (size_t i = 0; i < store_scope_bound_collector_.size(); ++i) {
      std::pair<PrimExpr, PrimExpr> buffer_to_mem = store_scope_bound_collector_[i];
      PrimExpr index = buffer_to_mem.first;
      PrimExpr upper_bound = buffer_to_mem.second;

      if (const RampNode *ramp_index = index.as<RampNode>()) {
        // In case index is base + stride * i.
        // Non inclusive range.
        index = AddNode::make(
            ramp_index->base,
            MulNode::make(ramp_index->stride, make_const(ramp_index->stride.dtype(),
                                                     ramp_index->lanes - 1)));
      }

      // Try to simplify index and bound.
      index = analyzer_.Simplify(index);
      upper_bound = analyzer_.Simplify(upper_bound);

      // Cast to the same type - signed, to be able to check lower bound.
      index = CastNode::make(DataType::Int(64), index);
      upper_bound = CastNode::make(DataType::Int(64), upper_bound);

      // Looks like a lower bound should always be zero after normalization.
      PrimExpr lower_bound = make_zero(DataType::Int(64));

      PrimExpr current_condition =
          AndNode::make(GENode::make(index, lower_bound), LTNode::make(index, upper_bound));
      condition =
          !i ? current_condition : AndNode::make(condition, current_condition);
    }
    return condition;
  }

  // Whether we process store value recursively.
  bool process_store_{false};
  // Whether we face tvm_if_then_else intrinsic.
  bool unsafe_rewritten_{false};
  // Pool which collects the pair of index and shape for specific store/load.
  std::vector<std::pair<PrimExpr, PrimExpr>> store_scope_bound_collector_;
  // Error message.
  const char *const error_message_ = "OUT OF THE BOUNDS";
  // Hashtable which maps buffer_var to shape.
  std::unordered_map<const VarNode *, PrimExpr> mem_to_shape_;
  // internal analyzer
  arith::Analyzer analyzer_;
};

Stmt InstrumentBoundCheckers(Stmt stmt) {
  BoundCollector bound_collector;
  // At first walk recursively and collect bound attributes.
  bound_collector(stmt);
  return BoundChecker(bound_collector.mem_to_shape)(std::move(stmt));
}

namespace transform {

Pass InstrumentBoundCheckers() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    BoundCollector bound_collector;
    // At first walk recursively and collect bound attributes.
    bound_collector(n->body);
    n->body = BoundChecker(bound_collector.mem_to_shape)(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InstrumentBoundCheckers", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InstrumentBoundCheckers")
.set_body_typed(InstrumentBoundCheckers);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
