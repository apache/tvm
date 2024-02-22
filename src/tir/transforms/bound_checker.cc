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

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <utility>
#include <vector>

#include "../../arith/unwrap_vector_expr.h"

namespace tvm {
namespace tir {

// TODO(Lunderberg): Move this pass to be before
// StorageFlatten/FlattenBuffer.  That will simplify this pass,
// because it can check directly against the buffer limits.
class BoundCollector : public StmtVisitor {
 public:
  BoundCollector() {}

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::buffer_bound) {
      const VarNode* key = op->node.as<VarNode>();
      const CallNode* container = op->value.as<CallNode>();
      if (key && container) {
        mem_to_shape[key] = container->args;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }
  // Hashtable which maps buffer_var to shape.
  std::unordered_map<const VarNode*, Array<PrimExpr>> mem_to_shape;
};

class BoundChecker : public StmtExprMutator {
 public:
  explicit BoundChecker(const std::unordered_map<const VarNode*, Array<PrimExpr>>& mem_to_shape)
      : mem_to_shape_(mem_to_shape) {}

  Stmt VisitStmt_(const AllocateNode* op) final {
    // If the shape was updated we should update the hashtable.
    if (UpdateIsNeeded(op->buffer_var)) {
      Update(op->buffer_var, op->extents, op->dtype);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (process_store_ && op->op.same_as(builtin::if_then_else())) {
      unsafe_rewritten_ = true;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    store_scope_bound_collector_.clear();
    process_store_ = true;
    unsafe_rewritten_ = false;
    StmtExprMutator::VisitStmt_(op);
    process_store_ = false;
    if (CanInstrument(op->indices, op->buffer->data)) {
      Collect(op->indices, op->buffer->data);
    }
    // The collector should has at least one item.
    if (store_scope_bound_collector_.size()) {
      PrimExpr condition = MakeCondition();
      if (!condition.as<StringImmNode>()) {
        Stmt nop = Evaluate(1);
        Stmt then_case = GetRef<Stmt>(op);
        Stmt else_case = AssertStmt(condition, StringImm(error_message_), nop);
        Stmt body = IfThenElse(condition, then_case, else_case);
        return body;
      }
    }
    return GetRef<Stmt>(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    if (CanInstrument(op->indices, op->buffer->data)) {
      Collect(op->indices, op->buffer->data);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

 private:
  bool UpdateIsNeeded(const Var& buffer_var) const {
    return (buffer_var.defined() && mem_to_shape_.count(buffer_var.get()));
  }

  void Update(const Var& buffer_var, Array<PrimExpr> new_shape, const DataType& type) {
    // Sanity check at first.
    if (!ShapeIsValid(new_shape)) {
      return;
    }

    new_shape.MutateByApply([&](const PrimExpr& dim) {
      // Cast to uint64 to avoid potential overflow.
      return make_const(DataType::UInt(64), type.lanes()) * dim;
    });
    mem_to_shape_[buffer_var.get()] = new_shape;
  }

  bool ShapeIsValid(const Array<PrimExpr>& shape) const {
    if (!shape.defined()) {
      return false;
    }
    for (const auto& dim : shape) {
      if (!IsValidScalar(dim) || is_negative_const(dim)) {
        return false;
      }
    }

    return true;
  }

  bool IndicesAreValid(const Array<PrimExpr>& indices) const {
    if (!indices.defined()) {
      return false;
    }

    for (const auto& index : indices) {
      if (!index.defined()) {
        return false;
      }

      if (const RampNode* ramp_index = index.as<RampNode>()) {
        if (!IsValidScalar(ramp_index->base)) {
          return false;
        }
        if (!IsValidScalar(ramp_index->stride)) {
          return false;
        }
        bool lanes_int = ramp_index->lanes->IsInstance<IntImmNode>();
        if (!lanes_int) {
          return false;
        }
        int lanes = static_cast<int>(Downcast<IntImm>(ramp_index->lanes)->value);
        if (lanes <= 0) {
          return false;
        }
      }
    }
    return true;
  }

  bool IsValidScalar(const PrimExpr& expr) const {
    return expr.defined() && expr.dtype().is_scalar();
  }

  bool CanInstrument(const Array<PrimExpr>& indices, const Var& buffer_var) const {
    return buffer_var.defined() && mem_to_shape_.count(buffer_var.get()) &&
           IndicesAreValid(indices) && !unsafe_rewritten_;
  }

  void Collect(Array<PrimExpr> indices, Var buffer_var) {
    store_scope_bound_collector_.push_back(
        std::make_pair(indices, mem_to_shape_[buffer_var.get()]));
  }

  PrimExpr MakeCondition() {
    PrimExpr condition;
    for (const auto& pair : store_scope_bound_collector_) {
      Array<PrimExpr> indices = pair.first;
      Array<PrimExpr> shape = pair.second;

      ICHECK_EQ(indices.size(), shape.size())
          << "Mismatch between dimension of physical shape and physical indices";

      for (size_t i = 0; i < indices.size(); i++) {
        PrimExpr index = indices[i];
        PrimExpr upper_bound = shape[i];

        if (const RampNode* ramp_index = index.as<RampNode>()) {
          index = arith::UnwrapVectorExpr(GetRef<Ramp>(ramp_index), ramp_index->lanes);
        }

        // Try to simplify index and bound.
        index = analyzer_.Simplify(index);
        upper_bound = analyzer_.Simplify(upper_bound);

        // Cast to the same type - signed, to be able to check lower bound.
        index = Cast(DataType::Int(64), index);
        upper_bound = Cast(DataType::Int(64), upper_bound);

        // Looks like a lower bound should always be zero after normalization.
        PrimExpr lower_bound = make_zero(DataType::Int(64));

        PrimExpr current_condition = And(GE(index, lower_bound), LT(index, upper_bound));
        condition = condition.defined() ? And(condition, current_condition) : current_condition;
      }
    }
    return condition;
  }

  // Whether we process store value recursively.
  bool process_store_{false};
  // Whether we face tvm_if_then_else intrinsic.
  bool unsafe_rewritten_{false};
  // Pool which collects the pair of index and shape for specific store/load.
  std::vector<std::pair<Array<PrimExpr>, Array<PrimExpr>>> store_scope_bound_collector_;
  // Error message.
  const char* const error_message_ = "OUT OF THE BOUNDS";
  // Hashtable which maps buffer_var to shape.
  std::unordered_map<const VarNode*, Array<PrimExpr>> mem_to_shape_;
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
