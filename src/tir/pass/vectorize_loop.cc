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
 * \file vectorize_loop.cc
 */
// Loop vectorizer as in Halide pipeline.
#include <tvm/tir/expr.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/arith/analyzer.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "../../arith/compute_expr.h"

namespace tvm {
namespace tir {

inline PrimExpr BroadcastTo(PrimExpr e, int lanes) {
  if (e.dtype().lanes() == lanes) return e;
  if (const BroadcastNode* op = e.as<BroadcastNode>()) {
    if (lanes % op->lanes == 0) {
      return BroadcastNode::make(op->value, lanes);
    }
  }
  CHECK_EQ(e.dtype().lanes(), 1)
      << "Cannot broadcast lane=" << e.dtype().lanes()
      << " to " << lanes;
  return BroadcastNode::make(e, lanes);
}

// Rewrite vectorized allocation access
// This is necessary for making each vector component containing its own workspace.
// Originates from Halide's loop vectorizer
//
// s[i] = s[i * lanes + var]
//
// The same principle applies when using one thread to simulate multiple context.
//
class VecAllocAccess : public StmtExprMutator {
 public:
  VecAllocAccess(const VarNode* buf, Var var, int var_lanes)
      : buf_(buf), var_(var), var_lanes_(var_lanes) {}
  // Load
  PrimExpr VisitExpr_(const LoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<LoadNode>();
    if (op->buffer_var.get() == buf_) {
      return LoadNode::make(op->dtype, op->buffer_var,
                        op->index * var_lanes_ + var_,
                        op->predicate);
    } else {
      return expr;
    }
  }
  // Store
  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<StoreNode>();
    if (op->buffer_var.get() == buf_) {
      return StoreNode::make(op->buffer_var,
                         op->value,
                         op->index * var_lanes_ + var_,
                         op->predicate);
    } else {
      return stmt;
    }
  }

 private:
  // buffer var
  const VarNode* buf_;
  // variable to be replaced
  Var var_;
  // the lanes.
  int var_lanes_;
};

class Vectorizer : public StmtExprMutator {
 public:
  Vectorizer(Var var, int var_lanes)
      : var_(var), var_lanes_(var_lanes) {
    ramp_ = RampNode::make(0, 1, var_lanes);
  }

  Stmt VisitStmt(const Stmt& stmt) final {
    CHECK(!need_scalarize_);
    Stmt ret = StmtExprMutator::VisitStmt(stmt);
    if (need_scalarize_) {
      need_scalarize_ = false;
      return Scalarize(stmt);
    } else {
      return ret;
    }
  }

  PrimExpr VisitExpr_(const AddNode* op) final {
    return AddSubVec(op);
  }
  PrimExpr VisitExpr_(const SubNode* op) final {
    return AddSubVec(op);
  }
  PrimExpr VisitExpr_(const MulNode* op) final {
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) &&
        b.same_as(op->b)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(a.dtype().lanes(), b.dtype().lanes());
      if (lanes != 1) {
        const RampNode* b_ramp = b.as<RampNode>();
        const RampNode* a_ramp = a.as<RampNode>();
        if (a_ramp && b.dtype().lanes() == 1 && analyzer_.CanProve(b > 0)) {
          return RampNode::make(
              a_ramp->base * b, a_ramp->stride * b, a_ramp->lanes);
        }
        if (b_ramp && a.dtype().lanes() == 1 && analyzer_.CanProve(a > 0)) {
          return RampNode::make(
              b_ramp->base * a, b_ramp->stride * a, b_ramp->lanes);
        }
      }
      return MulNode::make(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
    return BinaryVec(op);
  }
  PrimExpr VisitExpr_(const DivNode* op) final {
    return BinaryVec(op);
  }
  PrimExpr VisitExpr_(const ModNode* op) final {
    return BinaryVec(op);
  }
  PrimExpr VisitExpr_(const FloorDivNode* op) final {
    return BinaryVec(op);
  }
  PrimExpr VisitExpr_(const FloorModNode* op) final {
    return BinaryVec(op);
  }
  PrimExpr VisitExpr_(const MinNode* op) final {
    return BinaryVec(op);
  }
  PrimExpr VisitExpr_(const MaxNode* op) final {
    return BinaryVec(op);
  }
  PrimExpr VisitExpr_(const EQNode* op) final {
    return BinaryVec(op);
  }
  PrimExpr VisitExpr_(const NENode* op) final {
    return BinaryVec(op);
  }
  PrimExpr VisitExpr_(const LTNode* op) final {
    return BinaryVec(op);
  }
  PrimExpr VisitExpr_(const LENode* op) final {
    return BinaryVec(op);
  }
  PrimExpr VisitExpr_(const GTNode* op) final {
    return BinaryVec(op);
  }
  PrimExpr VisitExpr_(const GENode* op) final {
    return BinaryVec(op);
  }
  PrimExpr VisitExpr_(const AndNode* op) final {
    return BinaryVec(op);
  }
  PrimExpr VisitExpr_(const OrNode* op) final {
    return BinaryVec(op);
  }
  PrimExpr VisitExpr_(const RampNode* op) final {
    PrimExpr base = this->VisitExpr(op->base);
    PrimExpr stride = this->VisitExpr(op->stride);
    if (base.dtype().lanes() > 1 && stride.dtype().lanes() == 1) {
      const RampNode* base_ramp = base.as<RampNode>();
      if (analyzer_.CanProve(base_ramp->stride == stride * make_const(stride.dtype(), op->lanes))) {
        return RampNode::make(base_ramp->base, stride, op->lanes * base_ramp->lanes);
      }
    }
    int lanes = std::max(base.dtype().lanes(), stride.dtype().lanes());
    base = BroadcastTo(base, lanes);
    stride = BroadcastTo(stride, lanes);
    Array<PrimExpr> elems;
    for (int i = 0; i < lanes; ++i) {
      elems.push_back(
          RampNode::make(ShuffleNode::make_extract_element(base, i),
                     ShuffleNode::make_extract_element(stride, i),
                     op->lanes));
    }
    return ShuffleNode::make_concat(elems);
  }
  PrimExpr VisitExpr_(const SelectNode *op) final {
    PrimExpr cond = this->VisitExpr(op->condition);
    PrimExpr t = this->VisitExpr(op->true_value);
    PrimExpr f = this->VisitExpr(op->false_value);
    if (cond.same_as(op->condition) &&
        t.same_as(op->true_value) &&
        f.same_as(op->false_value)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(std::max(
          cond.dtype().lanes(),
          t.dtype().lanes()), f.dtype().lanes());
      return SelectNode::make(cond, BroadcastTo(t, lanes), BroadcastTo(f, lanes));
    }
  }
  PrimExpr VisitExpr_(const CastNode *op) final {
    PrimExpr value = this->VisitExpr(op->value);
    if (value.same_as(op->value)) {
      return GetRef<PrimExpr>(op);
    } else {
      return CastNode::make(op->dtype.with_lanes(value.dtype().lanes()), value);
    }
  }
  // Variable
  PrimExpr VisitExpr_(const VarNode* v) final {
    if (v == var_.get()) {
      return ramp_;
    } else if (lets_.count(v)) {
        return lets_[v];
    } else {
      return GetRef<PrimExpr>(v);
    }
  }
  // IfThenElse expr
  PrimExpr MutateIfThenElseExpr_(const CallNode *op) {
    PrimExpr cond = this->VisitExpr(op->args[0]);
    if (cond.dtype().is_vector())  {
      need_scalarize_ = true;
      return GetRef<PrimExpr>(op);
    }
    PrimExpr t = this->VisitExpr(op->args[1]);
    PrimExpr f = this->VisitExpr(op->args[2]);
    if (cond.same_as(op->args[0]) &&
        t.same_as(op->args[1]) &&
        f.same_as(op->args[2])) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(t.dtype().lanes(), f.dtype().lanes());
      t = BroadcastTo(t, lanes);
      f = BroadcastTo(f, lanes);
      return CallNode::make(
          op->dtype.with_lanes(lanes), op->name,
          {cond, t, f}, op->call_type, op->func, op->value_index);
    }
  }
  // Call
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->name == intrinsic::tvm_if_then_else) {
      return MutateIfThenElseExpr_(op);
    }
    if (!op->is_vectorizable()) {
      // Cannot vectorize this op
      Array<PrimExpr> new_args;
      for (auto arg : op->args) {
        auto new_arg = this->VisitExpr(arg);
        if (new_arg.dtype().is_vector()) {
          need_scalarize_ = true;
          return GetRef<PrimExpr>(op);
        }
        new_args.push_back(new_arg);
      }
      if (op->args.same_as(new_args)) {
        return GetRef<PrimExpr>(op);
      } else {
        return CallNode::make(
            op->dtype, op->name, new_args, op->call_type, op->func, op->value_index);
      }
    } else {
      int lane = 0;
      Array<PrimExpr> new_args = MutateArray(op->args, &lane);
      // normal code path.
      if (op->args.same_as(new_args)) {
        return GetRef<PrimExpr>(op);
      } else {
        return CallNode::make(
            op->dtype.with_lanes(lane), op->name, new_args,
            op->call_type, op->func, op->value_index);
      }
    }
  }
  // Load
  PrimExpr VisitExpr_(const LoadNode* op) final {
    PrimExpr index = this->VisitExpr(op->index);
    PrimExpr pred = this->VisitExpr(op->predicate);
    if (index.same_as(op->index) && pred.same_as(op->predicate)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(index.dtype().lanes(), pred.dtype().lanes());
      return LoadNode::make(
          op->dtype.with_lanes(lanes),
          op->buffer_var,
          BroadcastTo(index, lanes),
          BroadcastTo(pred, lanes));
    }
  }
  // Let
  PrimExpr VisitExpr_(const LetNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    CHECK(!lets_.count(op->var.get())) << "not SSA";
    if (value.dtype().lanes() != op->value.dtype().lanes()) {
      Var v(op->var->name_hint, value.dtype());
      lets_[op->var.get()] = v;
      return LetNode::make(v, value, this->VisitExpr(op->body));
    } else {
      PrimExpr body = this->VisitExpr(op->body);
      if (value.same_as(op->value) &&
          body.same_as(op->body)) {
        return GetRef<PrimExpr>(op);
      } else {
        return LetNode::make(op->var, value, body);
      }
    }
  }
  // Provide
  Stmt VisitStmt_(const ProvideNode* op) final {
    PrimExpr new_value = this->VisitExpr(op->value);
    int lane = new_value.dtype().lanes();
    Array<PrimExpr> new_args = MutateArray(op->args, &lane);
    if (op->args.same_as(new_args) && op->value.same_as(new_value)) {
      return GetRef<Stmt>(op);
    } else {
      new_value = BroadcastTo(new_value, lane);
      return ProvideNode::make(op->func, op->value_index, new_value, new_args);
    }
  }
  // Store
  Stmt VisitStmt_(const StoreNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    PrimExpr index = this->VisitExpr(op->index);
    PrimExpr pred = this->VisitExpr(op->predicate);
    if (value.same_as(op->value) && index.same_as(op->index)) {
      return GetRef<Stmt>(op);
    } else {
      int lanes = std::max(value.dtype().lanes(), index.dtype().lanes());
      lanes = std::max(lanes, pred.dtype().lanes());
      return StoreNode::make(op->buffer_var,
                         BroadcastTo(value, lanes),
                         BroadcastTo(index, lanes),
                         BroadcastTo(pred, lanes));
    }
  }
  // For
  Stmt VisitStmt_(const ForNode* op) final {
    if (op->for_type == ForType::Vectorized) {
      LOG(WARNING) << "Detect vectorize inside vectorized loop, ignoring...";
    }
    CHECK(is_zero(op->min));
    CHECK(!op->extent.dtype().is_vector());
    PrimExpr extent = this->VisitExpr(op->extent);
    if (extent.dtype().is_vector()) {
      return Scalarize(GetRef<Stmt>(op));
    }
    Stmt body = this->VisitStmt(op->body);
    if (extent.same_as(op->extent) &&
        body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      return ForNode::make(
          op->loop_var, op->min, extent,
          op->for_type, op->device_api, body);
    }
  }
  // IfThenElse
  Stmt VisitStmt_(const IfThenElseNode* op) final {
    CHECK(!op->condition.dtype().is_vector());
    PrimExpr condition = this->VisitExpr(op->condition);
    if (condition.dtype().is_vector()) {
      return Scalarize(GetRef<Stmt>(op));
    }
    Stmt then_case = this->VisitStmt(op->then_case);
    Stmt else_case;
    if (op->else_case.defined()) {
      else_case = this->VisitStmt(op->else_case);
    }
    if (condition.same_as(op->condition) &&
        then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return GetRef<Stmt>(op);
    } else {
      return IfThenElseNode::make(condition, then_case, else_case);
    }
  }
  // LetStmt
  Stmt VisitStmt_(const LetStmtNode* op) final {
    LOG(WARNING) << "Cannot vectorize with LetStmt, remove it with Simplify Before Vectorize";
    return Scalarize(GetRef<Stmt>(op));
  }
  // Allocate
  Stmt VisitStmt_(const AllocateNode* op) final {
    PrimExpr condition = this->VisitExpr(op->condition);
    if (condition.dtype().is_vector()) {
      LOG(WARNING) << "Cannot handle vector extent in alloc ";
      return Scalarize(GetRef<Stmt>(op));
    }
    Array<PrimExpr> extents;
    for (size_t i = 0; i < op->extents.size(); i++) {
      PrimExpr new_ext = this->VisitExpr(op->extents[i]);
      if (new_ext.dtype().is_vector()) {
        LOG(WARNING) << "Cannot handle vector extent in alloc ";
        return Scalarize(GetRef<Stmt>(op));
      }
      extents.push_back(new_ext);
    }
    // place the vector lanes in least significant dimension.
    extents.push_back(var_lanes_);
    // rewrite access to buffer internally.
    Stmt body = VecAllocAccess(
        op->buffer_var.get(), var_, var_lanes_)(op->body);
    body = this->VisitStmt(body);
    return AllocateNode::make(
        op->buffer_var, op->dtype,
        extents, condition, body);
  }
  // scalarize the statment
  Stmt Scalarize(Stmt stmt) {
    Var idx(var_->name_hint + ".s", var_->dtype);
    Map<Var, PrimExpr> values{{var_, idx}};
    stmt = Substitute(stmt, values);
    return ForNode::make(idx, 0, var_lanes_, ForType::Serial, DeviceAPI::None, stmt);
  }

 private:
  // analyzer
  arith::Analyzer analyzer_;
  // variable to be replaced
  Var var_;
  // the lanes.
  int var_lanes_;
  // ramp representing the var.
  PrimExpr ramp_;
  // flag to mark requirment of scalarization.
  bool need_scalarize_{false};
  // The lets
  std::unordered_map<const VarNode*, PrimExpr> lets_;
  // mutate array, with given lane requirement
  // when finished, p_lane updates the lane requirement.
  Array<PrimExpr> MutateArray(Array<PrimExpr> arr, int* p_lanes) {
    if (arr.size() == 0) return arr;
    int& lanes = *p_lanes;
    bool changed = false;
    std::vector<PrimExpr> new_arr(arr.size());
    for (size_t i = 0; i < arr.size(); i++) {
      PrimExpr old_elem = arr[i];
      PrimExpr new_elem = this->VisitExpr(old_elem);
      if (!new_elem.same_as(old_elem)) changed = true;
      new_arr[i] = new_elem;
      lanes = std::max(lanes, new_elem.dtype().lanes());
    }

    for (size_t i = 0; i < arr.size(); ++i) {
      if (new_arr[i].dtype().lanes() != lanes) {
        new_arr[i] = BroadcastTo(new_arr[i], lanes);
        changed = true;
      }
    }
    if (!changed) return arr;
    return Array<PrimExpr>(new_arr);
  }
  template<typename T>
  PrimExpr BinaryVec(const T* op) {
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) &&
        b.same_as(op->b)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(a.dtype().lanes(), b.dtype().lanes());
      return T::make(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
  }
  template<typename T>
  PrimExpr AddSubVec(const T* op) {
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) &&
        b.same_as(op->b)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(a.dtype().lanes(), b.dtype().lanes());
      if (lanes != 1) {
        const RampNode* b_ramp = b.as<RampNode>();
        const RampNode* a_ramp = a.as<RampNode>();
        if (a.dtype().lanes() == 1 && b_ramp) {
          return RampNode::make(
              arith::Compute<T>(a, b_ramp->base),
              arith::Compute<T>(make_zero(b_ramp->stride.dtype()), b_ramp->stride),
              b_ramp->lanes);
        }
        if (b.dtype().lanes() == 1 && a_ramp) {
          return RampNode::make(
              arith::Compute<T>(a_ramp->base, b), a_ramp->stride, a_ramp->lanes);
        }
      }
      return T::make(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
  }
};

class LoopVectorizer : public StmtMutator {
 public:
  Stmt VisitStmt_(const ForNode* op) final {
    if (op->for_type == ForType::Vectorized) {
      CHECK(is_zero(op->min));
      int lanes = 0;
      bool succ = arith::GetConstInt(op->extent, &lanes);
      if (!succ || lanes < 1) {
        LOG(FATAL) << "Failed to vectorize loop with extent " << op->extent;
      }
      return Vectorizer(op->loop_var, lanes)(op->body);
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }
};

Stmt VectorizeLoop(Stmt stmt) {
  return LoopVectorizer()(std::move(stmt));
}

class VectorizeSkipper : public StmtMutator {
 public:
  Stmt VisitStmt_(const ForNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<ForNode>();
    if (op->for_type == ForType::Vectorized) {
      return ForNode::make(op->loop_var, op->min, op->extent, ForType::Serial, op->device_api,
                       op->body);
    } else {
       return stmt;
    }
  }
};

Stmt SkipVectorize(Stmt stmt) {
  return VectorizeSkipper()(std::move(stmt));
}

}  // namespace tir
}  // namespace tvm
