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
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/arithmetic.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace ir {

inline Expr BroadcastTo(Expr e, int lanes) {
  if (e.type().lanes() == lanes) return e;
  if (const Broadcast* op = e.as<Broadcast>()) {
    if (lanes % op->lanes == 0) {
      return Broadcast::make(op->value, lanes);
    }
  }
  CHECK_EQ(e.type().lanes(), 1)
      << "Cannot broadcast lane=" << e.type().lanes()
      << " to " << lanes;
  return Broadcast::make(e, lanes);
}

// Rewrite vectorized allocation access
// This is necessary for making each vector component containing its own workspace.
// Originates from Halide's loop vectorizer
//
// s[i] = s[i * lanes + var]
//
// The same principle applies when using one thread to simulate multiple context.
//
class VecAllocAccess : public IRMutator {
 public:
  VecAllocAccess(const Variable* buf, Var var, int var_lanes)
      : buf_(buf), var_(var), var_lanes_(var_lanes) {}
  // Load
  Expr Mutate_(const Load* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Load>();
    if (op->buffer_var.get() == buf_) {
      return Load::make(op->type, op->buffer_var,
                        op->index * var_lanes_ + var_,
                        op->predicate);
    } else {
      return expr;
    }
  }
  // Store
  Stmt Mutate_(const Store* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Store>();
    if (op->buffer_var.get() == buf_) {
      return Store::make(op->buffer_var,
                         op->value,
                         op->index * var_lanes_ + var_,
                         op->predicate);
    } else {
      return stmt;
    }
  }

 private:
  // buffer var
  const Variable* buf_;
  // variable to be replaced
  Var var_;
  // the lanes.
  int var_lanes_;
};

class Vectorizer : public IRMutator {
 public:
  Vectorizer(Var var, int var_lanes)
      : var_(var), var_lanes_(var_lanes) {
    ramp_ = Ramp::make(0, 1, var_lanes);
  }
  // user mutate from parent.
  using IRMutator::Mutate;

  Stmt Mutate(Stmt stmt) final {
    CHECK(!need_scalarize_);

    Stmt ret = IRMutator::Mutate(stmt);
    if (need_scalarize_) {
      need_scalarize_ = false;
      return Scalarize(stmt);
    } else {
      return ret;
    }
  }


  Expr Mutate_(const Add* op, const Expr &e) final {
    return AddSubVec(op, e);
  }
  Expr Mutate_(const Sub* op, const Expr &e) final {
    return AddSubVec(op, e);
  }
  Expr Mutate_(const Mul* op, const Expr &e) final {
    Expr a = this->Mutate(op->a);
    Expr b = this->Mutate(op->b);
    if (a.same_as(op->a) &&
        b.same_as(op->b)) {
      return e;
    } else {
      int lanes = std::max(a.type().lanes(), b.type().lanes());
      if (lanes != 1) {
        const Ramp* b_ramp = b.as<Ramp>();
        const Ramp* a_ramp = a.as<Ramp>();
        if (a_ramp && b.type().lanes() == 1 && analyzer_.CanProve(b > 0)) {
          return Ramp::make(
              a_ramp->base * b, a_ramp->stride * b, a_ramp->lanes);
        }
        if (b_ramp && a.type().lanes() == 1 && analyzer_.CanProve(a > 0)) {
          return Ramp::make(
              b_ramp->base * a, b_ramp->stride * a, b_ramp->lanes);
        }
      }
      return Mul::make(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
    return BinaryVec(op, e);
  }
  Expr Mutate_(const Div* op, const Expr &e) final {
    return BinaryVec(op, e);
  }
  Expr Mutate_(const Mod* op, const Expr &e) final {
    return BinaryVec(op, e);
  }
  Expr Mutate_(const FloorDiv* op, const Expr &e) final {
    return BinaryVec(op, e);
  }
  Expr Mutate_(const FloorMod* op, const Expr &e) final {
    return BinaryVec(op, e);
  }
  Expr Mutate_(const Min* op, const Expr &e) final {
    return BinaryVec(op, e);
  }
  Expr Mutate_(const Max* op, const Expr &e) final {
    return BinaryVec(op, e);
  }
  Expr Mutate_(const EQ* op, const Expr &e) final {
    return BinaryVec(op, e);
  }
  Expr Mutate_(const NE* op, const Expr &e) final {
    return BinaryVec(op, e);
  }
  Expr Mutate_(const LT* op, const Expr &e) final {
    return BinaryVec(op, e);
  }
  Expr Mutate_(const LE* op, const Expr &e) final {
    return BinaryVec(op, e);
  }
  Expr Mutate_(const GT* op, const Expr &e) final {
    return BinaryVec(op, e);
  }
  Expr Mutate_(const GE* op, const Expr &e) final {
    return BinaryVec(op, e);
  }
  Expr Mutate_(const And* op, const Expr &e) final {
    return BinaryVec(op, e);
  }
  Expr Mutate_(const Or* op, const Expr &e) final {
    return BinaryVec(op, e);
  }
  Expr Mutate_(const Ramp* op, const Expr &e) final {
    Expr base = this->Mutate(op->base);
    Expr stride = this->Mutate(op->stride);
    if (base.type().lanes() > 1 && stride.type().lanes() == 1) {
      const Ramp* base_ramp = base.as<Ramp>();
      if (analyzer_.CanProve(base_ramp->stride == stride * make_const(stride.type(), op->lanes))) {
        return Ramp::make(base_ramp->base, stride, op->lanes * base_ramp->lanes);
      }
    }
    int lanes = std::max(base.type().lanes(), stride.type().lanes());
    base = BroadcastTo(base, lanes);
    stride = BroadcastTo(stride, lanes);
    Array<Expr> elems;
    for (int i = 0; i < lanes; ++i) {
      elems.push_back(
          Ramp::make(Shuffle::make_extract_element(base, i),
                     Shuffle::make_extract_element(stride, i),
                     op->lanes));
    }
    return Shuffle::make_concat(elems);
  }
  Expr Mutate_(const Select *op, const Expr& e) final {
    Expr cond = this->Mutate(op->condition);
    Expr t = this->Mutate(op->true_value);
    Expr f = this->Mutate(op->false_value);
    if (cond.same_as(op->condition) &&
        t.same_as(op->true_value) &&
        f.same_as(op->false_value)) {
      return e;
    } else {
      int lanes = std::max(std::max(
          cond.type().lanes(),
          t.type().lanes()), f.type().lanes());
      return Select::make(cond, BroadcastTo(t, lanes), BroadcastTo(f, lanes));
    }
  }
  Expr Mutate_(const Cast *op, const Expr& e) final {
    Expr value = this->Mutate(op->value);
    if (value.same_as(op->value)) {
      return e;
    } else {
      return Cast::make(op->type.with_lanes(value.type().lanes()), value);
    }
  }
  // Variable
  Expr Mutate_(const Variable* v, const Expr& e) final {
    if (v == var_.get()) {
      return ramp_;
    } else if (lets_.count(v)) {
        return lets_[v];
    } else {
      return e;
    }
  }
  // IfThenElse expr
  Expr MutateIfThenElseExpr_(const Call *op, const Expr& e) {
    Expr cond = this->Mutate(op->args[0]);
    if (cond.type().is_vector())  {
      need_scalarize_ = true;
      return e;
    }
    Expr t = this->Mutate(op->args[1]);
    Expr f = this->Mutate(op->args[2]);
    if (cond.same_as(op->args[0]) &&
        t.same_as(op->args[1]) &&
        f.same_as(op->args[2])) {
      return e;
    } else {
      int lanes = std::max(t.type().lanes(), f.type().lanes());
      t = BroadcastTo(t, lanes);
      f = BroadcastTo(f, lanes);
      return Call::make(
          op->type.with_lanes(lanes), op->name,
          {cond, t, f}, op->call_type, op->func, op->value_index);
    }
  }
  // Call
  Expr Mutate_(const Call* op, const Expr& e) final {
    if (op->name == intrinsic::tvm_if_then_else) {
      return MutateIfThenElseExpr_(op, e);
    }
    int lane = 0;
    Array<Expr> new_args = MutateArray(op->args, &lane);

    // normal code path.
    if (op->args.same_as(new_args)) {
      return e;
    } else {
      return Call::make(
          op->type.with_lanes(lane), op->name, new_args,
          op->call_type, op->func, op->value_index);
    }
  }
  // Load
  Expr Mutate_(const Load* op, const Expr& e) final {
    Expr index = this->Mutate(op->index);
    Expr pred = this->Mutate(op->predicate);
    if (index.same_as(op->index) && pred.same_as(op->predicate)) {
      return e;
    } else {
      int lanes = std::max(index.type().lanes(), pred.type().lanes());
      return Load::make(
          op->type.with_lanes(lanes),
          op->buffer_var,
          BroadcastTo(index, lanes),
          BroadcastTo(pred, lanes));
    }
  }
  // Let
  Expr Mutate_(const Let* op, const Expr& e) final {
    Expr value = this->Mutate(op->value);
    CHECK(!lets_.count(op->var.get())) << "not SSA";
    if (value.type().lanes() != op->value.type().lanes()) {
      Var v(op->var->name_hint, value.type());
      lets_[op->var.get()] = v;
      return Let::make(v, value, Mutate(op->body));
    } else {
      Expr body = this->Mutate(op->body);
      if (value.same_as(op->value) &&
          body.same_as(op->body)) {
        return e;
      } else {
        return Let::make(op->var, value, body);
      }
    }
  }
  // Provide
  Stmt Mutate_(const Provide* op, const Stmt& s) final {
    Expr new_value = this->Mutate(op->value);
    int lane = new_value.type().lanes();
    Array<Expr> new_args = MutateArray(op->args, &lane);
    if (op->args.same_as(new_args) && op->value.same_as(new_value)) {
      return s;
    } else {
      new_value = BroadcastTo(new_value, lane);
      return Provide::make(op->func, op->value_index, new_value, new_args);
    }
  }
  // Store
  Stmt Mutate_(const Store* op, const Stmt& s) final {
    Expr value = this->Mutate(op->value);
    Expr index = this->Mutate(op->index);
    Expr pred = this->Mutate(op->predicate);
    if (value.same_as(op->value) && index.same_as(op->index)) {
      return s;
    } else {
      int lanes = std::max(value.type().lanes(), index.type().lanes());
      lanes = std::max(lanes, pred.type().lanes());
      return Store::make(op->buffer_var,
                         BroadcastTo(value, lanes),
                         BroadcastTo(index, lanes),
                         BroadcastTo(pred, lanes));
    }
  }
  // For
  Stmt Mutate_(const For* op, const Stmt& s) final {
    if (op->for_type == ForType::Vectorized) {
      LOG(WARNING) << "Detect vectorize inside vectorized loop, ignoring...";
    }
    CHECK(is_zero(op->min));
    CHECK(!op->extent.type().is_vector());
    Expr extent = Mutate(op->extent);
    if (extent.type().is_vector()) {
      LOG(WARNING) << "Detect vectorized extent type, scalarizing...";
      return Scalarize(s);
    }
    Stmt body = Mutate(op->body);
    if (extent.same_as(op->extent) &&
        body.same_as(op->body)) {
      return s;
    } else {
      return For::make(
          op->loop_var, op->min, extent,
          op->for_type, op->device_api, body);
    }
  }
  // IfThenElse
  Stmt Mutate_(const IfThenElse* op, const Stmt& s) final {
    CHECK(!op->condition.type().is_vector());
    Expr condition = this->Mutate(op->condition);
    if (condition.type().is_vector()) {
      LOG(WARNING) << "Detect vector condition in Vectorized Loop, scalarizing...";
      return Scalarize(s);
    }
    Stmt then_case = this->Mutate(op->then_case);
    Stmt else_case;
    if (op->else_case.defined()) {
      else_case = this->Mutate(op->else_case);
    }
    if (condition.same_as(op->condition) &&
        then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return s;
    } else {
      return IfThenElse::make(condition, then_case, else_case);
    }
  }
  // LetStmt
  Stmt Mutate_(const LetStmt* op, const Stmt& s) final {
    LOG(WARNING) << "Cannot vectorize with LetStmt, remove it with Simplify Before Vectorize";
    return Scalarize(s);
  }
  // Allocate
  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    if (op->new_expr.defined()) {
      LOG(WARNING) << "Cannot vectorize with new expr";
      return Scalarize(s);
    }
    Expr condition = Mutate(op->condition);
    if (condition.type().is_vector()) {
      LOG(WARNING) << "Cannot handle vector extent in alloc ";
      return Scalarize(s);
    }
    Array<Expr> extents;
    for (size_t i = 0; i < op->extents.size(); i++) {
      Expr new_ext = Mutate(op->extents[i]);
      if (new_ext.type().is_vector()) {
        LOG(WARNING) << "Cannot handle vector extent in alloc ";
        return Scalarize(s);
      }
      extents.push_back(new_ext);
    }
    // place the vector lanes in least significant dimension.
    extents.push_back(var_lanes_);
    // rewrite access to buffer internally.
    Stmt body = VecAllocAccess(
        op->buffer_var.get(), var_, var_lanes_).Mutate(op->body);
    body = Mutate(body);
    return Allocate::make(
        op->buffer_var, op->type,
        extents, condition, body,
        op->new_expr, op->free_function);
  }
  // scalarize the statment
  Stmt Scalarize(Stmt stmt) {
    Var idx(var_->name_hint + ".s", var_->type);
    Map<Var, Expr> values{{var_, idx}};
    stmt = Substitute(stmt, values);
    return For::make(idx, 0, var_lanes_, ForType::Serial, DeviceAPI::None, stmt);
  }

 private:
  // analyzer
  arith::Analyzer analyzer_;
  // variable to be replaced
  Var var_;
  // the lanes.
  int var_lanes_;
  // ramp representing the var.
  Expr ramp_;
  // flag to mark requirment of scalarization.
  bool need_scalarize_{false};
  // The lets
  std::unordered_map<const Variable*, Expr> lets_;
  // mutate array, with given lane requirement
  // when finished, p_lane updates the lane requirement.
  Array<Expr> MutateArray(Array<Expr> arr, int* p_lanes) {
    if (arr.size() == 0) return arr;
    int& lanes = *p_lanes;
    bool changed = false;
    std::vector<Expr> new_arr(arr.size());
    for (size_t i = 0; i < arr.size(); i++) {
      Expr old_elem = arr[i];
      Expr new_elem = this->Mutate(old_elem);
      if (!new_elem.same_as(old_elem)) changed = true;
      new_arr[i] = new_elem;
      lanes = std::max(lanes, new_elem.type().lanes());
    }

    for (size_t i = 0; i < arr.size(); ++i) {
      if (new_arr[i].type().lanes() != lanes) {
        new_arr[i] = BroadcastTo(new_arr[i], lanes);
        changed = true;
      }
    }
    if (!changed) return arr;
    return Array<Expr>(new_arr);
  }
  template<typename T>
  Expr BinaryVec(const T* op, const Expr& e) {
    Expr a = this->Mutate(op->a);
    Expr b = this->Mutate(op->b);
    if (a.same_as(op->a) &&
        b.same_as(op->b)) {
      return e;
    } else {
      int lanes = std::max(a.type().lanes(), b.type().lanes());
      return T::make(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
  }
  template<typename T>
  Expr AddSubVec(const T* op, const Expr& e) {
    Expr a = this->Mutate(op->a);
    Expr b = this->Mutate(op->b);
    if (a.same_as(op->a) &&
        b.same_as(op->b)) {
      return e;
    } else {
      int lanes = std::max(a.type().lanes(), b.type().lanes());
      if (lanes != 1) {
        const Ramp* b_ramp = b.as<Ramp>();
        const Ramp* a_ramp = a.as<Ramp>();
        if (a.type().lanes() == 1 && b_ramp) {
          return Ramp::make(
              arith::Compute<T>(a, b_ramp->base),
              arith::Compute<T>(make_zero(b_ramp->stride.type()), b_ramp->stride),
              b_ramp->lanes);
        }
        if (b.type().lanes() == 1 && a_ramp) {
          return Ramp::make(
              arith::Compute<T>(a_ramp->base, b), a_ramp->stride, a_ramp->lanes);
        }
      }
      return T::make(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
  }
};

class LoopVectorizer : public IRMutator {
 public:
  Stmt Mutate_(const For* op, const Stmt& s) final {
    if (op->for_type == ForType::Vectorized) {
      CHECK(is_zero(op->min));
      int lanes = 0;
      bool succ = arith::GetConstInt(op->extent, &lanes);
      if (!succ || lanes < 1) {
        LOG(FATAL) << "Failed to vectorize loop with extent " << op->extent;
      }
      Var var(op->loop_var.node_);
      return Vectorizer(var, lanes).Mutate(op->body);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }
};

Stmt VectorizeLoop(Stmt stmt) {
  return LoopVectorizer().Mutate(stmt);
}

class VectorizeSkipper : public IRMutator {
 public:
  Stmt Mutate_(const For* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<For>();
    if (op->for_type == ForType::Vectorized) {
      return For::make(op->loop_var, op->min, op->extent, ForType::Serial, op->device_api,
                       op->body);
    } else {
       return stmt;
    }
  }
};

Stmt SkipVectorize(Stmt stmt) {
  return VectorizeSkipper().Mutate(stmt);
}

}  // namespace ir
}  // namespace tvm
