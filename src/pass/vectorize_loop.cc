/*!
 *  Copyright (c) 2017 by Contributors
 *  Vectorize the loop
 * \file vectorize_loop.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace ir {

inline Expr BroadcastTo(Expr e, int lanes) {
  if (e.type().lanes() == lanes) return e;
  CHECK_EQ(e.type().lanes(), 1)
      << "Cannot broadcast lane=" << e.type().lanes()
      << " to " << lanes;
  return Broadcast::make(e, lanes);
}

// Rewrite vectorized allocation access
// s[i] = s[i * lanes + var]
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
                        op->index * var_lanes_ + var_);
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
                         op->index * var_lanes_ + var_);
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
  // override mutate
  Expr Mutate(Expr expr) final {
    static const FMutateExpr& f = Vectorizer::vtable_expr();
    return  (f.can_dispatch(expr) ?
             f(expr, expr, this) : IRMutator::Mutate(expr));
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
  // Call
  Expr Mutate_(const Call* op, const Expr& e) final {
    int lane = 0;
    Array<Expr> new_args = MutateArray(op->args, &lane);
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
    if (index.same_as(op->index)) {
      return e;
    } else {
      return Load::make(op->type.with_lanes(index.type().lanes()),
                        op->buffer_var, index);
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
    if (value.same_as(op->value) && index.same_as(op->index)) {
      return s;
    } else {
      int lanes = std::max(value.type().lanes(), index.type().lanes());
      return Store::make(op->buffer_var,
                         BroadcastTo(value, lanes),
                         BroadcastTo(index, lanes));
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
    if (else_case.defined()) {
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
    stmt = Substitute(stmt, {{var_, idx}});
    return For::make(idx, 0, var_lanes_, ForType::Serial, DeviceAPI::None, stmt);
  }
  // The overloads for vectorize.
  static FMutateExpr& vtable_expr() {  // NOLINT(*)
    static FMutateExpr inst; return inst;
  }

 private:
  // variable to be replaced
  Var var_;
  // the lanes.
  int var_lanes_;
  // ramp representing the var.
  Expr ramp_;
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
};

// binary vectorize
template<typename T>
inline Expr BinaryVec(const T* op, const Expr& e, IRMutator* m) {
  Expr a = m->Mutate(op->a);
  Expr b = m->Mutate(op->b);
  if (a.same_as(op->a) &&
      b.same_as(op->b)) {
    return e;
  } else {
    int lanes = std::max(a.type().lanes(), b.type().lanes());
    return T::make(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
  }
}

template<typename T>
inline Expr AddSubVec(const T* op, const Expr& e, IRMutator* m) {
  Expr a = m->Mutate(op->a);
  Expr b = m->Mutate(op->b);
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
            arith::ComputeExpr<T>(a, b_ramp->base), b_ramp->stride, b_ramp->lanes);
      }
      if (b.type().lanes() == 1 && a_ramp) {
        return Ramp::make(
            arith::ComputeExpr<T>(a_ramp->base, b), a_ramp->stride, a_ramp->lanes);
      }
    }
    return T::make(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
  }
}

TVM_STATIC_IR_FUNCTOR(Vectorizer, vtable_expr)
.set_dispatch<Add>(AddSubVec<Add>)
.set_dispatch<Sub>(AddSubVec<Sub>)
.set_dispatch<Mul>(BinaryVec<Mul>)
.set_dispatch<Div>(BinaryVec<Div>)
.set_dispatch<Mod>(BinaryVec<Mod>)
.set_dispatch<Min>(BinaryVec<Min>)
.set_dispatch<Max>(BinaryVec<Max>)
.set_dispatch<EQ>(BinaryVec<EQ>)
.set_dispatch<NE>(BinaryVec<NE>)
.set_dispatch<LT>(BinaryVec<LT>)
.set_dispatch<LE>(BinaryVec<LE>)
.set_dispatch<GT>(BinaryVec<GT>)
.set_dispatch<GE>(BinaryVec<GE>)
.set_dispatch<And>(BinaryVec<And>)
.set_dispatch<Or>(BinaryVec<Or>);


TVM_STATIC_IR_FUNCTOR(Vectorizer, vtable_expr)
.set_dispatch<Select>([](const Select *op, const Expr& e, IRMutator* m) {
    Expr cond = m->Mutate(op->condition);
    Expr t = m->Mutate(op->true_value);
    Expr f = m->Mutate(op->false_value);
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
  })
.set_dispatch<Cast>([](const Cast *op, const Expr& e, IRMutator* m) {
    Expr value = m->Mutate(op->value);
    if (value.same_as(op->value)) {
      return e;
    } else {
      return Cast::make(op->type.with_lanes(value.type().lanes()), value);
    }
  });


class LoopVectorizer : public IRMutator {
 public:
  Stmt Mutate_(const For* op, const Stmt& s) final {
    if (op->for_type == ForType::Vectorized) {
      CHECK(is_zero(op->min));
      CHECK(is_positive_const(op->extent));
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

}  // namespace ir
}  // namespace tvm
