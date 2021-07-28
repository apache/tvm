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
 * \file vectorize_loop_scalable.cc
 */
// Loop vectorizer as in Halide pipeline.
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tir {

inline PrimExpr BroadcastToVL(PrimExpr e, int min_num_lanes) {
  if (e.dtype().is_scalable()) return e;
  // In the VLA world a ramp is always scalable
  if (e.as<RampNode>()) {
    return e;
  }
  if (const BroadcastNode* op = e.as<BroadcastNode>()) {
    return Broadcast(op->value, min_num_lanes, true);
  }
  return Broadcast(e, min_num_lanes, true);
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
    // type_ = expr->dtype;
    if (op->buffer_var.get() == buf_) {
      return Load(op->dtype, op->buffer_var, op->index * var_lanes_ + var_, op->predicate);
    } else {
      return expr;
    }
  }
  // Store
  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<StoreNode>();
    if (op->buffer_var.get() == buf_) {
      return Store(op->buffer_var, op->value, op->index * var_lanes_ + var_, op->predicate);
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
  // The type
  // DataType type_;
};

// We use ExprFunctor directly instead of StmtExprMutator
// This is because the transformation can change the dtype of the Expr
// The existing ExprMutator transformation rules may not be well defined.
class VectorizerVLA : public StmtMutator, public ExprFunctor<PrimExpr(const PrimExpr&)> {
 public:
  using ExprFunctor::VisitExpr;
  using StmtMutator::operator();

  VectorizerVLA(Var var, PrimExpr min, int var_lanes)
      : var_(var), min_(min), var_lanes_(var_lanes) {
    // ramp_ = Ramp(var_, 1);
  }

  Stmt VisitStmt(const Stmt& stmt) final {
    ICHECK(!need_scalarize_);
    Stmt ret = StmtMutator::VisitStmt(stmt);
    if (need_scalarize_) {
      need_scalarize_ = false;
      return Scalarize(stmt);
    } else {
      return ret;
    }
  }

  PrimExpr VisitExpr(const PrimExpr& e) final { return ExprFunctor::VisitExpr(e); }

  PrimExpr VisitExpr_(const AddNode* op) final {
    return AddSubVec(op, [](PrimExpr a, PrimExpr b) { return a + b; });
  }

  PrimExpr VisitExpr_(const SubNode* op) final {
    return AddSubVec(op, [](PrimExpr a, PrimExpr b) { return a - b; });
  }

  PrimExpr VisitExpr_(const MulNode* op) final {
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) && b.same_as(op->b)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(a.dtype().lanes(), b.dtype().lanes());
      if (lanes != 1) {
        const RampNode* b_ramp = b.as<RampNode>();
        const RampNode* a_ramp = a.as<RampNode>();

        // This happens when we have a stride*i index into the tensor with stride >1
        // In this case scalarize, since for now it is not supported
        if ((a_ramp && b.dtype().lanes() == 1 && analyzer_.CanProve(b > 0)) ||
            (b_ramp && a.dtype().lanes() == 1 && analyzer_.CanProve(a > 0))) {
          need_scalarize_ = true;
          return GetRef<PrimExpr>(op);
        }
      }
      return Mul(BroadcastToVL(a, type_.lanes()), BroadcastToVL(b, type_.lanes()));
    }
    return BinaryVec<Mul>(op);
  }
  PrimExpr VisitExpr_(const DivNode* op) final { return BinaryVec<Div>(op); }
  PrimExpr VisitExpr_(const ModNode* op) final { return BinaryVec<Mod>(op); }
  PrimExpr VisitExpr_(const FloorDivNode* op) final { return BinaryVec<FloorDiv>(op); }
  PrimExpr VisitExpr_(const FloorModNode* op) final { return BinaryVec<FloorMod>(op); }
  PrimExpr VisitExpr_(const MinNode* op) final { return BinaryVec<Min>(op); }
  PrimExpr VisitExpr_(const MaxNode* op) final { return BinaryVec<Max>(op); }
  PrimExpr VisitExpr_(const EQNode* op) final { return BinaryVec<EQ>(op); }
  PrimExpr VisitExpr_(const NENode* op) final { return BinaryVec<NE>(op); }
  PrimExpr VisitExpr_(const LTNode* op) final { return BinaryVec<LT>(op); }
  PrimExpr VisitExpr_(const LENode* op) final { return BinaryVec<LE>(op); }
  PrimExpr VisitExpr_(const GTNode* op) final { return BinaryVec<GT>(op); }
  PrimExpr VisitExpr_(const GENode* op) final { return BinaryVec<GE>(op); }
  PrimExpr VisitExpr_(const AndNode* op) final { return BinaryVec<And>(op); }
  PrimExpr VisitExpr_(const OrNode* op) final { return BinaryVec<Or>(op); }

  PrimExpr VisitExpr_(const NotNode* op) final {
    PrimExpr a = this->VisitExpr(op->a);
    if (a.same_as(op->a)) {
      return GetRef<PrimExpr>(op);
    } else {
      return !(a);
    }
  }

  PrimExpr VisitExpr_(const RampNode* op) final {
    // This happens when the data tensor is a vector type. We scalarize in this
    // case
    need_scalarize_ = true;
    return GetRef<PrimExpr>(op);
  }

  PrimExpr VisitExpr_(const BroadcastNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    if (value.dtype().lanes() != 1) {
      need_scalarize_ = true;
      return GetRef<PrimExpr>(op);
    }
    if (value.same_as(op->value)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Broadcast(op->value, op->lanes);
    }
  }

  PrimExpr VisitExpr_(const SelectNode* op) final {
    PrimExpr cond = this->VisitExpr(op->condition);
    PrimExpr t = this->VisitExpr(op->true_value);
    PrimExpr f = this->VisitExpr(op->false_value);
    if (cond.same_as(op->condition) && t.same_as(op->true_value) && f.same_as(op->false_value)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Select(cond, BroadcastToVL(t, type_.lanes()), BroadcastToVL(f, type_.lanes()));
    }
  }
  PrimExpr VisitExpr_(const CastNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    if (value.same_as(op->value)) {
      return GetRef<PrimExpr>(op);
    } else {
      auto base_type = op->dtype;
      auto variable_type = DataType(base_type.code(), base_type.bits(), type_.lanes(), true);
      return Cast(variable_type, value);
    }
  }

  PrimExpr VisitExpr_(const FloatImmNode* op) final { return GetRef<PrimExpr>(op); }

  PrimExpr VisitExpr_(const IntImmNode* op) final { return GetRef<PrimExpr>(op); }

  PrimExpr VisitExpr_(const StringImmNode* op) final { return GetRef<PrimExpr>(op); }

  // Variable
  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);

    if (var.same_as(var_)) {
      return Ramp(var_, 1, type_.lanes(), true);
    }
    auto it = let_binding_.find(var);
    if (it != let_binding_.end()) {
      return it->second;
    } else {
      return std::move(var);
    }
  }
  // IfThenElse expr
  PrimExpr MutateIfThenElseExpr_(const CallNode* op) {
    PrimExpr cond = this->VisitExpr(op->args[0]);
    if (cond.dtype().is_vector()) {
      need_scalarize_ = true;
      return GetRef<PrimExpr>(op);
    }
    PrimExpr t = this->VisitExpr(op->args[1]);
    PrimExpr f = this->VisitExpr(op->args[2]);
    if (cond.same_as(op->args[0]) && t.same_as(op->args[1]) && f.same_as(op->args[2])) {
      return GetRef<PrimExpr>(op);
    } else {
      t = BroadcastToVL(t, type_.lanes());
      f = BroadcastToVL(f, type_.lanes());
      return Call(op->dtype.with_scalable_lanes(), op->op, {cond, t, f});
    }
  }
  // Call
  PrimExpr VisitExpr_(const CallNode* op) final {
    // TODO (giuseros): we could remove @tir.likely since we are using
    // predication. It is not trivial to do that here (would be simpler to do when we split)
    // but should be doable.
    if (op->op.same_as(builtin::if_then_else())) {
      type_ = op->dtype.with_scalable_lanes();
      return MutateIfThenElseExpr_(op);
    }
    auto* op_ptr = op->op.as<OpNode>();
    bool vectorizable = op_ptr && op_vectorizable_.get(GetRef<Op>(op_ptr), false);

    if (!vectorizable) {
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
        return Call(op->dtype, op->op, new_args);
      }
    } else {
      int lane = 0;
      Array<PrimExpr> new_args = MutateArray(op->args, &lane);
      // normal code path.
      if (op->args.same_as(new_args)) {
        return GetRef<PrimExpr>(op);
      } else {
        auto base_type = op->dtype;
        return Call(DataType(base_type.code(), base_type.bits(), type_.lanes(), true), op->op,
                    new_args);
      }
    }
  }
  // Load
  PrimExpr VisitExpr_(const LoadNode* op) final {
    DataType base_type = op->dtype;
    auto load_type = DataType(base_type.code(), base_type.bits(), type_.lanes(), true);
    PrimExpr index = this->VisitExpr(op->index);
    PrimExpr pred = this->VisitExpr(op->predicate);
    if (index.same_as(op->index) && pred.same_as(op->predicate)) {
      return GetRef<PrimExpr>(op);
    } else {
      // int lanes = std::max(index.dtype().lanes(), pred.dtype().lanes());
      return Load(load_type, op->buffer_var, BroadcastToVL(index, type_.lanes()),
                  BroadcastToVL(pred, type_.lanes()));
    }
  }
  // Let
  PrimExpr VisitExpr_(const LetNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    // Weaker SSA condition
    // A single var can be binded in multiple lets
    // but they have to bind to the same value.
    // This is used to allow cases when we reuse a single let
    // expression to cosntruct a nested expr.
    // (let x = 1 in x + 1) * (let x = 1 in x + 1)
    auto it = let_binding_.find(op->var);
    if (it != let_binding_.end()) {
      ICHECK(deep_equal_(it->second, value))
          << "Let cannot bind the same var to two different values";
    }
    if (value.dtype().lanes() != op->value.dtype().lanes()) {
      Var new_var(op->var->name_hint, value.dtype().with_scalable_lanes());
      let_binding_[op->var] = new_var;
      return Let(new_var, value, this->VisitExpr(op->body));
    } else {
      let_binding_[op->var] = op->var;
      PrimExpr body = this->VisitExpr(op->body);
      if (value.same_as(op->value) && body.same_as(op->body)) {
        return GetRef<PrimExpr>(op);
      } else {
        return Let(op->var, value, body);
      }
    }
  }
  // Store
  Stmt VisitStmt_(const StoreNode* op) final {
    // type_ = op->buffer_var->dtype.with_scalable_lanes();
    DataType base_type = op->value.dtype();
    type_ =
        DataType(base_type.code(), base_type.bits(), min_vector_len_bits_ / base_type.bits(), true);

    PrimExpr value = this->VisitExpr(op->value);
    //    type_ = value.dtype().with_scalable_lanes();

    PrimExpr index = this->VisitExpr(op->index);
    PrimExpr pred = this->VisitExpr(op->predicate);
    if (value.same_as(op->value) && index.same_as(op->index)) {
      return GetRef<Stmt>(op);
    } else {
      int min_lanes = type_.lanes();
      auto vla_loop_body = Store(op->buffer_var, BroadcastToVL(value, min_lanes),
                                 BroadcastToVL(index, min_lanes), BroadcastToVL(pred, min_lanes));
      if (need_loop_) {
        need_loop_ = false;
        return For(var_, min_, var_lanes_, ForKind::kSerial, vla_loop_body, NullOpt, 
                   Map<String, ObjectRef>(), Span(), true, type_.lanes());

              // TVM_DLL For(Var loop_var, PrimExpr min, PrimExpr extent, ForKind kind, Stmt body,
              // Optional<IterVar> thread_binding = NullOpt,
              // Map<String, ObjectRef> annotations = Map<String, ObjectRef>(), Span span = Span(), 
              // bool is_vla = false, int stride = 1);
      } else {
        return vla_loop_body;
      }
    }
  }
  // For
  Stmt VisitStmt_(const ForNode* op) final {
    // TODO(giuseros): Add a configuration parameter to enable
    // For loop vectorization. For VLA it boils down to have a
    // gather primitive in LLVM
    return Scalarize(GetRef<Stmt>(op));
  }

  // IfThenElse
  Stmt VisitStmt_(const IfThenElseNode* op) final {
    ICHECK(!op->condition.dtype().is_vector());
    // Evaluating then_case first to get to the data type
    Stmt then_case = this->VisitStmt(op->then_case);

    PrimExpr condition = this->VisitExpr(op->condition);
    if (condition.dtype().is_vector()) {
      return Scalarize(GetRef<Stmt>(op));
    }
    Stmt else_case;
    if (op->else_case.defined()) {
      else_case = this->VisitStmt(op->else_case);
    }
    if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return GetRef<Stmt>(op);
    } else {
      return IfThenElse(condition, then_case, else_case);
    }
  }
  // LetStmt
  Stmt VisitStmt_(const LetStmtNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    ICHECK(!let_binding_.count(op->var)) << "SSA violation, a single var is binded twice";
    let_binding_[op->var] = value;

    if (value.dtype().lanes() != op->value.dtype().lanes()) {
      Var new_var(op->var->name_hint, value.dtype().with_scalable_lanes());
      let_binding_[op->var] = new_var;
      need_loop_ = false;
      auto let_stmt = LetStmt(new_var, value, this->VisitStmt(op->body));
      return For(var_, min_, var_lanes_, ForKind::kSerial, let_stmt, NullOpt, 
                 Map<String, ObjectRef>(), Span(), true, type_.lanes());
    } else {
      let_binding_[op->var] = op->var;
      Stmt body = this->VisitStmt(op->body);
      if (value.same_as(op->value) && body.same_as(op->body)) {
        return GetRef<Stmt>(op);
      } else {
        return LetStmt(op->var, value, body);
      }
    }
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
    Stmt body = VecAllocAccess(op->buffer_var.get(), var_, var_lanes_)(op->body);
    body = this->VisitStmt(body);
    return Allocate(op->buffer_var, op->dtype, extents, condition, body);
  }

  // scalarize the statment
  Stmt Scalarize(Stmt stmt) {
    Var idx(var_->name_hint + ".s", var_->dtype);
    Map<Var, PrimExpr> values{{var_, idx}};
    stmt = Substitute(stmt, values);
    return For(idx, 0, var_lanes_, ForKind::kSerial, stmt);
  }
  // ProducerStore
  Stmt VisitStmt_(const ProducerStoreNode* op) final {
    LOG(FATAL) << "ProducerProvide is cannot appear in a TIR PrimFunc";
    return Stmt();
  }

  DataType vla_type() { return type_; }

  int extent() { return var_lanes_; }

 private:
  // analyzer
  arith::Analyzer analyzer_;
  // deep equal
  ExprDeepEqual deep_equal_;
  // variable to be replaced
  Var var_;
  // the lanes.
  PrimExpr min_;
  int var_lanes_;
  // ramp representing the var.
  PrimExpr ramp_;
  DataType type_;
  // flag to mark requirment of scalarization.
  bool need_scalarize_{false};
  bool need_loop_{true};
  // Should be configured
  int min_vector_len_bits_{128};
  int scalable_lanes_;
  // Let binding
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> let_binding_;
  // vectorizable property
  OpAttrMap<TVectorizable> op_vectorizable_ = Op::GetAttrMap<TVectorizable>("TVectorizable");

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
        new_arr[i] = BroadcastToVL(new_arr[i], type_.lanes());
        changed = true;
      }
    }
    if (!changed) return arr;
    return Array<PrimExpr>(new_arr);
  }
  template <typename TOp, typename T>
  PrimExpr BinaryVec(const T* op) {
    static_assert(std::is_same<typename TOp::ContainerType, T>::value, "constraint");
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) && b.same_as(op->b)) {
      return GetRef<PrimExpr>(op);
    } else {
      auto ba = BroadcastToVL(a, type_.lanes());
      auto bb = BroadcastToVL(b, type_.lanes());
      auto bin_op = TOp(ba, bb);
      return bin_op;
    }
  }
  template <typename T, typename FCompute>
  PrimExpr AddSubVec(const T* op, FCompute fcompute) {
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) && b.same_as(op->b)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(a.dtype().lanes(), b.dtype().lanes());
      if (lanes != 1) {
        const RampNode* b_ramp = b.as<RampNode>();
        const RampNode* a_ramp = a.as<RampNode>();
        if (a.dtype().lanes() == 1 && b_ramp) {
          PrimExpr new_stride = fcompute(make_zero(b_ramp->stride.dtype()), b_ramp->stride);

          if (analyzer_.CanProve(new_stride != 1)) {
            // TODO(giuros01): add support for gather also when stride != 1
            need_scalarize_ = true;
            return GetRef<PrimExpr>(op);
          }

          return Ramp(fcompute(a, b_ramp->base), new_stride, b_ramp->lanes, true);
        }
        if (b.dtype().lanes() == 1 && a_ramp) {
          return Ramp(fcompute(a_ramp->base, b), a_ramp->stride, a_ramp->lanes, true);
        }
      }
      return fcompute(BroadcastToVL(a, type_.lanes()), BroadcastToVL(b, type_.lanes()));
    }
  }
};

class LoopVectorizerVLA : public StmtMutator {
 public:
  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kVectorizedScalable) {
      ICHECK(is_zero(op->min));
      auto* extent_as_int = op->extent.as<IntImmNode>();
      if (!extent_as_int || extent_as_int->value < 1) {
        LOG(FATAL) << "Failed to vectorize loop with extent " << op->extent;
      }
      VectorizerVLA vla_vectorizer(op->loop_var, op->min, static_cast<int>(extent_as_int->value));
      auto vla_loop_body = vla_vectorizer(op->body);
      return vla_loop_body;
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }
};

Stmt VectorizeLoopScalable(Stmt stmt) { return LoopVectorizerVLA()(std::move(stmt)); }

class VectorizeVLASkipper : public StmtMutator {
 public:
  Stmt VisitStmt_(const ForNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<ForNode>();
    if (op->kind == ForKind::kVectorized) {
      return For(op->loop_var, op->min, op->extent, ForKind::kSerial, op->body);
    } else {
      return stmt;
    }
  }
};

Stmt SkipVectorizeScalable(Stmt stmt) { return VectorizeVLASkipper()(std::move(stmt)); }

namespace transform {

// TODO(tvm-team): Make it as a target property.
Pass VectorizeLoopScalable(bool enable_vectorize) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    if (enable_vectorize) {
      n->body = LoopVectorizerVLA()(std::move(n->body));
    } else {
      n->body = VectorizeVLASkipper()(std::move(n->body));
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.VectorizeLoopScalable", {});
}

TVM_REGISTER_GLOBAL("tir.transform.VectorizeLoopScalable").set_body_typed(VectorizeLoopScalable);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
