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

inline PrimExpr BroadcastTo(PrimExpr e, int lanes) {
  if (e.dtype().lanes() == lanes) return e;
  if (const BroadcastNode* op = e.as<BroadcastNode>()) {
    if (lanes % op->lanes == 0) {
      return Broadcast(op->value, lanes);
    }
  }
  ICHECK_EQ(e.dtype().lanes(), 1) << "Cannot broadcast lane=" << e.dtype().lanes() << " to "
                                  << lanes;
  return Broadcast(e, lanes);
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

  PrimExpr VisitExpr_(const LoadNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
    return PrimExpr();
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
    return Stmt();
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return UpdateBufferAccess(load);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return UpdateBufferAccess(store);
  }

 private:
  template <typename Node>
  Node UpdateBufferAccess(Node node) {
    // Only update the buffer that's being replaced.
    if (node->buffer->data.get() != buf_) {
      return node;
    }

    // Find/make a Buffer object with the correct updated shape.
    Buffer buf;
    auto it = buffer_map_.find(node->buffer.get());
    if (it != buffer_map_.end()) {
      buf = it->second;
    } else {
      // Extend the least significant dimension by a factor of
      // var_lanes_.  Typically, this will be a 1-d index into a flat
      // memory space.
      Array<PrimExpr> shape = node->buffer->shape;
      shape.Set(shape.size() - 1, analyzer_.Simplify(shape[shape.size() - 1] * var_lanes_));

      // TODO(Lunderberg): Move this pass to be prior to
      // StorageFlatten/FlattenBuffer, implement by appending a
      // dimension to the buffer.  Since it is currently after the
      // flattening, the strides are not technically necessary, but
      // are updated for consistency.

      // Update strides if defined.
      Array<PrimExpr> strides;
      for (size_t i = 0; i < strides.size(); i++) {
        PrimExpr stride = strides[i];
        if (i != strides.size() - 1) {
          stride *= var_lanes_;
        }
        strides.push_back(analyzer_.Simplify(stride));
      }

      // Copy everything into the new buffer.
      buf = node->buffer;
      auto buf_writer = buf.CopyOnWrite();
      buf_writer->shape = shape;
      buf_writer->strides = strides;
      buffer_map_[buf.get()] = buf;
    }

    // Extend the last index by the number of lanes in the vectorized
    // variable.
    Array<PrimExpr> indices = node->indices;
    indices.Set(indices.size() - 1,
                analyzer_.Simplify(indices[indices.size() - 1] * var_lanes_ + var_));

    auto writer = node.CopyOnWrite();
    writer->buffer = buf;
    writer->indices = indices;
    return node;
  }

  // buffer var
  const VarNode* buf_;
  // Updated buffer objects.
  std::unordered_map<const BufferNode*, Buffer> buffer_map_;
  // variable to be replaced
  Var var_;
  // the lanes.
  int var_lanes_;
  // Analyzer for simplifications
  arith::Analyzer analyzer_;
};

// We use ExprFunctor directly instead of StmtExprMutator
// This is because the transformation can change the dtype of the Expr
// The existing ExprMutator transformation rules may not be well defined.
class Vectorizer : public StmtMutator, public ExprFunctor<PrimExpr(const PrimExpr&)> {
 public:
  using ExprFunctor::VisitExpr;
  using StmtMutator::operator();

  Vectorizer(Var var, int var_lanes) : var_(var), var_lanes_(var_lanes) {
    ramp_ = Ramp(IntImm(var->dtype, 0), IntImm(var->dtype, 1), var_lanes);
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
        if (a_ramp && b.dtype().lanes() == 1 && analyzer_.CanProve(b > 0)) {
          return Ramp(a_ramp->base * b, a_ramp->stride * b, a_ramp->lanes);
        }
        if (b_ramp && a.dtype().lanes() == 1 && analyzer_.CanProve(a > 0)) {
          return Ramp(b_ramp->base * a, b_ramp->stride * a, b_ramp->lanes);
        }
      }
      return Mul(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
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
    PrimExpr base = this->VisitExpr(op->base);
    PrimExpr stride = this->VisitExpr(op->stride);
    if (base.dtype().lanes() > 1 && stride.dtype().lanes() == 1) {
      const RampNode* base_ramp = base.as<RampNode>();
      if (analyzer_.CanProve(base_ramp->stride == stride * make_const(stride.dtype(), op->lanes))) {
        return Ramp(base_ramp->base, stride, op->lanes * base_ramp->lanes);
      }
    }
    int lanes = std::max(base.dtype().lanes(), stride.dtype().lanes());
    base = BroadcastTo(base, lanes);
    stride = BroadcastTo(stride, lanes);
    Array<PrimExpr> elems;
    for (int i = 0; i < lanes; ++i) {
      elems.push_back(
          Ramp(Shuffle::ExtractElement(base, i), Shuffle::ExtractElement(stride, i), op->lanes));
    }
    return Shuffle::Concat(elems);
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
      int lanes = std::max(std::max(cond.dtype().lanes(), t.dtype().lanes()), f.dtype().lanes());
      return Select(cond, BroadcastTo(t, lanes), BroadcastTo(f, lanes));
    }
  }
  PrimExpr VisitExpr_(const CastNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    if (value.same_as(op->value)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Cast(op->dtype.with_lanes(value.dtype().lanes()), value);
    }
  }

  PrimExpr VisitExpr_(const FloatImmNode* op) final { return GetRef<PrimExpr>(op); }

  PrimExpr VisitExpr_(const IntImmNode* op) final { return GetRef<PrimExpr>(op); }

  PrimExpr VisitExpr_(const StringImmNode* op) final { return GetRef<PrimExpr>(op); }

  // Variable
  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);

    if (var.same_as(var_)) {
      return ramp_;
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
      int lanes = std::max(t.dtype().lanes(), f.dtype().lanes());
      t = BroadcastTo(t, lanes);
      f = BroadcastTo(f, lanes);
      return Call(op->dtype.with_lanes(lanes), op->op, {cond, t, f});
    }
  }
  // Call
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::if_then_else())) {
      return MutateIfThenElseExpr_(op);
    } else if (op->op.same_as(builtin::texture2d_load())) {
      int lane = 0;
      Array<PrimExpr> fcd = MutateArray({op->args.back()}, &lane);
      auto new_args = op->args;
      new_args.pop_back();
      new_args.push_back(fcd[0]);
      return Call(op->dtype.with_lanes(4), op->op, new_args);
    } else if (op->op.same_as(builtin::texture2d_store())) {
      int lane = 0;
      // Vectorize the value to store
      Array<PrimExpr> value{op->args.back()};
      Array<PrimExpr> mutated_value = MutateArray(value, &lane);
      Array<PrimExpr> new_args{op->args[0], op->args[1], op->args[2], mutated_value[0]};
      return Call(op->dtype.with_lanes(lane), op->op, new_args);
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
        return Call(op->dtype.with_lanes(lane), op->op, new_args);
      }
    }
  }
  // Load
  PrimExpr VisitExpr_(const LoadNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
    return PrimExpr();
  }
  // BufferLoad
  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto load = GetRef<BufferLoad>(op);

    auto fmutate = [this](const PrimExpr& index) { return this->VisitExpr(index); };
    Array<PrimExpr> indices = op->indices.Map(fmutate);

    if (!indices.same_as(op->indices)) {
      auto writer = load.CopyOnWrite();
      writer->indices = indices;
      writer->LegalizeDType();
    }

    return std::move(load);
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
      Var new_var(op->var->name_hint, value.dtype());
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
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
    return Stmt();
  }
  // BufferStore
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto store = GetRef<BufferStore>(op);

    auto fmutate = [this](const PrimExpr& index) { return this->VisitExpr(index); };
    Array<PrimExpr> indices = op->indices.Map(fmutate);

    PrimExpr value = this->VisitExpr(op->value);

    if (!indices.same_as(op->indices) || !value.same_as(op->value)) {
      // How many lanes of indexing are present in the index and
      // buffer element type, excluding the last index.  T
      int other_index_lanes = op->buffer->dtype.lanes();
      for (size_t i = 0; i < indices.size() - 1; i++) {
        other_index_lanes *= indices[i].dtype().lanes();
      }

      // The total number of lanes of indexing, including the last index.
      int index_lanes = other_index_lanes * indices[indices.size() - 1].dtype().lanes();

      // The total number of lanes in this store operation.  Either
      // the index or the value will be broadcast out to this number
      // of lanes, depending on which has more lanes.
      int total_lanes = std::max(index_lanes, value.dtype().lanes());

      ICHECK_EQ(total_lanes % other_index_lanes, 0)
          << "When storing to buffer " << op->buffer->name << ", cannot produce " << total_lanes
          << " lanes of storage location by changing the last index.";
      int last_index_lanes = total_lanes / other_index_lanes;

      // Broadcast the last index such that the total number of index
      // lanes matches the desired number.
      indices.Set(indices.size() - 1, BroadcastTo(indices[indices.size() - 1], last_index_lanes));

      auto writer = store.CopyOnWrite();
      writer->indices = indices;
      writer->value = BroadcastTo(value, total_lanes);
    }

    return std::move(store);
  }
  // For
  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kVectorized) {
      LOG(WARNING) << "Detect vectorize inside vectorized loop, ignoring...";
    }
    ICHECK(is_zero(op->min));
    ICHECK(!op->extent.dtype().is_vector());
    PrimExpr extent = this->VisitExpr(op->extent);
    if (extent.dtype().is_vector()) {
      return Scalarize(GetRef<Stmt>(op));
    }
    Stmt body = this->VisitStmt(op->body);
    if (extent.same_as(op->extent) && body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      return For(op->loop_var, op->min, extent, op->kind, body, op->thread_binding,
                 op->annotations);
    }
  }
  // IfThenElse
  Stmt VisitStmt_(const IfThenElseNode* op) final {
    ICHECK(!op->condition.dtype().is_vector());
    PrimExpr condition = this->VisitExpr(op->condition);
    if (condition.dtype().is_vector()) {
      return Scalarize(GetRef<Stmt>(op));
    }
    Stmt then_case = this->VisitStmt(op->then_case);
    Optional<Stmt> else_case = NullOpt;
    if (op->else_case) {
      else_case = this->VisitStmt(op->else_case.value());
    }
    if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return GetRef<Stmt>(op);
    } else {
      return IfThenElse(condition, then_case, else_case);
    }
  }
  // While
  Stmt VisitStmt_(const WhileNode* op) final {
    LOG(FATAL) << "A while loop inside a vectorized loop not supported.";
    return Stmt();
  }
  // LetStmt
  Stmt VisitStmt_(const LetStmtNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    ICHECK(!let_binding_.count(op->var)) << "SSA violation, a single var is binded twice";
    let_binding_[op->var] = value;

    if (value.dtype().lanes() != op->value.dtype().lanes()) {
      Var new_var(op->var->name_hint, value.dtype());
      let_binding_[op->var] = new_var;
      return LetStmt(new_var, value, this->VisitStmt(op->body));
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
    // Mutate the condition
    PrimExpr condition = this->VisitExpr(op->condition);
    if (condition.dtype().is_vector()) {
      LOG(WARNING) << "Cannot handle vector extent in alloc of " << op->buffer_var->name_hint;
      return Scalarize(GetRef<Stmt>(op));
    }

    // Mutate the extents
    Array<PrimExpr> extents;
    for (const auto& extent : op->extents) {
      PrimExpr new_ext = this->VisitExpr(extent);
      if (new_ext.dtype().is_vector()) {
        LOG(WARNING) << "Cannot handle vector extent in alloc of " << op->buffer_var->name_hint;
        return Scalarize(GetRef<Stmt>(op));
      }
      extents.push_back(new_ext);
    }

    // TODO(Lunderberg): Move this pass to be prior to
    // StorageFlatten/FlattenBuffer.  That will allow this pass to be
    // implemented as adding a new buffer dimension, which is later
    // flattened.

    // Extend the least significant dimension by a factor of
    // var_lanes_.  Typically, this will be a 1-d index into a flat
    // memory space.
    extents.Set(extents.size() - 1, extents[extents.size() - 1] * var_lanes_);

    // Rewrite access to the buffer in the body.
    Stmt body = VecAllocAccess(op->buffer_var.get(), var_, var_lanes_)(op->body);
    body = this->VisitStmt(body);
    return Allocate(op->buffer_var, op->dtype, extents, condition, body);
  }

  // scalarize the statment
  Stmt Scalarize(Stmt stmt) {
    Var idx(var_->name_hint + ".s", var_->dtype);
    Map<Var, PrimExpr> values{{var_, idx}};
    stmt = Substitute(stmt, values);
    return For(idx, IntImm(var_->dtype, 0), IntImm(var_->dtype, var_lanes_), ForKind::kSerial,
               stmt);
  }
  // ProducerStore
  Stmt VisitStmt_(const ProducerStoreNode* op) final {
    LOG(FATAL) << "ProducerProvide cannot appear in a TIR PrimFunc";
    return Stmt();
  }

 private:
  // analyzer
  arith::Analyzer analyzer_;
  // deep equal
  ExprDeepEqual deep_equal_;
  // variable to be replaced
  Var var_;
  // the lanes.
  int var_lanes_;
  // ramp representing the var.
  PrimExpr ramp_;
  // flag to mark requirment of scalarization.
  bool need_scalarize_{false};
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
        new_arr[i] = BroadcastTo(new_arr[i], lanes);
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
      int lanes = std::max(a.dtype().lanes(), b.dtype().lanes());
      return TOp(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
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
          return Ramp(fcompute(a, b_ramp->base),
                      fcompute(make_zero(b_ramp->stride.dtype()), b_ramp->stride), b_ramp->lanes);
        }
        if (b.dtype().lanes() == 1 && a_ramp) {
          return Ramp(fcompute(a_ramp->base, b), a_ramp->stride, a_ramp->lanes);
        }
      }
      return fcompute(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
  }
};

class LoopVectorizer : public StmtMutator {
 public:
  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kVectorized) {
      ICHECK(is_zero(op->min));
      auto* extent_as_int = op->extent.as<IntImmNode>();
      if (!extent_as_int || extent_as_int->value < 1) {
        LOG(FATAL) << "Failed to vectorize loop with extent " << op->extent;
      }
      return Vectorizer(op->loop_var, static_cast<int>(extent_as_int->value))(op->body);
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }
};

Stmt VectorizeLoop(Stmt stmt) { return LoopVectorizer()(std::move(stmt)); }

class VectorizeSkipper : public StmtMutator {
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

Stmt SkipVectorize(Stmt stmt) { return VectorizeSkipper()(std::move(stmt)); }

namespace transform {

// TODO(tvm-team): Make it as a target property.
Pass VectorizeLoop(bool enable_vectorize) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    if (enable_vectorize) {
      n->body = LoopVectorizer()(std::move(n->body));
    } else {
      n->body = VectorizeSkipper()(std::move(n->body));
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.VectorizeLoop", {});
}

TVM_REGISTER_GLOBAL("tir.transform.VectorizeLoop").set_body_typed(VectorizeLoop);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
