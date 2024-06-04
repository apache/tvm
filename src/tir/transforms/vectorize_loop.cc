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
#include <vector>

#include "../../src/arith/scalable_expression.h"
#include "../../tir/analysis/check_contains.h"

namespace tvm {
namespace tir {

inline PrimExpr CreateNewLanes(bool is_scalable, int lanes_or_vscale_factor) {
  if (is_scalable) {
    return Mul(Call(DataType::Int(32), builtin::vscale(), {}), lanes_or_vscale_factor);
  } else {
    return lanes_or_vscale_factor;
  }
}

inline PrimExpr BroadcastTo(PrimExpr e, int lanes, bool is_scalable) {
  // Check if e is already in the expected form
  if (e.dtype().get_lanes_or_vscale_factor() == lanes &&
      e.dtype().is_scalable_vector() == is_scalable)
    return e;

  if (const BroadcastNode* op = e.as<BroadcastNode>()) {
    ICHECK(op->dtype.is_scalable_vector() == is_scalable)
        << "Can't broadcast between scalable and fixed length vectors.";
    int e_lanes = op->dtype.get_lanes_or_vscale_factor();

    if (lanes % e_lanes == 0) {
      return Broadcast(op->value, CreateNewLanes(is_scalable, lanes));
    }
  }

  ICHECK(e.dtype().is_scalar()) << "Cannot broadcast lanes="
                                << e.dtype().get_lanes_or_vscale_factor()
                                << " is_scalable=" << e.dtype().is_scalable_vector() << " to "
                                << lanes;

  return Broadcast(e, CreateNewLanes(is_scalable, lanes));
}

bool EnableBufferLevelPredication(Target target) {
  transform::PassContext pass_ctx = transform::PassContext::Current();
  Optional<Bool> enable_buffer_predication =
      pass_ctx->GetConfig<Bool>("tir.enable_buffer_level_predication");
  if (enable_buffer_predication.defined()) {
    return enable_buffer_predication.value();
  }

  // Use buffer-level predication by default for AArch64 SVE targets
  return arith::TargetHasSVE(target);
}

/*!
 * \brief A pass that tries to rewrite buffer accesses (loads and stores) with a
 * predicate expression where possible.
 *
 * \note For now we start with a minimal case targeting block-level predicates
 * produced by the split schedule primitive, with the potential for predicating
 * more complex terms in the future if needed.
 *
 * \example
 * Before:
 * for i_0 in T.serial(4):
 *     for i_1 in T.vectorized(4):
 *         if i_0 * 4 + i_1 < 14:
 *             B[i_0 * 4 + i_1] = A[i_0 * 4 + i_1] + 1.0
 *
 * After:
 * for i_0 in T.serial(4):
 *  predicate = T.get_active_lane_mask("uint1x4", i_0 * 4, 14)
 *  A_load = T.meta_var(A.vload([T.Ramp(i_0 * 4, 1, 4)], predicate=predicate))
 *  B.vstore([T.Ramp(i_0 * 4, 1, 4)], A_load, predicate=predicate)
 */
class TryPredicateBufferAccesses : public StmtExprMutator {
 public:
  TryPredicateBufferAccesses() {}

  /*!
   * \brief Run the pass to try to exact predicates.
   * \param stmt - The statement containing buffer accesses (loads and stores)
   * we want to attempt to predicate.
   * \param condition - The conditional expression (block-level predicate)
   * that we will try to remove.
   * \return pair<success, stmt> - Boolean value for success/failure, the rewritten
   * stmt if successful.
   */
  std::pair<bool, Stmt> Run(Stmt stmt, PrimExpr condition) {
    // Check that the condition provided is of the form a < b, for now.
    if (!condition->IsInstance<LTNode>()) {
      return {false, stmt};
    }

    LT lt = Downcast<LT>(condition);

    // Check the form of the vectorized condition, we're expecting
    // Ramp(...) < Broadcast(...)
    if (!lt->a->IsInstance<RampNode>() || !lt->b->IsInstance<BroadcastNode>()) {
      return {false, stmt};
    }

    base_ = Downcast<Ramp>(lt->a)->base;
    limit_ = Downcast<Broadcast>(lt->b)->value;

    // Now we can try to predicate
    Stmt predicated_stmt = StmtExprMutator::operator()(std::move(stmt));
    if (num_accesses_analyzed_ > 0 && num_accesses_analyzed_ == num_accesses_rewritten_) {
      return {true, predicated_stmt};
    }
    return {false, stmt};
  }

 private:
  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return TryPredicateBufferAccess(load);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return TryPredicateBufferAccess(store);
  }

  template <typename AccessNode>
  AccessNode TryPredicateBufferAccess(AccessNode node) {
    num_accesses_analyzed_ += 1;

    // Do not try to predicate non-vectorized accesses
    Array<PrimExpr> indices = node->indices;
    if (!indices.size() || !indices[0]->IsInstance<RampNode>()) {
      return node;
    }
    Ramp ramp = Downcast<Ramp>(node->indices[0]);

    // The vectorized access pattern must match the base of the predicate
    if (!tvm::StructuralEqual()(ramp->base, base_)) {
      return node;
    }

    DataType buf_predicate_dtype =
        DataType(DataType::kUInt, 1, ramp->dtype.get_lanes_or_vscale_factor(),
                 ramp->dtype.is_scalable_vector());
    Call lane_mask = Call(buf_predicate_dtype, builtin::get_active_lane_mask(), {base_, limit_});

    num_accesses_rewritten_ += 1;
    auto writer = node.CopyOnWrite();
    writer->predicate = lane_mask;
    return node;
  }

  /*! \brief The variable base expr of the predicate. */
  PrimExpr base_;
  /*! \brief The limit of the predicate. The expr specifies the upper bound of the base's
   * evaluated value. */
  PrimExpr limit_;
  /*! \brief The number of buffer accesses in the stmt we will analyze. */
  size_t num_accesses_analyzed_ = 0;
  /*! \brief The number of buffer accesses rewritten with predicates. */
  size_t num_accesses_rewritten_ = 0;
};

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
  VecAllocAccess(const VarNode* buf, Var var, PrimExpr var_lanes)
      : buf_(buf), var_(var), var_lanes_(var_lanes) {}

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
  PrimExpr var_lanes_;
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

  Vectorizer(Var var, PrimExpr var_lanes, Target target)
      : var_(var), var_lanes_(var_lanes), target_(target) {
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
      bool is_vec_a = a.dtype().is_scalable_or_fixed_length_vector();
      bool is_vec_b = b.dtype().is_scalable_or_fixed_length_vector();
      if (is_vec_a && is_vec_b) {
        // Let's not multiply scalable and fixed length vectors
        ICHECK(a.dtype().is_scalable_vector() == b.dtype().is_scalable_vector())
            << "Fixed length and scalable vectors can't be mixed in multiplication.";
      }
      if (is_vec_a || is_vec_b) {
        const RampNode* b_ramp = b.as<RampNode>();
        const RampNode* a_ramp = a.as<RampNode>();
        if (a_ramp && b.dtype().is_scalar() && analyzer_.CanProve(b > 0)) {
          PrimExpr lanes = a_ramp->lanes;
          return Ramp(a_ramp->base * b, a_ramp->stride * b, lanes);
        }
        if (b_ramp && a.dtype().is_scalar() && analyzer_.CanProve(a > 0)) {
          PrimExpr lanes = b_ramp->lanes;
          return Ramp(b_ramp->base * a, b_ramp->stride * a, lanes);
        }
        int a_lanes = a.dtype().get_lanes_or_vscale_factor();
        int b_lanes = b.dtype().get_lanes_or_vscale_factor();
        int max_lanes = std::max(a_lanes, b_lanes);
        bool is_scalable = a.dtype().is_scalable_vector() || b.dtype().is_scalable_vector();
        return Mul(BroadcastTo(a, max_lanes, is_scalable), BroadcastTo(b, max_lanes, is_scalable));
      }
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
    ICHECK(!base.dtype().is_scalable_vector())
        << "Creating scalable vectors from existing vectors is not supported.";
    ICHECK(!stride.dtype().is_scalable_vector())
        << "Ramp stride with scalable dtype is not supported";
    if (base.dtype().is_fixed_length_vector() && stride.dtype().is_scalar()) {
      ICHECK(op->lanes->IsInstance<IntImmNode>())
          << "Vectorizing over existing scalable vectors is not supported.";
      const RampNode* base_ramp = base.as<RampNode>();
      int op_lanes = static_cast<int>(Downcast<IntImm>(op->lanes)->value);
      int base_ramp_lanes = static_cast<int>(Downcast<IntImm>(base_ramp->lanes)->value);
      if (analyzer_.CanProve(base_ramp->stride ==
                             stride * make_const(stride.dtype(), base_ramp_lanes))) {
        return Ramp(base_ramp->base, stride, op_lanes * base_ramp_lanes);
      }
    }
    int lanes = std::max(base.dtype().lanes(), stride.dtype().lanes());
    base = BroadcastTo(base, lanes, false);
    stride = BroadcastTo(stride, lanes, false);
    Array<PrimExpr> elems;
    for (int i = 0; i < lanes; ++i) {
      elems.push_back(
          Ramp(Shuffle::ExtractElement(base, i), Shuffle::ExtractElement(stride, i), op->lanes));
    }
    return Shuffle::Concat(elems);
  }

  PrimExpr VisitExpr_(const BroadcastNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    if (value.dtype().is_scalable_or_fixed_length_vector()) {
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
      int cond_lanes = cond.dtype().get_lanes_or_vscale_factor();
      int t_lanes = t.dtype().get_lanes_or_vscale_factor();
      int f_lanes = f.dtype().get_lanes_or_vscale_factor();
      int lanes = std::max(std::max(cond_lanes, t_lanes), f_lanes);
      bool is_scalable = cond.dtype().is_scalable_vector() || t.dtype().is_scalable_vector() ||
                         f.dtype().is_scalable_vector();
      return Select(BroadcastTo(cond, lanes, is_scalable), BroadcastTo(t, lanes, is_scalable),
                    BroadcastTo(f, lanes, is_scalable));
    }
  }

  PrimExpr VisitExpr_(const CastNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    if (value.same_as(op->value)) {
      return GetRef<PrimExpr>(op);
    } else {
      if (value.dtype().is_scalable_vector()) {
        return Cast(op->dtype.with_scalable_vscale_factor(value.dtype().vscale_factor()), value);
      } else {
        return Cast(op->dtype.with_lanes(value.dtype().lanes()), value);
      }
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
    if (cond.dtype().is_scalable_or_fixed_length_vector()) {
      need_scalarize_ = true;
      return GetRef<PrimExpr>(op);
    }
    PrimExpr t = this->VisitExpr(op->args[1]);
    PrimExpr f = this->VisitExpr(op->args[2]);
    if (cond.same_as(op->args[0]) && t.same_as(op->args[1]) && f.same_as(op->args[2])) {
      return GetRef<PrimExpr>(op);
    } else {
      int t_lanes = t.dtype().get_lanes_or_vscale_factor();
      int f_lanes = f.dtype().get_lanes_or_vscale_factor();
      int lanes = std::max(t_lanes, f_lanes);
      bool is_scalable = t.dtype().is_scalable_vector() || f.dtype().is_scalable_vector();
      t = BroadcastTo(t, lanes, is_scalable);
      f = BroadcastTo(f, lanes, is_scalable);
      if (is_scalable) {
        return Call(op->dtype.with_scalable_vscale_factor(lanes), op->op, {cond, t, f});
      } else {
        return Call(op->dtype.with_lanes(lanes), op->op, {cond, t, f});
      }
    }
  }
  // Reinterpret expr
  PrimExpr MutateReinterpretExpr_(const CallNode* op) {
    ICHECK(op->op.same_as(builtin::reinterpret()));
    PrimExpr value = this->VisitExpr(op->args[0]);
    if (value.same_as(op->args[0])) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = value.dtype().get_lanes_or_vscale_factor();
      if (value.dtype().is_scalable_vector()) {
        return Call(op->dtype.with_scalable_vscale_factor(lanes), op->op, {value});
      } else {
        return Call(op->dtype.with_lanes(lanes), op->op, {value});
      }
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
    } else if (op->op.same_as(builtin::reinterpret())) {
      return MutateReinterpretExpr_(op);
    }
    auto optional_op = op->op.as<Op>();
    bool vectorizable = optional_op && op_vectorizable_.get(optional_op.value(), false) &&
                        !op->dtype.is_scalable_vector();

    if (!vectorizable) {
      // Cannot vectorize this op
      Array<PrimExpr> new_args;
      for (auto arg : op->args) {
        auto new_arg = this->VisitExpr(arg);
        if (new_arg.dtype().is_scalable_or_fixed_length_vector()) {
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
      Array<PrimExpr> new_args;
      if (op->op.same_as(builtin::call_llvm_pure_intrin())) {
        // op->args[1], will give us total number of arguments to intrinsic
        int num_signature = Downcast<IntImm>(op->args[1])->value;
        Array<PrimExpr> op_expr_args;
        for (int i = 0; i < num_signature; i++) {
          // Collect all intrinsic arguments
          op_expr_args.push_back(op->args[i + 2]);
        }
        // Generate RAMP nodes for intrinsic arguments
        Array<PrimExpr> updated_args = MutateArray(op_expr_args, &lane);
        // Collect Intrinsic ID and no. of argument
        for (int i = 0; i < 2; i++) {
          new_args.push_back(op->args[i]);
        }
        // Collect updated intrinsic arguments
        for (int i = 0; i < num_signature; i++) {
          new_args.push_back(updated_args[i]);
        }
      } else {
        new_args = MutateArray(op->args, &lane);
      }
      // normal code path.
      if (op->args.same_as(new_args)) {
        return GetRef<PrimExpr>(op);
      } else {
        return Call(op->dtype.with_lanes(lane), op->op, new_args);
      }
    }
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
    if (value.dtype().get_lanes_or_vscale_factor() !=
        op->value.dtype().get_lanes_or_vscale_factor()) {
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
  // BufferStore
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto store = GetRef<BufferStore>(op);

    auto fmutate = [this](const PrimExpr& index) { return this->VisitExpr(index); };
    Array<PrimExpr> indices = op->indices.Map(fmutate);

    PrimExpr value = this->VisitExpr(op->value);

    if (!indices.same_as(op->indices) || !value.same_as(op->value)) {
      ICHECK(!op->buffer->dtype.is_scalable_vector())
          << "Vectorizing over scalable buffer elements is not supported in vectorizer.";
      // How many lanes of indexing are present in the index and
      // buffer element type, excluding the last index.
      int other_index_lanes = op->buffer->dtype.lanes();
      for (size_t i = 0; i < indices.size() - 1; i++) {
        other_index_lanes *= indices[i].dtype().lanes();
        // Only allow the last index to be scalable
        ICHECK(!indices[i].dtype().is_scalable_vector()) << "Only the last index can be scalable.";
      }

      // The total number of lanes of indexing, including the last index.
      auto last_index_dtype = indices[indices.size() - 1].dtype();
      int lanes_in_last_index = last_index_dtype.get_lanes_or_vscale_factor();
      int index_lanes = other_index_lanes * lanes_in_last_index;

      // The total number of lanes in this store operation.  Either
      // the index or the value will be broadcast out to this number
      // of lanes, depending on which has more lanes.
      int value_dtype_lanes = value.dtype().get_lanes_or_vscale_factor();
      bool is_last_index_scalable = last_index_dtype.is_scalable_vector();
      int total_lanes = std::max(index_lanes, value_dtype_lanes);

      ICHECK_EQ(total_lanes % other_index_lanes, 0)
          << "When storing to buffer " << op->buffer->name << ", cannot produce " << total_lanes
          << " lanes of storage location by changing the last index.";
      int last_index_lanes = total_lanes / other_index_lanes;

      // Broadcast the last index such that the total number of index
      // lanes matches the desired number.
      indices.Set(indices.size() - 1, BroadcastTo(indices[indices.size() - 1], last_index_lanes,
                                                  is_last_index_scalable));

      auto writer = store.CopyOnWrite();
      writer->indices = indices;
      writer->value = BroadcastTo(value, total_lanes, is_last_index_scalable);
    }

    return std::move(store);
  }
  // For
  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kVectorized) {
      LOG(WARNING) << "Detect vectorize inside vectorized loop, ignoring...";
    }
    ICHECK(is_zero(op->min));
    ICHECK(!op->extent.dtype().is_scalable_or_fixed_length_vector());
    PrimExpr extent = this->VisitExpr(op->extent);
    if (extent.dtype().is_scalable_or_fixed_length_vector()) {
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
    ICHECK(!op->condition.dtype().is_scalable_or_fixed_length_vector());
    PrimExpr condition = this->VisitExpr(op->condition);
    // need scalarize can be marked as true during visit of condition
    bool cond_need_scalarize = false;
    std::swap(cond_need_scalarize, need_scalarize_);
    // temp clear need_scalarize flag, so VisitStmt
    // won't trigger an ICHECK eror
    Stmt then_case = this->VisitStmt(op->then_case);
    Optional<Stmt> else_case = NullOpt;
    if (op->else_case) {
      else_case = this->VisitStmt(op->else_case.value());
    }
    // Check if we can rewrite the condition with predicated buffers
    if (EnableBufferLevelPredication(target_) &&
        condition.dtype().is_scalable_or_fixed_length_vector() && !else_case.defined()) {
      std::pair<bool, Stmt> success_stmt_pair =
          TryPredicateBufferAccesses().Run(then_case, condition);
      bool can_remove_if_then_else = success_stmt_pair.first;
      if (can_remove_if_then_else) {
        return success_stmt_pair.second;
      }
    }

    if (cond_need_scalarize || condition.dtype().is_scalable_or_fixed_length_vector()) {
      return Scalarize(GetRef<Stmt>(op));
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
  }
  // LetStmt
  Stmt VisitStmt_(const LetStmtNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    // if visit of value triggers need scalarize
    // we need to scalarize the let
    if (need_scalarize_) {
      need_scalarize_ = false;
      Scalarize(GetRef<Stmt>(op));
    }
    ICHECK(!let_binding_.count(op->var)) << "SSA violation, a single var is binded twice";
    let_binding_[op->var] = value;

    if (value.dtype().get_lanes_or_vscale_factor() !=
        op->value.dtype().get_lanes_or_vscale_factor()) {
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
    if (condition.dtype().is_scalable_or_fixed_length_vector()) {
      LOG(WARNING) << "Cannot handle vector extent in alloc of " << op->buffer_var->name_hint;
      return Scalarize(GetRef<Stmt>(op));
    }

    // Mutate the extents
    Array<PrimExpr> extents;
    for (const auto& extent : op->extents) {
      PrimExpr new_ext = this->VisitExpr(extent);
      if (new_ext.dtype().is_scalable_or_fixed_length_vector()) {
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
    stmt = Substitute(stmt, {{var_, idx}});
    return For(idx, IntImm(var_->dtype, 0), var_lanes_, ForKind::kSerial, stmt);
  }
  // ProducerStore
  Stmt VisitStmt_(const ProducerStoreNode* op) final {
    LOG(FATAL) << "ProducerProvide cannot appear in a TIR PrimFunc";
  }

 private:
  // analyzer
  arith::Analyzer analyzer_;
  // deep equal
  ExprDeepEqual deep_equal_;
  // variable to be replaced
  Var var_;
  // the lanes.
  PrimExpr var_lanes_;
  // ramp representing the var.
  PrimExpr ramp_;
  // flag to mark requirment of scalarization.
  bool need_scalarize_{false};
  // Let binding
  std::unordered_map<Var, PrimExpr> let_binding_;
  // vectorizable property
  OpAttrMap<TVectorizable> op_vectorizable_ = Op::GetAttrMap<TVectorizable>("TVectorizable");
  /*! \brief The current target context. */
  Target target_;

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
        new_arr[i] = BroadcastTo(new_arr[i], lanes, false);
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
      int a_lanes = a.dtype().get_lanes_or_vscale_factor();
      int b_lanes = b.dtype().get_lanes_or_vscale_factor();
      int lanes = std::max(a_lanes, b_lanes);
      bool is_scalable = a.dtype().is_scalable_vector() || b.dtype().is_scalable_vector();
      return TOp(BroadcastTo(a, lanes, is_scalable), BroadcastTo(b, lanes, is_scalable));
    }
  }
  template <typename T, typename FCompute>
  PrimExpr AddSubVec(const T* op, FCompute fcompute) {
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) && b.same_as(op->b)) {
      return GetRef<PrimExpr>(op);
    } else {
      int a_lanes = a.dtype().get_lanes_or_vscale_factor();
      int b_lanes = b.dtype().get_lanes_or_vscale_factor();
      int lanes = std::max(a_lanes, b_lanes);
      if (lanes != 1) {
        const RampNode* b_ramp = b.as<RampNode>();
        const RampNode* a_ramp = a.as<RampNode>();
        if (a.dtype().is_scalar() && b_ramp) {
          return Ramp(fcompute(a, b_ramp->base),
                      fcompute(make_zero(b_ramp->stride.dtype()), b_ramp->stride), b_ramp->lanes);
        }
        if (b.dtype().is_scalar() && a_ramp) {
          return Ramp(fcompute(a_ramp->base, b), a_ramp->stride, a_ramp->lanes);
        }
      }
      bool is_scalable = a.dtype().is_scalable_vector() || b.dtype().is_scalable_vector();
      return fcompute(BroadcastTo(a, lanes, is_scalable), BroadcastTo(b, lanes, is_scalable));
    }
  }
};

class LoopVectorizer : public StmtMutator {
 public:
  explicit LoopVectorizer(DictAttrs attrs) {
    if (auto opt_target = attrs.GetAttr<Target>(tvm::attr::kTarget)) {
      target_ = opt_target.value();
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kVectorized) {
      auto* extent_as_int = op->extent.as<IntImmNode>();

      if (!extent_as_int || extent_as_int->value < 1) {
        bool is_scalable_expr = CheckContains::ExprContains(op->extent, arith::IsVScaleCall);
        ICHECK(is_scalable_expr && arith::TargetHasSVE(target_))
            << "Failed to vectorize loop with extent " << op->extent << " for target " << target_;
      }
      ICHECK(is_zero(op->min));
      return Vectorizer(op->loop_var, op->extent, target_)(op->body);
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tvm::attr::kTarget) {
      Target previous_target = target_;
      target_ = op->node.as<Target>().value();
      Stmt new_op = StmtMutator::VisitStmt_(op);
      target_ = previous_target;
      return new_op;
    }
    return StmtMutator::VisitStmt_(op);
  }

 private:
  Target target_ = Target::Current();
};

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
      n->body = LoopVectorizer(n->attrs)(std::move(n->body));
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
