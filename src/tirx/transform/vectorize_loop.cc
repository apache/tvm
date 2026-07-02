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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <vector>

#include "../../tirx/analysis/check_contains.h"
#include "tvm/ffi/dtype.h"
#include "tvm/tirx/buffer.h"

namespace tvm {
namespace tirx {

namespace {
int GetLanesOrVScaleFactor(const PrimType& ty) {
  if (ty.IsScalableVector()) {
    return ty.VScaleFactor();
  }
  return ty.lanes();
}

// File-local helper: true if `expr` is a call to tirx::builtin::vscale().
bool IsVScaleCall(const PrimExpr& expr) {
  if (const auto* call = expr.as<CallNode>()) {
    return call->op.same_as(builtin::vscale());
  }
  return false;
}

bool TargetHasRVV(Target target) {
  if (!target.defined()) return false;
  static auto target_has_feature_fn = tvm::ffi::Function::GetGlobal("target.target_has_feature");
  return target_has_feature_fn.has_value() && (*target_has_feature_fn)("v", target).cast<bool>();
}

// File-local helper: true if the target supports Variable-Length Array extensions
// (AArch64 SVE or RISC-V V).
bool TargetHasVLA(Target target) {
  if (!target.defined()) return false;
  bool has_vla = target->GetAttr<bool>("feature.has_sve").value_or(false);
  if (!has_vla) {
    if (auto mattr = target->GetAttr<ffi::Array<ffi::String>>("mattr")) {
      for (const ffi::String& attr : mattr.value()) {
        if (attr == "+sve") {
          has_vla = true;
          break;
        }
      }
    }
  }
  has_vla |= TargetHasRVV(target);
  return has_vla;
}

bool ContainsCallNode(const Stmt& stmt) {
  return CheckContains::StmtContains(
      stmt, [](const PrimExpr& expr) { return expr.as<CallNode>() != nullptr; });
}
}  // namespace

inline PrimExpr CreateNewLanes(bool is_scalable, int lanes_or_vscale_factor) {
  if (is_scalable) {
    return Mul(Call(PrimType::Int(32), builtin::vscale(), {}).as_or_throw<PrimExpr>(),
               lanes_or_vscale_factor);
  } else {
    return lanes_or_vscale_factor;
  }
}

inline PrimExpr BroadcastTo(PrimExpr e, int lanes, bool is_scalable) {
  // Check if e is already in the expected form
  if (GetLanesOrVScaleFactor(e.ty()) == lanes && e.ty().IsScalableVector() == is_scalable) return e;

  if (const BroadcastNode* op = e.as<BroadcastNode>()) {
    TVM_FFI_ICHECK(op->ty.as_or_throw<PrimType>().IsScalableVector() == is_scalable)
        << "Can't broadcast between scalable and fixed length vectors.";
    int e_lanes = GetLanesOrVScaleFactor(op->ty.as_or_throw<PrimType>());

    if (lanes % e_lanes == 0) {
      return Broadcast(op->value, CreateNewLanes(is_scalable, lanes));
    }
  }

  TVM_FFI_ICHECK(e.ty().IsScalar())
      << "Cannot broadcast lanes=" << GetLanesOrVScaleFactor(e.ty())
      << " is_scalable=" << e.ty().IsScalableVector() << " to " << lanes;

  return Broadcast(e, CreateNewLanes(is_scalable, lanes));
}

bool EnableBufferLevelPredication(Target target) {
  transform::PassContext pass_ctx = transform::PassContext::Current();
  ffi::Optional<bool> enable_buffer_predication =
      pass_ctx->GetConfig<bool>("tirx.enable_buffer_level_predication");
  if (enable_buffer_predication.has_value()) {
    return enable_buffer_predication.value();
  }

  // Use buffer-level predication by default for VLA targets
  return TargetHasVLA(target);
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
  explicit TryPredicateBufferAccesses(bool allow_offset_predication)
      : allow_offset_predication_(allow_offset_predication) {}

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

    LT lt = condition.as_or_throw<LT>();

    // Check the form of the vectorized condition, we're expecting
    // Ramp(...) < Broadcast(...)
    if (!lt->a->IsInstance<RampNode>() || !lt->b->IsInstance<BroadcastNode>()) {
      return {false, stmt};
    }

    Ramp pred_ramp = lt->a.as_or_throw<Ramp>();
    base_ = pred_ramp->base;
    stride_ = pred_ramp->stride;
    lanes_ = pred_ramp->lanes;
    limit_ = lt->b.as_or_throw<Broadcast>()->value;

    // Now we can try to predicate
    Stmt predicated_stmt = StmtExprMutator::operator()(std::move(stmt));
    if (num_accesses_analyzed_ > 0 && num_accesses_analyzed_ == num_accesses_rewritten_) {
      return {true, predicated_stmt};
    }
    return {false, stmt};
  }

 private:
  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto load = StmtExprMutator::VisitExpr_(op).as_or_throw<BufferLoad>();
    return TryPredicateBufferAccess(load);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto store = StmtExprMutator::VisitStmt_(op).as_or_throw<BufferStore>();
    return TryPredicateBufferAccess(store);
  }

  template <typename AccessNode>
  AccessNode TryPredicateBufferAccess(AccessNode node) {
    num_accesses_analyzed_ += 1;

    // Do not try to predicate non-vectorized accesses
    ffi::Array<PrimExpr> indices = node->indices;
    if (!indices.size() || !indices[0]->IsInstance<RampNode>()) {
      return node;
    }
    Ramp ramp = node->indices[0].template as_or_throw<Ramp>();

    if (!ffi::StructuralEqual()(ramp->stride, stride_) ||
        !ffi::StructuralEqual()(ramp->lanes, lanes_)) {
      return node;
    }

    bool same_base = ffi::StructuralEqual()(ramp->base, base_);
    if (!same_base) {
      // The lane mask describes which lanes are active, independent of the
      // memory base.  This covers accesses such as A[offset + i] guarded by
      // a predicate over i.
      if (!allow_offset_predication_) {
        return node;
      }
    }

    PrimType buf_predicate_dtype =
        ramp->ty.as_or_throw<PrimType>().IsScalableVector()
            ? PrimType::ScalableVector(DLDataTypeCode::kDLUInt, 1,
                                       GetLanesOrVScaleFactor(ramp->ty.as_or_throw<PrimType>()))
            : PrimType::UInt(1, GetLanesOrVScaleFactor(ramp->ty.as_or_throw<PrimType>()));
    PrimExpr lane_mask = Call(buf_predicate_dtype, builtin::get_active_lane_mask(), {base_, limit_})
                             .as_or_throw<PrimExpr>();

    num_accesses_rewritten_ += 1;
    auto writer = node.CopyOnWrite();
    if (node->predicate.defined() && allow_offset_predication_) {
      // Buffer predicates are uint1 lane masks, so mask merging uses bitwise
      // and rather than logical &&.
      writer->predicate = node->predicate.value() & lane_mask;
    } else {
      writer->predicate = lane_mask;
    }
    return node;
  }

  /*! \brief The variable base expr of the predicate. */
  PrimExpr base_;
  /*! \brief The lane stride of the predicate. */
  PrimExpr stride_;
  /*! \brief The lane count of the predicate. */
  PrimExpr lanes_;
  /*! \brief The limit of the predicate. The expr specifies the upper bound of the base's
   * evaluated value. */
  PrimExpr limit_;
  /*! \brief Whether to predicate offset buffer accesses that use the same lane layout. */
  bool allow_offset_predication_;
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
    auto load = StmtExprMutator::VisitExpr_(op).as_or_throw<BufferLoad>();
    return UpdateBufferAccess(load);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto store = StmtExprMutator::VisitStmt_(op).as_or_throw<BufferStore>();
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
      ffi::Array<PrimExpr> shape = node->buffer->shape;
      shape.Set(shape.size() - 1, analyzer_->Simplify(shape[shape.size() - 1] * var_lanes_));

      // TODO(Lunderberg): Move this pass to be prior to
      // FlattenBuffer, implement by appending a
      // dimension to the buffer.  Since it is currently after the
      // flattening, the strides are not technically necessary, but
      // are updated for consistency.

      // Update strides if defined.
      ffi::Array<PrimExpr> strides;
      for (size_t i = 0; i < strides.size(); i++) {
        PrimExpr stride = strides[i];
        if (i != strides.size() - 1) {
          stride *= var_lanes_;
        }
        strides.push_back(analyzer_->Simplify(stride));
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
    ffi::Array<PrimExpr> indices = node->indices;
    indices.Set(indices.size() - 1,
                analyzer_->Simplify(indices[indices.size() - 1] * var_lanes_ + var_));

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
    ramp_ = Ramp(IntImm(var.ty(), 0), IntImm(var.ty(), 1), var_lanes);
  }

  Stmt VisitStmt(const Stmt& stmt) final {
    TVM_FFI_ICHECK(!need_scalarize_);
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
      return ffi::GetRef<PrimExpr>(op);
    } else {
      bool is_vec_a = a.ty().IsScalableVector() || a.ty().IsFixedLengthVector();
      bool is_vec_b = b.ty().IsScalableVector() || b.ty().IsFixedLengthVector();
      if (is_vec_a && is_vec_b) {
        // Let's not multiply scalable and fixed length vectors
        TVM_FFI_ICHECK(a.ty().IsScalableVector() == b.ty().IsScalableVector())
            << "Fixed length and scalable vectors can't be mixed in multiplication.";
      }
      if (is_vec_a || is_vec_b) {
        const RampNode* b_ramp = b.as<RampNode>();
        const RampNode* a_ramp = a.as<RampNode>();
        if (a_ramp && b.ty().IsScalar() && analyzer_->CanProve(b > 0)) {
          PrimExpr lanes = a_ramp->lanes;
          return Ramp(a_ramp->base * b, a_ramp->stride * b, lanes);
        }
        if (b_ramp && a.ty().IsScalar() && analyzer_->CanProve(a > 0)) {
          PrimExpr lanes = b_ramp->lanes;
          return Ramp(b_ramp->base * a, b_ramp->stride * a, lanes);
        }
        int a_lanes = GetLanesOrVScaleFactor(a.ty());
        int b_lanes = GetLanesOrVScaleFactor(b.ty());
        int max_lanes = std::max(a_lanes, b_lanes);
        bool is_scalable = a.ty().IsScalableVector() || b.ty().IsScalableVector();
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
      return ffi::GetRef<PrimExpr>(op);
    } else {
      return !(a);
    }
  }

  PrimExpr VisitExpr_(const RampNode* op) final {
    PrimExpr base = this->VisitExpr(op->base);
    PrimExpr stride = this->VisitExpr(op->stride);
    TVM_FFI_ICHECK(!base.ty().IsScalableVector())
        << "Creating scalable vectors from existing vectors is not supported.";
    TVM_FFI_ICHECK(!stride.ty().IsScalableVector())
        << "Ramp stride with scalable dtype is not supported";
    if (base.ty().IsFixedLengthVector() && stride.ty().IsScalar()) {
      TVM_FFI_ICHECK(op->lanes->IsInstance<IntImmNode>())
          << "Vectorizing over existing scalable vectors is not supported.";
      const RampNode* base_ramp = base.as<RampNode>();
      int op_lanes = static_cast<int>(op->lanes.as_or_throw<IntImm>()->value);
      int base_ramp_lanes = static_cast<int>(base_ramp->lanes.as_or_throw<IntImm>()->value);
      if (analyzer_->CanProve(base_ramp->stride ==
                              stride * MakeConst(stride.ty(), base_ramp_lanes))) {
        return Ramp(base_ramp->base, stride, op_lanes * base_ramp_lanes);
      }
    }
    int lanes = std::max(base.ty().lanes(), stride.ty().lanes());
    base = BroadcastTo(base, lanes, false);
    stride = BroadcastTo(stride, lanes, false);
    ffi::Array<PrimExpr> elems;
    for (int i = 0; i < lanes; ++i) {
      elems.push_back(
          Ramp(Shuffle::ExtractElement(base, i), Shuffle::ExtractElement(stride, i), op->lanes));
    }
    return Shuffle::Concat(elems);
  }

  PrimExpr VisitExpr_(const BroadcastNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    if (value.ty().IsScalableVector() || value.ty().IsFixedLengthVector()) {
      need_scalarize_ = true;
      return ffi::GetRef<PrimExpr>(op);
    }
    if (value.same_as(op->value)) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      return Broadcast(op->value, op->lanes);
    }
  }

  PrimExpr VisitExpr_(const SelectNode* op) final {
    PrimExpr cond = this->VisitExpr(op->condition);
    PrimExpr t = this->VisitExpr(op->true_value);
    PrimExpr f = this->VisitExpr(op->false_value);
    if (cond.same_as(op->condition) && t.same_as(op->true_value) && f.same_as(op->false_value)) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      int cond_lanes = GetLanesOrVScaleFactor(cond.ty());
      int t_lanes = GetLanesOrVScaleFactor(t.ty());
      int f_lanes = GetLanesOrVScaleFactor(f.ty());
      int lanes = std::max(std::max(cond_lanes, t_lanes), f_lanes);
      bool is_scalable =
          cond.ty().IsScalableVector() || t.ty().IsScalableVector() || f.ty().IsScalableVector();
      return Select(BroadcastTo(cond, lanes, is_scalable), BroadcastTo(t, lanes, is_scalable),
                    BroadcastTo(f, lanes, is_scalable));
    }
  }

  PrimExpr VisitExpr_(const CastNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    if (value.same_as(op->value)) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      if (value.ty().IsScalableVector()) {
        return Cast(PrimType::ScalableVector(op->ty.as_or_throw<PrimType>().code(),
                                             op->ty.as_or_throw<PrimType>().bits(),
                                             value.ty().VScaleFactor()),
                    value);
      } else {
        return Cast(op->ty.as_or_throw<PrimType>().WithLanes(value.ty().lanes()), value);
      }
    }
  }

  PrimExpr VisitExpr_(const FloatImmNode* op) final { return ffi::GetRef<PrimExpr>(op); }

  PrimExpr VisitExpr_(const IntImmNode* op) final { return ffi::GetRef<PrimExpr>(op); }

  PrimExpr VisitExpr_(const StringImmNode* op) final { return ffi::GetRef<PrimExpr>(op); }

  // Variable
  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = ffi::GetRef<Var>(op);

    if (var.same_as(var_)) {
      return ramp_;
    }
    auto it = let_binding_.find(var);
    if (it != let_binding_.end()) {
      return it->second;
    } else {
      return var;
    }
  }
  // IfThenElse expr
  PrimExpr MutateIfThenElseExpr_(const CallNode* op) {
    PrimExpr cond = this->VisitExpr(op->args[0].as_or_throw<PrimExpr>());
    if (cond.ty().IsScalableVector() || cond.ty().IsFixedLengthVector()) {
      need_scalarize_ = true;
      return ffi::GetRef<Call>(op).as_or_throw<PrimExpr>();
    }
    PrimExpr t = this->VisitExpr(op->args[1].as_or_throw<PrimExpr>());
    PrimExpr f = this->VisitExpr(op->args[2].as_or_throw<PrimExpr>());
    if (cond.same_as(op->args[0]) && t.same_as(op->args[1]) && f.same_as(op->args[2])) {
      return ffi::GetRef<Call>(op).as_or_throw<PrimExpr>();
    } else {
      int t_lanes = GetLanesOrVScaleFactor(t.ty());
      int f_lanes = GetLanesOrVScaleFactor(f.ty());
      int lanes = std::max(t_lanes, f_lanes);
      bool is_scalable = t.ty().IsScalableVector() || f.ty().IsScalableVector();
      t = BroadcastTo(t, lanes, is_scalable);
      f = BroadcastTo(f, lanes, is_scalable);
      PrimType op_ty = op->ty.as_or_throw<PrimType>();
      if (is_scalable) {
        return Call(PrimType::ScalableVector(op_ty.code(), op_ty.bits(), lanes), op->op,
                    {cond, t, f}, op->attrs, {}, op->span)
            .as_or_throw<PrimExpr>();
      } else {
        return Call(op_ty.WithLanes(lanes), op->op, {cond, t, f}, op->attrs, {}, op->span)
            .as_or_throw<PrimExpr>();
      }
    }
  }
  // Reinterpret expr
  PrimExpr MutateReinterpretExpr_(const CallNode* op) {
    TVM_FFI_ICHECK(op->op.same_as(builtin::reinterpret()));
    PrimExpr input = op->args[0].as_or_throw<PrimExpr>();
    PrimExpr value = this->VisitExpr(input);
    if (value.same_as(op->args[0])) {
      return ffi::GetRef<Call>(op).as_or_throw<PrimExpr>();
    } else {
      int lanes = GetLanesOrVScaleFactor(value.ty());
      PrimType op_ty = op->ty.as_or_throw<PrimType>();
      if (value.ty().IsScalableVector()) {
        return Call(PrimType::ScalableVector(op_ty.code(), op_ty.bits(), lanes), op->op, {value},
                    op->attrs, {}, op->span)
            .as_or_throw<PrimExpr>();
      } else {
        int new_lanes = (op_ty.code() != DLDataTypeCode::kDLFloat4_e2m1fn &&
                         input.ty().code() != DLDataTypeCode::kDLFloat4_e2m1fn)
                            ? (value.ty().bits() * value.ty().lanes()) / op_ty.bits()
                            : value.ty().lanes();
        return Call(op_ty.WithLanes(new_lanes), op->op, {value}, op->attrs, {}, op->span)
            .as_or_throw<PrimExpr>();
      }
    }
  }
  // Call
  PrimExpr VisitExpr_(const CallNode* op) final {
    PrimType ret_ty = op->ty.as_or_throw<PrimType>();
    if (op->op.same_as(builtin::if_then_else())) {
      return MutateIfThenElseExpr_(op);
    } else if (op->op.same_as(builtin::texture2d_load())) {
      int lane = 0;
      ffi::Array<PrimExpr> fcd = MutateArray({op->args.back().as_or_throw<PrimExpr>()}, &lane);
      DLDataType dtype = op->args[0]
                             .as<VarNode>()
                             ->ty.as<PointerTypeNode>()
                             ->element_type.as<PrimTypeNode>()
                             ->dtype;
      TVM_FFI_ICHECK(lane * dtype.bits <= op->args[4].as<IntImmNode>()->value)
          << "Expected Data to be Read is lesser than or equal to Texture Load length";

      auto new_args = op->args;
      new_args.pop_back();
      new_args.push_back(fcd[0]);
      ffi::Array<PrimExpr> prim_args = new_args.as_or_throw<ffi::Array<PrimExpr>>();
      return Call(ret_ty.WithLanes(lane), op->op, prim_args, op->attrs, {}, op->span)
          .as_or_throw<PrimExpr>();
    } else if (op->op.same_as(builtin::texture2d_store())) {
      int lane = 0;
      ffi::Array<PrimExpr> prim_args = op->args.as_or_throw<ffi::Array<PrimExpr>>();
      // Vectorize the value to store
      ffi::Array<PrimExpr> value{prim_args.back()};
      ffi::Array<PrimExpr> mutated_value = MutateArray(value, &lane);
      DLDataType dtype = op->args[0]
                             .as<VarNode>()
                             ->ty.as<PointerTypeNode>()
                             ->element_type.as<PrimTypeNode>()
                             ->dtype;
      TVM_FFI_ICHECK(lane * dtype.bits == op->args[4].as<IntImmNode>()->value)
          << "Expected Data to be Written equal to Texture Store length";
      ffi::Array<PrimExpr> new_args{prim_args[0], prim_args[1], prim_args[2],
                                    prim_args[3], prim_args[4], mutated_value[0]};
      return Call(ret_ty.WithLanes(lane), op->op, new_args, op->attrs, {}, op->span)
          .as_or_throw<PrimExpr>();
    } else if (op->op.same_as(builtin::reinterpret())) {
      return MutateReinterpretExpr_(op);
    }
    auto optional_op = op->op.as<Op>();
    bool vectorizable = optional_op && op_vectorizable_.get(optional_op.value(), false) &&
                        !ret_ty.IsScalableVector();

    if (!vectorizable) {
      // Cannot vectorize this op
      ffi::Array<PrimExpr> new_args;
      for (const PrimExpr& arg : op->args.as_or_throw<ffi::Array<PrimExpr>>()) {
        auto new_arg = this->VisitExpr(arg);
        if (new_arg.ty().IsScalableVector() || new_arg.ty().IsFixedLengthVector()) {
          need_scalarize_ = true;
          return ffi::GetRef<Call>(op).as_or_throw<PrimExpr>();
        }
        new_args.push_back(new_arg);
      }
      if (op->args.same_as(new_args)) {
        return ffi::GetRef<Call>(op).as_or_throw<PrimExpr>();
      } else {
        return Call(ret_ty, op->op, new_args, op->attrs, {}, op->span).as_or_throw<PrimExpr>();
      }
    } else {
      int lane = 0;
      ffi::Array<PrimExpr> new_args;
      if (op->op.same_as(builtin::call_llvm_pure_intrin())) {
        // op->args[1], will give us total number of arguments to intrinsic
        ffi::Array<PrimExpr> op_expr_args;
        ffi::Array<PrimExpr> prim_args = op->args.as_or_throw<ffi::Array<PrimExpr>>();
        for (size_t i = 1; i < prim_args.size(); ++i) {
          // Collect all intrinsic arguments
          op_expr_args.push_back(prim_args[i]);
        }
        // Generate RAMP nodes for intrinsic arguments
        ffi::Array<PrimExpr> updated_args = MutateArray(op_expr_args, &lane);
        new_args.push_back(prim_args[0]);
        // Collect updated intrinsic arguments
        for (size_t i = 0; i < updated_args.size(); ++i) {
          new_args.push_back(updated_args[i]);
        }
      } else {
        ffi::Array<PrimExpr> prim_args = op->args.as_or_throw<ffi::Array<PrimExpr>>();
        new_args = MutateArray(prim_args, &lane);
      }
      // normal code path.
      if (op->args.same_as(new_args)) {
        return ffi::GetRef<Call>(op).as_or_throw<PrimExpr>();
      } else {
        return Call(ret_ty.WithLanes(lane), op->op, new_args, op->attrs, {}, op->span)
            .as_or_throw<PrimExpr>();
      }
    }
  }
  // BufferLoad
  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto load = ffi::GetRef<BufferLoad>(op);

    auto fmutate = [this](const PrimExpr& index) { return this->VisitExpr(index); };
    ffi::Array<PrimExpr> indices = op->indices.Map(fmutate);

    if (!indices.same_as(op->indices)) {
      auto writer = load.CopyOnWrite();
      writer->indices = indices;
      writer->LegalizeDType();
    }

    return load;
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
      TVM_FFI_ICHECK(deep_equal_(it->second, value))
          << "Let cannot bind the same var to two different values";
    }
    if (GetLanesOrVScaleFactor(value.ty()) != GetLanesOrVScaleFactor(op->value.ty())) {
      Var new_var(op->var->name_hint, value.ty());
      let_binding_[op->var] = new_var;
      return Let(new_var, value, this->VisitExpr(op->body));
    } else {
      let_binding_[op->var] = op->var;
      PrimExpr body = this->VisitExpr(op->body);
      if (value.same_as(op->value) && body.same_as(op->body)) {
        return ffi::GetRef<PrimExpr>(op);
      } else {
        return Let(op->var, value, body);
      }
    }
  }
  PrimExpr VisitExpr_(const ShuffleNode* op) final {
    TVM_FFI_ICHECK(op->vectors.size() == 1 && op->indices.size() == 1)
        << "Cannot vectorize ShuffleNode with multiple vectors or indices: the vector size is "
        << op->vectors.size() << " and the index size is " << op->indices.size();
    int lane_vectors = 0;
    int lane_indices = 0;
    ffi::Array<PrimExpr> vectors = MutateArray(op->vectors, &lane_vectors);
    ffi::Array<PrimExpr> indices = MutateArray(op->indices, &lane_indices);
    if (vectors.same_as(op->vectors) && indices.same_as(op->indices)) {
      return ffi::GetRef<PrimExpr>(op);
    }

    int new_vec_length = var_lanes_.as_or_throw<IntImm>()->value / op->vectors[0].ty().lanes();
    PrimExpr updated_index = indices[0];
    // Check that the indices satisfy the specific patterns.
    auto f_check_index = [this, op](const PrimExpr& index) {
      // Allowing Ramp(0, 1, var_lanes_)
      if (const auto* ramp = index.as<RampNode>()) {
        if (ramp->base->IsInstance<IntImmNode>() && ramp->base.as_or_throw<IntImm>()->value == 0 &&
            ramp->stride->IsInstance<IntImmNode>() &&
            ramp->stride.as_or_throw<IntImm>()->value == 1 &&
            ramp->lanes->IsInstance<IntImmNode>() &&
            ramp->lanes.as_or_throw<IntImm>()->value == var_lanes_.as_or_throw<IntImm>()->value) {
          return true;
        }
      }
      // Allowing FloorMod(Ramp(0, 1, var_lanes_), Broadcast(op->vectors[0]->lanes, var_lanes_))
      if (const auto* floordiv = index.as<FloorModNode>()) {
        if (const auto* ramp = floordiv->a.as<RampNode>()) {
          if (const auto* broadcast = floordiv->b.as<BroadcastNode>()) {
            if (ramp->base->IsInstance<IntImmNode>() &&
                ramp->base.as_or_throw<IntImm>()->value == 0 &&
                ramp->stride->IsInstance<IntImmNode>() &&
                ramp->stride.as_or_throw<IntImm>()->value == 1 &&
                ramp->lanes->IsInstance<IntImmNode>() &&
                ramp->lanes.as_or_throw<IntImm>()->value ==
                    var_lanes_.as_or_throw<IntImm>()->value &&
                broadcast->value->IsInstance<IntImmNode>() &&
                broadcast->value.as_or_throw<IntImm>()->value == op->vectors[0].ty().lanes() &&
                broadcast->lanes->IsInstance<IntImmNode>() &&
                broadcast->lanes.as_or_throw<IntImm>()->value ==
                    var_lanes_.as_or_throw<IntImm>()->value) {
              return true;
            }
          }
        }
      }

      return false;
    };
    TVM_FFI_ICHECK(f_check_index(updated_index));

    if (new_vec_length == 1) {
      return tirx::Substitute(op->vectors[0], {{var_, tvm::IntImm(var_.ty(), 0)}});
    } else {
      PrimExpr prev_ramp = ramp_;
      PrimExpr prev_var_lanes = var_lanes_;
      ramp_ = Ramp(IntImm(var_.ty(), 0), IntImm(var_.ty(), 2), new_vec_length);
      var_lanes_ = tvm::IntImm(var_lanes_.ty(), new_vec_length);
      lane_vectors = 0;
      vectors = MutateArray(op->vectors, &lane_vectors);
      ramp_ = prev_ramp;
      var_lanes_ = prev_var_lanes;
      return vectors[0];
    }
  }
  // BufferStore
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto store = ffi::GetRef<BufferStore>(op);

    auto fmutate = [this](const PrimExpr& index) { return this->VisitExpr(index); };
    ffi::Array<PrimExpr> indices = op->indices.Map(fmutate);

    PrimExpr value = this->VisitExpr(op->value);

    if (!indices.same_as(op->indices) || !value.same_as(op->value)) {
      TVM_FFI_ICHECK(!op->buffer->dtype.IsScalableVector())
          << "Vectorizing over scalable buffer elements is not supported in vectorizer.";
      // How many lanes of indexing are present in the index and
      // buffer element type, excluding the last index.
      int other_index_lanes = op->buffer->dtype.lanes();
      for (size_t i = 0; i < indices.size() - 1; i++) {
        other_index_lanes *= indices[i].ty().lanes();
        // Only allow the last index to be scalable
        TVM_FFI_ICHECK(!indices[i].ty().IsScalableVector())
            << "Only the last index can be scalable.";
      }

      // The total number of lanes of indexing, including the last index.
      PrimType last_index_dtype = indices[indices.size() - 1].ty();
      int lanes_in_last_index = GetLanesOrVScaleFactor(last_index_dtype);
      int index_lanes = other_index_lanes * lanes_in_last_index;

      // The total number of lanes in this store operation.  Either
      // the index or the value will be broadcast out to this number
      // of lanes, depending on which has more lanes.
      int value_dtype_lanes = GetLanesOrVScaleFactor(value.ty());
      bool is_last_index_scalable = last_index_dtype.IsScalableVector();
      int total_lanes = std::max(index_lanes, value_dtype_lanes);

      TVM_FFI_ICHECK_EQ(total_lanes % other_index_lanes, 0)
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

    return store;
  }
  // For
  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kVectorized) {
      LOG(WARNING) << "Detect vectorize inside vectorized loop, ignoring...";
    }
    TVM_FFI_ICHECK(is_zero(op->min));
    TVM_FFI_ICHECK(!op->extent.ty().IsScalableVector() && !op->extent.ty().IsFixedLengthVector());
    PrimExpr extent = this->VisitExpr(op->extent);
    if (extent.ty().IsScalableVector() || extent.ty().IsFixedLengthVector()) {
      return Scalarize(ffi::GetRef<Stmt>(op));
    }
    Stmt body = this->VisitStmt(op->body);
    if (extent.same_as(op->extent) && body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->extent = extent;
      n->body = body;
      return For(n);
    }
  }
  // IfThenElse
  Stmt VisitStmt_(const IfThenElseNode* op) final {
    TVM_FFI_ICHECK(!op->condition.ty().IsScalableVector() &&
                   !op->condition.ty().IsFixedLengthVector());
    PrimExpr condition = this->VisitExpr(op->condition);
    // need scalarize can be marked as true during visit of condition
    bool cond_need_scalarize = false;
    std::swap(cond_need_scalarize, need_scalarize_);
    // temp clear need_scalarize flag, so VisitStmt
    // won't trigger an TVM_FFI_ICHECK eror
    Stmt then_case = this->VisitStmt(op->then_case);
    ffi::Optional<Stmt> else_case = std::nullopt;
    if (op->else_case) {
      else_case = this->VisitStmt(op->else_case.value());
    }
    // Check if we can rewrite the condition with predicated buffers
    if (EnableBufferLevelPredication(target_) &&
        (condition.ty().IsScalableVector() || condition.ty().IsFixedLengthVector()) &&
        !else_case.defined()) {
      std::pair<bool, Stmt> success_stmt_pair =
          TryPredicateBufferAccesses(TargetHasRVV(target_)).Run(then_case, condition);
      bool can_remove_if_then_else = success_stmt_pair.first;
      if (can_remove_if_then_else) {
        return success_stmt_pair.second;
      }
    }

    if (cond_need_scalarize || condition.ty().IsScalableVector() ||
        condition.ty().IsFixedLengthVector()) {
      return Scalarize(ffi::GetRef<Stmt>(op));
    }
    if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      return IfThenElse(condition, then_case, else_case);
    }
  }
  // While
  Stmt VisitStmt_(const WhileNode* op) final {
    TVM_FFI_THROW(InternalError) << "A while loop inside a vectorized loop not supported.";
    TVM_FFI_UNREACHABLE();
  }
  // Bind
  Stmt VisitStmt_(const BindNode* op) final {
    auto prim_value = op->value.as<PrimExpr>();
    if (!prim_value) {
      return StmtMutator::VisitStmt_(op);
    }
    PrimExpr value = this->VisitExpr(prim_value.value());
    // if visit of value triggers need scalarize
    // we need to scalarize the let
    if (need_scalarize_) {
      need_scalarize_ = false;
      return Scalarize(ffi::GetRef<Stmt>(op));
    }
    TVM_FFI_ICHECK(!let_binding_.count(op->var)) << "SSA violation, a single var is binded twice";
    let_binding_[op->var] = value;

    if (GetLanesOrVScaleFactor(value.ty()) != GetLanesOrVScaleFactor(prim_value.value().ty())) {
      Var new_var(op->var->name_hint, value.ty());
      let_binding_[op->var] = new_var;
      return Bind(new_var, value);
    } else {
      let_binding_[op->var] = op->var;
      if (value.same_as(op->value)) {
        return ffi::GetRef<Stmt>(op);
      } else {
        return Bind(op->var, value, op->span);
      }
    }
  }
  // AllocBuffer: just visit the body (vectorization of AllocBuffer not yet implemented)
  Stmt VisitStmt_(const AllocBufferNode* op) final { return StmtMutator::VisitStmt_(op); }

  // scalarize the statment
  Stmt Scalarize(Stmt stmt) {
    Var idx(var_->name_hint + ".s", var_.ty());
    stmt = Substitute(stmt, {{var_, idx}});
    return For(idx, IntImm(var_.ty(), 0), var_lanes_, ForKind::kSerial, stmt);
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
  ffi::Array<PrimExpr> MutateArray(ffi::Array<PrimExpr> arr, int* p_lanes) {
    if (arr.size() == 0) return arr;
    int& lanes = *p_lanes;
    bool changed = false;
    std::vector<PrimExpr> new_arr(arr.size());
    for (size_t i = 0; i < arr.size(); i++) {
      PrimExpr old_elem = arr[i];
      PrimExpr new_elem = this->VisitExpr(old_elem);
      if (!new_elem.same_as(old_elem)) changed = true;
      new_arr[i] = new_elem;
      lanes = std::max(lanes, new_elem.ty().lanes());
    }

    for (size_t i = 0; i < arr.size(); ++i) {
      if (new_arr[i].ty().lanes() != lanes) {
        new_arr[i] = BroadcastTo(new_arr[i], lanes, false);
        changed = true;
      }
    }
    if (!changed) return arr;
    return ffi::Array<PrimExpr>(new_arr);
  }
  template <typename TOp, typename T>
  PrimExpr BinaryVec(const T* op) {
    static_assert(std::is_same<typename TOp::ContainerType, T>::value, "constraint");
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) && b.same_as(op->b)) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      int a_lanes = GetLanesOrVScaleFactor(a.ty());
      int b_lanes = GetLanesOrVScaleFactor(b.ty());
      int lanes = std::max(a_lanes, b_lanes);
      bool is_scalable = a.ty().IsScalableVector() || b.ty().IsScalableVector();
      return TOp(BroadcastTo(a, lanes, is_scalable), BroadcastTo(b, lanes, is_scalable));
    }
  }
  template <typename T, typename FCompute>
  PrimExpr AddSubVec(const T* op, FCompute fcompute) {
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) && b.same_as(op->b)) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      int a_lanes = GetLanesOrVScaleFactor(a.ty());
      int b_lanes = GetLanesOrVScaleFactor(b.ty());
      int lanes = std::max(a_lanes, b_lanes);
      if (lanes != 1) {
        const RampNode* b_ramp = b.as<RampNode>();
        const RampNode* a_ramp = a.as<RampNode>();
        if (a.ty().IsScalar() && b_ramp) {
          return Ramp(fcompute(a, b_ramp->base),
                      fcompute(IntImm(b_ramp->stride.ty(), 0), b_ramp->stride), b_ramp->lanes);
        }
        if (b.ty().IsScalar() && a_ramp) {
          return Ramp(fcompute(a_ramp->base, b), a_ramp->stride, a_ramp->lanes);
        }
      }
      bool is_scalable = a.ty().IsScalableVector() || b.ty().IsScalableVector();
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

      TVM_FFI_ICHECK(is_zero(op->min));
      // General calls still have vectorization paths that query a compile-time
      // lane count, so keep them on the existing fixed-width path for now.
      if (extent_as_int && extent_as_int->value > 1 && TargetHasRVV(target_) &&
          !ContainsCallNode(op->body)) {
        return VectorizeFixedLoopForRVV(op, extent_as_int->value);
      }

      if (!extent_as_int || extent_as_int->value < 1) {
        bool is_scalable_expr = CheckContains::ExprContains(op->extent, IsVScaleCall);
        TVM_FFI_ICHECK(is_scalable_expr && TargetHasVLA(target_))
            << "Failed to vectorize loop with extent " << op->extent << " for target " << target_;
      }
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
  Stmt VectorizeFixedLoopForRVV(const ForNode* op, int64_t extent) {
    // Match the existing TIRx scalable-vector convention.  LLVM/RVV still
    // selects the runtime vector length with vsetvli.
    static constexpr int kDefaultVScaleFactor = 4;
    PrimType index_dtype = op->loop_var.ty();
    PrimExpr zero = IntImm(index_dtype, 0);
    PrimExpr fixed_extent = IntImm(index_dtype, extent);
    PrimExpr scalable_lanes = CreateNewLanes(/*is_scalable=*/true, kDefaultVScaleFactor);
    PrimType lane_dtype = scalable_lanes.ty();
    PrimExpr scalable_lanes_index = scalable_lanes;
    if (scalable_lanes_index.ty() != index_dtype) {
      scalable_lanes_index = Cast(index_dtype, scalable_lanes_index);
    }
    PrimExpr num_chunks = ceildiv(fixed_extent, scalable_lanes_index);

    Var outer(op->loop_var->name_hint + ".vla.o", index_dtype);
    Var inner(op->loop_var->name_hint + ".vla.i", lane_dtype);
    PrimExpr inner_index = inner;
    if (inner_index.ty() != index_dtype) {
      inner_index = Cast(index_dtype, inner_index);
    }
    PrimExpr index = outer * scalable_lanes_index + inner_index;
    Stmt body = Substitute(op->body, {{op->loop_var, index}});
    Stmt guarded_body = IfThenElse(index < fixed_extent, body, std::nullopt, op->span);
    Stmt vector_loop = For(inner, IntImm(lane_dtype, 0), scalable_lanes, ForKind::kVectorized,
                           guarded_body, std::nullopt, op->annotations, std::nullopt, op->span);
    Stmt loop = For(outer, zero, num_chunks, ForKind::kSerial, vector_loop, std::nullopt, {},
                    std::nullopt, op->span);

    return this->VisitStmt(loop);
  }

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
  return CreatePrimFuncPass(pass_func, 0, "tirx.VectorizeLoop", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.VectorizeLoop", VectorizeLoop);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
