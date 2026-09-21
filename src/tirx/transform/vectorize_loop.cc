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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
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
using namespace tvm::prim;

namespace {
int GetLanesOrVScaleFactor(const PrimType& ty) {
  if (ty.IsScalableVector()) {
    return ty.VScaleFactor();
  }
  return ty.lanes();
}

// File-local helper: true if `expr` is a call to prim::builtin::vscale().
bool IsVScaleCall(const PrimExpr& expr) {
  if (const auto* call = expr.as<CallNode>()) {
    return call->op.same_as(prim::builtin::vscale());
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

PrimType GetTextureElementType(const Expr& texture) {
  const auto* pointer_type = texture->ty.as<PointerTypeNode>();
  TVM_FFI_ICHECK(pointer_type) << "Texture arguments must have PointerType";
  return pointer_type->element_type.as_or_throw<PrimType>();
}
}  // namespace

inline PrimExpr CreateNewLanes(bool is_scalable, int lanes_or_vscale_factor) {
  if (is_scalable) {
    return prim::Mul(Call(PrimType::Int(32), prim::builtin::vscale(), {}).as_or_throw<PrimExpr>(),
                     lanes_or_vscale_factor);
  } else {
    return lanes_or_vscale_factor;
  }
}

inline PrimExpr BroadcastTo(PrimExpr e, int lanes, bool is_scalable) {
  // Check if e is already in the expected form
  if (GetLanesOrVScaleFactor(e.ty()) == lanes && e.ty().IsScalableVector() == is_scalable) return e;

  if (const prim::BroadcastNode* op = e.as<prim::BroadcastNode>()) {
    TVM_FFI_ICHECK(op->ty.as_or_throw<PrimType>().IsScalableVector() == is_scalable)
        << "Can't broadcast between scalable and fixed length vectors.";
    int e_lanes = GetLanesOrVScaleFactor(op->ty.as_or_throw<PrimType>());

    if (lanes % e_lanes == 0) {
      return prim::Broadcast(op->value, CreateNewLanes(is_scalable, lanes));
    }
  }

  TVM_FFI_ICHECK(e.ty().IsScalar())
      << "Cannot broadcast lanes=" << GetLanesOrVScaleFactor(e.ty())
      << " is_scalable=" << e.ty().IsScalableVector() << " to " << lanes;

  return prim::Broadcast(e, CreateNewLanes(is_scalable, lanes));
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
 *  A_load = T.meta_var(T.call_intrin("float32x4", "tirx.masked_load", A,
 *                                    T.Ramp(i_0 * 4, 1, 4), predicate))
 *  T.evaluate(T.call_intrin("void", "tirx.masked_store", B, A_load,
 *                           T.Ramp(i_0 * 4, 1, 4), predicate))
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
    if (!condition->IsInstance<prim::LTNode>()) {
      return {false, stmt};
    }

    prim::LT lt = condition.as_or_throw<prim::LT>();

    // Check the form of the vectorized condition, we're expecting
    // Ramp(...) < Broadcast(...)
    if (!lt->a->IsInstance<prim::RampNode>() || !lt->b->IsInstance<prim::BroadcastNode>()) {
      return {false, stmt};
    }

    prim::Ramp pred_ramp = lt->a.as_or_throw<prim::Ramp>();
    base_ = pred_ramp->base;
    stride_ = pred_ramp->stride;
    lanes_ = pred_ramp->lanes;
    limit_ = lt->b.as_or_throw<prim::Broadcast>()->value;

    // Now we can try to predicate
    // Keep stmt intact when predication only succeeds for some accesses.
    Stmt predicated_stmt = Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
    if (num_accesses_analyzed_ > 0 && num_accesses_analyzed_ == num_accesses_rewritten_) {
      return {true, predicated_stmt};
    }
    return {false, stmt};
  }

 private:
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    auto load = StmtExprMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                    .as_or_throw<TensorLoad>();
    return TryPredicateBufferAccess(load);
  }

  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) final {
    auto store = StmtExprMutator::Mutate_(op, inplace_mode)
                     .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                     .as_or_throw<BufferStore>();
    return TryPredicateBufferAccess(store);
  }

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    Call call = StmtExprMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Expr>(op))
                    .as_or_throw<Call>();
    if (!call->op.same_as(builtin::masked_load()) && !call->op.same_as(builtin::masked_store())) {
      return call;
    }

    bool is_load = call->op.same_as(builtin::masked_load());
    ffi::Array<PrimExpr> indices;
    for (size_t i = is_load ? 1 : 2; i + 1 < call->args.size(); ++i) {
      indices.push_back(call->args[i].as_or_throw<PrimExpr>());
    }
    if (auto lane_mask = GetLaneMask(indices)) {
      PrimExpr predicate = call->args.back().as_or_throw<PrimExpr>();
      predicate = allow_offset_predication_ ? predicate & lane_mask.value() : lane_mask.value();
      ffi::Array<Expr> args = call->args;
      args.Set(args.size() - 1, predicate);
      return Call(call->ty, call->op, args, call->attrs, call->ty_args, call->span);
    }
    return call;
  }

  ffi::Optional<PrimExpr> GetLaneMask(const ffi::Array<PrimExpr>& indices) {
    num_accesses_analyzed_ += 1;

    // Do not try to predicate non-vectorized accesses
    if (!indices.size() || !indices[0]->IsInstance<prim::RampNode>()) {
      return std::nullopt;
    }
    prim::Ramp ramp = indices[0].as_or_throw<prim::Ramp>();

    if (!ffi::StructuralEqual()(ramp->stride, stride_) ||
        !ffi::StructuralEqual()(ramp->lanes, lanes_)) {
      return std::nullopt;
    }

    bool same_base = ffi::StructuralEqual()(ramp->base, base_);
    if (!same_base) {
      // The lane mask describes which lanes are active, independent of the
      // memory base.  This covers accesses such as A[offset + i] guarded by
      // a predicate over i.
      if (!allow_offset_predication_) {
        return std::nullopt;
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
    return lane_mask;
  }

  PrimExpr TryPredicateBufferAccess(TensorLoad load) {
    if (auto mask = GetLaneMask(load->indices)) {
      ffi::Array<Expr> args{load->source.as_or_throw<tvm::tirx::BufferVar>().var()};
      for (const PrimExpr& index : load->indices) args.push_back(index);
      args.push_back(mask.value());
      return Call(load->ty, builtin::masked_load(), args, {}, {}, load->span);
    }
    return load;
  }

  Stmt TryPredicateBufferAccess(BufferStore store) {
    if (auto mask = GetLaneMask(store->indices)) {
      ffi::Array<Expr> args{store->buffer.var(), store->value};
      for (const PrimExpr& index : store->indices) args.push_back(index);
      args.push_back(mask.value());
      return Evaluate(Call(PrimType::Void(), builtin::masked_store(), args, {}, {}, store->span),
                      store->span);
    }
    return store;
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

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    auto indices =
        Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad load = indices.UnchangedOrSameAs(op->indices)
                          ? ffi::GetRef<TensorLoad>(op)
                          : BufferLoad(op->source.as_or_throw<BufferVar>(),
                                       std::move(indices).ValueUnchecked(), op->span);
    return UpdateBufferAccess(load);
  }

  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) final {
    auto value = Mutate(op->value, inplace_mode);
    auto indices =
        Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    BufferStore store =
        value.UnchangedOrSameAs(op->value) && indices.UnchangedOrSameAs(op->indices)
            ? ffi::GetRef<BufferStore>(op)
            : BufferStore(op->buffer, std::move(value).ValueOrUnchanged(op->value),
                          std::move(indices).ValueOrUnchanged(op->indices), op->span);
    return UpdateBufferAccess(store);
  }

 private:
  template <typename Node>
  Node UpdateBufferAccess(Node node) {
    // Only update the buffer that's being replaced.
    if (node->buffer.get() != buf_) {
      return node;
    }

    // Find/make a BufferVar object with the correct updated shape.
    BufferVar buf;
    ffi::Any mapped = VarRemapGet(node->buffer);
    if (mapped != nullptr) {
      buf = mapped.as_or_throw<BufferVar>();
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
      ffi::Array<PrimExpr> strides = node->buffer->strides;
      for (size_t i = 0; i < strides.size(); i++) {
        PrimExpr stride = strides[i];
        if (i != strides.size() - 1) {
          stride *= var_lanes_;
        }
        strides.Set(i, analyzer_->Simplify(stride));
      }

      // Copy everything into the new buffer.
      auto type = CopyBufferType(node->buffer);
      type->shape = shape;
      type->strides = strides;
      buf = RebuildBufferVar(node->buffer, std::move(type));
      VarRemapSet(node->buffer, buf);
    }

    // Extend the last index by the number of lanes in the vectorized
    // variable.
    ffi::Array<PrimExpr> indices = node->indices;
    indices.Set(indices.size() - 1, analyzer_->Simplify(indices[indices.size() - 1] * var_lanes_ +
                                                        var_.as_or_throw<PrimExpr>()));

    auto writer = node.CopyOnWrite();
    writer->buffer = buf;
    writer->indices = indices;
    return node;
  }

  TensorLoad UpdateBufferAccess(TensorLoad node) {
    BufferVar buffer = node->source.as_or_throw<tvm::tirx::BufferVar>();
    if (buffer.get() != buf_) return node;
    BufferVar buf;
    auto mapped = VarRemapGet(buffer);
    if (mapped != nullptr) {
      buf = mapped.as_or_throw<BufferVar>();
    } else {
      ffi::Array<PrimExpr> shape = buffer->shape;
      shape.Set(shape.size() - 1, analyzer_->Simplify(shape.back() * var_lanes_));
      ffi::Array<PrimExpr> strides = buffer->strides;
      for (size_t i = 0; i < strides.size(); ++i) {
        PrimExpr stride = strides[i];
        if (i + 1 != strides.size()) stride *= var_lanes_;
        strides.Set(i, analyzer_->Simplify(stride));
      }
      auto type = CopyBufferType(buffer);
      type->shape = shape;
      type->strides = strides;
      buf = RebuildBufferVar(buffer, std::move(type));
      VarRemapSet(buffer, buf);
    }
    ffi::Array<PrimExpr> indices = node->indices;
    indices.Set(indices.size() - 1,
                analyzer_->Simplify(indices.back() * var_lanes_ + var_.as_or_throw<PrimExpr>()));
    return BufferLoad(buf, indices, node->span);
  }

  // buffer var
  const VarNode* buf_;
  // variable to be replaced
  Var var_;
  // the lanes.
  PrimExpr var_lanes_;
  // Analyzer for simplifications
  sym::Analyzer analyzer_;
};

// Vectorization supplies its own dtype-aware expression traversal.
class Vectorizer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  Vectorizer(Var var, PrimExpr var_lanes, Target target)
      : var_(var), var_lanes_(var_lanes), target_(target) {
    PrimType var_ty = var->ty.as_or_throw<PrimType>();
    ramp_ = prim::Ramp(IntImm(var_ty, 0), IntImm(var_ty, 1), var_lanes);
    VarRemapSet(var_, ramp_);
  }

  // Rewriting may change vector dtypes and later request scalarization of the
  // original statement.  Keep the complete speculative traversal out of place.
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    if (const auto* stmt = value.as<StmtNode>()) {
      TVM_FFI_ICHECK(!need_scalarize_);
      auto result = StmtExprMutator::Mutate(value, InplaceMode::kDisallow);
      if (need_scalarize_) {
        need_scalarize_ = false;
        return ffi::Any(Scalarize(ffi::GetRef<Stmt>(stmt)));
      }
      return result;
    }
    return StmtExprMutator::Mutate(value, InplaceMode::kDisallow);
  }

  UnchangedOr<Expr> Mutate_(const OpaqueExprNode* op, InplaceMode inplace_mode) final {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  UnchangedOr<Expr> Mutate_(const TupleNode* op, InplaceMode inplace_mode) final {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  UnchangedOr<Expr> Mutate_(const TupleGetItemNode* op, InplaceMode inplace_mode) final {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionNode* op, InplaceMode inplace_mode) final {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  UnchangedOr<Expr> Mutate_(const GlobalVarNode* op, InplaceMode inplace_mode) final {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  UnchangedOr<Expr> Mutate_(const OpNode* op, InplaceMode inplace_mode) final {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::AddNode* op, InplaceMode inplace_mode) final {
    return AddSubVec(op, [](PrimExpr a, PrimExpr b) { return a + b; }, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::SubNode* op, InplaceMode inplace_mode) final {
    return AddSubVec(op, [](PrimExpr a, PrimExpr b) { return a - b; }, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::MulNode* op, InplaceMode inplace_mode) final {
    auto a_update = this->Mutate(op->a, inplace_mode);
    bool a_unchanged = a_update.UnchangedOrSameAs(op->a);
    PrimExpr a = std::move(a_update).ValueOrUnchanged(op->a);
    auto b_update = this->Mutate(op->b, inplace_mode);
    bool b_unchanged = b_update.UnchangedOrSameAs(op->b);
    PrimExpr b = std::move(b_update).ValueOrUnchanged(op->b);
    if (a_unchanged && b_unchanged) {
      return ffi::Unchanged();
    } else {
      bool is_vec_a = a.ty().IsScalableVector() || a.ty().IsFixedLengthVector();
      bool is_vec_b = b.ty().IsScalableVector() || b.ty().IsFixedLengthVector();
      if (is_vec_a && is_vec_b) {
        // Let's not multiply scalable and fixed length vectors
        TVM_FFI_ICHECK(a.ty().IsScalableVector() == b.ty().IsScalableVector())
            << "Fixed length and scalable vectors can't be mixed in multiplication.";
      }
      if (is_vec_a || is_vec_b) {
        const prim::RampNode* b_ramp = b.as<prim::RampNode>();
        const prim::RampNode* a_ramp = a.as<prim::RampNode>();
        if (a_ramp && b.ty().IsScalar() && analyzer_->CanProve(b > 0)) {
          PrimExpr lanes = a_ramp->lanes;
          return prim::Ramp(a_ramp->base * b, a_ramp->stride * b, lanes);
        }
        if (b_ramp && a.ty().IsScalar() && analyzer_->CanProve(a > 0)) {
          PrimExpr lanes = b_ramp->lanes;
          return prim::Ramp(b_ramp->base * a, b_ramp->stride * a, lanes);
        }
        int a_lanes = GetLanesOrVScaleFactor(a.ty());
        int b_lanes = GetLanesOrVScaleFactor(b.ty());
        int max_lanes = std::max(a_lanes, b_lanes);
        bool is_scalable = a.ty().IsScalableVector() || b.ty().IsScalableVector();
        return prim::Mul(BroadcastTo(a, max_lanes, is_scalable),
                         BroadcastTo(b, max_lanes, is_scalable));
      }
    }
    return BinaryVec<prim::Mul>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::DivNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::Div>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::ModNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::Mod>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::FloorDivNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::FloorDiv>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::FloorModNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::FloorMod>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::MinNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::Min>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::MaxNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::Max>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::EQNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::EQ>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::NENode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::NE>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::LTNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::LT>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::LENode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::LE>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::GTNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::GT>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::GENode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::GE>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::AndNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::And>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::OrNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::Or>(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::LShiftNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::LShift>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::RShiftNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::RShift>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseAndNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::BitwiseAnd>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseOrNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::BitwiseOr>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseXorNode* op, InplaceMode inplace_mode) final {
    return BinaryVec<prim::BitwiseXor>(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseNotNode* op, InplaceMode inplace_mode) final {
    auto a = this->Mutate(op->a, inplace_mode);
    if (a.UnchangedOrSameAs(op->a)) return ffi::Unchanged();
    return prim::BitwiseNot(std::move(a).ValueOrUnchanged(op->a), op->span);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::NotNode* op, InplaceMode inplace_mode) final {
    auto a_update = this->Mutate(op->a, inplace_mode);
    bool a_unchanged = a_update.UnchangedOrSameAs(op->a);
    PrimExpr a = std::move(a_update).ValueOrUnchanged(op->a);
    if (a_unchanged) {
      return ffi::Unchanged();
    } else {
      return !(a);
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::RampNode* op, InplaceMode inplace_mode) final {
    PrimExpr base = this->Mutate(op->base, inplace_mode).ValueOrUnchanged(op->base);
    PrimExpr stride = this->Mutate(op->stride, inplace_mode).ValueOrUnchanged(op->stride);
    TVM_FFI_ICHECK(!base.ty().IsScalableVector())
        << "Creating scalable vectors from existing vectors is not supported.";
    TVM_FFI_ICHECK(!stride.ty().IsScalableVector())
        << "Ramp stride with scalable dtype is not supported";
    if (base.ty().IsFixedLengthVector() && stride.ty().IsScalar()) {
      TVM_FFI_ICHECK(op->lanes->IsInstance<IntImmNode>())
          << "Vectorizing over existing scalable vectors is not supported.";
      const prim::RampNode* base_ramp = base.as<prim::RampNode>();
      int op_lanes = op->lanes.as_or_throw<IntImm>()->value.as<int>().value();
      int base_ramp_lanes = base_ramp->lanes.as_or_throw<IntImm>()->value.as<int>().value();
      if (analyzer_->CanProve(base_ramp->stride ==
                              stride * MakeConst(stride.ty(), base_ramp_lanes))) {
        return prim::Ramp(base_ramp->base, stride, op_lanes * base_ramp_lanes);
      }
    }
    int lanes = std::max(base.ty().lanes(), stride.ty().lanes());
    base = BroadcastTo(base, lanes, false);
    stride = BroadcastTo(stride, lanes, false);
    ffi::Array<PrimExpr> elems;
    for (int i = 0; i < lanes; ++i) {
      elems.push_back(prim::Ramp(prim::Shuffle::ExtractElement(base, i),
                                 prim::Shuffle::ExtractElement(stride, i), op->lanes));
    }
    return prim::Shuffle::Concat(elems);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::BroadcastNode* op, InplaceMode inplace_mode) final {
    auto value_update = this->Mutate(op->value, inplace_mode);
    bool value_unchanged = value_update.UnchangedOrSameAs(op->value);
    PrimExpr value = std::move(value_update).ValueOrUnchanged(op->value);
    if (value.ty().IsScalableVector() || value.ty().IsFixedLengthVector()) {
      need_scalarize_ = true;
      return ffi::Unchanged();
    }
    if (value_unchanged) {
      return ffi::Unchanged();
    } else {
      return prim::Broadcast(op->value, op->lanes);
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::SelectNode* op, InplaceMode inplace_mode) final {
    auto cond_update = this->Mutate(op->condition, inplace_mode);
    bool cond_unchanged = cond_update.UnchangedOrSameAs(op->condition);
    PrimExpr cond = std::move(cond_update).ValueOrUnchanged(op->condition);
    auto t_update = this->Mutate(op->true_value, inplace_mode);
    bool t_unchanged = t_update.UnchangedOrSameAs(op->true_value);
    PrimExpr t = std::move(t_update).ValueOrUnchanged(op->true_value);
    auto f_update = this->Mutate(op->false_value, inplace_mode);
    bool f_unchanged = f_update.UnchangedOrSameAs(op->false_value);
    PrimExpr f = std::move(f_update).ValueOrUnchanged(op->false_value);
    if (cond_unchanged && t_unchanged && f_unchanged) {
      return ffi::Unchanged();
    } else {
      int cond_lanes = GetLanesOrVScaleFactor(cond.ty());
      int t_lanes = GetLanesOrVScaleFactor(t.ty());
      int f_lanes = GetLanesOrVScaleFactor(f.ty());
      int lanes = std::max(std::max(cond_lanes, t_lanes), f_lanes);
      bool is_scalable =
          cond.ty().IsScalableVector() || t.ty().IsScalableVector() || f.ty().IsScalableVector();
      return prim::Select(BroadcastTo(cond, lanes, is_scalable), BroadcastTo(t, lanes, is_scalable),
                          BroadcastTo(f, lanes, is_scalable));
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::CastNode* op, InplaceMode inplace_mode) final {
    auto value_update = this->Mutate(op->value, inplace_mode);
    bool value_unchanged = value_update.UnchangedOrSameAs(op->value);
    PrimExpr value = std::move(value_update).ValueOrUnchanged(op->value);
    if (value_unchanged) {
      return ffi::Unchanged();
    } else {
      if (value.ty().IsScalableVector()) {
        return prim::Cast(PrimType::ScalableVector(op->ty.as_or_throw<PrimType>().code(),
                                                   op->ty.as_or_throw<PrimType>().bits(),
                                                   value.ty().VScaleFactor()),
                          value);
      } else {
        return prim::Cast(op->ty.as_or_throw<PrimType>().WithLanes(value.ty().lanes()), value);
      }
    }
  }

  // IfThenElse expr
  PrimExpr MutateIfThenElseExpr_(const CallNode* op, InplaceMode inplace_mode) {
    PrimExpr cond = this->Mutate(op->args[0].as_or_throw<PrimExpr>())
                        .ValueOrUnchanged(op->args[0].as_or_throw<PrimExpr>());
    if (cond.ty().IsScalableVector() || cond.ty().IsFixedLengthVector()) {
      need_scalarize_ = true;
      return ffi::GetRef<Call>(op).as_or_throw<PrimExpr>();
    }
    PrimExpr t = this->Mutate(op->args[1].as_or_throw<PrimExpr>())
                     .ValueOrUnchanged(op->args[1].as_or_throw<PrimExpr>());
    PrimExpr f = this->Mutate(op->args[2].as_or_throw<PrimExpr>())
                     .ValueOrUnchanged(op->args[2].as_or_throw<PrimExpr>());
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
  PrimExpr MutateReinterpretExpr_(const CallNode* op, InplaceMode inplace_mode) {
    TVM_FFI_ICHECK(op->op.same_as(builtin::reinterpret()));
    PrimExpr input = op->args[0].as_or_throw<PrimExpr>();
    PrimExpr value = this->Mutate(input).ValueOrUnchanged(input);
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
  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    auto optional_ret_ty = op->ty.as<PrimType>();
    if (!optional_ret_ty) {
      // Non-primitive calls are not vectorized themselves.  Visit their general Expr operands
      // to preserve pointer values and rewrite scalar values, scalarizing the surrounding
      // statement if rewriting produces a vector operand.
      ffi::Array<Expr> new_args;
      for (const Expr& arg : op->args) {
        Expr new_arg = this->Mutate(arg).ValueOrUnchanged(arg);
        if (auto prim_arg = new_arg.as<PrimExpr>();
            prim_arg && (prim_arg.value().ty().IsScalableVector() ||
                         prim_arg.value().ty().IsFixedLengthVector())) {
          need_scalarize_ = true;
          return ffi::Unchanged();
        }
        new_args.push_back(new_arg);
      }
      if (op->args.same_as(new_args)) {
        return ffi::Unchanged();
      }
      return Call(op->ty, op->op, new_args, op->attrs, {}, op->span);
    }
    PrimType ret_ty = optional_ret_ty.value();
    if (op->op.same_as(prim::builtin::if_then_else())) {
      return MutateIfThenElseExpr_(op, inplace_mode);
    } else if (op->op.same_as(builtin::texture2d_load())) {
      int lane = 0;
      ffi::Array<PrimExpr> fcd =
          MutateArray({op->args.back().as_or_throw<PrimExpr>()}, &lane, inplace_mode);
      PrimType dtype = GetTextureElementType(op->args[0]);
      TVM_FFI_ICHECK(lane * dtype.bits() <= op->args[4].as<IntImmNode>()->value)
          << "Expected Data to be Read is lesser than or equal to Texture Load length";

      auto new_args = op->args;
      new_args.pop_back();
      new_args.push_back(fcd[0]);
      return Call(ret_ty.WithLanes(lane), op->op, new_args, op->attrs, {}, op->span);
    } else if (op->op.same_as(builtin::texture2d_store())) {
      int lane = 0;
      // Vectorize the value to store
      ffi::Array<PrimExpr> value{op->args.back().as_or_throw<PrimExpr>()};
      ffi::Array<PrimExpr> mutated_value = MutateArray(value, &lane, inplace_mode);
      PrimType dtype = GetTextureElementType(op->args[0]);
      TVM_FFI_ICHECK(lane * dtype.bits() == op->args[4].as<IntImmNode>()->value)
          << "Expected Data to be Written equal to Texture Store length";
      ffi::Array<Expr> new_args = op->args;
      new_args.Set(new_args.size() - 1, mutated_value[0]);
      return Call(ret_ty, op->op, new_args, op->attrs, {}, op->span);
    } else if (op->op.same_as(builtin::reinterpret())) {
      return MutateReinterpretExpr_(op, inplace_mode);
    }
    auto optional_op = op->op.as<Op>();
    bool vectorizable = optional_op && op_vectorizable_.get(optional_op.value(), false) &&
                        !ret_ty.IsScalableVector();

    if (!vectorizable) {
      // Cannot vectorize this op
      ffi::Array<Expr> new_args;
      for (const Expr& arg : op->args) {
        Expr new_arg = this->Mutate(arg).ValueOrUnchanged(arg);
        if (auto prim_arg = new_arg.as<PrimExpr>();
            prim_arg && (prim_arg.value().ty().IsScalableVector() ||
                         prim_arg.value().ty().IsFixedLengthVector())) {
          need_scalarize_ = true;
          return ffi::Unchanged();
        }
        new_args.push_back(new_arg);
      }
      if (op->args.same_as(new_args)) {
        return ffi::Unchanged();
      } else {
        return Call(ret_ty, op->op, new_args, op->attrs, {}, op->span);
      }
    } else {
      int lane = 0;
      ffi::Array<Expr> new_args;
      if (op->op.same_as(builtin::call_llvm_pure_intrin())) {
        // op->args[1], will give us total number of arguments to intrinsic
        ffi::Array<Expr> op_expr_args;
        for (size_t i = 1; i < op->args.size(); ++i) {
          // Collect all intrinsic arguments
          op_expr_args.push_back(op->args[i]);
        }
        // Generate RAMP nodes for intrinsic arguments
        ffi::Array<Expr> updated_args = MutateCallArgs(op_expr_args, &lane, inplace_mode);
        new_args.push_back(op->args[0]);
        // Collect updated intrinsic arguments
        for (size_t i = 0; i < updated_args.size(); ++i) {
          new_args.push_back(updated_args[i]);
        }
      } else {
        new_args = MutateCallArgs(op->args, &lane, inplace_mode);
      }
      // normal code path.
      if (op->args.same_as(new_args)) {
        return ffi::Unchanged();
      } else {
        return Call(ret_ty.WithLanes(lane), op->op, new_args, op->attrs, {}, op->span);
      }
    }
  }
  // BufferLoad
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    auto load = ffi::GetRef<TensorLoad>(op);

    auto fmutate = [this](const PrimExpr& index) {
      return this->Mutate(index).ValueOrUnchanged(index);
    };
    ffi::Array<PrimExpr> indices = op->indices.Map(fmutate);

    if (!indices.same_as(op->indices)) {
      return BufferLoad(op->source.as_or_throw<tvm::tirx::BufferVar>(), indices, op->span);
    }

    return load;
  }
  // Let
  UnchangedOr<PrimExpr> Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) final {
    auto value_update = this->Mutate(op->value, inplace_mode);
    bool value_unchanged = value_update.UnchangedOrSameAs(op->value);
    PrimExpr value = std::move(value_update).ValueOrUnchanged(op->value);
    // Weaker SSA condition
    // A single var can be binded in multiple lets
    // but they have to bind to the same value.
    // This is used to allow cases when we reuse a single let
    // expression to cosntruct a nested expr.
    // (let x = 1 in x + 1) * (let x = 1 in x + 1)
    auto mapped = VarRemapGet(op->var);
    if (mapped != nullptr) {
      TVM_FFI_ICHECK(deep_equal_(mapped.as_or_throw<PrimExpr>(), value))
          << "Let cannot bind the same var to two different values";
    }
    if (GetLanesOrVScaleFactor(value.ty()) != GetLanesOrVScaleFactor(op->value.ty())) {
      Var new_var(op->var->name, value.ty());
      VarRemapSet(op->var, new_var);
      return prim::Let(new_var, value,
                       this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body));
    } else {
      VarRemapSet(op->var, op->var);
      auto body_update = this->Mutate(op->body, inplace_mode);
      bool body_unchanged = body_update.UnchangedOrSameAs(op->body);
      PrimExpr body = std::move(body_update).ValueOrUnchanged(op->body);
      if (value_unchanged && body_unchanged) {
        return ffi::Unchanged();
      } else {
        return prim::Let(op->var, value, body);
      }
    }
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::ShuffleNode* op, InplaceMode inplace_mode) final {
    TVM_FFI_ICHECK(op->vectors.size() == 1 && op->indices.size() == 1)
        << "Cannot vectorize ShuffleNode with multiple vectors or indices: the vector size is "
        << op->vectors.size() << " and the index size is " << op->indices.size();
    int lane_vectors = 0;
    int lane_indices = 0;
    ffi::Array<PrimExpr> vectors = MutateArray(op->vectors, &lane_vectors, inplace_mode);
    ffi::Array<PrimExpr> indices = MutateArray(op->indices, &lane_indices, inplace_mode);
    if (vectors.same_as(op->vectors) && indices.same_as(op->indices)) {
      return ffi::Unchanged();
    }

    int new_vec_length =
        var_lanes_.as_or_throw<IntImm>()->value.as<int>().value() / op->vectors[0].ty().lanes();
    PrimExpr updated_index = indices[0];
    // Check that the indices satisfy the specific patterns.
    auto f_check_index = [this, op](const PrimExpr& index) {
      // Allowing Ramp(0, 1, var_lanes_)
      if (const auto* ramp = index.as<prim::RampNode>()) {
        if (ramp->base->IsInstance<IntImmNode>() && ramp->base.as_or_throw<IntImm>()->value == 0 &&
            ramp->stride->IsInstance<IntImmNode>() &&
            ramp->stride.as_or_throw<IntImm>()->value == 1 &&
            ramp->lanes->IsInstance<IntImmNode>() &&
            ramp->lanes.as_or_throw<IntImm>()->value == var_lanes_.as_or_throw<IntImm>()->value) {
          return true;
        }
      }
      // Allowing FloorMod(Ramp(0, 1, var_lanes_), Broadcast(op->vectors[0]->lanes, var_lanes_))
      if (const auto* floordiv = index.as<prim::FloorModNode>()) {
        if (const auto* ramp = floordiv->a.as<prim::RampNode>()) {
          if (const auto* broadcast = floordiv->b.as<prim::BroadcastNode>()) {
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
      PrimType var_ty = var_->ty.as_or_throw<PrimType>();
      auto f_substitute = [old_var = var_, replacement = tvm::IntImm(var_ty, 0)](
                              const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (var.same_as(old_var)) return ffi::Any(replacement);
        return ffi::Unchanged();
      };
      return ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(op->vectors[0], f_substitute)
          .as_or_throw<PrimExpr>();
    } else {
      PrimExpr prev_ramp = ramp_;
      PrimExpr prev_var_lanes = var_lanes_;
      PrimType var_ty = var_->ty.as_or_throw<PrimType>();
      ramp_ = prim::Ramp(IntImm(var_ty, 0), IntImm(var_ty, 2), new_vec_length);
      var_lanes_ = tvm::IntImm(var_lanes_.ty(), new_vec_length);
      lane_vectors = 0;
      vectors = MutateArray(op->vectors, &lane_vectors, inplace_mode);
      ramp_ = prev_ramp;
      var_lanes_ = prev_var_lanes;
      return vectors[0];
    }
  }
  // BufferStore
  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) final {
    auto store = ffi::GetRef<BufferStore>(op);

    auto fmutate = [this](const PrimExpr& index) {
      return this->Mutate(index).ValueOrUnchanged(index);
    };
    ffi::Array<PrimExpr> indices = op->indices.Map(fmutate);

    auto value_update = this->Mutate(op->value, inplace_mode);
    bool value_unchanged = value_update.UnchangedOrSameAs(op->value);
    PrimExpr value = std::move(value_update).ValueOrUnchanged(op->value);

    if (!indices.same_as(op->indices) || !value_unchanged) {
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
          << "When storing to buffer " << op->buffer.name() << ", cannot produce " << total_lanes
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
  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    if (op->kind == ForKind::kVectorized) {
      LOG(WARNING) << "Detect vectorize inside vectorized loop, ignoring...";
    }
    TVM_FFI_ICHECK(is_zero(op->min));
    TVM_FFI_ICHECK(!op->extent.ty().IsScalableVector() && !op->extent.ty().IsFixedLengthVector());
    auto extent_update = this->Mutate(op->extent, inplace_mode);
    bool extent_unchanged = extent_update.UnchangedOrSameAs(op->extent);
    PrimExpr extent = std::move(extent_update).ValueOrUnchanged(op->extent);
    if (extent.ty().IsScalableVector() || extent.ty().IsFixedLengthVector()) {
      return Scalarize(ffi::GetRef<Stmt>(op));
    }
    auto body_update = this->Mutate(op->body, inplace_mode);
    bool body_unchanged = body_update.UnchangedOrSameAs(op->body);
    Stmt body = std::move(body_update).ValueOrUnchanged(op->body);
    if (extent_unchanged && body_unchanged) {
      return ffi::Unchanged();
    } else {
      auto n = ffi::make_object<ForNode>(*op);
      n->extent = extent;
      n->body = body;
      return For(n);
    }
  }
  // IfThenElse
  UnchangedOr<Stmt> Mutate_(const IfThenElseNode* op, InplaceMode inplace_mode) final {
    TVM_FFI_ICHECK(!op->condition.ty().IsScalableVector() &&
                   !op->condition.ty().IsFixedLengthVector());
    auto condition_update = this->Mutate(op->condition, inplace_mode);
    bool condition_unchanged = condition_update.UnchangedOrSameAs(op->condition);
    PrimExpr condition = std::move(condition_update).ValueOrUnchanged(op->condition);
    // need scalarize can be marked as true during visit of condition
    bool cond_need_scalarize = false;
    std::swap(cond_need_scalarize, need_scalarize_);
    // Clear the flag while recursively rewriting branch statements.
    auto then_case_update = this->Mutate(op->then_case, inplace_mode);
    bool then_case_unchanged = then_case_update.UnchangedOrSameAs(op->then_case);
    Stmt then_case = std::move(then_case_update).ValueOrUnchanged(op->then_case);
    ffi::Optional<Stmt> else_case = std::nullopt;
    if (op->else_case) {
      else_case =
          this->Mutate(op->else_case.value(), inplace_mode).ValueOrUnchanged(op->else_case.value());
    }
    // Check if we can rewrite the condition with predicated buffers
    if (EnableBufferLevelPredication(target_) &&
        (condition.ty().IsScalableVector() || condition.ty().IsFixedLengthVector()) &&
        !else_case.has_value()) {
      std::pair<bool, Stmt> success_stmt_pair =
          ffi::make_object<TryPredicateBufferAccesses>(TargetHasRVV(target_))
              ->Run(then_case, condition);
      bool can_remove_if_then_else = success_stmt_pair.first;
      if (can_remove_if_then_else) {
        return success_stmt_pair.second;
      }
    }

    if (cond_need_scalarize || condition.ty().IsScalableVector() ||
        condition.ty().IsFixedLengthVector()) {
      return Scalarize(ffi::GetRef<Stmt>(op));
    }
    if (condition_unchanged && then_case_unchanged && else_case.same_as(op->else_case)) {
      return ffi::Unchanged();
    } else {
      return IfThenElse(condition, then_case, else_case);
    }
  }
  // While
  UnchangedOr<Stmt> Mutate_(const WhileNode* op, InplaceMode inplace_mode) final {
    TVM_FFI_THROW(InternalError) << "A while loop inside a vectorized loop not supported.";
    TVM_FFI_UNREACHABLE();
  }
  // Bind
  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final {
    auto prim_value = op->value.as<PrimExpr>();
    if (!prim_value) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    PrimExpr value =
        this->Mutate(prim_value.value(), inplace_mode).ValueOrUnchanged(prim_value.value());
    // if visit of value triggers need scalarize
    // we need to scalarize the let
    if (need_scalarize_) {
      need_scalarize_ = false;
      return Scalarize(ffi::GetRef<Stmt>(op));
    }
    TVM_FFI_ICHECK(VarRemapGet(op->var) == nullptr)
        << "SSA violation, a single var is binded twice";

    if (GetLanesOrVScaleFactor(value.ty()) != GetLanesOrVScaleFactor(prim_value.value().ty())) {
      Var new_var(op->var->name, value.ty());
      VarRemapSet(op->var, new_var);
      return Bind(new_var, value);
    } else {
      VarRemapSet(op->var, op->var);
      if (value.same_as(op->value)) {
        return ffi::Unchanged();
      } else {
        return Bind(op->var, value, op->span);
      }
    }
  }
  // scalarize the statment
  Stmt Scalarize(Stmt stmt) {
    PrimType var_ty = var_->ty.as_or_throw<PrimType>();
    Var idx(var_->name + ".s", var_ty);
    auto substituter = ffi::make_object<StmtExprMutator>();
    substituter->VarRemapSet(var_, idx);
    stmt = substituter->Mutate(stmt).ValueOrUnchanged(stmt);
    return For(idx.as_or_throw<PrimVar>(), IntImm(var_ty, 0), var_lanes_, ForKind::kSerial, stmt);
  }

 private:
  // analyzer
  sym::Analyzer analyzer_;
  // deep equal
  prim::ExprDeepEqual deep_equal_;
  // variable to be replaced
  Var var_;
  // the lanes.
  PrimExpr var_lanes_;
  // ramp representing the var.
  PrimExpr ramp_;
  // flag to mark requirment of scalarization.
  bool need_scalarize_{false};
  // vectorizable property
  OpAttrMap<TVectorizable> op_vectorizable_ = Op::GetAttrMap<TVectorizable>("TVectorizable");
  /*! \brief The current target context. */
  Target target_;

  // mutate array, with given lane requirement
  // when finished, p_lane updates the lane requirement.
  ffi::Array<PrimExpr> MutateArray(const ffi::Array<PrimExpr>& arr, int* p_lanes,
                                   InplaceMode inplace_mode) {
    if (arr.size() == 0) return arr;
    int& lanes = *p_lanes;
    bool changed = false;
    std::vector<PrimExpr> new_arr(arr.size());
    for (size_t i = 0; i < arr.size(); i++) {
      PrimExpr old_elem = arr[i];
      auto new_elem_update = this->Mutate(old_elem);
      bool new_elem_unchanged = new_elem_update.UnchangedOrSameAs(old_elem);
      PrimExpr new_elem = std::move(new_elem_update).ValueOrUnchanged(old_elem);
      if (!new_elem_unchanged) changed = true;
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

  // Mutate primitive call operands together so their lanes stay aligned, while
  // preserving pointer-valued operands as general Expr.  Pointer expressions
  // may themselves request scalarization if an address depends on the
  // vectorized loop variable.
  ffi::Array<Expr> MutateCallArgs(const ffi::Array<Expr>& arr, int* p_lanes,
                                  InplaceMode inplace_mode) {
    if (arr.empty()) return arr;
    int& lanes = *p_lanes;
    bool changed = false;
    std::vector<Expr> new_arr(arr.size());
    for (size_t i = 0; i < arr.size(); ++i) {
      const Expr& old_elem = arr[i];
      auto new_elem_update = this->Mutate(old_elem);
      bool new_elem_unchanged = new_elem_update.UnchangedOrSameAs(old_elem);
      Expr new_elem = std::move(new_elem_update).ValueOrUnchanged(old_elem);
      changed = changed || !new_elem_unchanged;
      new_arr[i] = new_elem;
      if (auto prim_elem = new_elem.as<PrimExpr>()) {
        lanes = std::max(lanes, prim_elem.value().ty().lanes());
      }
    }
    for (size_t i = 0; i < new_arr.size(); ++i) {
      if (auto prim_elem = new_arr[i].as<PrimExpr>();
          prim_elem && prim_elem.value().ty().lanes() != lanes) {
        new_arr[i] = BroadcastTo(prim_elem.value(), lanes, false);
        changed = true;
      }
    }
    if (!changed) return arr;
    return ffi::Array<Expr>(new_arr);
  }
  template <typename TOp, typename T>
  PrimExpr BinaryVec(const T* op, InplaceMode inplace_mode) {
    static_assert(std::is_same<typename TOp::ContainerType, T>::value, "constraint");
    auto a_update = this->Mutate(op->a, inplace_mode);
    bool a_unchanged = a_update.UnchangedOrSameAs(op->a);
    PrimExpr a = std::move(a_update).ValueOrUnchanged(op->a);
    auto b_update = this->Mutate(op->b, inplace_mode);
    bool b_unchanged = b_update.UnchangedOrSameAs(op->b);
    PrimExpr b = std::move(b_update).ValueOrUnchanged(op->b);
    if (a_unchanged && b_unchanged) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      int a_lanes = GetLanesOrVScaleFactor(a.ty());
      int b_lanes = GetLanesOrVScaleFactor(b.ty());
      int lanes = std::max(a_lanes, b_lanes);
      bool is_scalable = a.ty().IsScalableVector() || b.ty().IsScalableVector();
      return TOp(BroadcastTo(a, lanes, is_scalable), BroadcastTo(b, lanes, is_scalable), op->span);
    }
  }
  template <typename T, typename FCompute>
  PrimExpr AddSubVec(const T* op, FCompute fcompute, InplaceMode inplace_mode) {
    auto a_update = this->Mutate(op->a, inplace_mode);
    bool a_unchanged = a_update.UnchangedOrSameAs(op->a);
    PrimExpr a = std::move(a_update).ValueOrUnchanged(op->a);
    auto b_update = this->Mutate(op->b, inplace_mode);
    bool b_unchanged = b_update.UnchangedOrSameAs(op->b);
    PrimExpr b = std::move(b_update).ValueOrUnchanged(op->b);
    if (a_unchanged && b_unchanged) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      int a_lanes = GetLanesOrVScaleFactor(a.ty());
      int b_lanes = GetLanesOrVScaleFactor(b.ty());
      int lanes = std::max(a_lanes, b_lanes);
      if (lanes != 1) {
        const prim::RampNode* b_ramp = b.as<prim::RampNode>();
        const prim::RampNode* a_ramp = a.as<prim::RampNode>();
        if (a.ty().IsScalar() && b_ramp) {
          return prim::Ramp(fcompute(a, b_ramp->base),
                            fcompute(IntImm(b_ramp->stride.ty(), 0), b_ramp->stride),
                            b_ramp->lanes);
        }
        if (b.ty().IsScalar() && a_ramp) {
          return prim::Ramp(fcompute(a_ramp->base, b), a_ramp->stride, a_ramp->lanes);
        }
      }
      bool is_scalable = a.ty().IsScalableVector() || b.ty().IsScalableVector();
      return fcompute(BroadcastTo(a, lanes, is_scalable), BroadcastTo(b, lanes, is_scalable));
    }
  }
};

class LoopVectorizer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    if (value.as<ExprNode>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  explicit LoopVectorizer(DictAttrs attrs) {
    if (auto opt_target = attrs.GetAttr<Target>(tvm::attr::kTarget)) {
      target_ = opt_target.value();
    }
  }

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    if (op->kind == ForKind::kVectorized) {
      auto* extent_as_int = op->extent.as<IntImmNode>();

      TVM_FFI_ICHECK(is_zero(op->min));
      // General calls still have vectorization paths that query a compile-time
      // lane count, so keep them on the existing fixed-width path for now.
      if (extent_as_int && extent_as_int->value > 1 && TargetHasRVV(target_) &&
          !ContainsCallNode(op->body)) {
        return VectorizeFixedLoopForRVV(op, extent_as_int->value.as<int>().value());
      }

      if (!extent_as_int || extent_as_int->value < 1) {
        bool is_scalable_expr = CheckContains::ExprContains(op->extent, IsVScaleCall);
        TVM_FFI_ICHECK(is_scalable_expr && TargetHasVLA(target_))
            << "Failed to vectorize loop with extent " << op->extent << " for target " << target_;
      }
      return ffi::make_object<Vectorizer>(op->loop_var, op->extent, target_)
          ->Mutate(op->body, inplace_mode)
          .ValueOrUnchanged(op->body);
    } else {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
  }

  UnchangedOr<Stmt> Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) final {
    if (op->attr_key == tvm::attr::kTarget) {
      Target previous_target = target_;
      target_ = op->node.as<Target>().value();
      auto result = StmtExprMutator::Mutate_(op, inplace_mode);
      target_ = previous_target;
      return result;
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
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
      scalable_lanes_index = prim::Cast(index_dtype, scalable_lanes_index);
    }
    PrimExpr num_chunks = ceildiv(fixed_extent, scalable_lanes_index);

    PrimVar outer(op->loop_var->name + ".vla.o", index_dtype);
    PrimVar inner(op->loop_var->name + ".vla.i", lane_dtype);
    PrimExpr inner_index = inner;
    if (inner_index.ty() != index_dtype) {
      inner_index = prim::Cast(index_dtype, inner_index);
    }
    PrimExpr index = outer * scalable_lanes_index + inner_index;
    auto substituter = ffi::make_object<StmtExprMutator>();
    substituter->VarRemapSet(op->loop_var, index);
    Stmt body = substituter->Mutate(op->body).ValueOrUnchanged(op->body);
    Stmt guarded_body = IfThenElse(index < fixed_extent, body, std::nullopt, op->span);
    Stmt vector_loop = For(inner, IntImm(lane_dtype, 0), scalable_lanes, ForKind::kVectorized,
                           guarded_body, op->annotations, std::nullopt, op->span);
    Stmt loop =
        For(outer, zero, num_chunks, ForKind::kSerial, vector_loop, {}, std::nullopt, op->span);

    return this->Mutate(loop, InplaceMode::kDisallow).ValueOrUnchanged(loop);
  }

  Target target_ = Target::Current();
};

class VectorizeSkipper : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    if (value.as<ExprNode>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = stmt.as<ForNode>();
    if (op->kind == ForKind::kVectorized) {
      return For(op->loop_var, op->min, op->extent, ForKind::kSerial, op->body);
    } else {
      return stmt;
    }
  }
};

Stmt SkipVectorize(Stmt stmt) {
  return ffi::make_object<VectorizeSkipper>()
      ->Mutate(stmt, InplaceMode::kAllow)
      .ValueOrUnchanged(std::move(stmt));
}

namespace transform {

// TODO(tvm-team): Make it as a target property.
Pass VectorizeLoop(bool enable_vectorize) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    if (enable_vectorize) {
      n->body = ffi::make_object<LoopVectorizer>(n->attrs)
                    ->Mutate(n->body, InplaceMode::kAllow)
                    .ValueOrUnchanged(n->body);
    } else {
      n->body = ffi::make_object<VectorizeSkipper>()
                    ->Mutate(n->body, InplaceMode::kAllow)
                    .ValueOrUnchanged(n->body);
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
