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
 * \file unsupported_dtype_legalize.cc
 * \brief legalize bf16/fp8 type by adding cast_to_fp32
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <cmath>
#include <tuple>

#include "dtype_conversion.h"

namespace tvm {
namespace tirx {
using namespace tvm::prim;

namespace {

bool IsBFloat16Type(const PrimType& type) {
  return type.MatchesElementType(DLDataTypeCode::kDLBfloat, 16);
}

bool IsFloat8Type(const PrimType& type) {
  DLDataTypeCode code = type.code();
  return code == DLDataTypeCode::kDLFloat8_e3m4 || code == DLDataTypeCode::kDLFloat8_e4m3 ||
         code == DLDataTypeCode::kDLFloat8_e4m3b11fnuz ||
         code == DLDataTypeCode::kDLFloat8_e4m3fn || code == DLDataTypeCode::kDLFloat8_e4m3fnuz ||
         code == DLDataTypeCode::kDLFloat8_e5m2 || code == DLDataTypeCode::kDLFloat8_e5m2fnuz ||
         code == DLDataTypeCode::kDLFloat8_e8m0fnu;
}

template <typename F>
bool MatchPrimType(const Type& type, F f) {
  if (const auto* prim_type = type.as<PrimTypeNode>()) {
    return f(ffi::GetRef<PrimType>(prim_type));
  }
  return false;
}

}  // namespace

// NOTE: do not touch buffer on function boundary
// remap internal fp8/bf16 buffer to f32 if they meet the following condition
// - constant allocation size
// - do not have raw pointer access to the buffer
//
// populate candidate variable replacements before opaque-access filtering.
class ComputeLegalizePlanner : public StmtExprVisitor {
 public:
  explicit ComputeLegalizePlanner(PrimType promote_dtype) : promote_dtype_(promote_dtype) {}

  void Plan(PrimFunc func) {
    this->Visit(func->body);
    // A later opaque access can veto an earlier allocation candidate.
    for (const Var& var : opaque_var_access_) {
      compute_var_remap_.erase(var);
    }
  }

  void SeedRemaps(StmtExprMutator* mutator) const {
    for (const auto& [var, replacement] : compute_var_remap_) {
      mutator->VarRemapSet(var, replacement);
    }
  }

  virtual bool MatchType(const Type& type) const = 0;

  ffi::Optional<VisitInterrupt> Visit_(const AllocBufferNode* op) final {
    // remap all intermediate constant buffer to promote data types (fp16/fp32)
    if (MatchType(op->buffer->dtype)) {
      PrimType dtype = promote_dtype_.WithLanes(op->buffer->dtype.lanes());
      auto type = CopyBufferType(op->buffer);
      type->dtype = dtype;
      BufferVar buffer_var = RebuildBufferVar(op->buffer, std::move(type));
      compute_var_remap_[op->buffer.var()] = buffer_var.var();
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const CallNode* op) final {
    if (op->op.same_as(builtin::buffer_data()) && op->args.size() == 1) {
      if (auto buffer = op->args[0].as<Var>()) {
        opaque_var_access_.insert(buffer.value());
      }
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarNode* op) final {
    if (op->ty.as<PointerTypeNode>()) {
      opaque_var_access_.insert(ffi::GetRef<Var>(op));
    }
    return std::nullopt;
  }

 private:
  std::unordered_map<Var, Var> compute_var_remap_;
  std::unordered_set<Var> opaque_var_access_;
  PrimType promote_dtype_;
};

class BF16ComputeLegalizePlanner : public ComputeLegalizePlanner {
 public:
  using ComputeLegalizePlanner::ComputeLegalizePlanner;
  bool MatchType(const Type& type) const {
    return MatchPrimType(type, [](const PrimType& prim_type) { return IsBFloat16Type(prim_type); });
  }
};

class FP8ComputeLegalizePlanner : public ComputeLegalizePlanner {
 public:
  using ComputeLegalizePlanner::ComputeLegalizePlanner;
  bool MatchType(const Type& type) const {
    return MatchPrimType(type, [](const PrimType& prim_type) { return IsFloat8Type(prim_type); });
  }
};

#define DEFINE_BIOP_EXPR_LEGALIZE(OP, FUNC)                                         \
  UnchangedOr<PrimExpr> Mutate_(const OP* op, InplaceMode inplace_mode) final {     \
    PrimExpr origin_a =                                                             \
        PromoteToTarget(this->Mutate(op->a, inplace_mode).ValueOrUnchanged(op->a)); \
    PrimExpr origin_b =                                                             \
        PromoteToTarget(this->Mutate(op->b, inplace_mode).ValueOrUnchanged(op->b)); \
                                                                                    \
    if (origin_a.same_as(op->a) && origin_b.same_as(op->b)) {                       \
      return ffi::Unchanged();                                                      \
    } else {                                                                        \
      return FUNC(origin_a, origin_b);                                              \
    }                                                                               \
  }

// NOTE: Legalize the FP8/BF16 computations
// to floating point computations and only keeps the
// fp8/bf16 storage which can further be legalized by FP8/BF16StorageLegalizer
// FP8/BF16StorageLegalizer will be called at a much later time
// point in the TIR lowering phases.
class ComputeLegalizer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit ComputeLegalizer(PrimType promote_dtype) : promote_dtype_(promote_dtype) {}

  PrimFunc LegalizeWithPlanner(PrimFunc func, ComputeLegalizePlanner* planner) {
    planner->Plan(func);
    planner->SeedRemaps(this);
    auto* n = func.CopyOnWrite();
    n->body = this->Mutate(n->body, InplaceMode::kDisallow).ValueOrUnchanged(n->body);
    return func;
  }

  virtual PrimFunc Legalize(PrimFunc func) = 0;

  virtual bool MatchType(const Type& type) const = 0;

 protected:
  UnchangedOr<PrimExpr> Mutate_(const prim::CastNode* op, InplaceMode inplace_mode) final {
    auto op_val =
        PromoteToTarget(this->Mutate(op->value, inplace_mode).ValueOrUnchanged(op->value));

    // all casts to matched data type (fp8/bf16) becomes f32
    PrimType op_ty = op->ty.as_or_throw<PrimType>();
    if (MatchType(op_ty)) {
      return prim::cast(promote_dtype_.WithLanes(op_ty.lanes()), op_val);
    }

    if (op_val.same_as(op->value)) {
      return ffi::Unchanged();
    } else {
      return prim::cast(op_ty, op_val);
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::SelectNode* op, InplaceMode inplace_mode) final {
    auto condition_result = this->Mutate(op->condition, inplace_mode);
    bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
    PrimExpr condition = std::move(condition_result).ValueOrUnchanged(op->condition);
    PrimExpr true_value = PromoteToTarget(
        this->Mutate(op->true_value, inplace_mode).ValueOrUnchanged(op->true_value));
    PrimExpr false_value = PromoteToTarget(
        this->Mutate(op->false_value, inplace_mode).ValueOrUnchanged(op->false_value));
    if (condition_unchanged && true_value.same_as(op->true_value) &&
        false_value.same_as(op->false_value)) {
      return ffi::Unchanged();
    } else {
      return prim::Select(condition, true_value, false_value);
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::BroadcastNode* op, InplaceMode inplace_mode) final {
    PrimExpr value =
        PromoteToTarget(this->Mutate(op->value, inplace_mode).ValueOrUnchanged(op->value));
    if (value.same_as(op->value)) {
      return ffi::Unchanged();
    } else {
      return prim::Broadcast(value, op->lanes);
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::ShuffleNode* op, InplaceMode inplace_mode) final {
    auto vectors = op->vectors.Map([this](const PrimExpr& value) {
      return PromoteToTarget(Mutate(value).ValueOrUnchanged(value));
    });
    if (vectors.same_as(op->vectors)) {
      return ffi::Unchanged();
    } else {
      return prim::Shuffle(vectors, op->indices);
    }
  }

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(builtin::masked_load()) || op->op.same_as(builtin::masked_store())) {
      bool is_load = op->op.same_as(builtin::masked_load());
      BufferVar original(op->args[0].as_or_throw<Var>());
      BufferVar buffer = GetRemappedBuffer(original);
      ffi::Array<Expr> args{buffer.var()};
      PrimExpr value;
      if (!is_load) {
        value = this->Mutate(op->args[1]).ValueOrUnchanged(op->args[1]).as_or_throw<PrimExpr>();
      }
      ffi::Array<PrimExpr> indices;
      for (size_t i = is_load ? 1 : 2; i + 1 < op->args.size(); ++i) {
        indices.push_back(
            this->Mutate(op->args[i]).ValueOrUnchanged(op->args[i]).as_or_throw<PrimExpr>());
      }
      PrimExpr predicate = this->Mutate(op->args[op->args.size() - 1])
                               .ValueOrUnchanged(op->args[op->args.size() - 1])
                               .as_or_throw<PrimExpr>();
      if (is_load) {
        for (const PrimExpr& index : indices) args.push_back(index);
        args.push_back(predicate);
        Type type = BufferLoad(buffer, indices).ty();
        return Call(type, op->op, args, op->attrs, op->ty_args, op->span);
      }
      if (MatchType(buffer->dtype)) {
        value = CastTargetToDType(value, BufferLoad(buffer, indices).ty());
      }
      PrimType storage_dtype = BufferLoad(buffer, indices).ty();
      if (value.ty() != storage_dtype) {
        TVM_FFI_ICHECK(MatchType(value.ty()));
        value = DTypeConversion(value, storage_dtype);
      }
      args.push_back(value);
      for (const PrimExpr& index : indices) args.push_back(index);
      args.push_back(predicate);
      return Call(PrimType::Void(), op->op, args, op->attrs, op->ty_args, op->span);
    }
    if (!op->ty.as<PrimTypeNode>()) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    // presertve reinterpret<bf16>() behavior.
    if (op->op.same_as(builtin::reinterpret())) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    // update normal computations to return f32 instead.
    auto args = op->args.Map([this](const Expr& before) {
      bool is_prim = before.as<PrimExpr>().has_value();
      Expr value = Mutate(before).ValueOrUnchanged(before);
      if (is_prim) value = PromoteToTarget(value.as_or_throw<PrimExpr>());
      return value;
    });
    PrimType op_ty = op->ty.as_or_throw<PrimType>();
    if (MatchType(op_ty)) {
      return Call(promote_dtype_.WithLanes(op_ty.lanes()), op->op, args, op->attrs, {}, op->span)
          .as_or_throw<PrimExpr>();
    }
    if (args.same_as(op->args)) {
      return ffi::GetRef<Call>(op).as_or_throw<PrimExpr>();
    } else {
      return Call(op->ty.as_or_throw<PrimType>(), op->op, args, op->attrs, {}, op->span)
          .as_or_throw<PrimExpr>();
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::FloatImmNode* op, InplaceMode inplace_mode) final {
    if (MatchType(op->ty.as_or_throw<PrimType>())) {
      return prim::FloatImm(promote_dtype_, op->value);
    }
    return ffi::Unchanged();
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) final {
    PrimExpr value = PromoteToTarget(op->value);
    Var var = op->var;
    if (value.ty() != op->value.ty()) {
      var = op->var.CopyWithDType(op->value.ty());
      VarRemapSet(op->var, var);
    }
    auto body_result = Mutate(op->body, inplace_mode);
    bool body_unchanged = body_result.UnchangedOrSameAs(op->body);
    PrimExpr body = std::move(body_result).ValueOrUnchanged(op->body);

    if (value.same_as(op->value) && var.same_as(op->var) && body_unchanged) {
      return ffi::Unchanged();
    } else {
      return prim::Let(var, value, body);
    }
  }

  DEFINE_BIOP_EXPR_LEGALIZE(prim::AddNode, operator+);
  DEFINE_BIOP_EXPR_LEGALIZE(prim::SubNode, operator-);
  DEFINE_BIOP_EXPR_LEGALIZE(prim::MulNode, operator*);
  DEFINE_BIOP_EXPR_LEGALIZE(prim::DivNode, div);
  DEFINE_BIOP_EXPR_LEGALIZE(prim::MinNode, min);
  DEFINE_BIOP_EXPR_LEGALIZE(prim::MaxNode, max);
  DEFINE_BIOP_EXPR_LEGALIZE(prim::LTNode, operator<);  // NOLINT(*)
  DEFINE_BIOP_EXPR_LEGALIZE(prim::LENode, operator<=);
  DEFINE_BIOP_EXPR_LEGALIZE(prim::GTNode, operator>);  // NOLINT(*)
  DEFINE_BIOP_EXPR_LEGALIZE(prim::GENode, operator>=);
  DEFINE_BIOP_EXPR_LEGALIZE(prim::EQNode, operator==);
  DEFINE_BIOP_EXPR_LEGALIZE(prim::NENode, operator!=);

  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final {
    auto prim_value = op->value.as<PrimExpr>();
    if (!prim_value) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    PrimExpr value = PromoteToTarget(prim_value.value());
    Var var = op->var;
    if (value.ty() != prim_value.value().ty()) {
      var = op->var.CopyWithDType(prim_value.value().ty());
      VarRemapSet(op->var, var);
    }

    if (value.same_as(op->value) && var.same_as(op->var)) {
      return ffi::Unchanged();
    } else {
      return Bind(var, value);
    }
  }

  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) final {
    auto value_result = this->Mutate(op->value, inplace_mode);
    bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
    PrimExpr value = std::move(value_result).ValueOrUnchanged(op->value);
    auto indices = Mutate(op->indices, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                       .ValueOrUnchanged(op->indices);
    BufferVar new_buf = GetRemappedBuffer(op->buffer);

    if (value_unchanged && indices.same_as(op->indices) && new_buf.same_as(op->buffer)) {
      return ffi::Unchanged();
    } else {
      if (MatchType(new_buf->dtype)) {
        value = CastTargetToDType(value, BufferLoad(new_buf, indices).ty());
      }
      PrimType storage_dtype = BufferLoad(new_buf, indices).ty();
      if (value.ty() != storage_dtype) {
        // this happens when buffer get rewritten to f32
        // but values remain as fp8/bf16
        TVM_FFI_ICHECK(MatchType(value.ty()));
        value = DTypeConversion(value, storage_dtype);
      }
      return BufferStore(new_buf, value, indices);
    }
  }

  UnchangedOr<Stmt> Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) final {
    Stmt ret = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = ret.as<AttrStmtNode>();
    if (auto var = op->node.as<Var>()) {
      auto mapped = VarRemapGet(var.value());
      if (mapped != nullptr) {
        return AttrStmt(mapped.as_or_throw<Var>(), op->attr_key, op->value, op->body);
      }
    } else if (auto reducer = op->node.as<te::CommReducerNode>()) {
      auto reducer_mode = op->unique() && reducer->unique() ? inplace_mode : InplaceMode::kDisallow;
      auto legalized_identity_elements = Mutate(reducer->identity_element, reducer_mode)
                                             .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                                             .ValueOrUnchanged(reducer->identity_element);

      // Remap input variables
      for (size_t i = 0; i < legalized_identity_elements.size(); i++) {
        Var lhs_var = reducer->lhs[i];
        if (lhs_var->ty.as_or_throw<PrimType>() != legalized_identity_elements[i].ty()) {
          VarRemapSet(lhs_var, lhs_var.CopyWithDType(legalized_identity_elements[i].ty()));
        }
        Var rhs_var = reducer->rhs[i];
        if (rhs_var->ty.as_or_throw<PrimType>() != legalized_identity_elements[i].ty()) {
          VarRemapSet(rhs_var, rhs_var.CopyWithDType(legalized_identity_elements[i].ty()));
        }
      }

      auto legalized_results = Mutate(reducer->result, reducer_mode)
                                   .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                                   .ValueOrUnchanged(reducer->result);

      auto legalized_lhs = reducer->lhs.Map([this](PrimVar var) {
        auto mapped = VarRemapGet(var);
        return mapped == nullptr ? var : mapped.as_or_throw<PrimVar>();
      });

      auto legalized_rhs = reducer->rhs.Map([this](PrimVar var) {
        auto mapped = VarRemapGet(var);
        return mapped == nullptr ? var : mapped.as_or_throw<PrimVar>();
      });
      return AttrStmt(te::CommReducer(legalized_lhs, legalized_rhs, legalized_results,
                                      legalized_identity_elements, reducer->span),
                      op->attr_key, op->value, op->body);
    }
    return ret;
  }

  UnchangedOr<Stmt> Mutate_(const DeclBufferNode* op, InplaceMode inplace_mode) final {
    auto data_result = Mutate(op->data, inplace_mode);
    bool data_unchanged = data_result.UnchangedOrSameAs(op->data);
    Expr data = std::move(data_result).ValueOrUnchanged(op->data);
    BufferVar new_buf = GetRemappedBuffer(op->buffer);
    if (new_buf.same_as(op->buffer) && data_unchanged) {
      return ffi::Unchanged();
    }
    return DeclBuffer(new_buf, std::move(data), op->span);
  }

  UnchangedOr<Stmt> Mutate_(const AllocBufferNode* op, InplaceMode inplace_mode) final {
    Stmt ret = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = ret.as<AllocBufferNode>();

    BufferVar new_buf = GetRemappedBuffer(op->buffer);
    if (new_buf.same_as(op->buffer)) {
      return ret;
    } else {
      auto node = ret.as_or_throw<AllocBuffer>();
      node.CopyOnWrite()->buffer = new_buf;
      return node;
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    BufferVar buffer = GetRemappedBuffer(op->source.as_or_throw<BufferVar>());
    auto indices = Mutate(op->indices, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                       .ValueOrUnchanged(op->indices);
    if (buffer.same_as(op->source) && indices.same_as(op->indices)) {
      return ffi::Unchanged();
    }
    return BufferLoad(buffer, indices, op->span);
  }

 private:
  /*!
   * \brief promote value to target datatype F16/F32 and keep other values unchanged.
   * \param value The input value.
   * \return The converted value.
   */
  PrimExpr PromoteToTarget(PrimExpr value) {
    PrimType value_ty = value.ty();
    if (!MatchType(value_ty)) return value;
    if (const prim::CastNode* cast = value.as<prim::CastNode>()) {
      if (cast->value.ty() == promote_dtype_.WithLanes(value_ty.lanes())) return cast->value;
    }
    return DTypeConversion(value, promote_dtype_.WithLanes(value_ty.lanes()));
  }

  /*!
   * \brief Cast value from promoted datatype (FP16/FP32) back to BF16/FP8 and keep other values
   *   unchanged.
   * \param value The input value
   * \return The converted value.
   */
  PrimExpr CastTargetToDType(PrimExpr value, PrimType dtype) {
    PrimType value_ty = value.ty();
    if (value_ty.code() != DLDataTypeCode::kDLFloat) return value;
    TVM_FFI_ICHECK_EQ(value.ty(), this->promote_dtype_.WithLanes(value_ty.lanes()));
    return DTypeConversion(value, dtype);
  }

  BufferVar GetRemappedBuffer(BufferVar buf) {
    auto mapped = VarRemapGet(buf);
    return mapped == nullptr ? buf : mapped.as_or_throw<BufferVar>();
  }

 protected:
  PrimType promote_dtype_;
};

class BF16ComputeLegalizer : public ComputeLegalizer {
 public:
  using ComputeLegalizer::Mutate;
  using ComputeLegalizer::Mutate_;
  BF16ComputeLegalizer() : ComputeLegalizer(PrimType::Float(32)) {}
  PrimFunc Legalize(PrimFunc func) {
    auto planner = ffi::make_object<BF16ComputeLegalizePlanner>(promote_dtype_);
    return LegalizeWithPlanner(func, planner.get());
  }
  bool MatchType(const Type& type) const {
    return MatchPrimType(type, [](const PrimType& prim_type) { return IsBFloat16Type(prim_type); });
  }
};

class FP8ComputeLegalizer : public ComputeLegalizer {
 public:
  using ComputeLegalizer::Mutate;
  using ComputeLegalizer::Mutate_;
  explicit FP8ComputeLegalizer(PrimType promote_dtype) : ComputeLegalizer(promote_dtype) {}
  PrimFunc Legalize(PrimFunc func) {
    auto planner = ffi::make_object<FP8ComputeLegalizePlanner>(promote_dtype_);
    return LegalizeWithPlanner(func, planner.get());
  }
  bool MatchType(const Type& type) const {
    return MatchPrimType(type, [](const PrimType& prim_type) { return IsFloat8Type(prim_type); });
  }
};

/*!
 * \brief This Pass legalizes remaining FP8/BF16 storages to unsigned integers with equal number of
 * bits.
 *
 * This pass needs to happens after FP8/BF16ComputeLegalizer and serves
 * as a way to support FP8/BF16 on platforms that do not have native support.
 */
class StorageLegalizer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  PrimFunc Legalize(PrimFunc func) {
    for (const Var& param : func->params) {
      TVM_FFI_ICHECK(!param->ty.as<BufferTypeNode>())
          << "This pass must be called after MakePackedAPI";
    }
    auto* n = func.CopyOnWrite();
    n->params = n->params.Map([this](Var var) { return this->RemapVarDef(var); });
    n->body = this->Mutate(n->body, InplaceMode::kDisallow).ValueOrUnchanged(n->body);
    return func;
  }

 private:
  UnchangedOr<Stmt> Mutate_(const AllocBufferNode* op, InplaceMode inplace_mode) final {
    BufferVar buf = GetRemappedBuffer(op->buffer, /*allow_definition=*/true);
    // in a rare case the buffer didn't get remapped
    // because the original var is not bfloat*
    // force remap here
    if (MatchType(buf->dtype)) {
      PrimType new_dtype = GetStorageUIntDType(buf->dtype);
      auto type = CopyBufferType(buf);
      type->dtype = new_dtype;
      BufferVar new_buf = RebuildBufferVar(buf, std::move(type));
      VarRemapSet(op->buffer, new_buf);
      buf = std::move(new_buf);
    }
    if (buf.same_as(op->buffer)) {
      return ffi::Unchanged();
    } else {
      auto node = ffi::GetRef<AllocBuffer>(op);
      node.CopyOnWrite()->buffer = buf;
      return node;
    }
  }

  UnchangedOr<Stmt> Mutate_(const DeclBufferNode* op, InplaceMode inplace_mode) final {
    BufferVar buf = GetRemappedBuffer(op->buffer, /*allow_definition=*/true);
    auto data_result = Mutate(op->data, inplace_mode);
    bool data_unchanged = data_result.UnchangedOrSameAs(op->data);
    Expr data = std::move(data_result).ValueOrUnchanged(op->data);
    // in a rare case the buffer didn't get remapped
    // because the original var is not bfloat*
    // force remap here
    if (MatchType(buf->dtype)) {
      PrimType new_dtype = GetStorageUIntDType(buf->dtype);
      auto type = CopyBufferType(buf);
      type->dtype = new_dtype;
      buf = RebuildBufferVar(buf, std::move(type));
      VarRemapSet(op->buffer, buf);
    }
    if (buf.same_as(op->buffer) && data_unchanged) {
      return ffi::Unchanged();
    }
    return DeclBuffer(buf, std::move(data), op->span);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) final {
    auto value_result = Mutate(op->value, inplace_mode);
    bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
    PrimExpr value = std::move(value_result).ValueOrUnchanged(op->value);
    Var var = RemapVarDef(op->var);
    auto body_result = Mutate(op->body, inplace_mode);
    bool body_unchanged = body_result.UnchangedOrSameAs(op->body);
    PrimExpr body = std::move(body_result).ValueOrUnchanged(op->body);

    if (value_unchanged && var.same_as(op->var) && body_unchanged) {
      return ffi::Unchanged();
    } else {
      return prim::Let(var, value, body);
    }
  }

  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) final {
    auto value_result = Mutate(op->value, inplace_mode);
    bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
    Expr value = std::move(value_result).ValueOrUnchanged(op->value);
    Var var = RemapVarDef(op->var);

    if (value_unchanged && var.same_as(op->var)) {
      return ffi::Unchanged();
    } else {
      return Bind(var, value);
    }
  }

  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) final {
    PrimExpr value =
        this->ChangeToUInt(Mutate(op->value, inplace_mode).ValueOrUnchanged(op->value));
    BufferVar new_buf = GetRemappedBuffer(op->buffer);
    auto indices = Mutate(op->indices, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                       .ValueOrUnchanged(op->indices);
    if (new_buf.same_as(op->buffer) && indices.same_as(op->indices) && value.same_as(op->value)) {
      return ffi::Unchanged();
    } else {
      if (MatchType(op->value.ty())) {
        TVM_FFI_ICHECK(new_buf->dtype.MatchesCode(DLDataTypeCode::kDLUInt));
      }
      return BufferStore(new_buf, value, indices);
    }
  }

  UnchangedOr<Stmt> Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) final {
    Stmt ret = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = ret.as<AttrStmtNode>();

    if (auto var = op->node.as<Var>()) {
      auto mapped = VarRemapGet(var.value());
      if (mapped != nullptr) {
        return AttrStmt(mapped.as_or_throw<Var>(), op->attr_key, op->value, op->body);
      }
    }
    return ret;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    BufferVar buffer = GetRemappedBuffer(op->source.as_or_throw<BufferVar>());
    auto indices = Mutate(op->indices, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                       .ValueOrUnchanged(op->indices);
    if (buffer.same_as(op->source) && indices.same_as(op->indices)) {
      return ffi::Unchanged();
    }
    return BufferLoad(buffer, indices, op->span);
  }

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(builtin::masked_load()) || op->op.same_as(builtin::masked_store())) {
      bool is_load = op->op.same_as(builtin::masked_load());
      BufferVar buffer = GetRemappedBuffer(BufferVar(op->args[0].as_or_throw<Var>()));
      ffi::Array<Expr> args{buffer.var()};
      PrimExpr value;
      if (!is_load) {
        PrimExpr original_value = op->args[1].as_or_throw<PrimExpr>();
        value = this->ChangeToUInt(this->Mutate(original_value).ValueOrUnchanged(original_value));
        if (MatchType(original_value.ty())) {
          TVM_FFI_ICHECK(buffer->dtype.MatchesCode(DLDataTypeCode::kDLUInt));
        }
        args.push_back(value);
      }
      ffi::Array<PrimExpr> indices;
      for (size_t i = is_load ? 1 : 2; i + 1 < op->args.size(); ++i) {
        PrimExpr index =
            this->Mutate(op->args[i]).ValueOrUnchanged(op->args[i]).as_or_throw<PrimExpr>();
        indices.push_back(index);
        args.push_back(index);
      }
      args.push_back(this->Mutate(op->args[op->args.size() - 1])
                         .ValueOrUnchanged(op->args[op->args.size() - 1])
                         .as_or_throw<Expr>());
      if (is_load) {
        Type type = BufferLoad(buffer, indices).ty();
        return Call(type, op->op, args, op->attrs, op->ty_args, op->span);
      } else {
        return Call(PrimType::Void(), op->op, args, op->attrs, op->ty_args, op->span);
      }
    }
    if (const auto* pointer_type = op->ty.as<PointerTypeNode>()) {
      Expr ret = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Expr>(op));
      const auto* element_type = pointer_type->element_type.as<PrimTypeNode>();
      if (!element_type || !MatchType(ffi::GetRef<PrimType>(element_type))) {
        return ret;
      }
      Call call = ret.as_or_throw<Call>();
      Type new_element_type = GetStorageUIntDType(ffi::GetRef<PrimType>(element_type));
      return Call(PointerType(new_element_type, pointer_type->storage_scope), call->op, call->args,
                  call->attrs, call->ty_args, call->span);
    }
    if (!op->ty.as<PrimTypeNode>()) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    // remap re-interpret so un-necessary reinterpret can be skipped.
    if (op->op.same_as(builtin::reinterpret())) {
      PrimExpr value = Mutate(op->args[0]).ValueOrUnchanged(op->args[0]).as_or_throw<PrimExpr>();
      // sometimes the input dtype can change and we can skip.
      PrimType op_dtype = op->ty.as_or_throw<PrimType>();
      if (value.ty() == op_dtype) return value;
      if (MatchType(op_dtype)) {
        return reinterpret(GetStorageUIntDType(op_dtype), value);
      }
      if (op->args[0].same_as(value)) {
        return ffi::GetRef<Call>(op).as_or_throw<PrimExpr>();
      } else {
        return reinterpret(op_dtype, value);
      }
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  virtual bool MatchType(const Type& type) const = 0;

 private:
  /*!
   * \brief Change float value to uint value.
   * \param value The input value.
   * \return The converted value.
   */
  PrimExpr ChangeToUInt(PrimExpr value) {
    PrimType value_dtype = value.ty();
    if (!MatchType(value_dtype)) return value;
    auto* call = value.as<CallNode>();
    if (call && call->op.same_as(builtin::reinterpret())) {
      return reinterpret(GetStorageUIntDType(value_dtype), call->args[0].as_or_throw<PrimExpr>());
    } else {
      return value;
    }
  }

  Var RemapVarDef(Var var) {
    // remap the var
    if (auto* ptr_type = var->ty.as<PointerTypeNode>()) {
      if (auto* elem_type = ptr_type->element_type.as<PrimTypeNode>()) {
        PrimType elem_prim_type = ffi::GetRef<PrimType>(elem_type);
        if (MatchType(elem_prim_type)) {
          Var new_var = Var(
              var->name, PointerType(GetStorageUIntDType(elem_prim_type), ptr_type->storage_scope));
          VarRemapSet(var, new_var);
          return new_var;
        }
      }
    }
    return var;
  }

  BufferVar GetRemappedBuffer(BufferVar buf, bool allow_definition = false) {
    auto mapped = VarRemapGet(buf);
    if (mapped != nullptr) return mapped.as_or_throw<BufferVar>();
    if (!allow_definition) {
      TVM_FFI_ICHECK(!MatchType(buf->dtype)) << "Cannot find var remap for " << buf;
    }
    return buf;
  }
};

class BF16StorageLegalizer : public StorageLegalizer {
 public:
  using StmtExprMutator::Mutate_;
  using StorageLegalizer::Mutate;
  bool MatchType(const Type& type) const {
    return MatchPrimType(type, [](const PrimType& prim_type) { return IsBFloat16Type(prim_type); });
  }
};

class FP8StorageLegalizer : public StorageLegalizer {
 public:
  using StmtExprMutator::Mutate_;
  using StorageLegalizer::Mutate;
  bool MatchType(const Type& type) const {
    return MatchPrimType(type, [](const PrimType& prim_type) { return IsFloat8Type(prim_type); });
  }
};

namespace transform {

bool CheckDataTypeSupport(const Target& target, const std::string& support_func_name) {
  bool has_native_support = false;
  if (target->kind->name == "cuda") {
    if (auto get_cv = tvm::ffi::Function::GetGlobal("tvm.support.nvcc.get_compute_version")) {
      std::string compute_version = (*get_cv)(target).cast<std::string>();
      if (auto check_support = tvm::ffi::Function::GetGlobal(support_func_name)) {
        has_native_support = (*check_support)(compute_version).cast<bool>();
      }
    }
  }
  return has_native_support;
}

Pass BF16ComputeLegalize() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (opt_target.has_value() &&
        CheckDataTypeSupport(opt_target.value(), "tvm.support.nvcc.supports_bf16")) {
      return f;
    }
    return ffi::make_object<BF16ComputeLegalizer>()->Legalize(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.BF16ComputeLegalize", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.BF16ComputeLegalize", BF16ComputeLegalize);
}

Pass BF16StorageLegalize() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (opt_target.has_value() &&
        CheckDataTypeSupport(opt_target.value(), "tvm.support.nvcc.supports_bf16")) {
      return f;
    }
    return ffi::make_object<BF16StorageLegalizer>()->Legalize(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.BF16StorageLegalize", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.BF16StorageLegalize", BF16StorageLegalize);
}

Pass FP8ComputeLegalize(ffi::String promote_dtype) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (opt_target.has_value() &&
        CheckDataTypeSupport(opt_target.value(), "tvm.support.nvcc.supports_fp8")) {
      return f;
    }
    return ffi::make_object<FP8ComputeLegalizer>(PrimType(ffi::StringToDLDataType(promote_dtype)))
        ->Legalize(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.FP8ComputeLegalize", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.FP8ComputeLegalize", FP8ComputeLegalize);
}

Pass FP8StorageLegalize() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (opt_target.has_value() &&
        CheckDataTypeSupport(opt_target.value(), "tvm.support.nvcc.supports_fp8")) {
      return f;
    }
    return ffi::make_object<FP8StorageLegalizer>()->Legalize(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.FP8StorageLegalize", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.FP8StorageLegalize", FP8StorageLegalize);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
