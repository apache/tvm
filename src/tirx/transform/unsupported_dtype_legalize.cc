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
#include <tvm/ir/expr.h>
#include <tvm/ir/prim/op.h>
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
// Remap internal fp8/bf16 allocations without raw pointer access to the promoted dtype.
//
// Select allocation calls before opaque-access filtering.
class ComputeLegalizePlanner : public StmtExprVisitor {
 public:
  void Plan(Function func) {
    this->Visit(func->body);
    // A later opaque access can veto an earlier allocation candidate.
    for (const Var& var : opaque_var_access_) {
      candidates_.erase(var);
    }
  }

  const std::unordered_set<Var>& Allocations() const { return candidates_; }

  virtual bool MatchType(const Type& type) const = 0;

  ffi::Optional<VisitInterrupt> Visit_(const BindNode* op) final {
    if (const auto* call = op->value.as<CallNode>();
        call && call->op.same_as(tirx::alloc_tensor_op()))
      return DispatchAllocTensor(op, call);
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> DispatchAllocTensor(const BindNode* op, const CallNode* call) {
    DLDataType dtype = call->args[1].as_or_throw<DataTypeImm>()->value;
    PrimType alloc_dtype(dtype);
    // Select intermediate buffers with an unsupported element type.
    if (MatchType(alloc_dtype)) {
      candidates_.insert(op->var);
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const CallNode* op) final {
    if (op->op.same_as(tirx::buffer_data_op()) && op->args.size() == 1) {
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
    return StmtExprVisitor::Visit_(op);
  }

 private:
  std::unordered_set<Var> candidates_;
  std::unordered_set<Var> opaque_var_access_;
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
    if (origin_a.same_as(op->a) && origin_b.same_as(op->b) && !MatchType(op->ty)) { \
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

  Function LegalizeWithPlanner(Function func, ComputeLegalizePlanner* planner) {
    planner->Plan(func);
    promoted_buffers_ = planner->Allocations();
    auto* n = func.CopyOnWrite();
    n->body = this->Mutate(n->body, InplaceMode::kDisallow).ValueOrUnchanged(n->body);
    return func;
  }

  virtual Function Legalize(Function func) = 0;

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
        false_value.same_as(op->false_value) && !MatchType(op->ty)) {
      return ffi::Unchanged();
    } else {
      return prim::Select(condition, true_value, false_value);
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::BroadcastNode* op, InplaceMode inplace_mode) final {
    PrimExpr value =
        PromoteToTarget(this->Mutate(op->value, inplace_mode).ValueOrUnchanged(op->value));
    if (value.same_as(op->value) && !MatchType(op->ty)) {
      return ffi::Unchanged();
    } else {
      return prim::Broadcast(value, op->lanes);
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::ShuffleNode* op, InplaceMode inplace_mode) final {
    auto vectors = op->vectors.Map([this](const PrimExpr& value) {
      return PromoteToTarget(Mutate(value).ValueOrUnchanged(value));
    });
    if (vectors.same_as(op->vectors) && !MatchType(op->ty)) {
      return ffi::Unchanged();
    } else {
      return prim::Shuffle(vectors, op->indices);
    }
  }

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(tirx::tvm_thread_allreduce_op())) {
      return LegalizeThreadAllreduce(op);
    }
    if (op->op.same_as(tirx::alloc_tensor_op()) || op->op.same_as(tirx::decl_tensor_op())) {
      Call call = StmtExprMutator::Mutate_(op, inplace_mode)
                      .ValueOrUnchanged(ffi::GetRef<Expr>(op))
                      .as_or_throw<Call>();
      if (op == allocation_to_promote_) {
        auto dtype = call->args[1].as_or_throw<DataTypeImm>();
        auto* node = call.CopyOnWrite();
        node->args.Set(1,
                       DataTypeImm(promote_dtype_.WithLanes(PrimType(dtype->value).lanes())->dtype,
                                   dtype->span));
      }
      return ReinferMutatedCallType(call, op, inplace_mode);
    }
    if (op->op.same_as(tirx::masked_load_op()) || op->op.same_as(tirx::masked_store_op())) {
      bool is_load = op->op.same_as(tirx::masked_load_op());
      TensorVar original = op->args[0].as_or_throw<TensorVar>();
      TensorVar buffer = GetRemappedBuffer(original);
      ffi::Array<Expr> args{buffer.var()};
      ffi::Optional<PrimExpr> value;
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
        Type type = MakeTensorLoad(buffer, indices).ty();
        return Call(type, op->op, args, op->attrs, op->ty_args, op->span);
      }
      if (MatchType(buffer->dtype)) {
        value = CastTargetToDType(value.value(), MakeTensorLoad(buffer, indices).ty());
      }
      PrimType storage_dtype = MakeTensorLoad(buffer, indices).ty();
      if (value.value().ty() != storage_dtype) {
        TVM_FFI_ICHECK(MatchType(value.value().ty()));
        value = DTypeConversion(value.value(), storage_dtype);
      }
      args.push_back(value.value());
      for (const PrimExpr& index : indices) args.push_back(index);
      args.push_back(predicate);
      return Call(PrimType::Void(), op->op, args, op->attrs, op->ty_args, op->span);
    }
    if (!op->ty.as<PrimTypeNode>()) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    // presertve reinterpret<bf16>() behavior.
    if (op->op.same_as(tirx::reinterpret_op())) {
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

  UnchangedOr<PrimExpr> Mutate_(const FloatImmNode* op, InplaceMode inplace_mode) final {
    if (MatchType(op->ty.as_or_throw<PrimType>())) {
      return FloatImm(promote_dtype_, op->value);
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
      // Eligibility belongs to the binding, even when several bindings share one Call.
      const CallNode* previous = allocation_to_promote_;
      allocation_to_promote_ =
          promoted_buffers_.count(op->var) ? op->value.as<CallNode>() : nullptr;
      auto result = StmtExprMutator::Mutate_(op, inplace_mode);
      allocation_to_promote_ = previous;
      return result;
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

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode* op, InplaceMode inplace_mode) final {
    auto value_result = this->Mutate(op->value, inplace_mode);
    bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
    PrimExpr value = std::move(value_result).ValueOrUnchanged(op->value);
    auto indices = Mutate(op->indices, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                       .ValueOrUnchanged(op->indices);
    TensorVar new_buf = GetRemappedBuffer(op->buffer);

    if (value_unchanged && indices.same_as(op->indices) && new_buf.same_as(op->buffer)) {
      return ffi::Unchanged();
    } else {
      if (MatchType(new_buf->dtype)) {
        value = CastTargetToDType(value, MakeTensorLoad(new_buf, indices).ty());
      }
      PrimType storage_dtype = MakeTensorLoad(new_buf, indices).ty();
      if (value.ty() != storage_dtype) {
        // this happens when buffer get rewritten to f32
        // but values remain as fp8/bf16
        TVM_FFI_ICHECK(MatchType(value.ty()));
        value = DTypeConversion(value, storage_dtype);
      }
      return TensorStore(new_buf, value, indices);
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    TensorVar buffer = GetRemappedBuffer(op->source.as_or_throw<TensorVar>());
    auto indices = Mutate(op->indices, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                       .ValueOrUnchanged(op->indices);
    if (buffer.same_as(op->source) && indices.same_as(op->indices)) {
      return ffi::Unchanged();
    }
    return MakeTensorLoad(buffer, indices, op->span);
  }

 private:
  // Preserve scalar operands and explicit Tuple grouping while promoting computation.
  Expr LegalizeThreadAllreduce(const CallNode* op) {
    LambdaExpr combine = op->args[0].as_or_throw<LambdaExpr>();
    auto map_operand = [](const Expr& operand, const auto& transform) -> Expr {
      if (const auto* tuple = operand.as<tvm::TupleNode>()) {
        return tvm::Tuple(tuple->fields.Map(transform), operand->span);
      }
      return transform(operand);
    };
    auto promote = [this](const Expr& value) -> Expr {
      return PromoteToTarget(Mutate(value).ValueOrUnchanged(value).as_or_throw<PrimExpr>());
    };
    Expr identity = map_operand(op->args[1], promote);
    Expr values = map_operand(op->args[2], promote);
    ffi::Array<Expr> value_fields = tirx::GetAllreduceFields(values);
    ffi::Array<Var> vars;
    ffi::Array<Expr> arguments;
    for (size_t i = 0; i < combine->vars.size(); ++i) {
      Var promoted = combine->vars[i].CopyWithDType(
          value_fields[i % value_fields.size()]->ty.as_or_throw<PrimType>());
      vars.push_back(promoted);
      arguments.push_back(promoted);
    }
    LambdaExpr legalized_combine(vars, map_operand(combine->Apply(arguments), promote));
    auto mutate = [this](const Expr& value) { return Mutate(value).ValueOrUnchanged(value); };
    Expr predicate = mutate(op->args[3]);
    // Destinations are lvalues: remap promoted allocations without adding compute casts.
    Expr destinations = map_operand(op->args[4], mutate);
    Expr axes = map_operand(op->args[5], mutate);
    return Call(PrimType::Void(), op->op,
                {legalized_combine, identity, values, predicate, destinations, axes}, op->attrs,
                op->ty_args, op->span);
  }

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

  TensorVar GetRemappedBuffer(TensorVar buf) {
    auto mapped = VarRemapGet(buf);
    return mapped == nullptr ? buf : mapped.as_or_throw<TensorVar>();
  }

 protected:
  std::unordered_set<Var> promoted_buffers_;
  const CallNode* allocation_to_promote_{nullptr};
  PrimType promote_dtype_;
};

class BF16ComputeLegalizer : public ComputeLegalizer {
 public:
  using ComputeLegalizer::Mutate;
  using ComputeLegalizer::Mutate_;
  BF16ComputeLegalizer() : ComputeLegalizer(PrimType::Float(32)) {}
  Function Legalize(Function func) {
    auto planner = ffi::make_object<BF16ComputeLegalizePlanner>();
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
  Function Legalize(Function func) {
    auto planner = ffi::make_object<FP8ComputeLegalizePlanner>();
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
  Function Legalize(Function func) {
    for (const Var& param : func->params) {
      TVM_FFI_ICHECK(!param->ty.as<TensorTypeNode>())
          << "This pass must be called after MakePackedAPI";
    }
    auto* n = func.CopyOnWrite();
    n->params = n->params.Map([this](Var var) { return this->RemapVarDef(var); });
    n->body = this->Mutate(n->body, InplaceMode::kDisallow).ValueOrUnchanged(n->body);
    return func;
  }

 private:
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
    if (op->value->ty.as<TensorTypeNode>()) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
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

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode* op, InplaceMode inplace_mode) final {
    PrimExpr value =
        this->ChangeToUInt(Mutate(op->value, inplace_mode).ValueOrUnchanged(op->value));
    TensorVar new_buf = GetRemappedBuffer(op->buffer);
    auto indices = Mutate(op->indices, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                       .ValueOrUnchanged(op->indices);
    if (new_buf.same_as(op->buffer) && indices.same_as(op->indices) && value.same_as(op->value)) {
      return ffi::Unchanged();
    } else {
      if (MatchType(op->value.ty())) {
        TVM_FFI_ICHECK(new_buf->dtype.MatchesCode(DLDataTypeCode::kDLUInt));
      }
      return TensorStore(new_buf, value, indices);
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    TensorVar buffer = GetRemappedBuffer(op->source.as_or_throw<TensorVar>());
    auto indices = Mutate(op->indices, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                       .ValueOrUnchanged(op->indices);
    if (buffer.same_as(op->source) && indices.same_as(op->indices)) {
      return ffi::Unchanged();
    }
    return MakeTensorLoad(buffer, indices, op->span);
  }

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(tirx::alloc_tensor_op()) || op->op.same_as(tirx::decl_tensor_op())) {
      Call call = StmtExprMutator::Mutate_(op, inplace_mode)
                      .ValueOrUnchanged(ffi::GetRef<Expr>(op))
                      .as_or_throw<Call>();
      int dtype_index = op->op.same_as(tirx::alloc_tensor_op()) ? 1 : 2;
      auto dtype = call->args[dtype_index].as_or_throw<DataTypeImm>();
      if (MatchType(PrimType(dtype->value))) {
        call.CopyOnWrite()->args.Set(
            dtype_index,
            DataTypeImm(GetStorageUIntDType(PrimType(dtype->value))->dtype, dtype->span));
      }
      return ReinferMutatedCallType(call, op, inplace_mode);
    }
    if (op->op.same_as(tirx::masked_load_op()) || op->op.same_as(tirx::masked_store_op())) {
      bool is_load = op->op.same_as(tirx::masked_load_op());
      TensorVar buffer = GetRemappedBuffer(op->args[0].as_or_throw<TensorVar>());
      ffi::Array<Expr> args{buffer.var()};
      ffi::Optional<PrimExpr> value;
      if (!is_load) {
        PrimExpr original_value = op->args[1].as_or_throw<PrimExpr>();
        value = this->ChangeToUInt(this->Mutate(original_value).ValueOrUnchanged(original_value));
        if (MatchType(original_value.ty())) {
          TVM_FFI_ICHECK(buffer->dtype.MatchesCode(DLDataTypeCode::kDLUInt));
        }
        args.push_back(value.value());
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
        Type type = MakeTensorLoad(buffer, indices).ty();
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
    if (op->op.same_as(tirx::reinterpret_op())) {
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
    if (call && call->op.same_as(tirx::reinterpret_op())) {
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

  TensorVar GetRemappedBuffer(TensorVar buf) {
    auto mapped = VarRemapGet(buf);
    if (mapped != nullptr) return mapped.as_or_throw<TensorVar>();
    TVM_FFI_ICHECK(!MatchType(buf->dtype)) << "Cannot find var remap for " << buf;
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
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    auto opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (opt_target.has_value() &&
        CheckDataTypeSupport(opt_target.value(), "tvm.support.nvcc.supports_bf16")) {
      return f;
    }
    return ffi::make_object<BF16ComputeLegalizer>()->Legalize(f);
  };
  return CreateFunctionPass(pass_func, 0, "tirx.BF16ComputeLegalize", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.BF16ComputeLegalize", BF16ComputeLegalize);
}

Pass BF16StorageLegalize() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    auto opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (opt_target.has_value() &&
        CheckDataTypeSupport(opt_target.value(), "tvm.support.nvcc.supports_bf16")) {
      return f;
    }
    return ffi::make_object<BF16StorageLegalizer>()->Legalize(f);
  };
  return CreateFunctionPass(pass_func, 0, "tirx.BF16StorageLegalize", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.BF16StorageLegalize", BF16StorageLegalize);
}

Pass FP8ComputeLegalize(ffi::String promote_dtype) {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    auto opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (opt_target.has_value() &&
        CheckDataTypeSupport(opt_target.value(), "tvm.support.nvcc.supports_fp8")) {
      return f;
    }
    return ffi::make_object<FP8ComputeLegalizer>(PrimType(ffi::StringToDLDataType(promote_dtype)))
        ->Legalize(f);
  };
  return CreateFunctionPass(pass_func, 0, "tirx.FP8ComputeLegalize", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.FP8ComputeLegalize", FP8ComputeLegalize);
}

Pass FP8StorageLegalize() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    auto opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (opt_target.has_value() &&
        CheckDataTypeSupport(opt_target.value(), "tvm.support.nvcc.supports_fp8")) {
      return f;
    }
    return ffi::make_object<FP8StorageLegalizer>()->Legalize(f);
  };
  return CreateFunctionPass(pass_func, 0, "tirx.FP8StorageLegalize", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.FP8StorageLegalize", FP8StorageLegalize);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
