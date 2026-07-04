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
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <cmath>
#include <tuple>

#include "dtype_conversion.h"

namespace tvm {
namespace tirx {

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
// populate the buffer_remap and var_remap accordingly.
class ComputeLegalizePlanner : public StmtExprVisitor {
 public:
  ComputeLegalizePlanner(
      std::unordered_map<Buffer, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>* buffer_remap,
      std::unordered_map<Var, Var>* var_remap, PrimType promote_dtype)
      : buffer_remap_(buffer_remap), var_remap_(var_remap), promote_dtype_(promote_dtype) {}

  // run planning to populate buffer remap and var remap.
  void Plan(PrimFunc func) {
    this->VisitStmt(func->body);
    // if there are opaque var access, then we cannot
    // do remap of var and buffer, post-hoc remove these items.
    for (Var var : opaque_var_access_) {
      auto it = var_remap_->find(var);
      if (it != var_remap_->end()) {
        var_remap_->erase(it);
      }
    }
    ffi::Array<Buffer> drop_buffers;
    for (auto kv : *buffer_remap_) {
      if (opaque_var_access_.count(kv.first->data)) {
        drop_buffers.push_back(kv.first);
      }
    }
    for (Buffer buffer : drop_buffers) {
      auto it = buffer_remap_->find(buffer);
      TVM_FFI_ICHECK(it != buffer_remap_->end());
      buffer_remap_->erase(it);
    }
  }

  virtual bool MatchType(const Type& type) const = 0;

  void VisitStmt_(const BufferStoreNode* op) final {
    StmtExprVisitor::VisitStmt_(op);
    this->PopulateBufferRemap(op->buffer);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    StmtExprVisitor::VisitExpr_(op);
    this->PopulateBufferRemap(op->buffer);
  }

  void VisitStmt_(const AllocBufferNode* op) final {
    // remap all intermediate constant buffer to promote data types (fp16/fp32)
    if (MatchType(op->buffer->dtype)) {
      PrimType dtype = promote_dtype_.WithLanes(op->buffer->dtype.lanes());
      ffi::String storage_scope = "global";
      if (auto* ptr_type = op->buffer->data->ty.as<PointerTypeNode>()) {
        storage_scope = ptr_type->storage_scope;
      }
      Var buffer_var = Var(op->buffer->data->name_hint, PointerType(dtype, storage_scope));
      (*var_remap_)[op->buffer->data] = buffer_var;
    }
    return StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode* op) final {
    StmtExprVisitor::VisitStmt_(op);
    this->PopulateBufferRemap(op->buffer);
  }

  void VisitExpr_(const VarNode* op) final {
    StmtExprVisitor::VisitExpr_(op);
    Var buffer_var = ffi::GetRef<Var>(op);
    if (PrimType(GetRuntimeDataType(buffer_var->ty)).IsHandle()) {
      opaque_var_access_.insert(buffer_var);
    }
  }

 private:
  void PopulateBufferRemap(Buffer buf) {
    auto var_it = var_remap_->find(buf->data);
    if (var_it == var_remap_->end()) return;

    Buffer new_buffer(var_it->second, promote_dtype_.WithLanes(buf->dtype.lanes()), buf->shape,
                      buf->strides, buf->elem_offset, buf->name, buf->data_alignment,
                      buf->offset_factor, buf->buffer_type, buf->axis_separators, buf->span,
                      buf->layout, buf->allocated_addr);
    (*buffer_remap_)[buf] = new_buffer;
  }

  std::unordered_map<Buffer, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>* buffer_remap_;
  std::unordered_map<Var, Var>* var_remap_;
  std::unordered_set<Var> opaque_var_access_;
  PrimType promote_dtype_;
};

class BF16ComputeLegalizePlanner : public ComputeLegalizePlanner {
 public:
  explicit BF16ComputeLegalizePlanner(
      std::unordered_map<Buffer, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>* buffer_remap,
      std::unordered_map<Var, Var>* var_remap, PrimType promote_dtype)
      : ComputeLegalizePlanner(buffer_remap, var_remap, promote_dtype) {}
  bool MatchType(const Type& type) const {
    return MatchPrimType(type, [](const PrimType& prim_type) { return IsBFloat16Type(prim_type); });
  }
};

class FP8ComputeLegalizePlanner : public ComputeLegalizePlanner {
 public:
  explicit FP8ComputeLegalizePlanner(
      std::unordered_map<Buffer, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>* buffer_remap,
      std::unordered_map<Var, Var>* var_remap, PrimType promote_dtype)
      : ComputeLegalizePlanner(buffer_remap, var_remap, promote_dtype) {}
  bool MatchType(const Type& type) const {
    return MatchPrimType(type, [](const PrimType& prim_type) { return IsFloat8Type(prim_type); });
  }
};

#define DEFINE_BIOP_EXPR_LEGALIZE(OP, FUNC)                          \
  Expr VisitExpr_(const OP* op) final {                              \
    PrimExpr origin_a = PromoteToTarget(this->VisitPrimExpr(op->a)); \
    PrimExpr origin_b = PromoteToTarget(this->VisitPrimExpr(op->b)); \
                                                                     \
    if (origin_a.same_as(op->a) && origin_b.same_as(op->b)) {        \
      return ffi::GetRef<PrimExpr>(op);                              \
    } else {                                                         \
      return FUNC(origin_a, origin_b);                               \
    }                                                                \
  }

// NOTE: Legalize the FP8/BF16 computations
// to floating point computations and only keeps the
// fp8/bf16 storage which can further be legalized by FP8/BF16StorageLegalizer
// FP8/BF16StorageLegalizer will be called at a much later time
// point in the TIR lowering phases.
class ComputeLegalizer : public StmtExprMutator {
 public:
  explicit ComputeLegalizer(PrimType promote_dtype) : promote_dtype_(promote_dtype) {}

  PrimFunc LegalizeWithPlanner(PrimFunc func, ComputeLegalizePlanner* planner) {
    planner->Plan(func);
    auto* n = func.CopyOnWrite();
    n->body = this->VisitStmt(std::move(n->body));
    return func;
  }

  virtual PrimFunc Legalize(PrimFunc func) = 0;

  virtual bool MatchType(const Type& type) const = 0;

 protected:
  Expr VisitExpr_(const CastNode* op) final {
    auto op_val = PromoteToTarget(this->VisitPrimExpr(op->value));

    // all casts to matched data type (fp8/bf16) becomes f32
    PrimType op_ty = op->ty.as_or_throw<PrimType>();
    if (MatchType(op_ty)) {
      return cast(promote_dtype_.WithLanes(op_ty.lanes()), op_val);
    }

    if (op_val.same_as(op->value)) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      return cast(op_ty, op_val);
    }
  }

  Expr VisitExpr_(const SelectNode* op) final {
    PrimExpr condition = this->VisitPrimExpr(op->condition);
    PrimExpr true_value = PromoteToTarget(this->VisitPrimExpr(op->true_value));
    PrimExpr false_value = PromoteToTarget(this->VisitPrimExpr(op->false_value));
    if (condition.same_as(op->condition) && true_value.same_as(op->true_value) &&
        false_value.same_as(op->false_value)) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      return Select(condition, true_value, false_value);
    }
  }

  Expr VisitExpr_(const BroadcastNode* op) final {
    PrimExpr value = PromoteToTarget(this->VisitPrimExpr(op->value));
    if (value.same_as(op->value)) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      return Broadcast(value, op->lanes);
    }
  }

  Expr VisitExpr_(const ShuffleNode* op) final {
    auto fexpr = [this](const PrimExpr& e) { return PromoteToTarget(this->VisitPrimExpr(e)); };
    auto vectors = op->vectors.Map(fexpr);
    if (vectors.same_as(op->vectors)) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      return Shuffle(vectors, op->indices);
    }
  }

  Expr VisitExpr_(const CallNode* op) final {
    if (!op->ty.as<PrimTypeNode>()) {
      return StmtExprMutator::VisitExpr_(op);
    }
    // presertve reinterpret<bf16>() behavior.
    if (op->op.same_as(builtin::reinterpret())) {
      return StmtExprMutator::VisitExpr_(op);
    }
    // update normal computations to return f32 instead.
    auto fmutate = [this](const Expr& e) -> Expr {
      if (auto prim = e.as<PrimExpr>()) {
        return PromoteToTarget(this->VisitPrimExpr(prim.value()));
      }
      return this->VisitExpr(e);
    };
    ffi::Array<Expr> args = op->args.Map(fmutate);
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

  Expr VisitExpr_(const FloatImmNode* op) final {
    if (MatchType(op->ty.as_or_throw<PrimType>())) {
      return FloatImm(promote_dtype_, op->value);
    }
    return ffi::GetRef<PrimExpr>(op);
  }

  Expr VisitExpr_(const VarNode* op) final {
    Var var = ffi::GetRef<Var>(op);

    auto itr = var_remap_.find(var);
    if (itr != var_remap_.end()) {
      return itr->second;
    } else {
      return var;
    }
  }

  Expr VisitExpr_(const LetNode* op) final {
    PrimExpr value = PromoteToTarget(op->value);
    Var var = op->var;
    if (value.ty() != op->value.ty()) {
      var = op->var.copy_with_dtype(op->value.ty());
      var_remap_[op->var] = var;
    }

    PrimExpr body = VisitPrimExpr(op->body);

    if (value.same_as(op->value) && var.same_as(op->var) && body.same_as(op->body)) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      return Let(var, value, body);
    }
  }

  DEFINE_BIOP_EXPR_LEGALIZE(AddNode, operator+);
  DEFINE_BIOP_EXPR_LEGALIZE(SubNode, operator-);
  DEFINE_BIOP_EXPR_LEGALIZE(MulNode, operator*);
  DEFINE_BIOP_EXPR_LEGALIZE(DivNode, div);
  DEFINE_BIOP_EXPR_LEGALIZE(MinNode, min);
  DEFINE_BIOP_EXPR_LEGALIZE(MaxNode, max);
  DEFINE_BIOP_EXPR_LEGALIZE(LTNode, operator<);  // NOLINT(*)
  DEFINE_BIOP_EXPR_LEGALIZE(LENode, operator<=);
  DEFINE_BIOP_EXPR_LEGALIZE(GTNode, operator>);  // NOLINT(*)
  DEFINE_BIOP_EXPR_LEGALIZE(GENode, operator>=);
  DEFINE_BIOP_EXPR_LEGALIZE(EQNode, operator==);
  DEFINE_BIOP_EXPR_LEGALIZE(NENode, operator!=);

  Stmt VisitStmt_(const BindNode* op) final {
    auto prim_value = op->value.as<PrimExpr>();
    if (!prim_value) {
      return StmtExprMutator::VisitStmt_(op);
    }
    PrimExpr value = PromoteToTarget(prim_value.value());
    Var var = op->var;
    if (value.ty() != prim_value.value().ty()) {
      var = op->var.copy_with_dtype(prim_value.value().ty());
      var_remap_[op->var] = var;
    }

    if (value.same_as(op->value) && var.same_as(op->var)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      return Bind(var, value);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    PrimExpr value = this->VisitPrimExpr(op->value);
    auto fmutate = [this](const PrimExpr& e) { return this->VisitPrimExpr(e); };

    ffi::Array<PrimExpr> indices = op->indices.Map(fmutate);
    ffi::Optional<PrimExpr> predicate = std::nullopt;
    if (op->predicate.defined()) {
      predicate = this->VisitPrimExpr(op->predicate.value());
    }

    Buffer new_buf = GetRemappedBuffer(op->buffer);

    if (value.same_as(op->value) && indices.same_as(op->indices) &&
        predicate.same_as(op->predicate) && new_buf.same_as(op->buffer)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      if (MatchType(new_buf->dtype)) {
        int index_lanes = indices.size() ? indices.back().ty().lanes() : 1;
        int buffer_lanes = new_buf->dtype.lanes();
        PrimType legalized_dtype = new_buf->dtype.WithLanes(index_lanes * buffer_lanes);
        value = CastTargetToDType(value, legalized_dtype);
      }
      if (value.ty() != new_buf->dtype) {
        // this happens when buffer get rewritten to f32
        // but values remain as fp8/bf16
        TVM_FFI_ICHECK(MatchType(value.ty()));
        value = DTypeConversion(value, new_buf->dtype.WithLanes(value.ty().lanes()));
      }
      return BufferStore(new_buf, value, indices, predicate);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();
    if (auto buffer = op->node.as<Buffer>()) {
      auto it = buffer_remap_.find(buffer.value());
      if (it != buffer_remap_.end()) {
        return AttrStmt(it->second, op->attr_key, op->value, op->body);
      }
    } else if (auto var = op->node.as<Var>()) {
      auto it = var_remap_.find(var.value());
      if (it != var_remap_.end()) {
        return AttrStmt(it->second, op->attr_key, op->value, op->body);
      }
    } else if (auto reducer = op->node.as<CommReducerNode>()) {
      auto legalized_identity_elements = reducer->identity_element.Map(
          [this](PrimExpr expr) { return this->VisitPrimExpr(expr); });

      // Remap input variables
      for (size_t i = 0; i < legalized_identity_elements.size(); i++) {
        Var lhs_var = reducer->lhs[i];
        if (lhs_var->ty.as_or_throw<PrimType>() != legalized_identity_elements[i].ty()) {
          var_remap_[lhs_var] = lhs_var.copy_with_dtype(legalized_identity_elements[i].ty());
        }
        Var rhs_var = reducer->rhs[i];
        if (rhs_var->ty.as_or_throw<PrimType>() != legalized_identity_elements[i].ty()) {
          var_remap_[rhs_var] = rhs_var.copy_with_dtype(legalized_identity_elements[i].ty());
        }
      }

      auto legalized_results =
          reducer->result.Map([this](PrimExpr expr) { return this->VisitPrimExpr(expr); });

      auto legalized_lhs = reducer->lhs.Map([this](PrimVar var) {
        auto it = var_remap_.find(var);
        if (it != var_remap_.end()) {
          return it->second.as_or_throw<PrimVar>();
        }
        return var;
      });

      auto legalized_rhs = reducer->rhs.Map([this](PrimVar var) {
        auto it = var_remap_.find(var);
        if (it != var_remap_.end()) {
          return it->second.as_or_throw<PrimVar>();
        }
        return var;
      });
      return AttrStmt(CommReducer(legalized_lhs, legalized_rhs, legalized_results,
                                  legalized_identity_elements, reducer->span),
                      op->attr_key, op->value, op->body);
    }
    return ret;
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<DeclBufferNode>();

    Buffer new_buf = GetRemappedBuffer(op->buffer);
    if (new_buf.same_as(op->buffer)) {
      return ret;
    } else {
      return DeclBuffer(new_buf);
    }
  }

  Stmt VisitStmt_(const AllocBufferNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AllocBufferNode>();

    Buffer new_buf = GetRemappedBuffer(op->buffer);
    if (new_buf.same_as(op->buffer)) {
      return ret;
    } else {
      auto node = ret.as_or_throw<AllocBuffer>();
      node.CopyOnWrite()->buffer = new_buf;
      return node;
    }
  }

  Expr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr ret = StmtExprMutator::VisitExpr_(op).as_or_throw<PrimExpr>();
    op = ret.as<BufferLoadNode>();

    Buffer new_buf = GetRemappedBuffer(op->buffer);
    if (new_buf.same_as(op->buffer)) {
      return ret;
    } else {
      return BufferLoad(new_buf, op->indices, op->predicate);
    }
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
    if (const CastNode* cast = value.as<CastNode>()) {
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

  Buffer GetRemappedBuffer(Buffer buf) {
    auto buf_it = buffer_remap_.find(buf);
    if (buf_it != buffer_remap_.end()) {
      return buf_it->second;
    }
    return buf;
  }

 protected:
  PrimType promote_dtype_;
  std::unordered_map<Buffer, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> buffer_remap_;
  std::unordered_map<Var, Var> var_remap_;
};

class BF16ComputeLegalizer : public ComputeLegalizer {
 public:
  BF16ComputeLegalizer() : ComputeLegalizer(PrimType::Float(32)) {}
  PrimFunc Legalize(PrimFunc func) {
    BF16ComputeLegalizePlanner planner(&buffer_remap_, &var_remap_, promote_dtype_);
    return LegalizeWithPlanner(func, &planner);
  }
  bool MatchType(const Type& type) const {
    return MatchPrimType(type, [](const PrimType& prim_type) { return IsBFloat16Type(prim_type); });
  }
};

class FP8ComputeLegalizer : public ComputeLegalizer {
 public:
  explicit FP8ComputeLegalizer(PrimType promote_dtype) : ComputeLegalizer(promote_dtype) {}
  PrimFunc Legalize(PrimFunc func) {
    FP8ComputeLegalizePlanner planner(&buffer_remap_, &var_remap_, promote_dtype_);
    return LegalizeWithPlanner(func, &planner);
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
  PrimFunc Legalize(PrimFunc func) {
    TVM_FFI_ICHECK_EQ(func->buffer_map.size(), 0) << "This pass must be called after MakePackedAPI";
    auto* n = func.CopyOnWrite();
    n->params = n->params.Map([this](Var var) { return this->RemapVarDef(var); });
    n->body = this->VisitStmt(std::move(n->body));
    return func;
  }

 private:
  Expr VisitExpr_(const VarNode* op) final {
    Var var = ffi::GetRef<Var>(op);
    auto itr = var_remap_.find(var);
    if (itr != var_remap_.end()) {
      return itr->second;
    } else {
      return var;
    }
  }

  Stmt VisitStmt_(const AllocBufferNode* op) final {
    Buffer buf = GetRemappedBuffer(op->buffer);
    // in a rare case the buffer didn't get remapped
    // because the original var is not bfloat*
    // force remap here
    if (MatchType(buf->dtype)) {
      PrimType new_dtype = GetStorageUIntDType(buf->dtype);
      ffi::String storage_scope = "global";
      if (auto* ptr_type = buf->data->ty.as<PointerTypeNode>()) {
        storage_scope = ptr_type->storage_scope;
      }
      Var new_data = Var(buf->data->name_hint, PointerType(new_dtype, storage_scope));
      var_remap_[buf->data] = new_data;
      buf = Buffer(new_data, new_dtype, buf->shape, buf->strides, buf->elem_offset, buf->name,
                   buf->data_alignment, buf->offset_factor, buf->buffer_type, buf->axis_separators,
                   buf->span, buf->layout, buf->allocated_addr);
      buffer_remap_[op->buffer] = buf;
    }
    if (buf.same_as(op->buffer)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      auto node = ffi::GetRef<AllocBuffer>(op);
      node.CopyOnWrite()->buffer = buf;
      return node;
    }
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    Buffer buf = GetRemappedBuffer(op->buffer);
    // in a rare case the buffer didn't get remapped
    // because the original var is not bfloat*
    // force remap here
    if (MatchType(buf->dtype)) {
      buf = Buffer(buf->data, GetStorageUIntDType(buf->dtype), buf->shape, buf->strides,
                   buf->elem_offset, buf->name, buf->data_alignment, buf->offset_factor,
                   buf->buffer_type, buf->axis_separators, buf->span, buf->layout,
                   buf->allocated_addr);
      buffer_remap_[op->buffer] = buf;
    }
    if (buf.same_as(op->buffer)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      return DeclBuffer(buf, op->span);
    }
  }

  Expr VisitExpr_(const LetNode* op) final {
    PrimExpr value = VisitPrimExpr(op->value);
    Var var = RemapVarDef(op->var);
    PrimExpr body = VisitPrimExpr(op->body);

    if (value.same_as(op->value) && var.same_as(op->var) && body.same_as(op->body)) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      return Let(var, value, body);
    }
  }

  Stmt VisitStmt_(const BindNode* op) final {
    Expr value = VisitExpr(op->value);
    Var var = RemapVarDef(op->var);

    if (value.same_as(op->value) && var.same_as(op->var)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      return Bind(var, value);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    PrimExpr value = this->ChangeToUInt(VisitPrimExpr(op->value));
    Buffer new_buf = GetRemappedBuffer(op->buffer);
    auto indices = op->indices.Map([this](PrimExpr expr) { return this->VisitPrimExpr(expr); });
    ffi::Optional<PrimExpr> predicate = std::nullopt;
    if (op->predicate.defined()) {
      predicate = this->VisitPrimExpr(op->predicate.value());
    }
    if (new_buf.same_as(op->buffer) && indices.same_as(op->indices) &&
        predicate.same_as(op->predicate) && value.same_as(op->value)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      if (MatchType(op->value.ty())) {
        TVM_FFI_ICHECK(new_buf->dtype.MatchesCode(DLDataTypeCode::kDLUInt));
      }
      return BufferStore(new_buf, value, indices, predicate);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();

    if (auto buffer = op->node.as<Buffer>()) {
      auto it = buffer_remap_.find(buffer.value());
      if (it != buffer_remap_.end()) {
        return AttrStmt(it->second, op->attr_key, op->value, op->body);
      }
    } else if (auto var = op->node.as<Var>()) {
      auto it = var_remap_.find(var.value());
      if (it != var_remap_.end()) {
        return AttrStmt(it->second, op->attr_key, op->value, op->body);
      }
    }
    return ret;
  }

  Expr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr ret = StmtExprMutator::VisitExpr_(op).as_or_throw<PrimExpr>();
    op = ret.as<BufferLoadNode>();
    Buffer new_buf = GetRemappedBuffer(op->buffer);
    if (new_buf.same_as(op->buffer)) {
      return ret;
    } else {
      return BufferLoad(new_buf, op->indices, op->predicate);
    }
  }

  Expr VisitExpr_(const CallNode* op) final {
    if (!op->ty.as<PrimTypeNode>()) {
      return StmtExprMutator::VisitExpr_(op);
    }
    // remap re-interpret so un-necessary reinterpret can be skipped.
    if (op->op.same_as(builtin::reinterpret())) {
      PrimExpr value = VisitPrimExpr(op->args[0].as_or_throw<PrimExpr>());
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
    return StmtExprMutator::VisitExpr_(op);
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
    if (PrimType(GetRuntimeDataType(var->ty)).IsHandle()) {
      if (auto* ptr_type = var->ty.as<PointerTypeNode>()) {
        if (auto* elem_type = ptr_type->element_type.as<PrimTypeNode>()) {
          PrimType elem_prim_type = ffi::GetRef<PrimType>(elem_type);
          if (MatchType(elem_prim_type)) {
            Var new_var = Var(var->name_hint, PointerType(GetStorageUIntDType(elem_prim_type),
                                                          ptr_type->storage_scope));
            var_remap_[var] = new_var;
            return new_var;
          }
        }
      }
    }
    return var;
  }

  Buffer GetRemappedBuffer(Buffer buf) {
    auto buf_it = buffer_remap_.find(buf);
    if (buf_it != buffer_remap_.end()) {
      return buf_it->second;
    }
    Buffer new_buf = buf;
    auto var_it = var_remap_.find(buf->data);
    if (var_it != var_remap_.end()) {
      PrimType dtype = MatchType(buf->dtype) ? GetStorageUIntDType(buf->dtype) : buf->dtype;
      new_buf = Buffer(var_it->second, dtype, buf->shape, buf->strides, buf->elem_offset, buf->name,
                       buf->data_alignment, buf->offset_factor, buf->buffer_type,
                       buf->axis_separators, buf->span, buf->layout, buf->allocated_addr);
    } else {
      TVM_FFI_ICHECK(!MatchType(buf->dtype)) << "Cannot find var remap for " << buf;
    }

    buffer_remap_[buf] = new_buf;

    return new_buf;
  }

  std::unordered_map<Buffer, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> buffer_remap_;
  std::unordered_map<Var, Var> var_remap_;
};

class BF16StorageLegalizer : public StorageLegalizer {
 public:
  bool MatchType(const Type& type) const {
    return MatchPrimType(type, [](const PrimType& prim_type) { return IsBFloat16Type(prim_type); });
  }
};

class FP8StorageLegalizer : public StorageLegalizer {
 public:
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
    if (opt_target.defined() &&
        CheckDataTypeSupport(opt_target.value(), "tvm.support.nvcc.supports_bf16")) {
      return f;
    }
    return BF16ComputeLegalizer().Legalize(f);
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
    if (opt_target.defined() &&
        CheckDataTypeSupport(opt_target.value(), "tvm.support.nvcc.supports_bf16")) {
      return f;
    }
    return BF16StorageLegalizer().Legalize(f);
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
    if (opt_target.defined() &&
        CheckDataTypeSupport(opt_target.value(), "tvm.support.nvcc.supports_fp8")) {
      return f;
    }
    return FP8ComputeLegalizer(PrimType(ffi::StringToDLDataType(promote_dtype))).Legalize(f);
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
    if (opt_target.defined() &&
        CheckDataTypeSupport(opt_target.value(), "tvm.support.nvcc.supports_fp8")) {
      return f;
    }
    return FP8StorageLegalizer().Legalize(f);
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
