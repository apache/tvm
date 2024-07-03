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
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <cmath>
#include <tuple>

#include "dtype_conversion.h"

namespace tvm {
namespace tir {

// NOTE: do not touch buffer on function boundary
// remap internal fp8/bf16 buffer to f32 if they meet the following condition
// - constant allocation size
// - do not have raw pointer access to the buffer
//
// populate the buffer_remap and var_remap accordingly.
class ComputeLegalizePlanner : public StmtExprVisitor {
 public:
  ComputeLegalizePlanner(
      std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>* buffer_remap,
      std::unordered_map<Var, Var>* var_remap, DataType promote_dtype)
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
    Array<Buffer> drop_buffers;
    for (auto kv : *buffer_remap_) {
      if (opaque_var_access_.count(kv.first->data)) {
        drop_buffers.push_back(kv.first);
      }
    }
    for (Buffer buffer : drop_buffers) {
      auto it = buffer_remap_->find(buffer);
      ICHECK(it != buffer_remap_->end());
      buffer_remap_->erase(it);
    }
  }

  virtual bool MatchDType(DataType dtype) const = 0;

  void VisitStmt_(const AllocateNode* op) final {
    // remap all intermediate constant buffer to promote data types (fp16/fp32)
    if (MatchDType(op->dtype) && op->ConstantAllocationSize() != 0) {
      DataType dtype = promote_dtype_.with_lanes(op->dtype.lanes());
      String storage_scope = "global";
      if (auto* ptr_type = op->buffer_var->type_annotation.as<PointerTypeNode>()) {
        storage_scope = ptr_type->storage_scope;
      }
      Var buffer_var = Var(op->buffer_var->name_hint, PointerType(PrimType(dtype), storage_scope));
      (*var_remap_)[op->buffer_var] = buffer_var;
    }
    return StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    StmtExprVisitor::VisitStmt_(op);
    this->PopulateBufferRemap(op->buffer);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    StmtExprVisitor::VisitExpr_(op);
    this->PopulateBufferRemap(op->buffer);
  }

  void VisitStmt_(const DeclBufferNode* op) final {
    StmtExprVisitor::VisitStmt_(op);
    this->PopulateBufferRemap(op->buffer);
  }

  void VisitExpr_(const VarNode* op) final {
    StmtExprVisitor::VisitExpr_(op);
    Var buffer_var = GetRef<Var>(op);
    if (buffer_var.dtype().is_handle()) {
      opaque_var_access_.insert(buffer_var);
    }
  }

 private:
  void PopulateBufferRemap(Buffer buf) {
    auto var_it = var_remap_->find(buf->data);
    if (var_it == var_remap_->end()) return;

    Buffer new_buffer(var_it->second, promote_dtype_.with_lanes(buf->dtype.lanes()), buf->shape,
                      buf->strides, buf->elem_offset, buf->name, buf->data_alignment,
                      buf->offset_factor, buf->buffer_type, buf->axis_separators, buf->span);
    (*buffer_remap_)[buf] = new_buffer;
  }

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>* buffer_remap_;
  std::unordered_map<Var, Var>* var_remap_;
  std::unordered_set<Var> opaque_var_access_;
  DataType promote_dtype_;
};

class BF16ComputeLegalizePlanner : public ComputeLegalizePlanner {
 public:
  explicit BF16ComputeLegalizePlanner(
      std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>* buffer_remap,
      std::unordered_map<Var, Var>* var_remap, DataType promote_dtype)
      : ComputeLegalizePlanner(buffer_remap, var_remap, promote_dtype) {}
  bool MatchDType(DataType dtype) const { return dtype.is_bfloat16(); }
};

class FP8ComputeLegalizePlanner : public ComputeLegalizePlanner {
 public:
  explicit FP8ComputeLegalizePlanner(
      std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>* buffer_remap,
      std::unordered_map<Var, Var>* var_remap, DataType promote_dtype)
      : ComputeLegalizePlanner(buffer_remap, var_remap, promote_dtype) {}
  bool MatchDType(DataType dtype) const { return dtype.is_float8(); }
};

#define DEFINE_BIOP_EXPR_LEGALIZE(OP, FUNC)                      \
  PrimExpr VisitExpr_(const OP* op) final {                      \
    PrimExpr origin_a = PromoteToTarget(this->VisitExpr(op->a)); \
    PrimExpr origin_b = PromoteToTarget(this->VisitExpr(op->b)); \
                                                                 \
    if (origin_a.same_as(op->a) && origin_b.same_as(op->b)) {    \
      return GetRef<PrimExpr>(op);                               \
    } else {                                                     \
      return FUNC(origin_a, origin_b);                           \
    }                                                            \
  }

// NOTE: Legalize the FP8/BF16 computations
// to floating point computations and only keeps the
// fp8/bf16 storage which can further be legalized by FP8/BF16StorageLegalizer
// FP8/BF16StorageLegalizer will be called at a much later time
// point in the TIR lowering phases.
class ComputeLegalizer : public StmtExprMutator {
 public:
  explicit ComputeLegalizer(DataType promote_dtype) : promote_dtype_(promote_dtype) {}

  PrimFunc LegalizeWithPlanner(PrimFunc func, ComputeLegalizePlanner* planner) {
    planner->Plan(func);
    auto* n = func.CopyOnWrite();
    n->body = this->VisitStmt(std::move(n->body));
    return func;
  }

  virtual PrimFunc Legalize(PrimFunc func) = 0;

  virtual bool MatchDType(DataType dtype) const = 0;

 protected:
  PrimExpr VisitExpr_(const CastNode* op) final {
    auto op_val = PromoteToTarget(this->VisitExpr(op->value));

    // all casts to matched data type (fp8/bf16) becomes f32
    if (MatchDType(op->dtype)) {
      return cast(promote_dtype_.with_lanes(op->dtype.lanes()), op_val);
    }

    if (op_val.same_as(op->value)) {
      return GetRef<PrimExpr>(op);
    } else {
      return cast(op->dtype, op_val);
    }
  }

  PrimExpr VisitExpr_(const SelectNode* op) final {
    PrimExpr condition = this->VisitExpr(op->condition);
    PrimExpr true_value = PromoteToTarget(this->VisitExpr(op->true_value));
    PrimExpr false_value = PromoteToTarget(this->VisitExpr(op->false_value));
    if (condition.same_as(op->condition) && true_value.same_as(op->true_value) &&
        false_value.same_as(op->false_value)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Select(condition, true_value, false_value);
    }
  }

  PrimExpr VisitExpr_(const BroadcastNode* op) final {
    PrimExpr value = PromoteToTarget(this->VisitExpr(op->value));
    if (value.same_as(op->value)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Broadcast(value, op->lanes);
    }
  }

  PrimExpr VisitExpr_(const ShuffleNode* op) final {
    auto fexpr = [this](const PrimExpr& e) { return PromoteToTarget(this->VisitExpr(e)); };
    auto vectors = op->vectors.Map(fexpr);
    if (vectors.same_as(op->vectors)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Shuffle(vectors, op->indices);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    // presertve reinterpret<bf16>() behavior.
    if (op->op.same_as(builtin::reinterpret())) {
      return StmtExprMutator::VisitExpr_(op);
    }
    // update normal computations to return f32 instead.
    auto fmutate = [this](const PrimExpr& e) { return PromoteToTarget(this->VisitExpr(e)); };
    Array<PrimExpr> args = op->args.Map(fmutate);
    if (MatchDType(op->dtype)) {
      return Call(promote_dtype_.with_lanes(op->dtype.lanes()), op->op, args);
    }
    if (args.same_as(op->args)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Call(op->dtype, op->op, args);
    }
  }

  PrimExpr VisitExpr_(const FloatImmNode* op) final {
    if (MatchDType(op->dtype)) {
      return FloatImm(promote_dtype_, op->value);
    }
    return GetRef<PrimExpr>(op);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);

    auto itr = var_remap_.find(var);
    if (itr != var_remap_.end()) {
      return itr->second;
    } else {
      return std::move(var);
    }
  }

  PrimExpr VisitExpr_(const LetNode* op) final {
    PrimExpr value = PromoteToTarget(op->value);
    Var var = op->var;
    if (value.dtype() != op->value.dtype()) {
      var = op->var.copy_with_dtype(op->value.dtype());
      var_remap_[op->var] = var;
    }

    PrimExpr body = VisitExpr(op->body);

    if (value.same_as(op->value) && var.same_as(op->var) && body.same_as(op->body)) {
      return GetRef<PrimExpr>(op);
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

  Stmt VisitStmt_(const LetStmtNode* op) final {
    PrimExpr value = PromoteToTarget(op->value);
    Var var = op->var;
    if (value.dtype() != op->value.dtype()) {
      var = op->var.copy_with_dtype(op->value.dtype());
      var_remap_[op->var] = var;
    }
    Stmt body = VisitStmt(op->body);

    if (value.same_as(op->value) && var.same_as(op->var) && body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      return LetStmt(var, value, body);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    auto fmutate = [this](const PrimExpr& e) { return this->VisitExpr(e); };

    Array<PrimExpr> indices = op->indices.Map(fmutate);

    Buffer new_buf = GetRemappedBuffer(op->buffer);

    if (value.same_as(op->value) && indices.same_as(op->indices) && new_buf.same_as(op->buffer)) {
      return GetRef<Stmt>(op);
    } else {
      if (MatchDType(new_buf->dtype)) {
        int index_lanes = indices.size() ? indices.back().dtype().lanes() : 1;
        int buffer_lanes = new_buf->dtype.lanes();
        DataType legalized_dtype = new_buf->dtype.with_lanes(index_lanes * buffer_lanes);
        value = CastTargetToDType(value, legalized_dtype);
      }
      if (value.dtype() != new_buf->dtype) {
        // this happens when buffer get rewritten to f32
        // but values remain as fp8/bf16
        ICHECK(MatchDType(value->dtype));
        value = cast(new_buf->dtype.with_lanes(value.dtype().lanes()), value);
      }
      ICHECK(!op->predicate.defined()) << "Predicated buffer store is not currently supported in "
                                          "data type legalizer pass.";
      return BufferStore(new_buf, value, indices);
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

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<BufferRealizeNode>();

    Buffer new_buf = GetRemappedBuffer(op->buffer);
    if (new_buf.same_as(op->buffer)) {
      return ret;
    } else {
      return BufferRealize(new_buf, op->bounds, op->condition, op->body);
    }
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<DeclBufferNode>();

    Buffer new_buf = GetRemappedBuffer(op->buffer);
    if (new_buf.same_as(op->buffer)) {
      return ret;
    } else {
      return DeclBuffer(new_buf, op->body);
    }
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AllocateNode>();

    auto it = var_remap_.find(op->buffer_var);
    if (it != var_remap_.end()) {
      Var remapped_var = it->second;
      auto* ptr = remapped_var->type_annotation.as<PointerTypeNode>();
      ICHECK(ptr);
      auto* prim_type = ptr->element_type.as<PrimTypeNode>();
      ICHECK(prim_type);
      return Allocate(remapped_var, prim_type->dtype, op->extents, op->condition, op->body);
    } else {
      return ret;
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr ret = StmtExprMutator::VisitExpr_(op);
    op = ret.as<BufferLoadNode>();

    Buffer new_buf = GetRemappedBuffer(op->buffer);
    if (new_buf.same_as(op->buffer)) {
      return ret;
    } else {
      ICHECK(!op->predicate.defined()) << "Predicated buffer load is not currently supported in "
                                          "data type legalizer pass.";
      return BufferLoad(new_buf, op->indices);
    }
  }

 private:
  /*!
   * \brief promote value to target datatype F16/F32 and keep other values unchanged.
   * \param value The input value.
   * \return The converted value.
   */
  PrimExpr PromoteToTarget(PrimExpr value) {
    if (!MatchDType(value.dtype())) return value;
    if (const CastNode* cast = value.as<CastNode>()) {
      if (cast->value.dtype() == promote_dtype_.with_lanes(value.dtype().lanes()))
        return cast->value;
    }
    return DTypeConversion(value, promote_dtype_.with_lanes(value.dtype().lanes()));
  }

  /*!
   * \brief Cast value from promoted datatype (FP16/FP32) back to BF16/FP8 and keep other values
   *   unchanged.
   * \param value The input value
   * \return The converted value.
   */
  PrimExpr CastTargetToDType(PrimExpr value, DataType dtype) {
    if (!value.dtype().is_float()) return value;
    ICHECK_EQ(value.dtype(), this->promote_dtype_.with_lanes(value.dtype().lanes()));
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
  DataType promote_dtype_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_remap_;
  std::unordered_map<Var, Var> var_remap_;
};

class BF16ComputeLegalizer : public ComputeLegalizer {
 public:
  BF16ComputeLegalizer() : ComputeLegalizer(DataType::Float(32)) {}
  PrimFunc Legalize(PrimFunc func) {
    BF16ComputeLegalizePlanner planner(&buffer_remap_, &var_remap_, promote_dtype_);
    return LegalizeWithPlanner(func, &planner);
  }
  bool MatchDType(DataType dtype) const { return dtype.is_bfloat16(); }
};

class FP8ComputeLegalizer : public ComputeLegalizer {
 public:
  explicit FP8ComputeLegalizer(DataType promote_dtype) : ComputeLegalizer(promote_dtype) {}
  PrimFunc Legalize(PrimFunc func) {
    FP8ComputeLegalizePlanner planner(&buffer_remap_, &var_remap_, promote_dtype_);
    return LegalizeWithPlanner(func, &planner);
  }
  bool MatchDType(DataType dtype) const { return dtype.is_float8(); }
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
    ICHECK_EQ(func->buffer_map.size(), 0) << "This pass must be called after MakePackedAPI";
    auto* n = func.CopyOnWrite();
    n->params = n->params.Map([this](Var var) { return this->RemapVarDef(var); });
    n->body = this->VisitStmt(std::move(n->body));
    return func;
  }

 private:
  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    auto itr = var_remap_.find(var);
    if (itr != var_remap_.end()) {
      return itr->second;
    } else {
      return std::move(var);
    }
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    if (MatchDType(op->dtype)) {
      DataType dtype = GetStorageUIntDType(op->dtype);
      String storage_scope = "global";
      if (auto* ptr_type = op->buffer_var->type_annotation.as<PointerTypeNode>()) {
        storage_scope = ptr_type->storage_scope;
      }
      Var buffer_var = Var(op->buffer_var->name_hint, PointerType(PrimType(dtype), storage_scope));
      var_remap_[op->buffer_var] = buffer_var;
      return VisitStmt(Allocate(buffer_var, dtype, op->extents, op->condition, op->body));
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    Buffer buf = GetRemappedBuffer(op->buffer);
    // in a rare case the buffer didn't get remapped
    // because the original var is not bfloat*
    // force remap here
    if (MatchDType(buf->dtype)) {
      buf = Buffer(buf->data, GetStorageUIntDType(buf->dtype), buf->shape, buf->strides,
                   buf->elem_offset, buf->name, buf->data_alignment, buf->offset_factor,
                   buf->buffer_type, buf->axis_separators, buf->span);
      buffer_remap_[op->buffer] = buf;
    }
    Stmt body = VisitStmt(op->body);
    if (buf.same_as(op->buffer) && body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      return DeclBuffer(buf, body, op->span);
    }
  }

  PrimExpr VisitExpr_(const LetNode* op) final {
    PrimExpr value = VisitExpr(op->value);
    Var var = RemapVarDef(op->var);
    PrimExpr body = VisitExpr(op->body);

    if (value.same_as(op->value) && var.same_as(op->var) && body.same_as(op->body)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Let(var, value, body);
    }
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    PrimExpr value = VisitExpr(op->value);
    Var var = RemapVarDef(op->var);
    Stmt body = VisitStmt(op->body);

    if (value.same_as(op->value) && var.same_as(op->var) && body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      return LetStmt(var, value, body);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    PrimExpr value = this->ChangeToUInt(VisitExpr(op->value));
    Buffer new_buf = GetRemappedBuffer(op->buffer);
    auto indices = op->indices.Map([this](PrimExpr expr) { return this->VisitExpr(expr); });
    if (new_buf.same_as(op->buffer) && indices.same_as(op->indices) && value.same_as(op->value)) {
      return GetRef<Stmt>(op);
    } else {
      if (MatchDType(op->value.dtype())) {
        ICHECK(new_buf->dtype.is_uint());
      }
      ICHECK(!op->predicate.defined()) << "Predicated buffer store is not currently supported in "
                                          "data type legalizer pass.";
      return BufferStore(new_buf, value, indices);
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

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    LOG(FATAL) << "Do not expect buffer realize";
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr ret = StmtExprMutator::VisitExpr_(op);
    op = ret.as<BufferLoadNode>();
    Buffer new_buf = GetRemappedBuffer(op->buffer);
    if (new_buf.same_as(op->buffer)) {
      return ret;
    } else {
      ICHECK(!op->predicate.defined()) << "Predicated buffer load is not currently supported in "
                                          "data type legalizer pass.";
      return BufferLoad(new_buf, op->indices);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    // remap re-interpret so un-necessary reinterpret can be skipped.
    if (op->op.same_as(builtin::reinterpret())) {
      PrimExpr value = VisitExpr(op->args[0]);
      // sometimes the input dtype can change and we can skip.
      if (value.dtype() == op->dtype) return value;
      if (MatchDType(op->dtype)) {
        return reinterpret(GetStorageUIntDType(op->dtype), value);
      }
      if (op->args[0].same_as(value)) {
        return GetRef<PrimExpr>(op);
      } else {
        return reinterpret(op->dtype, value);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  virtual bool MatchDType(DataType dtype) const = 0;

 private:
  /*!
   * \brief Change float value to uint value.
   * \param value The input value.
   * \return The converted value.
   */
  PrimExpr ChangeToUInt(PrimExpr value) {
    if (!MatchDType(value->dtype)) return value;
    auto* call = value.as<CallNode>();
    if (call && call->op.same_as(builtin::reinterpret())) {
      return reinterpret(GetStorageUIntDType(value->dtype), call->args[0]);
    } else {
      return value;
    }
  }

  Var RemapVarDef(Var var) {
    // remap the var
    if (var.dtype().is_handle()) {
      if (auto* ptr_type = var->type_annotation.as<PointerTypeNode>()) {
        if (auto* elem_type = ptr_type->element_type.as<PrimTypeNode>()) {
          if (MatchDType(elem_type->dtype)) {
            Var new_var =
                Var(var->name_hint, PointerType(PrimType(GetStorageUIntDType(elem_type->dtype)),
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
      DataType dtype = MatchDType(buf->dtype) ? GetStorageUIntDType(buf->dtype) : buf->dtype;
      new_buf = Buffer(var_it->second, dtype, buf->shape, buf->strides, buf->elem_offset, buf->name,
                       buf->data_alignment, buf->offset_factor, buf->buffer_type,
                       buf->axis_separators, buf->span);
    } else {
      ICHECK(!MatchDType(buf->dtype)) << "Cannot find var remap for " << buf;
    }

    buffer_remap_[buf] = new_buf;

    return new_buf;
  }

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_remap_;
  std::unordered_map<Var, Var> var_remap_;
};

class BF16StorageLegalizer : public StorageLegalizer {
 public:
  bool MatchDType(DataType dtype) const { return dtype.is_bfloat16(); }
};

class FP8StorageLegalizer : public StorageLegalizer {
 public:
  bool MatchDType(DataType dtype) const { return dtype.is_float8(); }
};

namespace transform {

bool CheckDataTypeSupport(const Target& target, const std::string& support_func_name) {
  bool has_native_support = false;
  if (target->kind->name == "cuda") {
    if (const PackedFunc* get_cv =
            tvm::runtime::Registry::Get("tvm.contrib.nvcc.get_compute_version")) {
      std::string compute_version = (*get_cv)(target);
      if (const PackedFunc* check_support = tvm::runtime::Registry::Get(support_func_name)) {
        has_native_support = (*check_support)(compute_version);
      }
    }
  }
  return has_native_support;
}

Pass BF16ComputeLegalize() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    // TODO(tvm-team): skip if the target supports bf16
    return BF16ComputeLegalizer().Legalize(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.BF16ComputeLegalize", {});
}

TVM_REGISTER_GLOBAL("tir.transform.BF16ComputeLegalize").set_body_typed(BF16ComputeLegalize);

Pass BF16StorageLegalize() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    // TODO(tvm-team): skip if the target supports bf16
    return BF16StorageLegalizer().Legalize(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.BF16StorageLegalize", {});
}

TVM_REGISTER_GLOBAL("tir.transform.BF16StorageLegalize").set_body_typed(BF16StorageLegalize);

Pass FP8ComputeLegalize(String promote_dtype_str) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget).value();
    if (CheckDataTypeSupport(target, "tvm.contrib.nvcc.supports_fp8")) {
      return f;
    }
    return FP8ComputeLegalizer(DataType(String2DLDataType(promote_dtype_str))).Legalize(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.FP8ComputeLegalize", {});
}

TVM_REGISTER_GLOBAL("tir.transform.FP8ComputeLegalize").set_body_typed(FP8ComputeLegalize);

Pass FP8StorageLegalize() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget).value();
    if (CheckDataTypeSupport(target, "tvm.contrib.nvcc.supports_fp8")) {
      return f;
    }
    return FP8StorageLegalizer().Legalize(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.FP8StorageLegalize", {});
}

TVM_REGISTER_GLOBAL("tir.transform.FP8StorageLegalize").set_body_typed(FP8StorageLegalize);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
