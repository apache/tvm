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
 * \file bf16_legalize.cc
 * \brief legalize bf16 type by adding cast_to_fp32
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <cmath>
#include <tuple>

namespace tvm {
namespace tir {

// NOTE: do not touch buffer on function boundary
// remap internal bf16 buffer to f32 if they meet the following condition
// - constant allocation size
// - do not have raw pointer access to the buffer
//
// populate the buffer_remap and var_remap accordingly.
class BF16ComputeLegalizePlanner : public StmtExprVisitor {
 public:
  BF16ComputeLegalizePlanner(
      std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>* buffer_remap,
      std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual>* var_remap)
      : buffer_remap_(buffer_remap), var_remap_(var_remap) {}

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

  void VisitStmt_(const AllocateNode* op) final {
    // remap all intermediate constant buffr to fp32
    if (op->dtype.is_bfloat16() && op->ConstantAllocationSize() != 0) {
      DataType dtype = DataType::Float(32, op->dtype.lanes());
      Var buffer_var = Var(op->buffer_var->name_hint, PointerType(PrimType(dtype)));
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

    Buffer new_buffer(var_it->second, DataType::Float(32, buf->dtype.lanes()), buf->shape,
                      buf->strides, buf->elem_offset, buf->name, buf->data_alignment,
                      buf->offset_factor, buf->buffer_type, buf->axis_separators, buf->span);
    (*buffer_remap_)[buf] = new_buffer;
  }

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>* buffer_remap_;
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual>* var_remap_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> opaque_var_access_;
};

#define DEFINE_BIOP_EXPR_LEGALIZE(OP, FUNC)                       \
  PrimExpr VisitExpr_(const OP* op) final {                       \
    PrimExpr origin_a = PromoteBF16ToF32(this->VisitExpr(op->a)); \
    PrimExpr origin_b = PromoteBF16ToF32(this->VisitExpr(op->b)); \
                                                                  \
    if (origin_a.same_as(op->a) && origin_b.same_as(op->b)) {     \
      return GetRef<PrimExpr>(op);                                \
    } else {                                                      \
      return FUNC(origin_a, origin_b);                            \
    }                                                             \
  }

// NOTE: Legalize the BF16 computations
// to floating point computations and only keeps the
// bf16 storage which can further be legalized by BF16StorageLegalizer
// BF16StorageLegalizer will be called at a much later time
// point in the TIR lowering phases.
class BF16ComputeLegalizer : public StmtExprMutator {
 public:
  PrimFunc Legalize(PrimFunc func) {
    BF16ComputeLegalizePlanner planner(&buffer_remap_, &var_remap_);
    planner.Plan(func);
    auto* n = func.CopyOnWrite();
    n->body = this->VisitStmt(std::move(n->body));
    return func;
  }

 protected:
  PrimExpr VisitExpr_(const CastNode* op) final {
    auto op_val = PromoteBF16ToF32(this->VisitExpr(op->value));

    // all casts to BF16 becomes f32
    if (op->dtype.is_bfloat16()) {
      return cast(DataType::Float(32, op->dtype.lanes()), op_val);
    }

    if (op_val.same_as(op->value)) {
      return GetRef<PrimExpr>(op);
    } else {
      return cast(op->dtype, op_val);
    }
  }

  PrimExpr VisitExpr_(const SelectNode* op) final {
    PrimExpr condition = this->VisitExpr(op->condition);
    PrimExpr true_value = PromoteBF16ToF32(this->VisitExpr(op->true_value));
    PrimExpr false_value = PromoteBF16ToF32(this->VisitExpr(op->false_value));
    if (condition.same_as(op->condition) && true_value.same_as(op->true_value) &&
        false_value.same_as(op->false_value)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Select(condition, true_value, false_value);
    }
  }

  PrimExpr VisitExpr_(const BroadcastNode* op) final {
    PrimExpr value = PromoteBF16ToF32(this->VisitExpr(op->value));
    if (value.same_as(op->value)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Broadcast(value, op->lanes);
    }
  }

  PrimExpr VisitExpr_(const ShuffleNode* op) final {
    auto fexpr = [this](const PrimExpr& e) { return PromoteBF16ToF32(this->VisitExpr(e)); };
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
    auto fmutate = [this](const PrimExpr& e) { return PromoteBF16ToF32(this->VisitExpr(e)); };
    Array<PrimExpr> args = op->args.Map(fmutate);
    if (op->dtype.is_bfloat16()) {
      return Call(DataType::Float(32, op->dtype.lanes()), op->op, args);
    }
    if (args.same_as(op->args)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Call(op->dtype, op->op, args);
    }
  }

  PrimExpr VisitExpr_(const FloatImmNode* op) final {
    if (op->dtype.is_bfloat16()) {
      return FloatImm(DataType::Float(32), op->value);
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
    PrimExpr value = PromoteBF16ToF32(op->value);
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
    PrimExpr value = PromoteBF16ToF32(op->value);
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
      if (new_buf->dtype.is_bfloat16()) {
        value = CastF32ToBF16(value);
      }
      if (value.dtype() != new_buf->dtype) {
        // this happens when buffer get rewritten to f32
        // but values remain as bf16
        ICHECK(value.dtype().is_bfloat16());
        value = cast(new_buf->dtype.with_lanes(value.dtype().lanes()), value);
      }
      return BufferStore(new_buf, value, indices);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();

    if (auto* buffer = op->node.as<BufferNode>()) {
      auto it = buffer_remap_.find(GetRef<Buffer>(buffer));
      if (it != buffer_remap_.end()) {
        return AttrStmt(it->second, op->attr_key, op->value, op->body);
      }
    } else if (auto* var = op->node.as<VarNode>()) {
      auto it = var_remap_.find(GetRef<Var>(var));
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
      return BufferLoad(new_buf, op->indices);
    }
  }

 private:
  /*!
   * \brief promote BF16 to F32 and keep other values unchanged.
   * \param value The input value.
   * \return The converted value.
   */
  PrimExpr PromoteBF16ToF32(PrimExpr value) {
    if (!value.dtype().is_bfloat16()) return value;
    if (const CastNode* cast = value.as<CastNode>()) {
      if (cast->value.dtype() == DataType::Float(32)) return cast->value;
    }
    DataType f32 = DataType::Float(32, value.dtype().lanes());
    DataType u16 = DataType::UInt(16, value.dtype().lanes());
    DataType u32 = DataType::UInt(32, value.dtype().lanes());
    // reinterpret<f32>((cast<u32>(reinterpret<u16>(bf16_value)) << 16))
    return reinterpret(f32, cast(u32, reinterpret(u16, value)) << 16);
  }

  /*!
   * \brief Cast value to F32 to BF16 and keep other values unchanged.
   * \param value The input value
   * \return The converted value.
   */
  PrimExpr CastF32ToBF16(PrimExpr value) {
    if (!value.dtype().is_float()) return value;
    ICHECK_EQ(value.dtype().bits(), 32);
    DataType bf16 = DataType::BFloat(16, value.dtype().lanes());
    DataType u16 = DataType::UInt(16, value.dtype().lanes());
    DataType u32 = DataType::UInt(32, value.dtype().lanes());
    PrimExpr u32_val = reinterpret(u32, value);

    if (round_to_even_) {
      PrimExpr rounding_bias = ((u32_val >> 16) & 1) + make_const(u32, 0x7FFF);
      u32_val = u32_val + rounding_bias;
    }
    // reinterpret<bf16>((cast<u16>(reinterpret<u32>(f32_value)) >> 16))
    return reinterpret(bf16, cast(u16, u32_val >> 16));
  }

  Buffer GetRemappedBuffer(Buffer buf) {
    auto buf_it = buffer_remap_.find(buf);
    if (buf_it != buffer_remap_.end()) {
      return buf_it->second;
    }
    return buf;
  }

  bool round_to_even_{true};

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_remap_;
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> var_remap_;
};

/*!
 * \brief This Pass legalizes remaining BF16 storages to u16
 *
 * This pass needs to happens after BF16ComputeLegalizer and serves
 * as a way to support BF16 on platforms that do not have native support.
 */
class BF16StorageLegalizer : public StmtExprMutator {
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
    if (op->dtype.is_bfloat16()) {
      DataType dtype = DataType::UInt(16, op->dtype.lanes());
      Var buffer_var = Var(op->buffer_var->name_hint, PointerType(PrimType(dtype)));
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
    if (buf->dtype.is_bfloat16()) {
      buf = Buffer(buf->data, DataType::UInt(16, buf->dtype.lanes()), buf->shape, buf->strides,
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
    PrimExpr value = this->ChangeBF16ToU16(VisitExpr(op->value));
    Buffer new_buf = GetRemappedBuffer(op->buffer);
    auto indices = op->indices.Map([this](PrimExpr expr) { return this->VisitExpr(expr); });
    if (new_buf.same_as(op->buffer) && indices.same_as(op->indices) && value.same_as(op->value)) {
      return GetRef<Stmt>(op);
    } else {
      if (op->value.dtype().is_bfloat16()) {
        ICHECK(new_buf->dtype.is_uint());
      }
      return BufferStore(new_buf, value, indices);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();

    if (auto* buffer = op->node.as<BufferNode>()) {
      auto it = buffer_remap_.find(GetRef<Buffer>(buffer));
      if (it != buffer_remap_.end()) {
        return AttrStmt(it->second, op->attr_key, op->value, op->body);
      }
    } else if (auto* var = op->node.as<VarNode>()) {
      auto it = var_remap_.find(GetRef<Var>(var));
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
      return BufferLoad(new_buf, op->indices);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    // remap re-interpret so un-necessary reinterpret can be skipped.
    if (op->op.same_as(builtin::reinterpret())) {
      PrimExpr value = VisitExpr(op->args[0]);
      // sometimes the input dtype can change and we can skip.
      if (value.dtype() == op->dtype) return value;
      if (op->dtype.is_bfloat16()) {
        return reinterpret(DataType::UInt(16, op->dtype.lanes()), value);
      }
      if (op->args[0].same_as(value)) {
        return GetRef<PrimExpr>(op);
      } else {
        return reinterpret(op->dtype, value);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

 private:
  /*!
   * \brief Change BF16 value to U16 value.
   * \param value The input value.
   * \return The converted value.
   */
  PrimExpr ChangeBF16ToU16(PrimExpr value) {
    if (!value.dtype().is_bfloat16()) return value;
    auto* call = value.as<CallNode>();
    if (call && call->op.same_as(builtin::reinterpret())) {
      return reinterpret(DataType::UInt(16, value.dtype().lanes()), call->args[0]);
    } else {
      return value;
    }
  }

  Var RemapVarDef(Var var) {
    // remap the var
    if (var.dtype().is_handle()) {
      if (auto* ptr_type = var->type_annotation.as<PointerTypeNode>()) {
        if (auto* elem_type = ptr_type->element_type.as<PrimTypeNode>()) {
          if (elem_type->dtype.is_bfloat16()) {
            Var new_var = Var(var->name_hint,
                              PointerType(PrimType(DataType::UInt(16, elem_type->dtype.lanes()))));
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
      DataType dtype =
          buf->dtype.is_bfloat16() ? DataType::UInt(16, buf->dtype.lanes()) : buf->dtype;
      new_buf = Buffer(var_it->second, dtype, buf->shape, buf->strides, buf->elem_offset, buf->name,
                       buf->data_alignment, buf->offset_factor, buf->buffer_type,
                       buf->axis_separators, buf->span);
    } else {
      ICHECK(!buf->dtype.is_bfloat16()) << "Cannot find var remap for " << buf;
    }

    buffer_remap_[buf] = new_buf;

    return new_buf;
  }

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_remap_;
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> var_remap_;
};

namespace transform {

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

}  // namespace transform
}  // namespace tir
}  // namespace tvm
