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
 * \file data_type_rewriter.cc
 * \brief Rewrite the data type of expressions.
 */

#include "data_type_rewriter.h"

#include <tvm/ffi/cast.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "tvm/ir/expr.h"
#include "tvm/ir/prim/expr.h"
#include "tvm/tirx/stmt.h"
#include "tvm/tirx/var.h"

namespace tvm {
namespace tirx {

UnchangedOr<Stmt> DataTypeLegalizer::Mutate_(const ForNode* op, InplaceMode inplace_mode) {
  auto result = StmtExprMutator::Mutate_(op, inplace_mode);
  if (!result.IsUnchanged()) {
    op = ffi::AnyView(result).as<ForNode>();
    TVM_FFI_ICHECK(op != nullptr) << "Expected type to be ForNode, but get "
                                  << ffi::AnyView(result).GetTypeKey();
    if (!op->unique()) inplace_mode = InplaceMode::kDisallow;
  }
  PrimExpr e = Mutate(op->loop_var, inplace_mode).ValueOrUnchanged(op->loop_var);
  Var var = e.as_or_throw<Var>();
  PrimType var_ty = var->ty.as_or_throw<PrimType>();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* n = const_cast<ForNode*>(op);
    n->min = cast(var_ty, op->min);
    n->extent = cast(var_ty, op->extent);
    if (op->step.has_value()) {
      n->step = cast(var_ty, *op->step);
    }
    return result;
  }
  auto n = ffi::make_object<ForNode>(*op);
  n->min = cast(var_ty, op->min);
  n->extent = cast(var_ty, op->extent);
  if (op->step.has_value()) {
    n->step = cast(var_ty, *op->step);
  }
  return For(n);
}

UnchangedOr<Stmt> DataTypeLegalizer::Mutate_(const SBlockRealizeNode* op,
                                             InplaceMode inplace_mode) {
  SBlockRealize realize = StmtExprMutator::Mutate_(op, inplace_mode)
                              .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                              .as_or_throw<SBlockRealize>();
  ffi::Array<PrimExpr> new_iter_values;
  bool changed = false;
  for (int i = 0; i < static_cast<int>(op->iter_values.size()); ++i) {
    PrimType dtype = realize->block->iter_vars[i]->var.ty();
    if (op->iter_values[i].ty() != dtype) {
      new_iter_values.push_back(cast(dtype, realize->iter_values[i]));
      changed = true;
    } else {
      new_iter_values.push_back(realize->iter_values[i]);
    }
  }
  if (changed) {
    realize.CopyOnWrite()->iter_values = std::move(new_iter_values);
  }
  return realize;
}

UnchangedOr<Stmt> DataTypeLegalizer::Mutate_(const SBlockNode* op, InplaceMode inplace_mode) {
  SBlock new_block = StmtExprMutator::Mutate_(op, inplace_mode)
                         .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                         .as_or_throw<SBlock>();
  ffi::Array<IterVar> new_iter_vars = new_block->iter_vars.Map([](const IterVar& iter) {
    PrimType dtype = iter->var.ty();
    if (iter->dom->min.ty() != dtype || iter->dom->extent.ty() != dtype) {
      IterVar new_iter = iter;
      new_iter.CopyOnWrite()->dom =
          Range(cast(dtype, iter->dom->min), cast(dtype, iter->dom->extent));
      return new_iter;
    } else {
      return iter;
    }
  });
  if (!op->iter_vars.same_as(new_iter_vars)) {
    new_block.CopyOnWrite()->iter_vars = std::move(new_iter_vars);
  }
  return new_block;
}

UnchangedOr<Stmt> DataTypeLegalizer::Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) {
  if (op->attr_key == attr::thread_extent || op->attr_key == s_tir::attr::virtual_thread) {
    Stmt s = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = s.as<AttrStmtNode>();
    TVM_FFI_ICHECK(op != nullptr) << "Expected type to be AttrStmtNode"
                                  << ", but get " << s->GetTypeKey();
    const IterVarNode* iv = op->node.as<IterVarNode>();
    TVM_FFI_ICHECK(iv != nullptr) << "Expected type to be IterVarNode"
                                  << ", but get " << op->node.GetTypeKey();
    PrimExpr e = Mutate(iv->var).ValueOrUnchanged(iv->var);
    PrimVar var = e.as_or_throw<PrimVar>();
    if (ivmap_.find(iv) == ivmap_.end()) {
      Range dom = iv->dom;
      if (dom.defined()) {
        PrimExpr extend = dom->extent;
        PrimType extend_ty = extend.ty();
        PrimType var_ty = var.ty();
        TVM_FFI_ICHECK(extend_ty.MatchesCode(DLDataTypeCode::kDLInt) &&
                       var_ty.MatchesCode(DLDataTypeCode::kDLInt));
        if (var_ty.bits() != extend_ty.bits()) {
          dom = Range(cast(var_ty, dom->min), cast(var_ty, extend), dom->span);
        }
      }
      ivmap_[iv] = IterVar(dom, var, iv->iter_type, iv->thread_tag);
    }
    return AttrStmt(ivmap_[iv], op->attr_key, cast(var.ty(), op->value), op->body);
  }
  return StmtExprMutator::Mutate_(op, inplace_mode);
}

UnchangedOr<PrimExpr> DataTypeLegalizer::Mutate_(const prim::LetNode* op,
                                                 InplaceMode inplace_mode) {
  auto value_result = this->Mutate(op->value, inplace_mode);
  bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
  PrimExpr value = std::move(value_result).ValueOrUnchanged(op->value);
  Var var = Mutate(op->var).ValueOrUnchanged(op->var).as_or_throw<Var>();
  if (value.ty() != var->ty.as_or_throw<PrimType>()) {
    if (var.same_as(op->var)) {
      var = op->var.CopyWithDType(value.ty());
      VarRemapSet(op->var, var);
    } else {
      value = cast(var->ty.as_or_throw<PrimType>(), value);
      value_unchanged = false;
    }
  }
  auto new_body_result = this->Mutate(op->body, inplace_mode);
  bool new_body_unchanged = new_body_result.UnchangedOrSameAs(op->body);
  PrimExpr new_body = std::move(new_body_result).ValueOrUnchanged(op->body);

  if (value_unchanged && new_body_unchanged && var.same_as(op->var)) {
    return ffi::Unchanged();
  } else {
    return prim::Let(var, value, new_body, op->span);
  }
}

UnchangedOr<Stmt> DataTypeLegalizer::Mutate_(const BindNode* op, InplaceMode inplace_mode) {
  auto value_result = this->Mutate(op->value, inplace_mode);
  bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
  Expr value = std::move(value_result).ValueOrUnchanged(op->value);
  Var var = op->var;

  if (auto prim_value = value.as<PrimExpr>()) {
    if (prim_value.value().ty() != op->var->ty.as_or_throw<PrimType>()) {
      var = op->var.CopyWithDType(prim_value.value().ty());
      VarRemapSet(op->var, var);
    }
  }

  if (value_unchanged && var.same_as(op->var)) {
    return ffi::Unchanged();
  } else {
    return Bind(var, value, op->span);
  }
}

UnchangedOr<PrimExpr> DataTypeLegalizer::Mutate_(const prim::SelectNode* op,
                                                 InplaceMode inplace_mode) {
  auto condition_result = this->Mutate(op->condition, inplace_mode);
  bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
  PrimExpr condition = std::move(condition_result).ValueOrUnchanged(op->condition);
  auto true_value_result = this->Mutate(op->true_value, inplace_mode);
  bool true_value_unchanged = true_value_result.UnchangedOrSameAs(op->true_value);
  PrimExpr true_value = std::move(true_value_result).ValueOrUnchanged(op->true_value);
  auto false_value_result = this->Mutate(op->false_value, inplace_mode);
  bool false_value_unchanged = false_value_result.UnchangedOrSameAs(op->false_value);
  PrimExpr false_value = std::move(false_value_result).ValueOrUnchanged(op->false_value);
  if (condition_unchanged && true_value_unchanged && false_value_unchanged &&
      true_value.ty() == false_value.ty()) {
    return ffi::Unchanged();
  } else {
    PrimType true_dtype = true_value.ty();
    PrimType false_dtype = false_value.ty();
    int bits = std::max(true_dtype.bits(), false_dtype.bits());
    PrimType dtype = true_dtype.WithBits(bits);
    if (true_dtype != dtype) true_value = cast(dtype, true_value);
    if (false_dtype != dtype) false_value = cast(dtype, false_value);
    return prim::Select(condition, true_value, false_value);
  }
}

UnchangedOr<PrimExpr> DataTypeLegalizer::Mutate_(const prim::RampNode* op,
                                                 InplaceMode inplace_mode) {
  auto base_result = Mutate(op->base, inplace_mode);
  bool base_unchanged = base_result.UnchangedOrSameAs(op->base);
  PrimExpr base = std::move(base_result).ValueOrUnchanged(op->base);
  auto stride_result = Mutate(op->stride, inplace_mode);
  bool stride_unchanged = stride_result.UnchangedOrSameAs(op->stride);
  PrimExpr stride = std::move(stride_result).ValueOrUnchanged(op->stride);
  if (base_unchanged && stride_unchanged && base.ty() == stride.ty()) {
    return ffi::Unchanged();
  } else {
    PrimType base_dtype = base.ty();
    PrimType stride_dtype = stride.ty();
    TVM_FFI_ICHECK(base_dtype.MatchesCode(DLDataTypeCode::kDLInt) &&
                   stride_dtype.MatchesCode(DLDataTypeCode::kDLInt));
    int bits = std::max(base_dtype.bits(), stride_dtype.bits());
    PrimType dtype = base_dtype.WithBits(bits);
    if (base_dtype->dtype != dtype->dtype) base = cast(dtype, base);
    if (stride_dtype->dtype != dtype->dtype) stride = cast(dtype, stride);
    return prim::Ramp(base, stride, op->lanes);
  }
}

UnchangedOr<PrimExpr> DataTypeLegalizer::Mutate_(const prim::BroadcastNode* op,
                                                 InplaceMode inplace_mode) {
  auto value = Mutate(op->value, inplace_mode);
  auto lanes = Mutate(op->lanes, inplace_mode);
  if (value.UnchangedOrSameAs(op->value) && lanes.UnchangedOrSameAs(op->lanes)) {
    return ffi::Unchanged();
  }
  // Construction re-infers the dtype; the shared structural hook retains the old dtype.
  return prim::Broadcast(std::move(value).ValueOrUnchanged(op->value),
                         std::move(lanes).ValueOrUnchanged(op->lanes));
}

UnchangedOr<PrimExpr> DataTypeLegalizer::Mutate_(const prim::ShuffleNode* op,
                                                 InplaceMode inplace_mode) {
  auto vectors = Mutate(op->vectors, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  auto indices = Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  bool unchanged = vectors.UnchangedOrSameAs(op->vectors) && indices.UnchangedOrSameAs(op->indices);
  if (unchanged && inplace_mode == InplaceMode::kDisallow) return ffi::Unchanged();
  // Arrays can change in place; reconstruct to infer the dtype from the final vectors.
  PrimExpr updated = prim::Shuffle(std::move(vectors).ValueOrUnchanged(op->vectors),
                                   std::move(indices).ValueOrUnchanged(op->indices));
  if (unchanged && updated.ty() == op->ty.as_or_throw<PrimType>()) return ffi::Unchanged();
  return updated;
}

#define TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC)                                \
  UnchangedOr<PrimExpr> DataTypeLegalizer::Mutate_(const OP* op, InplaceMode inplace_mode) { \
    auto a_result = this->Mutate(op->a, inplace_mode);                                       \
    bool a_unchanged = a_result.UnchangedOrSameAs(op->a);                                    \
    PrimExpr a = std::move(a_result).ValueOrUnchanged(op->a);                                \
    auto b_result = this->Mutate(op->b, inplace_mode);                                       \
    bool b_unchanged = b_result.UnchangedOrSameAs(op->b);                                    \
    PrimExpr b = std::move(b_result).ValueOrUnchanged(op->b);                                \
    if (a_unchanged && b_unchanged && a.ty() == b.ty()) {                                    \
      return ffi::Unchanged();                                                               \
    } else {                                                                                 \
      return FUNC(a, b);                                                                     \
    }                                                                                        \
  }

TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::AddNode, operator+);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::SubNode, operator-);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::MulNode, operator*);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::DivNode, div);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::ModNode, truncmod);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::FloorDivNode, floordiv);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::FloorModNode, floormod);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::MinNode, min);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::MaxNode, max);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::EQNode, operator==);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::NENode, operator!=);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::LENode, operator<=);
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::LTNode, operator<);  // NOLINT(*)
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::GTNode, operator>);  // NOLINT(*)
TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::GENode, operator>=);

#undef TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH

UnchangedOr<Expr> DataTypeLegalizer::Mutate_(const CallNode* op, InplaceMode inplace_mode) {
  Call before = ffi::GetRef<Call>(op);
  // Keep the original argument dtypes available for shift and clz correction below.
  Expr e =
      StmtExprMutator::Mutate_(op, InplaceMode::kDisallow).ValueOrUnchanged(ffi::GetRef<Expr>(op));
  op = e.as<CallNode>();
  TVM_FFI_ICHECK(op != nullptr) << "Expected type to be CallNode"
                                << ", but get " << e->GetTypeKey();
  if (!op->ty.as<PrimTypeNode>()) {
    return e;
  }
  PrimExpr prim_e = e.as_or_throw<PrimExpr>();
  if (op->op.same_as(prim::builtin::shift_right())) {
    PrimExpr lhs = op->args[0].as_or_throw<PrimExpr>();
    PrimExpr rhs = op->args[1].as_or_throw<PrimExpr>();
    PrimType before_dtype = before->args[0].as_or_throw<PrimExpr>().ty();
    PrimType after_dtype = lhs.ty();
    if (ShouldClampShiftAmounts() && before_dtype.code() == DLDataTypeCode::kDLInt &&
        after_dtype.code() == DLDataTypeCode::kDLInt && before_dtype.bits() > after_dtype.bits()) {
      // Values are assumed to fit in the narrowed dtype.  An arithmetic right
      // shift at or beyond its sign bit therefore has the same value as a shift
      // by the new sign-bit position.  Clamp lane-wise so dynamic and vector
      // shift amounts remain valid for the narrowed dtype.
      rhs = min(rhs, MakeConst(rhs.ty(), after_dtype.bits() - 1, op->span), op->span);
    }
    return lhs >> rhs;
  } else if (op->op.same_as(prim::builtin::shift_left())) {
    PrimExpr lhs = op->args[0].as_or_throw<PrimExpr>();
    PrimExpr rhs = op->args[1].as_or_throw<PrimExpr>();
    PrimType before_dtype = before->args[0].as_or_throw<PrimExpr>().ty();
    PrimType after_dtype = lhs.ty();
    if (ShouldClampShiftAmounts() && before_dtype.code() == DLDataTypeCode::kDLInt &&
        after_dtype.code() == DLDataTypeCode::kDLInt && before_dtype.bits() > after_dtype.bits()) {
      // Keep dynamic and vector shift amounts valid for the narrowed dtype.  Under the pass's
      // representability precondition, a left shift at or beyond the narrowed width can only
      // produce a representable result when lhs is zero, so clamping does not alter valid cases.
      rhs = min(rhs, MakeConst(rhs.ty(), after_dtype.bits() - 1, op->span), op->span);
    }
    return lhs << rhs;
  } else if (op->op.same_as(prim::builtin::bitwise_and())) {
    return op->args[0].as_or_throw<PrimExpr>() & op->args[1].as_or_throw<PrimExpr>();
  } else if (op->op.same_as(prim::builtin::bitwise_or())) {
    return op->args[0].as_or_throw<PrimExpr>() | op->args[1].as_or_throw<PrimExpr>();
  } else if (op->op.same_as(prim::builtin::bitwise_xor())) {
    return op->args[0].as_or_throw<PrimExpr>() ^ op->args[1].as_or_throw<PrimExpr>();
  }
  static const Op& pow_op = Op::Get("tirx.pow");
  static const Op& clz_op = Op::Get("tirx.clz");
  if (op->op.same_as(pow_op)) {
    return pow(op->args[0].as_or_throw<PrimExpr>(), op->args[1].as_or_throw<PrimExpr>());
  } else if (op->op.same_as(prim::builtin::if_then_else())) {
    return Call(op->ty.as_or_throw<PrimType>(), op->op,
                {op->args[0].as_or_throw<PrimExpr>(), op->args[1].as_or_throw<PrimExpr>(),
                 op->args[2].as_or_throw<PrimExpr>()},
                op->attrs, {}, op->span)
        .as_or_throw<PrimExpr>();
  } else if (op->op.same_as(clz_op)) {
    PrimType before_dtype = before->args[0].as_or_throw<PrimExpr>().ty();
    PrimType after_dtype = op->args[0].as_or_throw<PrimExpr>().ty();
    TVM_FFI_ICHECK((before_dtype.code() == DLDataTypeCode::kDLInt ||
                    before_dtype.code() == DLDataTypeCode::kDLUInt) &&
                   (before_dtype.bits() == 32 || before_dtype.bits() == 64))
        << "clz only supports 32 or 64 bit integer types, but get type before legalizing: "
        << before_dtype;
    TVM_FFI_ICHECK((after_dtype.code() == DLDataTypeCode::kDLInt ||
                    after_dtype.code() == DLDataTypeCode::kDLUInt) &&
                   (after_dtype.bits() == 32 || after_dtype.bits() == 64))
        << "clz only supports 32 or 64 bit integer types, but get type after legalizing: "
        << after_dtype;
    return prim_e - after_dtype.bits() + before_dtype.bits();
  }
  return prim_e;
}

UnchangedOr<Stmt> IndexDataTypeRewriter::Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) {
  if (op->attr_key == attr::thread_extent || op->attr_key == s_tir::attr::virtual_thread) {
    bool is_enabled = is_enabled_;
    is_enabled_ = true;
    auto stmt = DataTypeLegalizer::Mutate_(op, inplace_mode);
    is_enabled_ = is_enabled;
    return stmt;
  }
  return DataTypeLegalizer::Mutate_(op, inplace_mode);
}

UnchangedOr<ffi::Any> IndexDataTypeRewriter::Mutate(ffi::AnyView value, InplaceMode inplace_mode) {
  bool is_enabled = is_enabled_;
  if (value.as<BufferTypeNode>()) is_enabled_ = true;
  auto result = DataTypeLegalizer::Mutate(value, inplace_mode);
  is_enabled_ = is_enabled;
  return result;
}

UnchangedOr<Stmt> IndexDataTypeRewriter::Mutate_(const SBlockRealizeNode* op,
                                                 InplaceMode inplace_mode) {
  bool is_condition = is_condition_;
  is_condition_ = true;
  auto new_predicate_result = Mutate(op->predicate, inplace_mode);
  bool new_predicate_unchanged = new_predicate_result.UnchangedOrSameAs(op->predicate);
  auto new_predicate = std::move(new_predicate_result).ValueOrUnchanged(op->predicate);
  is_condition_ = is_condition;

  bool is_enabled = is_enabled_;
  is_enabled_ = true;
  auto new_iter_values = Mutate(op->iter_values, inplace_mode)
                             .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                             .ValueOrUnchanged(op->iter_values);
  is_enabled_ = is_enabled;
  SBlock new_body =
      this->Mutate(op->block, inplace_mode).ValueOrUnchanged(op->block).as_or_throw<SBlock>();
  if (!new_predicate_unchanged || !new_iter_values.same_as(op->iter_values) ||
      !new_body.same_as(op->block)) {
    SBlockRealize new_block_realize = ffi::GetRef<SBlockRealize>(op);
    auto* n = new_block_realize.CopyOnWrite();
    n->predicate = std::move(new_predicate);
    n->iter_values = std::move(new_iter_values);
    n->block = std::move(new_body);
    return new_block_realize;

  } else {
    return ffi::Unchanged();
  }
}

UnchangedOr<Stmt> IndexDataTypeRewriter::Mutate_(const SBlockNode* op, InplaceMode inplace_mode) {
  auto new_alloc_buffers = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
    return Mutate(op->alloc_buffers, inplace_mode)
        .as_or_throw<UnchangedOr<ffi::Array<BufferVar>>>()
        .ValueOrUnchanged(op->alloc_buffers);
  });
  auto new_match_buffers = op->match_buffers.Map([this](const MatchBufferRegion& match) {
    BufferVar buffer = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
      return Mutate(match->buffer, InplaceMode::kDisallow)
          .as_or_throw<UnchangedOr<BufferVar>>()
          .ValueOrUnchanged(match->buffer);
    });
    BufferRegion source = VisitBufferRegion(match->source);
    if (buffer.same_as(match->buffer) && source.same_as(match->source)) return match;
    return MatchBufferRegion(buffer, source);
  });
  ffi::Array<BufferRegion> new_reads = op->reads.Map(
      [this](const BufferRegion& buffer_region) { return this->VisitBufferRegion(buffer_region); });
  ffi::Array<BufferRegion> new_writes = op->writes.Map(
      [this](const BufferRegion& buffer_region) { return this->VisitBufferRegion(buffer_region); });
  ffi::Array<IterVar> new_iter_vars =
      op->iter_vars.Map([this](const IterVar& iter_var) { return this->VisitIterVar(iter_var); });
  ffi::Optional<Stmt> new_init = std::nullopt;
  if (op->init.has_value()) {
    new_init = this->Mutate(op->init.value(), inplace_mode).ValueOrUnchanged(op->init.value());
  }
  ffi::Map<ffi::String, ffi::Any> new_annotations = VisitBlockAnnotations(op->annotations);
  auto new_body_result = this->Mutate(op->body, inplace_mode);
  bool new_body_unchanged = new_body_result.UnchangedOrSameAs(op->body);
  Stmt new_body = std::move(new_body_result).ValueOrUnchanged(op->body);

  if (!new_init.same_as(op->init) || !new_body_unchanged ||
      !new_alloc_buffers.same_as(op->alloc_buffers) ||
      !new_match_buffers.same_as(op->match_buffers) || !new_reads.same_as(op->reads) ||
      !new_writes.same_as(op->writes) || new_iter_vars.same_as(op->iter_vars) ||
      !new_annotations.same_as(op->annotations)) {
    SBlock new_block = ffi::GetRef<SBlock>(op);
    SBlockNode* n = new_block.CopyOnWrite();
    n->alloc_buffers = std::move(new_alloc_buffers);
    n->match_buffers = std::move(new_match_buffers);
    n->reads = std::move(new_reads);
    n->writes = std::move(new_writes);
    n->iter_vars = std::move(new_iter_vars);
    n->init = std::move(new_init);
    n->annotations = std::move(new_annotations);
    n->body = std::move(new_body);
    return new_block;
  }
  return ffi::Unchanged();
}

ffi::Map<ffi::String, ffi::Any> IndexDataTypeRewriter::VisitBlockAnnotations(
    const ffi::Map<ffi::String, ffi::Any>& annotations) {
  auto new_annotations = annotations;

  std::function<Any(const Any&)> f_mutate_obj = [this, &f_mutate_obj](const Any& obj) -> Any {
    if (obj == nullptr) {
      return obj;
    }
    if (auto var = obj.as<Var>(); var && var.value()->ty.as<BufferTypeNode>()) {
      BufferVar buffer(var.value());
      if (BufferVar new_buffer = Mutate(buffer, InplaceMode::kDisallow)
                                     .as_or_throw<UnchangedOr<BufferVar>>()
                                     .ValueOrUnchanged(buffer);
          !new_buffer.same_as(buffer)) {
        return new_buffer;
      }
    } else if (obj.as<ffi::ArrayObj>()) {
      return obj.as_or_throw<ffi::Array<Any>>().Map(f_mutate_obj);
    }
    return obj;
  };
  for (const auto& [key, value] : annotations) {
    if (auto opt_object_ref = value.as<ffi::ObjectRef>()) {
      auto new_value = f_mutate_obj(*opt_object_ref);
      if (!new_value.same_as(*opt_object_ref)) {
        new_annotations.Set(key, new_value);
      }
    }
  }
  return new_annotations;
}

IterVar IndexDataTypeRewriter::VisitIterVar(const IterVar& iter_var) {
  bool is_enabled = is_enabled_;
  is_enabled_ = true;
  PrimVar new_var = Mutate(iter_var->var, InplaceMode::kDisallow)
                        .ValueOrUnchanged(iter_var->var)
                        .as_or_throw<PrimVar>();
  PrimExpr min =
      Mutate(iter_var->dom->min, InplaceMode::kDisallow).ValueOrUnchanged(iter_var->dom->min);
  PrimExpr extent =
      Mutate(iter_var->dom->extent, InplaceMode::kDisallow).ValueOrUnchanged(iter_var->dom->extent);
  is_enabled_ = is_enabled;
  if (!new_var.same_as(iter_var->var) || !min.same_as(iter_var->dom->min) ||
      !extent.same_as(iter_var->dom->extent)) {
    IterVar new_iter_var = iter_var;
    IterVarNode* n = new_iter_var.CopyOnWrite();
    n->var = std::move(new_var);
    n->dom = Range(min, extent);
    return new_iter_var;
  }
  return iter_var;
}

BufferRegion IndexDataTypeRewriter::VisitBufferRegion(const BufferRegion& buffer_region) {
  BufferVar remapped_buffer = Mutate(buffer_region->buffer, InplaceMode::kDisallow)
                                  .as_or_throw<UnchangedOr<BufferVar>>()
                                  .ValueOrUnchanged(buffer_region->buffer);

  bool is_enabled = is_enabled_;
  is_enabled_ = true;
  auto new_region = buffer_region->region.Map([&](const Range& range) {
    return Range::FromMinExtent(
        this->Mutate(range->min, InplaceMode::kDisallow).ValueOrUnchanged(range->min),
        this->Mutate(range->extent, InplaceMode::kDisallow).ValueOrUnchanged(range->extent));
  });
  is_enabled_ = is_enabled;

  if (!remapped_buffer.same_as(buffer_region->buffer) ||
      !new_region.same_as(buffer_region->region)) {
    return BufferRegion(remapped_buffer, new_region);
  } else {
    return buffer_region;
  }
}

UnchangedOr<Stmt> IndexDataTypeRewriter::Mutate_(const BufferStoreNode* op,
                                                 InplaceMode inplace_mode) {
  BufferStore store = ffi::GetRef<BufferStore>(op);

  BufferVar new_buffer = Mutate(op->buffer, inplace_mode)
                             .as_or_throw<UnchangedOr<BufferVar>>()
                             .ValueOrUnchanged(op->buffer);
  auto value_result = this->Mutate(op->value, inplace_mode);
  bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
  auto value = std::move(value_result).ValueOrUnchanged(op->value);
  PrimType value_dtype = value.ty();
  if (new_buffer->dtype != value_dtype && value_dtype.IsScalar()) {
    value = cast(new_buffer->dtype, value);
    value_unchanged = false;
  }
  auto indices = VisitIndices(op->indices, inplace_mode);

  if (!new_buffer.same_as(op->buffer) || !value_unchanged || !indices.same_as(op->indices)) {
    auto writer = store.CopyOnWrite();
    writer->buffer = new_buffer;
    writer->value = value;
    writer->indices = indices;
  }

  return store;
}

UnchangedOr<PrimExpr> IndexDataTypeRewriter::Mutate_(const TensorLoadNode* op,
                                                     InplaceMode inplace_mode) {
  TensorLoad load = ffi::GetRef<TensorLoad>(op);

  BufferVar new_buffer =
      Mutate(op->source, inplace_mode).ValueOrUnchanged(op->source).as_or_throw<BufferVar>();
  auto indices = VisitIndices(op->indices, inplace_mode);

  if (!new_buffer.same_as(op->source.as_or_throw<tvm::tirx::BufferVar>()) ||
      !indices.same_as(op->indices)) {
    return BufferLoad(new_buffer, indices, op->span);
  }

  return load;
}

ffi::Array<PrimExpr> IndexDataTypeRewriter::VisitIndices(const ffi::Array<PrimExpr>& indices,
                                                         InplaceMode inplace_mode) {
  bool is_enabled = is_enabled_;
  is_enabled_ = true;
  auto result = Mutate(indices, inplace_mode)
                    .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                    .ValueOrUnchanged(indices);
  is_enabled_ = is_enabled;
  return result;
}

UnchangedOr<Stmt> IndexDataTypeRewriter::Mutate_(const IfThenElseNode* op,
                                                 InplaceMode inplace_mode) {
  bool is_condition = is_condition_;
  is_condition_ = true;
  auto cond_result = Mutate(op->condition, inplace_mode);
  bool cond_unchanged = cond_result.UnchangedOrSameAs(op->condition);
  PrimExpr cond = std::move(cond_result).ValueOrUnchanged(op->condition);
  is_condition_ = is_condition;
  auto then_case_result = Mutate(op->then_case, inplace_mode);
  bool then_case_unchanged = then_case_result.UnchangedOrSameAs(op->then_case);
  Stmt then_case = std::move(then_case_result).ValueOrUnchanged(op->then_case);
  ffi::Optional<Stmt> else_case =
      op->else_case.has_value() ? ffi::Optional<Stmt>{Mutate(op->else_case.value(), inplace_mode)
                                                          .ValueOrUnchanged(op->else_case.value())}
                                : std::nullopt;
  if (!cond_unchanged || !then_case_unchanged || !else_case.same_as(op->else_case)) {
    IfThenElse new_stmt = ffi::GetRef<IfThenElse>(op);
    auto* n = new_stmt.CopyOnWrite();
    n->condition = std::move(cond);
    n->then_case = std::move(then_case);
    n->else_case = std::move(else_case);
    return new_stmt;
  }
  return ffi::Unchanged();
}

UnchangedOr<Stmt> IndexDataTypeRewriter::Mutate_(const ForNode* op, InplaceMode inplace_mode) {
  bool is_enabled = is_enabled_;
  is_enabled_ = true;
  PrimVar new_loop_var =
      Mutate(op->loop_var, inplace_mode).ValueOrUnchanged(op->loop_var).as_or_throw<PrimVar>();
  auto min_result = Mutate(op->min, inplace_mode);
  bool min_unchanged = min_result.UnchangedOrSameAs(op->min);
  PrimExpr min = std::move(min_result).ValueOrUnchanged(op->min);
  auto extent_result = Mutate(op->extent, inplace_mode);
  bool extent_unchanged = extent_result.UnchangedOrSameAs(op->extent);
  PrimExpr extent = std::move(extent_result).ValueOrUnchanged(op->extent);
  is_enabled_ = is_enabled;
  auto new_body_result = Mutate(op->body, inplace_mode);
  bool new_body_unchanged = new_body_result.UnchangedOrSameAs(op->body);
  Stmt new_body = std::move(new_body_result).ValueOrUnchanged(op->body);

  if (!new_loop_var.same_as(op->loop_var) || !min_unchanged || !extent_unchanged ||
      !new_body_unchanged) {
    For new_for = ffi::GetRef<For>(op);
    auto* n = new_for.CopyOnWrite();
    n->loop_var = new_loop_var;
    n->min = cast(new_loop_var.ty(), min);
    n->extent = cast(new_loop_var.ty(), extent);
    if (op->thread_binding.has_value()) {
      auto old_thread_binding = op->thread_binding.value();
      auto* ptr = old_thread_binding.CopyOnWrite();
      ptr->var = old_thread_binding->var.CopyWithDType(new_loop_var.ty());
      n->thread_binding = ffi::Optional<IterVar>(std::move(old_thread_binding));
    }
    n->body = new_body;
    return new_for;

  } else {
    return ffi::Unchanged();
  }
}

UnchangedOr<Stmt> IndexDataTypeRewriter::Mutate_(const BindNode* op, InplaceMode inplace_mode) {
  auto mapped = VarRemapGet(op->var);
  if (mapped == nullptr || mapped.type_index() == ffi::TypeIndex::kTVMFFIUnchanged) {
    return DataTypeLegalizer::Mutate_(op, inplace_mode);
  }
  Var var = mapped.as_or_throw<Var>();
  bool is_enabled = is_enabled_;
  is_enabled_ = true;
  PrimExpr value =
      Mutate(op->value, inplace_mode).ValueOrUnchanged(op->value).as_or_throw<PrimExpr>();
  is_enabled_ = is_enabled;
  // The collected index requirement need not apply to every variable in the RHS.
  return Bind(var, cast(var->ty.as_or_throw<PrimType>(), value), op->span);
}

#define TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC)                                   \
  UnchangedOr<PrimExpr> IndexDataTypeRewriter::Mutate_(const OP* op, InplaceMode inplace_mode) { \
    bool is_enabled = is_enabled_;                                                               \
    is_enabled_ = is_condition_ && op->a.ty().MatchesCode(DLDataTypeCode::kDLInt) &&             \
                  op->b.ty().MatchesCode(DLDataTypeCode::kDLInt);                                \
    auto result = Parent::Mutate_(op, inplace_mode);                                             \
    is_enabled_ = is_enabled;                                                                    \
    return result;                                                                               \
  }

TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::EQNode, operator==);
TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::NENode, operator!=);
TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::LENode, operator<=);
TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::LTNode, operator<);  // NOLINT(*)
TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::GTNode, operator>);  // NOLINT(*)
TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::GENode, operator>=);

UnchangedOr<Expr> IndexDataTypeRewriter::Mutate_(const CallNode* op, InplaceMode inplace_mode) {
  // handle if_then_else condition
  if (op->op.same_as(prim::builtin::if_then_else())) {
    bool is_condition = is_condition_;
    is_condition_ = true;
    PrimExpr cond = Mutate(op->args[0]).ValueOrUnchanged(op->args[0]).as_or_throw<PrimExpr>();
    is_condition_ = is_condition;
    PrimExpr true_value = Mutate(op->args[1]).ValueOrUnchanged(op->args[1]).as_or_throw<PrimExpr>();
    PrimExpr false_value =
        Mutate(op->args[2]).ValueOrUnchanged(op->args[2]).as_or_throw<PrimExpr>();
    PrimType true_dtype = true_value.ty();
    PrimType false_dtype = false_value.ty();
    PrimType dtype = true_dtype.WithBits(std::max(true_dtype.bits(), false_dtype.bits()));
    if (true_dtype != dtype) true_value = cast(dtype, true_value);
    if (false_dtype != dtype) false_value = cast(dtype, false_value);
    return Call(dtype, op->op, {cond, true_value, false_value}, op->attrs, {}, op->span)
        .as_or_throw<PrimExpr>();
  }
  return Parent::Mutate_(op, inplace_mode);
}

UnchangedOr<PrimExpr> IndexDataTypeRewriter::Mutate_(const prim::SelectNode* op,
                                                     InplaceMode inplace_mode) {
  bool is_condition = true;
  std::swap(is_condition_, is_condition);
  auto condition_result = this->Mutate(op->condition, inplace_mode);
  bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
  PrimExpr condition = std::move(condition_result).ValueOrUnchanged(op->condition);
  std::swap(is_condition_, is_condition);
  auto true_value_result = this->Mutate(op->true_value, inplace_mode);
  bool true_value_unchanged = true_value_result.UnchangedOrSameAs(op->true_value);
  PrimExpr true_value = std::move(true_value_result).ValueOrUnchanged(op->true_value);
  auto false_value_result = this->Mutate(op->false_value, inplace_mode);
  bool false_value_unchanged = false_value_result.UnchangedOrSameAs(op->false_value);
  PrimExpr false_value = std::move(false_value_result).ValueOrUnchanged(op->false_value);

  if (condition_unchanged && true_value_unchanged && false_value_unchanged &&
      true_value.ty() == false_value.ty()) {
    return ffi::Unchanged();
  } else {
    PrimType true_dtype = true_value.ty();
    PrimType false_dtype = false_value.ty();
    int bits = std::max(true_dtype.bits(), false_dtype.bits());
    PrimType dtype = true_dtype.WithBits(bits);
    if (true_dtype->dtype != dtype->dtype) true_value = cast(dtype, true_value);
    if (false_dtype->dtype != dtype->dtype) false_value = cast(dtype, false_value);
    return prim::Select(condition, true_value, false_value);
  }
}

#undef TVM_DEFINE_CMPOP_EXPR_MUTATE_WITH_TYPE_MATCH

IndexDataTypeNormalizer::IndexDataTypeNormalizer(PrimType target_data_type)
    : target_data_type_(std::move(target_data_type)) {}

PrimFunc IndexDataTypeNormalizer::Rewrite(PrimFunc func) {
  // Collect scalar dtype requirements without changing types.  Buffer definitions
  // are rewritten only after every scalar replacement has been seeded.
  class IndexVarCollector : public IndexDataTypeRewriter {
   public:
    explicit IndexVarCollector(std::function<void(const VarNode*)> collect)
        : collect_(std::move(collect)) {}
    using IndexDataTypeRewriter::Mutate;
    using IndexDataTypeRewriter::Mutate_;
    UnchangedOr<Expr> Mutate_(const VarNode* op, InplaceMode mode) final {
      if (def_region_kind() == kTVMFFIDefRegionKindNone && is_enabled_) collect_(op);
      return IndexDataTypeRewriter::Mutate_(op, mode);
    }

   private:
    std::function<void(const VarNode*)> collect_;
  };
  auto seed = [this](const VarNode* var) {
    auto dtype = var->ty.as<PrimType>();
    if (dtype && CanRewriteDType(dtype.value()) && dtype.value() != target_data_type_ &&
        VarRemapGet(ffi::AnyView(var)) == nullptr) {
      VarRemapSet(ffi::AnyView(var), ffi::GetRef<Var>(var).CopyWithDType(target_data_type_));
    }
  };
  auto collector = ffi::make_object<IndexVarCollector>(seed);
  collector->Mutate(func->body);
  for (const Var& param : func->params) {
    if (param.as<BufferVar>()) {
      collector->WithDefRegionKind(kTVMFFIDefRegionKindSimple,
                                   [&] { return collector->Mutate(param); });
    } else {
      seed(param.get());
    }
  }
  ffi::Array<Var> params = func->params.Map([this](const Var& param) {
    return WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
      return Mutate(param).ValueOrUnchanged(param).as_or_throw<Var>();
    });
  });
  PrimFuncNode* new_func = func.CopyOnWrite();
  new_func->params = std::move(params);
  new_func->body = Mutate(new_func->body).ValueOrUnchanged(new_func->body);
  return func;
}

bool IndexDataTypeNormalizer::CanRewriteDType(PrimType dtype) const {
  return dtype.code() == DLDataTypeCode::kDLInt && dtype.bits() >= 32;
}

UnchangedOr<PrimExpr> IndexDataTypeNormalizer::Mutate_(const IntImmNode* op,
                                                       InplaceMode inplace_mode) {
  if (is_enabled_ && CanRewriteDType(op->ty.as_or_throw<PrimType>())) {
    TVM_FFI_ICHECK_LE(op->value, max_value(target_data_type_).as_or_throw<IntImm>()->value);
    return cast(target_data_type_, ffi::GetRef<IntImm>(op));
  }
  return ffi::Unchanged();
}

UnchangedOr<PrimExpr> IndexDataTypeNormalizer::Mutate_(const prim::CastNode* op,
                                                       InplaceMode inplace_mode) {
  // Unwrap the cast only when the dtype of this cast is integer dtype.
  // When the dtype of this cast is not integer dtype, it means that this cast
  // has some other purpose, and we should not unwrap the cast.
  PrimType dtype = op->ty.as_or_throw<PrimType>();
  if (is_enabled_ && CanRewriteDType(dtype)) {
    PrimExpr value = this->Mutate(op->value, inplace_mode).ValueOrUnchanged(op->value);
    return value.ty() == target_data_type_ ? value : prim::Cast(target_data_type_, value);
  }
  return IndexDataTypeRewriter::Mutate_(op, inplace_mode);
}

}  // namespace tirx
}  // namespace tvm
