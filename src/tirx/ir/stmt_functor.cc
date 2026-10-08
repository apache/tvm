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
 * \file stmt_functor.cc
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "data_type_rewriter.h"
#include "seq_stmt_mutate.h"

namespace tvm {
namespace tirx {

void StmtExprVisitor::InitVTable(VTable* vtable) {
  tvm::ExprVisitor::InitVTable(vtable);
  SetDispatch<StmtExprVisitor, BindNode>(vtable);
  SetDispatch<StmtExprVisitor, RegionStmtNode>(vtable);
  SetDispatch<StmtExprVisitor, IfThenElseNode>(vtable);
  SetDispatch<StmtExprVisitor, ForNode>(vtable);
  SetDispatch<StmtExprVisitor, WhileNode>(vtable);
  SetDispatch<StmtExprVisitor, ReturnNode>(vtable);
  SetDispatch<StmtExprVisitor, BreakNode>(vtable);
  SetDispatch<StmtExprVisitor, ContinueNode>(vtable);
  SetDispatch<StmtExprVisitor, TensorStoreNode>(vtable);
  SetDispatch<StmtExprVisitor, AssertStmtNode>(vtable);
  SetDispatch<StmtExprVisitor, SeqStmtNode>(vtable);
  SetDispatch<StmtExprVisitor, EvaluateNode>(vtable);
  SetDispatch<StmtExprVisitor, ScopeIdDefStmtNode>(vtable);
  SetDispatch<StmtExprVisitor, TilePrimitiveCallNode>(vtable);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const VarNode* op) {
  // Types belong to definitions; uses retain the identity introduced in that region.
  if (def_region_kind() == kTVMFFIDefRegionKindNone) return std::nullopt;
  return ExprVisitor::Visit_(op);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const OpaqueExprNode* op) {
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const BreakNode* op) { return std::nullopt; }

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const ContinueNode* op) {
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const TensorLoadNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->source));
  for (const auto& child : op->indices) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const TupleNode* op) {
  for (const auto& child : op->fields) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const TupleGetItemNode* op) {
  return this->Visit(op->tuple);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const prim::LetNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->value));
  return this->Visit(op->body);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const CallNode* op) {
  if (op->op.as<OpaqueExprNode>()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->op));
  }
  for (const auto& child : op->args) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const prim::RampNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->base));
  return this->Visit(op->stride);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const prim::BroadcastNode* op) {
  return this->Visit(op->value);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const prim::ShuffleNode* op) {
  for (const auto& child : op->indices) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  for (const auto& child : op->vectors) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const BindNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->value));
  return this->WithDefRegionKind(kTVMFFIDefRegionKindSimple,
                                 [&]() { return this->Visit(op->var); });
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const RegionStmtNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->attrs));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->args));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return this->Visit(op->body_params); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->body));
  return this->WithDefRegionKind(kTVMFFIDefRegionKindSimple,
                                 [&]() { return this->Visit(op->result_vars); });
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const ForNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->min));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->extent));
  if (op->step.has_value()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(*op->step));
  }
  return this->Visit(op->body);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const WhileNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->condition));
  return this->Visit(op->body);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const ReturnNode* op) {
  return this->Visit(op->value);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const TensorStoreNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->buffer));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->value));
  for (const auto& child : op->indices) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const IfThenElseNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->then_case));
  if (op->else_case) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->else_case.value()));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const AssertStmtNode* op) {
  // Constant message_parts are intentionally skipped.
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->condition));
  return this->Visit(op->error_kind);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const SeqStmtNode* op) {
  for (const auto& child : op->seq) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const EvaluateNode* op) {
  return this->Visit(op->value);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const ScopeIdDefStmtNode* op) {
  // Flat stmt -- no body. Visit extents (skip deferred defs whose extents
  // are NullOpt) and any preferred_extents.
  if (op->def->extents.has_value()) {
    for (const auto& child : op->def->extents.value()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
    }
  }
  if (op->def->preferred_extents.has_value()) {
    for (const auto& child : op->def->preferred_extents.value()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
    }
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const TilePrimitiveCallNode* op) {
  for (const Expr& arg : op->args) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(arg));
  }
  for (const auto& [key, value] : op->config) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(value));
  }
  return std::nullopt;
}

void StmtExprMutator::InitVTable(VTable* vtable) {
  tvm::ExprMutator::InitVTable(vtable);
  SetDispatch<StmtExprMutator, BindNode>(vtable);
  SetDispatch<StmtExprMutator, RegionStmtNode>(vtable);
  SetDispatch<StmtExprMutator, IfThenElseNode>(vtable);
  SetDispatch<StmtExprMutator, ForNode>(vtable);
  SetDispatch<StmtExprMutator, WhileNode>(vtable);
  SetDispatch<StmtExprMutator, ReturnNode>(vtable);
  SetDispatch<StmtExprMutator, BreakNode>(vtable);
  SetDispatch<StmtExprMutator, ContinueNode>(vtable);
  SetDispatch<StmtExprMutator, TensorStoreNode>(vtable);
  SetDispatch<StmtExprMutator, AssertStmtNode>(vtable);
  SetDispatch<StmtExprMutator, SeqStmtNode>(vtable);
  SetDispatch<StmtExprMutator, EvaluateNode>(vtable);
  SetDispatch<StmtExprMutator, ScopeIdDefStmtNode>(vtable);
  SetDispatch<StmtExprMutator, TilePrimitiveCallNode>(vtable);
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const BindNode* op, InplaceMode inplace_mode) {
  Type old_value_type = op->value->ty;
  auto value_u = Mutate(op->value, inplace_mode);
  bool value_changed = !value_u.UnchangedOrSameAs(op->value);
  Expr value = std::move(value_u).ValueOrUnchanged(op->value);
  value_changed |= !old_value_type.same_as(value->ty);
  Type old_var_type = op->var->ty;
  auto var_u = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
                 return Mutate(op->var, inplace_mode);
               }).as_or_throw<UnchangedOr<Var>>();
  bool var_changed = !var_u.UnchangedOrSameAs(op->var);
  Var var = std::move(var_u).ValueOrUnchanged(op->var);
  var_changed |= !old_var_type.same_as(var->ty);
  if ((value_changed || var_changed) && !var->ty.same_as(value->ty)) {
    var = var.CopyWithType(value->ty);
  }
  if (!var.same_as(op->var)) VarRemapSet(op->var, var);
  if (value.same_as(op->value) && var.same_as(op->var)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<BindNode*>(op);
    writable->value = std::move(value);
    writable->var = std::move(var);
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<BindNode>(*op);
  copy->value = std::move(value);
  copy->var = std::move(var);
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) {
  auto attrs = Mutate(op->attrs, inplace_mode).as_or_throw<UnchangedOr<DictAttrs>>();
  auto args = Mutate(op->args, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Expr>>>();
  auto body_params = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
    return Mutate(op->body_params, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Var>>>();
  });
  auto body = Mutate(op->body, inplace_mode);
  auto result_vars = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
    return Mutate(op->result_vars, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Var>>>();
  });
  if (attrs.UnchangedOrSameAs(op->attrs) && args.UnchangedOrSameAs(op->args) &&
      body_params.UnchangedOrSameAs(op->body_params) && body.UnchangedOrSameAs(op->body) &&
      result_vars.UnchangedOrSameAs(op->result_vars)) {
    return ffi::Unchanged();
  }
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<RegionStmtNode*>(op);
    if (!attrs.IsUnchanged()) writable->attrs = std::move(attrs).ValueUnchecked();
    if (!args.IsUnchanged()) writable->args = std::move(args).ValueUnchecked();
    if (!body_params.IsUnchanged()) writable->body_params = std::move(body_params).ValueUnchecked();
    if (!body.IsUnchanged()) writable->body = std::move(body).ValueUnchecked();
    if (!result_vars.IsUnchanged()) writable->result_vars = std::move(result_vars).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<RegionStmtNode>(*op);
  if (!attrs.IsUnchanged()) copy->attrs = std::move(attrs).ValueUnchecked();
  if (!args.IsUnchanged()) copy->args = std::move(args).ValueUnchecked();
  if (!body_params.IsUnchanged()) copy->body_params = std::move(body_params).ValueUnchecked();
  if (!body.IsUnchanged()) copy->body = std::move(body).ValueUnchecked();
  if (!result_vars.IsUnchanged()) copy->result_vars = std::move(result_vars).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const ForNode* op, InplaceMode inplace_mode) {
  auto min = Mutate(op->min, inplace_mode);
  auto extent = Mutate(op->extent, inplace_mode);
  auto step = Mutate(op->step, inplace_mode).as_or_throw<UnchangedOr<ffi::Optional<PrimExpr>>>();
  auto body = Mutate(op->body, inplace_mode);
  if (min.UnchangedOrSameAs(op->min) && extent.UnchangedOrSameAs(op->extent) &&
      step.UnchangedOrSameAs(op->step) && body.UnchangedOrSameAs(op->body))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<ForNode*>(op);
    if (!min.IsUnchanged()) writable->min = std::move(min).ValueUnchecked();
    if (!extent.IsUnchanged()) writable->extent = std::move(extent).ValueUnchecked();
    if (!step.IsUnchanged()) writable->step = std::move(step).ValueUnchecked();
    if (!body.IsUnchanged()) writable->body = std::move(body).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<ForNode>(*op);
  if (!min.IsUnchanged()) copy->min = std::move(min).ValueUnchecked();
  if (!extent.IsUnchanged()) copy->extent = std::move(extent).ValueUnchecked();
  if (!step.IsUnchanged()) copy->step = std::move(step).ValueUnchecked();
  if (!body.IsUnchanged()) copy->body = std::move(body).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const WhileNode* op, InplaceMode inplace_mode) {
  auto condition = Mutate(op->condition, inplace_mode);
  auto body = Mutate(op->body, inplace_mode);
  if (condition.UnchangedOrSameAs(op->condition) && body.UnchangedOrSameAs(op->body))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<WhileNode*>(op);
    if (!condition.IsUnchanged()) writable->condition = std::move(condition).ValueUnchecked();
    if (!body.IsUnchanged()) writable->body = std::move(body).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<WhileNode>(*op);
  if (!condition.IsUnchanged()) copy->condition = std::move(condition).ValueUnchecked();
  if (!body.IsUnchanged()) copy->body = std::move(body).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const ReturnNode* op, InplaceMode inplace_mode) {
  auto value = Mutate(op->value, inplace_mode);
  if (value.UnchangedOrSameAs(op->value)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<ReturnNode*>(op);
    if (!value.IsUnchanged()) writable->value = std::move(value).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<ReturnNode>(*op);
  if (!value.IsUnchanged()) copy->value = std::move(value).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const IfThenElseNode* op, InplaceMode inplace_mode) {
  auto condition = Mutate(op->condition, inplace_mode);
  auto then_case = Mutate(op->then_case, inplace_mode);
  auto else_case = Mutate(op->else_case, inplace_mode);
  if (condition.UnchangedOrSameAs(op->condition) && then_case.UnchangedOrSameAs(op->then_case) &&
      else_case.UnchangedOrSameAs(op->else_case))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<IfThenElseNode*>(op);
    if (!condition.IsUnchanged()) writable->condition = std::move(condition).ValueUnchecked();
    if (!then_case.IsUnchanged()) writable->then_case = std::move(then_case).ValueUnchecked();
    if (!else_case.IsUnchanged()) writable->else_case = std::move(else_case).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<IfThenElseNode>(*op);
  if (!condition.IsUnchanged()) copy->condition = std::move(condition).ValueUnchecked();
  if (!then_case.IsUnchanged()) copy->then_case = std::move(then_case).ValueUnchecked();
  if (!else_case.IsUnchanged()) copy->else_case = std::move(else_case).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const AssertStmtNode* op, InplaceMode inplace_mode) {
  auto condition = Mutate(op->condition, inplace_mode);
  auto error_kind = Mutate(op->error_kind, inplace_mode).as_or_throw<UnchangedOr<StringImm>>();
  if (condition.UnchangedOrSameAs(op->condition) && error_kind.UnchangedOrSameAs(op->error_kind))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<AssertStmtNode*>(op);
    if (!condition.IsUnchanged()) writable->condition = std::move(condition).ValueUnchecked();
    if (!error_kind.IsUnchanged()) writable->error_kind = std::move(error_kind).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<AssertStmtNode>(*op);
  if (!condition.IsUnchanged()) copy->condition = std::move(condition).ValueUnchecked();
  if (!error_kind.IsUnchanged()) copy->error_kind = std::move(error_kind).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const EvaluateNode* op, InplaceMode inplace_mode) {
  auto value = Mutate(op->value, inplace_mode);
  if (value.UnchangedOrSameAs(op->value)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<EvaluateNode*>(op);
    if (!value.IsUnchanged()) writable->value = std::move(value).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<EvaluateNode>(*op);
  if (!value.IsUnchanged()) copy->value = std::move(value).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const BreakNode* op, InplaceMode inplace_mode) {
  return ffi::Unchanged();
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const ContinueNode* op, InplaceMode inplace_mode) {
  return ffi::Unchanged();
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const TensorStoreNode* op, InplaceMode inplace_mode) {
  auto buffer = Mutate(op->buffer, inplace_mode).as_or_throw<UnchangedOr<TensorVar>>();
  auto value = Mutate(op->value, inplace_mode);
  auto indices = Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  if (buffer.UnchangedOrSameAs(op->buffer) && value.UnchangedOrSameAs(op->value) &&
      indices.UnchangedOrSameAs(op->indices))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<TensorStoreNode*>(op);
    if (!buffer.IsUnchanged()) writable->buffer = std::move(buffer).ValueUnchecked();
    if (!value.IsUnchanged()) writable->value = std::move(value).ValueUnchecked();
    if (!indices.IsUnchanged()) writable->indices = std::move(indices).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<TensorStoreNode>(*op);
  if (!buffer.IsUnchanged()) copy->buffer = std::move(buffer).ValueUnchecked();
  if (!value.IsUnchanged()) copy->value = std::move(value).ValueUnchecked();
  if (!indices.IsUnchanged()) copy->indices = std::move(indices).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const SeqStmtNode* op, InplaceMode inplace_mode) {
  return detail::MutateSeqStmt(op, inplace_mode, [this](ffi::AnyView element, InplaceMode mode) {
    return Mutate(element, mode).as_or_throw<UnchangedOr<Stmt>>();
  });
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const ScopeIdDefStmtNode* op, InplaceMode inplace_mode) {
  // The definition owns both optional arrays; it is skipped by this semantic hook.
  InplaceMode def_mode = op->def.unique() ? inplace_mode : InplaceMode::kDisallow;
  auto extents = Mutate(op->def->extents, def_mode)
                     .as_or_throw<UnchangedOr<ffi::Optional<ffi::Array<PrimExpr>>>>();
  auto preferred = Mutate(op->def->preferred_extents, def_mode)
                       .as_or_throw<UnchangedOr<ffi::Optional<ffi::Array<PrimExpr>>>>();
  if (extents.UnchangedOrSameAs(op->def->extents) &&
      preferred.UnchangedOrSameAs(op->def->preferred_extents))
    return ffi::Unchanged();
  ScopeIdDef def(op->def->def_ids, std::move(extents).ValueOrUnchanged(op->def->extents),
                 op->def->scope, std::move(preferred).ValueOrUnchanged(op->def->preferred_extents));
  if (inplace_mode == InplaceMode::kAllow) {
    const_cast<ScopeIdDefStmtNode*>(op)->def = std::move(def);
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<ScopeIdDefStmtNode>(*op);
  copy->def = std::move(def);
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const TilePrimitiveCallNode* op,
                                           InplaceMode inplace_mode) {
  auto args = Mutate(op->args, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Expr>>>();
  auto config =
      Mutate(op->config, inplace_mode).as_or_throw<UnchangedOr<ffi::Map<ffi::String, Expr>>>();
  if (args.UnchangedOrSameAs(op->args) && config.UnchangedOrSameAs(op->config)) {
    return ffi::Unchanged();
  }
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<TilePrimitiveCallNode*>(op);
    if (!args.IsUnchanged()) writable->args = std::move(args).ValueUnchecked();
    if (!config.IsUnchanged()) writable->config = std::move(config).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<TilePrimitiveCallNode>(*op);
  if (!args.IsUnchanged()) copy->args = std::move(args).ValueUnchecked();
  if (!config.IsUnchanged()) copy->config = std::move(config).ValueUnchecked();
  return Stmt(std::move(copy));
}

class IRSubstituteWithDataTypeLegalization : public DataTypeLegalizer {
 public:
  IRSubstituteWithDataTypeLegalization(ffi::AnyView root,
                                       const std::function<ffi::Optional<Expr>(const Var&)>& vmap) {
    ffi::Map<Var, bool> visited;
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        root, [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
          if (visited.count(var)) return ffi::WalkResult::Skip();
          visited.Set(var, true);
          if (auto replacement = vmap(var)) VarRemapSet(var, replacement.value());
          return ffi::WalkResult::Advance();
        });
  }

  using DataTypeLegalizer::Mutate;
  using DataTypeLegalizer::Mutate_;

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    auto result = StmtExprMutator::Mutate_(op, InplaceMode::kDisallow);
    if (result.UnchangedOrSameAs(ffi::GetRef<PrimExpr>(op))) return ffi::Unchanged();
    auto load = std::move(result).ValueUnchecked().as_or_throw<TensorLoad>();
    if (auto buffer = load->source.as<TensorVar>()) {
      return MakeTensorLoad(buffer.value(), load->indices, load->span);
    }
    return load;
  }
};

Stmt SubstituteWithDataTypeLegalization(Stmt stmt,
                                        std::function<ffi::Optional<PrimExpr>(const Var&)> vmap) {
  auto general_vmap = [vmap = std::move(vmap)](const Var& var) -> ffi::Optional<Expr> {
    if (auto replacement = vmap(var)) return Expr(replacement.value());
    return std::nullopt;
  };
  return ffi::make_object<IRSubstituteWithDataTypeLegalization>(stmt, general_vmap)
      ->Mutate(stmt, InplaceMode::kAllow)
      .ValueOrUnchanged(std::move(stmt));
}

PrimExpr SubstituteWithDataTypeLegalization(
    PrimExpr expr, std::function<ffi::Optional<PrimExpr>(const Var&)> vmap) {
  auto general_vmap = [vmap = std::move(vmap)](const Var& var) -> ffi::Optional<Expr> {
    if (auto replacement = vmap(var)) return Expr(replacement.value());
    return std::nullopt;
  };
  return ffi::make_object<IRSubstituteWithDataTypeLegalization>(expr, general_vmap)
      ->Mutate(expr, InplaceMode::kAllow)
      .ValueOrUnchanged(std::move(expr));
}

}  // namespace tirx
}  // namespace tvm
