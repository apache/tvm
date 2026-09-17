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
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/stmt_functor.h>

#include <cstdint>
#include <functional>
#include <utility>

#include "data_type_rewriter.h"
#include "seq_stmt_mutate.h"

namespace tvm {
namespace tirx {

void StmtExprVisitor::InitVTable(VTable* vtable) {
  tvm::ExprVisitor::InitVTable(vtable);
  SetDispatch<StmtExprVisitor, BindNode>(vtable);
  SetDispatch<StmtExprVisitor, AttrStmtNode>(vtable);
  SetDispatch<StmtExprVisitor, IfThenElseNode>(vtable);
  SetDispatch<StmtExprVisitor, ForNode>(vtable);
  SetDispatch<StmtExprVisitor, WhileNode>(vtable);
  SetDispatch<StmtExprVisitor, ReturnNode>(vtable);
  SetDispatch<StmtExprVisitor, BreakNode>(vtable);
  SetDispatch<StmtExprVisitor, ContinueNode>(vtable);
  SetDispatch<StmtExprVisitor, AllocBufferNode>(vtable);
  SetDispatch<StmtExprVisitor, DeclBufferNode>(vtable);
  SetDispatch<StmtExprVisitor, BufferStoreNode>(vtable);
  SetDispatch<StmtExprVisitor, AssertStmtNode>(vtable);
  SetDispatch<StmtExprVisitor, SeqStmtNode>(vtable);
  SetDispatch<StmtExprVisitor, EvaluateNode>(vtable);
  SetDispatch<StmtExprVisitor, SBlockNode>(vtable);
  SetDispatch<StmtExprVisitor, SBlockRealizeNode>(vtable);
  SetDispatch<StmtExprVisitor, ScopeIdDefStmtNode>(vtable);
  SetDispatch<StmtExprVisitor, TilePrimitiveCallNode>(vtable);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const VarNode* op) { return std::nullopt; }

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
  // Bind has no body -- only visit the value expression.
  return this->Visit(op->value);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const AttrStmtNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->value));
  return this->Visit(op->body);
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

ffi::Optional<VisitInterrupt> StmtExprVisitor::VisitBufferMetadata(const BufferVar& buffer) {
  for (const auto& child : buffer->shape) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  for (const auto& child : buffer->strides) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(buffer->elem_offset));
  for (const auto& child : buffer->allocated_addr) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  if (buffer->layout.has_value()) {
    const auto* layout = buffer->layout.value().as<TileLayoutNode>();
    if (layout == nullptr) return std::nullopt;
    for (const Iter& iter : layout->shard) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(iter->extent));
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(iter->stride));
    }
    for (const Iter& iter : layout->replica) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(iter->extent));
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(iter->stride));
    }
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const AllocBufferNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return this->Visit(op->buffer); }));
  return VisitBufferMetadata(op->buffer);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const DeclBufferNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->data));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return this->Visit(op->buffer); }));
  return VisitBufferMetadata(op->buffer);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const BufferStoreNode* op) {
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

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const SBlockNode* op) {
  for (const IterVar& iter_var : op->iter_vars) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(iter_var->dom->min));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(iter_var->dom->extent));
  }
  for (const BufferVar& buf : op->alloc_buffers) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(
        this->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() { return this->Visit(buf); }));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(VisitBufferMetadata(buf));
  }
  for (const TensorRegion& region : op->reads) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(region));
  }
  for (const TensorRegion& region : op->writes) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(region));
  }
  for (const MatchBufferRegion& match_buffer_region : op->match_buffers) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->WithDefRegionKind(
        kTVMFFIDefRegionKindSimple, [&]() { return this->Visit(match_buffer_region->buffer); }));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(VisitBufferMetadata(match_buffer_region->buffer));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(match_buffer_region->source));
  }
  if (op->init.has_value()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->init.value()));
  }
  return this->Visit(op->body);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const SBlockRealizeNode* op) {
  for (const auto& child : op->iter_values) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->predicate));
  return this->Visit(op->block);
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
  std::function<ffi::Optional<VisitInterrupt>(const ffi::Any&)> fvisit;
  fvisit = [this, &fvisit](const ffi::Any& e) -> ffi::Optional<VisitInterrupt> {
    if (e == nullptr) return std::nullopt;
    if (auto buffer_region = e.as<TensorRegion>()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(buffer_region.value()));
    } else if (auto var = e.as<Var>(); var && var.value()->ty.as<BufferTypeNode>()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(BufferVar(var.value())));
    } else if (auto expr = e.as<PrimExpr>()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(expr.value()));
    } else if (auto stmt = e.as<Stmt>()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(stmt.value()));
    } else if (auto array = e.as<ffi::Array<ffi::Any>>()) {
      for (const ffi::Any& item : array.value()) {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(fvisit(item));
      }
    }
    return std::nullopt;
  };
  for (const ffi::Any& arg : op->args) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(fvisit(arg));
  }
  for (const auto& [key, value] : op->config) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(fvisit(value));
  }
  return std::nullopt;
}

void StmtExprMutator::InitVTable(VTable* vtable) {
  tvm::ExprMutator::InitVTable(vtable);
  SetDispatch<StmtExprMutator, BindNode>(vtable);
  SetDispatch<StmtExprMutator, AttrStmtNode>(vtable);
  SetDispatch<StmtExprMutator, IfThenElseNode>(vtable);
  SetDispatch<StmtExprMutator, ForNode>(vtable);
  SetDispatch<StmtExprMutator, WhileNode>(vtable);
  SetDispatch<StmtExprMutator, ReturnNode>(vtable);
  SetDispatch<StmtExprMutator, BreakNode>(vtable);
  SetDispatch<StmtExprMutator, ContinueNode>(vtable);
  SetDispatch<StmtExprMutator, AllocBufferNode>(vtable);
  SetDispatch<StmtExprMutator, DeclBufferNode>(vtable);
  SetDispatch<StmtExprMutator, BufferStoreNode>(vtable);
  SetDispatch<StmtExprMutator, AssertStmtNode>(vtable);
  SetDispatch<StmtExprMutator, SeqStmtNode>(vtable);
  SetDispatch<StmtExprMutator, EvaluateNode>(vtable);
  SetDispatch<StmtExprMutator, SBlockNode>(vtable);
  SetDispatch<StmtExprMutator, SBlockRealizeNode>(vtable);
  SetDispatch<StmtExprMutator, ScopeIdDefStmtNode>(vtable);
  SetDispatch<StmtExprMutator, TilePrimitiveCallNode>(vtable);
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const BindNode* op, InplaceMode inplace_mode) {
  auto value = Mutate(op->value, inplace_mode);
  if (value.UnchangedOrSameAs(op->value)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<BindNode*>(op);
    if (!value.IsUnchanged()) writable->value = std::move(value).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<BindNode>(*op);
  if (!value.IsUnchanged()) copy->value = std::move(value).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) {
  auto value = Mutate(op->value, inplace_mode);
  auto body = Mutate(op->body, inplace_mode);
  if (value.UnchangedOrSameAs(op->value) && body.UnchangedOrSameAs(op->body))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<AttrStmtNode*>(op);
    if (!value.IsUnchanged()) writable->value = std::move(value).ValueUnchecked();
    if (!body.IsUnchanged()) writable->body = std::move(body).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<AttrStmtNode>(*op);
  if (!value.IsUnchanged()) copy->value = std::move(value).ValueUnchecked();
  if (!body.IsUnchanged()) copy->body = std::move(body).ValueUnchecked();
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
  auto else_case =
      Mutate(op->else_case, inplace_mode).as_or_throw<UnchangedOr<ffi::Optional<Stmt>>>();
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
  auto error_kind =
      Mutate(op->error_kind, inplace_mode).as_or_throw<UnchangedOr<prim::StringImm>>();
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

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const SBlockRealizeNode* op, InplaceMode inplace_mode) {
  auto iter_values =
      Mutate(op->iter_values, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  auto predicate = Mutate(op->predicate, inplace_mode);
  auto block = Mutate(op->block, inplace_mode).as_or_throw<UnchangedOr<SBlock>>();
  if (iter_values.UnchangedOrSameAs(op->iter_values) &&
      predicate.UnchangedOrSameAs(op->predicate) && block.UnchangedOrSameAs(op->block))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<SBlockRealizeNode*>(op);
    if (!iter_values.IsUnchanged()) writable->iter_values = std::move(iter_values).ValueUnchecked();
    if (!predicate.IsUnchanged()) writable->predicate = std::move(predicate).ValueUnchecked();
    if (!block.IsUnchanged()) writable->block = std::move(block).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<SBlockRealizeNode>(*op);
  if (!iter_values.IsUnchanged()) copy->iter_values = std::move(iter_values).ValueUnchecked();
  if (!predicate.IsUnchanged()) copy->predicate = std::move(predicate).ValueUnchecked();
  if (!block.IsUnchanged()) copy->block = std::move(block).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const BreakNode* op, InplaceMode inplace_mode) {
  return ffi::Unchanged();
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const ContinueNode* op, InplaceMode inplace_mode) {
  return ffi::Unchanged();
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const AllocBufferNode* op, InplaceMode inplace_mode) {
  auto buffer = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
                  return Mutate(op->buffer, inplace_mode);
                }).as_or_throw<UnchangedOr<BufferVar>>();
  if (buffer.UnchangedOrSameAs(op->buffer)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<AllocBufferNode*>(op);
    if (!buffer.IsUnchanged()) writable->buffer = std::move(buffer).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<AllocBufferNode>(*op);
  if (!buffer.IsUnchanged()) copy->buffer = std::move(buffer).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const DeclBufferNode* op, InplaceMode inplace_mode) {
  auto data = Mutate(op->data, inplace_mode);
  auto buffer = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
                  return Mutate(op->buffer, inplace_mode);
                }).as_or_throw<UnchangedOr<BufferVar>>();
  if (data.UnchangedOrSameAs(op->data) && buffer.UnchangedOrSameAs(op->buffer))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<DeclBufferNode*>(op);
    if (!data.IsUnchanged()) writable->data = std::move(data).ValueUnchecked();
    if (!buffer.IsUnchanged()) writable->buffer = std::move(buffer).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<DeclBufferNode>(*op);
  if (!data.IsUnchanged()) copy->data = std::move(data).ValueUnchecked();
  if (!buffer.IsUnchanged()) copy->buffer = std::move(buffer).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) {
  auto buffer = Mutate(op->buffer, inplace_mode).as_or_throw<UnchangedOr<BufferVar>>();
  auto value = Mutate(op->value, inplace_mode);
  auto indices = Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  if (buffer.UnchangedOrSameAs(op->buffer) && value.UnchangedOrSameAs(op->value) &&
      indices.UnchangedOrSameAs(op->indices))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<BufferStoreNode*>(op);
    if (!buffer.IsUnchanged()) writable->buffer = std::move(buffer).ValueUnchecked();
    if (!value.IsUnchanged()) writable->value = std::move(value).ValueUnchecked();
    if (!indices.IsUnchanged()) writable->indices = std::move(indices).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<BufferStoreNode>(*op);
  if (!buffer.IsUnchanged()) copy->buffer = std::move(buffer).ValueUnchecked();
  if (!value.IsUnchanged()) copy->value = std::move(value).ValueUnchecked();
  if (!indices.IsUnchanged()) copy->indices = std::move(indices).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const SBlockNode* op, InplaceMode inplace_mode) {
  // SBlock iteration variables keep their binders; only their domains are expressions here.
  const auto* iters = op->iter_vars.GetArrayObj();
  InplaceMode iter_mode = iters->unique() ? inplace_mode : InplaceMode::kDisallow;
  std::vector<std::pair<size_t, IterVar>> replacements;
  for (size_t i = 0; i < iters->size(); ++i) {
    const auto* iter = (*iters)[i].as<IterVarNode>();
    InplaceMode domain_mode = iter->unique() ? iter_mode : InplaceMode::kDisallow;
    auto domain = Mutate(iter->dom, domain_mode).as_or_throw<UnchangedOr<Range>>();
    if (domain.UnchangedOrSameAs(iter->dom)) continue;
    if (domain_mode == InplaceMode::kAllow) {
      const_cast<IterVarNode*>(iter)->dom = std::move(domain).ValueUnchecked();
    } else {
      auto updated = ffi::make_object<IterVarNode>(*iter);
      updated->dom = std::move(domain).ValueUnchecked();
      replacements.emplace_back(i, IterVar(std::move(updated)));
    }
  }
  UnchangedOr<ffi::Array<IterVar>> iter_vars = ffi::Unchanged();
  if (!replacements.empty()) {
    if (iter_mode == InplaceMode::kAllow) {
      for (auto& [i, iter] : replacements) {
        const_cast<ffi::ArrayObj*>(iters)->SetItem(i, std::move(iter));
      }
    } else {
      ffi::Array<IterVar> updated = op->iter_vars;
      for (auto& [i, iter] : replacements) updated.Set(i, std::move(iter));
      iter_vars = std::move(updated);
    }
  }
  auto alloc_buffers = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
                         return Mutate(op->alloc_buffers, inplace_mode);
                       }).as_or_throw<UnchangedOr<ffi::Array<BufferVar>>>();
  auto reads = Mutate(op->reads, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<TensorRegion>>>();
  auto writes =
      Mutate(op->writes, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<TensorRegion>>>();
  auto match_buffers = Mutate(op->match_buffers, inplace_mode)
                           .as_or_throw<UnchangedOr<ffi::Array<MatchBufferRegion>>>();
  auto init = Mutate(op->init, inplace_mode).as_or_throw<UnchangedOr<ffi::Optional<Stmt>>>();
  auto body = Mutate(op->body, inplace_mode);
  if (iter_vars.UnchangedOrSameAs(op->iter_vars) &&
      alloc_buffers.UnchangedOrSameAs(op->alloc_buffers) && reads.UnchangedOrSameAs(op->reads) &&
      writes.UnchangedOrSameAs(op->writes) && match_buffers.UnchangedOrSameAs(op->match_buffers) &&
      init.UnchangedOrSameAs(op->init) && body.UnchangedOrSameAs(op->body))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<SBlockNode*>(op);
    if (!iter_vars.IsUnchanged()) writable->iter_vars = std::move(iter_vars).ValueUnchecked();
    if (!alloc_buffers.IsUnchanged())
      writable->alloc_buffers = std::move(alloc_buffers).ValueUnchecked();
    if (!reads.IsUnchanged()) writable->reads = std::move(reads).ValueUnchecked();
    if (!writes.IsUnchanged()) writable->writes = std::move(writes).ValueUnchecked();
    if (!match_buffers.IsUnchanged())
      writable->match_buffers = std::move(match_buffers).ValueUnchecked();
    if (!init.IsUnchanged()) writable->init = std::move(init).ValueUnchecked();
    if (!body.IsUnchanged()) writable->body = std::move(body).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<SBlockNode>(*op);
  if (!iter_vars.IsUnchanged()) copy->iter_vars = std::move(iter_vars).ValueUnchecked();
  if (!alloc_buffers.IsUnchanged()) copy->alloc_buffers = std::move(alloc_buffers).ValueUnchecked();
  if (!reads.IsUnchanged()) copy->reads = std::move(reads).ValueUnchecked();
  if (!writes.IsUnchanged()) copy->writes = std::move(writes).ValueUnchecked();
  if (!match_buffers.IsUnchanged()) copy->match_buffers = std::move(match_buffers).ValueUnchecked();
  if (!init.IsUnchanged()) copy->init = std::move(init).ValueUnchecked();
  if (!body.IsUnchanged()) copy->body = std::move(body).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const SeqStmtNode* op, InplaceMode inplace_mode) {
  return detail::MutateSeqStmt(op, inplace_mode, [this](ffi::AnyView element, InplaceMode mode) {
    return Mutate(element, mode).as_or_throw<UnchangedOr<Stmt>>();
  });
}

namespace {

template <typename T, typename F>
UnchangedOr<ffi::Array<T>> MutateTileArray(const ffi::ArrayObj* values, InplaceMode mode,
                                           F fmutate) {
  // Borrow both the owning container and its elements throughout recursion.
  if (!values->unique()) mode = InplaceMode::kDisallow;
  std::vector<std::pair<size_t, T>> replacements;
  for (size_t i = 0; i < values->size(); ++i) {
    UnchangedOr<ffi::Any> result = fmutate(ffi::AnyView((*values)[i]), mode);
    if (!result.UnchangedOrSameAs((*values)[i])) {
      replacements.emplace_back(i, std::move(result).ValueUnchecked().template as_or_throw<T>());
    }
  }
  if (replacements.empty()) return ffi::Unchanged();
  if (mode == InplaceMode::kAllow) {
    auto* writable = const_cast<ffi::ArrayObj*>(values);
    for (auto& [i, value] : replacements) writable->SetItem(i, std::move(value));
    return ffi::Unchanged();
  }
  ffi::Array<T> result(ffi::GetObjectPtr<ffi::ArrayObj>(const_cast<ffi::ArrayObj*>(values)));
  for (auto& [i, value] : replacements) result.Set(i, std::move(value));
  return result;
}

}  // namespace

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
  std::function<UnchangedOr<ffi::Any>(ffi::AnyView, InplaceMode)> mutate_arg;
  mutate_arg = [&](ffi::AnyView value, InplaceMode mode) -> UnchangedOr<ffi::Any> {
    if (value.as<TensorRegionNode>()) {
      return Mutate(value, mode);
    }
    if (const auto* var = value.as<VarNode>(); var && var->ty.as<BufferTypeNode>()) {
      return Mutate(value, mode);
    }
    if (value.as<PrimExpr>() || value.as<StmtNode>()) return Mutate(value, mode);
    if (const auto* array = value.as<ffi::ArrayObj>()) {
      return MutateTileArray<ffi::Any>(array, mode, mutate_arg);
    }
    return ffi::Unchanged();
  };
  auto args = MutateTileArray<ffi::Any>(op->args.GetArrayObj(), inplace_mode, mutate_arg);
  // A config map is another owning container on the path to its values.
  auto config_mode = op->config.unique() ? inplace_mode : InplaceMode::kDisallow;
  UnchangedOr<ffi::Map<ffi::String, ffi::Any>> config = ffi::Unchanged();
  std::vector<std::pair<ffi::String, ffi::Any>> replacements;
  for (const auto& [key, value] : *static_cast<const ffi::MapObj*>(op->config.get())) {
    auto result = mutate_arg(value, config_mode);
    if (!result.UnchangedOrSameAs(value)) {
      replacements.emplace_back(key.as_or_throw<ffi::String>(), std::move(result).ValueUnchecked());
    }
  }
  if (!replacements.empty()) {
    if (config_mode == InplaceMode::kAllow) {
      for (auto& [key, value] : replacements)
        const_cast<ffi::MapObj*>(static_cast<const ffi::MapObj*>(op->config.get()))->at(key) =
            std::move(value);
    } else {
      ffi::Map<ffi::String, ffi::Any> replacement = op->config;
      for (auto& [key, value] : replacements) replacement.Set(key, std::move(value));
      config = std::move(replacement);
    }
  }
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
    if (auto buffer = load->source.as<BufferVar>()) {
      return BufferLoad(buffer.value(), load->indices, load->span);
    }
    return load;
  }

  UnchangedOr<Stmt> Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) final {
    Stmt ret = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = ret.as<AttrStmtNode>();
    // remap var node in attr
    if (auto var_node = op->node.as<Var>()) {
      ffi::Any mapped = VarRemapGet(var_node.value());
      if (mapped.type_index() != ffi::TypeIndex::kTVMFFINone) {
        Expr node =
            std::move(mapped).as_or_throw<UnchangedOr<Expr>>().ValueOrUnchanged(var_node.value());
        if (!node.same_as(var_node.value())) {
          return AttrStmt(node, op->attr_key, op->value, op->body);
        }
      }
    }
    return ret;
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
