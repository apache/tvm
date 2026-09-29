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
 * \file tvm/tirx/stmt.cc
 */
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>

#include <iterator>
#include <limits>
#include <utility>
#include <vector>

#include "buffer_common.h"
#include "seq_stmt_mutate.h"

namespace tvm {
namespace tirx {
using namespace tvm::prim;

namespace {

/*!
 * \brief Whether an integer literal can be represented exactly by `ty`.
 * \note Mirrors the range checks performed by the IntImm constructor.
 */
bool IntImmValueFits(const ffi::BigInt& value, const PrimType& ty) {
  int bits = ty.bits();
  if (ty.MatchesCode(DLDataTypeCode::kDLUInt)) {
    if (bits <= 64) {
      uint64_t maximum =
          bits == 64 ? std::numeric_limits<uint64_t>::max() : (uint64_t{1} << bits) - 1;
      return value >= 0 && value <= maximum;
    }
    return value >= 0 && value < (ffi::BigInt(1) << bits);
  }
  if (bits <= 64) {
    int64_t maximum =
        bits == 64 ? std::numeric_limits<int64_t>::max() : (int64_t{1} << (bits - 1)) - 1;
    return value >= -maximum - 1 && value <= maximum;
  }
  ffi::BigInt limit = ffi::BigInt(1) << (bits - 1);
  return value >= -limit && value < limit;
}

// Structural traversal hooks

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> BindVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const BindNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BindNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->var); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> BindMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const BindNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BindNode>(value);
  Type old_value_type = self->value->ty;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MutateExpected(self->value));
  bool value_changed = !mapped_value.UnchangedOrSameAs(self->value);
  Expr new_value = std::move(mapped_value).ValueOrUnchanged(self->value);
  value_changed |= !old_value_type.same_as(new_value->ty);
  Type old_var_type = self->var->ty;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Var>, mapped_var,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->var);
                                    }));
  bool var_changed = !mapped_var.UnchangedOrSameAs(self->var);
  Var new_var = std::move(mapped_var).ValueOrUnchanged(self->var);
  var_changed |= !old_var_type.same_as(new_var->ty);
  if ((value_changed || var_changed) && !new_var->ty.same_as(new_value->ty)) {
    new_var = new_var.CopyWithType(new_value->ty);
  }
  if (!new_var.same_as(self->var)) {
    auto remap = mutator->VarRemapSetExpected(self->var, new_var);
    TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap);
  }
  if (new_var.same_as(self->var) && new_value.same_as(self->value)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<BindNode> copy = ffi::make_object<BindNode>(*self);
  copy->var = std::move(new_var);
  copy->value = std::move(new_value);
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> BindMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  BindNode* self = const_cast<BindNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BindNode>(value));
  Type old_value_type = self->value->ty;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MutateExpected(self->value, ffi::InplaceMode::kAllow));
  bool value_changed = !mapped_value.UnchangedOrSameAs(self->value);
  Expr new_value = std::move(mapped_value).ValueOrUnchanged(self->value);
  value_changed |= !old_value_type.same_as(new_value->ty);
  Type old_var_type = self->var->ty;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Var>, mapped_var,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->var,
                                                                     ffi::InplaceMode::kAllow);
                                    }));
  bool var_changed = !mapped_var.UnchangedOrSameAs(self->var);
  Var new_var = std::move(mapped_var).ValueOrUnchanged(self->var);
  var_changed |= !old_var_type.same_as(new_var->ty);
  if ((value_changed || var_changed) && !new_var->ty.same_as(new_value->ty)) {
    new_var = new_var.CopyWithType(new_value->ty);
  }
  if (!new_var.same_as(self->var)) {
    auto remap = mutator->VarRemapSetExpected(self->var, new_var);
    TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap);
  }
  if (new_var.same_as(self->var) && new_value.same_as(self->value)) {
    return ffi::Unchanged();
  }
  self->var = std::move(new_var);
  self->value = std::move(new_value);
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> AttrStmtVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: attr_key
  const AttrStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AttrStmtNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->node));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> AttrStmtMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attr_key
  const AttrStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AttrStmtNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Any>, mapped_node,
                                    mutator->MutateExpected(self->node));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MutateExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body));
  if (mapped_node.UnchangedOrSameAs(self->node) && mapped_value.UnchangedOrSameAs(self->value) &&
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<AttrStmtNode> copy = ffi::make_object<AttrStmtNode>(*self);
  copy->node = std::move(mapped_node).ValueOrUnchanged(std::move(copy->node));
  copy->value = std::move(mapped_value).ValueOrUnchanged(std::move(copy->value));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> AttrStmtMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attr_key
  AttrStmtNode* self = const_cast<AttrStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AttrStmtNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Any>, mapped_node,
                                    mutator->MutateExpected(self->node, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MutateExpected(self->value, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body, ffi::InplaceMode::kAllow));
  if (mapped_node.UnchangedOrSameAs(self->node) && mapped_value.UnchangedOrSameAs(self->value) &&
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged();
  }
  if (!mapped_node.IsUnchanged()) self->node = std::move(mapped_node).ValueUnchecked();
  if (!mapped_value.IsUnchanged()) self->value = std::move(mapped_value).ValueUnchecked();
  if (!mapped_body.IsUnchanged()) self->body = std::move(mapped_body).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> AssertStmtVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: error_kind and message_parts, which are constant assertion metadata.
  const AssertStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AssertStmtNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->condition));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> AssertStmtMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: error_kind and message_parts, which are constant assertion metadata.
  const AssertStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AssertStmtNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_condition,
                                    mutator->MutateExpected(self->condition));
  if (mapped_condition.UnchangedOrSameAs(self->condition)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<AssertStmtNode> copy = ffi::make_object<AssertStmtNode>(*self);
  copy->condition = std::move(mapped_condition).ValueOrUnchanged(std::move(copy->condition));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> AssertStmtMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: error_kind and message_parts, which are constant assertion metadata.
  AssertStmtNode* self = const_cast<AssertStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AssertStmtNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_condition,
      mutator->MutateExpected(self->condition, ffi::InplaceMode::kAllow));
  if (mapped_condition.UnchangedOrSameAs(self->condition)) {
    return ffi::Unchanged();
  }
  if (!mapped_condition.IsUnchanged())
    self->condition = std::move(mapped_condition).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> ForVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: kind and constant annotations; unlike SBlock annotations, these carry no expressions.
  const ForNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ForNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->loop_var); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->min));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->extent));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->thread_binding));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->step));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ForMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: kind and constant annotations; unlike SBlock annotations, these carry no expressions.
  const ForNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ForNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimVar>, mapped_loop_var,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->loop_var);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_min,
                                    mutator->MutateExpected(self->min));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_extent,
                                    mutator->MutateExpected(self->extent));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<IterVar>>, mapped_thread_binding,
                                    mutator->MutateExpected(self->thread_binding));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<PrimExpr>>, mapped_step,
                                    mutator->MutateExpected(self->step));
  if (mapped_loop_var.UnchangedOrSameAs(self->loop_var) &&
      mapped_min.UnchangedOrSameAs(self->min) && mapped_extent.UnchangedOrSameAs(self->extent) &&
      mapped_body.UnchangedOrSameAs(self->body) &&
      mapped_thread_binding.UnchangedOrSameAs(self->thread_binding) &&
      mapped_step.UnchangedOrSameAs(self->step)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<ForNode> copy = ffi::make_object<ForNode>(*self);
  copy->loop_var = std::move(mapped_loop_var).ValueOrUnchanged(std::move(copy->loop_var));
  copy->min = std::move(mapped_min).ValueOrUnchanged(std::move(copy->min));
  copy->extent = std::move(mapped_extent).ValueOrUnchanged(std::move(copy->extent));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  copy->thread_binding =
      std::move(mapped_thread_binding).ValueOrUnchanged(std::move(copy->thread_binding));
  copy->step = std::move(mapped_step).ValueOrUnchanged(std::move(copy->step));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ForMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: kind and constant annotations; unlike SBlock annotations, these carry no expressions.
  ForNode* self = const_cast<ForNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ForNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimVar>, mapped_loop_var,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->loop_var,
                                                                     ffi::InplaceMode::kAllow);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_min,
                                    mutator->MutateExpected(self->min, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_extent,
      mutator->MutateExpected(self->extent, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Optional<IterVar>>, mapped_thread_binding,
      mutator->MutateExpected(self->thread_binding, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<PrimExpr>>, mapped_step,
                                    mutator->MutateExpected(self->step, ffi::InplaceMode::kAllow));
  if (mapped_loop_var.UnchangedOrSameAs(self->loop_var) &&
      mapped_min.UnchangedOrSameAs(self->min) && mapped_extent.UnchangedOrSameAs(self->extent) &&
      mapped_body.UnchangedOrSameAs(self->body) &&
      mapped_thread_binding.UnchangedOrSameAs(self->thread_binding) &&
      mapped_step.UnchangedOrSameAs(self->step)) {
    return ffi::Unchanged();
  }
  if (!mapped_loop_var.IsUnchanged()) self->loop_var = std::move(mapped_loop_var).ValueUnchecked();
  if (!mapped_min.IsUnchanged()) self->min = std::move(mapped_min).ValueUnchecked();
  if (!mapped_extent.IsUnchanged()) self->extent = std::move(mapped_extent).ValueUnchecked();
  if (!mapped_body.IsUnchanged()) self->body = std::move(mapped_body).ValueUnchecked();
  if (!mapped_thread_binding.IsUnchanged()) {
    self->thread_binding = std::move(mapped_thread_binding).ValueUnchecked();
  }
  if (!mapped_step.IsUnchanged()) self->step = std::move(mapped_step).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> WhileVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const WhileNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const WhileNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> WhileMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const WhileNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const WhileNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_condition,
                                    mutator->MutateExpected(self->condition));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body));
  if (mapped_condition.UnchangedOrSameAs(self->condition) &&
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<WhileNode> copy = ffi::make_object<WhileNode>(*self);
  copy->condition = std::move(mapped_condition).ValueOrUnchanged(std::move(copy->condition));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> WhileMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  WhileNode* self = const_cast<WhileNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const WhileNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_condition,
      mutator->MutateExpected(self->condition, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body, ffi::InplaceMode::kAllow));
  if (mapped_condition.UnchangedOrSameAs(self->condition) &&
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged();
  }
  if (!mapped_condition.IsUnchanged())
    self->condition = std::move(mapped_condition).ValueUnchecked();
  if (!mapped_body.IsUnchanged()) self->body = std::move(mapped_body).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> ReturnVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const ReturnNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ReturnNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ReturnMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const ReturnNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ReturnNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MutateExpected(self->value));
  if (mapped_value.UnchangedOrSameAs(self->value)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<ReturnNode> copy = ffi::make_object<ReturnNode>(*self);
  copy->value = std::move(mapped_value).ValueOrUnchanged(std::move(copy->value));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ReturnMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  ReturnNode* self = const_cast<ReturnNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ReturnNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MutateExpected(self->value, ffi::InplaceMode::kAllow));
  if (mapped_value.UnchangedOrSameAs(self->value)) {
    return ffi::Unchanged();
  }
  if (!mapped_value.IsUnchanged()) self->value = std::move(mapped_value).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> BreakVisit(
    ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> BreakMutate(ffi::StructuralMutatorObj*,
                                                                     ffi::AnyView) noexcept {
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> BreakMaybeInplaceMutate(
    ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> ContinueVisit(
    ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ContinueMutate(ffi::StructuralMutatorObj*,
                                                                        ffi::AnyView) noexcept {
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ContinueMaybeInplaceMutate(
    ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> SeqStmtVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const SeqStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->seq));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> MutateSeqStmtStructural(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value,
    ffi::InplaceMode inplace_mode) noexcept {
  const auto* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(value);
  auto mutate = [mutator](ffi::AnyView element,
                          ffi::InplaceMode mode) -> ffi::Expected<ffi::UnchangedOr<Stmt>> {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped,
                                      mutator->MutateExpected(element, mode));
    return mapped;
  };
  return detail::MutateSeqStmt(self, inplace_mode, mutate);
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> SeqStmtMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  return MutateSeqStmtStructural(mutator, value, ffi::InplaceMode::kDisallow);
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> SeqStmtMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  return MutateSeqStmtStructural(mutator, value, ffi::InplaceMode::kAllow);
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> IfThenElseVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const IfThenElseNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfThenElseNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->then_case));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->else_case));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> IfThenElseMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const IfThenElseNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfThenElseNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_condition,
                                    mutator->MutateExpected(self->condition));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_then_case,
                                    mutator->MutateExpected(self->then_case));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Stmt>>, mapped_else_case,
                                    mutator->MutateExpected(self->else_case));
  if (mapped_condition.UnchangedOrSameAs(self->condition) &&
      mapped_then_case.UnchangedOrSameAs(self->then_case) &&
      mapped_else_case.UnchangedOrSameAs(self->else_case)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<IfThenElseNode> copy = ffi::make_object<IfThenElseNode>(*self);
  copy->condition = std::move(mapped_condition).ValueOrUnchanged(std::move(copy->condition));
  copy->then_case = std::move(mapped_then_case).ValueOrUnchanged(std::move(copy->then_case));
  copy->else_case = std::move(mapped_else_case).ValueOrUnchanged(std::move(copy->else_case));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> IfThenElseMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  IfThenElseNode* self = const_cast<IfThenElseNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfThenElseNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_condition,
      mutator->MutateExpected(self->condition, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Stmt>, mapped_then_case,
      mutator->MutateExpected(self->then_case, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Optional<Stmt>>, mapped_else_case,
      mutator->MutateExpected(self->else_case, ffi::InplaceMode::kAllow));
  if (mapped_condition.UnchangedOrSameAs(self->condition) &&
      mapped_then_case.UnchangedOrSameAs(self->then_case) &&
      mapped_else_case.UnchangedOrSameAs(self->else_case)) {
    return ffi::Unchanged();
  }
  if (!mapped_condition.IsUnchanged())
    self->condition = std::move(mapped_condition).ValueUnchecked();
  if (!mapped_then_case.IsUnchanged())
    self->then_case = std::move(mapped_then_case).ValueUnchecked();
  if (!mapped_else_case.IsUnchanged())
    self->else_case = std::move(mapped_else_case).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> EvaluateVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const EvaluateNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> EvaluateMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const EvaluateNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MutateExpected(self->value));
  if (mapped_value.UnchangedOrSameAs(self->value)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<EvaluateNode> copy = ffi::make_object<EvaluateNode>(*self);
  copy->value = std::move(mapped_value).ValueOrUnchanged(std::move(copy->value));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> EvaluateMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  EvaluateNode* self = const_cast<EvaluateNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MutateExpected(self->value, ffi::InplaceMode::kAllow));
  if (mapped_value.UnchangedOrSameAs(self->value)) {
    return ffi::Unchanged();
  }
  if (!mapped_value.IsUnchanged()) self->value = std::move(mapped_value).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> BufferStoreVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const BufferStoreNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferStoreNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->buffer));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->indices));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> BufferStoreMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const BufferStoreNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferStoreNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<BufferVar>, mapped_buffer,
                                    mutator->MutateExpected(self->buffer));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value,
                                    mutator->MutateExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_indices,
                                    mutator->MutateExpected(self->indices));
  if (mapped_buffer.UnchangedOrSameAs(self->buffer) &&
      mapped_value.UnchangedOrSameAs(self->value) &&
      mapped_indices.UnchangedOrSameAs(self->indices)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<BufferStoreNode> copy = ffi::make_object<BufferStoreNode>(*self);
  copy->buffer = std::move(mapped_buffer).ValueOrUnchanged(std::move(copy->buffer));
  copy->value = std::move(mapped_value).ValueOrUnchanged(std::move(copy->value));
  copy->indices = std::move(mapped_indices).ValueOrUnchanged(std::move(copy->indices));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> BufferStoreMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  BufferStoreNode* self = const_cast<BufferStoreNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferStoreNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<BufferVar>, mapped_buffer,
      mutator->MutateExpected(self->buffer, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value,
                                    mutator->MutateExpected(self->value, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_indices,
      mutator->MutateExpected(self->indices, ffi::InplaceMode::kAllow));
  if (mapped_buffer.UnchangedOrSameAs(self->buffer) &&
      mapped_value.UnchangedOrSameAs(self->value) &&
      mapped_indices.UnchangedOrSameAs(self->indices)) {
    return ffi::Unchanged();
  }
  if (!mapped_buffer.IsUnchanged()) self->buffer = std::move(mapped_buffer).ValueUnchecked();
  if (!mapped_value.IsUnchanged()) self->value = std::move(mapped_value).ValueUnchecked();
  if (!mapped_indices.IsUnchanged()) self->indices = std::move(mapped_indices).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> ScopeIdDefStmtVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const ScopeIdDefStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ScopeIdDefStmtNode>(value);
  // ScopeIdDef reflection marks only def_ids as definitions; its extent fields remain uses.
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->def));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ScopeIdDefStmtMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const ScopeIdDefStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ScopeIdDefStmtNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ScopeIdDef>, mapped_def,
                                    mutator->MutateExpected(self->def));
  if (mapped_def.UnchangedOrSameAs(self->def)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<ScopeIdDefStmtNode> copy = ffi::make_object<ScopeIdDefStmtNode>(*self);
  copy->def = std::move(mapped_def).ValueOrUnchanged(std::move(copy->def));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ScopeIdDefStmtMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  ScopeIdDefStmtNode* self = const_cast<ScopeIdDefStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ScopeIdDefStmtNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ScopeIdDef>, mapped_def,
                                    mutator->MutateExpected(self->def, ffi::InplaceMode::kAllow));
  if (mapped_def.UnchangedOrSameAs(self->def)) {
    return ffi::Unchanged();
  }
  if (!mapped_def.IsUnchanged()) self->def = std::move(mapped_def).ValueUnchecked();
  return ffi::Unchanged();
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() { StmtNode::RegisterReflection(); }

// Bind
Bind::Bind(Var var, Expr value, Span span) {
  TVM_FFI_ICHECK(value.defined());
  TVM_FFI_ICHECK(ffi::StructuralEqual()(value->ty, var->ty));

  ffi::ObjectPtr<BindNode> node = ffi::make_object<BindNode>();
  node->var = std::move(var);
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BindNode::RegisterReflection();
  refl::TypeAttrDef<BindNode>()
      .attr(refl::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&BindVisit>())
      .attr(refl::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&BindMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BindMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.Bind",
                        [](Var var, Expr value, Span span) { return Bind(var, value, span); });
}

// AttrStmt
AttrStmt::AttrStmt(ffi::Any node, ffi::String attr_key, Expr value, Stmt body, Span span) {
  auto n = ffi::make_object<AttrStmtNode>();
  n->node = node;
  n->attr_key = std::move(attr_key);
  n->value = std::move(value);
  n->body = std::move(body);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  AttrStmtNode::RegisterReflection();
  refl::TypeAttrDef<AttrStmtNode>()
      .attr(refl::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&AttrStmtVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&AttrStmtMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&AttrStmtMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.AttrStmt",
                        [](Any node, ffi::String attr_key, Expr value, Stmt body, Span span) {
                          return AttrStmt(node, attr_key, value, body, span);
                        });
}

// AssertStmt
AssertStmt::AssertStmt(PrimExpr condition, StringImm error_kind,
                       ffi::Array<StringImm> message_parts, Span span) {
  TVM_FFI_ICHECK(condition.defined());
  PrimType condition_ty = condition.ty();
  TVM_FFI_ICHECK(condition_ty.MatchesCode(DLDataTypeCode::kDLBool))
      << "AssertStmt should have boolean condition, "
      << "but received " << condition << " with dtype " << condition_ty;
  TVM_FFI_ICHECK(error_kind.defined());

  ffi::ObjectPtr<AssertStmtNode> node = ffi::make_object<AssertStmtNode>();
  node->condition = std::move(condition);
  node->error_kind = std::move(error_kind);
  node->message_parts = std::move(message_parts);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  AssertStmtNode::RegisterReflection();
  refl::TypeAttrDef<AssertStmtNode>()
      .attr(refl::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&AssertStmtVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&AssertStmtMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&AssertStmtMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.AssertStmt", [](PrimExpr condition, StringImm error_kind,
                                              ffi::Array<StringImm> message_parts, Span span) {
    return AssertStmt(condition, error_kind, message_parts, span);
  });
}

// For
For::For(PrimVar loop_var, PrimExpr min, PrimExpr extent, ForKind kind, Stmt body,
         ffi::Optional<IterVar> thread_binding, ffi::Map<ffi::String, Any> annotations,
         ffi::Optional<PrimExpr> step, Span span) {
  TVM_FFI_ICHECK(loop_var.defined());
  TVM_FFI_ICHECK(min.defined());
  TVM_FFI_ICHECK(extent.defined());
  TVM_FFI_ICHECK(body.defined());

  auto require_scalar_int_dtype = [&](PrimExpr expr, const char* field_name) {
    PrimType dtype = expr.ty();
    TVM_FFI_ICHECK(dtype.IsScalar() &&
                   (dtype.MatchesCode(DLDataTypeCode::kDLUInt, DLDataTypeCode::kDLInt)))
        << "TIR For nodes require a scalar integer as the " << field_name << ", but received "
        << expr << " with dtype " << dtype;
  };
  require_scalar_int_dtype(loop_var, "loop_var");
  require_scalar_int_dtype(min, "min");
  require_scalar_int_dtype(extent, "extent");

  // When extent, min or step is an IntImm whose dtype differs from loop_var's
  // (narrower bits and/or a different signedness code), we directly promote it
  // to the loop var's dtype as long as the value stays representable.
  auto try_promote_imm_dtype = [&](const PrimExpr& e) -> PrimExpr {
    PrimType e_ty = e.ty();
    PrimType loop_var_ty = loop_var.ty();
    if (e_ty == loop_var_ty) return e;
    if (const IntImmNode* a = e.as<IntImmNode>()) {
      TVM_FFI_ICHECK(IntImmValueFits(a->value, loop_var_ty))
          << "Literal value " << a->value << " is not representable in the loop variable's dtype ("
          << loop_var_ty << ")";
      return IntImm(loop_var_ty, a->value);
    }
    TVM_FFI_ICHECK(e_ty.bits() <= loop_var_ty.bits())
        << " Loop variable's dtype (" << loop_var_ty
        << ") is narrower than that of `min` or `extent` (" << e_ty << ")";
    return e;
  };

  min = try_promote_imm_dtype(min);
  extent = try_promote_imm_dtype(extent);

  TVM_FFI_ICHECK(loop_var.ty() == min.ty()) << loop_var.ty() << " vs " << min.ty();
  TVM_FFI_ICHECK(loop_var.ty() == extent.ty()) << loop_var.ty() << " vs " << extent.ty();

  if (step.has_value()) {
    require_scalar_int_dtype(*step, "step");
    step = try_promote_imm_dtype(*step);
    TVM_FFI_ICHECK(loop_var.ty() == step.value().ty())
        << loop_var.ty() << " vs " << step.value().ty();
  }

  ffi::ObjectPtr<ForNode> node = ffi::make_object<ForNode>();
  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->extent = std::move(extent);
  node->kind = kind;
  node->body = std::move(body);
  node->thread_binding = std::move(thread_binding);
  node->annotations = std::move(annotations);
  node->step = std::move(step);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ForNode::RegisterReflection();
  refl::TypeAttrDef<ForNode>()
      .attr(refl::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&ForVisit>())
      .attr(refl::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&ForMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&ForMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.For", [](PrimVar loop_var, PrimExpr min, PrimExpr extent, int kind,
                                       Stmt body, ffi::Optional<IterVar> thread_binding,
                                       ffi::Optional<ffi::Map<ffi::String, Any>> annotations,
                                       ffi::Optional<PrimExpr> step, Span span) {
    return For(loop_var, min, extent, static_cast<ForKind>(kind), body, thread_binding,
               annotations.value_or(ffi::Map<ffi::String, Any>()), step, span);
  });
}

bool ForNode::HasTrivialStep() const { return !step.has_value() || is_one(*step); }

std::ostream& operator<<(std::ostream& out, ForKind type) {  // NOLINT(*)
  switch (type) {
    case ForKind::kSerial:
      out << "for";
      break;
    case ForKind::kParallel:
      out << "parallel";
      break;
    case ForKind::kUnrolled:
      out << "unrolled";
      break;
    case ForKind::kVectorized:
      out << "vectorized";
      break;
    case ForKind::kThreadBinding:
      out << "launch_thread";
      break;
  }
  return out;
}

// While
While::While(PrimExpr condition, Stmt body, Span span) {
  TVM_FFI_ICHECK(condition.defined());
  TVM_FFI_ICHECK(condition.ty().IsScalar());
  TVM_FFI_ICHECK(body.defined());

  ffi::ObjectPtr<WhileNode> node = ffi::make_object<WhileNode>();
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  WhileNode::RegisterReflection();
  refl::TypeAttrDef<WhileNode>()
      .attr(refl::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&WhileVisit>())
      .attr(refl::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&WhileMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&WhileMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.While", [](PrimExpr condition, Stmt body, Span span) {
    return While(condition, body, span);
  });
}

// Return
Return::Return(Expr value, Span span) {
  TVM_FFI_ICHECK(value.defined());

  ffi::ObjectPtr<ReturnNode> node = ffi::make_object<ReturnNode>();
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ReturnNode::RegisterReflection();
  refl::TypeAttrDef<ReturnNode>()
      .attr(refl::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&ReturnVisit>())
      .attr(refl::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&ReturnMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&ReturnMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.Return", [](Expr value, Span span) { return Return(value, span); });
}

// Break
Break::Break(Span span) {
  ffi::ObjectPtr<BreakNode> node = ffi::make_object<BreakNode>();
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BreakNode::RegisterReflection();
  refl::TypeAttrDef<BreakNode>()
      .attr(refl::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&BreakVisit>())
      .attr(refl::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&BreakMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BreakMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.Break", [](Span span) { return Break(span); });
}

// Continue
Continue::Continue(Span span) {
  ffi::ObjectPtr<ContinueNode> node = ffi::make_object<ContinueNode>();
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ContinueNode::RegisterReflection();
  refl::TypeAttrDef<ContinueNode>()
      .attr(refl::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&ContinueVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&ContinueMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&ContinueMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.Continue", [](Span span) { return Continue(span); });
}

// SeqStmt
SeqStmt::SeqStmt(ffi::Array<Stmt> seq, Span span) {
  bool requires_flattening = std::any_of(
      seq.begin(), seq.end(), [](const Stmt& stmt) { return stmt->IsInstance<SeqStmtNode>(); });

  if (requires_flattening) {
    auto flattened = SeqStmt::Flatten(seq);
    if (auto* ptr = flattened.as<SeqStmtNode>()) {
      seq = ptr->seq;
    } else {
      seq = {flattened};
    }
  }

  TVM_FFI_ICHECK_NE(seq.size(), 0) << "An empty SeqStmt is prohibited.  "
                                   << "To write a no-op, use Evaluate(0), "
                                   << "or the result of SeqStmt::Flatten()";
  TVM_FFI_ICHECK_NE(seq.size(), 1) << "A SeqStmt of length 1 is prohibited.  "
                                   << "Use the node " << seq[0] << "directly, "
                                   << "or for dynamic usage, normalize using SeqStmt::Flatten()";

  auto node = ffi::make_object<SeqStmtNode>();
  node->seq = std::move(seq);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SeqStmtNode::RegisterReflection();
  refl::TypeAttrDef<SeqStmtNode>()
      .attr(refl::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&SeqStmtVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&SeqStmtMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&SeqStmtMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.SeqStmt", [](ffi::Array<Stmt> seq, Span span) {
    return SeqStmt(std::move(seq), span);
  });
}

// IfThenElse
IfThenElse::IfThenElse(PrimExpr condition, Stmt then_case, ffi::Optional<Stmt> else_case,
                       Span span) {
  TVM_FFI_ICHECK(condition.defined());
  TVM_FFI_ICHECK(then_case.defined());
  // else_case may be null.
  ffi::ObjectPtr<IfThenElseNode> node = ffi::make_object<IfThenElseNode>();
  node->condition = std::move(condition);
  node->then_case = std::move(then_case);
  node->else_case = std::move(else_case);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  IfThenElseNode::RegisterReflection();
  refl::TypeAttrDef<IfThenElseNode>()
      .attr(refl::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&IfThenElseVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&IfThenElseMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&IfThenElseMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.IfThenElse",
                        [](PrimExpr condition, Stmt then_case, Stmt else_case, Span span) {
                          return IfThenElse(condition, then_case, else_case, span);
                        });
}

// Evaluate
Evaluate::Evaluate(Expr value, Span span) {
  TVM_FFI_ICHECK(value.defined());
  TVM_FFI_ICHECK(!(value->IsInstance<VarNode>() && value->ty.as<BufferTypeNode>()))
      << "A buffer variable cannot be used as a scalar Evaluate value; "
      << "use buffer.data to evaluate its physical pointer";

  ffi::ObjectPtr<EvaluateNode> node = ffi::make_object<EvaluateNode>();
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  EvaluateNode::RegisterReflection();
  refl::TypeAttrDef<EvaluateNode>()
      .attr(refl::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&EvaluateVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&EvaluateMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&EvaluateMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.Evaluate",
                        [](Expr value, Span span) { return Evaluate(value, span); });
}

// BufferStore
TVM_FFI_INLINE int GetLanesOrVScaleFactor(const PrimType& ty) {
  return ty.IsScalableVector() ? ty.VScaleFactor() : ty.lanes();
}

BufferStore::BufferStore(BufferVar buffer, PrimExpr value, ffi::Array<PrimExpr> indices,
                         Span span) {
  TVM_FFI_ICHECK_EQ(buffer->shape.size(), indices.size())
      << "BufferVar " << buffer.name() << " is " << buffer->shape.size()
      << "-dimensional, cannot be indexed with the " << indices.size()
      << "-dimensional indices provided.";

  for (int i = 0; i < static_cast<int>(indices.size()) - 1; i++) {
    TVM_FFI_ICHECK(indices[i].ty().IsScalar())
        << "Only the last index of a buffer access may be a vector type.";
  }

  bool is_index_scalable = indices.empty() ? false : indices.back().ty().IsScalableVector();
  int16_t buffer_encoded_lanes = static_cast<int16_t>(buffer->dtype->dtype.lanes);
  bool is_buffer_dtype_scalable = buffer_encoded_lanes < -1;
  PrimType value_ty = value.ty();
  bool is_value_dtype_scalable = value_ty.IsScalableVector();

  TVM_FFI_ICHECK(!(is_index_scalable && is_buffer_dtype_scalable))
      << "Index dtype and buffer dtype can't both be scalable.";

  if (is_index_scalable || is_buffer_dtype_scalable) {
    TVM_FFI_ICHECK(is_value_dtype_scalable) << "Can't store non-scalable data into scalable buffer";
  }

  int index_lanes = indices.empty() ? 1 : GetLanesOrVScaleFactor(indices.back().ty());
  int buffer_lanes = is_buffer_dtype_scalable ? -buffer_encoded_lanes : buffer_encoded_lanes;
  int value_dtype_lanes = GetLanesOrVScaleFactor(value_ty);

  TVM_FFI_ICHECK_EQ(index_lanes * buffer_lanes, value_dtype_lanes)
      << "Cannot store value with " << value_dtype_lanes << ", expected value with "
      << index_lanes * buffer_lanes << " (" << index_lanes << " index lanes * " << buffer_lanes
      << " buffer element lanes)";

  PrimType buffer_dtype = PrimType::Void();
  if (is_index_scalable || is_buffer_dtype_scalable) {
    buffer_dtype = PrimType::ScalableVector(buffer->dtype.code(), buffer->dtype.bits(),
                                            buffer_lanes * index_lanes);
  } else {
    buffer_dtype = buffer->dtype.WithLanes(buffer_lanes * index_lanes);
  }
  if (buffer_dtype != value_ty) {
    TVM_FFI_THROW(TypeError) << "dtype mismatch on BufferStore: "                 //
                             << "buffer's dtype is `" << buffer->dtype            //
                             << "`, the lanes of indexing are: `" << index_lanes  //
                             << "`, the scalability is: `" << buffer_dtype.IsScalableVector()
                             << "`, but RHS's dtype is `" << value_ty << "`";
  }

  ffi::ObjectPtr<BufferStoreNode> node = ffi::make_object<BufferStoreNode>();
  node->buffer = std::move(buffer);
  node->value = std::move(value);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BufferStoreNode::RegisterReflection();
  refl::TypeAttrDef<BufferStoreNode>()
      .attr(refl::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BufferStoreVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BufferStoreMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BufferStoreMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.BufferStore",
                        [](BufferVar buffer, PrimExpr value, ffi::Array<PrimExpr> indices,
                           Span span) { return BufferStore(buffer, value, indices, span); });
}

// ScopeIdDefStmt
ScopeIdDefStmt::ScopeIdDefStmt(ScopeIdDef def, Span span) {
  TVM_FFI_ICHECK(def.defined());
  ffi::ObjectPtr<ScopeIdDefStmtNode> node = ffi::make_object<ScopeIdDefStmtNode>();
  node->def = std::move(def);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ScopeIdDefStmtNode::RegisterReflection();
  refl::TypeAttrDef<ScopeIdDefStmtNode>()
      .attr(refl::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&ScopeIdDefStmtVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&ScopeIdDefStmtMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&ScopeIdDefStmtMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.ScopeIdDefStmt",
                        [](ScopeIdDef def, Span span) { return ScopeIdDefStmt(def, span); });
}

PrimExpr TypeAnnotation(PrimType dtype, Span span) {
  static const Op type_annotation_op = Op::Get("tirx.type_annotation");
  return Call(dtype, type_annotation_op, {}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.type_annotation")
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("type_annotation"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                           static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));
}

}  // namespace tirx
}  // namespace tvm
