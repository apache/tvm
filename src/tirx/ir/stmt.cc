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
#include <tvm/arith/analyzer.h>
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
#include <utility>
#include <vector>

#include "buffer_common.h"

namespace tvm {
namespace tirx {

namespace {

using SubscriptSlice = ffi::Array<ffi::Variant<
    ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>>,
    PrimExpr>>;

/*!
 * \brief Whether an integer literal can be represented exactly by `ty`.
 * \note Mirrors the range checks performed by the IntImm constructor.
 */
bool IntImmValueFits(int64_t value, const PrimType& ty) {
  int bits = ty.bits();
  if (ty.MatchesCode(DLDataTypeCode::kDLUInt)) {
    return value >= 0 && (bits >= 64 || value < (int64_t{1} << bits));
  }
  if (bits >= 64) return true;
  return value >= -(int64_t{1} << (bits - 1)) && value < (int64_t{1} << (bits - 1));
}

// Structural traversal hooks

TVMFFIAny BindVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const BindNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BindNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->var); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny BindMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const BindNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BindNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Var>, mapped_var,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->var);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MutateExpected(self->value));
  if (mapped_var.UnchangedOrSameAs(self->var) && mapped_value.UnchangedOrSameAs(self->value)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<BindNode> copy = ffi::make_object<BindNode>(*self);
  copy->var = std::move(mapped_var).ValueOrUnchanged(std::move(copy->var));
  copy->value = std::move(mapped_value).ValueOrUnchanged(std::move(copy->value));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny BindMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  BindNode* self = const_cast<BindNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BindNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Var>, mapped_var,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MaybeInplaceMutateIfUniqueExpected(self->var);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  if (mapped_var.UnchangedOrSameAs(self->var) && mapped_value.UnchangedOrSameAs(self->value)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_var.IsUnchanged()) self->var = std::move(mapped_var).ValueUnchecked();
  if (!mapped_value.IsUnchanged()) self->value = std::move(mapped_value).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny AttrStmtVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: attr_key
  const AttrStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AttrStmtNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->node));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny AttrStmtMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attr_key
  const AttrStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AttrStmtNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Any>, mapped_node,
                                    mutator->MutateExpected(self->node));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value,
                                    mutator->MutateExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body));
  if (mapped_node.UnchangedOrSameAs(self->node) && mapped_value.UnchangedOrSameAs(self->value) &&
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<AttrStmtNode> copy = ffi::make_object<AttrStmtNode>(*self);
  copy->node = std::move(mapped_node).ValueOrUnchanged(std::move(copy->node));
  copy->value = std::move(mapped_value).ValueOrUnchanged(std::move(copy->value));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny AttrStmtMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                     ffi::AnyView value) noexcept {
  // skips: attr_key
  AttrStmtNode* self = const_cast<AttrStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AttrStmtNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Any>, mapped_node,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->node));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->body));
  if (mapped_node.UnchangedOrSameAs(self->node) && mapped_value.UnchangedOrSameAs(self->value) &&
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_node.IsUnchanged()) self->node = std::move(mapped_node).ValueUnchecked();
  if (!mapped_value.IsUnchanged()) self->value = std::move(mapped_value).ValueUnchecked();
  if (!mapped_body.IsUnchanged()) self->body = std::move(mapped_body).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny AssertStmtVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: error_kind and message_parts, which are constant assertion metadata.
  const AssertStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AssertStmtNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->condition));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny AssertStmtMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: error_kind and message_parts, which are constant assertion metadata.
  const AssertStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AssertStmtNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_condition,
                                    mutator->MutateExpected(self->condition));
  if (mapped_condition.UnchangedOrSameAs(self->condition)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<AssertStmtNode> copy = ffi::make_object<AssertStmtNode>(*self);
  copy->condition = std::move(mapped_condition).ValueOrUnchanged(std::move(copy->condition));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny AssertStmtMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                       ffi::AnyView value) noexcept {
  // skips: error_kind and message_parts, which are constant assertion metadata.
  AssertStmtNode* self = const_cast<AssertStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AssertStmtNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_condition,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->condition));
  if (mapped_condition.UnchangedOrSameAs(self->condition)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_condition.IsUnchanged())
    self->condition = std::move(mapped_condition).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny ForVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
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
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny ForMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
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
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<ForNode> copy = ffi::make_object<ForNode>(*self);
  copy->loop_var = std::move(mapped_loop_var).ValueOrUnchanged(std::move(copy->loop_var));
  copy->min = std::move(mapped_min).ValueOrUnchanged(std::move(copy->min));
  copy->extent = std::move(mapped_extent).ValueOrUnchanged(std::move(copy->extent));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  copy->thread_binding =
      std::move(mapped_thread_binding).ValueOrUnchanged(std::move(copy->thread_binding));
  copy->step = std::move(mapped_step).ValueOrUnchanged(std::move(copy->step));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny ForMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: kind and constant annotations; unlike SBlock annotations, these carry no expressions.
  ForNode* self = const_cast<ForNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ForNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimVar>, mapped_loop_var,
      mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
        return mutator->MaybeInplaceMutateIfUniqueExpected(self->loop_var);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_min,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->min));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_extent,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->extent));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->body));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Optional<IterVar>>, mapped_thread_binding,
      mutator->MaybeInplaceMutateIfUniqueExpected(self->thread_binding));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<PrimExpr>>, mapped_step,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->step));
  if (mapped_loop_var.UnchangedOrSameAs(self->loop_var) &&
      mapped_min.UnchangedOrSameAs(self->min) && mapped_extent.UnchangedOrSameAs(self->extent) &&
      mapped_body.UnchangedOrSameAs(self->body) &&
      mapped_thread_binding.UnchangedOrSameAs(self->thread_binding) &&
      mapped_step.UnchangedOrSameAs(self->step)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_loop_var.IsUnchanged()) self->loop_var = std::move(mapped_loop_var).ValueUnchecked();
  if (!mapped_min.IsUnchanged()) self->min = std::move(mapped_min).ValueUnchecked();
  if (!mapped_extent.IsUnchanged()) self->extent = std::move(mapped_extent).ValueUnchecked();
  if (!mapped_body.IsUnchanged()) self->body = std::move(mapped_body).ValueUnchecked();
  if (!mapped_thread_binding.IsUnchanged()) {
    self->thread_binding = std::move(mapped_thread_binding).ValueUnchecked();
  }
  if (!mapped_step.IsUnchanged()) self->step = std::move(mapped_step).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny WhileVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const WhileNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const WhileNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny WhileMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const WhileNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const WhileNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_condition,
                                    mutator->MutateExpected(self->condition));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body));
  if (mapped_condition.UnchangedOrSameAs(self->condition) &&
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<WhileNode> copy = ffi::make_object<WhileNode>(*self);
  copy->condition = std::move(mapped_condition).ValueOrUnchanged(std::move(copy->condition));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny WhileMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  WhileNode* self = const_cast<WhileNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const WhileNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_condition,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->condition));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->body));
  if (mapped_condition.UnchangedOrSameAs(self->condition) &&
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_condition.IsUnchanged())
    self->condition = std::move(mapped_condition).ValueUnchecked();
  if (!mapped_body.IsUnchanged()) self->body = std::move(mapped_body).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny ReturnVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const ReturnNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ReturnNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny ReturnMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const ReturnNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ReturnNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MutateExpected(self->value));
  if (mapped_value.UnchangedOrSameAs(self->value)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<ReturnNode> copy = ffi::make_object<ReturnNode>(*self);
  copy->value = std::move(mapped_value).ValueOrUnchanged(std::move(copy->value));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny ReturnMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                   ffi::AnyView value) noexcept {
  ReturnNode* self = const_cast<ReturnNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ReturnNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  if (mapped_value.UnchangedOrSameAs(self->value)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_value.IsUnchanged()) self->value = std::move(mapped_value).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny BreakVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny BreakMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny BreakMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny ContinueVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny ContinueMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny ContinueMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny DeclBufferVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const DeclBufferNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DeclBufferNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->buffer); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->data));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny DeclBufferMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const DeclBufferNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DeclBufferNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<BufferVar>, mapped_buffer,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->buffer);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_data,
                                    mutator->MutateExpected(self->data));
  if (mapped_buffer.UnchangedOrSameAs(self->buffer) && mapped_data.UnchangedOrSameAs(self->data)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<DeclBufferNode> copy = ffi::make_object<DeclBufferNode>(*self);
  copy->buffer = std::move(mapped_buffer).ValueOrUnchanged(std::move(copy->buffer));
  copy->data = std::move(mapped_data).ValueOrUnchanged(std::move(copy->data));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny DeclBufferMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                       ffi::AnyView value) noexcept {
  DeclBufferNode* self = const_cast<DeclBufferNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DeclBufferNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<BufferVar>, mapped_buffer,
      mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
        return mutator->MaybeInplaceMutateIfUniqueExpected(self->buffer);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_data,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->data));
  if (mapped_buffer.UnchangedOrSameAs(self->buffer) && mapped_data.UnchangedOrSameAs(self->data)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_buffer.IsUnchanged()) self->buffer = std::move(mapped_buffer).ValueUnchecked();
  if (!mapped_data.IsUnchanged()) self->data = std::move(mapped_data).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny AllocBufferVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: constant annotations; unlike SBlock annotations, these carry no expressions.
  const AllocBufferNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AllocBufferNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->buffer); }));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny AllocBufferMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: constant annotations; unlike SBlock annotations, these carry no expressions.
  const AllocBufferNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AllocBufferNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<BufferVar>, mapped_buffer,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->buffer);
                                    }));
  if (mapped_buffer.UnchangedOrSameAs(self->buffer)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<AllocBufferNode> copy = ffi::make_object<AllocBufferNode>(*self);
  copy->buffer = std::move(mapped_buffer).ValueOrUnchanged(std::move(copy->buffer));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny AllocBufferMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                        ffi::AnyView value) noexcept {
  // skips: constant annotations; unlike SBlock annotations, these carry no expressions.
  AllocBufferNode* self = const_cast<AllocBufferNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AllocBufferNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<BufferVar>, mapped_buffer,
      mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
        return mutator->MaybeInplaceMutateIfUniqueExpected(self->buffer);
      }));
  if (mapped_buffer.UnchangedOrSameAs(self->buffer)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_buffer.IsUnchanged()) self->buffer = std::move(mapped_buffer).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny SeqStmtVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const SeqStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->seq));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

bool IsSeqStmtNoOp(ffi::AnyView stmt) {
  const auto* evaluate = stmt.as<EvaluateNode>();
  const auto* value = evaluate == nullptr ? nullptr : evaluate->value.as<IntImmNode>();
  return value != nullptr && value->value == 0;
}

// Move the non-None slots to the front, preserving order; return the compacted size.
size_t SeqStmtCompactNop(ffi::Any* begin, size_t current_size) {
  size_t write = 0;
  for (size_t read = 0; read < current_size; ++read) {
    if (begin[read].type_index() == ffi::TypeIndex::kTVMFFINone) {
      continue;
    }
    if (write != read) {
      begin[write] = std::move(begin[read]);
    }
    ++write;
  }
  return write;
}

// Expand each nested SeqStmt in [0, current_size) into its final position in [0, target_size).
// Walk backward so a destination is never below its source; requires every nested SeqStmt to be
// non-empty and target_size slots to exist.
void SeqStmtExpandNested(ffi::Any* begin, size_t current_size, size_t target_size) {
  size_t write = target_size;
  for (size_t read = current_size; read-- > 0;) {
    if (const auto* nested = begin[read].as<SeqStmtNode>()) {
      for (size_t i = nested->seq.size(); i-- > 0;) {
        begin[--write] = ffi::Any(nested->seq[i]);
      }
    } else {
      --write;
      if (write != read) {
        begin[write] = std::move(begin[read]);
      }
    }
  }
}

TVMFFIAny MutateSeqStmtChanged(ffi::StructuralMutatorObj* mutator, const SeqStmtNode* self,
                               size_t index, Stmt mapped) noexcept {
  const size_t size = self->seq.size();
  std::vector<Stmt> results;
  results.reserve(size);
  results.assign(self->seq.begin(), self->seq.begin() + index);
  auto append = [&](Stmt stmt) {
    if (IsSeqStmtNoOp(stmt)) {
      return;
    }
    if (const auto* nested = stmt.as<SeqStmtNode>()) {
      for (const Stmt& child : nested->seq) {
        results.emplace_back(child);
      }
    } else {
      results.emplace_back(std::move(stmt));
    }
  };
  append(std::move(mapped));
  for (size_t i = index + 1; i < size; ++i) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, element,
                                      mutator->MutateExpected(self->seq[i]));
    append(element.IsUnchanged() ? self->seq[i] : std::move(element).ValueUnchecked());
  }
  if (results.empty()) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(Evaluate(0)));
  }
  if (results.size() == 1) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(results[0])));
  }
  ffi::ObjectPtr<SeqStmtNode> copy = ffi::make_object<SeqStmtNode>(*self);
  copy->seq = ffi::Array<Stmt>(std::make_move_iterator(results.begin()),
                               std::make_move_iterator(results.end()));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny MutateSeqStmtRaw(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const SeqStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(value);
  const size_t size = self->seq.size();
  for (size_t i = 0; i < size; ++i) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped,
                                      mutator->MutateExpected(self->seq[i]));
    if (!mapped.UnchangedOrSameAs(self->seq[i])) {
      return MutateSeqStmtChanged(mutator, self, i, std::move(mapped).ValueUnchecked());
    }
  }
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny MaybeInplaceMutateSeqStmtRaw(ffi::StructuralMutatorObj* mutator,
                                       ffi::AnyView value) noexcept {
  SeqStmtNode* self = const_cast<SeqStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(value));
  // The engine establishes ownership of the SeqStmt, but seq is a field and needs its own check.
  if (!self->seq.unique()) {
    return MutateSeqStmtRaw(mutator, value);
  }
  ffi::ArrayObj* seq = self->seq.GetArrayObj();
  ffi::Any* slots = const_cast<ffi::Any*>(seq->begin());
  const size_t size = seq->size();
  size_t delete_count = 0;
  size_t nested_extra = 0;

  // Pass 1: rebuild each element in place and classify the final slot contents.
  for (size_t i = 0; i < size; ++i) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(slots[i]));
    if (!mapped.UnchangedOrSameAs(slots[i].cast<Stmt>())) {
      slots[i] = ffi::Any(std::move(mapped).ValueUnchecked());
    }
    if (IsSeqStmtNoOp(slots[i])) {
      slots[i] = ffi::Any();  // A None slot is the delete marker consumed by compaction.
      ++delete_count;
      continue;
    }
    if (const auto* nested = slots[i].as<SeqStmtNode>()) {
      if (nested->seq.empty()) {
        slots[i] = ffi::Any();
        ++delete_count;
      } else {
        nested_extra += nested->seq.size() - 1;
      }
    }
  }

  const size_t final_size = size - delete_count + nested_extra;
  if (delete_count == 0 && nested_extra == 0) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }

  // Splice in place when the result fits: compact out None slots, resize once, then expand.
  if (final_size <= seq->SeqBaseObj::capacity()) {
    size_t compacted = SeqStmtCompactNop(slots, size);
    seq->resize(final_size);
    SeqStmtExpandNested(slots, compacted, final_size);
  } else {
    // Splice beyond capacity: flatten into one exact-size array with pointer moves.
    std::vector<Stmt> results;
    results.reserve(final_size);
    for (size_t i = 0; i < size; ++i) {
      if (slots[i].type_index() == ffi::TypeIndex::kTVMFFINone) {
        continue;
      }
      if (const auto* nested = slots[i].as<SeqStmtNode>()) {
        for (const Stmt& child : nested->seq) {
          results.emplace_back(child);
        }
      } else {
        results.emplace_back(
            ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<Stmt>(std::move(slots[i])));
      }
    }
    self->seq = ffi::Array<Stmt>(std::make_move_iterator(results.begin()),
                                 std::make_move_iterator(results.end()));
    seq = self->seq.GetArrayObj();
  }

  if (final_size == 0) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(Evaluate(0)));
  }
  if (final_size == 1) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(seq->begin()[0].cast<Stmt>()));
  }
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny SeqStmtMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  return MutateSeqStmtRaw(mutator, value);
}

TVMFFIAny SeqStmtMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                    ffi::AnyView value) noexcept {
  return MaybeInplaceMutateSeqStmtRaw(mutator, value);
}

TVMFFIAny IfThenElseVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const IfThenElseNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfThenElseNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->then_case));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->else_case));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny IfThenElseMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
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
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<IfThenElseNode> copy = ffi::make_object<IfThenElseNode>(*self);
  copy->condition = std::move(mapped_condition).ValueOrUnchanged(std::move(copy->condition));
  copy->then_case = std::move(mapped_then_case).ValueOrUnchanged(std::move(copy->then_case));
  copy->else_case = std::move(mapped_else_case).ValueOrUnchanged(std::move(copy->else_case));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny IfThenElseMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                       ffi::AnyView value) noexcept {
  IfThenElseNode* self = const_cast<IfThenElseNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfThenElseNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_condition,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->condition));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_then_case,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->then_case));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Stmt>>, mapped_else_case,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->else_case));
  if (mapped_condition.UnchangedOrSameAs(self->condition) &&
      mapped_then_case.UnchangedOrSameAs(self->then_case) &&
      mapped_else_case.UnchangedOrSameAs(self->else_case)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_condition.IsUnchanged())
    self->condition = std::move(mapped_condition).ValueUnchecked();
  if (!mapped_then_case.IsUnchanged())
    self->then_case = std::move(mapped_then_case).ValueUnchecked();
  if (!mapped_else_case.IsUnchanged())
    self->else_case = std::move(mapped_else_case).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny EvaluateVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const EvaluateNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny EvaluateMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const EvaluateNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MutateExpected(self->value));
  if (mapped_value.UnchangedOrSameAs(self->value)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<EvaluateNode> copy = ffi::make_object<EvaluateNode>(*self);
  copy->value = std::move(mapped_value).ValueOrUnchanged(std::move(copy->value));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny EvaluateMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                     ffi::AnyView value) noexcept {
  EvaluateNode* self = const_cast<EvaluateNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  if (mapped_value.UnchangedOrSameAs(self->value)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_value.IsUnchanged()) self->value = std::move(mapped_value).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny BufferStoreVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const BufferStoreNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferStoreNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->buffer));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->indices));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny BufferStoreMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
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
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<BufferStoreNode> copy = ffi::make_object<BufferStoreNode>(*self);
  copy->buffer = std::move(mapped_buffer).ValueOrUnchanged(std::move(copy->buffer));
  copy->value = std::move(mapped_value).ValueOrUnchanged(std::move(copy->value));
  copy->indices = std::move(mapped_indices).ValueOrUnchanged(std::move(copy->indices));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny BufferStoreMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                        ffi::AnyView value) noexcept {
  BufferStoreNode* self = const_cast<BufferStoreNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferStoreNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<BufferVar>, mapped_buffer,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->buffer));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_indices,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->indices));
  if (mapped_buffer.UnchangedOrSameAs(self->buffer) &&
      mapped_value.UnchangedOrSameAs(self->value) &&
      mapped_indices.UnchangedOrSameAs(self->indices)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_buffer.IsUnchanged()) self->buffer = std::move(mapped_buffer).ValueUnchecked();
  if (!mapped_value.IsUnchanged()) self->value = std::move(mapped_value).ValueUnchecked();
  if (!mapped_indices.IsUnchanged()) self->indices = std::move(mapped_indices).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

ffi::ObjectRef RealizeBufferRegionSubscript(Expr value, SubscriptSlice slice, Span span) {
  BufferRegion source = value.as_or_throw<BufferRegion>();
  TVM_FFI_CHECK_LE(slice.size(), source->region.size(), IndexError)
      << "Too many indices for a " << source->region.size() << "-dimensional buffer region";

  bool all_points = slice.size() == source->region.size();
  for (const auto& item : slice) {
    if (auto descriptor = item.as<ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>,
                                             ffi::Optional<PrimExpr>>>()) {
      all_points = false;
      ffi::Optional<PrimExpr> step = descriptor.value().get<2>();
      TVM_FFI_CHECK(!step.has_value() || is_one(step.value()), ValueError)
          << "BufferRegion slices with a non-unit step are not supported";
    }
  }

  if (all_points) {
    ffi::Array<PrimExpr> indices;
    indices.reserve(slice.size());
    for (size_t i = 0; i < slice.size(); ++i) {
      indices.push_back(source->region[i]->min + slice[i].as<PrimExpr>().value());
    }
    return BufferLoad(source->buffer, indices, span);
  }

  arith::Analyzer analyzer;
  ffi::Array<Range> region;
  region.reserve(source->region.size());
  for (size_t i = 0; i < slice.size(); ++i) {
    const Range& old_range = source->region[i];
    if (auto point = slice[i].as<PrimExpr>()) {
      PrimExpr new_min = old_range->min + point.value();
      region.push_back(Range::FromMinExtent(new_min, IntImm(point.value().ty(), 1)));
    } else {
      auto descriptor = slice[i]
                            .as<ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>,
                                           ffi::Optional<PrimExpr>>>()
                            .value();
      PrimExpr start = descriptor.get<0>().value_or(IntImm(old_range->extent.ty(), 0));
      PrimExpr stop = descriptor.get<1>().value_or(old_range->extent);
      region.push_back(
          Range::FromMinExtent(old_range->min + start, analyzer->Simplify(stop - start)));
    }
  }
  for (size_t i = slice.size(); i < source->region.size(); ++i) {
    region.push_back(source->region[i]);
  }
  return BufferRegion(source->buffer, region, span);
}

TVMFFIAny BufferRegionTypeVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny BufferRegionTypeMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny BufferRegionTypeMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny BufferRegionVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const BufferRegionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferRegionNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->buffer));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->region));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny BufferRegionMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const BufferRegionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferRegionNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<BufferVar>, mapped_buffer,
                                    mutator->MutateExpected(self->buffer));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Range>>, mapped_region,
                                    mutator->MutateExpected(self->region));
  if (mapped_buffer.UnchangedOrSameAs(self->buffer) &&
      mapped_region.UnchangedOrSameAs(self->region)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<BufferRegionNode> copy = ffi::make_object<BufferRegionNode>(*self);
  copy->buffer = std::move(mapped_buffer).ValueOrUnchanged(std::move(copy->buffer));
  copy->region = std::move(mapped_region).ValueOrUnchanged(std::move(copy->region));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny BufferRegionMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                         ffi::AnyView value) noexcept {
  BufferRegionNode* self = const_cast<BufferRegionNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferRegionNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<BufferVar>, mapped_buffer,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->buffer));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Range>>, mapped_region,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->region));
  if (mapped_buffer.UnchangedOrSameAs(self->buffer) &&
      mapped_region.UnchangedOrSameAs(self->region)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_buffer.IsUnchanged()) self->buffer = std::move(mapped_buffer).ValueUnchecked();
  if (!mapped_region.IsUnchanged()) self->region = std::move(mapped_region).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny MatchBufferRegionVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const MatchBufferRegionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MatchBufferRegionNode>(
          value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->buffer); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->source));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny MatchBufferRegionMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const MatchBufferRegionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MatchBufferRegionNode>(
          value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<BufferVar>, mapped_buffer,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->buffer);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<BufferRegion>, mapped_source,
                                    mutator->MutateExpected(self->source));
  if (mapped_buffer.UnchangedOrSameAs(self->buffer) &&
      mapped_source.UnchangedOrSameAs(self->source)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<MatchBufferRegionNode> copy = ffi::make_object<MatchBufferRegionNode>(*self);
  copy->buffer = std::move(mapped_buffer).ValueOrUnchanged(std::move(copy->buffer));
  copy->source = std::move(mapped_source).ValueOrUnchanged(std::move(copy->source));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny MatchBufferRegionMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                              ffi::AnyView value) noexcept {
  MatchBufferRegionNode* self = const_cast<MatchBufferRegionNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MatchBufferRegionNode>(
          value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<BufferVar>, mapped_buffer,
      mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
        return mutator->MaybeInplaceMutateIfUniqueExpected(self->buffer);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<BufferRegion>, mapped_source,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->source));
  if (mapped_buffer.UnchangedOrSameAs(self->buffer) &&
      mapped_source.UnchangedOrSameAs(self->source)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_buffer.IsUnchanged()) self->buffer = std::move(mapped_buffer).ValueUnchecked();
  if (!mapped_source.IsUnchanged()) self->source = std::move(mapped_source).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny SBlockVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: name_hint
  const SBlockNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->iter_vars));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->reads));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->writes));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->alloc_buffers); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->match_buffers));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->annotations));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->init));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny SBlockMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: name_hint
  const SBlockNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<IterVar>>, mapped_iter_vars,
                                    mutator->MutateExpected(self->iter_vars));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<BufferRegion>>, mapped_reads,
                                    mutator->MutateExpected(self->reads));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<BufferRegion>>, mapped_writes,
                                    mutator->MutateExpected(self->writes));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<BufferVar>>, mapped_alloc_buffers,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->alloc_buffers);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<MatchBufferRegion>>,
                                    mapped_match_buffers,
                                    mutator->MutateExpected(self->match_buffers));
  using AnnotationMap = ffi::Map<ffi::String, ffi::Any>;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<AnnotationMap>, mapped_annotations,
                                    mutator->MutateExpected(self->annotations));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Stmt>>, mapped_init,
                                    mutator->MutateExpected(self->init));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body));
  if (mapped_iter_vars.UnchangedOrSameAs(self->iter_vars) &&
      mapped_reads.UnchangedOrSameAs(self->reads) &&
      mapped_writes.UnchangedOrSameAs(self->writes) &&
      mapped_alloc_buffers.UnchangedOrSameAs(self->alloc_buffers) &&
      mapped_match_buffers.UnchangedOrSameAs(self->match_buffers) &&
      mapped_annotations.UnchangedOrSameAs(self->annotations) &&
      mapped_init.UnchangedOrSameAs(self->init) && mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<SBlockNode> copy = ffi::make_object<SBlockNode>(*self);
  copy->iter_vars = std::move(mapped_iter_vars).ValueOrUnchanged(std::move(copy->iter_vars));
  copy->reads = std::move(mapped_reads).ValueOrUnchanged(std::move(copy->reads));
  copy->writes = std::move(mapped_writes).ValueOrUnchanged(std::move(copy->writes));
  copy->alloc_buffers =
      std::move(mapped_alloc_buffers).ValueOrUnchanged(std::move(copy->alloc_buffers));
  copy->match_buffers =
      std::move(mapped_match_buffers).ValueOrUnchanged(std::move(copy->match_buffers));
  copy->annotations = std::move(mapped_annotations).ValueOrUnchanged(std::move(copy->annotations));
  copy->init = std::move(mapped_init).ValueOrUnchanged(std::move(copy->init));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny SBlockMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                   ffi::AnyView value) noexcept {
  // skips: name_hint
  SBlockNode* self = const_cast<SBlockNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<IterVar>>, mapped_iter_vars,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->iter_vars));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<BufferRegion>>, mapped_reads,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->reads));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<BufferRegion>>, mapped_writes,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->writes));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<BufferVar>>, mapped_alloc_buffers,
      mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
        return mutator->MaybeInplaceMutateIfUniqueExpected(self->alloc_buffers);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<MatchBufferRegion>>, mapped_match_buffers,
      mutator->MaybeInplaceMutateIfUniqueExpected(self->match_buffers));
  using AnnotationMap = ffi::Map<ffi::String, ffi::Any>;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<AnnotationMap>, mapped_annotations,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->annotations));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Stmt>>, mapped_init,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->init));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->body));
  if (mapped_iter_vars.UnchangedOrSameAs(self->iter_vars) &&
      mapped_reads.UnchangedOrSameAs(self->reads) &&
      mapped_writes.UnchangedOrSameAs(self->writes) &&
      mapped_alloc_buffers.UnchangedOrSameAs(self->alloc_buffers) &&
      mapped_match_buffers.UnchangedOrSameAs(self->match_buffers) &&
      mapped_annotations.UnchangedOrSameAs(self->annotations) &&
      mapped_init.UnchangedOrSameAs(self->init) && mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_iter_vars.IsUnchanged())
    self->iter_vars = std::move(mapped_iter_vars).ValueUnchecked();
  if (!mapped_reads.IsUnchanged()) self->reads = std::move(mapped_reads).ValueUnchecked();
  if (!mapped_writes.IsUnchanged()) self->writes = std::move(mapped_writes).ValueUnchecked();
  if (!mapped_alloc_buffers.IsUnchanged()) {
    self->alloc_buffers = std::move(mapped_alloc_buffers).ValueUnchecked();
  }
  if (!mapped_match_buffers.IsUnchanged()) {
    self->match_buffers = std::move(mapped_match_buffers).ValueUnchecked();
  }
  if (!mapped_annotations.IsUnchanged())
    self->annotations = std::move(mapped_annotations).ValueUnchecked();
  if (!mapped_init.IsUnchanged()) self->init = std::move(mapped_init).ValueUnchecked();
  if (!mapped_body.IsUnchanged()) self->body = std::move(mapped_body).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny ScopeIdDefStmtVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const ScopeIdDefStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ScopeIdDefStmtNode>(value);
  // ScopeIdDef reflection marks only def_ids as definitions; its extent fields remain uses.
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->def));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny ScopeIdDefStmtMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const ScopeIdDefStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ScopeIdDefStmtNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ScopeIdDef>, mapped_def,
                                    mutator->MutateExpected(self->def));
  if (mapped_def.UnchangedOrSameAs(self->def)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<ScopeIdDefStmtNode> copy = ffi::make_object<ScopeIdDefStmtNode>(*self);
  copy->def = std::move(mapped_def).ValueOrUnchanged(std::move(copy->def));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny ScopeIdDefStmtMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                           ffi::AnyView value) noexcept {
  ScopeIdDefStmtNode* self = const_cast<ScopeIdDefStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ScopeIdDefStmtNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ScopeIdDef>, mapped_def,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->def));
  if (mapped_def.UnchangedOrSameAs(self->def)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_def.IsUnchanged()) self->def = std::move(mapped_def).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny SBlockRealizeVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const SBlockRealizeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockRealizeNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->iter_values));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->predicate));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->block));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny SBlockRealizeMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const SBlockRealizeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockRealizeNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_iter_values,
                                    mutator->MutateExpected(self->iter_values));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_predicate,
                                    mutator->MutateExpected(self->predicate));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SBlock>, mapped_block,
                                    mutator->MutateExpected(self->block));
  if (mapped_iter_values.UnchangedOrSameAs(self->iter_values) &&
      mapped_predicate.UnchangedOrSameAs(self->predicate) &&
      mapped_block.UnchangedOrSameAs(self->block)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<SBlockRealizeNode> copy = ffi::make_object<SBlockRealizeNode>(*self);
  copy->iter_values = std::move(mapped_iter_values).ValueOrUnchanged(std::move(copy->iter_values));
  copy->predicate = std::move(mapped_predicate).ValueOrUnchanged(std::move(copy->predicate));
  copy->block = std::move(mapped_block).ValueOrUnchanged(std::move(copy->block));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny SBlockRealizeMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                          ffi::AnyView value) noexcept {
  SBlockRealizeNode* self = const_cast<SBlockRealizeNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockRealizeNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_iter_values,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->iter_values));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_predicate,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->predicate));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SBlock>, mapped_block,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->block));
  if (mapped_iter_values.UnchangedOrSameAs(self->iter_values) &&
      mapped_predicate.UnchangedOrSameAs(self->predicate) &&
      mapped_block.UnchangedOrSameAs(self->block)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_iter_values.IsUnchanged())
    self->iter_values = std::move(mapped_iter_values).ValueUnchecked();
  if (!mapped_predicate.IsUnchanged())
    self->predicate = std::move(mapped_predicate).ValueUnchecked();
  if (!mapped_block.IsUnchanged()) self->block = std::move(mapped_block).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
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
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BindVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BindMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BindMaybeInplaceMutate));

  refl::GlobalDef().def("tirx.Bind",
                        [](Var var, Expr value, Span span) { return Bind(var, value, span); });
}

// AttrStmt
AttrStmt::AttrStmt(ffi::Any node, ffi::String attr_key, PrimExpr value, Stmt body, Span span) {
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
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&AttrStmtVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&AttrStmtMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&AttrStmtMaybeInplaceMutate));

  refl::GlobalDef().def("tirx.AttrStmt",
                        [](Any node, ffi::String attr_key, PrimExpr value, Stmt body, Span span) {
                          return AttrStmt(node, attr_key, value, body, span);
                        });
}

// AssertStmt
AssertStmt::AssertStmt(PrimExpr condition, prim::StringImm error_kind,
                       ffi::Array<prim::StringImm> message_parts, Span span) {
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
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&AssertStmtVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&AssertStmtMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&AssertStmtMaybeInplaceMutate));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.AssertStmt",
      [](PrimExpr condition, prim::StringImm error_kind, ffi::Array<prim::StringImm> message_parts,
         Span span) { return AssertStmt(condition, error_kind, message_parts, span); });
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
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&ForVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&ForMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&ForMaybeInplaceMutate));

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
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&WhileVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&WhileMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&WhileMaybeInplaceMutate));

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
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&ReturnVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&ReturnMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&ReturnMaybeInplaceMutate));

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
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BreakVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BreakMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BreakMaybeInplaceMutate));

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
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&ContinueVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&ContinueMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&ContinueMaybeInplaceMutate));

  refl::GlobalDef().def("tirx.Continue", [](Span span) { return Continue(span); });
}

// DeclBuffer
DeclBuffer::DeclBuffer(BufferVar buffer, Expr data, Span span) {
  // Enforce storage scope rules for DeclBuffer.
  std::string scope = static_cast<std::string>(buffer.scope());
  if (scope.empty()) {
    scope = "global";
  }
  if (scope == "tmem") {
    TVM_FFI_ICHECK_EQ(buffer->allocated_addr.size(), 1U)
        << "ValueError: For `tmem` scope, DeclBuffer requires exactly one `allocated_addr` "
           "PrimExpr";
  } else if (scope == "global" || scope == "shared" || scope == "shared.dyn" || scope == "local") {
    TVM_FFI_ICHECK(buffer->allocated_addr.empty())
        << "ValueError: For `" << scope << "` scope, DeclBuffer does not accept `allocated_addr`";
  }
  ffi::ObjectPtr<DeclBufferNode> node = ffi::make_object<DeclBufferNode>();
  node->buffer = std::move(buffer);
  node->data = std::move(data);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  DeclBufferNode::RegisterReflection();
  refl::TypeAttrDef<DeclBufferNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&DeclBufferVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&DeclBufferMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&DeclBufferMaybeInplaceMutate));

  refl::GlobalDef().def("tirx.DeclBuffer", [](BufferVar buffer, Expr data, Span span) {
    return DeclBuffer(buffer, data, span);
  });
}

// AllocBuffer
AllocBuffer::AllocBuffer(BufferVar buffer, ffi::Map<ffi::String, Any> annotations, Span span) {
  ffi::ObjectPtr<AllocBufferNode> node = ffi::make_object<AllocBufferNode>();
  node->buffer = std::move(buffer);
  node->annotations = std::move(annotations);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  AllocBufferNode::RegisterReflection();
  refl::TypeAttrDef<AllocBufferNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&AllocBufferVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&AllocBufferMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&AllocBufferMaybeInplaceMutate));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.AllocBuffer",
      [](BufferVar buffer, ffi::Optional<ffi::Map<ffi::String, Any>> annotations, Span span) {
        return AllocBuffer(buffer, annotations.value_or(ffi::Map<ffi::String, Any>()), span);
      });
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
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&SeqStmtVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&SeqStmtMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&SeqStmtMaybeInplaceMutate));

  refl::GlobalDef().def("tirx.SeqStmt", [](ffi::Array<Stmt> seq, Span span) {
    return SeqStmt(std::move(seq), span);
  });
}

// A direct child that mutates into a SeqStmt is spliced into this sequence in the same pass,
// matching StmtMutator's normalized result without a second scan and allocation. A well-formed
// mapped SeqStmt already holds no SeqStmt child, so this single-level splice is sufficient and an
// unchanged element never needs splicing.

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
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&IfThenElseVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&IfThenElseMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&IfThenElseMaybeInplaceMutate));

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
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&EvaluateVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&EvaluateMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&EvaluateMaybeInplaceMutate));

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
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BufferStoreVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BufferStoreMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BufferStoreMaybeInplaceMutate));

  refl::GlobalDef().def("tirx.BufferStore",
                        [](BufferVar buffer, PrimExpr value, ffi::Array<PrimExpr> indices,
                           Span span) { return BufferStore(buffer, value, indices, span); });
}

// BufferRegion
BufferRegionType::BufferRegionType() : Type(ffi::UnsafeInit{}) {
  static ffi::ObjectPtr<BufferRegionTypeNode> singleton = ffi::make_object<BufferRegionTypeNode>();
  data_ = singleton;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BufferRegionTypeNode::RegisterReflection();
  refl::TypeAttrDef<BufferRegionTypeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BufferRegionTypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BufferRegionTypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BufferRegionTypeMaybeInplaceMutate))
      .def("__subscript_expr_realize__", RealizeBufferRegionSubscript);

  refl::GlobalDef().def("tirx.BufferRegionType", []() { return BufferRegionType(); });
}

BufferRegion::BufferRegion(BufferVar buffer, ffi::Array<Range> region, Span span) {
  TVM_FFI_ICHECK_EQ(buffer->shape.size(), region.size())
      << "The dimension between " << buffer << " and region " << region
      << " mismatched, the buffer is " << buffer;
  ffi::ObjectPtr<BufferRegionNode> node = ffi::make_object<BufferRegionNode>();
  node->ty = BufferRegionType();
  node->span = std::move(span);
  node->buffer = std::move(buffer);
  node->region = std::move(region);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BufferRegionNode::RegisterReflection();
  refl::TypeAttrDef<BufferRegionNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BufferRegionVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BufferRegionMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BufferRegionMaybeInplaceMutate));

  refl::GlobalDef().def("tirx.BufferRegion", [](BufferVar buffer, ffi::Array<Range> region) {
    return BufferRegion(buffer, region);
  });
}

BufferRegion BufferRegion::FullRegion(BufferVar buffer) {
  ffi::Array<Range> region;
  for (PrimExpr extent : buffer->shape) {
    region.push_back(Range::FromMinExtent(0, extent));
  }
  return BufferRegion(buffer, region);
}

BufferRegion BufferRegion::FromPoint(BufferVar buffer, ffi::Array<PrimExpr> indices) {
  ffi::Array<Range> region;
  for (const PrimExpr& index : indices) {
    if (const prim::RampNode* ramp_index = index.as<prim::RampNode>()) {
      region.push_back(
          Range::FromMinExtent(ramp_index->base, ramp_index->stride * ramp_index->lanes));
    } else {
      region.push_back(Range::FromMinExtent(index, MakeConst(index.ty(), 1)));
    }
  }
  return BufferRegion(buffer, region);
}

// MatchBufferRegion
MatchBufferRegion::MatchBufferRegion(BufferVar buffer, BufferRegion source) {
  const BufferVar& source_buffer = source->buffer;
  arith::Analyzer analyzer;
  // Check scope and dtype
  TVM_FFI_ICHECK_EQ(buffer.scope(), source_buffer.scope())
      << "MatchBuffer " << buffer << " scope mismatch:" << buffer.scope() << " vs. "
      << source_buffer.scope();
  TVM_FFI_ICHECK_EQ(buffer->dtype, source_buffer->dtype)
      << "MatchBuffer " << buffer << " data type mismatch:" << buffer->dtype << " vs. "
      << source_buffer->dtype;

  // Check data_alignment
  TVM_FFI_ICHECK(source_buffer->data_alignment % buffer->data_alignment == 0)
      << "Trying to match buffer to another one with lower alignment requirement "
      << " required alignment=" << buffer->data_alignment
      << ", provided alignment=" << source_buffer->data_alignment;

  // Validate shape
  TVM_FFI_ICHECK(source->region.size() >= buffer->shape.size())
      << "Dimension of source ffi::Array<Range> expected to be larger or equal than target buffer "
         "shape, but "
         "got "
      << source->region.size() << " vs. " << buffer->shape.size();
  size_t offset = source->region.size() - buffer->shape.size();
  for (size_t i = 0; i < offset; ++i) {
    TVM_FFI_ICHECK(analyzer->CanProve(source->region[i]->extent == 1))
        << "The higher dimension should be 1, but got " << source->region[i]->extent << ".";
  }
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    const Range& source_range = source->region[i + offset];
    const PrimExpr& buffer_shape = buffer->shape[i];
    if (!buffer_shape.as<PrimVar>()) {
      TVM_FFI_ICHECK(analyzer->CanProve(source_range->extent == buffer_shape))
          << "The dimension mismatched between source region and target buffer shape, got "
          << source_range->extent << " vs. " << buffer_shape << ".";
    }
  }
  // Note that we do not check elem_offset and strides in this function

  // Construction
  ffi::ObjectPtr<MatchBufferRegionNode> node = ffi::make_object<MatchBufferRegionNode>();
  node->buffer = std::move(buffer);
  node->source = std::move(source);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MatchBufferRegionNode::RegisterReflection();
  refl::TypeAttrDef<MatchBufferRegionNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&MatchBufferRegionVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&MatchBufferRegionMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&MatchBufferRegionMaybeInplaceMutate));

  refl::GlobalDef().def("tirx.MatchBufferRegion", [](BufferVar buffer, BufferRegion source) {
    return MatchBufferRegion(buffer, source);
  });
}

// Block
SBlock::SBlock(ffi::Array<IterVar> iter_vars, ffi::Array<BufferRegion> reads,
               ffi::Array<BufferRegion> writes, ffi::String name_hint, Stmt body,
               ffi::Optional<Stmt> init, ffi::Array<BufferVar> alloc_buffers,
               ffi::Array<MatchBufferRegion> match_buffers, ffi::Map<ffi::String, Any> annotations,
               Span span) {
  ffi::ObjectPtr<SBlockNode> node = ffi::make_object<SBlockNode>();
  node->iter_vars = std::move(iter_vars);
  node->reads = std::move(reads);
  node->writes = std::move(writes);
  node->name_hint = std::move(name_hint);
  node->body = std::move(body);
  node->init = std::move(init);
  node->alloc_buffers = std::move(alloc_buffers);
  node->match_buffers = std::move(match_buffers);
  node->annotations = std::move(annotations);
  node->span = std::move(span);
  data_ = std::move(node);
}

SBlock::SBlock(ffi::String name_hint, Stmt body, ffi::Array<BufferVar> alloc_buffers, Span span) {
  ffi::ObjectPtr<SBlockNode> node = ffi::make_object<SBlockNode>();
  node->iter_vars = {};
  node->reads = {};
  node->writes = {};
  node->name_hint = std::move(name_hint);
  node->body = std::move(body);
  node->init = std::nullopt;
  node->alloc_buffers = std::move(alloc_buffers);
  node->match_buffers = {};
  node->annotations = {};
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SBlockNode::RegisterReflection();
  refl::TypeAttrDef<SBlockNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&SBlockVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&SBlockMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&SBlockMaybeInplaceMutate));

  refl::GlobalDef().def("tirx.SBlock",
                        [](ffi::Array<IterVar> iter_vars, ffi::Array<BufferRegion> reads,
                           ffi::Array<BufferRegion> writes, ffi::String name_hint, Stmt body,
                           ffi::Optional<Stmt> init, ffi::Array<BufferVar> alloc_buffers,
                           ffi::Array<MatchBufferRegion> match_buffers,
                           ffi::Map<ffi::String, Any> annotations, Span span) {
                          return SBlock(iter_vars, reads, writes, name_hint, body, init,
                                        alloc_buffers, match_buffers, annotations, span);
                        });
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
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&ScopeIdDefStmtVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&ScopeIdDefStmtMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&ScopeIdDefStmtMaybeInplaceMutate));

  refl::GlobalDef().def("tirx.ScopeIdDefStmt",
                        [](ScopeIdDef def, Span span) { return ScopeIdDefStmt(def, span); });
}

// BlockRealize
SBlockRealize::SBlockRealize(ffi::Array<PrimExpr> values, PrimExpr predicate, SBlock block,
                             Span span) {
  TVM_FFI_CHECK_EQ(block->iter_vars.size(), values.size(), ValueError)
      << "BlockRealize needs to have the same number of iter_vars and binding values";
  PrimType predicate_ty = predicate.ty();
  TVM_FFI_CHECK(predicate_ty.MatchesCode(DLDataTypeCode::kDLBool), TypeError)
      << "Expect Block.predicate to be a bool expression";
  ffi::ObjectPtr<SBlockRealizeNode> node = ffi::make_object<SBlockRealizeNode>();
  node->iter_values = std::move(values);
  node->predicate = std::move(predicate);
  node->block = std::move(block);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SBlockRealizeNode::RegisterReflection();
  refl::TypeAttrDef<SBlockRealizeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&SBlockRealizeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&SBlockRealizeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&SBlockRealizeMaybeInplaceMutate));

  refl::GlobalDef().def("tirx.SBlockRealize", [](ffi::Array<PrimExpr> iter_values,
                                                 PrimExpr predicate, SBlock block, Span span) {
    return SBlockRealize(iter_values, predicate, block, span);
  });
}

PrimExpr TypeAnnotation(PrimType dtype, Span span) {
  static const Op& type_annotation_op = Op::Get("tirx.type_annotation");
  return Call(dtype, type_annotation_op, {}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_TIRX_REGISTER_OP("type_annotation")
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

}  // namespace tirx
}  // namespace tvm
