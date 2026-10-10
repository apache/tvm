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
 * \file tvm/ir/stmt.cc
 */
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/stmt.h>

#include <iterator>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "seq_stmt_mutate.h"

namespace tvm {
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

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> RegionStmtVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const RegionStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RegionStmtNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->attrs));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->args));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->body_params); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  return visitor->WithDefRegionKind(kTVMFFIDefRegionKindSimple,
                                    [&]() { return visitor->VisitExpected(self->result_vars); });
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> RegionStmtMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const RegionStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RegionStmtNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<DictAttrs>, mapped_attrs,
                                    mutator->MutateExpected(self->attrs));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Expr>>, mapped_args,
                                    mutator->MutateExpected(self->args));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_body_params,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->body_params);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_result_vars,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->result_vars);
                                    }));
  if (mapped_args.UnchangedOrSameAs(self->args) && mapped_attrs.UnchangedOrSameAs(self->attrs) &&
      mapped_body_params.UnchangedOrSameAs(self->body_params) &&
      mapped_body.UnchangedOrSameAs(self->body) &&
      mapped_result_vars.UnchangedOrSameAs(self->result_vars)) {
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<RegionStmtNode>(*self);
  if (!mapped_args.IsUnchanged()) copy->args = std::move(mapped_args).ValueUnchecked();
  if (!mapped_attrs.IsUnchanged()) copy->attrs = std::move(mapped_attrs).ValueUnchecked();
  if (!mapped_body_params.IsUnchanged())
    copy->body_params = std::move(mapped_body_params).ValueUnchecked();
  if (!mapped_body.IsUnchanged()) copy->body = SeqStmt(std::move(mapped_body).ValueUnchecked());
  if (!mapped_result_vars.IsUnchanged())
    copy->result_vars = std::move(mapped_result_vars).ValueUnchecked();
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> RegionStmtMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const RegionStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RegionStmtNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<DictAttrs>, mapped_attrs,
                                    mutator->MutateExpected(self->attrs, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Expr>>, mapped_args,
                                    mutator->MutateExpected(self->args, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_body_params,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->body_params,
                                                                     ffi::InplaceMode::kAllow);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_result_vars,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->result_vars,
                                                                     ffi::InplaceMode::kAllow);
                                    }));
  if (mapped_args.UnchangedOrSameAs(self->args) && mapped_attrs.UnchangedOrSameAs(self->attrs) &&
      mapped_body_params.UnchangedOrSameAs(self->body_params) &&
      mapped_body.UnchangedOrSameAs(self->body) &&
      mapped_result_vars.UnchangedOrSameAs(self->result_vars)) {
    return ffi::Unchanged();
  }
  auto* copy = const_cast<RegionStmtNode*>(self);
  if (!mapped_args.IsUnchanged()) copy->args = std::move(mapped_args).ValueUnchecked();
  if (!mapped_attrs.IsUnchanged()) copy->attrs = std::move(mapped_attrs).ValueUnchecked();
  if (!mapped_body_params.IsUnchanged())
    copy->body_params = std::move(mapped_body_params).ValueUnchecked();
  if (!mapped_body.IsUnchanged()) copy->body = SeqStmt(std::move(mapped_body).ValueUnchecked());
  if (!mapped_result_vars.IsUnchanged())
    copy->result_vars = std::move(mapped_result_vars).ValueUnchecked();
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
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<PrimExpr>>, mapped_step,
                                    mutator->MutateExpected(self->step));
  if (mapped_loop_var.UnchangedOrSameAs(self->loop_var) &&
      mapped_min.UnchangedOrSameAs(self->min) && mapped_extent.UnchangedOrSameAs(self->extent) &&
      mapped_body.UnchangedOrSameAs(self->body) && mapped_step.UnchangedOrSameAs(self->step)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<ForNode> copy = ffi::make_object<ForNode>(*self);
  copy->loop_var = std::move(mapped_loop_var).ValueOrUnchanged(std::move(copy->loop_var));
  copy->min = std::move(mapped_min).ValueOrUnchanged(std::move(copy->min));
  copy->extent = std::move(mapped_extent).ValueOrUnchanged(std::move(copy->extent));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
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
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<PrimExpr>>, mapped_step,
                                    mutator->MutateExpected(self->step, ffi::InplaceMode::kAllow));
  if (mapped_loop_var.UnchangedOrSameAs(self->loop_var) &&
      mapped_min.UnchangedOrSameAs(self->min) && mapped_extent.UnchangedOrSameAs(self->extent) &&
      mapped_body.UnchangedOrSameAs(self->body) && mapped_step.UnchangedOrSameAs(self->step)) {
    return ffi::Unchanged();
  }
  if (!mapped_loop_var.IsUnchanged()) self->loop_var = std::move(mapped_loop_var).ValueUnchecked();
  if (!mapped_min.IsUnchanged()) self->min = std::move(mapped_min).ValueUnchecked();
  if (!mapped_extent.IsUnchanged()) self->extent = std::move(mapped_extent).ValueUnchecked();
  if (!mapped_body.IsUnchanged()) self->body = SeqStmt(std::move(mapped_body).ValueUnchecked());
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
  if (!mapped_body.IsUnchanged()) self->body = SeqStmt(std::move(mapped_body).ValueUnchecked());
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

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> IfVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const IfNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->then_case));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->else_case));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> IfMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const IfNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_condition,
                                    mutator->MutateExpected(self->condition));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_then_case,
                                    mutator->MutateExpected(self->then_case));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Stmt>>, mapped_else_case,
                                    mutator->MutateExpected(self->else_case));
  if (mapped_condition.UnchangedOrSameAs(self->condition) &&
      mapped_then_case.UnchangedOrSameAs(self->then_case) &&
      (mapped_else_case.IsUnchanged() || ffi::AnyView(mapped_else_case).same_as(self->else_case))) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<IfNode> copy = ffi::make_object<IfNode>(*self);
  copy->condition = std::move(mapped_condition).ValueOrUnchanged(std::move(copy->condition));
  if (!mapped_then_case.IsUnchanged())
    copy->then_case = SeqStmt(std::move(mapped_then_case).ValueUnchecked());
  if (!mapped_else_case.IsUnchanged()) {
    auto replacement = std::move(mapped_else_case).ValueUnchecked();
    copy->else_case = replacement.has_value()
                          ? ffi::Optional<SeqStmt>(SeqStmt(std::move(replacement).value()))
                          : std::nullopt;
  }
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> IfMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  IfNode* self = const_cast<IfNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfNode>(value));
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
      (mapped_else_case.IsUnchanged() || ffi::AnyView(mapped_else_case).same_as(self->else_case))) {
    return ffi::Unchanged();
  }
  if (!mapped_condition.IsUnchanged())
    self->condition = std::move(mapped_condition).ValueUnchecked();
  if (!mapped_then_case.IsUnchanged())
    self->then_case = SeqStmt(std::move(mapped_then_case).ValueUnchecked());
  if (!mapped_else_case.IsUnchanged()) {
    auto replacement = std::move(mapped_else_case).ValueUnchecked();
    self->else_case = replacement.has_value()
                          ? ffi::Optional<SeqStmt>(SeqStmt(std::move(replacement).value()))
                          : std::nullopt;
  }
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

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> TensorStoreVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const TensorStoreNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorStoreNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->dest));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->indices));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TensorStoreMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const TensorStoreNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorStoreNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_dest,
                                    mutator->MutateExpected(self->dest));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_indices,
                                    mutator->MutateExpected(self->indices));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value,
                                    mutator->MutateExpected(self->value));
  if (mapped_dest.UnchangedOrSameAs(self->dest) && mapped_value.UnchangedOrSameAs(self->value) &&
      mapped_indices.UnchangedOrSameAs(self->indices)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<TensorStoreNode> copy = ffi::make_object<TensorStoreNode>(*self);
  copy->dest = std::move(mapped_dest).ValueOrUnchanged(std::move(copy->dest));
  copy->value = std::move(mapped_value).ValueOrUnchanged(std::move(copy->value));
  copy->indices = std::move(mapped_indices).ValueOrUnchanged(std::move(copy->indices));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TensorStoreMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  TensorStoreNode* self = const_cast<TensorStoreNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorStoreNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_dest,
                                    mutator->MutateExpected(self->dest, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_indices,
      mutator->MutateExpected(self->indices, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value,
                                    mutator->MutateExpected(self->value, ffi::InplaceMode::kAllow));
  if (mapped_dest.UnchangedOrSameAs(self->dest) && mapped_value.UnchangedOrSameAs(self->value) &&
      mapped_indices.UnchangedOrSameAs(self->indices)) {
    return ffi::Unchanged();
  }
  if (!mapped_dest.IsUnchanged()) self->dest = std::move(mapped_dest).ValueUnchecked();
  if (!mapped_value.IsUnchanged()) self->value = std::move(mapped_value).ValueUnchecked();
  if (!mapped_indices.IsUnchanged()) self->indices = std::move(mapped_indices).ValueUnchecked();
  return ffi::Unchanged();
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  StmtNode::RegisterReflection();
  ffi::reflection::EnsureTypeAttrColumn(tvm::type_attr::kTensorStoreValidate);
  ffi::reflection::EnsureTypeAttrColumn(tvm::type_attr::kEvaluateValidate);
}

// Bind
Bind::Bind(Var var, Expr value, ffi::Optional<Location> loc) : Stmt(ffi::UnsafeInit{}) {
  TVM_FFI_ICHECK(value.defined());
  TVM_FFI_ICHECK(ffi::StructuralEqual()(value->ty, var->ty));

  ffi::ObjectPtr<BindNode> node = ffi::make_object<BindNode>(std::move(var), std::move(value));
  node->loc = loc.value_or(UnknownLoc());
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

  refl::GlobalDef().def("ir.Bind", [](Var var, Expr value, ffi::Optional<Location> loc) {
    return Bind(var, value, loc);
  });
}

// RegionStmt
bool IsRegionOp(const Op& op) {
  if (!Op::HasAttrMap(tvm::op_attr::kRegionGetBodyParams)) return false;
  static auto get_body_params =
      Op::GetAttrMap<FRegionGetBodyParams>(tvm::op_attr::kRegionGetBodyParams);
  return get_body_params.count(op);
}

ffi::Array<Var> GetRegionBodyParams(Op op, ffi::Array<Expr> args, DictAttrs attrs) {
  TVM_FFI_CHECK(IsRegionOp(op), ValueError)
      << op->name << " does not support region construction: FRegionGetBodyParams is required";
  CallNode signature(op);
  signature.args = std::move(args);
  signature.attrs = std::move(attrs);
  op.Validate(&signature);
  static auto get_body_params =
      Op::GetAttrMap<FRegionGetBodyParams>(tvm::op_attr::kRegionGetBodyParams);
  ffi::Array<Var> params = get_body_params[op].CallExpected(&signature).value();
  std::unordered_set<const VarNode*> definitions;
  for (const Var& param : params) {
    TVM_FFI_CHECK(!param->ty.as<MissingType>().has_value(), ValueError)
        << "FRegionGetBodyParams for " << op->name << " must return typed variables";
    TVM_FFI_CHECK(definitions.insert(param.get()).second, ValueError)
        << "FRegionGetBodyParams for " << op->name << " must return distinct definitions";
  }
  return params;
}

RegionStmt::RegionStmt(Op op, ffi::Array<Expr> args, ffi::Array<Var> body_params, DictAttrs attrs,
                       SeqStmt body, ffi::Array<Var> result_vars, ffi::Optional<Location> loc)
    : Stmt(ffi::UnsafeInit{}) {
  TVM_FFI_CHECK(op.defined() && body.defined(), ValueError)
      << "RegionStmt requires an operator and a body";
  ffi::Array<Var> expected_params = GetRegionBodyParams(op, args, attrs);
  TVM_FFI_CHECK_EQ(body_params.size(), expected_params.size(), ValueError)
      << op->name << " expects " << expected_params.size() << " body parameters";
  for (size_t i = 0; i < body_params.size(); ++i) {
    TVM_FFI_CHECK(ffi::StructuralEqual()(body_params[i]->ty, expected_params[i]->ty), ValueError)
        << op->name << " body parameter " << i << " has a type inconsistent with its contract";
  }
  std::unordered_set<const VarNode*> definitions;
  for (const auto& vars : {body_params, result_vars}) {
    for (const Var& var : vars) {
      TVM_FFI_CHECK(definitions.insert(var.get()).second, ValueError)
          << "RegionStmt parameters and results must be distinct definitions";
    }
  }
  auto n = ffi::make_object<RegionStmtNode>(std::move(op), std::move(body));
  n->args = std::move(args);
  n->body_params = std::move(body_params);
  n->attrs = std::move(attrs);
  n->result_vars = std::move(result_vars);
  n->loc = loc.value_or(UnknownLoc());
  if (Op::HasAttrMap(tvm::op_attr::kRegionValidate)) {
    static auto validate = Op::GetAttrMap<FRegionValidate>(tvm::op_attr::kRegionValidate);
    if (validate.count(n->op)) validate[n->op].CallExpected(n.get()).value();
  }
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  RegionStmtNode::RegisterReflection();
  refl::TypeAttrDef<RegionStmtNode>()
      .attr(refl::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&RegionStmtVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&RegionStmtMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&RegionStmtMaybeInplaceMutate>());
  refl::GlobalDef().def(
      "ir.RegionStmt",
      [](Op op, ffi::Array<Expr> args, ffi::Array<Var> body_params, DictAttrs attrs, SeqStmt body,
         ffi::Array<Var> result_vars, ffi::Optional<Location> loc) {
        return RegionStmt(op, args, body_params, attrs, body, result_vars, loc);
      });
}

// AssertStmt
AssertStmt::AssertStmt(PrimExpr condition, StringImm error_kind,
                       ffi::Array<StringImm> message_parts, ffi::Optional<Location> loc)
    : Stmt(ffi::UnsafeInit{}) {
  TVM_FFI_ICHECK(condition.defined());
  PrimType condition_ty = condition.ty();
  TVM_FFI_ICHECK(condition_ty.MatchesCode(DLDataTypeCode::kDLBool))
      << "AssertStmt should have boolean condition, "
      << "but received " << condition << " with dtype " << condition_ty;
  TVM_FFI_ICHECK(error_kind.defined());

  ffi::ObjectPtr<AssertStmtNode> node =
      ffi::make_object<AssertStmtNode>(std::move(condition), std::move(error_kind));
  node->message_parts = std::move(message_parts);
  node->loc = loc.value_or(UnknownLoc());
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

  refl::GlobalDef().def("ir.AssertStmt",
                        [](PrimExpr condition, StringImm error_kind,
                           ffi::Array<StringImm> message_parts, ffi::Optional<Location> loc) {
                          return AssertStmt(condition, error_kind, message_parts, loc);
                        });
}

// For
For::For(PrimVar loop_var, PrimExpr min, PrimExpr extent, ForKind kind, SeqStmt body,
         ffi::Map<ffi::String, Any> annotations, ffi::Optional<PrimExpr> step,
         ffi::Optional<Location> loc)
    : Stmt(ffi::UnsafeInit{}) {
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

  ffi::ObjectPtr<ForNode> node = ffi::make_object<ForNode>(std::move(loop_var), std::move(min),
                                                           std::move(extent), std::move(body));
  node->kind = kind;
  node->annotations = std::move(annotations);
  node->step = std::move(step);
  node->loc = loc.value_or(UnknownLoc());
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

  refl::GlobalDef().def("ir.For",
                        [](PrimVar loop_var, PrimExpr min, PrimExpr extent, int kind, SeqStmt body,
                           ffi::Optional<ffi::Map<ffi::String, Any>> annotations,
                           ffi::Optional<PrimExpr> step, ffi::Optional<Location> loc) {
                          return For(loop_var, min, extent, static_cast<ForKind>(kind), body,
                                     annotations.value_or(ffi::Map<ffi::String, Any>()), step, loc);
                        });
}

bool ForNode::HasTrivialStep() const { return !step.has_value() || IsOne(*step); }

std::ostream& operator<<(std::ostream& out, ForKind type) {  // NOLINT(*)
  switch (type) {
    case ForKind::kDefault:
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
  }
  return out;
}

// While
While::While(PrimExpr condition, SeqStmt body, ffi::Optional<Location> loc)
    : Stmt(ffi::UnsafeInit{}) {
  TVM_FFI_ICHECK(condition.defined());
  TVM_FFI_ICHECK(condition.ty().IsScalar());
  TVM_FFI_ICHECK(body.defined());

  ffi::ObjectPtr<WhileNode> node =
      ffi::make_object<WhileNode>(std::move(condition), std::move(body));
  node->loc = loc.value_or(UnknownLoc());
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

  refl::GlobalDef().def("ir.While",
                        [](PrimExpr condition, SeqStmt body, ffi::Optional<Location> loc) {
                          return While(condition, body, loc);
                        });
}

// Return
Return::Return(Expr value, ffi::Optional<Location> loc) : Stmt(ffi::UnsafeInit{}) {
  TVM_FFI_ICHECK(value.defined());

  ffi::ObjectPtr<ReturnNode> node = ffi::make_object<ReturnNode>(std::move(value));
  node->loc = loc.value_or(UnknownLoc());
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

  refl::GlobalDef().def("ir.Return",
                        [](Expr value, ffi::Optional<Location> loc) { return Return(value, loc); });
}

// Break
Break::Break(ffi::Optional<Location> loc) : Stmt(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<BreakNode> node = ffi::make_object<BreakNode>();
  node->loc = loc.value_or(UnknownLoc());
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

  refl::GlobalDef().def("ir.Break", [](ffi::Optional<Location> loc) { return Break(loc); });
}

// Continue
Continue::Continue(ffi::Optional<Location> loc) : Stmt(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<ContinueNode> node = ffi::make_object<ContinueNode>();
  node->loc = loc.value_or(UnknownLoc());
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

  refl::GlobalDef().def("ir.Continue", [](ffi::Optional<Location> loc) { return Continue(loc); });
}

// SeqStmt
SeqStmt::SeqStmt(ffi::Array<Stmt> seq, ffi::Optional<Location> loc) : Stmt(ffi::UnsafeInit{}) {
  bool requires_flattening = std::any_of(seq.begin(), seq.end(), [](const Stmt& stmt) {
    return stmt.as<SeqStmtNode>() || detail::IsSeqStmtNoOp(stmt);
  });
  if (requires_flattening) {
    ffi::Array<Stmt> flattened;
    auto append = [&](auto&& append, const Stmt& stmt) -> void {
      if (const auto* nested = stmt.as<SeqStmtNode>()) {
        for (const Stmt& child : nested->seq) append(append, child);
      } else if (!detail::IsSeqStmtNoOp(stmt)) {
        flattened.push_back(stmt);
      }
    };
    for (const Stmt& stmt : seq) append(append, stmt);
    seq = std::move(flattened);
  }
  auto node = ffi::make_object<SeqStmtNode>();
  node->seq = std::move(seq);
  node->loc = loc.value_or(UnknownLoc());
  data_ = std::move(node);
}

SeqStmt::SeqStmt(Stmt stmt, ffi::Optional<Location> loc) : Stmt(ffi::UnsafeInit{}) {
  Location location = loc.value_or(stmt->loc);
  if (const auto* sequence = stmt.as<SeqStmtNode>()) {
    if (!loc.has_value() || location.same_as(sequence->loc)) {
      data_ = ffi::GetObjectPtr<SeqStmtNode>(const_cast<SeqStmtNode*>(sequence));
      return;
    }
    *this = SeqStmt(sequence->seq, std::move(location));
  } else {
    *this = SeqStmt(ffi::Array<Stmt>{std::move(stmt)}, std::move(location));
  }
}

void SeqStmtNode::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  struct CanonicalSequence : refl::InfoTrait {
    static int Set(void* field, const TVMFFIAny* value) {
      TVM_FFI_SAFE_CALL_BEGIN();
      auto seq = ffi::AnyView::CopyFromTVMFFIAny(*value).cast<ffi::Array<Stmt>>();
      *static_cast<ffi::Array<Stmt>*>(field) = SeqStmt(std::move(seq))->seq;
      TVM_FFI_SAFE_CALL_END();
    }
    void Apply(refl::FieldInfoBuilder* info) const { info->setter = reinterpret_cast<void*>(&Set); }
  };
  refl::ObjectDef<SeqStmtNode>().def_ro("seq", &SeqStmtNode::seq, CanonicalSequence{});
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

  refl::GlobalDef().def("ir.SeqStmt", [](SeqStmt seq, ffi::Optional<Location> loc) {
    return SeqStmt(std::move(seq), std::move(loc));
  });
}

// If
If::If(PrimExpr condition, SeqStmt then_case, ffi::Optional<SeqStmt> else_case,
       ffi::Optional<Location> loc)
    : Stmt(ffi::UnsafeInit{}) {
  TVM_FFI_ICHECK(condition.defined());
  TVM_FFI_ICHECK(then_case.defined());
  // else_case may be null.
  ffi::ObjectPtr<IfNode> node =
      ffi::make_object<IfNode>(std::move(condition), std::move(then_case));
  node->else_case = std::move(else_case);
  node->loc = loc.value_or(UnknownLoc());
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  IfNode::RegisterReflection();
  refl::TypeAttrDef<IfNode>()
      .attr(refl::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&IfVisit>())
      .attr(refl::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&IfMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&IfMaybeInplaceMutate>());

  refl::GlobalDef().def("ir.If", [](PrimExpr condition, SeqStmt then_case,
                                    ffi::Optional<SeqStmt> else_case, ffi::Optional<Location> loc) {
    return If(condition, then_case, else_case, loc);
  });
}

// Evaluate
Evaluate::Evaluate(Expr value, ffi::Optional<Location> loc) : Stmt(ffi::UnsafeInit{}) {
  TVM_FFI_ICHECK(value.defined());
  static const ffi::reflection::TypeAttrColumn validate_column(tvm::type_attr::kEvaluateValidate);
  if (auto validate = validate_column[value->ty->type_index()]; validate != nullptr) {
    validate.cast<ffi::Function>()(value);
  }

  ffi::ObjectPtr<EvaluateNode> node = ffi::make_object<EvaluateNode>(std::move(value));
  node->loc = loc.value_or(UnknownLoc());
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

  refl::GlobalDef().def(
      "ir.Evaluate", [](Expr value, ffi::Optional<Location> loc) { return Evaluate(value, loc); });
}

// TensorStore
TensorStore::TensorStore(Expr dest, ffi::Array<PrimExpr> indices, PrimExpr value,
                         ffi::Optional<Location> loc)
    : Stmt(ffi::UnsafeInit{}) {
  namespace refl = ffi::reflection;
  static const refl::TypeAttrColumn validate_column(tvm::type_attr::kTensorStoreValidate);
  auto validate = validate_column[dest->ty->type_index()];
  TVM_FFI_CHECK(validate != nullptr, TypeError)
      << "Type " << dest->ty->GetTypeKey() << " does not support TensorStore";
  validate.cast<ffi::Function>()(dest, indices, value);
  auto node = ffi::make_object<TensorStoreNode>(std::move(dest), std::move(value));
  node->indices = std::move(indices);
  node->loc = loc.value_or(UnknownLoc());
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TensorStoreNode::RegisterReflection();
  refl::TypeAttrDef<TensorStoreNode>()
      .attr(refl::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&TensorStoreVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&TensorStoreMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TensorStoreMaybeInplaceMutate>());

  refl::GlobalDef().def("ir.TensorStore", [](Expr dest, ffi::Array<PrimExpr> indices,
                                             PrimExpr value, ffi::Optional<Location> loc) {
    return TensorStore(dest, indices, value, loc);
  });
}

}  // namespace tvm
