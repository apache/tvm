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
 * \file src/ir/expr.cc
 * \brief The expression AST nodes for the common IR infra.
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/type.h>

#include <cmath>
#include <utility>

#include "../support/limits.h"

namespace tvm {

namespace {

TVMFFIAny OpaqueExprVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const OpaqueExprNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const OpaqueExprNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  // Any None is Expected<Optional<VisitInterrupt>>'s successful empty value.
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny OpaqueExprMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const OpaqueExprNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const OpaqueExprNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty_u,
                                    mutator->MutateExpected(self->ty));
  if (mapped_ty_u.UnchangedOrSameAs(self->ty)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<OpaqueExprNode> copy = ffi::make_object<OpaqueExprNode>(*self);
  if (!mapped_ty_u.IsUnchanged()) copy->ty = std::move(mapped_ty_u).ValueUnchecked();
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny OpaqueExprMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                       ffi::AnyView value) noexcept {
  OpaqueExprNode* self = const_cast<OpaqueExprNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const OpaqueExprNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty_u,
                                    mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow));
  if (mapped_ty_u.UnchangedOrSameAs(self->ty)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_ty_u.IsUnchanged()) self->ty = std::move(mapped_ty_u).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny TensorLoadVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const TensorLoadNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorLoadNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->source));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->indices));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny TensorLoadMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const TensorLoadNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorLoadNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty_u,
                                    mutator->MutateExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_source_u,
                                    mutator->MutateExpected(self->source));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_indices_u,
                                    mutator->MutateExpected(self->indices));
  if (mapped_ty_u.UnchangedOrSameAs(self->ty) && mapped_source_u.UnchangedOrSameAs(self->source) &&
      mapped_indices_u.UnchangedOrSameAs(self->indices)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<TensorLoadNode> copy = ffi::make_object<TensorLoadNode>(*self);
  if (!mapped_ty_u.IsUnchanged()) copy->ty = std::move(mapped_ty_u).ValueUnchecked();
  if (!mapped_source_u.IsUnchanged()) copy->source = std::move(mapped_source_u).ValueUnchecked();
  if (!mapped_indices_u.IsUnchanged()) copy->indices = std::move(mapped_indices_u).ValueUnchecked();
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny TensorLoadMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                       ffi::AnyView value) noexcept {
  TensorLoadNode* self = const_cast<TensorLoadNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorLoadNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty_u,
                                    mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Expr>, mapped_source_u,
      mutator->MutateExpected(self->source, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_indices_u,
      mutator->MutateExpected(self->indices, ffi::InplaceMode::kAllow));
  if (mapped_ty_u.UnchangedOrSameAs(self->ty) && mapped_source_u.UnchangedOrSameAs(self->source) &&
      mapped_indices_u.UnchangedOrSameAs(self->indices)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_ty_u.IsUnchanged()) self->ty = std::move(mapped_ty_u).ValueUnchecked();
  if (!mapped_source_u.IsUnchanged()) self->source = std::move(mapped_source_u).ValueUnchecked();
  if (!mapped_indices_u.IsUnchanged()) self->indices = std::move(mapped_indices_u).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny TensorRegionVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const TensorRegionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorRegionNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->source));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->region));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny TensorRegionMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const TensorRegionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorRegionNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty_u,
                                    mutator->MutateExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_source_u,
                                    mutator->MutateExpected(self->source));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Range>>, mapped_region_u,
                                    mutator->MutateExpected(self->region));
  if (mapped_ty_u.UnchangedOrSameAs(self->ty) && mapped_source_u.UnchangedOrSameAs(self->source) &&
      mapped_region_u.UnchangedOrSameAs(self->region)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<TensorRegionNode> copy = ffi::make_object<TensorRegionNode>(*self);
  if (!mapped_ty_u.IsUnchanged()) copy->ty = std::move(mapped_ty_u).ValueUnchecked();
  if (!mapped_source_u.IsUnchanged()) copy->source = std::move(mapped_source_u).ValueUnchecked();
  if (!mapped_region_u.IsUnchanged()) copy->region = std::move(mapped_region_u).ValueUnchecked();
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny TensorRegionMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                         ffi::AnyView value) noexcept {
  TensorRegionNode* self = const_cast<TensorRegionNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorRegionNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty_u,
                                    mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Expr>, mapped_source_u,
      mutator->MutateExpected(self->source, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<Range>>, mapped_region_u,
      mutator->MutateExpected(self->region, ffi::InplaceMode::kAllow));
  if (mapped_ty_u.UnchangedOrSameAs(self->ty) && mapped_source_u.UnchangedOrSameAs(self->source) &&
      mapped_region_u.UnchangedOrSameAs(self->region)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_ty_u.IsUnchanged()) self->ty = std::move(mapped_ty_u).ValueUnchecked();
  if (!mapped_source_u.IsUnchanged()) self->source = std::move(mapped_source_u).ValueUnchecked();
  if (!mapped_region_u.IsUnchanged()) self->region = std::move(mapped_region_u).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny TupleVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const TupleNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->fields));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny TupleMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const TupleNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty_u,
                                    mutator->MutateExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Expr>>, mapped_fields_u,
                                    mutator->MutateExpected(self->fields));
  if (mapped_ty_u.UnchangedOrSameAs(self->ty) && mapped_fields_u.UnchangedOrSameAs(self->fields)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<TupleNode> copy = ffi::make_object<TupleNode>(*self);
  if (!mapped_ty_u.IsUnchanged()) copy->ty = std::move(mapped_ty_u).ValueUnchecked();
  if (!mapped_fields_u.IsUnchanged()) copy->fields = std::move(mapped_fields_u).ValueUnchecked();
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny TupleMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  TupleNode* self = const_cast<TupleNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty_u,
                                    mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<Expr>>, mapped_fields_u,
      mutator->MutateExpected(self->fields, ffi::InplaceMode::kAllow));
  if (mapped_ty_u.UnchangedOrSameAs(self->ty) && mapped_fields_u.UnchangedOrSameAs(self->fields)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_ty_u.IsUnchanged()) self->ty = std::move(mapped_ty_u).ValueUnchecked();
  if (!mapped_fields_u.IsUnchanged()) self->fields = std::move(mapped_fields_u).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny TupleGetItemVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: index
  const TupleGetItemNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleGetItemNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->tuple));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny TupleGetItemMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: index
  const TupleGetItemNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleGetItemNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty_u,
                                    mutator->MutateExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_tuple_u,
                                    mutator->MutateExpected(self->tuple));
  if (mapped_ty_u.UnchangedOrSameAs(self->ty) && mapped_tuple_u.UnchangedOrSameAs(self->tuple)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<TupleGetItemNode> copy = ffi::make_object<TupleGetItemNode>(*self);
  if (!mapped_ty_u.IsUnchanged()) copy->ty = std::move(mapped_ty_u).ValueUnchecked();
  if (!mapped_tuple_u.IsUnchanged()) copy->tuple = std::move(mapped_tuple_u).ValueUnchecked();
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny TupleGetItemMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                         ffi::AnyView value) noexcept {
  // skips: index
  TupleGetItemNode* self = const_cast<TupleGetItemNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleGetItemNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty_u,
                                    mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_tuple_u,
                                    mutator->MutateExpected(self->tuple, ffi::InplaceMode::kAllow));
  if (mapped_ty_u.UnchangedOrSameAs(self->ty) && mapped_tuple_u.UnchangedOrSameAs(self->tuple)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_ty_u.IsUnchanged()) self->ty = std::move(mapped_ty_u).ValueUnchecked();
  if (!mapped_tuple_u.IsUnchanged()) self->tuple = std::move(mapped_tuple_u).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny IntImmVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  // skips: value
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny IntImmMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  // skips: value
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny IntImmMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  // skips: value
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny FloatImmVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  // skips: value
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny FloatImmMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  // skips: value
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny FloatImmMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  // skips: value
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny RangeVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const RangeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RangeNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->min));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->extent));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny RangeMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const RangeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RangeNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_min_u,
                                    mutator->MutateExpected(self->min));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_extent_u,
                                    mutator->MutateExpected(self->extent));
  if (mapped_min_u.UnchangedOrSameAs(self->min) &&
      mapped_extent_u.UnchangedOrSameAs(self->extent)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<RangeNode> copy = ffi::make_object<RangeNode>(*self);
  if (!mapped_min_u.IsUnchanged()) copy->min = std::move(mapped_min_u).ValueUnchecked();
  if (!mapped_extent_u.IsUnchanged()) copy->extent = std::move(mapped_extent_u).ValueUnchecked();
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny RangeMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  RangeNode* self = const_cast<RangeNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RangeNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_min_u,
                                    mutator->MutateExpected(self->min, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_extent_u,
      mutator->MutateExpected(self->extent, ffi::InplaceMode::kAllow));
  if (mapped_min_u.UnchangedOrSameAs(self->min) &&
      mapped_extent_u.UnchangedOrSameAs(self->extent)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_min_u.IsUnchanged()) self->min = std::move(mapped_min_u).ValueUnchecked();
  if (!mapped_extent_u.IsUnchanged()) self->extent = std::move(mapped_extent_u).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

// DataflowVarNode duplicates this protocol because structural hooks do not inherit.  Keep the two
// hook triples in lockstep when changing remap, PrimType-skip, or definition-region behavior.
TVMFFIAny VarVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: name
  const VarNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const VarNode>(value);
  // A PrimType carries only a dtype, so it has nothing to visit.  Broad callbacks do not see this
  // skipped field; dynamically typed Vars still descend through the Type value.
  if (!self->ty.as<PrimTypeNode>()) {
    // Only Simple is clamped: Pattern co-introduces type fields such as BufferType shape
    // variables, so that ambient region must continue through the dynamic type.
    if (visitor->def_region_kind() == kTVMFFIDefRegionKindSimple) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
          kTVMFFIDefRegionKindNone, [&]() { return visitor->VisitExpected(self->ty); }));
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
    }
  }
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny VarMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: name
  const VarNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const VarNode>(value);
  ffi::Expected<ffi::Any> remap_result = mutator->VarRemapGetExpected(value);
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap_result);
  if (ffi::details::ExpectedUnsafe::GetData(remap_result).type_index() !=
      ffi::TypeIndex::kTVMFFINone) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(remap_result));
  }
  if (mutator->def_region_kind() == kTVMFFIDefRegionKindNone) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::UnchangedOr<ffi::Any> result_u = ffi::Unchanged();
  ffi::Any mapped_value = ffi::Unchanged();
  // A PrimType carries only a dtype, so it has nothing to substitute.  Broad callbacks do not see
  // this skipped field; dynamically typed Vars still descend through the Type value.
  if (!self->ty.as<PrimTypeNode>()) {
    // Pattern co-introduces type fields such as BufferType shape variables; Simple does not.
    ffi::Expected<ffi::UnchangedOr<ffi::Any>> mapped_ty_result =
        mutator->def_region_kind() == kTVMFFIDefRegionKindSimple
            ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone,
                                         [&]() { return mutator->MutateExpected(self->ty); })
            : mutator->MutateExpected(self->ty);
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty_u,
                                      std::move(mapped_ty_result));
    if (!mapped_ty_u.UnchangedOrSameAs(self->ty)) {
      ffi::ObjectPtr<VarNode> copy = ffi::make_object<VarNode>(*self);
      copy->ty = std::move(mapped_ty_u).ValueUnchecked();
      mapped_value = ffi::Any(std::move(copy));
      result_u = mapped_value;
    }
  }
  if (!result_u.IsUnchanged() || mutator->def_region_kind() == kTVMFFIDefRegionKindPattern) {
    auto set_result = mutator->VarRemapSetExpected(value, mapped_value);
    if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
      return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(set_result).error()));
    }
  }
  return ffi::details::UnchangedOrUnsafe::MoveToTVMFFIAny(std::move(result_u));
}

TVMFFIAny VarMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: name
  VarNode* self = const_cast<VarNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const VarNode>(value));
  ffi::Expected<ffi::Any> remap_result = mutator->VarRemapGetExpected(value);
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap_result);
  if (ffi::details::ExpectedUnsafe::GetData(remap_result).type_index() !=
      ffi::TypeIndex::kTVMFFINone) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(remap_result));
  }
  if (mutator->def_region_kind() == kTVMFFIDefRegionKindNone) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::UnchangedOr<ffi::Any> result_u = ffi::Unchanged();
  ffi::Any mapped_value = ffi::Unchanged();
  // A PrimType carries only a dtype, so it has nothing to substitute.  Broad callbacks do not see
  // this skipped field; dynamically typed Vars still descend through the Type value.
  if (!self->ty.as<PrimTypeNode>()) {
    // Pattern co-introduces type fields such as BufferType shape variables; Simple does not.
    ffi::Expected<ffi::UnchangedOr<ffi::Any>> mapped_ty_result =
        mutator->def_region_kind() == kTVMFFIDefRegionKindSimple
            ? mutator->WithDefRegionKind(
                  kTVMFFIDefRegionKindNone,
                  [&]() { return mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow); })
            : mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow);
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty_u,
                                      std::move(mapped_ty_result));
    if (!mapped_ty_u.UnchangedOrSameAs(self->ty)) {
      self->ty = std::move(mapped_ty_u).ValueUnchecked();
      mapped_value = ffi::Any(self);
      result_u = mapped_value;
    }
  }
  if (!result_u.IsUnchanged() || mutator->def_region_kind() == kTVMFFIDefRegionKindPattern) {
    auto set_result = mutator->VarRemapSetExpected(value, mapped_value);
    if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
      return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(set_result).error()));
    }
  }
  return ffi::details::UnchangedOrUnsafe::MoveToTVMFFIAny(std::move(result_u));
}

TVMFFIAny GlobalVarVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  // GlobalVar is a module-level symbol.  name_hint is scalar identity and ty is derived from the
  // referenced function, matching GlobalVarNode's custom structural equality/hash definition.
  // It has no definition site where this hook could establish a VarRemap.  A callback that renames
  // GlobalVars is therefore responsible for returning one stable replacement per module symbol.
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny GlobalVarMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny GlobalVarMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny CallVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: attrs, constant metadata left untouched like the classic Expr functors.
  const CallNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(value);
  // A PrimType carries only a dtype, so it has nothing to visit.  Broad callbacks do not see this
  // skipped field; dynamically typed Call results still descend through the Type value.
  if (!self->ty.as<PrimTypeNode>()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  }
  // An Op is an interned registry singleton, so it has nothing to visit.  Broad callbacks do not
  // see this skipped field; function-valued Call operators still descend through the Expr value.
  if (!self->op.as<OpNode>()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->op));
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->args));
  // An empty ty_args has no element to traverse.  Broad callbacks do not see the empty container;
  // nonempty type arguments retain normal container descent and callback behavior.
  if (!self->ty_args.empty()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty_args));
  }
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny CallMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attrs, constant metadata left untouched like the classic Expr functors.
  const CallNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(value);
  // A PrimType carries only a dtype, so it has nothing to substitute.  Broad callbacks do not see
  // this skipped field; dynamically typed Call results still descend through the Type value.
  ffi::UnchangedOr<Type> mapped_ty_u = ffi::Unchanged();
  if (!self->ty.as<PrimTypeNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, descended_ty_u,
                                      mutator->MutateExpected(self->ty));
    mapped_ty_u = std::move(descended_ty_u);
  }
  // An Op is an interned registry singleton, so it has nothing to substitute.  Broad callbacks do
  // not see this skipped field; function-valued Call operators still descend through the Expr.
  ffi::UnchangedOr<Expr> mapped_op_u = ffi::Unchanged();
  if (!self->op.as<OpNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, descended_op_u,
                                      mutator->MutateExpected(self->op));
    mapped_op_u = std::move(descended_op_u);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Expr>>, mapped_args_u,
                                    mutator->MutateExpected(self->args));
  // An empty ty_args has no element to substitute.  Broad callbacks do not see the empty
  // container; nonempty type arguments retain normal container descent and callback behavior.
  ffi::UnchangedOr<ffi::Array<Type>> mapped_ty_args_u = ffi::Unchanged();
  if (!self->ty_args.empty()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Type>>, descended_ty_args_u,
                                      mutator->MutateExpected(self->ty_args));
    mapped_ty_args_u = std::move(descended_ty_args_u);
  }
  if (mapped_ty_u.UnchangedOrSameAs(self->ty) && mapped_op_u.UnchangedOrSameAs(self->op) &&
      mapped_args_u.UnchangedOrSameAs(self->args) &&
      mapped_ty_args_u.UnchangedOrSameAs(self->ty_args)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<CallNode> copy = ffi::make_object<CallNode>(*self);
  if (!mapped_ty_u.IsUnchanged()) copy->ty = std::move(mapped_ty_u).ValueUnchecked();
  if (!mapped_op_u.IsUnchanged()) copy->op = std::move(mapped_op_u).ValueUnchecked();
  if (!mapped_args_u.IsUnchanged()) copy->args = std::move(mapped_args_u).ValueUnchecked();
  if (!mapped_ty_args_u.IsUnchanged()) copy->ty_args = std::move(mapped_ty_args_u).ValueUnchecked();
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny CallMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attrs, constant metadata left untouched like the classic Expr functors.
  CallNode* self = const_cast<CallNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(value));
  // A PrimType carries only a dtype, so it has nothing to substitute.  Broad callbacks do not see
  // this skipped field; dynamically typed Call results still descend through the Type value.
  ffi::UnchangedOr<Type> mapped_ty_u = ffi::Unchanged();
  if (!self->ty.as<PrimTypeNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, descended_ty_u,
                                      mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow));
    mapped_ty_u = std::move(descended_ty_u);
  }
  // An Op is an interned registry singleton, so it has nothing to substitute.  Broad callbacks do
  // not see this skipped field; function-valued Call operators still descend through the Expr.
  ffi::UnchangedOr<Expr> mapped_op_u = ffi::Unchanged();
  if (!self->op.as<OpNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, descended_op_u,
                                      mutator->MutateExpected(self->op, ffi::InplaceMode::kAllow));
    mapped_op_u = std::move(descended_op_u);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Expr>>, mapped_args_u,
                                    mutator->MutateExpected(self->args, ffi::InplaceMode::kAllow));
  // An empty ty_args has no element to substitute.  Broad callbacks do not see the empty
  // container; nonempty type arguments retain normal container descent and callback behavior.
  ffi::UnchangedOr<ffi::Array<Type>> mapped_ty_args_u = ffi::Unchanged();
  if (!self->ty_args.empty()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
        ffi::UnchangedOr<ffi::Array<Type>>, descended_ty_args_u,
        mutator->MutateExpected(self->ty_args, ffi::InplaceMode::kAllow));
    mapped_ty_args_u = std::move(descended_ty_args_u);
  }
  if (mapped_ty_u.UnchangedOrSameAs(self->ty) && mapped_op_u.UnchangedOrSameAs(self->op) &&
      mapped_args_u.UnchangedOrSameAs(self->args) &&
      mapped_ty_args_u.UnchangedOrSameAs(self->ty_args)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_ty_u.IsUnchanged()) self->ty = std::move(mapped_ty_u).ValueUnchecked();
  if (!mapped_op_u.IsUnchanged()) self->op = std::move(mapped_op_u).ValueUnchecked();
  if (!mapped_args_u.IsUnchanged()) self->args = std::move(mapped_args_u).ValueUnchecked();
  if (!mapped_ty_args_u.IsUnchanged()) self->ty_args = std::move(mapped_ty_args_u).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() { ExprNode::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() { BaseFuncNode::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  OpaqueExprNode::RegisterReflection();
  refl::TypeAttrDef<OpaqueExprNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&OpaqueExprVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&OpaqueExprMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&OpaqueExprMaybeInplaceMutate));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TensorLoadNode::RegisterReflection();
  refl::TypeAttrDef<TensorLoadNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&TensorLoadVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&TensorLoadMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&TensorLoadMaybeInplaceMutate));
}

TensorRegion::TensorRegion(Expr source, ffi::Array<Range> region, Type ty, Span span) {
  auto node = ffi::make_object<TensorRegionNode>();
  node->source = std::move(source);
  node->region = std::move(region);
  node->ty = std::move(ty);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TensorRegionNode::RegisterReflection();
  refl::TypeAttrDef<TensorRegionNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&TensorRegionVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&TensorRegionMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&TensorRegionMaybeInplaceMutate));
  refl::GlobalDef().def("ir.TensorRegion",
                        [](Expr source, ffi::Array<Range> region, Type ty, Span span) {
                          return TensorRegion(source, region, ty, span);
                        });
}

// Tuple
Tuple::Tuple(ffi::Array<Expr> fields, Span span) {
  ffi::Optional<Type> tuple_ty = [&]() -> ffi::Optional<Type> {
    ffi::Array<Type> field_ty;
    for (const Expr& field : fields) {
      if (field->ty.IsMissing()) {
        return std::nullopt;
      }
      field_ty.push_back(field->ty);
    }
    return TupleType(field_ty);
  }();

  ffi::ObjectPtr<TupleNode> node = ffi::make_object<TupleNode>();
  node->fields = std::move(fields);
  node->span = std::move(span);
  if (tuple_ty.has_value()) {
    node->ty = tuple_ty.value();
  }
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TupleNode::RegisterReflection();
  refl::TypeAttrDef<TupleNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&TupleVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&TupleMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&TupleMaybeInplaceMutate));

  refl::GlobalDef().def("ir.Tuple",
                        [](ffi::Array<Expr> fields, Span span) { return Tuple(fields, span); });
}

// TupleGetItem
TupleGetItem::TupleGetItem(Expr tuple, int index, Span span) {
  TVM_FFI_CHECK_GE(index, 0, IndexError) << "Index out of bounds: Tuple " << tuple
                                         << " cannot be accessed with negative index " << index;
  ffi::ObjectPtr<TupleGetItemNode> node = ffi::make_object<TupleGetItemNode>();
  if (const auto* tuple_type = tuple->ty.as<TupleTypeNode>()) {
    TVM_FFI_CHECK_LT(index, tuple_type->fields.size(), IndexError)
        << "Index out of bounds: Tuple " << tuple << " is of size " << tuple_type->fields.size()
        << ", and cannot be accessed with index " << index;
    node->ty = tuple_type->fields[index];
  }
  node->tuple = std::move(tuple);
  node->index = index;
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TupleGetItemNode::RegisterReflection();
  refl::TypeAttrDef<TupleGetItemNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&TupleGetItemVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&TupleGetItemMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&TupleGetItemMaybeInplaceMutate));

  refl::GlobalDef().def("ir.TupleGetItem", [](Expr tuple, int index, Span span) {
    return TupleGetItem(tuple, index, span);
  });
}

PrimExpr::PrimExpr(Call call) : PrimExpr(std::move(call).as_or_throw<PrimExpr>()) {}

PrimExpr::PrimExpr(int32_t value) : PrimExpr(IntImm::Int32(value)) {}

PrimExpr::PrimExpr(float value) : PrimExpr(FloatImm(PrimType::Float(32), value)) {}

PrimExpr PrimExpr::ConvertFallbackValue(ffi::String value) { return prim::StringImm(value); }

namespace ffi {

PrimExpr TypeTraits<PrimExpr>::ConvertFallbackValue(StrictBool value) {
  return IntImm::Bool(value);
}

PrimExpr TypeTraits<PrimExpr>::ConvertFallbackValue(int64_t value) {
  return TypeTraits<IntImm>::ConvertFallbackValue(value);
}

PrimExpr TypeTraits<PrimExpr>::ConvertFallbackValue(double value) {
  return TypeTraits<FloatImm>::ConvertFallbackValue(value);
}

}  // namespace ffi

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.Tuple", [](ffi::Array<Expr> fields, Span span) { return Tuple(fields, span); })
      .def("relax.TupleGetItem",
           [](Expr tuple, int index, Span span) { return TupleGetItem(tuple, index, span); });
}

// IntImm
IntImm::IntImm(PrimType value_ty, int64_t value, Span span) {
  DLDataType runtime_dtype = value_ty->dtype;
  DLDataTypeCode code = value_ty.code();
  int32_t bits = value_ty.bits();
  TVM_FFI_CHECK(!value_ty.IsScalableVector() && !value_ty.IsFixedLengthVector(), ValueError)
      << "IntImm can only take scalar, but " << runtime_dtype << " was supplied.";
  TVM_FFI_CHECK(value_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                                     DLDataTypeCode::kDLBool),
                ValueError)
      << "IntImm supports only int or uint or bool type, but " << runtime_dtype << " was supplied.";
  if (code == DLDataTypeCode::kDLUInt) {
    TVM_FFI_CHECK_GE(value, 0U, ValueError)
        << "Literal value " << value << " is negative for unsigned integer type " << runtime_dtype;
    if (bits < 64) {
      TVM_FFI_CHECK_LT(value, 1LL << bits, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
    }
  } else if (bits == 1 || code == DLDataTypeCode::kDLBool) {
    // int(1)
    TVM_FFI_CHECK(value == 0 || value == 1, ValueError)
        << value << " exceeds range of " << runtime_dtype;
  } else if (bits < 64) {
    TVM_FFI_CHECK_GE(value, -(1LL << (bits - 1)), ValueError)
        << "Literal value " << value << " exceeds minimum of " << runtime_dtype;
    TVM_FFI_CHECK_LT(value, 1LL << (bits - 1), ValueError)
        << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
  }
  ffi::ObjectPtr<IntImmNode> node = ffi::make_object<IntImmNode>();
  node->ExprNode::ty = std::move(value_ty);
  node->value = value;
  node->span = span;
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  IntImmNode::RegisterReflection();
  refl::TypeAttrDef<IntImmNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&IntImmVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&IntImmMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&IntImmMaybeInplaceMutate));

  refl::GlobalDef().def("ir.IntImm", [](DLDataType dtype, int64_t value, Span span) {
    return IntImm(PrimType(dtype), value, span);
  });
}

// FloatImm
FloatImm::FloatImm(PrimType value_ty, double value, Span span) {
  DLDataType runtime_dtype = value_ty->dtype;
  DLDataTypeCode code = value_ty.code();
  int32_t bits = value_ty.bits();
  TVM_FFI_CHECK(!value_ty.IsScalableVector() && !value_ty.IsFixedLengthVector(), ValueError)
      << "FloatImm can only take scalar.";

  TVM_FFI_CHECK(
      value_ty.MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLFloat8_e3m4,
                           DLDataTypeCode::kDLFloat8_e4m3, DLDataTypeCode::kDLFloat8_e4m3b11fnuz,
                           DLDataTypeCode::kDLFloat8_e4m3fn, DLDataTypeCode::kDLFloat8_e4m3fnuz,
                           DLDataTypeCode::kDLFloat8_e5m2, DLDataTypeCode::kDLFloat8_e5m2fnuz,
                           DLDataTypeCode::kDLFloat8_e8m0fnu, DLDataTypeCode::kDLFloat6_e2m3fn,
                           DLDataTypeCode::kDLFloat6_e3m2fn) ||
          value_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16) ||
          value_ty.MatchesElementType(DLDataTypeCode::kDLFloat4_e2m1fn, 4) ||
          static_cast<int>(code) >= static_cast<int>(ffi::DLExtDataTypeCode::kDLExtCustomBegin),
      ValueError)
      << "FloatImm supports only float, but " << runtime_dtype << " was supplied.";

  // check range for float32 and float16 since they have specified range.
  if (!std::isinf(value) && !std::isnan(value)) {
    if (bits == 32) {
      TVM_FFI_CHECK_GE(value, std::numeric_limits<float>::lowest(), ValueError)
          << "Literal value " << value << " exceeds minimum of " << runtime_dtype;
      TVM_FFI_CHECK_LE(value, std::numeric_limits<float>::max(), ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
    } else if (value_ty.MatchesElementType(DLDataTypeCode::kDLFloat, 16)) {
      TVM_FFI_CHECK_GE(value, -support::kMaxFloat16, ValueError)
          << "Literal value " << value << " exceeds minimum of " << runtime_dtype;
      TVM_FFI_CHECK_LE(value, support::kMaxFloat16, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
    } else if (value_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {
      TVM_FFI_CHECK_GE(value, -support::kMaxBFloat16, ValueError)
          << "Literal value " << value << " exceeds minimum of " << runtime_dtype;
      TVM_FFI_CHECK_LE(value, support::kMaxBFloat16, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
    } else if (value_ty.MatchesCode(
                   DLDataTypeCode::kDLFloat8_e3m4, DLDataTypeCode::kDLFloat8_e4m3,
                   DLDataTypeCode::kDLFloat8_e4m3b11fnuz, DLDataTypeCode::kDLFloat8_e4m3fn,
                   DLDataTypeCode::kDLFloat8_e4m3fnuz, DLDataTypeCode::kDLFloat8_e5m2,
                   DLDataTypeCode::kDLFloat8_e5m2fnuz, DLDataTypeCode::kDLFloat8_e8m0fnu)) {
      double bound = 0.0;
      bool nonneg = false;

      switch (code) {
        case DLDataTypeCode::kDLFloat8_e3m4:
          bound = support::kMaxE3M4;
          break;
        case DLDataTypeCode::kDLFloat8_e4m3:
          bound = support::kMaxE4M3;
          break;
        case DLDataTypeCode::kDLFloat8_e4m3b11fnuz:
          bound = support::kMaxE4M3B11FNUZ;
          nonneg = true;
          break;
        case DLDataTypeCode::kDLFloat8_e4m3fn:
          bound = support::kMaxE4M3FN;
          break;
        case DLDataTypeCode::kDLFloat8_e4m3fnuz:
          bound = support::kMaxE4M3FNUZ;
          nonneg = true;
          break;
        case DLDataTypeCode::kDLFloat8_e5m2:
          bound = support::kMaxE5M2;
          break;
        case DLDataTypeCode::kDLFloat8_e5m2fnuz:
          bound = support::kMaxE5M2FNUZ;
          nonneg = true;
          break;
        case DLDataTypeCode::kDLFloat8_e8m0fnu:
          bound = support::kMaxE8M0FNU;
          nonneg = true;
          break;
        default:
          TVM_FFI_THROW(InternalError) << "Unhandled float8 type: " << runtime_dtype;
      }

      if (nonneg) {
        TVM_FFI_CHECK_GE(value, 0, ValueError)
            << "Literal value " << value << " below zero for unsigned " << runtime_dtype;
      } else {
        TVM_FFI_CHECK_GE(value, -bound, ValueError)
            << "Literal value " << value << " below minimum of " << runtime_dtype;
      }
      TVM_FFI_CHECK_LE(value, bound, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;

    } else if (value_ty.MatchesCode(DLDataTypeCode::kDLFloat6_e2m3fn,
                                    DLDataTypeCode::kDLFloat6_e3m2fn)) {
      double bound =
          (code == DLDataTypeCode::kDLFloat6_e2m3fn) ? support::kMaxE2M3FN : support::kMaxE3M2FN;
      TVM_FFI_CHECK_GE(value, -bound, ValueError)
          << "Literal value " << value << " below minimum of " << runtime_dtype;
      TVM_FFI_CHECK_LE(value, bound, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;

    } else if (code == DLDataTypeCode::kDLFloat4_e2m1fn) {
      double bound = support::kMaxE2M1FN;
      TVM_FFI_CHECK_GE(value, -bound, ValueError)
          << "Literal value " << value << " below minimum of " << runtime_dtype;
      TVM_FFI_CHECK_LE(value, bound, ValueError)
          << "Literal value " << value << " exceeds maximum of " << runtime_dtype;
    }
  }
  ffi::ObjectPtr<FloatImmNode> node = ffi::make_object<FloatImmNode>();
  node->ExprNode::ty = std::move(value_ty);
  node->value = value;
  node->span = span;
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  FloatImmNode::RegisterReflection();
  refl::TypeAttrDef<FloatImmNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&FloatImmVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&FloatImmMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&FloatImmMaybeInplaceMutate));

  refl::GlobalDef().def("ir.FloatImm", [](DLDataType dtype, double value, Span span) {
    return FloatImm(PrimType(dtype), value, span);
  });
}

// Range
Range::Range(PrimExpr begin, PrimExpr end, Span span)
    : Range(ffi::make_object<RangeNode>(begin, tvm::prim::is_zero(begin) ? end : (end - begin),
                                        span)) {}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  RangeNode::RegisterReflection();
  refl::TypeAttrDef<RangeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&RangeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&RangeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&RangeMaybeInplaceMutate));

  refl::GlobalDef().def("ir.Range",
                        [](PrimExpr begin, ffi::Optional<PrimExpr> end, Span span) -> Range {
                          if (end.has_value()) {
                            return Range(begin, end.value(), span);
                          } else {
                            return Range(IntImm(begin.ty(), 0), begin, span);
                          }
                        });
}

Range Range::FromMinExtent(PrimExpr min, PrimExpr extent, Span span) {
  return Range(ffi::make_object<RangeNode>(min, extent, span));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.Range_from_min_extent", Range::FromMinExtent);
}

// Var
Var::Var(ffi::String name, ffi::Optional<Type> ty_annotation, Span span) {
  ffi::ObjectPtr<VarNode> n = ffi::make_object<VarNode>();
  n->name = std::move(name);
  if (ty_annotation.has_value()) {
    n->ty = ty_annotation.value();
  }
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  VarNode::RegisterReflection();
  refl::TypeAttrDef<VarNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&VarVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&VarMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&VarMaybeInplaceMutate));

  refl::GlobalDef().def("ir.Var", [](ffi::String name, ffi::Optional<Type> ty_annotation,
                                     Span span) { return Var(name, ty_annotation, span); });
}

Var Var::CopyWithName(const ffi::String& name) const {
  TVM_FFI_CHECK_EQ(type_index(), VarNode::RuntimeTypeIndex(), TypeError)
      << "Cannot copy a Var runtime subtype as an ordinary Var";
  ffi::ObjectPtr<VarNode> copy = ffi::make_object<VarNode>(*get());
  copy->name = name;
  return Var(std::move(copy));
}

Var Var::CopyWithSuffix(const ffi::String& suffix) const {
  return CopyWithName(get()->name + suffix);
}

Var Var::CopyWithDType(PrimType dtype) const {
  TVM_FFI_CHECK_EQ(type_index(), VarNode::RuntimeTypeIndex(), TypeError)
      << "Cannot copy a Var runtime subtype as an ordinary Var";
  ffi::ObjectPtr<VarNode> copy = ffi::make_object<VarNode>(*get());
  copy->ExprNode::ty = std::move(dtype);
  return Var(std::move(copy));
}

// GlobalVar
GlobalVar::GlobalVar(ffi::String name_hint, Span span) {
  ffi::ObjectPtr<GlobalVarNode> n = ffi::make_object<GlobalVarNode>();
  n->name_hint = std::move(name_hint);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  GlobalVarNode::RegisterReflection();
  refl::TypeAttrDef<GlobalVarNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&GlobalVarVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&GlobalVarMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&GlobalVarMaybeInplaceMutate));

  refl::GlobalDef().def("ir.GlobalVar", [](ffi::String name) { return GlobalVar(name); });
}

// Call
Call::Call(Type ret_ty, Expr op, ffi::Array<Expr> args, Attrs attrs, ffi::Array<Type> ty_args,
           Span span) {
  TVM_FFI_CHECK(op.defined(), ValueError) << "Call expects a defined operator";

  ffi::ObjectPtr<CallNode> n = ffi::make_object<CallNode>();
  n->ExprNode::ty = std::move(ret_ty);
  n->op = std::move(op);
  n->args = std::move(args);
  n->attrs = std::move(attrs);
  n->ty_args = std::move(ty_args);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  CallNode::RegisterReflection();
  refl::TypeAttrDef<CallNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&CallVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&CallMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&CallMaybeInplaceMutate));

  refl::GlobalDef().def("ir.Call", [](Type ret_ty, Expr op, ffi::Array<Expr> args, Attrs attrs,
                                      ffi::Array<Type> ty_args, Span span) {
    return Call(ret_ty, op, args, attrs, ty_args, span);
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.DebugPrint", [](ffi::ObjectRef ref) {
    std::stringstream ss;
    ss << ref;
    return ss.str();
  });
  // Note: kRepr for GlobalVarNode is registered in script/printer/ir/ir.cc
  // via TVM_REGISTER_SCRIPT_AS_REPR(GlobalVarNode, ReprPrintIR).
}

}  // namespace tvm
