/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include <unordered_set>

namespace tvm {
namespace relax {

namespace {

// Traverses only ExprNode::ty.  The per-node payload is an intentional constant leaf:
// ConstantNode::data is tensor data; StringImmNode::value and DataTypeImmNode::value are scalars;
// ExternFuncNode::global_symbol is scalar and BaseFuncNode::attrs is metadata.
template <typename TNode>
TVMFFIAny TypeOnlyExprVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const TNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

template <typename TNode>
TVMFFIAny TypeOnlyExprMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const TNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MutateExpected(self->ty));
  if (mapped_ty.UnchangedOrSameAs(self->ty)) return ffi::Unchanged().CopyToTVMFFIAny();
  ffi::ObjectPtr<TNode> copy = ffi::make_object<TNode>(*self);
  copy->ty = std::move(mapped_ty).ValueOrUnchanged(std::move(copy->ty));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

template <typename TNode>
TVMFFIAny TypeOnlyExprMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                         ffi::AnyView value) noexcept {
  TNode* self = const_cast<TNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->ty));
  if (!mapped_ty.UnchangedOrSameAs(self->ty)) {
    self->ty = std::move(mapped_ty).ValueUnchecked();
  }
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny ShapeExprVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const ShapeExprNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShapeExprNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->values));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny ShapeExprMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const ShapeExprNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShapeExprNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MutateExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_values,
                                    mutator->MutateExpected(self->values));
  if (mapped_ty.UnchangedOrSameAs(self->ty) && mapped_values.UnchangedOrSameAs(self->values)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<ShapeExprNode> copy = ffi::make_object<ShapeExprNode>(*self);
  copy->ty = std::move(mapped_ty).ValueOrUnchanged(std::move(copy->ty));
  copy->values = std::move(mapped_values).ValueOrUnchanged(std::move(copy->values));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny ShapeExprMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                      ffi::AnyView value) noexcept {
  ShapeExprNode* self = const_cast<ShapeExprNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShapeExprNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_values,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->values));
  if (!mapped_ty.UnchangedOrSameAs(self->ty)) self->ty = std::move(mapped_ty).ValueUnchecked();
  if (!mapped_values.UnchangedOrSameAs(self->values)) {
    self->values = std::move(mapped_values).ValueUnchecked();
  }
  return ffi::Unchanged().CopyToTVMFFIAny();
}

// Hooks do not inherit, so DataflowVar must mirror the base VarNode remap, PrimType-skip, and
// Simple-to-None definition-region protocol.  Keep this hook triple in lockstep with VarNode.
TVMFFIAny DataflowVarVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const DataflowVarNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DataflowVarNode>(value);
  if (!self->ty.as<PrimTypeNode>()) {
    if (visitor->def_region_kind() == kTVMFFIDefRegionKindSimple) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
          kTVMFFIDefRegionKindNone, [&]() { return visitor->VisitExpected(self->ty); }));
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
    }
  }
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny DataflowVarMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const DataflowVarNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DataflowVarNode>(value);
  ffi::Expected<ffi::Any> remap_result = mutator->VarRemapGetExpected(value);
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap_result);
  if (ffi::details::ExpectedUnsafe::GetData(remap_result).type_index() !=
      ffi::TypeIndex::kTVMFFINone) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(remap_result));
  }
  if (mutator->def_region_kind() == kTVMFFIDefRegionKindNone) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::UnchangedOr<ffi::Any> result = ffi::Unchanged();
  ffi::Any mapped_value;
  if (!self->ty.as<PrimTypeNode>()) {
    ffi::Expected<ffi::UnchangedOr<ffi::Any>> mapped_ty_result =
        mutator->def_region_kind() == kTVMFFIDefRegionKindSimple
            ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone,
                                         [&]() { return mutator->MutateExpected(self->ty); })
            : mutator->MutateExpected(self->ty);
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                      std::move(mapped_ty_result));
    if (!mapped_ty.UnchangedOrSameAs(self->ty)) {
      ffi::ObjectPtr<DataflowVarNode> copy = ffi::make_object<DataflowVarNode>(*self);
      copy->ty = std::move(mapped_ty).ValueUnchecked();
      mapped_value = ffi::Any(std::move(copy));
      result = mapped_value;
    }
  }
  if (!result.IsUnchanged() || mutator->def_region_kind() == kTVMFFIDefRegionKindPattern) {
    ffi::AnyView value_to_store = result.IsUnchanged() ? value : ffi::AnyView(mapped_value);
    auto set_result = mutator->VarRemapSetExpected(value, value_to_store);
    if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
      return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(set_result).error()));
    }
  }
  return ffi::details::UnchangedOrUnsafe::MoveToTVMFFIAny(std::move(result));
}

TVMFFIAny DataflowVarMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                        ffi::AnyView value) noexcept {
  DataflowVarNode* self = const_cast<DataflowVarNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DataflowVarNode>(value));
  ffi::Expected<ffi::Any> remap_result = mutator->VarRemapGetExpected(value);
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap_result);
  if (ffi::details::ExpectedUnsafe::GetData(remap_result).type_index() !=
      ffi::TypeIndex::kTVMFFINone) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(remap_result));
  }
  if (mutator->def_region_kind() == kTVMFFIDefRegionKindNone) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::UnchangedOr<ffi::Any> result = ffi::Unchanged();
  ffi::Any mapped_value;
  if (!self->ty.as<PrimTypeNode>()) {
    ffi::Expected<ffi::UnchangedOr<ffi::Any>> mapped_ty_result =
        mutator->def_region_kind() == kTVMFFIDefRegionKindSimple
            ? mutator->WithDefRegionKind(
                  kTVMFFIDefRegionKindNone,
                  [&]() { return mutator->MaybeInplaceMutateIfUniqueExpected(self->ty); })
            : mutator->MaybeInplaceMutateIfUniqueExpected(self->ty);
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                      std::move(mapped_ty_result));
    if (!mapped_ty.UnchangedOrSameAs(self->ty)) {
      self->ty = std::move(mapped_ty).ValueUnchecked();
      mapped_value = ffi::Any(self);
      result = mapped_value;
    }
  }
  if (!result.IsUnchanged() || mutator->def_region_kind() == kTVMFFIDefRegionKindPattern) {
    ffi::AnyView value_to_store = result.IsUnchanged() ? value : ffi::AnyView(mapped_value);
    auto set_result = mutator->VarRemapSetExpected(value, value_to_store);
    if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
      return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(set_result).error()));
    }
  }
  return ffi::details::UnchangedOrUnsafe::MoveToTVMFFIAny(std::move(result));
}

TVMFFIAny SeqExprVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const SeqExprNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqExprNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->blocks));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny SeqExprMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const SeqExprNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqExprNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<BindingBlock>>, mapped_blocks,
                                    mutator->MutateExpected(self->blocks));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MutateExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_body,
                                    mutator->MutateExpected(self->body));
  if (mapped_blocks.UnchangedOrSameAs(self->blocks) && mapped_ty.UnchangedOrSameAs(self->ty) &&
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<SeqExprNode> copy = ffi::make_object<SeqExprNode>(*self);
  copy->blocks = std::move(mapped_blocks).ValueOrUnchanged(std::move(copy->blocks));
  copy->ty = std::move(mapped_ty).ValueOrUnchanged(std::move(copy->ty));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny SeqExprMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                    ffi::AnyView value) noexcept {
  SeqExprNode* self = const_cast<SeqExprNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqExprNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<BindingBlock>>, mapped_blocks,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->blocks));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_body,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->body));
  if (!mapped_blocks.UnchangedOrSameAs(self->blocks)) {
    self->blocks = std::move(mapped_blocks).ValueUnchecked();
  }
  if (!mapped_ty.UnchangedOrSameAs(self->ty)) self->ty = std::move(mapped_ty).ValueUnchecked();
  if (!mapped_body.UnchangedOrSameAs(self->body)) {
    self->body = std::move(mapped_body).ValueUnchecked();
  }
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny IfVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const IfNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->cond));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->true_branch));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->false_branch));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny IfMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const IfNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MutateExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_cond,
                                    mutator->MutateExpected(self->cond));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SeqExpr>, mapped_true_branch,
                                    mutator->MutateExpected(self->true_branch));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SeqExpr>, mapped_false_branch,
                                    mutator->MutateExpected(self->false_branch));
  if (mapped_ty.UnchangedOrSameAs(self->ty) && mapped_cond.UnchangedOrSameAs(self->cond) &&
      mapped_true_branch.UnchangedOrSameAs(self->true_branch) &&
      mapped_false_branch.UnchangedOrSameAs(self->false_branch)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<IfNode> copy = ffi::make_object<IfNode>(*self);
  copy->ty = std::move(mapped_ty).ValueOrUnchanged(std::move(copy->ty));
  copy->cond = std::move(mapped_cond).ValueOrUnchanged(std::move(copy->cond));
  copy->true_branch = std::move(mapped_true_branch).ValueOrUnchanged(std::move(copy->true_branch));
  copy->false_branch =
      std::move(mapped_false_branch).ValueOrUnchanged(std::move(copy->false_branch));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny IfMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  IfNode* self = const_cast<IfNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_cond,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->cond));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SeqExpr>, mapped_true_branch,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->true_branch));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<SeqExpr>, mapped_false_branch,
      mutator->MaybeInplaceMutateIfUniqueExpected(self->false_branch));
  if (!mapped_ty.UnchangedOrSameAs(self->ty)) self->ty = std::move(mapped_ty).ValueUnchecked();
  if (!mapped_cond.UnchangedOrSameAs(self->cond)) {
    self->cond = std::move(mapped_cond).ValueUnchecked();
  }
  if (!mapped_true_branch.UnchangedOrSameAs(self->true_branch)) {
    self->true_branch = std::move(mapped_true_branch).ValueUnchecked();
  }
  if (!mapped_false_branch.UnchangedOrSameAs(self->false_branch)) {
    self->false_branch = std::move(mapped_false_branch).ValueUnchecked();
  }
  return ffi::Unchanged().CopyToTVMFFIAny();
}

// Parameters precede the reflected ty field in this hook triple so their pattern region establishes
// the remap before the derived function type can refer to those symbolic definitions.
TVMFFIAny FunctionVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: attrs (metadata), is_pure (scalar)
  const FunctionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FunctionNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindPattern, [&]() { return visitor->VisitExpected(self->params); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ret_ty));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny FunctionMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attrs (metadata), is_pure (scalar)
  const FunctionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FunctionNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_params,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&]() {
                                      return mutator->MutateExpected(self->params);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MutateExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SeqExpr>, mapped_body,
                                    mutator->MutateExpected(self->body));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ret_ty,
                                    mutator->MutateExpected(self->ret_ty));
  if (mapped_params.UnchangedOrSameAs(self->params) && mapped_ty.UnchangedOrSameAs(self->ty) &&
      mapped_body.UnchangedOrSameAs(self->body) && mapped_ret_ty.UnchangedOrSameAs(self->ret_ty)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<FunctionNode> copy = ffi::make_object<FunctionNode>(*self);
  copy->params = std::move(mapped_params).ValueOrUnchanged(std::move(copy->params));
  copy->ty = std::move(mapped_ty).ValueOrUnchanged(std::move(copy->ty));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  copy->ret_ty = std::move(mapped_ret_ty).ValueOrUnchanged(std::move(copy->ret_ty));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny FunctionMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                     ffi::AnyView value) noexcept {
  // skips: attrs (metadata), is_pure (scalar)
  FunctionNode* self = const_cast<FunctionNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FunctionNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<Var>>, mapped_params,
      mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&]() {
        return mutator->MaybeInplaceMutateIfUniqueExpected(self->params);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SeqExpr>, mapped_body,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->body));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ret_ty,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->ret_ty));
  if (!mapped_params.UnchangedOrSameAs(self->params)) {
    self->params = std::move(mapped_params).ValueUnchecked();
  }
  if (!mapped_ty.UnchangedOrSameAs(self->ty)) self->ty = std::move(mapped_ty).ValueUnchecked();
  if (!mapped_body.UnchangedOrSameAs(self->body)) {
    self->body = std::move(mapped_body).ValueUnchecked();
  }
  if (!mapped_ret_ty.UnchangedOrSameAs(self->ret_ty)) {
    self->ret_ty = std::move(mapped_ret_ty).ValueUnchecked();
  }
  return ffi::Unchanged().CopyToTVMFFIAny();
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ShapeExprNode::RegisterReflection();
  BindingNode::RegisterReflection();
  DataflowVarNode::RegisterReflection();
  ConstantNode::RegisterReflection();
  StringImmNode::RegisterReflection();
  DataTypeImmNode::RegisterReflection();
  MatchCastNode::RegisterReflection();
  VarBindingNode::RegisterReflection();
  BindingBlockNode::RegisterReflection();
  DataflowBlockNode::RegisterReflection();
  SeqExprNode::RegisterReflection();
  IfNode::RegisterReflection();
  FunctionNode::RegisterReflection();
  ExternFuncNode::RegisterReflection();
  refl::TypeAttrDef<ShapeExprNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&ShapeExprVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&ShapeExprMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&ShapeExprMaybeInplaceMutate));
  refl::TypeAttrDef<DataflowVarNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&DataflowVarVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&DataflowVarMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&DataflowVarMaybeInplaceMutate));
  refl::TypeAttrDef<ConstantNode>()
      .attr(refl::type_attr::kStructuralVisit,
            reinterpret_cast<void*>(&TypeOnlyExprVisit<ConstantNode>))
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(&TypeOnlyExprMutate<ConstantNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&TypeOnlyExprMaybeInplaceMutate<ConstantNode>));
  refl::TypeAttrDef<StringImmNode>()
      .attr(refl::type_attr::kStructuralVisit,
            reinterpret_cast<void*>(&TypeOnlyExprVisit<StringImmNode>))
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(&TypeOnlyExprMutate<StringImmNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&TypeOnlyExprMaybeInplaceMutate<StringImmNode>));
  refl::TypeAttrDef<DataTypeImmNode>()
      .attr(refl::type_attr::kStructuralVisit,
            reinterpret_cast<void*>(&TypeOnlyExprVisit<DataTypeImmNode>))
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(&TypeOnlyExprMutate<DataTypeImmNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&TypeOnlyExprMaybeInplaceMutate<DataTypeImmNode>));
  refl::TypeAttrDef<SeqExprNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&SeqExprVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&SeqExprMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&SeqExprMaybeInplaceMutate));
  refl::TypeAttrDef<IfNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&IfVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&IfMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&IfMaybeInplaceMutate));
  refl::TypeAttrDef<FunctionNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&FunctionVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&FunctionMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&FunctionMaybeInplaceMutate));
  refl::TypeAttrDef<ExternFuncNode>()
      .attr(refl::type_attr::kStructuralVisit,
            reinterpret_cast<void*>(&TypeOnlyExprVisit<ExternFuncNode>))
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(&TypeOnlyExprMutate<ExternFuncNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&TypeOnlyExprMaybeInplaceMutate<ExternFuncNode>));
}

If::If(Expr cond, Expr true_branch, Expr false_branch, Span span) {
  ffi::ObjectPtr<IfNode> n = ffi::make_object<IfNode>();
  n->cond = std::move(cond);
  n->true_branch = std::move(true_branch);
  n->false_branch = std::move(false_branch);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.If", [](Expr cond, Expr true_branch, Expr false_branch, Span span) {
    return If(cond, true_branch, false_branch, span);
  });
}

ShapeExpr::ShapeExpr(ffi::Array<PrimExpr> values, Span span) {
  ffi::ObjectPtr<ShapeExprNode> n = ffi::make_object<ShapeExprNode>();

  n->values = values.Map([](PrimExpr value) {
    if (value->IsInstance<IntImmNode>()) {
      return tvm::cast(PrimType::Int(64), value);
    }
    TVM_FFI_ICHECK(value.ty().MatchesElementType(DLDataTypeCode::kDLInt, 64))
        << "the value in ShapeType can only have dtype of int64";
    return value;
  });
  n->span = span;
  n->ty = ShapeType(values, span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ShapeExpr", [](ffi::Array<PrimExpr> values, Span span) {
    return ShapeExpr(values, span);
  });
}

DataflowVar::DataflowVar(ffi::String name, ffi::Optional<Type> ty_annotation, Span span) {
  ffi::ObjectPtr<DataflowVarNode> n = ffi::make_object<DataflowVarNode>();
  n->name = std::move(name);
  if (ty_annotation.has_value()) {
    n->ty = ty_annotation.value();
  }
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.DataflowVar",
                        [](ffi::String name, ffi::Optional<Type> ty_annotation, Span span) {
                          return DataflowVar(name, ty_annotation, span);
                        });
}

Constant::Constant(runtime::Tensor data, ffi::Optional<Type> ty_annotation, Span span) {
  ffi::ObjectPtr<ConstantNode> n = ffi::make_object<ConstantNode>();
  n->data = std::move(data);
  n->span = std::move(span);

  // set type.
  ffi::Array<PrimExpr> values;
  auto shape_tuple = n->data.Shape();
  for (size_t dim = 0; dim < shape_tuple.size(); ++dim) {
    values.push_back(IntImm::Int64(shape_tuple[dim]));
  }
  if (ty_annotation.has_value()) {
    n->ty = ty_annotation.value();
  } else {
    TensorType tinfo(ShapeExpr(values), PrimType(n->data.DataType()), VDevice(), span);
    n->ty = tinfo;
  }

  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.Constant",
                        [](runtime::Tensor data, ffi::Optional<Type> ty_annotation = std::nullopt,
                           Span span = Span()) { return Constant(data, ty_annotation, span); });
}

StringImm::StringImm(ffi::String value, Span span) {
  ffi::ObjectPtr<StringImmNode> n = ffi::make_object<StringImmNode>();
  n->value = std::move(value);
  n->span = std::move(span);
  n->ty = AnyType();
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.StringImm",
                        [](ffi::String value, Span span) { return StringImm(value, span); });
}

DataTypeImm::DataTypeImm(DLDataType value, Span span) {
  ffi::ObjectPtr<DataTypeImmNode> n = ffi::make_object<DataTypeImmNode>();
  n->value = value;
  n->span = std::move(span);
  n->ty = AnyType();
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.DataTypeImm",
                        [](DLDataType value, Span span) { return DataTypeImm(value, span); });
}

MatchCast::MatchCast(Var var, Expr value, Type ty, Span span) {
  ffi::ObjectPtr<MatchCastNode> n = ffi::make_object<MatchCastNode>();
  TVM_FFI_ICHECK(var.defined()) << "MatchCast requires var to be defined";
  n->var = std::move(var);
  n->value = std::move(value);
  n->ty = std::move(ty);
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.MatchCast", [](Var var, Expr value, Type ty, Span span) {
    return MatchCast(var, value, ty, span);
  });
}

VarBinding::VarBinding(Var var, Expr value, Span span) {
  ffi::ObjectPtr<VarBindingNode> n = ffi::make_object<VarBindingNode>();
  n->var = std::move(var);
  n->value = std::move(value);
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.VarBinding", [](Var var, Expr value, Span span) {
    return VarBinding(var, value, span);
  });
}

bool VarBindingNode::SEqual(const VarBindingNode* other,
                            ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const {
  if (value->IsInstance<FunctionNode>()) {
    // Recursive function definitions may reference the bound variable
    // within the value being bound.  In these cases, the
    // var comparison must occur first to define the var, to ensure it is
    // defined at point of use.
    return equal(var, other->var, true, "var") && equal(value, other->value, false, "value");
  } else {
    // In all other cases, visit the bound value before the variable
    // it is bound to, in order to provide better error messages.
    return equal(value, other->value, false, "value") && equal(var, other->var, true, "var");
  }
}

int64_t VarBindingNode::SHash(int64_t init_hash,
                              ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const {
  int64_t hash_value = init_hash;
  if (value->IsInstance<FunctionNode>()) {
    hash_value = hash(var, hash_value, true);
    hash_value = hash(value, hash_value, false);
  } else {
    hash_value = hash(value, hash_value, false);
    hash_value = hash(var, hash_value, true);
  }
  return hash_value;
}

BindingBlock::BindingBlock(ffi::Array<Binding> bindings, Span span) {
  ffi::ObjectPtr<BindingBlockNode> n = ffi::make_object<BindingBlockNode>();
  n->bindings = std::move(bindings);
  n->span = span;
  data_ = std::move(n);
}

BindingBlockNode* BindingBlock::CopyOnWrite() {
  // The `TVM_DEFINE_OBJECT_REF_COW_METHOD` cannot be used for
  // BindingBlock, because it is the base class for `DataflowBlock`.
  // If the `TVM_DEFINE_OBJECT_REF_COW_METHOD` were used, the
  // automatic implementation would erroneously convert from a
  // `DataflowBlock` to a `BindingBlock`.
  TVM_FFI_ICHECK(data_ != nullptr);
  if (!data_.unique()) {
    ffi::ObjectPtr<BindingBlockNode> node;
    if (auto dataflow_block = as<DataflowBlockNode>()) {
      node = ffi::make_object<DataflowBlockNode>(*dataflow_block);
    } else {
      node = ffi::make_object<BindingBlockNode>(*(operator->()));
    }
    ffi::ObjectPtr<ffi::Object>(std::move(node)).swap(data_);
  }
  return static_cast<BindingBlockNode*>(data_.get());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.BindingBlock", [](ffi::Array<Binding> bindings, Span span) {
    return BindingBlock(bindings, span);
  });
}

DataflowBlock::DataflowBlock(ffi::Array<Binding> bindings, Span span) {
  ffi::ObjectPtr<DataflowBlockNode> n = ffi::make_object<DataflowBlockNode>();
  n->bindings = std::move(bindings);
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.DataflowBlock", [](ffi::Array<Binding> bindings, Span span) {
    return DataflowBlock(bindings, span);
  });
}

SeqExpr::SeqExpr(Expr body) {
  if (auto seq = body.as<SeqExpr>()) {
    *this = seq.value();
  } else {
    *this = SeqExpr(ffi::Array<BindingBlock>{}, body);
  }
}

SeqExpr::SeqExpr(ffi::Array<BindingBlock> blocks, Expr body, Span span) {
  ffi::ObjectPtr<SeqExprNode> n = ffi::make_object<SeqExprNode>();
  n->blocks = std::move(blocks);
  n->body = std::move(body);
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.SeqExpr", [](ffi::Array<BindingBlock> blocks, Expr body, Span span) {
    return SeqExpr(blocks, body, span);
  });
}

Function::Function(ffi::Array<Var> params, Expr body, ffi::Optional<Type> ret_ty, bool is_pure,
                   DictAttrs attrs, Span span) {
  // Set the function type.
  // For function, we take a conservative approach and require the function type
  // to be known at construction time.
  ffi::Array<Type> param_ty;

  for (const Var& param : params) {
    TVM_FFI_ICHECK(!param->ty.IsMissing()) << "relax.Function requires params to contain ty";
    param_ty.push_back(GetType(param));
  }

  ffi::Optional<Type> body_ty;

  if (!body->ty.IsMissing()) {
    body_ty = GetType(body);
  }

  TVM_FFI_ICHECK(body_ty.has_value() || ret_ty.has_value())
      << "Function must be constructed with either "
      << "an explicit type for the return type, "
      << "or a normalized body with type.";

  // Use the body's type if there is no explicit return type,
  // or if the body may provide a more granular return type.
  bool use_body_ty =
      !ret_ty.has_value() || (body_ty && ret_ty && IsBaseOf(ret_ty.value(), body_ty.value()));

  if (use_body_ty) {
    // MatchCast nodes within the body may introduce new symbolic
    // variables.  These are in-scope for the function body, but not
    // for the function's return type.  When hoisting the body's type
    // to the function return type, symbolic variables may only be
    // used if they were defined by the function's parameters.
    auto f_var_map = [&] {
      auto tir_vars = DefinableTIRVarsInType(TupleType(params.Map(GetType)));
      std::unordered_set<tirx::Var> lookup(tir_vars.begin(), tir_vars.end());
      return [lookup = std::move(lookup)](const Var& var) -> ffi::Optional<Expr> {
        if (auto prim_var = var.as<tirx::PrimVar>(); prim_var && lookup.count(prim_var.value())) {
          return prim_var.value().as_or_throw<PrimExpr>();
        }
        return std::nullopt;
      };
    }();
    ret_ty = EraseToWellDefined(body_ty.value(), f_var_map);
  }

  FuncType func_ty(param_ty, ret_ty.value(), is_pure);

  // set the fields
  ffi::ObjectPtr<FunctionNode> n = ffi::make_object<FunctionNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_ty = ret_ty.value();
  n->is_pure = is_pure;
  n->ty = std::move(func_ty);
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.Function",
                        [](ffi::Array<Var> params, Expr body, ffi::Optional<Type> ret_ty,
                           bool is_pure, DictAttrs attrs, Span span) {
                          return Function(params, body, ret_ty, is_pure, attrs, span);
                        });
}

Function Function::CreateEmpty(ffi::Array<Var> params, Type ret_ty, bool is_pure, DictAttrs attrs,
                               Span span) {
  ffi::Array<Type> param_ty;
  for (const Var& param : params) {
    TVM_FFI_ICHECK(!param->ty.IsMissing()) << "relax.Function requires params to contain ty.";
    param_ty.push_back(GetType(param));
  }

  FuncType finfo(param_ty, ret_ty, is_pure);

  // A dummy body, to ensure that the empty function is still well-formed.
  Expr body = [&]() -> Expr {
    Var output("output", ret_ty);
    Call expr(Type::Missing(), ExternFunc("_dummy_function", FuncType({}, ret_ty)), {});

    return SeqExpr({BindingBlock({VarBinding(output, expr)})}, output);
  }();

  // set the fields
  ffi::ObjectPtr<FunctionNode> n = ffi::make_object<FunctionNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->is_pure = is_pure;
  n->ty = std::move(finfo);
  n->ret_ty = std::move(ret_ty);
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  return Function(std::move(n));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.FunctionCreateEmpty", [](ffi::Array<Var> params, Type ret_ty,
                                                        bool is_pure, DictAttrs attrs, Span span) {
    return Function::CreateEmpty(params, ret_ty, is_pure, attrs, span);
  });
}

// Special opaque derivation function for ExternFunc
// Take look at ty_args to figure out the return Type.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  auto infer_by_ty_args = [](const Call& call, const BlockBuilder& ctx) -> Type {
    TVM_FFI_ICHECK(call->ty_args.defined()) << "ty_args field of CallNode should always be defined";
    if (call->ty_args.empty()) {
      return AnyType();
    } else if (call->ty_args.size() == 1) {
      return call->ty_args[0];
    } else {
      return TupleType(call->ty_args);
    }
  };
  refl::GlobalDef().def("tvm.relax.type.infer_by_ty_args", infer_by_ty_args);
}

// Get the derive function.
FuncType GetExternFuncType() {
  EnvFunc fn = EnvFunc::Get("tvm.relax.type.infer_by_ty_args");
  TypeDeriveFunc derive;
  derive = fn;
  return FuncType::OpaqueFunc(derive);
}

ExternFunc::ExternFunc(ffi::String global_symbol, Span span)
    : ExternFunc(global_symbol, GetExternFuncType(), span) {}

ExternFunc::ExternFunc(ffi::String global_symbol, Type ty, Span span) {
  TVM_FFI_ICHECK(ty.as<FuncTypeNode>())
      << "ExternFunc must have FuncType, "
      << "but declaration of '" << global_symbol << "' received " << ty;

  ffi::ObjectPtr<ExternFuncNode> n = ffi::make_object<ExternFuncNode>();
  n->global_symbol = std::move(global_symbol);
  n->span = span;
  n->ty = ty;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ExternFunc",
                        [](ffi::String global_symbol, ffi::Optional<Type> ty, Span span) {
                          if (ty.has_value()) {
                            return ExternFunc(global_symbol, ty.value(), span);
                          } else {
                            return ExternFunc(global_symbol, span);
                          }
                        });
}

Expr GetShapeOf(const Expr& expr) {
  // default case, to be normalized.
  TVM_FFI_ICHECK(!expr->ty.IsMissing()) << "GetShapeOf can only be applied to normalized expr";
  auto* tinfo = GetTypeAs<TensorTypeNode>(expr);

  TVM_FFI_ICHECK(tinfo != nullptr) << "ShapeOf can only be applied to expr with TensorType";
  if (tinfo->shape.has_value()) return tinfo->shape.value();

  static const Op& op = Op::Get("relax.shape_of");
  // default case, call shape of, eagerly normalize the expr.
  Call call_shape_of(Type::Missing(), op, {expr}, {}, {});
  UpdateType(call_shape_of, ShapeType(tinfo->ndim));
  return call_shape_of;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.GetShapeOf", [](const Expr& expr) { return GetShapeOf(expr); })
      .def("relax.FuncWithAttr",
           [](BaseFunc func, ffi::String key, ffi::ObjectRef value) -> ffi::Optional<Function> {
             if (func->IsInstance<relax::FunctionNode>()) {
               return WithAttr(std::move(func).as_or_throw<relax::Function>(), key, value);
             }
             return std::nullopt;
           })
      .def("relax.FuncWithAttrs",
           [](BaseFunc func, ffi::Map<ffi::String, ffi::Any> attr_map) -> ffi::Optional<Function> {
             if (func->IsInstance<relax::FunctionNode>()) {
               return WithAttrs(std::move(func).as_or_throw<relax::Function>(), attr_map);
             }
             return std::nullopt;
           })
      .def("relax.FuncWithoutAttr", [](BaseFunc func, ffi::String key) -> ffi::Optional<Function> {
        if (func->IsInstance<relax::FunctionNode>()) {
          return WithoutAttr(std::move(func).as_or_throw<relax::Function>(), key);
        }
        return std::nullopt;
      });
}

}  // namespace relax
}  // namespace tvm
