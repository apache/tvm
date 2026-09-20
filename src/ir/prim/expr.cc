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
 * \file expr.cc
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>

#include <utility>

namespace tvm {
namespace prim {

namespace {

int GetLanesOrVScaleFactor(const PrimType& ty) {
  return ty.IsScalableVector() ? ty.VScaleFactor() : ty.lanes();
}

TVM_FFI_INLINE const PrimTypeNode* GetPrimTypeNode(const PrimExpr& expr) {
  const auto* node = expr.get();
  TVM_FFI_DCHECK(node != nullptr);
  TVM_FFI_DCHECK(!node->ExprNode::ty.IsMissing());
  const auto* prim_ty = node->ExprNode::ty.as<PrimTypeNode>();
  TVM_FFI_DCHECK(prim_ty != nullptr);
  return prim_ty;
}

// Structural traversal hooks

template <typename TNode>
TVMFFIAny BinaryVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const TNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->b));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

template <typename TNode>
TVMFFIAny BinaryMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const TNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, a_u,
                                    mutator->MutateExpected(self->a));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, b_u,
                                    mutator->MutateExpected(self->b));
  if (a_u.UnchangedOrSameAs(self->a) && b_u.UnchangedOrSameAs(self->b)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<TNode> copy = ffi::make_object<TNode>(*self);
  // StructuralMap preserves node types. A rewrite that changes operand dtypes must keep the
  // operands compatible and set the result type itself; a generic traversal cannot infer the
  // casts that would require.
  if (!a_u.IsUnchanged()) copy->a = std::move(a_u).ValueUnchecked();
  if (!b_u.IsUnchanged()) copy->b = std::move(b_u).ValueUnchecked();
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

template <typename TNode>
TVMFFIAny BinaryMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                   ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  TNode* self = const_cast<TNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, a_u,
                                    mutator->MutateExpected(self->a, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, b_u,
                                    mutator->MutateExpected(self->b, ffi::InplaceMode::kAllow));
  if (a_u.UnchangedOrSameAs(self->a) && b_u.UnchangedOrSameAs(self->b)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!a_u.IsUnchanged()) self->a = std::move(a_u).ValueUnchecked();
  if (!b_u.IsUnchanged()) self->b = std::move(b_u).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny CastVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const CastNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CastNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny CastMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const CastNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CastNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value_u,
                                    mutator->MutateExpected(self->value));
  if (mapped_value_u.UnchangedOrSameAs(self->value)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<CastNode> copy = ffi::make_object<CastNode>(*self);
  if (!mapped_value_u.IsUnchanged()) copy->value = std::move(mapped_value_u).ValueUnchecked();
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny CastMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  CastNode* self = const_cast<CastNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CastNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value_u,
                                    mutator->MutateExpected(self->value, ffi::InplaceMode::kAllow));
  if (mapped_value_u.UnchangedOrSameAs(self->value)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_value_u.IsUnchanged()) self->value = std::move(mapped_value_u).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny NotVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const NotNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const NotNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->a));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny NotMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const NotNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const NotNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_a_u,
                                    mutator->MutateExpected(self->a));
  if (mapped_a_u.UnchangedOrSameAs(self->a)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<NotNode> copy = ffi::make_object<NotNode>(*self);
  if (!mapped_a_u.IsUnchanged()) copy->a = std::move(mapped_a_u).ValueUnchecked();
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny NotMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  NotNode* self = const_cast<NotNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const NotNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_a_u,
                                    mutator->MutateExpected(self->a, ffi::InplaceMode::kAllow));
  if (mapped_a_u.UnchangedOrSameAs(self->a)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_a_u.IsUnchanged()) self->a = std::move(mapped_a_u).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny BitwiseNotVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const BitwiseNotNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BitwiseNotNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->a));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny BitwiseNotMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const BitwiseNotNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BitwiseNotNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_a_u,
                                    mutator->MutateExpected(self->a));
  if (mapped_a_u.UnchangedOrSameAs(self->a)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<BitwiseNotNode> copy = ffi::make_object<BitwiseNotNode>(*self);
  if (!mapped_a_u.IsUnchanged()) copy->a = std::move(mapped_a_u).ValueUnchecked();
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny BitwiseNotMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                       ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  BitwiseNotNode* self = const_cast<BitwiseNotNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BitwiseNotNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_a_u,
                                    mutator->MutateExpected(self->a, ffi::InplaceMode::kAllow));
  if (mapped_a_u.UnchangedOrSameAs(self->a)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_a_u.IsUnchanged()) self->a = std::move(mapped_a_u).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny SelectVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const SelectNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SelectNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->true_value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->false_value));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny SelectMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const SelectNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SelectNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_condition_u,
                                    mutator->MutateExpected(self->condition));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_true_value_u,
                                    mutator->MutateExpected(self->true_value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_false_value_u,
                                    mutator->MutateExpected(self->false_value));
  if (mapped_condition_u.UnchangedOrSameAs(self->condition) &&
      mapped_true_value_u.UnchangedOrSameAs(self->true_value) &&
      mapped_false_value_u.UnchangedOrSameAs(self->false_value)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<SelectNode> copy = ffi::make_object<SelectNode>(*self);
  if (!mapped_condition_u.IsUnchanged())
    copy->condition = std::move(mapped_condition_u).ValueUnchecked();
  if (!mapped_true_value_u.IsUnchanged())
    copy->true_value = std::move(mapped_true_value_u).ValueUnchecked();
  if (!mapped_false_value_u.IsUnchanged())
    copy->false_value = std::move(mapped_false_value_u).ValueUnchecked();
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny SelectMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                   ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  SelectNode* self = const_cast<SelectNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SelectNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_condition_u,
      mutator->MutateExpected(self->condition, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_true_value_u,
      mutator->MutateExpected(self->true_value, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_false_value_u,
      mutator->MutateExpected(self->false_value, ffi::InplaceMode::kAllow));
  if (mapped_condition_u.UnchangedOrSameAs(self->condition) &&
      mapped_true_value_u.UnchangedOrSameAs(self->true_value) &&
      mapped_false_value_u.UnchangedOrSameAs(self->false_value)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_condition_u.IsUnchanged())
    self->condition = std::move(mapped_condition_u).ValueUnchecked();
  if (!mapped_true_value_u.IsUnchanged())
    self->true_value = std::move(mapped_true_value_u).ValueUnchecked();
  if (!mapped_false_value_u.IsUnchanged())
    self->false_value = std::move(mapped_false_value_u).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny LetVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const LetNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const LetNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->var); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny LetMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const LetNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const LetNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Var>, mapped_var_u,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->var);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value_u,
                                    mutator->MutateExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_body_u,
                                    mutator->MutateExpected(self->body));
  if (mapped_var_u.UnchangedOrSameAs(self->var) && mapped_value_u.UnchangedOrSameAs(self->value) &&
      mapped_body_u.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<LetNode> copy = ffi::make_object<LetNode>(*self);
  if (!mapped_var_u.IsUnchanged()) copy->var = std::move(mapped_var_u).ValueUnchecked();
  if (!mapped_value_u.IsUnchanged()) copy->value = std::move(mapped_value_u).ValueUnchecked();
  if (!mapped_body_u.IsUnchanged()) copy->body = std::move(mapped_body_u).ValueUnchecked();
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny LetMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  LetNode* self = const_cast<LetNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const LetNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Var>, mapped_var_u,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->var,
                                                                     ffi::InplaceMode::kAllow);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value_u,
                                    mutator->MutateExpected(self->value, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_body_u,
                                    mutator->MutateExpected(self->body, ffi::InplaceMode::kAllow));
  if (mapped_var_u.UnchangedOrSameAs(self->var) && mapped_value_u.UnchangedOrSameAs(self->value) &&
      mapped_body_u.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_var_u.IsUnchanged()) self->var = std::move(mapped_var_u).ValueUnchecked();
  if (!mapped_value_u.IsUnchanged()) self->value = std::move(mapped_value_u).ValueUnchecked();
  if (!mapped_body_u.IsUnchanged()) self->body = std::move(mapped_body_u).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

}  // namespace

/* \brief Convert an object to a PrimExpr
 *
 * All conversions to a PrimExpr are performed as part of the FFI,
 * when calling a function that accepts a PrimExpr as an argument.  If
 * a function must normalize to a PrimExpr (e.g. before accessing the
 * `expr.dtype` field), this function allows the FFI conversions to be
 * explicitly invoked.
 */
#define TVM_DEFINE_BINOP_CONSTRUCTOR(Name)                                        \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                                 \
    using T = Name::ContainerType;                                                \
    TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined\n";                 \
    TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined\n";                 \
    const PrimTypeNode* a_ty = GetPrimTypeNode(a);                                \
    const PrimTypeNode* b_ty = GetPrimTypeNode(b);                                \
    TVM_FFI_CHECK(a_ty->dtype == b_ty->dtype, TypeError)                          \
        << "mismatched types. " << a_ty->dtype << " vs. " << b_ty->dtype << "\n"; \
    ffi::ObjectPtr<T> node = ffi::make_object<T>();                               \
    node->ExprNode::ty = a.get()->ExprNode::ty;                                   \
    node->a = std::move(a);                                                       \
    node->b = std::move(b);                                                       \
    node->span = std::move(span);                                                 \
    data_ = std::move(node);                                                      \
  }

#define TVM_DEFINE_BITWISE_CONSTRUCTOR(Name, AllowBool)                                     \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                                           \
    using T = Name::ContainerType;                                                          \
    TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined\n";                           \
    TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined\n";                           \
    const PrimTypeNode* a_ty = GetPrimTypeNode(a);                                          \
    const PrimTypeNode* b_ty = GetPrimTypeNode(b);                                          \
    TVM_FFI_CHECK(a_ty->dtype == b_ty->dtype, TypeError)                                    \
        << "mismatched types. " << a_ty->dtype << " vs. " << b_ty->dtype << "\n";           \
    TVM_FFI_CHECK(a.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) ||    \
                      (AllowBool && a.ty().MatchesCode(DLDataTypeCode::kDLBool)),           \
                  TypeError)                                                                \
        << #Name << " requires integer" << (AllowBool ? " or boolean" : "") << " operands"; \
    ffi::ObjectPtr<T> node = ffi::make_object<T>();                                         \
    node->ExprNode::ty = a.get()->ExprNode::ty;                                             \
    node->a = std::move(a);                                                                 \
    node->b = std::move(b);                                                                 \
    node->span = std::move(span);                                                           \
    data_ = std::move(node);                                                                \
  }

#define TVM_DEFINE_CMPOP_CONSTRUCTOR(Name)                                        \
  Name::Name(PrimExpr a, PrimExpr b, Span span) {                                 \
    using T = Name::ContainerType;                                                \
    TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined\n";                 \
    TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined\n";                 \
    const PrimTypeNode* a_ty = GetPrimTypeNode(a);                                \
    const PrimTypeNode* b_ty = GetPrimTypeNode(b);                                \
    TVM_FFI_CHECK(a_ty->dtype == b_ty->dtype, TypeError)                          \
        << "mismatched types. " << a_ty->dtype << " vs. " << b_ty->dtype << "\n"; \
    ffi::ObjectPtr<T> node = ffi::make_object<T>();                               \
    node->ExprNode::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});     \
    node->a = std::move(a);                                                       \
    node->b = std::move(b);                                                       \
    node->span = std::move(span);                                                 \
    data_ = std::move(node);                                                      \
  }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("prim.convert",
                        [](ffi::Variant<PrimExpr, ffi::Array<PrimExpr>> expr) { return expr; });
  // Note: kRepr for VarNode is registered via TVM_REGISTER_SCRIPT_AS_REPR in
  // src/script/printer/tirx/expr.cc (-> ReprPrintTIR which delegates to TVMScriptPrinter).
}

// Cast
Cast::Cast(PrimType value_ty, PrimExpr value, Span span) {
  TVM_FFI_ICHECK(value.defined());
  PrimType value_expr_ty = value.ty();
  TVM_FFI_ICHECK_EQ(value_ty->dtype.lanes, value_expr_ty->dtype.lanes);
  ffi::ObjectPtr<CastNode> node = ffi::make_object<CastNode>();
  node->ExprNode::ty = std::move(value_ty);
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  CastNode::RegisterReflection();
  refl::TypeAttrDef<CastNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&CastVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&CastMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&CastMaybeInplaceMutate));

  refl::GlobalDef().def("prim.Cast", [](PrimType dtype, PrimExpr value, Span span) {
    return Cast(dtype, value, span);
  });
}

// Add
TVM_DEFINE_BINOP_CONSTRUCTOR(Add);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  AddNode::RegisterReflection();
  refl::GlobalDef().def("prim.Add",
                        [](PrimExpr a, PrimExpr b, Span span) { return Add(a, b, span); });
  refl::TypeAttrDef<AddNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<AddNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<AddNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<AddNode>));
}

// LShift
TVM_DEFINE_BITWISE_CONSTRUCTOR(LShift, false);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  LShiftNode::RegisterReflection();
  refl::GlobalDef().def("prim.LShift",
                        [](PrimExpr a, PrimExpr b, Span span) { return LShift(a, b, span); });
  refl::TypeAttrDef<LShiftNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<LShiftNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<LShiftNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<LShiftNode>));
}

// RShift
TVM_DEFINE_BITWISE_CONSTRUCTOR(RShift, false);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  RShiftNode::RegisterReflection();
  refl::GlobalDef().def("prim.RShift",
                        [](PrimExpr a, PrimExpr b, Span span) { return RShift(a, b, span); });
  refl::TypeAttrDef<RShiftNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<RShiftNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<RShiftNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<RShiftNode>));
}

// BitwiseAnd
TVM_DEFINE_BITWISE_CONSTRUCTOR(BitwiseAnd, true);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BitwiseAndNode::RegisterReflection();
  refl::GlobalDef().def("prim.BitwiseAnd",
                        [](PrimExpr a, PrimExpr b, Span span) { return BitwiseAnd(a, b, span); });
  refl::TypeAttrDef<BitwiseAndNode>()
      .attr(refl::type_attr::kStructuralVisit,
            reinterpret_cast<void*>(&BinaryVisit<BitwiseAndNode>))
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(&BinaryMutate<BitwiseAndNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<BitwiseAndNode>));
}

// BitwiseOr
TVM_DEFINE_BITWISE_CONSTRUCTOR(BitwiseOr, true);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BitwiseOrNode::RegisterReflection();
  refl::GlobalDef().def("prim.BitwiseOr",
                        [](PrimExpr a, PrimExpr b, Span span) { return BitwiseOr(a, b, span); });
  refl::TypeAttrDef<BitwiseOrNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<BitwiseOrNode>))
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(&BinaryMutate<BitwiseOrNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<BitwiseOrNode>));
}

// BitwiseXor
TVM_DEFINE_BITWISE_CONSTRUCTOR(BitwiseXor, true);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BitwiseXorNode::RegisterReflection();
  refl::GlobalDef().def("prim.BitwiseXor",
                        [](PrimExpr a, PrimExpr b, Span span) { return BitwiseXor(a, b, span); });
  refl::TypeAttrDef<BitwiseXorNode>()
      .attr(refl::type_attr::kStructuralVisit,
            reinterpret_cast<void*>(&BinaryVisit<BitwiseXorNode>))
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(&BinaryMutate<BitwiseXorNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<BitwiseXorNode>));
}

// Sub
TVM_DEFINE_BINOP_CONSTRUCTOR(Sub);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SubNode::RegisterReflection();
  refl::GlobalDef().def("prim.Sub",
                        [](PrimExpr a, PrimExpr b, Span span) { return Sub(a, b, span); });
  refl::TypeAttrDef<SubNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<SubNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<SubNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<SubNode>));
}

// Mul
TVM_DEFINE_BINOP_CONSTRUCTOR(Mul);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MulNode::RegisterReflection();
  refl::GlobalDef().def("prim.Mul",
                        [](PrimExpr a, PrimExpr b, Span span) { return Mul(a, b, span); });
  refl::TypeAttrDef<MulNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<MulNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<MulNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<MulNode>));
}

// Div
TVM_DEFINE_BINOP_CONSTRUCTOR(Div);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  DivNode::RegisterReflection();
  refl::GlobalDef().def("prim.Div",
                        [](PrimExpr a, PrimExpr b, Span span) { return Div(a, b, span); });
  refl::TypeAttrDef<DivNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<DivNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<DivNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<DivNode>));
}

// Mod
TVM_DEFINE_BINOP_CONSTRUCTOR(Mod);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ModNode::RegisterReflection();
  refl::GlobalDef().def("prim.Mod",
                        [](PrimExpr a, PrimExpr b, Span span) { return Mod(a, b, span); });
  refl::TypeAttrDef<ModNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<ModNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<ModNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<ModNode>));
}

// FloorDiv
TVM_DEFINE_BINOP_CONSTRUCTOR(FloorDiv);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  FloorDivNode::RegisterReflection();
  refl::GlobalDef().def("prim.FloorDiv",
                        [](PrimExpr a, PrimExpr b, Span span) { return FloorDiv(a, b, span); });
  refl::TypeAttrDef<FloorDivNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<FloorDivNode>))
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(&BinaryMutate<FloorDivNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<FloorDivNode>));
}

// FloorMod
TVM_DEFINE_BINOP_CONSTRUCTOR(FloorMod);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  FloorModNode::RegisterReflection();
  refl::GlobalDef().def("prim.FloorMod",
                        [](PrimExpr a, PrimExpr b, Span span) { return FloorMod(a, b, span); });
  refl::TypeAttrDef<FloorModNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<FloorModNode>))
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(&BinaryMutate<FloorModNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<FloorModNode>));
}

// Min
TVM_DEFINE_BINOP_CONSTRUCTOR(Min);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MinNode::RegisterReflection();
  refl::GlobalDef().def("prim.Min",
                        [](PrimExpr a, PrimExpr b, Span span) { return Min(a, b, span); });
  refl::TypeAttrDef<MinNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<MinNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<MinNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<MinNode>));
}

// Max
TVM_DEFINE_BINOP_CONSTRUCTOR(Max);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MaxNode::RegisterReflection();
  refl::GlobalDef().def("prim.Max",
                        [](PrimExpr a, PrimExpr b, Span span) { return Max(a, b, span); });
  refl::TypeAttrDef<MaxNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<MaxNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<MaxNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<MaxNode>));
}

// EQ
TVM_DEFINE_CMPOP_CONSTRUCTOR(EQ);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  EQNode::RegisterReflection();
  refl::GlobalDef().def("prim.EQ",
                        [](PrimExpr a, PrimExpr b, Span span) { return EQ(a, b, span); });
  refl::TypeAttrDef<EQNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<EQNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<EQNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<EQNode>));
}

// NE
TVM_DEFINE_CMPOP_CONSTRUCTOR(NE);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  NENode::RegisterReflection();
  refl::GlobalDef().def("prim.NE",
                        [](PrimExpr a, PrimExpr b, Span span) { return NE(a, b, span); });
  refl::TypeAttrDef<NENode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<NENode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<NENode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<NENode>));
}

// LT
TVM_DEFINE_CMPOP_CONSTRUCTOR(LT);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  LTNode::RegisterReflection();
  refl::GlobalDef().def("prim.LT",
                        [](PrimExpr a, PrimExpr b, Span span) { return LT(a, b, span); });
  refl::TypeAttrDef<LTNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<LTNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<LTNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<LTNode>));
}

// LE
TVM_DEFINE_CMPOP_CONSTRUCTOR(LE);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  LENode::RegisterReflection();
  refl::GlobalDef().def("prim.LE",
                        [](PrimExpr a, PrimExpr b, Span span) { return LE(a, b, span); });
  refl::TypeAttrDef<LENode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<LENode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<LENode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<LENode>));
}

// GT
TVM_DEFINE_CMPOP_CONSTRUCTOR(GT);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  GTNode::RegisterReflection();
  refl::GlobalDef().def("prim.GT",
                        [](PrimExpr a, PrimExpr b, Span span) { return GT(a, b, span); });
  refl::TypeAttrDef<GTNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<GTNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<GTNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<GTNode>));
}

// GE
TVM_DEFINE_CMPOP_CONSTRUCTOR(GE);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  GENode::RegisterReflection();
  refl::GlobalDef().def("prim.GE",
                        [](PrimExpr a, PrimExpr b, Span span) { return GE(a, b, span); });
  refl::TypeAttrDef<GENode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<GENode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<GENode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<GENode>));
}

// And
And::And(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined";
  PrimType a_ty = a.ty();
  PrimType b_ty = b.ty();
  TVM_FFI_ICHECK(a_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_ICHECK(b_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_CHECK(a_ty == b_ty, TypeError) << "mismatched types";

  ffi::ObjectPtr<AndNode> node = ffi::make_object<AndNode>();
  node->ExprNode::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  AndNode::RegisterReflection();
  refl::GlobalDef().def("prim.And",
                        [](PrimExpr a, PrimExpr b, Span span) { return And(a, b, span); });
  refl::TypeAttrDef<AndNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<AndNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<AndNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<AndNode>));
}

// Or
Or::Or(PrimExpr a, PrimExpr b, Span span) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined";
  PrimType a_ty = a.ty();
  PrimType b_ty = b.ty();
  TVM_FFI_ICHECK(a_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_ICHECK(b_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_CHECK(a_ty == b_ty, TypeError) << "mismatched types";

  ffi::ObjectPtr<OrNode> node = ffi::make_object<OrNode>();
  node->ExprNode::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});
  node->a = std::move(a);
  node->b = std::move(b);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  OrNode::RegisterReflection();
  refl::GlobalDef().def("prim.Or",
                        [](PrimExpr a, PrimExpr b, Span span) { return Or(a, b, span); });
  refl::TypeAttrDef<OrNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BinaryVisit<OrNode>))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BinaryMutate<OrNode>))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BinaryMaybeInplaceMutate<OrNode>));
}

// Not
Not::Not(PrimExpr a, Span span) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  PrimType a_ty = a.ty();
  TVM_FFI_ICHECK(a_ty.MatchesCode(DLDataTypeCode::kDLBool));

  ffi::ObjectPtr<NotNode> node = ffi::make_object<NotNode>();
  node->ExprNode::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});
  node->a = std::move(a);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  NotNode::RegisterReflection();
  refl::TypeAttrDef<NotNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&NotVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&NotMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&NotMaybeInplaceMutate));

  refl::GlobalDef().def("prim.Not", [](PrimExpr a, Span span) { return Not(a, span); });
}

// BitwiseNot
BitwiseNot::BitwiseNot(PrimExpr a, Span span) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  PrimType a_ty = a.ty();
  TVM_FFI_CHECK(a_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) ||
                    a_ty.MatchesCode(DLDataTypeCode::kDLBool),
                TypeError)
      << "BitwiseNot requires an integer or boolean operand";

  ffi::ObjectPtr<BitwiseNotNode> node = ffi::make_object<BitwiseNotNode>();
  node->ExprNode::ty = a_ty;
  node->a = std::move(a);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BitwiseNotNode::RegisterReflection();
  refl::TypeAttrDef<BitwiseNotNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BitwiseNotVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BitwiseNotMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BitwiseNotMaybeInplaceMutate));

  refl::GlobalDef().def("prim.BitwiseNot",
                        [](PrimExpr a, Span span) { return BitwiseNot(a, span); });
}

// Select
Select::Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span) {
  TVM_FFI_CHECK(condition.defined(), ValueError) << "condition is undefined";
  TVM_FFI_CHECK(true_value.defined(), ValueError) << "true_value is undefined";
  TVM_FFI_CHECK(false_value.defined(), ValueError) << "true_value is undefined";
  PrimType condition_ty = condition.ty();
  PrimType true_ty = true_value.ty();
  PrimType false_ty = false_value.ty();
  TVM_FFI_ICHECK(condition_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_ICHECK(GetLanesOrVScaleFactor(condition_ty) == GetLanesOrVScaleFactor(true_ty) ||
                 condition_ty.IsScalar());
  TVM_FFI_CHECK(false_ty == true_ty, TypeError)
      << "mismatched types. "
      << "False type: " << false_ty->dtype << "; True type: " << true_ty->dtype;

  ffi::ObjectPtr<SelectNode> node = ffi::make_object<SelectNode>();
  node->ExprNode::ty = true_ty;
  node->condition = std::move(condition);
  node->true_value = std::move(true_value);
  node->false_value = std::move(false_value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SelectNode::RegisterReflection();
  refl::TypeAttrDef<SelectNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&SelectVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&SelectMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&SelectMaybeInplaceMutate));

  refl::GlobalDef().def(
      "prim.Select", [](PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Span span) {
        return Select(condition, true_value, false_value, span);
      });
}

// Let
Let::Let(Var var, PrimExpr value, PrimExpr body, Span span) {
  TVM_FFI_ICHECK(value.defined());
  TVM_FFI_ICHECK(body.defined());
  TVM_FFI_ICHECK(value.ty() == var->ty.as_or_throw<PrimType>());

  ffi::ObjectPtr<LetNode> node = ffi::make_object<LetNode>();
  node->ExprNode::ty = body.ty();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  LetNode::RegisterReflection();
  refl::TypeAttrDef<LetNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&LetVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&LetMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&LetMaybeInplaceMutate));

  refl::GlobalDef().def("prim.Let", [](Var var, PrimExpr value, PrimExpr body, Span span) {
    return Let(var, value, body, span);
  });
}

}  // namespace prim

}  // namespace tvm
