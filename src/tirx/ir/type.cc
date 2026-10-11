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
 * \file tirx/ir/type.cc
 * \brief Types specific to TIRX.
 */
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/type.h>

#include <utility>

namespace tvm::tirx {

void MutableCellTypeNode::Validate() const {
  const PrimType& type = element_type;
  bool numeric = type.MatchesCode(kDLInt, kDLUInt, kDLFloat, kDLBfloat, kDLBool) ||
                 (type.code() >= kDLFloat8_e3m4 && type.code() <= kDLFloat4_e2m1fn);
  TVM_FFI_CHECK(numeric && !type.IsScalableVector() && type.bits() > 0 && type.lanes() > 0,
                TypeError)
      << "MutableCellType requires a numeric or boolean scalar or fixed-vector element type";
  TVM_FFI_CHECK(scope == "local", ValueError)
      << "MutableCellType only supports the local storage scope";
}

MutableCellType::MutableCellType(PrimType element_type, ffi::String scope, Location loc)
    : Type(ffi::UnsafeInit{}) {
  auto node = ffi::make_object<MutableCellTypeNode>();
  node->element_type = std::move(element_type);
  node->scope = std::move(scope);
  node->loc = std::move(loc);
  node->Validate();
  data_ = std::move(node);
}

bool TensorTypeNode::IsScalar(bool alloc_or_decl) const {
  // TODO(@bohan): logical scope is not considered
  return shape.size() == 1 && tvm::prim::IsOne(shape[0]) && strides.empty() &&
         (!alloc_or_decl || tvm::prim::IsZero(elem_offset)) && data_alignment == 64 &&
         offset_factor == 1 && layout.has_value() &&
         ffi::StructuralEqual()(layout.value(), TileLayoutNode::DefaultLayout({1}));
}

std::optional<int64_t> TensorTypeNode::ConstantAllocationSize() const {
  int64_t result = 1;
  for (const PrimExpr& extent : shape) {
    const auto* size = extent.as<IntImmNode>();
    if (!size) return std::nullopt;
    auto product = (result * size->value).as<int64_t>();
    if (!product.has_value()) return std::nullopt;
    result = *product;
  }
  return result;
}

namespace {

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> MutableCellTypeVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const auto* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MutableCellTypeNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->element_type));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->scope));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> MutableCellTypeMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const auto* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MutableCellTypeNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimType>, element_type,
                                    mutator->MutateExpected(self->element_type));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::String>, scope,
                                    mutator->MutateExpected(self->scope));
  if (element_type.UnchangedOrSameAs(self->element_type) && scope.UnchangedOrSameAs(self->scope)) {
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<MutableCellTypeNode>(*self);
  copy->element_type = std::move(element_type).ValueOrUnchanged(std::move(copy->element_type));
  copy->scope = std::move(scope).ValueOrUnchanged(std::move(copy->scope));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> MutableCellTypeMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  auto* self = const_cast<MutableCellTypeNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MutableCellTypeNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimType>, element_type,
      mutator->MutateExpected(self->element_type, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::String>, scope,
                                    mutator->MutateExpected(self->scope, ffi::InplaceMode::kAllow));
  if (!element_type.IsUnchanged()) self->element_type = std::move(element_type).ValueUnchecked();
  if (!scope.IsUnchanged()) self->scope = std::move(scope).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> TensorMapTypeVisit(
    ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TensorMapTypeMutate(
    ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TensorMapTypeMaybeInplaceMutate(
    ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> TensorTypeVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: storage_scope, data_alignment, offset_factor
  const TensorTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorTypeNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->dtype));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->shape));
  // Empty strides denote the common compact layout.  Broad callbacks do not see the empty
  // container; explicit strides retain normal container descent and callback behavior.
  if (!self->strides.empty()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->strides));
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->elem_offset));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->layout));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TensorTypeMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: storage_scope, data_alignment, offset_factor
  const TensorTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorTypeNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimType>, mapped_dtype,
                                    mutator->MutateExpected(self->dtype));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_shape,
                                    mutator->MutateExpected(self->shape));
  // Empty strides denote the common compact layout.  Broad callbacks do not see the empty
  // container; explicit strides retain normal container descent and callback behavior.
  ffi::UnchangedOr<ffi::Array<PrimExpr>> mapped_strides = ffi::Unchanged();
  if (!self->strides.empty()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, descended_strides,
                                      mutator->MutateExpected(self->strides));
    mapped_strides = std::move(descended_strides);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_elem_offset,
                                    mutator->MutateExpected(self->elem_offset));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Layout>>, mapped_layout,
                                    mutator->MutateExpected(self->layout));
  if (mapped_dtype.UnchangedOrSameAs(self->dtype) && mapped_shape.UnchangedOrSameAs(self->shape) &&
      mapped_strides.UnchangedOrSameAs(self->strides) &&
      mapped_elem_offset.UnchangedOrSameAs(self->elem_offset) &&
      mapped_layout.UnchangedOrSameAs(self->layout)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<TensorTypeNode> copy = ffi::make_object<TensorTypeNode>(*self);
  copy->dtype = std::move(mapped_dtype).ValueOrUnchanged(std::move(copy->dtype));
  copy->shape = std::move(mapped_shape).ValueOrUnchanged(std::move(copy->shape));
  copy->strides = std::move(mapped_strides).ValueOrUnchanged(std::move(copy->strides));
  copy->elem_offset = std::move(mapped_elem_offset).ValueOrUnchanged(std::move(copy->elem_offset));
  copy->layout = std::move(mapped_layout).ValueOrUnchanged(std::move(copy->layout));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TensorTypeMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: storage_scope, data_alignment, offset_factor
  TensorTypeNode* self = const_cast<TensorTypeNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorTypeNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimType>, mapped_dtype,
                                    mutator->MutateExpected(self->dtype, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_shape,
                                    mutator->MutateExpected(self->shape, ffi::InplaceMode::kAllow));
  // Empty strides denote the common compact layout.  Broad callbacks do not see the empty
  // container; explicit strides retain normal container descent and callback behavior.
  ffi::UnchangedOr<ffi::Array<PrimExpr>> mapped_strides = ffi::Unchanged();
  if (!self->strides.empty()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
        ffi::UnchangedOr<ffi::Array<PrimExpr>>, descended_strides,
        mutator->MutateExpected(self->strides, ffi::InplaceMode::kAllow));
    mapped_strides = std::move(descended_strides);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_elem_offset,
      mutator->MutateExpected(self->elem_offset, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Optional<Layout>>, mapped_layout,
      mutator->MutateExpected(self->layout, ffi::InplaceMode::kAllow));
  if (mapped_dtype.UnchangedOrSameAs(self->dtype) && mapped_shape.UnchangedOrSameAs(self->shape) &&
      mapped_strides.UnchangedOrSameAs(self->strides) &&
      mapped_elem_offset.UnchangedOrSameAs(self->elem_offset) &&
      mapped_layout.UnchangedOrSameAs(self->layout)) {
    return ffi::Unchanged();
  }
  if (!mapped_dtype.IsUnchanged()) self->dtype = std::move(mapped_dtype).ValueUnchecked();
  if (!mapped_shape.IsUnchanged()) self->shape = std::move(mapped_shape).ValueUnchecked();
  if (!mapped_strides.IsUnchanged()) self->strides = std::move(mapped_strides).ValueUnchecked();
  if (!mapped_elem_offset.IsUnchanged())
    self->elem_offset = std::move(mapped_elem_offset).ValueUnchecked();
  if (!mapped_layout.IsUnchanged()) self->layout = std::move(mapped_layout).ValueUnchecked();
  return ffi::Unchanged();
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MutableCellTypeNode::RegisterReflection();
  refl::TypeAttrDef<MutableCellTypeNode>()
      .attr(refl::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&MutableCellTypeVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&MutableCellTypeMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&MutableCellTypeMaybeInplaceMutate>());
  refl::GlobalDef().def(
      "tirx.MutableCellType", [](PrimType element_type, ffi::String scope, Location loc) {
        return MutableCellType(std::move(element_type), std::move(scope), std::move(loc));
      });
}

TensorType::TensorType(ffi::String storage_scope, PrimType dtype, ffi::Array<PrimExpr> shape,
                       ffi::Array<PrimExpr> strides, ffi::Optional<PrimExpr> elem_offset,
                       int data_alignment, int offset_factor, ffi::Optional<Layout> layout,
                       Location loc)
    : Type(ffi::UnsafeInit{}) {
  PrimExpr offset = elem_offset.value_or(
      IntImm(shape.empty() ? PrimType(tvm::tirx::DefaultIndexType()) : shape[0].ty(), 0));
  auto n = ffi::make_object<TensorTypeNode>(std::move(offset));
  n->dtype = std::move(dtype);
  n->storage_scope = storage_scope.empty() ? ffi::String("global") : std::move(storage_scope);
  n->shape = std::move(shape);
  n->strides = std::move(strides);
  n->data_alignment =
      data_alignment <= 0 ? static_cast<int>(runtime::kAllocAlignment) : data_alignment;
  n->offset_factor = offset_factor == 0 ? 1 : offset_factor;
  n->layout = std::move(layout);
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TensorTypeNode::RegisterReflection();
  refl::TypeAttrDef<TensorTypeNode>()
      .attr(refl::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&TensorTypeVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&TensorTypeMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TensorTypeMaybeInplaceMutate>());

  refl::GlobalDef().def(
      "tirx.TensorType",
      [](ffi::String storage_scope, PrimType dtype, ffi::Array<PrimExpr> shape,
         ffi::Array<PrimExpr> strides, ffi::Optional<PrimExpr> elem_offset, int data_alignment,
         int offset_factor, ffi::Optional<Layout> layout, Location loc) {
        return TensorType(std::move(storage_scope), std::move(dtype), std::move(shape),
                          std::move(strides), std::move(elem_offset), data_alignment, offset_factor,
                          std::move(layout), std::move(loc));
      });
}

TensorMapType::TensorMapType(Location loc) : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<TensorMapTypeNode> n = ffi::make_object<TensorMapTypeNode>();
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TensorMapTypeNode::RegisterReflection();
  refl::TypeAttrDef<TensorMapTypeNode>()
      .attr(refl::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&TensorMapTypeVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&TensorMapTypeMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TensorMapTypeMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.TensorMapType", [](Location loc) { return TensorMapType(loc); });
}

}  // namespace tvm::tirx
