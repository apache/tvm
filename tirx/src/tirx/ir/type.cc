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
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/device_api.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/type.h>

#include <utility>

namespace tvm::tirx {

bool TensorTypeNode::HasDefaultLayout(ffi::Optional<sym::Analyzer> analyzer) const {
  if (!layout.has_value()) return true;
  const auto* tile = layout.as<TileLayoutNode>();
  if (!tile || !tile->replica.empty()) return false;
  static const Axis memory_axis = Axis::Get("m");
  for (const auto& iter : tile->shard) {
    if (iter->axis != memory_axis) return false;
  }
  prim::ExprDeepEqual deep_equal;
  auto equal = [&](const PrimExpr& lhs, const PrimExpr& rhs) {
    if (deep_equal(lhs, rhs)) return true;
    if (!analyzer.has_value()) analyzer = sym::Analyzer();
    return analyzer.value()->CanProveEqual(lhs, rhs);
  };
  for (const auto& [axis, offset] : tile->offset) {
    if (axis != memory_axis || !equal(offset, IntImm(offset.ty(), 0))) return false;
  }
  PrimExpr extent = IntImm(DefaultIndexType(), 1);
  for (size_t i = tile->shard.size(); i > 0; --i) extent = extent * tile->shard[i - 1]->extent;
  PrimExpr size = IntImm(DefaultIndexType(), 1);
  for (size_t i = shape.size(); i > 0; --i) size = size * shape[i - 1];
  if (!equal(extent, size)) return false;
  // Delegate last so any analyzer created above is reused by the stride proof.
  return tile->IsContiguous(std::move(analyzer));
}

bool TensorTypeNode::IsScalar(bool alloc_or_decl) const {
  // TODO(@bohan): logical scope is not considered
  return shape.size() == 1 && tvm::prim::IsOne(shape[0]) && data_alignment == 64 &&
         (!layout.has_value() ||
          (layout.as<TileLayoutNode>() &&
           layout.value()->Canonicalize().as_or_throw<TileLayout>()->IsTrivial()));
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
  // skips: scope, data_alignment
  const TensorTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorTypeNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->dtype));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->shape));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->layout));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TensorTypeMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: scope, data_alignment
  const TensorTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorTypeNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimType>, mapped_dtype,
                                    mutator->MutateExpected(self->dtype));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_shape,
                                    mutator->MutateExpected(self->shape));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Layout>>, mapped_layout,
                                    mutator->MutateExpected(self->layout));
  if (mapped_dtype.UnchangedOrSameAs(self->dtype) && mapped_shape.UnchangedOrSameAs(self->shape) &&
      mapped_layout.UnchangedOrSameAs(self->layout)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<TensorTypeNode> copy = ffi::make_object<TensorTypeNode>(*self);
  copy->dtype = std::move(mapped_dtype).ValueOrUnchanged(std::move(copy->dtype));
  copy->shape = std::move(mapped_shape).ValueOrUnchanged(std::move(copy->shape));
  copy->layout = std::move(mapped_layout).ValueOrUnchanged(std::move(copy->layout));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TensorTypeMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: scope, data_alignment
  TensorTypeNode* self = const_cast<TensorTypeNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorTypeNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimType>, mapped_dtype,
                                    mutator->MutateExpected(self->dtype, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_shape,
                                    mutator->MutateExpected(self->shape, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Optional<Layout>>, mapped_layout,
      mutator->MutateExpected(self->layout, ffi::InplaceMode::kAllow));
  if (mapped_dtype.UnchangedOrSameAs(self->dtype) && mapped_shape.UnchangedOrSameAs(self->shape) &&
      mapped_layout.UnchangedOrSameAs(self->layout)) {
    return ffi::Unchanged();
  }
  if (!mapped_dtype.IsUnchanged()) self->dtype = std::move(mapped_dtype).ValueUnchecked();
  if (!mapped_shape.IsUnchanged()) self->shape = std::move(mapped_shape).ValueUnchecked();
  if (!mapped_layout.IsUnchanged()) self->layout = std::move(mapped_layout).ValueUnchecked();
  return ffi::Unchanged();
}

}  // namespace

TensorType::TensorType(ffi::Array<PrimExpr> shape, PrimType dtype, runtime::MemoryScope scope,
                       int data_alignment, ffi::Optional<Layout> layout, Location loc)
    : Type(ffi::UnsafeInit{}) {
  auto n = ffi::make_object<TensorTypeNode>();
  n->dtype = std::move(dtype);
  n->scope = std::move(scope);
  n->shape = std::move(shape);
  n->data_alignment =
      data_alignment <= 0 ? static_cast<int>(runtime::kAllocAlignment) : data_alignment;
  n->layout = std::move(layout);
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TensorTypeNode::RegisterReflection();
  refl::GlobalDef().def("tirx.TensorTypeHasDefaultLayout",
                        [](const TensorType& tensor, ffi::Optional<sym::Analyzer> analyzer) {
                          return tensor->HasDefaultLayout(std::move(analyzer));
                        });
  refl::TypeAttrDef<TensorTypeNode>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&TensorTypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&TensorTypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TensorTypeMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.TensorType",
                        [](ffi::Array<PrimExpr> shape, PrimType dtype, runtime::MemoryScope scope,
                           int data_alignment, ffi::Optional<Layout> layout, Location loc) {
                          return TensorType(std::move(shape), std::move(dtype), std::move(scope),
                                            data_alignment, std::move(layout), std::move(loc));
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
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&TensorMapTypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&TensorMapTypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TensorMapTypeMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.TensorMapType", [](Location loc) { return TensorMapType(loc); });
}

}  // namespace tvm::tirx
