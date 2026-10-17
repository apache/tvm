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
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/type.h>

#include <utility>

namespace tvm::tirx {

bool TensorTypeNode::IsScalar(bool alloc_or_decl) const {
  // TODO(@bohan): logical scope is not considered
  return shape.size() == 1 && tvm::prim::IsOne(shape[0]) && data_alignment == 64 &&
         offset_factor == 1 &&
         (!layout.has_value() ||
          (layout.as<TileLayoutNode>() && layout.as<TileLayoutNode>()->compact &&
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
  // skips: scope, data_alignment, offset_factor
  const TensorTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorTypeNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->dtype));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->shape));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->layout));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TensorTypeMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: scope, data_alignment, offset_factor
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
  // skips: scope, data_alignment, offset_factor
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
                       ffi::Array<PrimExpr> strides, ffi::Optional<PrimExpr> elem_offset,
                       int data_alignment, int offset_factor, ffi::Optional<Layout> layout,
                       Location loc)
    : Type(ffi::UnsafeInit{}) {
  PrimExpr offset = elem_offset.value_or(IntImm(DefaultIndexType(), 0));
  auto n = ffi::make_object<TensorTypeNode>();
  n->dtype = std::move(dtype);
  n->scope = std::move(scope);
  n->shape = std::move(shape);
  n->data_alignment =
      data_alignment <= 0 ? static_cast<int>(runtime::kAllocAlignment) : data_alignment;
  n->offset_factor = offset_factor == 0 ? 1 : offset_factor;
  n->layout = std::move(layout);
  if (!strides.empty()) {
    if (n->layout.has_value()) {
      TVM_FFI_CHECK(n->layout.as<TileLayoutNode>() &&
                        n->layout.as<TileLayoutNode>()->IsRegularMemory(n->shape),
                    ValueError)
          << "Explicit tensor strides require a regular memory layout";
      auto tile = n->layout.value().as_or_throw<TileLayout>();
      TVM_FFI_CHECK(tile->shard.size() == strides.size(), ValueError)
          << "Tensor strides must match the layout rank";
      sym::Analyzer analyzer;
      for (size_t i = 0; i < strides.size(); ++i) {
        TVM_FFI_CHECK(analyzer->CanProveEqual(strides[i], tile->shard[i]->stride), ValueError)
            << "Tensor strides disagree with the explicit layout";
      }
      tile.CopyOnWrite()->compact = false;
      n->layout = tile;
    } else {
      TVM_FFI_CHECK(strides.size() == n->shape.size(), ValueError)
          << "Tensor strides must match its rank";
      ffi::Array<Iter> shard;
      for (size_t i = 0; i < strides.size(); ++i) {
        shard.push_back(Iter(n->shape[i], strides[i], Axis::Get("m")));
      }
      n->layout = TileLayout(shard, {}, {});
    }
  }
  if (elem_offset.has_value() &&
      (n->layout.has_value() || !prim::IsZero(offset) || offset.ty() != DefaultIndexType())) {
    if (!n->layout.has_value()) {
      auto tile = TileLayoutNode::DefaultLayout(n->shape);
      tile.CopyOnWrite()->compact = true;
      n->layout = tile;
    }
    if (auto tile = n->layout.as<TileLayout>()) {
      auto copy = tile.value();
      auto origin = copy->memory_origin;
      copy.CopyOnWrite()->memory_origin = prim::IsZero(origin) ? offset : origin + offset;
      n->layout = copy;
    } else if (auto compose = n->layout.as<ComposeLayout>()) {
      auto copy = compose.value();
      auto origin = copy->memory_origin;
      copy.CopyOnWrite()->memory_origin = prim::IsZero(origin) ? offset : origin + offset;
      n->layout = copy;
    } else {
      TVM_FFI_THROW(ValueError) << "Tensor memory origin requires a tile or composed layout";
    }
  }
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TensorTypeNode::RegisterReflection();
  refl::TypeAttrDef<TensorTypeNode>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&TensorTypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&TensorTypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TensorTypeMaybeInplaceMutate>());

  refl::GlobalDef().def(
      "tirx.TensorType",
      [](ffi::Array<PrimExpr> shape, PrimType dtype, runtime::MemoryScope scope,
         ffi::Array<PrimExpr> strides, ffi::AnyView elem_offset, int data_alignment,
         int offset_factor, ffi::Optional<Layout> layout, Location loc) {
        ffi::Optional<PrimExpr> offset;
        if (auto integer = elem_offset.as<int64_t>()) {
          offset = IntImm(DefaultIndexType(), integer.value());
        } else {
          offset = elem_offset.cast<ffi::Optional<PrimExpr>>();
        }
        return TensorType(std::move(shape), std::move(dtype), std::move(scope), std::move(strides),
                          std::move(offset), data_alignment, offset_factor, std::move(layout),
                          std::move(loc));
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
