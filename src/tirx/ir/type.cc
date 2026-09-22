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
#include <tvm/runtime/device_api.h>
#include <tvm/tirx/type.h>

#include <utility>

namespace tvm::tirx {
namespace {

TVMFFIAny TensorMapTypeVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny TensorMapTypeMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny TensorMapTypeMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny BufferTypeVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: storage_scope, data_alignment, offset_factor
  const BufferTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferTypeNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->dtype));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->shape));
  // Empty strides denote the common compact layout.  Broad callbacks do not see the empty
  // container; explicit strides retain normal container descent and callback behavior.
  if (!self->strides.empty()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->strides));
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->elem_offset));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->layout));
  // allocated_addr is empty outside specialized storage scopes.  Broad callbacks do not see the
  // empty container; present addresses retain normal container descent and callback behavior.
  if (!self->allocated_addr.empty()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->allocated_addr));
  }
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny BufferTypeMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: storage_scope, data_alignment, offset_factor
  const BufferTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferTypeNode>(value);
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
  // allocated_addr is empty outside specialized storage scopes.  Broad callbacks do not see the
  // empty container; present addresses retain normal container descent and callback behavior.
  ffi::UnchangedOr<ffi::Array<PrimExpr>> mapped_allocated_addr = ffi::Unchanged();
  if (!self->allocated_addr.empty()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>,
                                      descended_allocated_addr,
                                      mutator->MutateExpected(self->allocated_addr));
    mapped_allocated_addr = std::move(descended_allocated_addr);
  }
  if (mapped_dtype.UnchangedOrSameAs(self->dtype) && mapped_shape.UnchangedOrSameAs(self->shape) &&
      mapped_strides.UnchangedOrSameAs(self->strides) &&
      mapped_elem_offset.UnchangedOrSameAs(self->elem_offset) &&
      mapped_layout.UnchangedOrSameAs(self->layout) &&
      mapped_allocated_addr.UnchangedOrSameAs(self->allocated_addr)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<BufferTypeNode> copy = ffi::make_object<BufferTypeNode>(*self);
  copy->dtype = std::move(mapped_dtype).ValueOrUnchanged(std::move(copy->dtype));
  copy->shape = std::move(mapped_shape).ValueOrUnchanged(std::move(copy->shape));
  copy->strides = std::move(mapped_strides).ValueOrUnchanged(std::move(copy->strides));
  copy->elem_offset = std::move(mapped_elem_offset).ValueOrUnchanged(std::move(copy->elem_offset));
  copy->layout = std::move(mapped_layout).ValueOrUnchanged(std::move(copy->layout));
  copy->allocated_addr =
      std::move(mapped_allocated_addr).ValueOrUnchanged(std::move(copy->allocated_addr));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny BufferTypeMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                       ffi::AnyView value) noexcept {
  // skips: storage_scope, data_alignment, offset_factor
  BufferTypeNode* self = const_cast<BufferTypeNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferTypeNode>(value));
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
  // allocated_addr is empty outside specialized storage scopes.  Broad callbacks do not see the
  // empty container; present addresses retain normal container descent and callback behavior.
  ffi::UnchangedOr<ffi::Array<PrimExpr>> mapped_allocated_addr = ffi::Unchanged();
  if (!self->allocated_addr.empty()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
        ffi::UnchangedOr<ffi::Array<PrimExpr>>, descended_allocated_addr,
        mutator->MutateExpected(self->allocated_addr, ffi::InplaceMode::kAllow));
    mapped_allocated_addr = std::move(descended_allocated_addr);
  }
  if (mapped_dtype.UnchangedOrSameAs(self->dtype) && mapped_shape.UnchangedOrSameAs(self->shape) &&
      mapped_strides.UnchangedOrSameAs(self->strides) &&
      mapped_elem_offset.UnchangedOrSameAs(self->elem_offset) &&
      mapped_layout.UnchangedOrSameAs(self->layout) &&
      mapped_allocated_addr.UnchangedOrSameAs(self->allocated_addr)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_dtype.IsUnchanged()) self->dtype = std::move(mapped_dtype).ValueUnchecked();
  if (!mapped_shape.IsUnchanged()) self->shape = std::move(mapped_shape).ValueUnchecked();
  if (!mapped_strides.IsUnchanged()) self->strides = std::move(mapped_strides).ValueUnchecked();
  if (!mapped_elem_offset.IsUnchanged())
    self->elem_offset = std::move(mapped_elem_offset).ValueUnchecked();
  if (!mapped_layout.IsUnchanged()) self->layout = std::move(mapped_layout).ValueUnchecked();
  if (!mapped_allocated_addr.IsUnchanged()) {
    self->allocated_addr = std::move(mapped_allocated_addr).ValueUnchecked();
  }
  return ffi::Unchanged().CopyToTVMFFIAny();
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

}  // namespace

BufferType::BufferType(ffi::String storage_scope, PrimType dtype, ffi::Array<PrimExpr> shape,
                       ffi::Array<PrimExpr> strides, PrimExpr elem_offset, int data_alignment,
                       int offset_factor, ffi::Optional<Layout> layout,
                       ffi::Array<PrimExpr> allocated_addr, Span span)
    : Type(ffi::UnsafeInit{}) {
  auto n = ffi::make_object<BufferTypeNode>();
  n->dtype = std::move(dtype);
  n->storage_scope = storage_scope.empty() ? ffi::String("global") : std::move(storage_scope);
  n->shape = std::move(shape);
  n->strides = std::move(strides);
  if (!elem_offset.defined()) {
    elem_offset = IntImm(PrimType(n->DefaultIndexType()), 0);
  }
  n->elem_offset = std::move(elem_offset);
  n->data_alignment =
      data_alignment <= 0 ? static_cast<int>(runtime::kAllocAlignment) : data_alignment;
  n->offset_factor = offset_factor == 0 ? 1 : offset_factor;
  n->layout = std::move(layout);
  n->allocated_addr = std::move(allocated_addr);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BufferTypeNode::RegisterReflection();
  refl::TypeAttrDef<BufferTypeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BufferTypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BufferTypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BufferTypeMaybeInplaceMutate));

  refl::GlobalDef().def(
      "tirx.BufferType",
      [](ffi::String storage_scope, PrimType dtype, ffi::Array<PrimExpr> shape,
         ffi::Array<PrimExpr> strides, PrimExpr elem_offset, int data_alignment, int offset_factor,
         ffi::Optional<Layout> layout, ffi::Array<PrimExpr> allocated_addr, Span span) {
        return BufferType(std::move(storage_scope), std::move(dtype), std::move(shape),
                          std::move(strides), std::move(elem_offset), data_alignment, offset_factor,
                          std::move(layout), std::move(allocated_addr), std::move(span));
      });
}

// TensorRegion
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
            reinterpret_cast<void*>(&BufferRegionTypeMaybeInplaceMutate));

  refl::GlobalDef().def("tirx.BufferRegionType", []() { return BufferRegionType(); });
}

TensorMapType::TensorMapType(Span span) : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<TensorMapTypeNode> n = ffi::make_object<TensorMapTypeNode>();
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TensorMapTypeNode::RegisterReflection();
  refl::TypeAttrDef<TensorMapTypeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&TensorMapTypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&TensorMapTypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&TensorMapTypeMaybeInplaceMutate));

  refl::GlobalDef().def("tirx.TensorMapType", [](Span span) { return TensorMapType(span); });
}

}  // namespace tvm::tirx
