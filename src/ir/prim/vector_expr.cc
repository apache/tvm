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

/*! \file vector_expr.cc */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/op.h>

#include <optional>

namespace tvm {
namespace prim {
namespace {
// File-local helper: returns the vscale multiplier if `lanes` is of the form
// `multiplier * vscale()` or `vscale() * multiplier`, nullopt otherwise.
std::optional<int> ExtractVscaleFactor(const PrimExpr& lanes) {
  auto is_vscale = [](const PrimExpr& e) -> bool {
    if (const auto* call = e.as<CallNode>()) {
      return call->op.same_as(prim::builtin::vscale());
    }
    return false;
  };
  if (const auto* mul = lanes.as<MulNode>()) {
    if (const auto* imm = mul->a.as<IntImmNode>(); imm && is_vscale(mul->b)) {
      return static_cast<int>(imm->value);
    }
    if (const auto* imm = mul->b.as<IntImmNode>(); imm && is_vscale(mul->a)) {
      return static_cast<int>(imm->value);
    }
  }
  return std::nullopt;
}
}  // namespace
// Ramp
Ramp::Ramp(PrimExpr base, PrimExpr stride, PrimExpr lanes, Span span) {
  TVM_FFI_ICHECK(base.defined());
  TVM_FFI_ICHECK(stride.defined());
  PrimType base_ty = base.ty();
  PrimType stride_ty = stride.ty();
  TVM_FFI_ICHECK(base_ty.IsScalar());
  TVM_FFI_ICHECK(stride_ty.IsScalar());
  if (stride_ty != base_ty) {
    stride = cast(base_ty, stride);
  }

  ffi::ObjectPtr<RampNode> node = ffi::make_object<RampNode>();
  auto* lanes_as_int = lanes.as<IntImmNode>();
  if (lanes_as_int) {
    int lanes = static_cast<int>(lanes_as_int->value);
    TVM_FFI_ICHECK_GT(lanes, 1);
    node->ExprNode::ty = base_ty.WithLanes(lanes);
    // Stick to int32 lanes for fixed length vectors
    node->lanes = lanes;
  } else { /* scalable vector */
    std::optional<int> vscale_factor = ExtractVscaleFactor(lanes);
    TVM_FFI_ICHECK(vscale_factor) << "Invalid expression for scalable lanes " << lanes;

    node->ExprNode::ty =
        PrimType::ScalableVector(base_ty.code(), base_ty.bits(), vscale_factor.value());
    lanes = Mul(Call(PrimType::Int(32), prim::builtin::vscale(), {}).as_or_throw<PrimExpr>(),
                vscale_factor.value());
    node->lanes = lanes;
  }
  node->base = base;
  node->stride = stride;
  node->span = std::move(span);
  data_ = std::move(node);
}

static TVMFFIAny RampVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const RampNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RampNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->base));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->stride));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->lanes));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

static TVMFFIAny RampMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const RampNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RampNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_base,
                                    mutator->MutateExpected(self->base));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_stride,
                                    mutator->MutateExpected(self->stride));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_lanes,
                                    mutator->MutateExpected(self->lanes));
  if (mapped_base.UnchangedOrSameAs(self->base) && mapped_stride.UnchangedOrSameAs(self->stride) &&
      mapped_lanes.UnchangedOrSameAs(self->lanes)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<RampNode> copy = ffi::make_object<RampNode>(*self);
  copy->base = std::move(mapped_base).ValueOrUnchanged(std::move(copy->base));
  copy->stride = std::move(mapped_stride).ValueOrUnchanged(std::move(copy->stride));
  copy->lanes = std::move(mapped_lanes).ValueOrUnchanged(std::move(copy->lanes));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

static TVMFFIAny RampMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                        ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  RampNode* self = const_cast<RampNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RampNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_base,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->base));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_stride,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->stride));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_lanes,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->lanes));
  if (mapped_base.UnchangedOrSameAs(self->base) && mapped_stride.UnchangedOrSameAs(self->stride) &&
      mapped_lanes.UnchangedOrSameAs(self->lanes)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_base.IsUnchanged()) self->base = std::move(mapped_base).ValueUnchecked();
  if (!mapped_stride.IsUnchanged()) self->stride = std::move(mapped_stride).ValueUnchecked();
  if (!mapped_lanes.IsUnchanged()) self->lanes = std::move(mapped_lanes).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  RampNode::RegisterReflection();
  refl::TypeAttrDef<RampNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&RampVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&RampMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&RampMaybeInplaceMutate));

  refl::GlobalDef().def("ir.prim.Ramp", [](PrimExpr base, PrimExpr stride, PrimExpr lanes,
                                           Span span) { return Ramp(base, stride, lanes, span); });
}

// Broadcast
Broadcast::Broadcast(PrimExpr value, PrimExpr lanes, Span span) {
  TVM_FFI_ICHECK(value.defined());
  PrimType value_ty = value.ty();
  TVM_FFI_ICHECK(value_ty.IsScalar());

  ffi::ObjectPtr<BroadcastNode> node = ffi::make_object<BroadcastNode>();
  auto* lanes_int = lanes.as<IntImmNode>();
  if (lanes_int) {
    int lanes = static_cast<int>(lanes_int->value);
    TVM_FFI_ICHECK_GT(lanes, 1);
    node->ExprNode::ty = value_ty.WithLanes(lanes);
    // Stick to int32 lanes for fixed length vectors
    node->lanes = lanes;
  } else { /* scalable vector */
    std::optional<int> vscale_factor = ExtractVscaleFactor(lanes);
    TVM_FFI_ICHECK(vscale_factor) << "Invalid expression for scalable lanes " << lanes;

    node->ExprNode::ty =
        PrimType::ScalableVector(value_ty.code(), value_ty.bits(), vscale_factor.value());
    lanes = Mul(Call(PrimType::Int(32), prim::builtin::vscale(), {}).as_or_throw<PrimExpr>(),
                vscale_factor.value());
    node->lanes = lanes;
  }
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = node;
}

static TVMFFIAny BroadcastVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const BroadcastNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BroadcastNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->lanes));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

static TVMFFIAny BroadcastMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const BroadcastNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BroadcastNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value,
                                    mutator->MutateExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_lanes,
                                    mutator->MutateExpected(self->lanes));
  if (mapped_value.UnchangedOrSameAs(self->value) && mapped_lanes.UnchangedOrSameAs(self->lanes)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<BroadcastNode> copy = ffi::make_object<BroadcastNode>(*self);
  copy->value = std::move(mapped_value).ValueOrUnchanged(std::move(copy->value));
  copy->lanes = std::move(mapped_lanes).ValueOrUnchanged(std::move(copy->lanes));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

static TVMFFIAny BroadcastMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                             ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  BroadcastNode* self = const_cast<BroadcastNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BroadcastNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_lanes,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->lanes));
  if (mapped_value.UnchangedOrSameAs(self->value) && mapped_lanes.UnchangedOrSameAs(self->lanes)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_value.IsUnchanged()) self->value = std::move(mapped_value).ValueUnchecked();
  if (!mapped_lanes.IsUnchanged()) self->lanes = std::move(mapped_lanes).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BroadcastNode::RegisterReflection();
  refl::TypeAttrDef<BroadcastNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BroadcastVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BroadcastMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BroadcastMaybeInplaceMutate));

  refl::GlobalDef().def("ir.prim.Broadcast", [](PrimExpr value, PrimExpr lanes, Span span) {
    return Broadcast(value, lanes, span);
  });
}

// Shuffle
Shuffle::Shuffle(ffi::Array<PrimExpr> vectors, ffi::Array<PrimExpr> indices, Span span) {
  TVM_FFI_ICHECK_NE(vectors.size(), 0U);
  TVM_FFI_ICHECK_NE(indices.size(), 0U);

  PrimType base_type = vectors[0].ty().WithLanes(1);
  int total_lanes = 0;

  for (PrimExpr val : vectors) {
    PrimType val_ty = val.ty();
    TVM_FFI_ICHECK(val_ty.WithLanes(1)->dtype == base_type->dtype);
    total_lanes += val_ty.lanes();
  }
  TVM_FFI_ICHECK_LE(indices.size(), static_cast<size_t>(total_lanes));

  ffi::ObjectPtr<ShuffleNode> node = ffi::make_object<ShuffleNode>();
  node->ExprNode::ty = base_type.WithLanes(static_cast<int>(indices.size()));
  node->vectors = std::move(vectors);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = node;
}

static TVMFFIAny ShuffleVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const ShuffleNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShuffleNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->vectors));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->indices));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

static TVMFFIAny ShuffleMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const ShuffleNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShuffleNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_vectors,
                                    mutator->MutateExpected(self->vectors));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_indices,
                                    mutator->MutateExpected(self->indices));
  if (mapped_vectors.UnchangedOrSameAs(self->vectors) &&
      mapped_indices.UnchangedOrSameAs(self->indices)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<ShuffleNode> copy = ffi::make_object<ShuffleNode>(*self);
  copy->vectors = std::move(mapped_vectors).ValueOrUnchanged(std::move(copy->vectors));
  copy->indices = std::move(mapped_indices).ValueOrUnchanged(std::move(copy->indices));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

static TVMFFIAny ShuffleMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                           ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  ShuffleNode* self = const_cast<ShuffleNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShuffleNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_vectors,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->vectors));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_indices,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->indices));
  if (mapped_vectors.UnchangedOrSameAs(self->vectors) &&
      mapped_indices.UnchangedOrSameAs(self->indices)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_vectors.IsUnchanged()) self->vectors = std::move(mapped_vectors).ValueUnchecked();
  if (!mapped_indices.IsUnchanged()) self->indices = std::move(mapped_indices).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ShuffleNode::RegisterReflection();
  refl::TypeAttrDef<ShuffleNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&ShuffleVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&ShuffleMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&ShuffleMaybeInplaceMutate));

  refl::GlobalDef().def("ir.prim.Shuffle",
                        [](ffi::Array<PrimExpr> vectors, ffi::Array<PrimExpr> indices, Span span) {
                          return Shuffle(vectors, indices, span);
                        });
}

PrimExpr Shuffle::Concat(ffi::Array<PrimExpr> vectors, Span span) {
  TVM_FFI_ICHECK_NE(vectors.size(), 0);
  if (vectors.size() == 1) {
    return vectors[0];
  }
  ffi::Array<PrimExpr> indices;
  int index = 0;
  for (const PrimExpr& e : vectors) {
    for (int i = 0; i < e.ty().lanes(); ++i) {
      indices.push_back(IntImm::Int32(index++));
    }
  }
  return Shuffle(vectors, indices, span);
}

PrimExpr Shuffle::ExtractElement(PrimExpr vector, int index, Span span) {
  return Shuffle({vector}, {IntImm::Int32(index)}, span);
}

}  // namespace prim
}  // namespace tvm
