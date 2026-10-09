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
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>

#include <iterator>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {
namespace {
TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> ScopeIdDefStmtVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const ScopeIdDefStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ScopeIdDefStmtNode>(value);
  // ScopeIdDef reflection marks only def_ids as definitions; its extent fields remain uses.
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->def));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ScopeIdDefStmtMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const ScopeIdDefStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ScopeIdDefStmtNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ScopeIdDef>, mapped_def,
                                    mutator->MutateExpected(self->def));
  if (mapped_def.UnchangedOrSameAs(self->def)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<ScopeIdDefStmtNode> copy = ffi::make_object<ScopeIdDefStmtNode>(*self);
  copy->def = std::move(mapped_def).ValueOrUnchanged(std::move(copy->def));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ScopeIdDefStmtMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  ScopeIdDefStmtNode* self = const_cast<ScopeIdDefStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ScopeIdDefStmtNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ScopeIdDef>, mapped_def,
                                    mutator->MutateExpected(self->def, ffi::InplaceMode::kAllow));
  if (mapped_def.UnchangedOrSameAs(self->def)) {
    return ffi::Unchanged();
  }
  if (!mapped_def.IsUnchanged()) self->def = std::move(mapped_def).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE int GetLanesOrVScaleFactor(const PrimType& ty) {
  return ty.IsScalableVector() ? ty.VScaleFactor() : ty.lanes();
}

void ValidateTensorStore(Expr dest, ffi::Array<PrimExpr> indices, PrimExpr value) {
  TensorType type = dest->ty.as_or_throw<TensorType>();
  TVM_FFI_ICHECK_EQ(type->shape.size(), indices.size())
      << "Destination " << dest << " is " << type->shape.size()
      << "-dimensional, cannot be indexed with the " << indices.size()
      << "-dimensional indices provided.";

  for (int i = 0; i < static_cast<int>(indices.size()) - 1; i++) {
    TVM_FFI_ICHECK(indices[i].ty().IsScalar())
        << "Only the last index of a buffer access may be a vector type.";
  }

  bool is_index_scalable = indices.empty() ? false : indices.back().ty().IsScalableVector();
  int16_t buffer_encoded_lanes = static_cast<int16_t>(type->dtype->dtype.lanes);
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
    buffer_dtype = PrimType::ScalableVector(type->dtype.code(), type->dtype.bits(),
                                            buffer_lanes * index_lanes);
  } else {
    buffer_dtype = type->dtype.WithLanes(buffer_lanes * index_lanes);
  }
  if (buffer_dtype != value_ty) {
    TVM_FFI_THROW(TypeError) << "dtype mismatch on TensorStore: "                 //
                             << "buffer's dtype is `" << type->dtype              //
                             << "`, the lanes of indexing are: `" << index_lanes  //
                             << "`, the scalability is: `" << buffer_dtype.IsScalableVector()
                             << "`, but RHS's dtype is `" << value_ty << "`";
  }
}

void ValidateTensorEvaluate(Expr value) {
  TVM_FFI_ICHECK(!value->IsInstance<VarNode>())
      << "A buffer variable cannot be used as a scalar Evaluate value; "
      << "use buffer.data to evaluate its physical pointer";
}
}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  refl::TypeAttrDef<TensorTypeNode>()
      .def("__tensor_store_validate__", ValidateTensorStore)
      .def("__evaluate_validate__", ValidateTensorEvaluate);
}

// ScopeIdDefStmt
ScopeIdDefStmt::ScopeIdDefStmt(ScopeIdDef def, Span span) : Stmt(ffi::UnsafeInit{}) {
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
      .attr(refl::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&ScopeIdDefStmtVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&ScopeIdDefStmtMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&ScopeIdDefStmtMaybeInplaceMutate>());

  refl::GlobalDef().def("tirx.ScopeIdDefStmt",
                        [](ScopeIdDef def, Span span) { return ScopeIdDefStmt(def, span); });
}

}  // namespace tirx
}  // namespace tvm
