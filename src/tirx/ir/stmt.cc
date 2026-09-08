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
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>

#include <utility>

#include "buffer_common.h"

namespace tvm {
namespace tirx {

namespace {

using SubscriptSlice = ffi::Array<ffi::Variant<
    ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>>,
    PrimExpr>>;

ffi::ObjectRef RealizeBufferRegionSubscript(Expr value, SubscriptSlice slice, Span span) {
  BufferRegion source = value.as_or_throw<BufferRegion>();
  TVM_FFI_CHECK_LE(slice.size(), source->region.size(), IndexError)
      << "Too many indices for a " << source->region.size() << "-dimensional buffer region";

  bool all_points = slice.size() == source->region.size();
  for (const auto& item : slice) {
    if (auto descriptor = item.as<ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>,
                                             ffi::Optional<PrimExpr>>>()) {
      all_points = false;
      ffi::Optional<PrimExpr> step = descriptor.value().get<2>();
      TVM_FFI_CHECK(!step.has_value() || is_one(step.value()), ValueError)
          << "BufferRegion slices with a non-unit step are not supported";
    }
  }

  if (all_points) {
    ffi::Array<PrimExpr> indices;
    indices.reserve(slice.size());
    for (size_t i = 0; i < slice.size(); ++i) {
      indices.push_back(source->region[i]->min + slice[i].as<PrimExpr>().value());
    }
    return BufferLoad(source->buffer, indices, span);
  }

  arith::Analyzer analyzer;
  ffi::Array<Range> region;
  region.reserve(source->region.size());
  for (size_t i = 0; i < slice.size(); ++i) {
    const Range& old_range = source->region[i];
    if (auto point = slice[i].as<PrimExpr>()) {
      PrimExpr new_min = old_range->min + point.value();
      region.push_back(Range::FromMinExtent(new_min, IntImm(point.value().ty(), 1)));
    } else {
      auto descriptor = slice[i]
                            .as<ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>,
                                           ffi::Optional<PrimExpr>>>()
                            .value();
      PrimExpr start = descriptor.get<0>().value_or(IntImm(old_range->extent.ty(), 0));
      PrimExpr stop = descriptor.get<1>().value_or(old_range->extent);
      region.push_back(
          Range::FromMinExtent(old_range->min + start, analyzer->Simplify(stop - start)));
    }
  }
  for (size_t i = slice.size(); i < source->region.size(); ++i) {
    region.push_back(source->region[i]);
  }
  return BufferRegion(source->buffer, region, span);
}

/*!
 * \brief Whether an integer literal can be represented exactly by `ty`.
 * \note Mirrors the range checks performed by the IntImm constructor.
 */
bool IntImmValueFits(int64_t value, const PrimType& ty) {
  int bits = ty.bits();
  if (ty.MatchesCode(DLDataTypeCode::kDLUInt)) {
    return value >= 0 && (bits >= 64 || value < (int64_t{1} << bits));
  }
  if (bits >= 64) return true;
  return value >= -(int64_t{1} << (bits - 1)) && value < (int64_t{1} << (bits - 1));
}

// Structural traversal hooks

TVMFFIAny BindVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const BindNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BindNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindNonRecursive, [&]() { return visitor->VisitExpected(self->var); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny BindMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const BindNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BindNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      Var, mapped_var, mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
        return mutator->MutateExpected(self->var);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Expr, mapped_value, mutator->MutateExpected(self->value));
  if (mapped_var.same_as(self->var) && mapped_value.same_as(self->value)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<BindNode> copy = ffi::make_object<BindNode>(*self);
  copy->var = std::move(mapped_var);
  copy->value = std::move(mapped_value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny BindMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  BindNode* self = const_cast<BindNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BindNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      Var, mapped_var, mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
        return mutator->MaybeInplaceMutateIfUniqueExpected(self->var);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Expr, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  if (mapped_var.same_as(self->var) && mapped_value.same_as(self->value)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->var = std::move(mapped_var);
  self->value = std::move(mapped_value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny AttrStmtVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: attr_key
  const AttrStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AttrStmtNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->node));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny AttrStmtMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attr_key
  const AttrStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AttrStmtNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Any, mapped_node, mutator->MutateExpected(self->node));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_value, mutator->MutateExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped_body, mutator->MutateExpected(self->body));
  if (mapped_node.same_as(self->node) && mapped_value.same_as(self->value) &&
      mapped_body.same_as(self->body)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<AttrStmtNode> copy = ffi::make_object<AttrStmtNode>(*self);
  copy->node = std::move(mapped_node);
  copy->value = std::move(mapped_value);
  copy->body = std::move(mapped_body);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny AttrStmtMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                     ffi::AnyView value) noexcept {
  // skips: attr_key
  AttrStmtNode* self = const_cast<AttrStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AttrStmtNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Any, mapped_node,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->node));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped_body,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->body));
  if (mapped_node.same_as(self->node) && mapped_value.same_as(self->value) &&
      mapped_body.same_as(self->body)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->node = std::move(mapped_node);
  self->value = std::move(mapped_value);
  self->body = std::move(mapped_body);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny AssertStmtVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: error_kind and message_parts, which are constant assertion metadata.
  const AssertStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AssertStmtNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->condition));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny AssertStmtMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: error_kind and message_parts, which are constant assertion metadata.
  const AssertStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AssertStmtNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_condition,
                                    mutator->MutateExpected(self->condition));
  if (mapped_condition.same_as(self->condition)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<AssertStmtNode> copy = ffi::make_object<AssertStmtNode>(*self);
  copy->condition = std::move(mapped_condition);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny AssertStmtMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                       ffi::AnyView value) noexcept {
  // skips: error_kind and message_parts, which are constant assertion metadata.
  AssertStmtNode* self = const_cast<AssertStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AssertStmtNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_condition,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->condition));
  if (mapped_condition.same_as(self->condition)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->condition = std::move(mapped_condition);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny ForVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: kind and constant annotations; unlike SBlock annotations, these carry no expressions.
  const ForNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ForNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindNonRecursive, [&]() { return visitor->VisitExpected(self->loop_var); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->min));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->extent));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->thread_binding));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->step));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny ForMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: kind and constant annotations; unlike SBlock annotations, these carry no expressions.
  const ForNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ForNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      PrimVar, mapped_loop_var, mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
        return mutator->MutateExpected(self->loop_var);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_min, mutator->MutateExpected(self->min));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_extent, mutator->MutateExpected(self->extent));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped_body, mutator->MutateExpected(self->body));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Optional<IterVar>, mapped_thread_binding,
                                    mutator->MutateExpected(self->thread_binding));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Optional<PrimExpr>, mapped_step,
                                    mutator->MutateExpected(self->step));
  if (mapped_loop_var.same_as(self->loop_var) && mapped_min.same_as(self->min) &&
      mapped_extent.same_as(self->extent) && mapped_body.same_as(self->body) &&
      mapped_thread_binding.same_as(self->thread_binding) && mapped_step.same_as(self->step)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<ForNode> copy = ffi::make_object<ForNode>(*self);
  copy->loop_var = std::move(mapped_loop_var);
  copy->min = std::move(mapped_min);
  copy->extent = std::move(mapped_extent);
  copy->body = std::move(mapped_body);
  copy->thread_binding = std::move(mapped_thread_binding);
  copy->step = std::move(mapped_step);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny ForMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: kind and constant annotations; unlike SBlock annotations, these carry no expressions.
  ForNode* self = const_cast<ForNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ForNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      PrimVar, mapped_loop_var, mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
        return mutator->MaybeInplaceMutateIfUniqueExpected(self->loop_var);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_min,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->min));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_extent,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->extent));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped_body,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->body));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::Optional<IterVar>, mapped_thread_binding,
      mutator->MaybeInplaceMutateIfUniqueExpected(self->thread_binding));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Optional<PrimExpr>, mapped_step,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->step));
  if (mapped_loop_var.same_as(self->loop_var) && mapped_min.same_as(self->min) &&
      mapped_extent.same_as(self->extent) && mapped_body.same_as(self->body) &&
      mapped_thread_binding.same_as(self->thread_binding) && mapped_step.same_as(self->step)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->loop_var = std::move(mapped_loop_var);
  self->min = std::move(mapped_min);
  self->extent = std::move(mapped_extent);
  self->body = std::move(mapped_body);
  self->thread_binding = std::move(mapped_thread_binding);
  self->step = std::move(mapped_step);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny WhileVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const WhileNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const WhileNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny WhileMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const WhileNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const WhileNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_condition,
                                    mutator->MutateExpected(self->condition));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped_body, mutator->MutateExpected(self->body));
  if (mapped_condition.same_as(self->condition) && mapped_body.same_as(self->body)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<WhileNode> copy = ffi::make_object<WhileNode>(*self);
  copy->condition = std::move(mapped_condition);
  copy->body = std::move(mapped_body);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny WhileMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  WhileNode* self = const_cast<WhileNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const WhileNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_condition,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->condition));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped_body,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->body));
  if (mapped_condition.same_as(self->condition) && mapped_body.same_as(self->body)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->condition = std::move(mapped_condition);
  self->body = std::move(mapped_body);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny ReturnVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const ReturnNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ReturnNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny ReturnMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const ReturnNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ReturnNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Expr, mapped_value, mutator->MutateExpected(self->value));
  if (mapped_value.same_as(self->value)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<ReturnNode> copy = ffi::make_object<ReturnNode>(*self);
  copy->value = std::move(mapped_value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny ReturnMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                   ffi::AnyView value) noexcept {
  ReturnNode* self = const_cast<ReturnNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ReturnNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Expr, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  if (mapped_value.same_as(self->value)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->value = std::move(mapped_value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny BreakVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny BreakMutate(ffi::StructuralMutatorObj*, ffi::AnyView value) noexcept {
  const BreakNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BreakNode>(value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny BreakMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView value) noexcept {
  const BreakNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BreakNode>(value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny ContinueVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny ContinueMutate(ffi::StructuralMutatorObj*, ffi::AnyView value) noexcept {
  const ContinueNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ContinueNode>(value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny ContinueMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView value) noexcept {
  const ContinueNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ContinueNode>(value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny DeclBufferVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const DeclBufferNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DeclBufferNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindNonRecursive, [&]() { return visitor->VisitExpected(self->buffer); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->data));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny DeclBufferMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const DeclBufferNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DeclBufferNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      BufferVar, mapped_buffer, mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
        return mutator->MutateExpected(self->buffer);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Expr, mapped_data, mutator->MutateExpected(self->data));
  if (mapped_buffer.same_as(self->buffer) && mapped_data.same_as(self->data)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<DeclBufferNode> copy = ffi::make_object<DeclBufferNode>(*self);
  copy->buffer = std::move(mapped_buffer);
  copy->data = std::move(mapped_data);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny DeclBufferMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                       ffi::AnyView value) noexcept {
  DeclBufferNode* self = const_cast<DeclBufferNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DeclBufferNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      BufferVar, mapped_buffer, mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
        return mutator->MaybeInplaceMutateIfUniqueExpected(self->buffer);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Expr, mapped_data,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->data));
  if (mapped_buffer.same_as(self->buffer) && mapped_data.same_as(self->data)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->buffer = std::move(mapped_buffer);
  self->data = std::move(mapped_data);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny AllocBufferVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: constant annotations; unlike SBlock annotations, these carry no expressions.
  const AllocBufferNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AllocBufferNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindNonRecursive, [&]() { return visitor->VisitExpected(self->buffer); }));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny AllocBufferMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: constant annotations; unlike SBlock annotations, these carry no expressions.
  const AllocBufferNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AllocBufferNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      BufferVar, mapped_buffer, mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
        return mutator->MutateExpected(self->buffer);
      }));
  if (mapped_buffer.same_as(self->buffer)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<AllocBufferNode> copy = ffi::make_object<AllocBufferNode>(*self);
  copy->buffer = std::move(mapped_buffer);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny AllocBufferMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                        ffi::AnyView value) noexcept {
  // skips: constant annotations; unlike SBlock annotations, these carry no expressions.
  AllocBufferNode* self = const_cast<AllocBufferNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const AllocBufferNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      BufferVar, mapped_buffer, mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
        return mutator->MaybeInplaceMutateIfUniqueExpected(self->buffer);
      }));
  if (mapped_buffer.same_as(self->buffer)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->buffer = std::move(mapped_buffer);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny SeqStmtVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const SeqStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->seq));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

bool IsSeqStmtNoOp(const Stmt& stmt) {
  const auto* evaluate = stmt.as<EvaluateNode>();
  const auto* value = evaluate == nullptr ? nullptr : evaluate->value.as<IntImmNode>();
  return value != nullptr && value->value == 0;
}

void AppendSeqStmtResult(ffi::Array<Stmt>* output, Stmt mapped) {
  if (IsSeqStmtNoOp(mapped)) {
    return;
  }
  if (const auto* nested = mapped.as<SeqStmtNode>()) {
    for (const Stmt& stmt : nested->seq) {
      output->push_back(stmt);
    }
  } else {
    output->push_back(std::move(mapped));
  }
}

TVMFFIAny MutateSeqStmtChanged(ffi::StructuralMutatorObj* mutator, const SeqStmtNode* self,
                               int64_t index, Stmt mapped) noexcept {
  const int64_t size = static_cast<int64_t>(self->seq.size());
  ffi::ObjectPtr<ffi::ArrayObj> output_obj = ffi::ArrayObj::CreateRepeated(size, ffi::Any());
  output_obj->InitRange(0, self->seq.begin(), self->seq.begin() + index);
  output_obj->resize(index);
  ffi::Array<Stmt> output(std::move(output_obj));
  AppendSeqStmtResult(&output, std::move(mapped));
  for (int64_t i = index + 1; i < size; ++i) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped, mutator->MutateExpected(self->seq[i]));
    AppendSeqStmtResult(&output, std::move(mapped));
  }
  if (output.empty()) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(Evaluate(0)));
  }
  if (output.size() == 1) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(output[0]));
  }
  ffi::ObjectPtr<SeqStmtNode> copy = ffi::make_object<SeqStmtNode>(*self);
  copy->seq = std::move(output);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny MutateSeqStmtRaw(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const SeqStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(value);
  const int64_t size = static_cast<int64_t>(self->seq.size());
  for (int64_t i = 0; i < size; ++i) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped, mutator->MutateExpected(self->seq[i]));
    if (!self->seq[i].same_as(mapped)) {
      return MutateSeqStmtChanged(mutator, self, i, std::move(mapped));
    }
  }
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny InplaceSplice(ffi::StructuralMutatorObj* mutator, SeqStmtNode* self, ffi::ArrayObj* seq,
                        int64_t total, int64_t index, const SeqStmtNode* nested) noexcept {
  const int64_t size = static_cast<int64_t>(seq->size());
  ffi::Array<Stmt> output;
  output.reserve(size + static_cast<int64_t>(nested->seq.size()) - 1);
  for (int64_t i = 0; i < total; ++i) {
    output.push_back(seq->begin()[i].cast<Stmt>());
  }
  for (const Stmt& stmt : nested->seq) {
    output.push_back(stmt);
  }
  for (int64_t i = index + 1; i < size; ++i) {
    const ffi::Any& item = seq->begin()[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));
    AppendSeqStmtResult(&output, std::move(mapped));
  }
  if (output.empty()) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(Evaluate(0)));
  }
  if (output.size() == 1) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(output[0]));
  }
  self->seq = std::move(output);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny MaybeInplaceMutateSeqStmtChanged(ffi::StructuralMutatorObj* mutator, SeqStmtNode* self,
                                           ffi::ArrayObj* seq, int64_t index,
                                           Stmt mapped) noexcept {
  const int64_t size = static_cast<int64_t>(seq->size());
  int64_t total = index;
  if (IsSeqStmtNoOp(mapped)) {
    // Drop Evaluate(0), matching SeqStmt::Flatten.
  } else if (const auto* nested = mapped.as<SeqStmtNode>()) {
    const int64_t nested_size = static_cast<int64_t>(nested->seq.size());
    if (total + nested_size > index + 1) {
      return InplaceSplice(mutator, self, seq, total, index, nested);
    }
    for (const Stmt& stmt : nested->seq) {
      seq->SetItemAfterCheck(total++, ffi::Any(stmt));
    }
  } else {
    seq->SetItemAfterCheck(total++, ffi::Any(std::move(mapped)));
  }
  for (int64_t i = index + 1; i < size; ++i) {
    const ffi::Any& item = seq->begin()[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));
    if (IsSeqStmtNoOp(mapped)) {
      continue;
    }
    if (const auto* nested = mapped.as<SeqStmtNode>()) {
      const int64_t nested_size = static_cast<int64_t>(nested->seq.size());
      if (total + nested_size > i + 1) {
        return InplaceSplice(mutator, self, seq, total, i, nested);
      }
      for (const Stmt& stmt : nested->seq) {
        seq->SetItemAfterCheck(total++, ffi::Any(stmt));
      }
      continue;
    }
    if (total != i || !item.same_as(mapped)) {
      seq->SetItemAfterCheck(total, ffi::Any(std::move(mapped)));
    }
    ++total;
  }
  seq->resize(total);
  if (total == 0) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(Evaluate(0)));
  }
  if (total == 1) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(seq->begin()[0].cast<Stmt>()));
  }
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny MaybeInplaceMutateSeqStmtRaw(ffi::StructuralMutatorObj* mutator,
                                       ffi::AnyView value) noexcept {
  SeqStmtNode* self = const_cast<SeqStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtNode>(value));
  // The engine establishes ownership of the SeqStmt, but seq is a field and needs its own check.
  if (!self->seq.unique()) {
    return MutateSeqStmtRaw(mutator, value);
  }
  ffi::ArrayObj* seq = self->seq.GetArrayObj();
  const int64_t size = static_cast<int64_t>(seq->size());
  for (int64_t i = 0; i < size; ++i) {
    // Borrow the storage slot so the element stays unique during in-place dispatch.
    const ffi::Any& item = seq->begin()[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));
    if (!item.same_as(mapped)) {
      return MaybeInplaceMutateSeqStmtChanged(mutator, self, seq, i, std::move(mapped));
    }
  }
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny SeqStmtMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  return MutateSeqStmtRaw(mutator, value);
}

TVMFFIAny SeqStmtMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                    ffi::AnyView value) noexcept {
  return MaybeInplaceMutateSeqStmtRaw(mutator, value);
}

TVMFFIAny IfThenElseVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const IfThenElseNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfThenElseNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->then_case));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->else_case));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny IfThenElseMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const IfThenElseNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfThenElseNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_condition,
                                    mutator->MutateExpected(self->condition));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped_then_case,
                                    mutator->MutateExpected(self->then_case));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Optional<Stmt>, mapped_else_case,
                                    mutator->MutateExpected(self->else_case));
  if (mapped_condition.same_as(self->condition) && mapped_then_case.same_as(self->then_case) &&
      mapped_else_case.same_as(self->else_case)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<IfThenElseNode> copy = ffi::make_object<IfThenElseNode>(*self);
  copy->condition = std::move(mapped_condition);
  copy->then_case = std::move(mapped_then_case);
  copy->else_case = std::move(mapped_else_case);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny IfThenElseMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                       ffi::AnyView value) noexcept {
  IfThenElseNode* self = const_cast<IfThenElseNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfThenElseNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_condition,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->condition));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped_then_case,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->then_case));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Optional<Stmt>, mapped_else_case,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->else_case));
  if (mapped_condition.same_as(self->condition) && mapped_then_case.same_as(self->then_case) &&
      mapped_else_case.same_as(self->else_case)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->condition = std::move(mapped_condition);
  self->then_case = std::move(mapped_then_case);
  self->else_case = std::move(mapped_else_case);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny EvaluateVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const EvaluateNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny EvaluateMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const EvaluateNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Expr, mapped_value, mutator->MutateExpected(self->value));
  if (mapped_value.same_as(self->value)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<EvaluateNode> copy = ffi::make_object<EvaluateNode>(*self);
  copy->value = std::move(mapped_value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny EvaluateMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                     ffi::AnyView value) noexcept {
  EvaluateNode* self = const_cast<EvaluateNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const EvaluateNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Expr, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  if (mapped_value.same_as(self->value)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->value = std::move(mapped_value);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny BufferStoreVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const BufferStoreNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferStoreNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->buffer));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->indices));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny BufferStoreMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const BufferStoreNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferStoreNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(BufferVar, mapped_buffer,
                                    mutator->MutateExpected(self->buffer));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_value, mutator->MutateExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<PrimExpr>, mapped_indices,
                                    mutator->MutateExpected(self->indices));
  if (mapped_buffer.same_as(self->buffer) && mapped_value.same_as(self->value) &&
      mapped_indices.same_as(self->indices)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<BufferStoreNode> copy = ffi::make_object<BufferStoreNode>(*self);
  copy->buffer = std::move(mapped_buffer);
  copy->value = std::move(mapped_value);
  copy->indices = std::move(mapped_indices);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny BufferStoreMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                        ffi::AnyView value) noexcept {
  BufferStoreNode* self = const_cast<BufferStoreNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferStoreNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(BufferVar, mapped_buffer,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->buffer));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_value,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<PrimExpr>, mapped_indices,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->indices));
  if (mapped_buffer.same_as(self->buffer) && mapped_value.same_as(self->value) &&
      mapped_indices.same_as(self->indices)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->buffer = std::move(mapped_buffer);
  self->value = std::move(mapped_value);
  self->indices = std::move(mapped_indices);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny BufferRegionVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const BufferRegionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferRegionNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->buffer));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->region));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny BufferRegionMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const BufferRegionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferRegionNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(BufferVar, mapped_buffer,
                                    mutator->MutateExpected(self->buffer));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<Range>, mapped_region,
                                    mutator->MutateExpected(self->region));
  if (mapped_buffer.same_as(self->buffer) && mapped_region.same_as(self->region)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<BufferRegionNode> copy = ffi::make_object<BufferRegionNode>(*self);
  copy->buffer = std::move(mapped_buffer);
  copy->region = std::move(mapped_region);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny BufferRegionMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                         ffi::AnyView value) noexcept {
  BufferRegionNode* self = const_cast<BufferRegionNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BufferRegionNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(BufferVar, mapped_buffer,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->buffer));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<Range>, mapped_region,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->region));
  if (mapped_buffer.same_as(self->buffer) && mapped_region.same_as(self->region)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->buffer = std::move(mapped_buffer);
  self->region = std::move(mapped_region);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny MatchBufferRegionVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const MatchBufferRegionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MatchBufferRegionNode>(
          value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindNonRecursive, [&]() { return visitor->VisitExpected(self->buffer); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->source));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny MatchBufferRegionMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const MatchBufferRegionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MatchBufferRegionNode>(
          value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      BufferVar, mapped_buffer, mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
        return mutator->MutateExpected(self->buffer);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(BufferRegion, mapped_source,
                                    mutator->MutateExpected(self->source));
  if (mapped_buffer.same_as(self->buffer) && mapped_source.same_as(self->source)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<MatchBufferRegionNode> copy = ffi::make_object<MatchBufferRegionNode>(*self);
  copy->buffer = std::move(mapped_buffer);
  copy->source = std::move(mapped_source);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny MatchBufferRegionMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                              ffi::AnyView value) noexcept {
  MatchBufferRegionNode* self = const_cast<MatchBufferRegionNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MatchBufferRegionNode>(
          value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      BufferVar, mapped_buffer, mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
        return mutator->MaybeInplaceMutateIfUniqueExpected(self->buffer);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(BufferRegion, mapped_source,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->source));
  if (mapped_buffer.same_as(self->buffer) && mapped_source.same_as(self->source)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->buffer = std::move(mapped_buffer);
  self->source = std::move(mapped_source);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny SBlockVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: name_hint
  const SBlockNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->iter_vars));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->reads));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->writes));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(
      visitor->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive,
                                 [&]() { return visitor->VisitExpected(self->alloc_buffers); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->match_buffers));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->annotations));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->init));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny SBlockMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: name_hint
  const SBlockNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<IterVar>, mapped_iter_vars,
                                    mutator->MutateExpected(self->iter_vars));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<BufferRegion>, mapped_reads,
                                    mutator->MutateExpected(self->reads));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<BufferRegion>, mapped_writes,
                                    mutator->MutateExpected(self->writes));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::Array<BufferVar>, mapped_alloc_buffers,
      mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive,
                                 [&]() { return mutator->MutateExpected(self->alloc_buffers); }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<MatchBufferRegion>, mapped_match_buffers,
                                    mutator->MutateExpected(self->match_buffers));
  auto mapped_annotations_result = mutator->MutateExpected(self->annotations);
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(mapped_annotations_result);
  bool mapped_annotations_type_ok =
      ffi::details::AnyUnsafe::CheckAnyStrict<ffi::Map<ffi::String, ffi::Any>>(
          ffi::details::ExpectedUnsafe::GetData(mapped_annotations_result));
  if (TVM_FFI_PREDICT_FALSE(!mapped_annotations_type_ok)) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(ffi::details::SMutateDeclaredTypeError());
  }
  ffi::Map<ffi::String, ffi::Any> mapped_annotations =
      ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<ffi::Map<ffi::String, ffi::Any>>(
          std::move(ffi::details::ExpectedUnsafe::GetData(mapped_annotations_result)));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Optional<Stmt>, mapped_init,
                                    mutator->MutateExpected(self->init));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped_body, mutator->MutateExpected(self->body));
  if (mapped_iter_vars.same_as(self->iter_vars) && mapped_reads.same_as(self->reads) &&
      mapped_writes.same_as(self->writes) && mapped_alloc_buffers.same_as(self->alloc_buffers) &&
      mapped_match_buffers.same_as(self->match_buffers) &&
      mapped_annotations.same_as(self->annotations) && mapped_init.same_as(self->init) &&
      mapped_body.same_as(self->body)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<SBlockNode> copy = ffi::make_object<SBlockNode>(*self);
  copy->iter_vars = std::move(mapped_iter_vars);
  copy->reads = std::move(mapped_reads);
  copy->writes = std::move(mapped_writes);
  copy->alloc_buffers = std::move(mapped_alloc_buffers);
  copy->match_buffers = std::move(mapped_match_buffers);
  copy->annotations = std::move(mapped_annotations);
  copy->init = std::move(mapped_init);
  copy->body = std::move(mapped_body);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny SBlockMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                   ffi::AnyView value) noexcept {
  // skips: name_hint
  SBlockNode* self = const_cast<SBlockNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<IterVar>, mapped_iter_vars,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->iter_vars));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<BufferRegion>, mapped_reads,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->reads));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<BufferRegion>, mapped_writes,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->writes));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::Array<BufferVar>, mapped_alloc_buffers,
      mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
        return mutator->MaybeInplaceMutateIfUniqueExpected(self->alloc_buffers);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::Array<MatchBufferRegion>, mapped_match_buffers,
      mutator->MaybeInplaceMutateIfUniqueExpected(self->match_buffers));
  auto mapped_annotations_result = mutator->MaybeInplaceMutateIfUniqueExpected(self->annotations);
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(mapped_annotations_result);
  bool mapped_annotations_type_ok =
      ffi::details::AnyUnsafe::CheckAnyStrict<ffi::Map<ffi::String, ffi::Any>>(
          ffi::details::ExpectedUnsafe::GetData(mapped_annotations_result));
  if (TVM_FFI_PREDICT_FALSE(!mapped_annotations_type_ok)) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(ffi::details::SMutateDeclaredTypeError());
  }
  ffi::Map<ffi::String, ffi::Any> mapped_annotations =
      ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<ffi::Map<ffi::String, ffi::Any>>(
          std::move(ffi::details::ExpectedUnsafe::GetData(mapped_annotations_result)));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Optional<Stmt>, mapped_init,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->init));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Stmt, mapped_body,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->body));
  if (mapped_iter_vars.same_as(self->iter_vars) && mapped_reads.same_as(self->reads) &&
      mapped_writes.same_as(self->writes) && mapped_alloc_buffers.same_as(self->alloc_buffers) &&
      mapped_match_buffers.same_as(self->match_buffers) &&
      mapped_annotations.same_as(self->annotations) && mapped_init.same_as(self->init) &&
      mapped_body.same_as(self->body)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->iter_vars = std::move(mapped_iter_vars);
  self->reads = std::move(mapped_reads);
  self->writes = std::move(mapped_writes);
  self->alloc_buffers = std::move(mapped_alloc_buffers);
  self->match_buffers = std::move(mapped_match_buffers);
  self->annotations = std::move(mapped_annotations);
  self->init = std::move(mapped_init);
  self->body = std::move(mapped_body);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny ScopeIdDefStmtVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const ScopeIdDefStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ScopeIdDefStmtNode>(value);
  // ScopeIdDef reflection marks only def_ids as definitions; its extent fields remain uses.
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->def));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny ScopeIdDefStmtMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const ScopeIdDefStmtNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ScopeIdDefStmtNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ScopeIdDef, mapped_def, mutator->MutateExpected(self->def));
  if (mapped_def.same_as(self->def)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<ScopeIdDefStmtNode> copy = ffi::make_object<ScopeIdDefStmtNode>(*self);
  copy->def = std::move(mapped_def);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny ScopeIdDefStmtMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                           ffi::AnyView value) noexcept {
  ScopeIdDefStmtNode* self = const_cast<ScopeIdDefStmtNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ScopeIdDefStmtNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ScopeIdDef, mapped_def,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->def));
  if (mapped_def.same_as(self->def)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->def = std::move(mapped_def);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

TVMFFIAny SBlockRealizeVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const SBlockRealizeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockRealizeNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->iter_values));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->predicate));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->block));
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny SBlockRealizeMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const SBlockRealizeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockRealizeNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<PrimExpr>, mapped_iter_values,
                                    mutator->MutateExpected(self->iter_values));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_predicate,
                                    mutator->MutateExpected(self->predicate));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(SBlock, mapped_block, mutator->MutateExpected(self->block));
  if (mapped_iter_values.same_as(self->iter_values) && mapped_predicate.same_as(self->predicate) &&
      mapped_block.same_as(self->block)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<SBlockRealizeNode> copy = ffi::make_object<SBlockRealizeNode>(*self);
  copy->iter_values = std::move(mapped_iter_values);
  copy->predicate = std::move(mapped_predicate);
  copy->block = std::move(mapped_block);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny SBlockRealizeMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                          ffi::AnyView value) noexcept {
  SBlockRealizeNode* self = const_cast<SBlockRealizeNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockRealizeNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<PrimExpr>, mapped_iter_values,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->iter_values));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(PrimExpr, mapped_predicate,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->predicate));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(SBlock, mapped_block,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->block));
  if (mapped_iter_values.same_as(self->iter_values) && mapped_predicate.same_as(self->predicate) &&
      mapped_block.same_as(self->block)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->iter_values = std::move(mapped_iter_values);
  self->predicate = std::move(mapped_predicate);
  self->block = std::move(mapped_block);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() { StmtNode::RegisterReflection(); }

// Bind
Bind::Bind(Var var, Expr value, Span span) {
  TVM_FFI_ICHECK(value.defined());
  TVM_FFI_ICHECK(ffi::StructuralEqual()(value->ty, var->ty));

  ffi::ObjectPtr<BindNode> node = ffi::make_object<BindNode>();
  node->var = std::move(var);
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BindNode::RegisterReflection();
  refl::GlobalDef().def("tirx.Bind",
                        [](Var var, Expr value, Span span) { return Bind(var, value, span); });
  refl::TypeAttrDef<BindNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BindVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BindMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BindMaybeInplaceMutate));
}

// AttrStmt
AttrStmt::AttrStmt(ffi::Any node, ffi::String attr_key, PrimExpr value, Stmt body, Span span) {
  auto n = ffi::make_object<AttrStmtNode>();
  n->node = node;
  n->attr_key = std::move(attr_key);
  n->value = std::move(value);
  n->body = std::move(body);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  AttrStmtNode::RegisterReflection();
  refl::GlobalDef().def("tirx.AttrStmt",
                        [](Any node, ffi::String attr_key, PrimExpr value, Stmt body, Span span) {
                          return AttrStmt(node, attr_key, value, body, span);
                        });
  refl::TypeAttrDef<AttrStmtNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&AttrStmtVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&AttrStmtMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&AttrStmtMaybeInplaceMutate));
}

// AssertStmt
AssertStmt::AssertStmt(PrimExpr condition, prim::StringImm error_kind,
                       ffi::Array<prim::StringImm> message_parts, Span span) {
  TVM_FFI_ICHECK(condition.defined());
  PrimType condition_ty = condition.ty();
  TVM_FFI_ICHECK(condition_ty.MatchesCode(DLDataTypeCode::kDLBool))
      << "AssertStmt should have boolean condition, "
      << "but received " << condition << " with dtype " << condition_ty;
  TVM_FFI_ICHECK(error_kind.defined());

  ffi::ObjectPtr<AssertStmtNode> node = ffi::make_object<AssertStmtNode>();
  node->condition = std::move(condition);
  node->error_kind = std::move(error_kind);
  node->message_parts = std::move(message_parts);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  AssertStmtNode::RegisterReflection();
  refl::GlobalDef().def(
      "tirx.AssertStmt",
      [](PrimExpr condition, prim::StringImm error_kind, ffi::Array<prim::StringImm> message_parts,
         Span span) { return AssertStmt(condition, error_kind, message_parts, span); });
  refl::TypeAttrDef<AssertStmtNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&AssertStmtVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&AssertStmtMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&AssertStmtMaybeInplaceMutate));
}

// For
For::For(PrimVar loop_var, PrimExpr min, PrimExpr extent, ForKind kind, Stmt body,
         ffi::Optional<IterVar> thread_binding, ffi::Map<ffi::String, Any> annotations,
         ffi::Optional<PrimExpr> step, Span span) {
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

  ffi::ObjectPtr<ForNode> node = ffi::make_object<ForNode>();
  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->extent = std::move(extent);
  node->kind = kind;
  node->body = std::move(body);
  node->thread_binding = std::move(thread_binding);
  node->annotations = std::move(annotations);
  node->step = std::move(step);
  node->span = std::move(span);
  data_ = std::move(node);
}

bool ForNode::HasTrivialStep() const { return !step.has_value() || is_one(*step); }

std::ostream& operator<<(std::ostream& out, ForKind type) {  // NOLINT(*)
  switch (type) {
    case ForKind::kSerial:
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
    case ForKind::kThreadBinding:
      out << "launch_thread";
      break;
  }
  return out;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ForNode::RegisterReflection();
  refl::GlobalDef().def("tirx.For", [](PrimVar loop_var, PrimExpr min, PrimExpr extent, int kind,
                                       Stmt body, ffi::Optional<IterVar> thread_binding,
                                       ffi::Optional<ffi::Map<ffi::String, Any>> annotations,
                                       ffi::Optional<PrimExpr> step, Span span) {
    return For(loop_var, min, extent, static_cast<ForKind>(kind), body, thread_binding,
               annotations.value_or(ffi::Map<ffi::String, Any>()), step, span);
  });
  refl::TypeAttrDef<ForNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&ForVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&ForMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&ForMaybeInplaceMutate));
}

// While
While::While(PrimExpr condition, Stmt body, Span span) {
  TVM_FFI_ICHECK(condition.defined());
  TVM_FFI_ICHECK(condition.ty().IsScalar());
  TVM_FFI_ICHECK(body.defined());

  ffi::ObjectPtr<WhileNode> node = ffi::make_object<WhileNode>();
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  WhileNode::RegisterReflection();
  refl::GlobalDef().def("tirx.While", [](PrimExpr condition, Stmt body, Span span) {
    return While(condition, body, span);
  });
  refl::TypeAttrDef<WhileNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&WhileVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&WhileMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&WhileMaybeInplaceMutate));
}

// Return
Return::Return(Expr value, Span span) {
  TVM_FFI_ICHECK(value.defined());

  ffi::ObjectPtr<ReturnNode> node = ffi::make_object<ReturnNode>();
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ReturnNode::RegisterReflection();
  refl::GlobalDef().def("tirx.Return", [](Expr value, Span span) { return Return(value, span); });
  refl::TypeAttrDef<ReturnNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&ReturnVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&ReturnMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&ReturnMaybeInplaceMutate));
}

// Break
Break::Break(Span span) {
  ffi::ObjectPtr<BreakNode> node = ffi::make_object<BreakNode>();
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BreakNode::RegisterReflection();
  refl::GlobalDef().def("tirx.Break", [](Span span) { return Break(span); });
  refl::TypeAttrDef<BreakNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BreakVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BreakMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BreakMaybeInplaceMutate));
}

// Continue
Continue::Continue(Span span) {
  ffi::ObjectPtr<ContinueNode> node = ffi::make_object<ContinueNode>();
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ContinueNode::RegisterReflection();
  refl::GlobalDef().def("tirx.Continue", [](Span span) { return Continue(span); });
  refl::TypeAttrDef<ContinueNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&ContinueVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&ContinueMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&ContinueMaybeInplaceMutate));
}

// DeclBuffer
DeclBuffer::DeclBuffer(BufferVar buffer, Expr data, Span span) {
  // Enforce storage scope rules for DeclBuffer.
  std::string scope = static_cast<std::string>(buffer.scope());
  if (scope.empty()) {
    scope = "global";
  }
  if (scope == "tmem") {
    TVM_FFI_ICHECK_EQ(buffer->allocated_addr.size(), 1U)
        << "ValueError: For `tmem` scope, DeclBuffer requires exactly one `allocated_addr` "
           "PrimExpr";
  } else if (scope == "global" || scope == "shared" || scope == "shared.dyn" || scope == "local") {
    TVM_FFI_ICHECK(buffer->allocated_addr.empty())
        << "ValueError: For `" << scope << "` scope, DeclBuffer does not accept `allocated_addr`";
  }
  ffi::ObjectPtr<DeclBufferNode> node = ffi::make_object<DeclBufferNode>();
  node->buffer = std::move(buffer);
  node->data = std::move(data);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  DeclBufferNode::RegisterReflection();
  refl::GlobalDef().def("tirx.DeclBuffer", [](BufferVar buffer, Expr data, Span span) {
    return DeclBuffer(buffer, data, span);
  });
  refl::TypeAttrDef<DeclBufferNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&DeclBufferVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&DeclBufferMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&DeclBufferMaybeInplaceMutate));
}

// AllocBuffer
AllocBuffer::AllocBuffer(BufferVar buffer, ffi::Map<ffi::String, Any> annotations, Span span) {
  ffi::ObjectPtr<AllocBufferNode> node = ffi::make_object<AllocBufferNode>();
  node->buffer = std::move(buffer);
  node->annotations = std::move(annotations);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  AllocBufferNode::RegisterReflection();
  refl::GlobalDef().def(
      "tirx.AllocBuffer",
      [](BufferVar buffer, ffi::Optional<ffi::Map<ffi::String, Any>> annotations, Span span) {
        return AllocBuffer(buffer, annotations.value_or(ffi::Map<ffi::String, Any>()), span);
      });
  refl::TypeAttrDef<AllocBufferNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&AllocBufferVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&AllocBufferMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&AllocBufferMaybeInplaceMutate));
}

// SeqStmt
SeqStmt::SeqStmt(ffi::Array<Stmt> seq, Span span) {
  bool requires_flattening = std::any_of(
      seq.begin(), seq.end(), [](const Stmt& stmt) { return stmt->IsInstance<SeqStmtNode>(); });

  if (requires_flattening) {
    auto flattened = SeqStmt::Flatten(seq);
    if (auto* ptr = flattened.as<SeqStmtNode>()) {
      seq = ptr->seq;
    } else {
      seq = {flattened};
    }
  }

  TVM_FFI_ICHECK_NE(seq.size(), 0) << "An empty SeqStmt is prohibited.  "
                                   << "To write a no-op, use Evaluate(0), "
                                   << "or the result of SeqStmt::Flatten()";
  TVM_FFI_ICHECK_NE(seq.size(), 1) << "A SeqStmt of length 1 is prohibited.  "
                                   << "Use the node " << seq[0] << "directly, "
                                   << "or for dynamic usage, normalize using SeqStmt::Flatten()";

  auto node = ffi::make_object<SeqStmtNode>();
  node->seq = std::move(seq);
  node->span = std::move(span);
  data_ = std::move(node);
}

// A direct child that mutates into a SeqStmt is spliced into this sequence in the same pass,
// matching StmtMutator's normalized result without a second scan and allocation. A well-formed
// mapped SeqStmt already holds no SeqStmt child, so this single-level splice is sufficient and an
// unchanged element never needs splicing.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SeqStmtNode::RegisterReflection();
  refl::GlobalDef().def("tirx.SeqStmt", [](ffi::Array<Stmt> seq, Span span) {
    return SeqStmt(std::move(seq), span);
  });
  refl::TypeAttrDef<SeqStmtNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&SeqStmtVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&SeqStmtMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&SeqStmtMaybeInplaceMutate));
}

// IfThenElse
IfThenElse::IfThenElse(PrimExpr condition, Stmt then_case, ffi::Optional<Stmt> else_case,
                       Span span) {
  TVM_FFI_ICHECK(condition.defined());
  TVM_FFI_ICHECK(then_case.defined());
  // else_case may be null.
  ffi::ObjectPtr<IfThenElseNode> node = ffi::make_object<IfThenElseNode>();
  node->condition = std::move(condition);
  node->then_case = std::move(then_case);
  node->else_case = std::move(else_case);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  IfThenElseNode::RegisterReflection();
  refl::GlobalDef().def("tirx.IfThenElse",
                        [](PrimExpr condition, Stmt then_case, Stmt else_case, Span span) {
                          return IfThenElse(condition, then_case, else_case, span);
                        });
  refl::TypeAttrDef<IfThenElseNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&IfThenElseVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&IfThenElseMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&IfThenElseMaybeInplaceMutate));
}

// Evaluate
Evaluate::Evaluate(Expr value, Span span) {
  TVM_FFI_ICHECK(value.defined());
  TVM_FFI_ICHECK(!(value->IsInstance<VarNode>() && value->ty.as<BufferTypeNode>()))
      << "A buffer variable cannot be used as a scalar Evaluate value; "
      << "use buffer.data to evaluate its physical pointer";

  ffi::ObjectPtr<EvaluateNode> node = ffi::make_object<EvaluateNode>();
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  EvaluateNode::RegisterReflection();
  refl::GlobalDef().def("tirx.Evaluate",
                        [](Expr value, Span span) { return Evaluate(value, span); });
  refl::TypeAttrDef<EvaluateNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&EvaluateVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&EvaluateMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&EvaluateMaybeInplaceMutate));
}

// BufferStore
TVM_FFI_INLINE int GetLanesOrVScaleFactor(const PrimType& ty) {
  return ty.IsScalableVector() ? ty.VScaleFactor() : ty.lanes();
}

BufferStore::BufferStore(BufferVar buffer, PrimExpr value, ffi::Array<PrimExpr> indices,
                         Span span) {
  TVM_FFI_ICHECK_EQ(buffer->shape.size(), indices.size())
      << "BufferVar " << buffer.name() << " is " << buffer->shape.size()
      << "-dimensional, cannot be indexed with the " << indices.size()
      << "-dimensional indices provided.";

  for (int i = 0; i < static_cast<int>(indices.size()) - 1; i++) {
    TVM_FFI_ICHECK(indices[i].ty().IsScalar())
        << "Only the last index of a buffer access may be a vector type.";
  }

  bool is_index_scalable = indices.empty() ? false : indices.back().ty().IsScalableVector();
  int16_t buffer_encoded_lanes = static_cast<int16_t>(buffer->dtype->dtype.lanes);
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
    buffer_dtype = PrimType::ScalableVector(buffer->dtype.code(), buffer->dtype.bits(),
                                            buffer_lanes * index_lanes);
  } else {
    buffer_dtype = buffer->dtype.WithLanes(buffer_lanes * index_lanes);
  }
  if (buffer_dtype != value_ty) {
    TVM_FFI_THROW(TypeError) << "dtype mismatch on BufferStore: "                 //
                             << "buffer's dtype is `" << buffer->dtype            //
                             << "`, the lanes of indexing are: `" << index_lanes  //
                             << "`, the scalability is: `" << buffer_dtype.IsScalableVector()
                             << "`, but RHS's dtype is `" << value_ty << "`";
  }

  ffi::ObjectPtr<BufferStoreNode> node = ffi::make_object<BufferStoreNode>();
  node->buffer = std::move(buffer);
  node->value = std::move(value);
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BufferStoreNode::RegisterReflection();
  refl::GlobalDef().def("tirx.BufferStore",
                        [](BufferVar buffer, PrimExpr value, ffi::Array<PrimExpr> indices,
                           Span span) { return BufferStore(buffer, value, indices, span); });
  refl::TypeAttrDef<BufferStoreNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BufferStoreVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BufferStoreMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BufferStoreMaybeInplaceMutate));
}

// BufferRegion
BufferRegionType::BufferRegionType() : Type(ffi::UnsafeInit{}) {
  static ffi::ObjectPtr<BufferRegionTypeNode> singleton = ffi::make_object<BufferRegionTypeNode>();
  data_ = singleton;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BufferRegionTypeNode::RegisterReflection();
  refl::TypeAttrDef<BufferRegionTypeNode>().def("__subscript_expr_realize__",
                                                RealizeBufferRegionSubscript);
}

BufferRegion::BufferRegion(BufferVar buffer, ffi::Array<Range> region, Span span) {
  TVM_FFI_ICHECK_EQ(buffer->shape.size(), region.size())
      << "The dimension between " << buffer << " and region " << region
      << " mismatched, the buffer is " << buffer;
  ffi::ObjectPtr<BufferRegionNode> node = ffi::make_object<BufferRegionNode>();
  node->ty = BufferRegionType();
  node->span = std::move(span);
  node->buffer = std::move(buffer);
  node->region = std::move(region);
  data_ = std::move(node);
}

BufferRegion BufferRegion::FullRegion(BufferVar buffer) {
  ffi::Array<Range> region;
  for (PrimExpr extent : buffer->shape) {
    region.push_back(Range::FromMinExtent(0, extent));
  }
  return BufferRegion(buffer, region);
}

BufferRegion BufferRegion::FromPoint(BufferVar buffer, ffi::Array<PrimExpr> indices) {
  ffi::Array<Range> region;
  for (const PrimExpr& index : indices) {
    if (const prim::RampNode* ramp_index = index.as<prim::RampNode>()) {
      region.push_back(
          Range::FromMinExtent(ramp_index->base, ramp_index->stride * ramp_index->lanes));
    } else {
      region.push_back(Range::FromMinExtent(index, MakeConst(index.ty(), 1)));
    }
  }
  return BufferRegion(buffer, region);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BufferRegionNode::RegisterReflection();
  refl::GlobalDef()
      .def("tirx.BufferRegionType", []() { return BufferRegionType(); })
      .def("tirx.BufferRegion",
           [](BufferVar buffer, ffi::Array<Range> region) { return BufferRegion(buffer, region); });
  refl::TypeAttrDef<BufferRegionNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&BufferRegionVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&BufferRegionMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&BufferRegionMaybeInplaceMutate));
}

// MatchBufferRegion
MatchBufferRegion::MatchBufferRegion(BufferVar buffer, BufferRegion source) {
  const BufferVar& source_buffer = source->buffer;
  arith::Analyzer analyzer;
  // Check scope and dtype
  TVM_FFI_ICHECK_EQ(buffer.scope(), source_buffer.scope())
      << "MatchBuffer " << buffer << " scope mismatch:" << buffer.scope() << " vs. "
      << source_buffer.scope();
  TVM_FFI_ICHECK_EQ(buffer->dtype, source_buffer->dtype)
      << "MatchBuffer " << buffer << " data type mismatch:" << buffer->dtype << " vs. "
      << source_buffer->dtype;

  // Check data_alignment
  TVM_FFI_ICHECK(source_buffer->data_alignment % buffer->data_alignment == 0)
      << "Trying to match buffer to another one with lower alignment requirement "
      << " required alignment=" << buffer->data_alignment
      << ", provided alignment=" << source_buffer->data_alignment;

  // Validate shape
  TVM_FFI_ICHECK(source->region.size() >= buffer->shape.size())
      << "Dimension of source ffi::Array<Range> expected to be larger or equal than target buffer "
         "shape, but "
         "got "
      << source->region.size() << " vs. " << buffer->shape.size();
  size_t offset = source->region.size() - buffer->shape.size();
  for (size_t i = 0; i < offset; ++i) {
    TVM_FFI_ICHECK(analyzer->CanProve(source->region[i]->extent == 1))
        << "The higher dimension should be 1, but got " << source->region[i]->extent << ".";
  }
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    const Range& source_range = source->region[i + offset];
    const PrimExpr& buffer_shape = buffer->shape[i];
    if (!buffer_shape.as<PrimVar>()) {
      TVM_FFI_ICHECK(analyzer->CanProve(source_range->extent == buffer_shape))
          << "The dimension mismatched between source region and target buffer shape, got "
          << source_range->extent << " vs. " << buffer_shape << ".";
    }
  }
  // Note that we do not check elem_offset and strides in this function

  // Construction
  ffi::ObjectPtr<MatchBufferRegionNode> node = ffi::make_object<MatchBufferRegionNode>();
  node->buffer = std::move(buffer);
  node->source = std::move(source);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MatchBufferRegionNode::RegisterReflection();
  refl::GlobalDef().def("tirx.MatchBufferRegion", [](BufferVar buffer, BufferRegion source) {
    return MatchBufferRegion(buffer, source);
  });
  refl::TypeAttrDef<MatchBufferRegionNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&MatchBufferRegionVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&MatchBufferRegionMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&MatchBufferRegionMaybeInplaceMutate));
}

// Block
SBlock::SBlock(ffi::Array<IterVar> iter_vars, ffi::Array<BufferRegion> reads,
               ffi::Array<BufferRegion> writes, ffi::String name_hint, Stmt body,
               ffi::Optional<Stmt> init, ffi::Array<BufferVar> alloc_buffers,
               ffi::Array<MatchBufferRegion> match_buffers, ffi::Map<ffi::String, Any> annotations,
               Span span) {
  ffi::ObjectPtr<SBlockNode> node = ffi::make_object<SBlockNode>();
  node->iter_vars = std::move(iter_vars);
  node->reads = std::move(reads);
  node->writes = std::move(writes);
  node->name_hint = std::move(name_hint);
  node->body = std::move(body);
  node->init = std::move(init);
  node->alloc_buffers = std::move(alloc_buffers);
  node->match_buffers = std::move(match_buffers);
  node->annotations = std::move(annotations);
  node->span = std::move(span);
  data_ = std::move(node);
}

SBlock::SBlock(ffi::String name_hint, Stmt body, ffi::Array<BufferVar> alloc_buffers, Span span) {
  ffi::ObjectPtr<SBlockNode> node = ffi::make_object<SBlockNode>();
  node->iter_vars = {};
  node->reads = {};
  node->writes = {};
  node->name_hint = std::move(name_hint);
  node->body = std::move(body);
  node->init = std::nullopt;
  node->alloc_buffers = std::move(alloc_buffers);
  node->match_buffers = {};
  node->annotations = {};
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SBlockNode::RegisterReflection();
  refl::GlobalDef().def("tirx.SBlock",
                        [](ffi::Array<IterVar> iter_vars, ffi::Array<BufferRegion> reads,
                           ffi::Array<BufferRegion> writes, ffi::String name_hint, Stmt body,
                           ffi::Optional<Stmt> init, ffi::Array<BufferVar> alloc_buffers,
                           ffi::Array<MatchBufferRegion> match_buffers,
                           ffi::Map<ffi::String, Any> annotations, Span span) {
                          return SBlock(iter_vars, reads, writes, name_hint, body, init,
                                        alloc_buffers, match_buffers, annotations, span);
                        });
  refl::TypeAttrDef<SBlockNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&SBlockVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&SBlockMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&SBlockMaybeInplaceMutate));
}

// ScopeIdDefStmt
ScopeIdDefStmt::ScopeIdDefStmt(ScopeIdDef def, Span span) {
  TVM_FFI_ICHECK(def.defined());
  ffi::ObjectPtr<ScopeIdDefStmtNode> node = ffi::make_object<ScopeIdDefStmtNode>();
  node->def = std::move(def);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ScopeIdDefStmtNode::RegisterReflection();
  refl::GlobalDef().def("tirx.ScopeIdDefStmt",
                        [](ScopeIdDef def, Span span) { return ScopeIdDefStmt(def, span); });
  refl::TypeAttrDef<ScopeIdDefStmtNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&ScopeIdDefStmtVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&ScopeIdDefStmtMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&ScopeIdDefStmtMaybeInplaceMutate));
}

// BlockRealize
SBlockRealize::SBlockRealize(ffi::Array<PrimExpr> values, PrimExpr predicate, SBlock block,
                             Span span) {
  TVM_FFI_CHECK_EQ(block->iter_vars.size(), values.size(), ValueError)
      << "BlockRealize needs to have the same number of iter_vars and binding values";
  PrimType predicate_ty = predicate.ty();
  TVM_FFI_CHECK(predicate_ty.MatchesCode(DLDataTypeCode::kDLBool), TypeError)
      << "Expect Block.predicate to be a bool expression";
  ffi::ObjectPtr<SBlockRealizeNode> node = ffi::make_object<SBlockRealizeNode>();
  node->iter_values = std::move(values);
  node->predicate = std::move(predicate);
  node->block = std::move(block);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SBlockRealizeNode::RegisterReflection();
  refl::GlobalDef().def("tirx.SBlockRealize", [](ffi::Array<PrimExpr> iter_values,
                                                 PrimExpr predicate, SBlock block, Span span) {
    return SBlockRealize(iter_values, predicate, block, span);
  });
  refl::TypeAttrDef<SBlockRealizeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&SBlockRealizeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&SBlockRealizeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&SBlockRealizeMaybeInplaceMutate));
}

PrimExpr TypeAnnotation(PrimType dtype, Span span) {
  static const Op& type_annotation_op = Op::Get("tirx.type_annotation");
  return Call(dtype, type_annotation_op, {}, {}, {}, span).as_or_throw<PrimExpr>();
}

TVM_TIRX_REGISTER_OP("type_annotation")
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TScriptDtypePrintLocation>("TScriptDtypePrintLocation",
                                         static_cast<int64_t>(ScriptDtypePrintLocation::kFirst));

}  // namespace tirx
}  // namespace tvm
