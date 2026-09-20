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
 * \file tvm/s_tir/stmt.cc
 * \brief Schedulable block definitions and structural traversal.
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/sym/analyzer.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;
using namespace tvm::prim;

namespace {

TVMFFIAny MatchBufferRegionVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const MatchBufferRegionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MatchBufferRegionNode>(
          value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->buffer); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->source));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny MatchBufferRegionMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const MatchBufferRegionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MatchBufferRegionNode>(
          value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<BufferVar>, mapped_buffer,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->buffer);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<TensorRegion>, mapped_source,
                                    mutator->MutateExpected(self->source));
  if (mapped_buffer.UnchangedOrSameAs(self->buffer) &&
      mapped_source.UnchangedOrSameAs(self->source)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<MatchBufferRegionNode> copy = ffi::make_object<MatchBufferRegionNode>(*self);
  copy->buffer = std::move(mapped_buffer).ValueOrUnchanged(std::move(copy->buffer));
  copy->source = std::move(mapped_source).ValueOrUnchanged(std::move(copy->source));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny MatchBufferRegionMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                              ffi::AnyView value) noexcept {
  MatchBufferRegionNode* self = const_cast<MatchBufferRegionNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MatchBufferRegionNode>(
          value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<BufferVar>, mapped_buffer,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->buffer,
                                                                     ffi::InplaceMode::kAllow);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<TensorRegion>, mapped_source,
      mutator->MutateExpected(self->source, ffi::InplaceMode::kAllow));
  if (mapped_buffer.UnchangedOrSameAs(self->buffer) &&
      mapped_source.UnchangedOrSameAs(self->source)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_buffer.IsUnchanged()) self->buffer = std::move(mapped_buffer).ValueUnchecked();
  if (!mapped_source.IsUnchanged()) self->source = std::move(mapped_source).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny SBlockVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // Establish allocation and match-buffer definitions before their region uses.
  // Whole iterators and annotations remain part of structural traversal.
  // skips: name_hint
  const SBlockNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->iter_vars));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->alloc_buffers); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->match_buffers));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->reads));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->writes));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->annotations));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->init));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny SBlockMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // Establish allocation and match-buffer definitions before their region uses.
  // Whole iterators and annotations remain part of structural traversal.
  // skips: name_hint
  const SBlockNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<IterVar>>, mapped_iter_vars,
                                    mutator->MutateExpected(self->iter_vars));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<BufferVar>>, mapped_alloc_buffers,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->alloc_buffers);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<MatchBufferRegion>>,
                                    mapped_match_buffers,
                                    mutator->MutateExpected(self->match_buffers));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<TensorRegion>>, mapped_reads,
                                    mutator->MutateExpected(self->reads));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<TensorRegion>>, mapped_writes,
                                    mutator->MutateExpected(self->writes));
  using AnnotationMap = ffi::Map<ffi::String, ffi::Any>;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<AnnotationMap>, mapped_annotations,
                                    mutator->MutateExpected(self->annotations));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Stmt>>, mapped_init,
                                    mutator->MutateExpected(self->init));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body));
  if (mapped_iter_vars.UnchangedOrSameAs(self->iter_vars) &&
      mapped_reads.UnchangedOrSameAs(self->reads) &&
      mapped_writes.UnchangedOrSameAs(self->writes) &&
      mapped_alloc_buffers.UnchangedOrSameAs(self->alloc_buffers) &&
      mapped_match_buffers.UnchangedOrSameAs(self->match_buffers) &&
      mapped_annotations.UnchangedOrSameAs(self->annotations) &&
      mapped_init.UnchangedOrSameAs(self->init) && mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<SBlockNode> copy = ffi::make_object<SBlockNode>(*self);
  copy->iter_vars = std::move(mapped_iter_vars).ValueOrUnchanged(std::move(copy->iter_vars));
  copy->reads = std::move(mapped_reads).ValueOrUnchanged(std::move(copy->reads));
  copy->writes = std::move(mapped_writes).ValueOrUnchanged(std::move(copy->writes));
  copy->alloc_buffers =
      std::move(mapped_alloc_buffers).ValueOrUnchanged(std::move(copy->alloc_buffers));
  copy->match_buffers =
      std::move(mapped_match_buffers).ValueOrUnchanged(std::move(copy->match_buffers));
  copy->annotations = std::move(mapped_annotations).ValueOrUnchanged(std::move(copy->annotations));
  copy->init = std::move(mapped_init).ValueOrUnchanged(std::move(copy->init));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny SBlockMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                   ffi::AnyView value) noexcept {
  // Establish allocation and match-buffer definitions before their region uses.
  // Whole iterators and annotations remain part of structural traversal.
  // skips: name_hint
  SBlockNode* self = const_cast<SBlockNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<IterVar>>, mapped_iter_vars,
      mutator->MutateExpected(self->iter_vars, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<BufferVar>>, mapped_alloc_buffers,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->alloc_buffers,
                                                                     ffi::InplaceMode::kAllow);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<MatchBufferRegion>>, mapped_match_buffers,
      mutator->MutateExpected(self->match_buffers, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<TensorRegion>>, mapped_reads,
                                    mutator->MutateExpected(self->reads, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<TensorRegion>>, mapped_writes,
      mutator->MutateExpected(self->writes, ffi::InplaceMode::kAllow));
  using AnnotationMap = ffi::Map<ffi::String, ffi::Any>;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<AnnotationMap>, mapped_annotations,
      mutator->MutateExpected(self->annotations, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Stmt>>, mapped_init,
                                    mutator->MutateExpected(self->init, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body, ffi::InplaceMode::kAllow));
  if (mapped_iter_vars.UnchangedOrSameAs(self->iter_vars) &&
      mapped_reads.UnchangedOrSameAs(self->reads) &&
      mapped_writes.UnchangedOrSameAs(self->writes) &&
      mapped_alloc_buffers.UnchangedOrSameAs(self->alloc_buffers) &&
      mapped_match_buffers.UnchangedOrSameAs(self->match_buffers) &&
      mapped_annotations.UnchangedOrSameAs(self->annotations) &&
      mapped_init.UnchangedOrSameAs(self->init) && mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_iter_vars.IsUnchanged())
    self->iter_vars = std::move(mapped_iter_vars).ValueUnchecked();
  if (!mapped_reads.IsUnchanged()) self->reads = std::move(mapped_reads).ValueUnchecked();
  if (!mapped_writes.IsUnchanged()) self->writes = std::move(mapped_writes).ValueUnchecked();
  if (!mapped_alloc_buffers.IsUnchanged()) {
    self->alloc_buffers = std::move(mapped_alloc_buffers).ValueUnchecked();
  }
  if (!mapped_match_buffers.IsUnchanged()) {
    self->match_buffers = std::move(mapped_match_buffers).ValueUnchecked();
  }
  if (!mapped_annotations.IsUnchanged())
    self->annotations = std::move(mapped_annotations).ValueUnchecked();
  if (!mapped_init.IsUnchanged()) self->init = std::move(mapped_init).ValueUnchecked();
  if (!mapped_body.IsUnchanged()) self->body = std::move(mapped_body).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny SBlockRealizeVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const SBlockRealizeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockRealizeNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->iter_values));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->predicate));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->block));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny SBlockRealizeMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const SBlockRealizeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockRealizeNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_iter_values,
                                    mutator->MutateExpected(self->iter_values));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_predicate,
                                    mutator->MutateExpected(self->predicate));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SBlock>, mapped_block,
                                    mutator->MutateExpected(self->block));
  if (mapped_iter_values.UnchangedOrSameAs(self->iter_values) &&
      mapped_predicate.UnchangedOrSameAs(self->predicate) &&
      mapped_block.UnchangedOrSameAs(self->block)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<SBlockRealizeNode> copy = ffi::make_object<SBlockRealizeNode>(*self);
  copy->iter_values = std::move(mapped_iter_values).ValueOrUnchanged(std::move(copy->iter_values));
  copy->predicate = std::move(mapped_predicate).ValueOrUnchanged(std::move(copy->predicate));
  copy->block = std::move(mapped_block).ValueOrUnchanged(std::move(copy->block));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny SBlockRealizeMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                          ffi::AnyView value) noexcept {
  SBlockRealizeNode* self = const_cast<SBlockRealizeNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockRealizeNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_iter_values,
      mutator->MutateExpected(self->iter_values, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_predicate,
      mutator->MutateExpected(self->predicate, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SBlock>, mapped_block,
                                    mutator->MutateExpected(self->block, ffi::InplaceMode::kAllow));
  if (mapped_iter_values.UnchangedOrSameAs(self->iter_values) &&
      mapped_predicate.UnchangedOrSameAs(self->predicate) &&
      mapped_block.UnchangedOrSameAs(self->block)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  if (!mapped_iter_values.IsUnchanged())
    self->iter_values = std::move(mapped_iter_values).ValueUnchecked();
  if (!mapped_predicate.IsUnchanged())
    self->predicate = std::move(mapped_predicate).ValueUnchecked();
  if (!mapped_block.IsUnchanged()) self->block = std::move(mapped_block).ValueUnchecked();
  return ffi::Unchanged().CopyToTVMFFIAny();
}

}  // namespace

// MatchBufferRegion
MatchBufferRegion::MatchBufferRegion(BufferVar buffer, TensorRegion source) {
  const BufferVar& source_buffer = source->source.as_or_throw<BufferVar>();
  TVM_FFI_ICHECK_EQ(source_buffer->shape.size(), source->region.size())
      << "MatchBufferRegion source must match its buffer rank";
  sym::Analyzer analyzer;
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
  ffi::ObjectPtr<MatchBufferRegionNode> node = ffi::make_object<MatchBufferRegionNode>();
  node->buffer = std::move(buffer);
  node->source = std::move(source);
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MatchBufferRegionNode::RegisterReflection();
  refl::TypeAttrDef<MatchBufferRegionNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&MatchBufferRegionVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&MatchBufferRegionMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&MatchBufferRegionMaybeInplaceMutate));

  refl::GlobalDef().def("s_tir.MatchBufferRegion", [](BufferVar buffer, TensorRegion source) {
    return MatchBufferRegion(buffer, source);
  });
}

// Block
SBlock::SBlock(ffi::Array<IterVar> iter_vars, ffi::Array<TensorRegion> reads,
               ffi::Array<TensorRegion> writes, ffi::String name_hint, Stmt body,
               ffi::Optional<Stmt> init, ffi::Array<BufferVar> alloc_buffers,
               ffi::Array<MatchBufferRegion> match_buffers, ffi::Map<ffi::String, Any> annotations,
               Span span) {
  for (const auto& regions : {reads, writes}) {
    for (const TensorRegion& region : regions) {
      const auto buffer = region->source.as_or_throw<BufferVar>();
      TVM_FFI_ICHECK_EQ(buffer->shape.size(), region->region.size())
          << "SBlock region must match its buffer rank";
    }
  }
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
  refl::TypeAttrDef<SBlockNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&SBlockVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&SBlockMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&SBlockMaybeInplaceMutate));

  refl::GlobalDef().def("s_tir.SBlock",
                        [](ffi::Array<IterVar> iter_vars, ffi::Array<TensorRegion> reads,
                           ffi::Array<TensorRegion> writes, ffi::String name_hint, Stmt body,
                           ffi::Optional<Stmt> init, ffi::Array<BufferVar> alloc_buffers,
                           ffi::Array<MatchBufferRegion> match_buffers,
                           ffi::Map<ffi::String, Any> annotations, Span span) {
                          return SBlock(iter_vars, reads, writes, name_hint, body, init,
                                        alloc_buffers, match_buffers, annotations, span);
                        });
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
  refl::TypeAttrDef<SBlockRealizeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&SBlockRealizeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&SBlockRealizeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&SBlockRealizeMaybeInplaceMutate));

  refl::GlobalDef().def("s_tir.SBlockRealize", [](ffi::Array<PrimExpr> iter_values,
                                                  PrimExpr predicate, SBlock block, Span span) {
    return SBlockRealize(iter_values, predicate, block, span);
  });
}

}  // namespace s_tir
}  // namespace tvm
