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

#include "../../tirx/ir/data_type_rewriter.h"

#include <tvm/s_tir/stmt_functor.h>
#include <tvm/tirx/op.h>

#include <functional>

namespace tvm {
namespace tirx {
using namespace tvm::prim;
using s_tir::MatchBufferRegion;
using s_tir::SBlock;
using s_tir::SBlockNode;
using s_tir::SBlockRealize;
using s_tir::SBlockRealizeNode;

class DataTypeLegalizer::Extension {
 public:
  static void InitVTable(VTable* vtable);
  static UnchangedOr<Stmt> MutateBlockRealize(DataTypeLegalizer* self,
                                              const s_tir::SBlockRealizeNode* op,
                                              InplaceMode inplace_mode);
  static UnchangedOr<Stmt> MutateBlock(DataTypeLegalizer* self, const s_tir::SBlockNode* op,
                                       InplaceMode inplace_mode);
};
class IndexDataTypeRewriter::Extension {
 public:
  static void InitVTable(VTable* vtable);
  static UnchangedOr<Stmt> MutateBlockRealize(IndexDataTypeRewriter* self,
                                              const s_tir::SBlockRealizeNode* op,
                                              InplaceMode inplace_mode);
  static UnchangedOr<Stmt> MutateBlock(IndexDataTypeRewriter* self, const s_tir::SBlockNode* op,
                                       InplaceMode inplace_mode);
  static ffi::Map<ffi::String, ffi::Any> VisitBlockAnnotations(
      IndexDataTypeRewriter* self, const ffi::Map<ffi::String, ffi::Any>& annotations);
  static IterVar VisitIterVar(IndexDataTypeRewriter* self, const IterVar& iter_var);
  static BufferRegion VisitBufferRegion(IndexDataTypeRewriter* self,
                                        const BufferRegion& buffer_region);
};

UnchangedOr<Stmt> DataTypeLegalizer::Extension::MutateBlockRealize(DataTypeLegalizer* self,
                                                                   const SBlockRealizeNode* op,
                                                                   InplaceMode inplace_mode) {
  SBlockRealize realize = s_tir::StmtExprMutator::MutateBlockRealize(self, op, inplace_mode)
                              .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                              .as_or_throw<SBlockRealize>();
  ffi::Array<PrimExpr> new_iter_values;
  bool changed = false;
  for (int i = 0; i < static_cast<int>(op->iter_values.size()); ++i) {
    PrimType dtype = realize->block->iter_vars[i]->var.ty();
    if (op->iter_values[i].ty() != dtype) {
      new_iter_values.push_back(prim::cast(dtype, realize->iter_values[i]));
      changed = true;
    } else {
      new_iter_values.push_back(realize->iter_values[i]);
    }
  }
  if (changed) {
    realize.CopyOnWrite()->iter_values = std::move(new_iter_values);
  }
  return realize;
}

UnchangedOr<Stmt> DataTypeLegalizer::Extension::MutateBlock(DataTypeLegalizer* self,
                                                            const SBlockNode* op,
                                                            InplaceMode inplace_mode) {
  SBlock new_block = s_tir::StmtExprMutator::MutateBlock(self, op, inplace_mode)
                         .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                         .as_or_throw<SBlock>();
  ffi::Array<IterVar> new_iter_vars = new_block->iter_vars.Map([](const IterVar& iter) {
    PrimType dtype = iter->var.ty();
    if (iter->dom->min.ty() != dtype || iter->dom->extent.ty() != dtype) {
      IterVar new_iter = iter;
      new_iter.CopyOnWrite()->dom =
          Range(prim::cast(dtype, iter->dom->min), prim::cast(dtype, iter->dom->extent));
      return new_iter;
    } else {
      return iter;
    }
  });
  if (!op->iter_vars.same_as(new_iter_vars)) {
    new_block.CopyOnWrite()->iter_vars = std::move(new_iter_vars);
  }
  return new_block;
}

void DataTypeLegalizer::Extension::InitVTable(VTable* vtable) {
  vtable->ClearDispatch<SBlockNode>();
  vtable->ClearDispatch<SBlockRealizeNode>();
  vtable->SetDispatch<SBlockNode>(
      [](const ffi::Object* node, ObjectMutator* base, InplaceMode mode) -> UnchangedOr<ffi::Any> {
        return MutateBlock(static_cast<DataTypeLegalizer*>(base),
                           static_cast<const SBlockNode*>(node), mode);
      });
  vtable->SetDispatch<SBlockRealizeNode>(
      [](const ffi::Object* node, ObjectMutator* base, InplaceMode mode) -> UnchangedOr<ffi::Any> {
        return MutateBlockRealize(static_cast<DataTypeLegalizer*>(base),
                                  static_cast<const SBlockRealizeNode*>(node), mode);
      });
}
TVM_FFI_STATIC_INIT_BLOCK() {
  DataTypeLegalizer::RegisterExtension(DataTypeLegalizer::Extension::InitVTable);
}

UnchangedOr<Stmt> IndexDataTypeRewriter::Extension::MutateBlockRealize(IndexDataTypeRewriter* self,
                                                                       const SBlockRealizeNode* op,
                                                                       InplaceMode inplace_mode) {
  bool is_condition = self->is_condition_;
  self->is_condition_ = true;
  auto new_predicate_result = self->Mutate(op->predicate, inplace_mode);
  bool new_predicate_unchanged = new_predicate_result.UnchangedOrSameAs(op->predicate);
  auto new_predicate = std::move(new_predicate_result).ValueOrUnchanged(op->predicate);
  self->is_condition_ = is_condition;

  bool is_enabled = self->is_enabled_;
  self->is_enabled_ = true;
  auto new_iter_values = self->Mutate(op->iter_values, inplace_mode)
                             .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>()
                             .ValueOrUnchanged(op->iter_values);
  self->is_enabled_ = is_enabled;
  SBlock new_body =
      self->Mutate(op->block, inplace_mode).ValueOrUnchanged(op->block).as_or_throw<SBlock>();
  if (!new_predicate_unchanged || !new_iter_values.same_as(op->iter_values) ||
      !new_body.same_as(op->block)) {
    SBlockRealize new_block_realize = ffi::GetRef<SBlockRealize>(op);
    auto* n = new_block_realize.CopyOnWrite();
    n->predicate = std::move(new_predicate);
    n->iter_values = std::move(new_iter_values);
    n->block = std::move(new_body);
    return new_block_realize;

  } else {
    return ffi::Unchanged();
  }
}

UnchangedOr<Stmt> IndexDataTypeRewriter::Extension::MutateBlock(IndexDataTypeRewriter* self,
                                                                const SBlockNode* op,
                                                                InplaceMode inplace_mode) {
  auto new_alloc_buffers = self->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
    return self->Mutate(op->alloc_buffers, inplace_mode)
        .as_or_throw<UnchangedOr<ffi::Array<BufferVar>>>()
        .ValueOrUnchanged(op->alloc_buffers);
  });
  auto new_match_buffers = op->match_buffers.Map([self](const MatchBufferRegion& match) {
    BufferVar buffer = self->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
      return self->Mutate(match->buffer, InplaceMode::kDisallow)
          .as_or_throw<UnchangedOr<BufferVar>>()
          .ValueOrUnchanged(match->buffer);
    });
    BufferRegion source = VisitBufferRegion(self, match->source);
    if (buffer.same_as(match->buffer) && source.same_as(match->source)) return match;
    return MatchBufferRegion(buffer, source);
  });
  ffi::Array<BufferRegion> new_reads = op->reads.Map(
      [self](const BufferRegion& buffer_region) { return VisitBufferRegion(self, buffer_region); });
  ffi::Array<BufferRegion> new_writes = op->writes.Map(
      [self](const BufferRegion& buffer_region) { return VisitBufferRegion(self, buffer_region); });
  ffi::Array<IterVar> new_iter_vars =
      op->iter_vars.Map([self](const IterVar& iter_var) { return VisitIterVar(self, iter_var); });
  ffi::Optional<Stmt> new_init = std::nullopt;
  if (op->init.has_value()) {
    new_init = self->Mutate(op->init.value(), inplace_mode).ValueOrUnchanged(op->init.value());
  }
  ffi::Map<ffi::String, ffi::Any> new_annotations = VisitBlockAnnotations(self, op->annotations);
  auto new_body_result = self->Mutate(op->body, inplace_mode);
  bool new_body_unchanged = new_body_result.UnchangedOrSameAs(op->body);
  Stmt new_body = std::move(new_body_result).ValueOrUnchanged(op->body);

  if (!new_init.same_as(op->init) || !new_body_unchanged ||
      !new_alloc_buffers.same_as(op->alloc_buffers) ||
      !new_match_buffers.same_as(op->match_buffers) || !new_reads.same_as(op->reads) ||
      !new_writes.same_as(op->writes) || !new_iter_vars.same_as(op->iter_vars) ||
      !new_annotations.same_as(op->annotations)) {
    SBlock new_block = ffi::GetRef<SBlock>(op);
    SBlockNode* n = new_block.CopyOnWrite();
    n->alloc_buffers = std::move(new_alloc_buffers);
    n->match_buffers = std::move(new_match_buffers);
    n->reads = std::move(new_reads);
    n->writes = std::move(new_writes);
    n->iter_vars = std::move(new_iter_vars);
    n->init = std::move(new_init);
    n->annotations = std::move(new_annotations);
    n->body = std::move(new_body);
    for (const auto& buffer : new_block->alloc_buffers) self->ValidateAllocation(buffer);
    return new_block;
  }
  for (const auto& buffer : op->alloc_buffers) self->ValidateAllocation(buffer);
  return ffi::Unchanged();
}

ffi::Map<ffi::String, ffi::Any> IndexDataTypeRewriter::Extension::VisitBlockAnnotations(
    IndexDataTypeRewriter* self, const ffi::Map<ffi::String, ffi::Any>& annotations) {
  auto new_annotations = annotations;

  std::function<Any(const Any&)> f_mutate_obj = [self, &f_mutate_obj](const Any& obj) -> Any {
    if (obj == nullptr) {
      return obj;
    }
    if (auto var = obj.as<Var>(); var && var.value()->ty.as<BufferTypeNode>()) {
      BufferVar buffer(var.value());
      if (BufferVar new_buffer = self->Mutate(buffer, InplaceMode::kDisallow)
                                     .as_or_throw<UnchangedOr<BufferVar>>()
                                     .ValueOrUnchanged(buffer);
          !new_buffer.same_as(buffer)) {
        return new_buffer;
      }
    } else if (obj.as<ffi::ArrayObj>()) {
      return obj.as_or_throw<ffi::Array<Any>>().Map(f_mutate_obj);
    }
    return obj;
  };
  for (const auto& [key, value] : annotations) {
    if (auto opt_object_ref = value.as<ffi::ObjectRef>()) {
      auto new_value = f_mutate_obj(*opt_object_ref);
      if (!new_value.same_as(*opt_object_ref)) {
        new_annotations.Set(key, new_value);
      }
    }
  }
  return new_annotations;
}

IterVar IndexDataTypeRewriter::Extension::VisitIterVar(IndexDataTypeRewriter* self,
                                                       const IterVar& iter_var) {
  bool is_enabled = self->is_enabled_;
  self->is_enabled_ = true;
  PrimVar new_var = self->Mutate(iter_var->var, InplaceMode::kDisallow)
                        .ValueOrUnchanged(iter_var->var)
                        .as_or_throw<PrimVar>();
  PrimExpr min =
      self->Mutate(iter_var->dom->min, InplaceMode::kDisallow).ValueOrUnchanged(iter_var->dom->min);
  PrimExpr extent = self->Mutate(iter_var->dom->extent, InplaceMode::kDisallow)
                        .ValueOrUnchanged(iter_var->dom->extent);
  self->is_enabled_ = is_enabled;
  if (!new_var.same_as(iter_var->var) || !min.same_as(iter_var->dom->min) ||
      !extent.same_as(iter_var->dom->extent)) {
    IterVar new_iter_var = iter_var;
    IterVarNode* n = new_iter_var.CopyOnWrite();
    n->var = std::move(new_var);
    n->dom = Range(min, extent);
    return new_iter_var;
  }
  return iter_var;
}

BufferRegion IndexDataTypeRewriter::Extension::VisitBufferRegion(
    IndexDataTypeRewriter* self, const BufferRegion& buffer_region) {
  BufferVar remapped_buffer = self->Mutate(buffer_region->buffer, InplaceMode::kDisallow)
                                  .as_or_throw<UnchangedOr<BufferVar>>()
                                  .ValueOrUnchanged(buffer_region->buffer);

  bool is_enabled = self->is_enabled_;
  self->is_enabled_ = true;
  auto new_region = buffer_region->region.Map([&](const Range& range) {
    return Range::FromMinExtent(
        self->Mutate(range->min, InplaceMode::kDisallow).ValueOrUnchanged(range->min),
        self->Mutate(range->extent, InplaceMode::kDisallow).ValueOrUnchanged(range->extent));
  });
  self->is_enabled_ = is_enabled;

  if (!remapped_buffer.same_as(buffer_region->buffer) ||
      !new_region.same_as(buffer_region->region)) {
    return BufferRegion(remapped_buffer, new_region);
  } else {
    return buffer_region;
  }
}

void IndexDataTypeRewriter::Extension::InitVTable(VTable* vtable) {
  vtable->ClearDispatch<SBlockNode>();
  vtable->SetDispatch<SBlockNode>(
      [](const ffi::Object* node, ObjectMutator* base, InplaceMode mode) -> UnchangedOr<ffi::Any> {
        return MutateBlock(static_cast<IndexDataTypeRewriter*>(base),
                           static_cast<const SBlockNode*>(node), mode);
      });
  vtable->ClearDispatch<SBlockRealizeNode>();
  vtable->SetDispatch<SBlockRealizeNode>(
      [](const ffi::Object* node, ObjectMutator* base, InplaceMode mode) -> UnchangedOr<ffi::Any> {
        return MutateBlockRealize(static_cast<IndexDataTypeRewriter*>(base),
                                  static_cast<const SBlockRealizeNode*>(node), mode);
      });
}
TVM_FFI_STATIC_INIT_BLOCK() {
  IndexDataTypeRewriter::RegisterExtension(IndexDataTypeRewriter::Extension::InitVTable);
}
}  // namespace tirx
}  // namespace tvm
