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

#include "../../tirx/transform/stmt_extension.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt_functor.h>

namespace tvm {
namespace s_tir {
using namespace tirx;
namespace {

UnchangedOr<Stmt> ConvertSSABlock(tirx::SSAStmtMutator* self, const SBlockNode* op,
                                  InplaceMode mode) {
  SBlock block = ffi::GetRef<SBlock>(op);
  return self->WithScope([&]() -> Stmt {
    auto iter_vars = op->iter_vars.Map([&](IterVar iter) {
      Var var = self->DefineVar(iter->var);
      if (!var.same_as(iter->var)) iter.CopyOnWrite()->var = var.as_or_throw<PrimVar>();
      return iter;
    });
    auto remap_region = [&](BufferRegion region) {
      BufferVar buffer = self->RemapBuffer(region->buffer);
      if (!buffer.same_as(region->buffer)) region.CopyOnWrite()->buffer = buffer;
      return region;
    };
    auto reads = block->reads.Map(remap_region);
    auto writes = block->writes.Map(remap_region);
    if (!reads.same_as(block->reads) || !writes.same_as(block->writes) ||
        !iter_vars.same_as(op->iter_vars)) {
      auto* writer = block.CopyOnWrite();
      writer->reads = reads;
      writer->writes = writes;
      writer->iter_vars = iter_vars;
    }
    return StmtExprMutator::MutateBlock(self, block.get(),
                                        block.unique() ? mode : InplaceMode::kDisallow)
        .ValueOrUnchanged(block);
  });
}

UnchangedOr<Stmt> FlattenBlock(tirx::FlattenStmtMutator* self, const SBlockNode* op,
                               InplaceMode mode) {
  TVM_FFI_ICHECK_EQ(op->match_buffers.size(), 0)
      << "Unexpected MatchBufferRegion found during tirx.transform.FlattenBuffer.  "
      << "All MatchBufferRegion should be removed in tirx.transform.LowerMatchBuffer.";
  SBlock block = ffi::GetRef<SBlock>(op);
  auto alloc_buffers = op->alloc_buffers;
  alloc_buffers.MutateByApply([&](BufferVar buffer) { return self->DefineBuffer(buffer); });
  if (!alloc_buffers.same_as(op->alloc_buffers)) block.CopyOnWrite()->alloc_buffers = alloc_buffers;
  auto reads = op->reads;
  reads.MutateByApply([&](BufferRegion region) { return self->RewriteRegion(region); });
  if (!reads.same_as(op->reads)) block.CopyOnWrite()->reads = reads;
  auto writes = op->writes;
  writes.MutateByApply([&](BufferRegion region) { return self->RewriteRegion(region); });
  if (!writes.same_as(op->writes)) block.CopyOnWrite()->writes = writes;
  return StmtExprMutator::MutateBlock(self, block.get(),
                                      block.unique() ? mode : InplaceMode::kDisallow)
      .ValueOrUnchanged(block);
}

// These hooks extend each pass's existing native table before it is finalized.
// All ordinary statements and expression remapping remain in the TIRX pass.
TVM_FFI_STATIC_INIT_BLOCK() {
  tirx::SSAStmtMutator::RegisterExtension([](tirx::SSAStmtMutator::VTable* table) {
    table->ClearDispatch<SBlockNode>();
    table->SetDispatch<SBlockNode>(
        [](const ffi::Object* node, ObjectMutator* self, InplaceMode mode) {
          return ffi::details::UnchangedOrUnsafe::MoveFromTVMFFIAny<ffi::Any>(
              ffi::details::UnchangedOrUnsafe::MoveToTVMFFIAny(
                  ConvertSSABlock(static_cast<tirx::SSAStmtMutator*>(self),
                                  static_cast<const SBlockNode*>(node), mode)));
        });
  });
  tirx::FlattenStmtMutator::RegisterExtension([](tirx::FlattenStmtMutator::VTable* table) {
    table->ClearDispatch<SBlockNode>();
    table->SetDispatch<SBlockNode>(
        [](const ffi::Object* node, ObjectMutator* self, InplaceMode mode) {
          return ffi::details::UnchangedOrUnsafe::MoveFromTVMFFIAny<ffi::Any>(
              ffi::details::UnchangedOrUnsafe::MoveToTVMFFIAny(
                  FlattenBlock(static_cast<tirx::FlattenStmtMutator*>(self),
                               static_cast<const SBlockNode*>(node), mode)));
        });
  });
  tirx::IndexDomainVisitor::RegisterExtension([](tirx::IndexDomainVisitor::VTable* table) {
    table->ClearDispatch<SBlockNode>();
    table->SetDispatch<SBlockNode>([](const ffi::Object* node, ObjectVisitor* base) {
      auto* self = static_cast<tirx::IndexDomainVisitor*>(base);
      auto* block = static_cast<const SBlockNode*>(node);
      for (const auto& iter : block->iter_vars) {
        self->BindDomain(iter->var, Range::FromMinExtent(iter->dom->min, iter->dom->extent));
      }
      return StmtExprVisitor::VisitBlock(self, block);
    });
  });
}

}  // namespace
}  // namespace s_tir
}  // namespace tvm
