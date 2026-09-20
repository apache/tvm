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
 * \file stmt_functor.cc
 * \brief Native traversal of schedulable TIR nodes.
 */
#include <tvm/s_tir/stmt_functor.h>

#include <utility>
#include <vector>

namespace tvm {
namespace s_tir {

using namespace tirx;

void StmtExprVisitor::InitVTable(VTable* vtable) {
  tirx::StmtExprVisitor::InitVTable(vtable);
  SetDispatch<StmtExprVisitor, SBlockNode>(vtable);
  SetDispatch<StmtExprVisitor, SBlockRealizeNode>(vtable);
}

void StmtExprMutator::InitVTable(VTable* vtable) {
  tirx::StmtExprMutator::InitVTable(vtable);
  SetDispatch<StmtExprMutator, SBlockNode>(vtable);
  SetDispatch<StmtExprMutator, SBlockRealizeNode>(vtable);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const SBlockNode* op) {
  return VisitBlock(this, op);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::VisitBlock(tirx::StmtExprVisitor* visitor,
                                                          const SBlockNode* op) {
  for (const IterVar& iter_var : op->iter_vars) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(iter_var->dom->min));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(iter_var->dom->extent));
  }
  for (const BufferVar& buf : op->alloc_buffers) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
        kTVMFFIDefRegionKindSimple, [&]() { return visitor->Visit(buf); }));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitBufferMetadata(buf));
  }
  // Define match-buffer targets before visiting reads/writes that may use them.
  // This differs from the old TIRX native order (reads/writes before matches)
  // and agrees with structural traversal's definition-before-use contract.
  for (const MatchBufferRegion& match_buffer_region : op->match_buffers) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
        kTVMFFIDefRegionKindSimple, [&]() { return visitor->Visit(match_buffer_region->buffer); }));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitBufferMetadata(match_buffer_region->buffer));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(match_buffer_region->source));
  }
  for (const TensorRegion& region : op->reads) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(region));
  }
  for (const TensorRegion& region : op->writes) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(region));
  }
  if (op->init.has_value()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(op->init.value()));
  }
  return visitor->Visit(op->body);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const SBlockRealizeNode* op) {
  return VisitBlockRealize(this, op);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::VisitBlockRealize(tirx::StmtExprVisitor* visitor,
                                                                 const SBlockRealizeNode* op) {
  for (const auto& child : op->iter_values) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(child));
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->Visit(op->predicate));
  return visitor->Visit(op->block);
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const SBlockNode* op, InplaceMode inplace_mode) {
  return MutateBlock(this, op, inplace_mode);
}

UnchangedOr<Stmt> StmtExprMutator::MutateBlock(tirx::StmtExprMutator* mutator, const SBlockNode* op,
                                               InplaceMode inplace_mode) {
  // SBlock iteration variables keep their binders; only their domains are expressions here.
  const auto* iters = op->iter_vars.GetArrayObj();
  InplaceMode iter_mode = iters->unique() ? inplace_mode : InplaceMode::kDisallow;
  std::vector<std::pair<size_t, IterVar>> replacements;
  for (size_t i = 0; i < iters->size(); ++i) {
    const auto* iter = (*iters)[i].as<IterVarNode>();
    InplaceMode domain_mode = iter->unique() ? iter_mode : InplaceMode::kDisallow;
    auto domain = mutator->Mutate(iter->dom, domain_mode).as_or_throw<UnchangedOr<Range>>();
    if (domain.UnchangedOrSameAs(iter->dom)) continue;
    if (domain_mode == InplaceMode::kAllow) {
      const_cast<IterVarNode*>(iter)->dom = std::move(domain).ValueUnchecked();
    } else {
      auto updated = ffi::make_object<IterVarNode>(*iter);
      updated->dom = std::move(domain).ValueUnchecked();
      replacements.emplace_back(i, IterVar(std::move(updated)));
    }
  }
  UnchangedOr<ffi::Array<IterVar>> iter_vars = ffi::Unchanged();
  if (!replacements.empty()) {
    if (iter_mode == InplaceMode::kAllow) {
      for (auto& [i, iter] : replacements) {
        const_cast<ffi::ArrayObj*>(iters)->SetItem(i, std::move(iter));
      }
    } else {
      ffi::Array<IterVar> updated = op->iter_vars;
      for (auto& [i, iter] : replacements) updated.Set(i, std::move(iter));
      iter_vars = std::move(updated);
    }
  }
  auto alloc_buffers =
      mutator
          ->WithDefRegionKind(kTVMFFIDefRegionKindSimple,
                              [&] { return mutator->Mutate(op->alloc_buffers, inplace_mode); })
          .as_or_throw<UnchangedOr<ffi::Array<BufferVar>>>();
  auto match_buffers = mutator->Mutate(op->match_buffers, inplace_mode)
                           .as_or_throw<UnchangedOr<ffi::Array<MatchBufferRegion>>>();
  auto reads =
      mutator->Mutate(op->reads, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<TensorRegion>>>();
  auto writes = mutator->Mutate(op->writes, inplace_mode)
                    .as_or_throw<UnchangedOr<ffi::Array<TensorRegion>>>();
  auto init =
      mutator->Mutate(op->init, inplace_mode).as_or_throw<UnchangedOr<ffi::Optional<Stmt>>>();
  auto body = mutator->Mutate(op->body, inplace_mode);
  if (iter_vars.UnchangedOrSameAs(op->iter_vars) &&
      alloc_buffers.UnchangedOrSameAs(op->alloc_buffers) && reads.UnchangedOrSameAs(op->reads) &&
      writes.UnchangedOrSameAs(op->writes) && match_buffers.UnchangedOrSameAs(op->match_buffers) &&
      init.UnchangedOrSameAs(op->init) && body.UnchangedOrSameAs(op->body))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<SBlockNode*>(op);
    if (!iter_vars.IsUnchanged()) writable->iter_vars = std::move(iter_vars).ValueUnchecked();
    if (!alloc_buffers.IsUnchanged())
      writable->alloc_buffers = std::move(alloc_buffers).ValueUnchecked();
    if (!reads.IsUnchanged()) writable->reads = std::move(reads).ValueUnchecked();
    if (!writes.IsUnchanged()) writable->writes = std::move(writes).ValueUnchecked();
    if (!match_buffers.IsUnchanged())
      writable->match_buffers = std::move(match_buffers).ValueUnchecked();
    if (!init.IsUnchanged()) writable->init = std::move(init).ValueUnchecked();
    if (!body.IsUnchanged()) writable->body = std::move(body).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<SBlockNode>(*op);
  if (!iter_vars.IsUnchanged()) copy->iter_vars = std::move(iter_vars).ValueUnchecked();
  if (!alloc_buffers.IsUnchanged()) copy->alloc_buffers = std::move(alloc_buffers).ValueUnchecked();
  if (!reads.IsUnchanged()) copy->reads = std::move(reads).ValueUnchecked();
  if (!writes.IsUnchanged()) copy->writes = std::move(writes).ValueUnchecked();
  if (!match_buffers.IsUnchanged()) copy->match_buffers = std::move(match_buffers).ValueUnchecked();
  if (!init.IsUnchanged()) copy->init = std::move(init).ValueUnchecked();
  if (!body.IsUnchanged()) copy->body = std::move(body).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const SBlockRealizeNode* op, InplaceMode inplace_mode) {
  return MutateBlockRealize(this, op, inplace_mode);
}

UnchangedOr<Stmt> StmtExprMutator::MutateBlockRealize(tirx::StmtExprMutator* mutator,
                                                      const SBlockRealizeNode* op,
                                                      InplaceMode inplace_mode) {
  auto iter_values = mutator->Mutate(op->iter_values, inplace_mode)
                         .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  auto predicate = mutator->Mutate(op->predicate, inplace_mode);
  auto block = mutator->Mutate(op->block, inplace_mode).as_or_throw<UnchangedOr<SBlock>>();
  if (iter_values.UnchangedOrSameAs(op->iter_values) &&
      predicate.UnchangedOrSameAs(op->predicate) && block.UnchangedOrSameAs(op->block))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<SBlockRealizeNode*>(op);
    if (!iter_values.IsUnchanged()) writable->iter_values = std::move(iter_values).ValueUnchecked();
    if (!predicate.IsUnchanged()) writable->predicate = std::move(predicate).ValueUnchecked();
    if (!block.IsUnchanged()) writable->block = std::move(block).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<SBlockRealizeNode>(*op);
  if (!iter_values.IsUnchanged()) copy->iter_values = std::move(iter_values).ValueUnchecked();
  if (!predicate.IsUnchanged()) copy->predicate = std::move(predicate).ValueUnchecked();
  if (!block.IsUnchanged()) copy->block = std::move(block).ValueUnchecked();
  return Stmt(std::move(copy));
}

}  // namespace s_tir
}  // namespace tvm
