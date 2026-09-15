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
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/stmt_functor.h>

#include <functional>

#include "data_type_rewriter.h"
#include "functor_common.h"

namespace tvm {
namespace tirx {

void StmtExprVisitor::InitVTable(VTable* vtable) {
  tvm::ExprVisitor::InitVTable(vtable);
  SetDispatch<StmtExprVisitor, BindNode>(vtable);
  SetDispatch<StmtExprVisitor, AttrStmtNode>(vtable);
  SetDispatch<StmtExprVisitor, IfThenElseNode>(vtable);
  SetDispatch<StmtExprVisitor, ForNode>(vtable);
  SetDispatch<StmtExprVisitor, WhileNode>(vtable);
  SetDispatch<StmtExprVisitor, ReturnNode>(vtable);
  SetDispatch<StmtExprVisitor, BreakNode>(vtable);
  SetDispatch<StmtExprVisitor, ContinueNode>(vtable);
  SetDispatch<StmtExprVisitor, AllocBufferNode>(vtable);
  SetDispatch<StmtExprVisitor, DeclBufferNode>(vtable);
  SetDispatch<StmtExprVisitor, BufferStoreNode>(vtable);
  SetDispatch<StmtExprVisitor, AssertStmtNode>(vtable);
  SetDispatch<StmtExprVisitor, SeqStmtNode>(vtable);
  SetDispatch<StmtExprVisitor, EvaluateNode>(vtable);
  SetDispatch<StmtExprVisitor, SBlockNode>(vtable);
  SetDispatch<StmtExprVisitor, SBlockRealizeNode>(vtable);
  SetDispatch<StmtExprVisitor, ScopeIdDefStmtNode>(vtable);
  SetDispatch<StmtExprVisitor, TilePrimitiveCallNode>(vtable);
  SetDispatch<StmtExprVisitor, BufferRegionNode>(vtable);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const VarNode* op) { return std::nullopt; }

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const OpaqueExprNode* op) {
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const BreakNode* op) { return std::nullopt; }

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const ContinueNode* op) {
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const TensorLoadNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->source));
  for (const auto& child : op->indices) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const BufferRegionNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->buffer));
  for (const auto& range : op->region) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(range->min));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(range->extent));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const TupleNode* op) {
  for (const auto& child : op->fields) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const TupleGetItemNode* op) {
  return this->Visit(op->tuple);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const prim::LetNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->value));
  return this->Visit(op->body);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const CallNode* op) {
  if (op->op.as<OpaqueExprNode>()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->op));
  }
  for (const auto& child : op->args) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const prim::RampNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->base));
  return this->Visit(op->stride);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const prim::BroadcastNode* op) {
  return this->Visit(op->value);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const prim::ShuffleNode* op) {
  for (const auto& child : op->indices) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  for (const auto& child : op->vectors) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const BindNode* op) {
  // Bind has no body -- only visit the value expression.
  return this->Visit(op->value);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const AttrStmtNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->value));
  return this->Visit(op->body);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const ForNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->min));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->extent));
  if (op->step.has_value()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(*op->step));
  }
  return this->Visit(op->body);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const WhileNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->condition));
  return this->Visit(op->body);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const ReturnNode* op) {
  return this->Visit(op->value);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::VisitBufferMetadata(const BufferVar& buffer) {
  for (const auto& child : buffer->shape) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  for (const auto& child : buffer->strides) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(buffer->elem_offset));
  for (const auto& child : buffer->allocated_addr) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  if (buffer->layout.has_value()) {
    const auto* layout = buffer->layout.value().as<TileLayoutNode>();
    if (layout == nullptr) return std::nullopt;
    for (const Iter& iter : layout->shard) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(iter->extent));
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(iter->stride));
    }
    for (const Iter& iter : layout->replica) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(iter->extent));
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(iter->stride));
    }
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const AllocBufferNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return this->Visit(op->buffer); }));
  return VisitBufferMetadata(op->buffer);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const DeclBufferNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->data));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return this->Visit(op->buffer); }));
  return VisitBufferMetadata(op->buffer);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const BufferStoreNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->buffer));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->value));
  for (const auto& child : op->indices) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const IfThenElseNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->then_case));
  if (op->else_case) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->else_case.value()));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const AssertStmtNode* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->condition));
  return this->Visit(op->error_kind);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const SeqStmtNode* op) {
  for (const auto& child : op->seq) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const EvaluateNode* op) {
  return this->Visit(op->value);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const SBlockNode* op) {
  for (const IterVar& iter_var : op->iter_vars) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(iter_var->dom->min));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(iter_var->dom->extent));
  }
  for (const BufferVar& buf : op->alloc_buffers) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(
        this->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() { return this->Visit(buf); }));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(VisitBufferMetadata(buf));
  }
  for (const BufferRegion& region : op->reads) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(region));
  }
  for (const BufferRegion& region : op->writes) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(region));
  }
  for (const MatchBufferRegion& match_buffer_region : op->match_buffers) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->WithDefRegionKind(
        kTVMFFIDefRegionKindSimple, [&]() { return this->Visit(match_buffer_region->buffer); }));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(VisitBufferMetadata(match_buffer_region->buffer));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(match_buffer_region->source));
  }
  if (op->init.has_value()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->init.value()));
  }
  return this->Visit(op->body);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const SBlockRealizeNode* op) {
  for (const auto& child : op->iter_values) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->predicate));
  return this->Visit(op->block);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const ScopeIdDefStmtNode* op) {
  // Flat stmt -- no body. Visit extents (skip deferred defs whose extents
  // are NullOpt) and any preferred_extents.
  if (op->def->extents.has_value()) {
    for (const auto& child : op->def->extents.value()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
    }
  }
  if (op->def->preferred_extents.has_value()) {
    for (const auto& child : op->def->preferred_extents.value()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
    }
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const TilePrimitiveCallNode* op) {
  std::function<ffi::Optional<VisitInterrupt>(const ffi::Any&)> fvisit;
  fvisit = [this, &fvisit](const ffi::Any& e) -> ffi::Optional<VisitInterrupt> {
    if (e == nullptr) return std::nullopt;
    if (auto buffer_region = e.as<BufferRegion>()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(buffer_region.value()));
    } else if (auto var = e.as<Var>(); var && var.value()->ty.as<BufferTypeNode>()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(BufferVar(var.value())));
    } else if (auto expr = e.as<PrimExpr>()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(expr.value()));
    } else if (auto stmt = e.as<Stmt>()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(stmt.value()));
    } else if (auto array = e.as<ffi::Array<ffi::Any>>()) {
      for (const ffi::Any& item : array.value()) {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(fvisit(item));
      }
    }
    return std::nullopt;
  };
  for (const ffi::Any& arg : op->args) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(fvisit(arg));
  }
  for (const auto& [key, value] : op->config) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(fvisit(value));
  }
  return std::nullopt;
}

class StmtMutator::Internal {
 public:
  /*!
   * \brief Mutate array's element by fmutate function.
   *
   * \note Use extra care for copy on write setting.
   *
   * In particular, consider the following case of two reference chains:
   * - strongref0 -> loop0 -> loop1 -> loop2
   * - strongref1 -> loop3 -> loop1 -> loop2
   *
   * Think of the case of calling MutateArray on loop1->loop2(as const reference).
   * When both strongref0 and strongref1 exists, the context does not allow copy
   * on write, even though loop1 uniquely refers to loop2.
   *
   * \param self The pointer to the mutator.
   * \param arr Array to be mutated, const reference is used to allow copy on write
   *            mutation in a recursive visitor.
   * \param fmutate The mutator function.
   * \return The mutated array, a new copy can be created.
   */
  template <typename T, typename F>
  static ffi::Array<T> MutateArray(StmtMutator* self, const ffi::Array<T>& arr, F fmutate) {
    if (self->allow_copy_on_write_ && arr.unique()) {
      // if we allow copy on write, we can directly
      // call the inplace mutate function.
      const_cast<ffi::Array<T>&>(arr).MutateByApply(fmutate);
      return arr;
    } else {
      bool allow_cow = false;
      std::swap(allow_cow, self->allow_copy_on_write_);
      ffi::Array<T> copy = arr.Map(fmutate);
      std::swap(allow_cow, self->allow_copy_on_write_);
      return copy;
    }
  }

  static ffi::Array<IterVar> Mutate(StmtMutator* self, const ffi::Array<IterVar>& arr) {
    auto fmutate = [self](const IterVar& iter_var) {
      PrimExpr min = self->VisitPrimExpr(iter_var->dom->min);
      PrimExpr extent = self->VisitPrimExpr(iter_var->dom->extent);
      if (min.same_as(iter_var->dom->min) && extent.same_as(iter_var->dom->extent)) {
        return iter_var;
      } else {
        return IterVar(Range(min, extent), iter_var->var, iter_var->iter_type,
                       iter_var->thread_tag);
      }
    };
    return MutateArray(self, arr, fmutate);
  }

  static ffi::Array<PrimExpr> Mutate(StmtMutator* self, const ffi::Array<PrimExpr>& arr) {
    auto fmutate = [self](const PrimExpr& e) { return self->VisitPrimExpr(e); };
    return MutateArray(self, arr, fmutate);
  }

  static ffi::Array<Stmt> Mutate(StmtMutator* self, const ffi::Array<Stmt>& arr) {
    auto fmutate = [self](const Stmt& s) { return self->VisitStmt(s); };
    return MutateArray(self, arr, fmutate);
  }

  static ffi::Array<Range> Mutate(StmtMutator* self, const ffi::Array<Range>& arr) {
    auto fmutate = [self](const Range& r) {
      PrimExpr min = self->VisitPrimExpr(r->min);
      PrimExpr extent = self->VisitPrimExpr(r->extent);
      if (min.same_as(r->min) && extent.same_as(r->extent)) {
        return r;
      } else {
        return Range::FromMinExtent(min, extent);
      }
    };
    return MutateArray(self, arr, fmutate);
  }

  static ffi::Array<BufferRegion> Mutate(StmtMutator* self, const ffi::Array<BufferRegion>& arr) {
    auto fmutate = [self](const BufferRegion& buffer_region) {
      BufferVar new_buf = self->VisitBufferUse(buffer_region->buffer);
      ffi::Array<Range> region = Mutate(self, buffer_region->region);
      if (new_buf.same_as(buffer_region->buffer) && region.same_as(buffer_region->region)) {
        return buffer_region;
      } else {
        return BufferRegion(std::move(new_buf), std::move(region));
      }
    };
    return MutateArray(self, arr, fmutate);
  }

  static ffi::Array<MatchBufferRegion> Mutate(StmtMutator* self,
                                              const ffi::Array<MatchBufferRegion>& arr) {
    auto fmutate = [self](const MatchBufferRegion& match_buffer_region) {
      BufferVar new_buf = self->VisitBufferDef(match_buffer_region->buffer, /*alloc_data=*/true);
      BufferVar new_source_buf = self->VisitBufferUse(match_buffer_region->source->buffer);
      ffi::Array<Range> region = Mutate(self, match_buffer_region->source->region);
      if (new_buf.same_as(match_buffer_region->buffer) &&
          new_source_buf.same_as(match_buffer_region->source->buffer) &&
          region.same_as(match_buffer_region->source->region)) {
        return match_buffer_region;
      } else {
        return MatchBufferRegion(std::move(new_buf),
                                 BufferRegion(std::move(new_source_buf), std::move(region)));
      }
    };
    return MutateArray(self, arr, fmutate);
  }
};

Stmt StmtMutator::VisitStmt_(const BindNode* op) {
  // Bind has no body -- only mutate the value expression.
  Expr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const AttrStmtNode* op) {
  PrimExpr value = this->VisitPrimExpr(op->value);
  Stmt body = this->VisitStmt(op->body);
  if (value.same_as(op->value) && body.same_as(op->body)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const ForNode* op) {
  PrimExpr min = this->VisitPrimExpr(op->min);
  PrimExpr extent = this->VisitPrimExpr(op->extent);
  ffi::Optional<PrimExpr> step{std::nullopt};
  if (op->step.has_value()) {
    step = this->VisitPrimExpr(*op->step);
  }
  Stmt body = this->VisitStmt(op->body);
  if (min.same_as(op->min) && extent.same_as(op->extent) && body.same_as(op->body) &&
      step.same_as(op->step)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->min = std::move(min);
    n->extent = std::move(extent);
    n->step = std::move(step);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const WhileNode* op) {
  PrimExpr condition = this->VisitPrimExpr(op->condition);
  Stmt body = this->VisitStmt(op->body);
  if (condition.same_as(op->condition) && body.same_as(op->body)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->condition = std::move(condition);
    n->body = std::move(body);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const ReturnNode* op) {
  Expr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const BreakNode* op) { return ffi::GetRef<Stmt>(op); }

Stmt StmtMutator::VisitStmt_(const ContinueNode* op) { return ffi::GetRef<Stmt>(op); }

BufferVar StmtMutator::VisitBufferDef(const BufferVar& buffer, bool alloc_data) {
  if (auto it = buffer_remap_.find(buffer); it != buffer_remap_.end()) {
    return (*it).second;
  }

  // Visit expression fields (shape, strides, elem_offset) but NOT data.
  // data is a Var definition owned by this buffer, not an expression use.
  // Subclasses that need to remap data can override.
  auto shape = buffer->shape.Map([this](const PrimExpr& e) { return this->VisitPrimExpr(e); });
  auto strides = buffer->strides.Map([this](const PrimExpr& e) { return this->VisitPrimExpr(e); });
  PrimExpr elem_offset = this->VisitPrimExpr(buffer->elem_offset);
  auto allocated_addr =
      buffer->allocated_addr.Map([this](const PrimExpr& e) { return this->VisitPrimExpr(e); });

  // Visit the layout's per-iter extent/stride PrimExprs too: they share dtype
  // semantics with the shape, e.g. ``IndexDataTypeRewriter`` (int32 -> int64)
  // must rewrite layout fields together with the shape, otherwise the layout
  // diverges from the rewritten shape and structural-equal mismatches occur.
  ffi::Optional<Layout> new_layout = buffer->layout;
  bool layout_changed = false;
  if (buffer->layout.has_value()) {
    if (auto opt_tile = buffer->layout.value().as<TileLayoutNode>()) {
      auto remap_iter = [this](const Iter& it) -> Iter {
        PrimExpr new_extent = this->VisitPrimExpr(it->extent);
        PrimExpr new_stride = this->VisitPrimExpr(it->stride);
        if (new_extent.same_as(it->extent) && new_stride.same_as(it->stride)) {
          return it;
        }
        return Iter(new_extent, new_stride, it->axis);
      };
      auto new_shard = opt_tile->shard.Map(remap_iter);
      auto new_replica = opt_tile->replica.Map(remap_iter);
      if (!new_shard.same_as(opt_tile->shard) || !new_replica.same_as(opt_tile->replica)) {
        new_layout = TileLayout(new_shard, new_replica, opt_tile->offset);
        layout_changed = true;
      }
    }
  }

  if (shape.same_as(buffer->shape) && strides.same_as(buffer->strides) &&
      elem_offset.same_as(buffer->elem_offset) && allocated_addr.same_as(buffer->allocated_addr) &&
      !layout_changed) {
    return buffer;
  }
  BufferType new_type(buffer->storage_scope, buffer->dtype, std::move(shape), std::move(strides),
                      std::move(elem_offset), buffer->data_alignment, buffer->offset_factor,
                      std::move(new_layout), std::move(allocated_addr), buffer->span);
  BufferVar new_buf(buffer.name(), std::move(new_type), buffer.span());
  buffer_remap_.Set(buffer, new_buf);
  return new_buf;
}

BufferVar StmtMutator::VisitBufferUse(const BufferVar& buffer) {
  if (auto it = buffer_remap_.find(buffer); it != buffer_remap_.end()) {
    return (*it).second;
  }
  return buffer;
}

Expr StmtExprMutator::VisitExpr_(const VarNode* op) {
  Var var = ffi::GetRef<Var>(op);
  if (var->ty.as<BufferTypeNode>()) {
    return VisitBufferUse(BufferVar(var)).var();
  }
  return var;
}

Expr StmtExprMutator::VisitExpr_(const TensorLoadNode* op) {
  BufferVar old_buf = op->source.as_or_throw<tvm::tirx::BufferVar>();
  BufferVar new_buf = this->VisitBufferUse(old_buf);
  PrimExpr expr = ExprMutator::VisitExpr_(op).as_or_throw<PrimExpr>();
  op = expr.as<TensorLoadNode>();
  TVM_FFI_ICHECK(op != nullptr);
  if (!new_buf.same_as(old_buf)) {
    return BufferLoad(std::move(new_buf), op->indices, op->span);
  }
  return expr;
}

Expr StmtExprMutator::VisitExpr_(const BufferRegionNode* op) {
  BufferVar new_buf = this->VisitBufferUse(op->buffer);
  ffi::Array<Range> new_region = op->region.Map([this](const Range& range) {
    PrimExpr min = this->VisitPrimExpr(range->min);
    PrimExpr extent = this->VisitPrimExpr(range->extent);
    return min.same_as(range->min) && extent.same_as(range->extent)
               ? range
               : Range::FromMinExtent(std::move(min), std::move(extent));
  });
  if (new_buf.same_as(op->buffer) && new_region.same_as(op->region)) {
    return ffi::GetRef<BufferRegion>(op);
  }
  return BufferRegion(std::move(new_buf), std::move(new_region), op->span);
}

Stmt StmtMutator::VisitStmt_(const AllocBufferNode* op) {
  BufferVar new_buf = this->VisitBufferDef(op->buffer, /*alloc_data=*/true);

  if (new_buf.same_as(op->buffer)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->buffer = std::move(new_buf);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const DeclBufferNode* op) {
  Expr data = this->VisitExpr(op->data);
  BufferVar new_buf = this->VisitBufferDef(op->buffer, /*alloc_data=*/false);

  if (new_buf.same_as(op->buffer) && data.same_as(op->data)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->data = std::move(data);
    n->buffer = std::move(new_buf);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const IfThenElseNode* op) {
  PrimExpr condition = this->VisitPrimExpr(op->condition);
  Stmt then_case = this->VisitStmt(op->then_case);
  ffi::Optional<Stmt> else_case = std::nullopt;
  if (op->else_case) {
    else_case = this->VisitStmt(op->else_case.value());
  }
  if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
      else_case.same_as(op->else_case)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->condition = std::move(condition);
    n->then_case = std::move(then_case);
    n->else_case = std::move(else_case);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const BufferStoreNode* op) {
  BufferVar new_buf = this->VisitBufferUse(op->buffer);
  PrimExpr value = this->VisitPrimExpr(op->value);
  ffi::Array<PrimExpr> indices = Internal::Mutate(this, op->indices);

  if (new_buf.same_as(op->buffer) && value.same_as(op->value) && indices.same_as(op->indices)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->buffer = std::move(new_buf);
    n->value = std::move(value);
    n->indices = std::move(indices);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const SeqStmtNode* op) {
  ffi::Array<Stmt> seq = Internal::Mutate(this, op->seq);
  if (seq.same_as(op->seq)) {
    return SeqStmt::Flatten(ffi::GetRef<Stmt>(op));
  } else {
    auto node = CopyOnWrite(op);
    node->seq = std::move(seq);
    return SeqStmt::Flatten(SeqStmt(node));
  }
}

// advanced visit function for seqstmt.
Stmt StmtMutator::VisitSeqStmt_(const SeqStmtNode* op, bool flatten_before_visit,
                                std::function<Stmt(const Stmt&)> fmutate) {
  if (flatten_before_visit) {
    // Pass 1, check if we need to flatten.
    bool need_flatten = false;
    for (size_t i = 0; i < op->seq.size(); ++i) {
      Stmt tmp = (*op)[i];
      if (tmp.as<SeqStmtNode>()) need_flatten = true;
    }
    flatten_before_visit = need_flatten;
  }
  // function to run the visit.
  auto frunvisit = [&](const SeqStmtNode* op) {
    ffi::Array<Stmt> seq = fmutate != nullptr ? Internal::MutateArray(this, op->seq, fmutate)
                                              : Internal::Mutate(this, op->seq);
    if (seq.same_as(op->seq)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->seq = std::move(seq);
      return Stmt(n);
    }
  };
  if (flatten_before_visit) {
    ffi::Array<Stmt> seq;
    SeqStmt::Flattener flattener(&seq);
    flattener(0, op->seq);
    // NOTE: If copy on write is allowed
    // the assignment to seq below will
    // destruct the original seq.
    //
    // Such destruction removes duplicated reference
    // count to children and still enables COW for
    // child Stmt.
    ffi::ObjectPtr<SeqStmtNode> n = CopyOnWrite(op);
    n->seq = std::move(seq);
    return frunvisit(n.operator->());
  } else {
    return frunvisit(op);
  }
}

Stmt StmtMutator::VisitStmt_(const AssertStmtNode* op) {
  PrimExpr condition = this->VisitPrimExpr(op->condition);
  PrimExpr error_kind = this->VisitPrimExpr(op->error_kind);
  ffi::Array<prim::StringImm> message_parts =
      Internal::MutateArray(this, op->message_parts, [this](const prim::StringImm& e) {
        return this->VisitPrimExpr(e).as_or_throw<prim::StringImm>();
      });

  if (condition.same_as(op->condition) && error_kind.same_as(op->error_kind) &&
      message_parts.same_as(op->message_parts)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->condition = std::move(condition);
    n->error_kind = std::move(error_kind).as_or_throw<prim::StringImm>();
    n->message_parts = std::move(message_parts);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const EvaluateNode* op) {
  Expr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->value = std::move(value);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const SBlockNode* op) {
  ffi::Array<IterVar> iter_vars = Internal::Mutate(this, op->iter_vars);
  ffi::Array<BufferVar> alloc_buffers = Internal::MutateArray(
      this, op->alloc_buffers,
      [this](const BufferVar& buf) { return this->VisitBufferDef(buf, /*alloc_data=*/true); });
  ffi::Array<BufferRegion> reads = Internal::Mutate(this, op->reads);
  ffi::Array<BufferRegion> writes = Internal::Mutate(this, op->writes);
  ffi::Array<MatchBufferRegion> match_buffers = Internal::Mutate(this, op->match_buffers);
  ffi::Optional<Stmt> init = std::nullopt;
  if (op->init.has_value()) {
    init = VisitStmt(op->init.value());
  }
  Stmt body = VisitStmt(op->body);
  if (iter_vars.same_as(op->iter_vars) && alloc_buffers.same_as(op->alloc_buffers) &&
      reads.same_as(op->reads) && writes.same_as(op->writes) && body.same_as(op->body) &&
      init.same_as(op->init) && match_buffers.same_as(op->match_buffers)) {
    return ffi::GetRef<SBlock>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->iter_vars = std::move(iter_vars);
    n->alloc_buffers = std::move(alloc_buffers);
    n->reads = std::move(reads);
    n->writes = std::move(writes);
    n->body = std::move(body);
    n->init = std::move(init);
    n->match_buffers = std::move(match_buffers);
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const SBlockRealizeNode* op) {
  ffi::Array<PrimExpr> v = Internal::Mutate(this, op->iter_values);
  PrimExpr pred = this->VisitPrimExpr(op->predicate);
  Stmt block = this->VisitStmt(op->block);
  if (v.same_as(op->iter_values) && pred.same_as(op->predicate) && block.same_as(op->block)) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->iter_values = std::move(v);
    n->predicate = std::move(pred);
    n->block = block.as_or_throw<SBlock>();
    return Stmt(n);
  }
}

Stmt StmtMutator::VisitStmt_(const ScopeIdDefStmtNode* op) {
  // Mutate extents and preferred_extents; deferred defs have nothing to
  // mutate -- pass through.
  bool changed = false;
  ffi::Optional<ffi::Array<PrimExpr>> new_extents = op->def->extents;
  if (op->def->extents.has_value()) {
    ffi::Array<PrimExpr> new_arr;
    for (const auto& e : op->def->extents.value()) {
      PrimExpr ne = this->VisitPrimExpr(e);
      if (!ne.same_as(e)) changed = true;
      new_arr.push_back(ne);
    }
    new_extents = new_arr;
  }
  ffi::Optional<ffi::Array<PrimExpr>> new_pref = op->def->preferred_extents;
  if (op->def->preferred_extents.has_value()) {
    ffi::Array<PrimExpr> new_arr;
    for (const auto& e : op->def->preferred_extents.value()) {
      PrimExpr ne = this->VisitPrimExpr(e);
      if (!ne.same_as(e)) changed = true;
      new_arr.push_back(ne);
    }
    new_pref = new_arr;
  }
  if (!changed) return ffi::GetRef<Stmt>(op);
  ScopeIdDef new_def(op->def->def_ids, new_extents, op->def->scope, new_pref);
  auto n = CopyOnWrite(op);
  n->def = std::move(new_def);
  return Stmt(n);
}

Stmt StmtMutator::VisitStmt_(const tirx::TilePrimitiveCallNode* op) {
  std::function<ffi::Any(const ffi::Any&)> fmutate;
  fmutate = [&](const ffi::Any& e) -> ffi::Any {
    if (e == nullptr) return e;
    if (auto buffer_region = e.as<BufferRegion>()) {
      return Internal::Mutate(this, {buffer_region.value()})[0];
    } else if (auto var = e.as<Var>(); var && var.value()->ty.as<BufferTypeNode>()) {
      return this->VisitBufferUse(BufferVar(var.value()));
    } else if (auto expr = e.as<PrimExpr>()) {
      return this->VisitPrimExpr(expr.value());
    } else if (auto stmt = e.as<Stmt>()) {
      return this->VisitStmt(stmt.value());
    } else if (auto array = e.as<ffi::Array<ffi::Any>>()) {
      return Internal::MutateArray(this, array.value(), fmutate);
    }
    return e;
  };
  ffi::Array<ffi::Any> args = Internal::MutateArray(this, op->args, fmutate);
  // Also mutate PrimExpr values in the config map
  ffi::Map<ffi::String, ffi::Any> config(op->config.begin(), op->config.end());
  bool config_changed = false;
  for (const auto& [key, value] : op->config) {
    ffi::Any new_value = fmutate(value);
    if (!new_value.same_as(value)) {
      config.Set(key, new_value);
      config_changed = true;
    }
  }
  if (args.same_as(op->args) && !config_changed) {
    return ffi::GetRef<Stmt>(op);
  } else {
    auto n = CopyOnWrite(op);
    n->args = std::move(args);
    if (config_changed) n->config = std::move(config);
    return Stmt(n);
  }
}

class IRSubstituteWithDataTypeLegalization : public DataTypeLegalizer {
 public:
  explicit IRSubstituteWithDataTypeLegalization(std::function<ffi::Optional<Expr>(const Var&)> vmap)
      : vmap_(vmap) {}

  using DataTypeLegalizer::VisitExpr_;
  using DataTypeLegalizer::VisitStmt_;

  Expr VisitExpr_(const VarNode* op) final {
    Var var = ffi::GetRef<Var>(op);
    auto ret = vmap_(var);
    if (ret.has_value()) {
      return ret.value();
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();
    // remap var node in attr
    if (auto var_node = op->node.as<Var>()) {
      if (auto mapped_var = vmap_(var_node.value())) {
        return AttrStmt(mapped_var, op->attr_key, op->value, op->body);
      }
    }
    return ret;
  }

 private:
  // Caller provided function that defines the variables to be remapped.
  std::function<ffi::Optional<Expr>(const Var&)> vmap_;
};

Stmt SubstituteWithDataTypeLegalization(Stmt stmt,
                                        std::function<ffi::Optional<PrimExpr>(const Var&)> vmap) {
  auto general_vmap = [vmap = std::move(vmap)](const Var& var) -> ffi::Optional<Expr> {
    if (auto replacement = vmap(var)) return Expr(replacement.value());
    return std::nullopt;
  };
  return IRSubstituteWithDataTypeLegalization(std::move(general_vmap))(std::move(stmt));
}

PrimExpr SubstituteWithDataTypeLegalization(
    PrimExpr expr, std::function<ffi::Optional<PrimExpr>(const Var&)> vmap) {
  auto general_vmap = [vmap = std::move(vmap)](const Var& var) -> ffi::Optional<Expr> {
    if (auto replacement = vmap(var)) return Expr(replacement.value());
    return std::nullopt;
  };
  return IRSubstituteWithDataTypeLegalization(std::move(general_vmap))(std::move(expr))
      .as_or_throw<PrimExpr>();
}

}  // namespace tirx
}  // namespace tvm
