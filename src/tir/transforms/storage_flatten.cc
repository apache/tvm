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
 * \file storage_flatten.cc
 * \brief Flattens storage from multi-dimensional array to 1D buffer access
 */
// The pass definition originates from Halide pipeline.

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target_info.h>
#include <tvm/te/operation.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

#include "../../arith/ir_visitor_with_analyzer.h"
#include "../../runtime/thread_storage_scope.h"
#include "arg_binder.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

using runtime::StorageRank;
using runtime::StorageScope;
using runtime::ThreadScope;

/* Make buffer realize extents and buffer shapes consistent
 *
 * For external buffers, verify that the extents of BufferRealize
 * nodes match the shape of the external buffer.  For internal
 * buffers, rewrite the shape of the Buffer objects to match the
 * extent of the BufferRealize, and rewrite indices of
 * BufferLoad/BufferStore nodes to match.
 */
class BufferShapeLegalize : public StmtExprMutator {
 public:
  explicit BufferShapeLegalize(const Map<Var, Buffer>& extern_buffer_map,
                               IRVisitorWithAnalyzer* bound_analyzer)
      : bound_analyzer_(bound_analyzer) {
    for (auto kv : extern_buffer_map) {
      extern_buffers_.insert(kv.second);
    }
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    // External buffers should not be changed.
    if (extern_buffers_.count(op->buffer)) {
      CHECK_EQ(op->buffer->shape.size(), op->bounds.size())
          << "External buffer realize has mismatched dimension";
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      op = stmt.as<BufferRealizeNode>();
      ICHECK(op);

      for (size_t i = 0; i < op->bounds.size(); i++) {
        PrimExpr eq = bound_analyzer_->Simplify(op->buffer->shape[i] == op->bounds[i]->extent);
        std::ostringstream ss;
        ss << "Dim " << i << " of external buffer " << op->buffer->name << " has shape "
           << op->buffer->shape[i] << ", but is only realized for extent " << op->bounds[i]->extent;
        if (auto eq_int = eq.as<IntImmNode>()) {
          ICHECK(eq_int->value) << ss.str();
        } else {
          stmt = AssertStmt(eq, tvm::tir::StringImm(ss.str()), stmt);
        }
      }
      return stmt;
    }

    // Compute the new buffer shape, new realization bounds, and the
    // offsets to be applied to buffer access.
    Array<PrimExpr> realized_shape;
    Array<PrimExpr> realized_begins;
    Array<Range> new_bounds;
    for (size_t i = 0; i < op->bounds.size(); i++) {
      const Range& bound = op->bounds[i];
      realized_shape.push_back(bound->extent);
      realized_begins.push_back(bound->min);
      new_bounds.push_back({0, bound->extent});
    }

    Buffer key = op->buffer;

    Buffer buf = op->buffer;
    auto write_ptr = buf.CopyOnWrite();
    write_ptr->shape = realized_shape;

    {
      InternalBufferRemap remap;
      remap.remap_to = buf;
      remap.realized_begins = realized_begins;
      remap.in_scope = true;
      internal_buf_map_[key] = remap;
    }

    Stmt stmt = BufferRealize(buf, new_bounds, op->condition, this->VisitStmt(op->body), op->span);

    internal_buf_map_.at(key).in_scope = false;

    return stmt;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();
    ICHECK(op);

    auto it = internal_buf_map_.find(op->buffer);
    if (it != internal_buf_map_.end()) {
      const InternalBufferRemap& entry = it->second;
      ICHECK(entry.in_scope) << "Cannot store to an out-of-scope buffer";
      ICHECK_EQ(entry.realized_begins.size(), op->indices.size())
          << "Inconsistent dimensions for buffer " << op->buffer->name;

      Array<PrimExpr> new_indices;
      for (size_t i = 0; i < entry.realized_begins.size(); i++) {
        new_indices.push_back(op->indices[i] - entry.realized_begins[i]);
      }

      BufferStore updated = GetRef<BufferStore>(op);
      auto write_ptr = updated.CopyOnWrite();
      write_ptr->indices = new_indices;
      write_ptr->buffer = entry.remap_to;
      stmt = updated;
    }

    return stmt;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();
    ICHECK(op);

    auto it = internal_buf_map_.find(op->buffer);
    if (it != internal_buf_map_.end()) {
      const InternalBufferRemap& entry = it->second;
      ICHECK(entry.in_scope) << "Cannot read from an out-of-scope buffer";
      ICHECK_EQ(entry.realized_begins.size(), op->indices.size())
          << "Inconsistent dimensions for buffer " << op->buffer->name;

      Array<PrimExpr> new_indices;
      for (size_t i = 0; i < entry.realized_begins.size(); i++) {
        new_indices.push_back(op->indices[i] - entry.realized_begins[i]);
      }

      BufferLoad updated = GetRef<BufferLoad>(op);
      auto write_ptr = updated.CopyOnWrite();
      write_ptr->indices = new_indices;
      write_ptr->buffer = entry.remap_to;
      expr = updated;
    }

    return expr;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->node->IsInstance<tir::BufferNode>()) {
      // Visit body before checking internal_buf_map_, because we
      // don't know if the BufferNode needs to be changed until we
      // look in the body for a BufferRealizeNode with different
      // extents.
      Stmt body = this->VisitStmt(op->body);

      Buffer buffer = Downcast<tir::Buffer>(op->node);
      auto it = internal_buf_map_.find(buffer);
      if (it != internal_buf_map_.end()) {
        buffer = it->second.remap_to;
        return AttrStmt(it->second.remap_to, op->attr_key, op->value, body);
      }
      return AttrStmt(buffer, op->attr_key, op->value, body);

    } else if (op->attr_key == attr::buffer_bind_scope) {
      return HandleBufferBindScope(op);
    }

    return StmtExprMutator::VisitStmt_(op);
  }

 private:
  Stmt HandleBufferBindScope(const AttrStmtNode* op) {
    Array<ObjectRef> arr = Downcast<Array<ObjectRef>>(op->node);
    ICHECK_EQ(arr.size(), 2U);
    Buffer buffer = Downcast<Buffer>(arr[0]);
    ICHECK(buffer.defined());
    Buffer target = Downcast<Buffer>(arr[1]);
    ICHECK(target.defined());

    auto it = internal_buf_map_.find(target);
    if (it == internal_buf_map_.end()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    const InternalBufferRemap& target_remap = it->second;

    ICHECK(target_remap.in_scope) << "Cannot bind " << buffer->name
                                  << " to the out-of-scope buffer " << target_remap.remap_to->name;

    Call tuple = Downcast<Call>(op->value);
    ICHECK(tuple.defined() && tuple->op.same_as(builtin::tvm_tuple()));

    Array<PrimExpr> new_tuple_args;
    Array<PrimExpr> realized_begins;
    Array<PrimExpr> realized_shape;
    ICHECK_EQ(tuple->args.size(), target_remap.realized_begins.size() * 2);
    for (size_t i = 0; i < target_remap.realized_begins.size(); i++) {
      PrimExpr parent_begin = tuple->args[2 * i];
      PrimExpr view_extent = tuple->args[2 * i + 1];
      // Offset the begin of the buffer view by the offset of the target buffer.
      new_tuple_args.push_back(parent_begin - target_remap.realized_begins[i]);
      // Keep the extent of the buffer view the same.
      new_tuple_args.push_back(view_extent);
      // Use the extent of the buffer view to define the buffer view's shape.
      realized_shape.push_back(view_extent);
      // Within the buffer view, indices start at 0.
      realized_begins.push_back(0);
    }

    Buffer key = buffer;

    auto write_ptr = buffer.CopyOnWrite();
    write_ptr->shape = realized_shape;

    {
      InternalBufferRemap remap;
      remap.realized_begins = realized_begins;
      remap.remap_to = buffer;
      remap.in_scope = true;
      internal_buf_map_[key] = remap;
    }

    Stmt stmt = AttrStmt(Array<ObjectRef>{buffer, target_remap.remap_to}, op->attr_key,
                         Call(tuple->dtype, tuple->op, new_tuple_args, tuple->span),
                         this->VisitStmt(op->body));
    internal_buf_map_.at(key).in_scope = false;
    return stmt;
  }

  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> extern_buffers_;

  struct InternalBufferRemap {
    Buffer remap_to;
    Array<PrimExpr> realized_begins;
    bool in_scope;
  };

  std::unordered_map<Buffer, InternalBufferRemap, ObjectPtrHash, ObjectPtrEqual> internal_buf_map_;

  IRVisitorWithAnalyzer* bound_analyzer_;
};

/* Apply dimension alignment restrictions
 *
 * Buffers annotated with attr::buffer_dim_align may need to have
 * strides defined such that they are no longer in a compact shape.
 * After this pass, buffers have stride definitions to include these
 * alignment restrictions, and attr::buffer_dim_align annotations have
 * been removed.
 */
class BufferStrideLegalize : public StmtExprMutator {
 public:
  explicit BufferStrideLegalize(const Map<Var, Buffer>& extern_buffer_map,
                                IRVisitorWithAnalyzer* bound_analyzer)
      : bound_analyzer_(bound_analyzer) {
    for (auto kv : extern_buffer_map) {
      Buffer buf = kv.second;
      Buffer with_strides = WithStrides(buf);
      {
        BufferEntry entry;
        entry.remap_to = with_strides;
        entry.in_scope = true;
        entry.is_external = true;
        buf_map_[buf] = entry;
      }
      updated_extern_buffer_map_.Set(kv.first, with_strides);
    }
  }

  Map<Var, Buffer> UpdatedExternBufferMap() const { return updated_extern_buffer_map_; }

  Buffer WithStrides(Buffer buf) {
    auto it = buf_map_.find(buf);
    if (it != buf_map_.end()) {
      const BufferEntry& entry = it->second;
      ICHECK(entry.in_scope) << "Cannot annotate an out-of-scope buffer";
      return entry.remap_to;
    }

    if (buf->strides.size()) {
      ICHECK_EQ(buf->strides.size(), buf->shape.size());
      return buf;
    }

    Array<PrimExpr> shape = buf->shape;

    // Keeping this to have matched behavior to previous version.
    // There are many parts of the codebase that assume that a strided
    // array cannot be compact.
    if (dim_align_.count(buf) == 0) {
      return buf;
    }

    std::vector<PrimExpr> rstrides;
    const std::vector<DimAlignInfo>& avec = dim_align_[buf];
    int first_dim = 0;
    PrimExpr stride = make_const(shape[first_dim].dtype(), 1);
    for (size_t i = shape.size(); i != 0; --i) {
      size_t dim = i - 1;
      if (dim < avec.size() && avec[dim].align_factor != 0) {
        PrimExpr factor = make_const(stride.dtype(), avec[dim].align_factor);
        PrimExpr offset = make_const(stride.dtype(), avec[dim].align_offset);
        stride = stride + indexmod(factor + offset - indexmod(stride, factor), factor);
        stride = bound_analyzer_->Simplify(stride);
      }
      rstrides.push_back(stride);
      stride = stride * shape[dim];
    }

    auto ptr = buf.CopyOnWrite();
    ptr->strides = Array<PrimExpr>(rstrides.rbegin(), rstrides.rend());

    return buf;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::buffer_dim_align) {
      auto buffer = Downcast<tir::Buffer>(op->node);
      const CallNode* tuple = op->value.as<CallNode>();
      ICHECK(tuple && tuple->op.same_as(builtin::tvm_tuple()));
      auto& vinfo = dim_align_[buffer];
      int dim = tuple->args[0].as<IntImmNode>()->value;
      if (static_cast<size_t>(dim) >= vinfo.size()) {
        vinfo.resize(dim + 1);
      }
      vinfo[dim].align_factor = tuple->args[1].as<IntImmNode>()->value;
      vinfo[dim].align_offset = tuple->args[2].as<IntImmNode>()->value;

      return this->VisitStmt(op->body);
    } else if (op->attr_key == attr::buffer_bind_scope) {
      Array<ObjectRef> arr = Downcast<Array<ObjectRef>>(op->node);
      ICHECK_EQ(arr.size(), 2U);
      Buffer source = Downcast<Buffer>(arr[0]);
      Buffer target_with_strides = WithStrides(Downcast<Buffer>(arr[1]));
      Buffer source_with_strides = WithStrides(source);

      {
        BufferEntry entry;
        entry.remap_to = source_with_strides;
        entry.in_scope = true;
        entry.is_external = false;
        buf_map_[source] = entry;
      }

      Stmt body = this->VisitStmt(op->body);

      return AttrStmt(Array<ObjectRef>{source_with_strides, target_with_strides}, op->attr_key,
                      op->value, body, op->span);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    Buffer key = op->buffer;
    Buffer with_strides = WithStrides(op->buffer);
    {
      BufferEntry entry;
      entry.remap_to = with_strides;
      entry.in_scope = true;
      entry.is_external = false;
      buf_map_[key] = entry;
    }

    Stmt stmt = StmtExprMutator::VisitStmt_(op);

    buf_map_[key].in_scope = false;
    op = stmt.as<BufferRealizeNode>();
    ICHECK(op);

    return BufferRealize(with_strides, op->bounds, op->condition, op->body, op->span);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();

    auto it = buf_map_.find(op->buffer);
    ICHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << op->buffer;
    const BufferEntry& e = it->second;
    ICHECK(e.in_scope) << "Cannot read a buffer that is already out of scope";

    return BufferLoad(e.remap_to, op->indices, op->span);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();

    auto it = buf_map_.find(op->buffer);
    ICHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << op->buffer;
    const BufferEntry& e = it->second;
    ICHECK(e.in_scope) << "Cannot write to a buffer that is already out of scope";

    return BufferStore(e.remap_to, op->value, op->indices, op->span);
  }

 private:
  Map<Var, Buffer> updated_extern_buffer_map_;

  struct DimAlignInfo {
    int align_factor{0};
    int align_offset{0};
  };

  // Dimension alignment
  std::unordered_map<Buffer, std::vector<DimAlignInfo>, ObjectPtrHash, ObjectPtrEqual> dim_align_;

  struct BufferEntry {
    Buffer remap_to;
    bool in_scope;
    bool is_external;
  };

  std::unordered_map<Buffer, BufferEntry, ObjectPtrHash, ObjectPtrEqual> buf_map_;

  IRVisitorWithAnalyzer* bound_analyzer_;
};

/* Use the scope of IterVar to determine storage scope.
 *
 * For buffers that do not have an explicit storage scope defined, a
 * reasonable storage scope may be defined based on the thread scope
 * that contains the buffer's allocation.  All other buffers without a
 * scope are assigned to global scope.
 */
class ThreadScopePropagate : public StmtExprMutator {
 public:
  explicit ThreadScopePropagate(const Map<Var, Buffer>& extern_buffer_map) {
    // External buffers shouldn't be overwritten, even if they have a
    // BufferRealizeNode.
    for (auto kv : extern_buffer_map) {
      external_buffers_.insert(kv.second);
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = buf_remap_.find(GetRef<Var>(op));
    if (it != buf_remap_.end()) {
      return it->second->data;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    ICHECK_NE(op->attr_key, attr::buffer_dim_align)
        << "StorageFlattener assumes that all buffers have accurate strides, "
        << "and all buffer_dim_align annotations are removed.  "
        << "Please run BufferStrideLegalize first.";

    if (op->attr_key == attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ThreadScope ts = ThreadScope::Create(iv->thread_tag);
      curr_thread_scope_.push_back(ts);
      Stmt stmt = StmtExprMutator::VisitStmt_(op);
      curr_thread_scope_.pop_back();
      return stmt;
    } else if (op->attr_key == attr::buffer_bind_scope) {
      return HandleBufferBindScope(op);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    Var old_var = op->buffer->data;

    // Don't remap buffers that already have an explicit scope,
    // or external buffers.
    std::string str_scope = GetPtrStorageScope(old_var);
    if ((str_scope.length() > 0) || external_buffers_.count(op->buffer)) {
      return StmtExprMutator::VisitStmt_(op);
    }

    ICHECK_EQ(buf_remap_.count(old_var), 0)
        << "Buffer var " << op->buffer->data << " appears in multiple BufferRealize nodes";

    StorageScope skey;
    if (curr_thread_scope_.size() == 0) {
      skey.rank = StorageRank::kGlobal;
    } else {
      skey.rank = runtime::DefaultStorageRank(curr_thread_scope_.back().rank);
    }

    auto ptr_type = old_var->type_annotation.as<PointerTypeNode>();
    ICHECK(ptr_type);
    Var new_var(old_var->name_hint, PointerType(ptr_type->element_type, skey.to_string()),
                old_var->span);

    Buffer buf = op->buffer;
    buf.CopyOnWrite()->data = new_var;

    buf_remap_[old_var] = buf;

    Stmt body = this->VisitStmt(op->body);
    return BufferRealize(buf, op->bounds, op->condition, body, op->span);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();
    ICHECK(op);

    auto it = buf_remap_.find(op->buffer->data);
    if (it != buf_remap_.end()) {
      return BufferLoad(it->second, op->indices, op->span);
    } else {
      return expr;
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();
    ICHECK(op);

    auto it = buf_remap_.find(op->buffer->data);
    if (it != buf_remap_.end()) {
      return BufferStore(it->second, op->value, op->indices, op->span);
    } else {
      return stmt;
    }
  }

 private:
  Stmt HandleBufferBindScope(const AttrStmtNode* op) {
    Array<ObjectRef> arr = Downcast<Array<ObjectRef>>(op->node);
    ICHECK_EQ(arr.size(), 2U);
    Buffer buffer = Downcast<Buffer>(arr[0]);
    ICHECK(buffer.defined());
    Buffer target = Downcast<Buffer>(arr[1]);
    ICHECK(target.defined());

    bool needs_rewrite = false;

    {
      auto it = buf_remap_.find(buffer->data);
      if (it != buf_remap_.end()) {
        needs_rewrite = true;
        buffer = it->second;
      }
    }

    {
      auto it = buf_remap_.find(target->data);
      if (it != buf_remap_.end()) {
        needs_rewrite = true;
        target = it->second;
      }
    }

    if (needs_rewrite) {
      Stmt body = this->VisitStmt(op->body);
      return AttrStmt(Array<ObjectRef>{buffer, target}, op->attr_key, op->value, body);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buf_remap_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> external_buffers_;

  // The current thread scope.
  std::vector<ThreadScope> curr_thread_scope_;
};

/* Map buffer binds to their source buffer
 *
 * Buffers defined using an attr::buffer_bind_scope annotation are
 * views into some linked buffer, potentially into some restricted
 * subregion of that buffer.  This pass identifies such buffers, then
 * rewrites all access of the bound buffers to be access into the
 * linked buffer.
 */
class BufferBindUnwrapper : public StmtExprMutator {
 public:
  explicit BufferBindUnwrapper(const Map<Var, Buffer>& extern_buffer_map,
                               IRVisitorWithAnalyzer* bound_analyzer)
      : bound_analyzer_(bound_analyzer) {
    for (auto kv : extern_buffer_map) {
      BufferEntry e;
      e.buffer = kv.second;
      e.external = true;
      buf_map_[kv.second.get()] = std::move(e);
    }
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<StoreNode>();
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() && !it->second.same_as(op->buffer_var)) {
      // TODO(Lunderberg): Change from warning to error once all mixed
      // use of physical/logical layouts is removed.
      DLOG(WARNING) << op->buffer_var << " was declared as buffer (buffer_bind_scope), "
                    << "but is accessed as a pointer (StoreNode).";

      ICHECK(it->second.as<VarNode>());
      Var new_buf_var = Downcast<Var>(it->second);
      return Store(new_buf_var, op->value, op->index, op->predicate);
    } else {
      return stmt;
    }
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<LoadNode>();
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() && !it->second.same_as(op->buffer_var)) {
      // TODO(Lunderberg): Change from warning to error once all mixed
      // use of physical/logical layouts is removed.
      DLOG(WARNING) << op->buffer_var << " was declared as buffer (buffer_bind_scope), "
                    << "but is accessed as a pointer (LoadNode).";

      ICHECK(it->second.as<VarNode>());
      Var new_buf_var = Downcast<Var>(it->second);
      return Load(op->dtype, new_buf_var, op->index, op->predicate);
    } else {
      return expr;
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    ICHECK_NE(op->attr_key, attr::buffer_dim_align)
        << "BufferBindUnwrapper assumes that all buffers have accurate strides, "
        << "and all buffer_dim_align annotations are removed.  "
        << "Please run BufferStrideLegalize first.";

    if (op->attr_key == attr::buffer_bind_scope) {
      return HandleBufferBindScope(op);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_remap_.find(op);
    if (it != var_remap_.end()) {
      return it->second;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  Array<PrimExpr> remap_indices(Array<PrimExpr> indices, Array<PrimExpr> begins,
                                Array<PrimExpr> extents) {
    ICHECK_EQ(begins.size(), extents.size());

    if (begins.size() == 0) {
      return indices;
    }

    ICHECK_EQ(begins.size(), indices.size());

    Array<PrimExpr> out;
    for (size_t i = 0; i < begins.size(); i++) {
      out.push_back(begins[i] + indices[i]);
    }
    return out;
  }

  Array<Range> remap_bounds(Array<Range> bounds, Array<PrimExpr> begins, Array<PrimExpr> extents) {
    ICHECK_EQ(begins.size(), extents.size());

    if (begins.size() == 0) {
      return bounds;
    }

    ICHECK_EQ(begins.size(), bounds.size());

    Array<Range> out;
    for (size_t i = 0; i < begins.size(); i++) {
      out.push_back(Range::FromMinExtent(bounds[i]->min + begins[i], bounds[i]->extent));
    }
    return out;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();

    auto it = buf_map_.find(op->buffer.get());
    ICHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << op->buffer;
    const BufferEntry& e = it->second;
    ICHECK(e.in_scope) << "Cannot read a buffer that is already out of scope";

    if (e.remap) {
      return BufferLoad(e.remap->target,
                        remap_indices(op->indices, e.remap->begins, e.remap->extents), op->span);
    } else {
      return expr;
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();

    auto it = buf_map_.find(op->buffer.get());
    ICHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << op->buffer;
    const BufferEntry& e = it->second;
    ICHECK(e.in_scope) << "Cannot write to a buffer that is already out of scope";

    if (e.remap) {
      return BufferStore(e.remap->target, op->value,
                         remap_indices(op->indices, e.remap->begins, e.remap->extents), op->span);
    } else {
      return stmt;
    }
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    const auto& key = op->buffer.get();

    bool is_external = false;

    if (buf_map_.count(key)) {
      ICHECK(buf_map_.at(key).external)
          << "BufferRealize node for internal buffer " << op->buffer << " occurred multiple times.";

      is_external = true;
    } else {
      BufferEntry e;
      e.bounds = op->bounds;
      e.buffer = op->buffer;
      buf_map_[key] = std::move(e);
    }

    Stmt stmt = StmtExprMutator::VisitStmt_(op);

    if (is_external) {
      buf_map_[key].in_scope = false;
    }

    return stmt;
  }

  Stmt VisitStmt_(const PrefetchNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<PrefetchNode>();
    ICHECK(op != nullptr);

    const auto& key = op->buffer.get();
    auto it = buf_map_.find(key);
    ICHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << key;
    const BufferEntry& e = it->second;

    ICHECK(e.in_scope) << "Read a buffer that is already out of scope";
    ICHECK_EQ(e.buffer->shape.size(), op->bounds.size())
        << "Prefetch dim should be the same as buffer dim";

    if (e.remap) {
      return Prefetch(e.remap->target, remap_bounds(op->bounds, e.remap->begins, e.remap->extents),
                      op->span);
    } else {
      return stmt;
    }
  }

 private:
  // The specific tensor data layout is not determined before
  // StorageFlatten pass. We use buffer_bind_scope
  // to specify before hand we want to bind a subregion
  // of tensor to a symbolic buffer, which get used in extern.
  //
  // Example:
  //
  // realize A in range [i*4, extent=10) {
  //   bind Ab to A in [i*4+1, extent=4) {
  //     call_func(Ab.ptr, Ab.shape[0])
  //   }
  // }
  //
  // After StorageFlatten
  //
  // alloc A[10]
  //   call(A + 1,  4)
  //
  // Buffer is a protocol to declare specific
  // data layout and shape we expect.
  // So this function need to check:
  // - If the bind range is within the realize range
  // - If we can match the requirement of buffer
  // - Remap variables such as Ab.ptr to the actual value.
  //
  // Here are a few possible failure cases:
  // - Buffer is declared to have constant shape,
  //   but we try to bind it to a different one.
  // - Buffer is declared to be compact(no strides)
  //   but this binded region is a subregion of
  //   a matrix(tensor), which means it requires strides.
  //
  // We do support a few relaxed case, such as bindingx
  // region with shape [1, 1, n, m] to buffer with shape [n, m]
  Stmt HandleBufferBindScope(const AttrStmtNode* op) {
    // Unpack information from Attribute node
    RemapInfo remap;

    Array<ObjectRef> arr = Downcast<Array<ObjectRef>>(op->node);
    ICHECK_EQ(arr.size(), 2U);
    const Buffer source = Downcast<Buffer>(arr[0]);
    ICHECK(source.defined());
    remap.target = Downcast<Buffer>(arr[1]);
    ICHECK(remap.target.defined());
    const CallNode* tuple = op->value.as<CallNode>();
    ICHECK(tuple && tuple->op.same_as(builtin::tvm_tuple()));

    for (size_t i = 0; i < tuple->args.size(); i += 2) {
      remap.begins.push_back(tuple->args[i]);
      remap.extents.push_back(tuple->args[i + 1]);
    }

    // Determine bounds in the target buffer
    auto it = buf_map_.find(remap.target.get());
    ICHECK(it != buf_map_.end()) << "Cannot find buffer " << remap.target << " @ "
                                 << remap.target.get();
    const BufferEntry& target_info = it->second;
    ICHECK(target_info.in_scope) << "Cannot bind to a buffer that is out of scope";
    ICHECK_EQ(remap.begins.size(), target_info.buffer->shape.size())
        << "Incorrect number of arguments in buffer_bind_scope attribute.  "
        << "Expected (min_0, extent_0, min_1, extent_0, ..., min_N, extent_N).";

    if (target_info.bounds.size() > 0) {
      Array<PrimExpr> mapped_begins;
      for (size_t i = 0; i < target_info.buffer->shape.size(); ++i) {
        mapped_begins.push_back(remap.begins[i] - target_info.bounds[i]->min);
      }
      remap.begins = std::move(mapped_begins);
    }

    ICHECK(target_info.remap == nullptr) << "Indirect remapping not handled";

    for (size_t i = 0; i < remap.begins.size(); i++) {
      remap.begins.Set(i, bound_analyzer_->Simplify(remap.begins[i]));
      remap.extents.Set(i, bound_analyzer_->Simplify(remap.extents[i]));
    }

    // Add a buffer remap entry
    {
      BufferEntry source_info;
      source_info.buffer = source;
      source_info.remap = std::make_unique<RemapInfo>(remap);

      buf_map_[source.get()] = std::move(source_info);
    }

    // Define remappings of any remaining Variables (e.g. Store/Load nodes).
    ArgBinder binder(&var_remap_);

    // Define a view that represents the source's view into the target
    // buffer.  This Buffer object is only used to define the mapping
    // to the target buffer, and never actually appears in the TIR
    // graph.
    Buffer view = remap.target.MakeSlice(remap.begins, remap.extents);
    if (source->strides.size() == 0) {
      ICHECK_EQ(view->strides.size(), 0U)
          << "Cannot bind a compact buffer to a strided buffer" << view->strides;
    } else {
      // Add explicit strides to the view, in order to bind to source.strides[i].
      view = view.MakeStrideView();
    }
    binder.BindBuffer(source, view, source->name, true);

    // Apply the remaps
    Stmt body = op->body;
    body = MergeNest(binder.asserts(), body);
    body = MergeNest(binder.init_nest(), body);
    body = this->VisitStmt(body);
    // remove the binds
    for (const Var& v : binder.defs()) {
      var_remap_.erase(v.get());
    }
    return body;
  }

  struct RemapInfo {
    Buffer target;
    Array<PrimExpr> begins;
    Array<PrimExpr> extents;
  };

  // The buffer entry in the flatten map
  struct BufferEntry {
    // The storage buffer
    Buffer buffer;
    // the bounds of realization, can be null, means everything
    Region bounds;
    // Whether the buffer is external
    bool external{false};
    // Whether we are within the allocation scope of the buffer.
    bool in_scope{true};

    // The buffer to which the storage buffer should be remapped.
    std::unique_ptr<RemapInfo> remap{nullptr};

    PrimExpr ElemOffset() const {
      ICHECK(remap);

      Buffer copy = remap->target;
      {
        Array<PrimExpr> shape;
        for (auto r : bounds) {
          shape.push_back(r->extent);
        }
        copy.CopyOnWrite()->shape = std::move(shape);
      }

      Buffer target_slice = copy.MakeSlice(remap->begins, remap->extents);
      if (buffer->strides.size() == 0) {
        ICHECK_EQ(target_slice->strides.size(), 0U)
            << "Trying to bind compact buffer to strided one strides=" << target_slice->strides;
      } else {
        target_slice = target_slice.MakeStrideView();
      }

      return copy->ElemOffset(remap->begins);
    }
  };

  // The buffer assignment map
  // Variable remap
  std::unordered_map<const VarNode*, PrimExpr> var_remap_;
  // Buffer map
  std::unordered_map<const BufferNode*, BufferEntry> buf_map_;
  // Analyzer for the variable bounds, used to simplify the bounds populator. We really need the
  // analyzer from it. However
  IRVisitorWithAnalyzer* bound_analyzer_;
};

class StorageFlattener : public StmtExprMutator {
 public:
  explicit StorageFlattener(const Map<Var, Buffer>& extern_buffer_map, int cache_line_size,
                            bool create_bound_attributes, IRVisitorWithAnalyzer* bound_analyzer)
      : bound_analyzer_(bound_analyzer), create_bound_attributes_(create_bound_attributes) {
    for (auto kv : extern_buffer_map) {
      BufferEntry e;
      e.buffer = kv.second;
      e.external = true;
      buf_map_[kv.second] = e;
    }
    cache_line_size_ = cache_line_size;
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<StoreNode>();
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() && !it->second.same_as(op->buffer_var)) {
      ICHECK(it->second.as<VarNode>());
      Var buf_var = Downcast<Var>(it->second);
      return Store(buf_var, op->value, op->index, op->predicate);
    } else {
      return stmt;
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    ICHECK_NE(op->attr_key, attr::buffer_dim_align)
        << "StorageFlattener assumes that all buffers have accurate strides, "
        << "and all buffer_dim_align annotations are removed.  "
        << "Please run BufferStrideLegalize first.";

    ICHECK_NE(op->attr_key, attr::buffer_bind_scope)
        << "StorageFlattener assumes that all buffer binds have already been applied.  "
        << "Please run BufferBindUnwrapper first.";

    if (op->attr_key == attr::double_buffer_scope && op->node->IsInstance<tir::BufferNode>()) {
      auto buffer = Downcast<tir::Buffer>(op->node);
      Stmt body = this->VisitStmt(op->body);
      auto it = buf_map_.find(buffer);
      ICHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << buffer;
      body = AttrStmt(it->second.buffer->data, op->attr_key, op->value, std::move(body));
      return body;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (create_bound_attributes_) shape_collector_.clear();
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();

    const auto& key = op->buffer;

    auto it = buf_map_.find(key);
    ICHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << key;

    const BufferEntry& e = it->second;
    ICHECK(!e.released) << "Read a buffer that is already out of scope";

    Stmt body = e.buffer.vstore(e.RelIndex(op->indices), op->value);
    if (create_bound_attributes_ && ShapeIsValid(e.buffer->shape)) {
      shape_collector_.push_back(std::make_pair(e.buffer->data, e.buffer->shape));
    }
    // To create bound attribute collector should has at least one item.
    if (create_bound_attributes_ && shape_collector_.size()) {
      for (size_t i = 0; i < shape_collector_.size(); ++i) {
        body = AttrStmt(shape_collector_[i].first, tir::attr::buffer_bound,
                        MakeBound(e.buffer->dtype, shape_collector_[i].second), body);
      }
    }
    return body;
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    const auto& key = op->buffer;

    if (buf_map_.count(key)) {
      ICHECK(buf_map_.at(key).external);
      return this->VisitStmt(op->body);
    } else {
      // create a buffer entry
      BufferEntry e;
      e.bounds = op->bounds;

      ICHECK_EQ(op->buffer->shape.size(), op->bounds.size())
          << "Inconsistent buffer shape and realization shape for " << op->buffer;

      Array<PrimExpr> shape = op->buffer->shape;
      StorageScope skey = StorageScope::Create(GetPtrStorageScope(op->buffer->data));

      // use small alignment for small arrays
      auto dtype = op->buffer->dtype;
      int32_t const_size = AllocateNode::constant_allocation_size(shape);
      int align = GetTempAllocaAlignment(dtype, const_size);
      if (skey.tag.length() != 0) {
        MemoryInfo info = GetMemoryInfo(skey.to_string());
        if (info.defined()) {
          align = (info->max_simd_bits + dtype.bits() - 1) / dtype.bits();
          ICHECK_LE(const_size * dtype.bits(), info->max_num_bits)
              << "Allocation exceed bound of memory tag " << skey.to_string();
        }
      }
      Array<PrimExpr> strides = op->buffer->strides;

      e.buffer = Buffer(op->buffer->data, op->buffer->dtype, shape, strides, PrimExpr(),
                        op->buffer->name, align, 0, kDefault);

      buf_map_[key] = e;
      Stmt body = this->VisitStmt(op->body);
      buf_map_[key].released = true;
      Stmt ret;

      DataType storage_type = e.buffer->dtype;
      // specially handle bool, lower its storage
      // type to beDataType::Int(8)(byte)
      if (storage_type == DataType::Bool()) {
        storage_type = DataType::Int(8);
      }
      if (strides.size() != 0) {
        int first_dim = 0;
        ret = Allocate(e.buffer->data, storage_type,
                       {e.buffer->strides[first_dim] * e.buffer->shape[first_dim]},
                       make_const(DataType::Bool(e.buffer->dtype.lanes()), true), body);
      } else {
        shape = e.buffer->shape;
        if (shape.size() == 0) {
          shape.push_back(make_const(DataType::Int(32), 1));
        }
        ret = Allocate(e.buffer->data, storage_type, shape,
                       make_const(DataType::Bool(e.buffer->dtype.lanes()), true), body);
      }

      if (create_bound_attributes_ && ShapeIsValid(e.buffer->shape)) {
        ret = AttrStmt(e.buffer->data, tir::attr::buffer_bound,
                       MakeBound(e.buffer->dtype, e.buffer->shape), ret);
      }
      return ret;
    }
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<LoadNode>();
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() && !it->second.same_as(op->buffer_var)) {
      ICHECK(it->second.as<VarNode>());
      Var buf_var = Downcast<Var>(it->second);
      return Load(op->dtype, buf_var, op->index, op->predicate);
    } else {
      return expr;
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_remap_.find(op);
    if (it != var_remap_.end()) {
      return it->second;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();

    const auto& key = op->buffer;

    auto it = buf_map_.find(key);
    ICHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << key;
    const BufferEntry& e = it->second;
    ICHECK(!e.released) << "Read a buffer that is already out of scope";

    if (create_bound_attributes_ && ShapeIsValid(e.buffer->shape)) {
      shape_collector_.push_back(std::make_pair(e.buffer->data, e.buffer->shape));
    }
    return e.buffer.vload(e.RelIndex(op->indices), e.buffer->dtype);
  }

  Stmt VisitStmt_(const PrefetchNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<PrefetchNode>();
    ICHECK(op != nullptr);

    const auto& key = op->buffer;
    auto it = buf_map_.find(key);
    ICHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << key;
    const BufferEntry& e = it->second;

    ICHECK(!e.released) << "Read a buffer that is already out of scope";
    ICHECK_EQ(e.buffer->shape.size(), op->bounds.size())
        << "Prefetch dim should be the same as buffer dim";

    int block_size = 1, elem_cnt = cache_line_size_ / e.buffer->dtype.bytes();

    int starts = op->bounds.size() - 1;

    while (starts > 0) {
      auto* shape_as_int = e.buffer->shape[starts].as<IntImmNode>();
      if (shape_as_int == nullptr || block_size * shape_as_int->value > elem_cnt) break;
      block_size *= static_cast<int>(shape_as_int->value);
      starts--;
    }
    PrimExpr stride(elem_cnt / block_size);

    Array<PrimExpr> args;
    std::vector<Var> vars;

    for (int i = op->bounds.size() - 1; i > starts; --i) {
      args.push_back(op->bounds[i]->min);
    }
    auto& func_name = op->buffer->name;
    vars.push_back(Var("prefetch." + func_name + "." + std::to_string(starts), DataType::Int(32)));
    args.push_back(op->bounds[starts]->min + stride * vars.back());
    for (int i = starts - 1; i >= 0; --i) {
      vars.push_back(Var("prefetch." + func_name + "." + std::to_string(i), DataType::Int(32)));
      args.push_back(vars.back() + op->bounds[i]->min);
    }
    for (int i = starts; i >= 0; --i) {
      if (i < starts) {
        stmt = For(vars[i], 0, op->bounds[i]->extent, ForKind::kSerial, stmt);
      } else {
        PrimExpr load = e.buffer.vload(e.RelIndex(args), e.buffer->dtype);
        PrimExpr address = Call(DataType::Handle(), builtin::address_of(), {load});
        PrimExpr prefetch = Call(op->buffer->dtype, builtin::prefetch(), {address, 0, 3, 1});
        stmt = Evaluate(prefetch);
        PrimExpr extent = (op->bounds[i]->extent - 1) / stride + 1;
        stmt = For(vars[i], 0, extent, ForKind::kSerial, stmt);
      }
    }
    return stmt;
  }

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    LOG(FATAL) << "ProducerLoad cannot appear in a valid TIR PrimFunc.  "
               << "Please run SchedulePostProcToPrimFunc first.";
    return PrimExpr();
  }

  Stmt VisitStmt_(const ProducerStoreNode* op) final {
    LOG(FATAL) << "ProducerStore cannot appear in a valid TIR PrimFunc.  "
               << "Please run SchedulePostProcToPrimFunc first.";
    return Stmt();
  }

  Stmt VisitStmt_(const ProducerRealizeNode* op) final {
    LOG(FATAL) << "ProducerRealize cannot appear in a valid TIR PrimFunc.  "
               << "Please run SchedulePostProcToPrimFunc first.";
    return Stmt();
  }

 private:
  // The buffer entry in the flatten map
  struct DimAlignInfo {
    int align_factor{0};
    int align_offset{0};
  };
  // The buffer entry in the flatten map
  struct BufferEntry {
    // the buffer of storage
    Buffer buffer;
    // the bounds of realization, can be null, means everything
    Region bounds;
    // Whether the buffer is external
    bool external{false};
    // Whether we are out of allocation bounds and buffer get released.
    bool released{false};
    // relative index
    inline Array<PrimExpr> RelIndex(Array<PrimExpr> args) const {
      if (bounds.size() != 0) {
        Array<PrimExpr> index;
        ICHECK_EQ(bounds.size(), args.size());
        for (size_t i = 0; i < bounds.size(); ++i) {
          index.push_back(args[i] - bounds[i]->min);
        }
        return index;
      } else {
        return args;
      }
    }
  };

  bool ShapeIsValid(const Array<PrimExpr>& shape) {
    // Zero-dimensional tensor does not need boundary check.
    if (!shape.size()) return false;

    for (size_t i = 0; i < shape.size(); ++i) {
      if (!shape[i].defined() || !shape[i].dtype().is_scalar() || is_negative_const(shape[i])) {
        return false;
      }
    }
    return true;
  }

  PrimExpr MakeBound(const DataType& type, const Array<PrimExpr>& shape) {
    // We have already checked the shape size to be greater then 0.
    PrimExpr bound = Mul(make_const(shape[0].dtype(), type.lanes()), shape[0]);
    for (size_t i = 1; i < shape.size(); ++i) {
      bound = Mul(bound, Mul(make_const(bound.dtype(), type.lanes()), shape[i]));
    }
    return bound;
  }

  // The buffer assignment map
  // Variable remap
  std::unordered_map<const VarNode*, PrimExpr> var_remap_;
  // Buffer map
  std::unordered_map<Buffer, BufferEntry, ObjectPtrHash, ObjectPtrEqual> buf_map_;
  // Collects shapes.
  std::vector<std::pair<Var, Array<PrimExpr>>> shape_collector_;
  // bounds populator. We really need the analyzer from it.
  // However
  IRVisitorWithAnalyzer* bound_analyzer_;
  // The size of cacheline
  int cache_line_size_;
  // Whether to mark load/store with theirs bounds.
  bool create_bound_attributes_{false};
};

// The specific tensor data layout is not determined before
// StorageFlatten pass. We use buffer_bind_scope
// to specify before hand we want to bind a subregion
// of tensor to a symbolic buffer, which get used in extern.
//
// Example:
//
// realize A in range [i*4, extent=10) {
//   bind Ab to A in [i*4+1, extent=4) {
//     call_func(Ab.ptr, Ab.shape[0])
//   }
// }
//
// After StorageFlatten
//
// alloc A[10]
//   call(A + 1,  4)
//
// Buffer is a protocol to declare specific
// data layout and shape we expect.
// So this function need to check:
// - If the bind range is within the realize range
// - If we can match the requirement of buffer
// - Remap variables such as Ab.ptr to the actual value.
//
// Here are a few possible failure cases:
// - Buffer is declared to have constant shape,
//   but we try to bind it to a different one.
// - Buffer is declared to be compact(no strides)
//   but this binded region is a subregion of
//   a matrix(tensor), which means it requires strides.
//
// We do support a few relaxed case, such as bindingx
// region with shape [1, 1, n, m] to buffer with shape [n, m]
PrimFunc StorageFlatten(PrimFunc func, int cache_line_size, bool create_bound_attributes) {
  // Only apply this pass to TIR from TE schedules
  Optional<Bool> from_legacy_te_schedule = func->GetAttr("from_legacy_te_schedule", Bool(false));
  if (from_legacy_te_schedule.value()) {
    auto fptr = func.CopyOnWrite();

    IRVisitorWithAnalyzer bound_analyzer;
    bound_analyzer(fptr->body);

    fptr->body = BufferShapeLegalize(fptr->buffer_map, &bound_analyzer)(std::move(fptr->body));

    auto stride_legalize = BufferStrideLegalize(fptr->buffer_map, &bound_analyzer);
    fptr->body = stride_legalize(std::move(fptr->body));
    fptr->buffer_map = stride_legalize.UpdatedExternBufferMap();

    fptr->body = ThreadScopePropagate(fptr->buffer_map)(std::move(fptr->body));

    fptr->body = BufferBindUnwrapper(fptr->buffer_map, &bound_analyzer)(std::move(fptr->body));

    fptr->body = StorageFlattener(fptr->buffer_map, cache_line_size, create_bound_attributes,
                                  &bound_analyzer)(std::move(fptr->body));

    return func;
  } else {
    return func;
  }
}

namespace transform {

// TODO(tvm-team): consolidate configs to the PassContext
Pass StorageFlatten(int cache_line_size, bool create_bound_attributes) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return StorageFlatten(std::move(f), cache_line_size, create_bound_attributes);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.StorageFlatten", {});
}

TVM_REGISTER_GLOBAL("tir.transform.StorageFlatten").set_body_typed(StorageFlatten);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
