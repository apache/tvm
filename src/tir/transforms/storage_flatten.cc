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
#include <unordered_set>

#include "../../arith/ir_visitor_with_analyzer.h"
#include "../../runtime/thread_storage_scope.h"
#include "arg_binder.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

using arith::IRVisitorWithAnalyzer;
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
  static transform::Pass Pass() {
    auto pass_func = [](PrimFunc func, IRModule m, transform::PassContext ctx) {
      IRVisitorWithAnalyzer bound_analyzer;

      bound_analyzer(func->body);

      auto pass = BufferShapeLegalize(func->buffer_map, &bound_analyzer);

      auto fptr = func.CopyOnWrite();
      fptr->body = pass(std::move(fptr->body));
      if (auto map = func->attrs.GetAttr<Map<Buffer, Array<IndexMap>>>("layout_transform_map")) {
        func = WithAttr(std::move(func), "layout_transform_map", pass.UpdateIndexMap(map.value()));
      }
      return func;
    };
    return transform::CreatePrimFuncPass(pass_func, 0, "tir.BufferShapeLegalize", {});
  }

  explicit BufferShapeLegalize(const Map<Var, Buffer>& extern_buffer_map,
                               IRVisitorWithAnalyzer* bound_analyzer)
      : bound_analyzer_(bound_analyzer) {
    for (auto kv : extern_buffer_map) {
      Buffer buf = kv.second;
      extern_buffers_.insert(buf);

      BufferEntry remap;
      remap.remap_to = buf;
      remap.index_offsets = Array<PrimExpr>(buf->shape.size(), 0);
      remap.in_scope = true;
      buf_map_[buf] = remap;
    }
  }

  Map<Buffer, Array<IndexMap>> UpdateIndexMap(const Map<Buffer, Array<IndexMap>>& orig) {
    Map<Buffer, Array<IndexMap>> output;
    for (const auto& kv : orig) {
      auto it = buf_map_.find(kv.first);
      if (it != buf_map_.end()) {
        output.Set(it->second.remap_to, kv.second);
      } else {
        output.Set(kv.first, kv.second);
      }
    }
    return output;
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_remap_.find(op);
    if (it != var_remap_.end()) {
      return it->second;
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    // BufferRealizeNode for an external buffer serves as an
    // annotation of the external buffers, and should not be changed.
    // Instead, verify that the bounds match the external
    // buffer.
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
    Array<PrimExpr> index_offsets;
    Array<Range> new_bounds;
    for (size_t i = 0; i < op->bounds.size(); i++) {
      const Range& bound = op->bounds[i];
      realized_shape.push_back(bound->extent);
      index_offsets.push_back(bound->min);
      new_bounds.push_back({0, bound->extent});
    }

    if (op->buffer->shape.size()) {
      ICHECK_EQ(op->buffer->shape.size(), realized_shape.size())
          << "Inconsistency between dimension of buffer " << op->buffer
          << " and dimension of its realized bounds.";
    }

    Buffer key = op->buffer;

    Buffer buf = op->buffer;
    auto write_ptr = buf.CopyOnWrite();
    write_ptr->shape = realized_shape;

    {
      BufferEntry remap;
      remap.remap_to = buf;
      remap.index_offsets = index_offsets;
      remap.in_scope = true;
      buf_map_[key] = remap;
    }

    Stmt stmt = BufferRealize(buf, new_bounds, op->condition, this->VisitStmt(op->body), op->span);

    buf_map_.at(key).in_scope = false;

    return stmt;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return VisitBufferAccess(std::move(node));
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return VisitBufferAccess(std::move(node));
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    auto it = buf_map_.find(node->buffer);
    if (it != buf_map_.end()) {
      const BufferEntry& entry = it->second;
      ICHECK(entry.in_scope) << "Cannot access an out-of-scope buffer";

      Array<PrimExpr> indices = node->indices;
      if (entry.index_offsets.size()) {
        ICHECK_GE(entry.index_offsets.size(), indices.size())
            << "Cannot bind buffer to a shape of lower dimension.";

        Array<PrimExpr> new_indices;

        // Pad leading indices with zero, matching the "fuzzy_match"
        // behavior from ArgBinder::BindBuffer.
        size_t diff = entry.index_offsets.size() - indices.size();
        for (size_t i = 0; i < diff; i++) {
          new_indices.push_back(0);
        }

        // Offset indices used to access buffers of a reduced size.
        for (size_t i = 0; i < indices.size(); i++) {
          PrimExpr offset = entry.index_offsets[i + diff];
          new_indices.push_back(indices[i] - offset);
        }
        indices = new_indices;
      }

      auto write_ptr = node.CopyOnWrite();
      write_ptr->indices = indices;
      write_ptr->buffer = entry.remap_to;
    }
    return node;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->node->IsInstance<tir::BufferNode>()) {
      // Visit body before checking internal_buf_map_, because we
      // don't know if the BufferNode needs to be changed until we
      // look in the body for a BufferRealizeNode with different
      // extents.
      Stmt body = this->VisitStmt(op->body);

      Buffer buffer = Downcast<tir::Buffer>(op->node);
      auto it = buf_map_.find(buffer);
      if (it != buf_map_.end()) {
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
  // Any buffers that give views into a resized buffer should be
  // updated, both to refer to the resized buffer and to have the view
  // window updated.  For example, suppose B1 is a 1-D buffer of size
  // 100 which is only realized on the range (10,50), and buffer V1 is
  // a view into B1[25:35].  When B1 is replaced with B2, a buffer of
  // size 40 realized on the range (0,40), V1 must be replaced to be a
  // view into B2[15:25].
  Stmt HandleBufferBindScope(const AttrStmtNode* op) {
    Array<ObjectRef> arr = Downcast<Array<ObjectRef>>(op->node);
    ICHECK_EQ(arr.size(), 2U);
    Buffer buffer = Downcast<Buffer>(arr[0]);
    ICHECK(buffer.defined());
    Buffer target = Downcast<Buffer>(arr[1]);
    ICHECK(target.defined());

    auto it = buf_map_.find(target);
    ICHECK(it != buf_map_.end()) << "attr::buffer_bind_scope target " << target << " not in scope.";
    const BufferEntry& target_remap = it->second;

    ICHECK(target_remap.in_scope) << "Cannot bind " << buffer->name
                                  << " to the out-of-scope buffer " << target_remap.remap_to->name;

    Call tuple = Downcast<Call>(op->value);
    ICHECK(tuple.defined() && tuple->op.same_as(builtin::tvm_tuple()));

    Array<PrimExpr> new_tuple_args;
    Array<PrimExpr> realized_begins;
    Array<PrimExpr> view_shape;
    ICHECK_EQ(tuple->args.size(), target_remap.index_offsets.size() * 2)
        << "attr::buffer_bind_scope to define " << buffer << " as a view into " << target
        << " does match dimensionality of " << target;
    for (size_t i = 0; i < target_remap.index_offsets.size(); i++) {
      PrimExpr parent_begin = tuple->args[2 * i];
      PrimExpr view_extent = tuple->args[2 * i + 1];
      // Offset the begin of the buffer view by the offset of the target buffer.
      new_tuple_args.push_back(parent_begin - target_remap.index_offsets[i]);
      // Keep the extent of the buffer view the same.
      new_tuple_args.push_back(view_extent);
      // Use the extent of the buffer view to define the buffer view's shape.
      view_shape.push_back(view_extent);
      // Within the buffer view, indices start at 0.
      realized_begins.push_back(0);
    }

    // If a view is binding to a buffer of a higher dimensionality,
    // then the leading dimensions should be padded out with shape of
    // 1.
    ICHECK_GE(view_shape.size(), buffer->shape.size())
        << "Cannot bind " << buffer << " to a shape of lower dimension.";
    if (view_shape.size() > buffer->shape.size()) {
      size_t diff = view_shape.size() - buffer->shape.size();
      Array<PrimExpr> padded_shape;
      for (size_t i = 0; i < diff; i++) {
        padded_shape.push_back(1);
      }
      for (auto dim : buffer->shape) {
        padded_shape.push_back(dim);
      }
      view_shape = std::move(padded_shape);
    }

    // If a buffer has strides defined, and is being remapped into a
    // shape with additional dimensions, then define dummy values for
    // the strides.
    Array<PrimExpr> realized_strides = buffer->strides;
    if ((realized_strides.size() > 0) && (realized_strides.size() != view_shape.size())) {
      ICHECK_GE(view_shape.size(), realized_strides.size())
          << "Cannot bind the strides of " << buffer << " to a shape of lower dimension";
      size_t diff = view_shape.size() - buffer->strides.size();

      Array<PrimExpr> updated_strides;
      for (size_t i = 0; i < diff; i++) {
        updated_strides.push_back(Var("stride", buffer->shape[0].dtype()));
      }
      for (auto stride : buffer->strides) {
        updated_strides.push_back(stride);
      }
      realized_strides = updated_strides;
    }

    Buffer key = buffer;

    auto write_ptr = buffer.CopyOnWrite();
    write_ptr->shape = view_shape;
    write_ptr->strides = realized_strides;

    {
      BufferEntry remap;
      remap.index_offsets = realized_begins;
      remap.remap_to = buffer;
      remap.in_scope = true;
      buf_map_[key] = remap;
    }

    // Define remappings of any Variables referencing Buffer internals
    // (e.g. Store/Load nodes).  Passing fuzzy_match=true allows the
    // remapped buffer to have a number of dimensions.
    ArgBinder binder(&var_remap_);
    binder.BindBuffer(key, buffer, key->name, true);

    Stmt body = this->VisitStmt(op->body);
    body = MergeNest(binder.asserts(), body);
    body = MergeNest(binder.init_nest(), body);

    Stmt stmt = AttrStmt(Array<ObjectRef>{buffer, target_remap.remap_to}, op->attr_key,
                         Call(tuple->dtype, tuple->op, new_tuple_args, tuple->span), body);

    for (const Var& v : binder.defs()) {
      var_remap_.erase(v.get());
    }

    buf_map_.at(key).in_scope = false;
    return stmt;
  }

  std::unordered_map<const VarNode*, PrimExpr> var_remap_;

  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> extern_buffers_;

  struct BufferEntry {
    Buffer remap_to;
    Array<PrimExpr> index_offsets;
    bool in_scope;
  };

  std::unordered_map<Buffer, BufferEntry, ObjectPtrHash, ObjectPtrEqual> buf_map_;

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
  static transform::Pass Pass() {
    auto pass_func = [](PrimFunc func, IRModule m, transform::PassContext ctx) {
      IRVisitorWithAnalyzer bound_analyzer;

      bound_analyzer(func->body);

      auto pass = BufferStrideLegalize(func->buffer_map, &bound_analyzer);

      auto fptr = func.CopyOnWrite();
      fptr->body = pass(std::move(fptr->body));
      fptr->buffer_map = pass.UpdatedExternBufferMap();
      if (auto map = func->attrs.GetAttr<Map<Buffer, Array<IndexMap>>>("layout_transform_map")) {
        func = WithAttr(std::move(func), "layout_transform_map", pass.UpdateIndexMap(map.value()));
      }
      return func;
    };
    return transform::CreatePrimFuncPass(pass_func, 0, "tir.BufferStrideLegalize", {});
  }

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
        buf_map_[buf] = entry;
      }
      updated_extern_buffer_map_.Set(kv.first, with_strides);
    }
  }

  Map<Buffer, Array<IndexMap>> UpdateIndexMap(const Map<Buffer, Array<IndexMap>>& orig) {
    Map<Buffer, Array<IndexMap>> output;
    for (const auto& kv : orig) {
      auto it = buf_map_.find(kv.first);
      if (it != buf_map_.end()) {
        output.Set(it->second.remap_to, kv.second);
      } else {
        output.Set(kv.first, kv.second);
      }
    }
    return output;
  }

  Map<Var, Buffer> UpdatedExternBufferMap() const { return updated_extern_buffer_map_; }

  Buffer WithStrides(Buffer buf) {
    auto cache_key = buf;

    auto it = buf_map_.find(cache_key);
    if (it != buf_map_.end()) {
      const BufferEntry& entry = it->second;
      ICHECK(entry.in_scope) << "Cannot annotate an out-of-scope buffer";
      return entry.remap_to;
    }

    Array<PrimExpr> shape = buf->shape;

    if (buf->strides.size()) {
      ICHECK_EQ(buf->strides.size(), buf->shape.size())
          << "Buffer " << buf << " has inconsistent strides/shape.";
    } else if (dim_align_.count(buf) == 0) {
      // Keeping this to have matched behavior to previous version.
      // There are many parts of the codebase that assume that a
      // strided array cannot be compact.  For example,
      // ArgBinder::BindBuffer and tir.Specialize.  To avoid breaking
      // these, do not define the strides unless required for a
      // non-compact array.
    } else if (shape.size() == 0) {
      // Can't define the strides for a buffer without a known shape.
    } else {
      // With everything checked, can now define the updated strides
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

      buf.CopyOnWrite()->strides = Array<PrimExpr>(rstrides.rbegin(), rstrides.rend());
    }

    BufferEntry entry;
    entry.remap_to = buf;
    entry.in_scope = true;
    buf_map_[cache_key] = entry;

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

      Stmt body = this->VisitStmt(op->body);

      buf_map_[source].in_scope = false;

      return AttrStmt(Array<ObjectRef>{source_with_strides, target_with_strides}, op->attr_key,
                      op->value, body, op->span);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  // AllocateNodes may be present from tvm.tir.ir_builder.  This can
  // be simplified in the future by having AllocateNode hold a buffer,
  // rather than a buffer_var.
  Stmt VisitStmt_(const AllocateNode* op) final {
    buffer_var_defines_.insert(op->buffer_var.get());
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateConstNode* op) final {
    buffer_var_defines_.insert(op->buffer_var.get());
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    if (op->var.dtype().is_handle()) {
      buffer_var_defines_.insert(op->var.get());
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const LetNode* op) final {
    if (op->var.dtype().is_handle()) {
      buffer_var_defines_.insert(op->var.get());
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    Buffer key = op->buffer;
    Buffer with_strides = WithStrides(op->buffer);

    Stmt stmt = StmtExprMutator::VisitStmt_(op);

    buf_map_[key].in_scope = false;
    op = stmt.as<BufferRealizeNode>();
    ICHECK(op);

    return BufferRealize(with_strides, op->bounds, op->condition, op->body, op->span);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return VisitBufferAccess(std::move(node));
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return VisitBufferAccess(std::move(node));
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    auto it = buf_map_.find(node->buffer);
    ICHECK(it == buf_map_.end() || it->second.in_scope)
        << "Cannot access a buffer " << node->buffer->name << ", out of scope";

    auto with_strides = WithStrides(node->buffer);
    if (!with_strides.same_as(node->buffer)) {
      node.CopyOnWrite()->buffer = with_strides;
    }

    return node;
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
  };

  std::unordered_map<Buffer, BufferEntry, ObjectPtrHash, ObjectPtrEqual> buf_map_;

  // Set of vars that have occurred in an AllocateNode, but haven't
  // yet occurred in a BufferLoad/BufferStore.
  std::unordered_set<const VarNode*> buffer_var_defines_;

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
  static transform::Pass Pass() {
    auto pass_func = [](PrimFunc func, IRModule m, transform::PassContext ctx) {
      auto pass = ThreadScopePropagate(func->buffer_map);

      auto fptr = func.CopyOnWrite();
      fptr->body = pass(std::move(fptr->body));
      if (auto map = func->attrs.GetAttr<Map<Buffer, Array<IndexMap>>>("layout_transform_map")) {
        func = WithAttr(std::move(func), "layout_transform_map", pass.UpdateIndexMap(map.value()));
      }
      return func;
    };
    return transform::CreatePrimFuncPass(pass_func, 0, "tir.ThreadScopePropagate", {});
  }

  explicit ThreadScopePropagate(const Map<Var, Buffer>& extern_buffer_map) {
    // External buffers shouldn't be overwritten, even if they have a
    // BufferRealizeNode.
    for (auto kv : extern_buffer_map) {
      external_buffers_.insert(kv.second);
    }
  }

  Map<Buffer, Array<IndexMap>> UpdateIndexMap(const Map<Buffer, Array<IndexMap>>& orig) {
    Map<Buffer, Array<IndexMap>> output;
    for (const auto& kv : orig) {
      auto it = buf_remap_.find(kv.first->data);
      if (it != buf_remap_.end()) {
        output.Set(it->second, kv.second);
      } else {
        output.Set(kv.first, kv.second);
      }
    }
    return output;
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
  // If the rewritten buffers are part of a buffer_bind_scope, either
  // as the buffer view or as the buffer being viewed, then the
  // buffer_bind_scope must be rewritten to refer to the updated
  // buffers.
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
  static transform::Pass Pass() {
    auto pass_func = [](PrimFunc func, IRModule m, transform::PassContext ctx) {
      IRVisitorWithAnalyzer bound_analyzer;

      bound_analyzer(func->body);

      auto pass = BufferBindUnwrapper(func->buffer_map, &bound_analyzer);

      auto fptr = func.CopyOnWrite();
      fptr->body = pass(std::move(fptr->body));
      return func;
    };
    return transform::CreatePrimFuncPass(pass_func, 0, "tir.BufferBindUnwrapper", {});
  }

  explicit BufferBindUnwrapper(const Map<Var, Buffer>& extern_buffer_map,
                               IRVisitorWithAnalyzer* bound_analyzer)
      : bound_analyzer_(bound_analyzer) {
    for (auto kv : extern_buffer_map) {
      BufferEntry e;
      e.buffer = kv.second;
      e.external = true;
      var_to_buffer_[kv.second->data.get()] = kv.second;
      buf_map_[kv.second.get()] = std::move(e);
    }
  }

  Map<Buffer, Array<IndexMap>> UpdateIndexMap(const Map<Buffer, Array<IndexMap>>& orig) {
    Map<Buffer, Array<IndexMap>> output;
    for (const auto& kv : orig) {
      const BufferEntry& e = GetBufferEntry(kv.first);

      if (e.remap) {
        output.Set(e.remap->target, kv.second);
      } else {
        output.Set(kv.first, kv.second);
      }
    }
    return output;
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
    return Stmt();
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
    return PrimExpr();
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
    ICHECK(!illegal_vars_.count(op)) << "Variable " << op->name_hint << " is not well defined.  "
                                     << "(e.g. use of buffer.elem_offset for a non-flat buffer)";

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

  // AllocateNodes may be present from tvm.tir.ir_builder.  This can
  // be simplified in the future by having AllocateNode hold a buffer,
  // rather than a buffer_var.
  Stmt VisitStmt_(const AllocateNode* op) final {
    buffer_var_defines_.insert(op->buffer_var.get());
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocateConstNode* op) final {
    buffer_var_defines_.insert(op->buffer_var.get());
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    if (op->var.dtype().is_handle()) {
      buffer_var_defines_.insert(op->var.get());
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const LetNode* op) final {
    if (op->var.dtype().is_handle()) {
      buffer_var_defines_.insert(op->var.get());
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();

    const BufferEntry& e = GetBufferEntry(op->buffer);

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

    const BufferEntry& e = GetBufferEntry(op->buffer);

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
      var_to_buffer_[op->buffer->data.get()] = op->buffer;
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

    const BufferEntry& e = GetBufferEntry(op->buffer);

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
  // Read the mapping from a buffer view to the actual buffer.  This
  // allows all later BufferStore/BufferLoad nodes to reference the
  // actual buffer, rather than the buffer view.
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
    ICHECK(it != buf_map_.end()) << "Cannot define " << source << " as a view into " << remap.target
                                 << ", " << remap.target << " was not defined.";
    const BufferEntry& target_info = it->second;
    ICHECK(target_info.in_scope) << "Cannot define " << source << " as a view into " << remap.target
                                 << ", " << remap.target << " is out of scope.";
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

    ICHECK(target_info.remap == nullptr)
        << "buffer_bind_scope defines " << source << " as a view into " << remap.target
        << ", which is itself a buffer view.  "
        << "Indirect remapping not currently supported.";

    for (size_t i = 0; i < remap.begins.size(); i++) {
      remap.begins.Set(i, bound_analyzer_->Simplify(remap.begins[i]));
      remap.extents.Set(i, bound_analyzer_->Simplify(remap.extents[i]));
    }

    // Add a buffer remap entry
    {
      BufferEntry source_info;
      source_info.buffer = source;
      source_info.remap = std::make_unique<RemapInfo>(remap);

      var_to_buffer_[source->data.get()] = source;
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
          << "Cannot bind a compact buffer " << source << " to a strided buffer " << view
          << " with strides " << view->strides;
    } else {
      // Add explicit strides to the view, in order to bind to source.strides[i].
      view = view.MakeStrideView();
    }

    // Match integer bits of source->elem_offset and view->elem_offset
    // as is required by ArgBinder::Bind_
    if (view->elem_offset.defined() && source->elem_offset.dtype() != view->elem_offset.dtype()) {
      view.CopyOnWrite()->elem_offset = cast(source->elem_offset.dtype(), view->elem_offset);
    }

    // Bind any variables that reference the view (e.g. elem_offset,
    // strides, shape).  Pass fuzzy_match=false, because all shape
    // transformations should have been handled in
    // BufferShapeLegalize.
    binder.BindBuffer(source, view, source->name, false);
    if (auto* elem_offset_var = source->elem_offset.as<VarNode>()) {
      if (!view->elem_offset.defined()) {
        illegal_vars_.insert(elem_offset_var);
      }
    }

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
  };

  const BufferEntry& GetBufferEntry(Buffer buffer) {
    if (buf_map_.count(buffer.get())) {
      const BufferEntry& e = buf_map_[buffer.get()];
      ICHECK(e.in_scope) << "Cannot access a buffer " << buffer->name << ", out of scope";
      return e;
    } else if (buffer_var_defines_.count(buffer->data.get())) {
      // The buffer var was defined, but the buffer hasn't been seen
      // before.
      BufferEntry entry;
      entry.buffer = buffer;
      var_to_buffer_[buffer->data.get()] = buffer;
      buf_map_[buffer.get()] = std::move(entry);
      return buf_map_[buffer.get()];
    } else if (var_remap_.count(buffer->data.get())) {
      // The buffer var is an alias of a bound buffer.  Only
      // supported if the bound buffer has no offsets.  In this
      // case, we just need to make a new aliasing buffer that
      // shares the remapped data variable.
      Var old_var = buffer->data;
      Var new_var = Downcast<Var>(var_remap_[old_var.get()]);

      {
        ICHECK(var_to_buffer_.count(old_var.get()))
            << "Cannot find remap information for aliased buffer var " << old_var->name_hint
            << ", required to verify this alias is legal.";
        const Buffer& aliased_buffer = var_to_buffer_[old_var.get()];
        const BufferEntry& entry = buf_map_[aliased_buffer.get()];
        if (entry.remap) {
          for (const auto& begin : entry.remap->begins) {
            ICHECK(is_zero(begin)) << "Aliasing of buffer with offset is not supported";
          }
        }
      }

      {
        Buffer new_buf = buffer;
        new_buf.CopyOnWrite()->data = new_var;

        RemapInfo remap_info;
        remap_info.target = new_buf;
        remap_info.begins = Array<PrimExpr>(buffer->shape.size(), 0);
        remap_info.extents = buffer->shape;

        BufferEntry entry;
        entry.buffer = buffer;
        entry.remap = std::make_unique<RemapInfo>(remap_info);
        entry.in_scope = true;
        var_to_buffer_[buffer->data.get()] = buffer;
        buf_map_[buffer.get()] = std::move(entry);
      }
      return buf_map_[buffer.get()];
    } else if (var_to_buffer_.count(buffer->data.get())) {
      // This buffer is an alias of a known buffer, with no remaps.  A
      // buffer entry should be generated and returned.
      BufferEntry entry;
      entry.buffer = buffer;
      entry.in_scope = true;
      var_to_buffer_[buffer->data.get()] = buffer;
      buf_map_[buffer.get()] = std::move(entry);

      return buf_map_[buffer.get()];
    } else {
      LOG(FATAL) << "Can't work around the undefined buffer";
      return *static_cast<BufferEntry*>(nullptr);
    }
  }

  // The buffer assignment map
  // Variable remap
  std::unordered_map<const VarNode*, PrimExpr> var_remap_;
  // Variables that may not occur within the body.
  std::unordered_set<const VarNode*> illegal_vars_;
  // Buffer map
  std::unordered_map<const BufferNode*, BufferEntry> buf_map_;
  // Map from Var to the Buffer they occurred in.  In case of aliased
  // buffers, contains the first buffer.
  std::unordered_map<const VarNode*, Buffer> var_to_buffer_;
  // Set of vars that have occurred in an AllocateNode, but haven't
  // yet occurred in a BufferLoad/BufferStore.
  std::unordered_set<const VarNode*> buffer_var_defines_;
  // Analyzer for the variable bounds, used to simplify the bounds populator. We really need the
  // analyzer from it. However
  IRVisitorWithAnalyzer* bound_analyzer_;
};

class ApplyLayoutTransforms : public StmtExprMutator {
 public:
  static transform::Pass Pass() {
    auto pass_func = [](PrimFunc func, IRModule m, transform::PassContext ctx) {
      auto lookup = func->attrs.GetAttr<Map<Buffer, Array<IndexMap>>>("layout_transform_map");

      if (!lookup) {
        return func;
      }

      Map<Buffer, Array<IndexMap>> layout_transforms = lookup.value();

      auto fptr = func.CopyOnWrite();

      auto mutator = ApplyLayoutTransforms(layout_transforms);
      fptr->buffer_map = mutator.UpdateExternBufferMap(fptr->buffer_map);
      fptr->body = mutator(std::move(fptr->body));

      return WithoutAttr(std::move(func), "layout_transform_map");
    };
    return transform::CreatePrimFuncPass(pass_func, 0, "tir.ApplyLayoutTransforms", {});
  }

  explicit ApplyLayoutTransforms(Map<Buffer, Array<IndexMap>> layout_transforms)
      : layout_transforms_(layout_transforms) {}

  Map<tir::Var, Buffer> UpdateExternBufferMap(const Map<tir::Var, Buffer>& buffer_map) {
    Map<tir::Var, Buffer> output;
    for (const auto& kv : buffer_map) {
      output.Set(kv.first, GetBufferRemap(kv.second, true));
    }
    return output;
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    // Call once so that load/store nodes can read from the cached
    // value.
    GetBufferRemap(op->buffer, true);

    auto realize = Downcast<BufferRealize>(StmtExprMutator::VisitStmt_(op));

    auto lookup = layout_transforms_.Get(op->buffer);
    if (lookup) {
      auto write_ptr = realize.CopyOnWrite();
      write_ptr->buffer = GetBufferRemap(op->buffer, true);

      Array<IndexMap> transforms = lookup.value();
      for (const auto& transform : transforms) {
        write_ptr->bounds = transform->MapRanges(realize->bounds);
      }
    }

    return std::move(realize);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return VisitBufferAccess(std::move(node));
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return VisitBufferAccess(std::move(node));
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    auto lookup = layout_transforms_.Get(node->buffer);
    if (lookup) {
      auto write_ptr = node.CopyOnWrite();

      write_ptr->buffer = GetBufferRemap(node->buffer);

      Array<IndexMap> transforms = lookup.value();
      for (const auto& transform : transforms) {
        write_ptr->indices = transform->MapIndices(node->indices);
      }
    }
    return node;
  }

 private:
  //! \brief Given a buffer, return the buffer it should be remapped into.
  Buffer GetBufferRemap(Buffer buf, bool allow_alloc = false) {
    auto key = buf.get();
    auto it = buf_map_.find(key);
    if (it != buf_map_.end()) {
      return it->second;
    }

    ICHECK(allow_alloc) << "Buffer " << buf << " accessed before declaration.";

    auto lookup = layout_transforms_.Get(buf);
    if (lookup) {
      Array<IndexMap> transforms = lookup.value();

      auto write_ptr = buf.CopyOnWrite();
      for (const auto& transform : transforms) {
        write_ptr->shape = transform->MapShape(buf->shape);
      }
    }

    buf_map_[key] = buf;
    return buf;
  }

  std::unordered_map<const BufferNode*, Buffer> buf_map_;

  Map<Buffer, Array<IndexMap>> layout_transforms_;
};

class StorageFlattener : public StmtExprMutator {
 public:
  static transform::Pass Pass(int cache_line_size, bool create_bound_attributes) {
    auto pass_func = [=](PrimFunc func, IRModule m, transform::PassContext ctx) {
      IRVisitorWithAnalyzer bound_analyzer;

      bound_analyzer(func->body);

      auto pass = StorageFlattener(func->buffer_map, cache_line_size, create_bound_attributes,
                                   &bound_analyzer);

      auto fptr = func.CopyOnWrite();
      fptr->body = pass(std::move(fptr->body));
      // The buffers in func->buffer_map are deliberately left
      // unflattened, as they are used for validation of user-provided
      // arguments.  The flattened buffers used in the updated
      // function body alias the argument buffers.
      return func;
    };
    return transform::CreatePrimFuncPass(pass_func, 0, "tir.StorageFlattener", {});
  }

  explicit StorageFlattener(const Map<Var, Buffer>& extern_buffer_map, int cache_line_size,
                            bool create_bound_attributes, IRVisitorWithAnalyzer* bound_analyzer)
      : bound_analyzer_(bound_analyzer), create_bound_attributes_(create_bound_attributes) {
    for (auto kv : extern_buffer_map) {
      BufferEntry e;
      e.buffer = kv.second;
      e.flattened_buffer = e.buffer.GetFlattenedBuffer();
      // TODO(Lunderberg): Move the handling of boolean into a
      // dedicated pass.

      // Boolean tensors are backed by a Int8 array.
      if (e.buffer->dtype == DataType::Bool()) {
        {
          auto writer = e.buffer.CopyOnWrite();
          writer->dtype = DataType::Int(8);
        }
        {
          auto writer = e.flattened_buffer.CopyOnWrite();
          writer->dtype = DataType::Int(8);
        }
      }
      e.external = true;
      buffer_var_defines_.insert(kv.second->data.get());
      buf_map_[kv.second] = e;
    }
    cache_line_size_ = cache_line_size;
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
    return Stmt();
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
    return PrimExpr();
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
      const auto& entry = GetBufferEntry(buffer);
      body = AttrStmt(entry.flattened_buffer->data, op->attr_key, op->value, std::move(body));
      return body;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (create_bound_attributes_) shape_collector_.clear();
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();

    const BufferEntry& e = GetBufferEntry(op->buffer);

    // Handle casts from the value's dtype to the dtype of the backing
    // array.
    PrimExpr value = op->value;
    if (value.dtype() == DataType::Bool()) {
      ICHECK_EQ(e.flattened_buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor, but received "
          << e.flattened_buffer->dtype;
      value = tir::Cast(DataType::Int(8), value);
    }

    auto flattened_indices = e.buffer->ElemOffset(op->indices);

    Stmt body = BufferStore(e.flattened_buffer, value, flattened_indices, op->span);
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

  // AllocateNodes may be present from tvm.tir.ir_builder.  This can
  // be simplified in the future by having AllocateNode hold a buffer,
  // rather than a buffer_var.
  Stmt VisitStmt_(const AllocateNode* op) final {
    buffer_var_defines_.insert(op->buffer_var.get());
    auto stmt = Downcast<Allocate>(StmtExprMutator::VisitStmt_(op));
    return Allocate(stmt->buffer_var, stmt->dtype, FlattenExtents(stmt), stmt->condition,
                    stmt->body, stmt->annotations, stmt->span);
  }

  Stmt VisitStmt_(const AllocateConstNode* op) final {
    buffer_var_defines_.insert(op->buffer_var.get());
    auto stmt = Downcast<AllocateConst>(StmtExprMutator::VisitStmt_(op));
    ObjectRef data_or_idx;
    if (stmt->data) {
      data_or_idx = stmt->data.value();
    } else if (stmt->irmod_storage_idx) {
      data_or_idx = stmt->irmod_storage_idx.value();
    } else {
      LOG(FATAL) << "Neither data array nor data index specified for allocation of const "
                 << op->buffer_var->name_hint;
    }
    return AllocateConst(stmt->buffer_var, stmt->dtype, FlattenExtents(stmt), data_or_idx,
                         stmt->body, stmt->annotations, stmt->span);
  }

  Stmt VisitStmt_(const LetStmtNode* op) final {
    if (op->var.dtype().is_handle()) {
      buffer_var_defines_.insert(op->var.get());
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const LetNode* op) final {
    if (op->var.dtype().is_handle()) {
      buffer_var_defines_.insert(op->var.get());
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    const auto& key = op->buffer;

    if (buf_map_.count(key)) {
      ICHECK(buf_map_.at(key).external)
          << "BufferRealize for internal buffer " << op->buffer << " appears multiple times.";
      return this->VisitStmt(op->body);
    } else {
      // create a buffer entry
      BufferEntry e;

      ICHECK_EQ(op->buffer->shape.size(), op->bounds.size())
          << "Inconsistent buffer shape and realization shape for " << op->buffer;

      for (size_t i = 0; i < op->bounds.size(); i++) {
        const auto& bound = op->bounds[i];
        const auto& dim_size = op->buffer->shape[i];
        ICHECK(is_zero(bound_analyzer_->Simplify(bound->min)))
            << "Buffer " << op->buffer << " has realization bounds that do not start at zero.  "
            << "Please run BufferShapeLegalize first.";
        ICHECK(is_one(bound_analyzer_->Simplify(bound->extent == dim_size)))
            << "Buffer " << op->buffer
            << " has realization extent that does not match its size.  "
               "Please run BufferShapeLegalize first.";
      }

      StorageScope skey = StorageScope::Create(GetPtrStorageScope(op->buffer->data));

      // use small alignment for small arrays
      auto dtype = op->buffer->dtype;
      size_t const_size = AllocateNode::ConstantAllocationSize(op->buffer->shape);
      int align = GetTempAllocaAlignment(dtype, const_size);
      if (skey.tag.length() != 0) {
        MemoryInfo info = GetMemoryInfo(skey.to_string());
        if (info.defined()) {
          align = (info->max_simd_bits + dtype.bits() - 1) / dtype.bits();
          ICHECK_LE(const_size * dtype.bits(), info->max_num_bits)
              << "Allocation exceed bound of memory tag " << skey.to_string();
        }
      }

      e.buffer = Buffer(op->buffer->data, op->buffer->dtype, op->buffer->shape, op->buffer->strides,
                        PrimExpr(), op->buffer->name, align, 0, kDefault,
                        op->buffer->axis_separators, op->buffer->span);
      e.flattened_buffer = e.buffer.GetFlattenedBuffer();

      // TODO(Lunderberg): Move the handling of boolean into a
      // dedicated pass.

      // Boolean tensors are backed by a Int8 array.
      if (e.flattened_buffer->dtype == DataType::Bool()) {
        auto writer = e.flattened_buffer.CopyOnWrite();
        writer->dtype = DataType::Int(8);
      }

      buffer_var_defines_.insert(op->buffer->data.get());
      buf_map_[key] = e;
      Stmt body = this->VisitStmt(op->body);
      buffer_var_defines_.erase(op->buffer->data.get());
      buf_map_[key].in_scope = false;

      Stmt ret =
          Allocate(e.flattened_buffer->data, e.flattened_buffer->dtype, e.flattened_buffer->shape,
                   make_const(DataType::Bool(e.flattened_buffer->dtype.lanes()), true), body);

      if (create_bound_attributes_ && ShapeIsValid(e.buffer->shape)) {
        ret = AttrStmt(e.buffer->data, tir::attr::buffer_bound,
                       MakeBound(e.buffer->dtype, e.buffer->shape), ret);
      }
      return ret;
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

    const BufferEntry& e = GetBufferEntry(op->buffer);

    if (create_bound_attributes_ && ShapeIsValid(e.buffer->shape)) {
      shape_collector_.push_back(std::make_pair(e.buffer->data, e.buffer->shape));
    }

    auto flattened_indices = e.buffer->ElemOffset(op->indices);
    PrimExpr val = BufferLoad(e.flattened_buffer, flattened_indices, op->span);

    if (op->dtype == DataType::Bool()) {
      ICHECK_EQ(e.flattened_buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor, but received "
          << e.flattened_buffer->dtype;
      val = tir::Cast(DataType::Bool(), val);
    }

    return val;
  }

  Stmt VisitStmt_(const PrefetchNode* op) final {
    const BufferEntry& e = GetBufferEntry(op->buffer);

    ICHECK(e.in_scope) << "Cannot prefetch " << op->buffer << ", out of scope.";
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

    Stmt stmt = GetRef<Stmt>(op);
    for (int i = starts; i >= 0; --i) {
      if (i < starts) {
        stmt = For(vars[i], 0, op->bounds[i]->extent, ForKind::kSerial, stmt);
      } else {
        PrimExpr load = e.buffer.vload(args, e.buffer->dtype);
        PrimExpr address = Call(DataType::Handle(), builtin::address_of(), {load});
        PrimExpr prefetch = Call(op->buffer->dtype, builtin::prefetch(), {address, 0, 3, 1});
        stmt = Evaluate(prefetch);
        PrimExpr extent = (op->bounds[i]->extent - 1) / stride + 1;
        stmt = For(vars[i], 0, extent, ForKind::kSerial, stmt);
      }
    }
    return this->VisitStmt(stmt);
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
  // Helper function for visiting Allocate and AllocateConst.  If, in
  // the future, these are updated to hold a buffer (Buffer) object
  // rather than a buffer_var (Var), this function can be replaced
  // with a call to GetBufferEntry.
  template <typename Node>
  Array<PrimExpr> FlattenExtents(const Node& node) {
    arith::Analyzer analyzer;

    // If an allocation has extents that match the buffer
    auto is_compatible_buffer = [&](const Buffer& buffer) {
      if (buffer->shape.size() != node->extents.size()) {
        return false;
      }
      for (size_t i = 0; i < buffer->shape.size(); i++) {
        if (!analyzer.CanProveEqual(buffer->shape[i], node->extents[i])) {
          return false;
        }
      }

      return true;
    };

    auto int_array_equal = [](const Array<IntImm>& a, const Array<IntImm>& b) {
      if (a.size() != b.size()) {
        return false;
      }

      for (size_t i = 0; i < a.size(); i++) {
        if (a[i]->value != b[i]->value) {
          return false;
        }
      }

      return true;
    };

    Array<IntImm> axis_separators;
    auto it = buffer_var_map_.find(node->buffer_var.get());
    if (it != buffer_var_map_.end()) {
      const auto& buffers = it->second;
      if (buffers.size() == 0) {
        // No buffers use this allocation, treat as flat and optimize
        // out later.
      } else if (buffers.size() == 1) {
        // Only one buffer uses this allocation, so use its axis
        // separators.
        axis_separators = buffers[0]->axis_separators;
      } else {
        // Try to find a buffer using this allocation with a matching
        // shape.
        Buffer compatible_buffer;
        for (const auto& buffer : buffers) {
          if (is_compatible_buffer(buffer)) {
            ICHECK(!compatible_buffer.defined() ||
                   int_array_equal(compatible_buffer->axis_separators, buffer->axis_separators))
                << "Cannot determine axis separators to use when flattening "
                << node->buffer_var->name_hint
                << ", multiple buffer objects found with conflicting axis separators";
            compatible_buffer = buffer;
          }
        }
        ICHECK(compatible_buffer.defined())
            << "Cannot determine axis separators to use when flattening "
            << node->buffer_var->name_hint << ", no buffers found with matching shape";
        axis_separators = compatible_buffer->axis_separators;
      }
    }

    // Use GetFlattenedBuffer to determine the flattened shape of the
    // output.  We only need the shape and axis separators defined,
    // everything else can be dummy values.
    Buffer dummy_buffer =
        decl_buffer(node->extents, DataType::Float(32), "buffer", "", axis_separators);
    return dummy_buffer.GetFlattenedBuffer()->shape;
  }

  // The buffer entry in the flatten map
  struct DimAlignInfo {
    int align_factor{0};
    int align_offset{0};
  };
  // The buffer entry in the flatten map
  struct BufferEntry {
    // The buffer object
    Buffer buffer;
    // The updated buffer object, after flattening has been applied.
    Buffer flattened_buffer;
    // Whether the buffer is external
    bool external{false};
    // Whether the buffer is currently in scope.
    bool in_scope{true};
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
    Array<PrimExpr> bounds{bound};

    return Call(DataType::Handle(), builtin::tvm_tuple(), bounds);
  }

  const BufferEntry& GetBufferEntry(Buffer buffer) {
    auto alloc_key = buffer->data.get();
    if (!buf_map_.count(buffer) && buffer_var_defines_.count(alloc_key)) {
      BufferEntry entry;
      entry.buffer = buffer;
      entry.flattened_buffer = buffer.GetFlattenedBuffer();
      // Boolean tensors are backed by a Int8 array.
      if (entry.flattened_buffer->dtype == DataType::Bool()) {
        auto writer = entry.flattened_buffer.CopyOnWrite();
        writer->dtype = DataType::Int(8);
      }
      buf_map_[buffer] = std::move(entry);
    }

    auto it = buf_map_.find(buffer);
    ICHECK(it != buf_map_.end()) << "Cannot find allocated buffer for " << buffer;
    const BufferEntry& e = it->second;
    ICHECK(e.in_scope) << "Cannot access a buffer " << buffer->name << ", out of scope";
    return it->second;
  }

  // The buffer assignment map
  // Variable remap
  std::unordered_map<const VarNode*, PrimExpr> var_remap_;
  // Set of vars that have occurred in an AllocateNode, but haven't
  // yet occurred in a BufferLoad/BufferStore.
  std::unordered_set<const VarNode*> buffer_var_defines_;
  // Map from an allocation variable to the buffer(s) that it backs.
  // Used to track the determine the axis_separators that should be
  // used for flattening the extents of an AllocateNode.
  std::unordered_map<const VarNode*, std::vector<Buffer>> buffer_var_map_;
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

/*!
 * \brief Simplify assert statements.
 *
 * If an assert statement can be statically verified to be true,
 * remove the assert statement.  Otherwise, keep the assert statement
 * unmodified.
 */
class AssertSimplifier : public StmtMutator {
 public:
  static transform::Pass Pass() {
    auto pass_func = [=](PrimFunc func, IRModule m, transform::PassContext ctx) {
      IRVisitorWithAnalyzer bound_analyzer;

      bound_analyzer(func->body);

      auto fptr = func.CopyOnWrite();
      fptr->body = AssertSimplifier(&bound_analyzer)(std::move(fptr->body));
      return func;
    };
    return transform::CreatePrimFuncPass(pass_func, 0, "tir.AssertSimplifier", {});
  }

  explicit AssertSimplifier(IRVisitorWithAnalyzer* bound_analyzer)
      : bound_analyzer_(bound_analyzer) {}

  Stmt VisitStmt_(const AssertStmtNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<AssertStmtNode>();

    PrimExpr condition = bound_analyzer_->Simplify(op->condition);
    if (is_one(condition)) {
      return op->body;
    }

    return stmt;
  }

 private:
  IRVisitorWithAnalyzer* bound_analyzer_;
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
// We do support a few relaxed case, such as binding a
// region with shape [1, 1, n, m] to buffer with shape [n, m]
PrimFunc StorageFlatten(PrimFunc func, int cache_line_size, bool create_bound_attributes) {
  // Only apply this pass to TIR from TE schedules.  Because this is a
  // per-function attribute, we can't just check it once for the
  // entire module and apply the Sequential transform.
  Optional<Bool> from_legacy_te_schedule = func->GetAttr("from_legacy_te_schedule", Bool(false));
  if (from_legacy_te_schedule.value()) {
    auto seq = transform::Sequential(
        {
            BufferShapeLegalize::Pass(),
            BufferStrideLegalize::Pass(),
            ThreadScopePropagate::Pass(),
            BufferBindUnwrapper::Pass(),
            ApplyLayoutTransforms::Pass(),
            StorageFlattener::Pass(cache_line_size, create_bound_attributes),
            AssertSimplifier::Pass(),
        },
        "tir.StorageFlatten_impl");
    GlobalVar dummy_func_name("dummy_func");
    IRModule mod(Map<GlobalVar, BaseFunc>({{dummy_func_name, func}}));
    mod = seq(mod);
    return Downcast<PrimFunc>(mod->Lookup(dummy_func_name));
  } else {
    return func;
  }
}

namespace transform {

TVM_REGISTER_GLOBAL("tir.transform.ApplyLayoutTransforms")
    .set_body_typed(ApplyLayoutTransforms::Pass);

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
