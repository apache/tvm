/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file compact_buffer_region.cc
 * \brief Compact the buffer size into its exact need.
 */

#include <tvm/arith/int_set.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <stack>

#include "../../support/arena.h"
#include "../../support/nd_int_set.h"
#include "../../support/utils.h"
#include "../schedule/utils.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

using support::NDIntSet;

/*!
 * \brief return the region collected by NDIntSet. return the oroginal buffer shape if the
 *        int_set is empty.
 */
Region NarrowBufferRegionFromNDIntSet(const NDIntSet& nd_int_set,
                                      const Array<PrimExpr>& original_shape) {
  Array<Range> result;
  result.reserve(nd_int_set.size());
  for (size_t i = 0; i < nd_int_set.size(); ++i) {
    const arith::IntSet& int_set = nd_int_set[i];
    result.push_back(int_set.CoverRange(Range(/*begin=*/0, /*end=*/original_shape[i])));
  }
  return result;
}

/*!
 * \brief Collect the access region of each buffer.
 * \note The param buffer regions will not be collected.
 */
class BufferAccessRegionCollector : public StmtExprVisitor {
 public:
  static std::unordered_map<Buffer, Region, ObjectPtrHash, ObjectPtrEqual> Collect(
      const PrimFunc& f) {
    BufferAccessRegionCollector collector;
    collector(f->body);
    return std::move(collector.buffer_access_region_);
  }

 private:
  struct BufferAccessInfo {
    /*! \brief The buffer. */
    Buffer buffer;
    /*! \brief The buffer access region, which can be updated during visiting. */
    NDIntSet accessed_region;

    explicit BufferAccessInfo(const Buffer& buffer, const NDIntSet& region)
        : buffer(buffer), accessed_region(region) {}
  };

  BufferAccessRegionCollector() = default;

  /**************** Visitor overload ****************/

  void VisitStmt_(const BufferStoreNode* op) final {
    VisitBufferAccess(BufferRegion::FromPoint(op->buffer, op->indices));
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    VisitBufferAccess(BufferRegion::FromPoint(op->buffer, op->indices));
  }

  void VisitExpr_(const VarNode* op) final { VisitBufferVar(GetRef<Var>(op)); }

  void VisitExpr_(const LoadNode* op) final {
    StmtExprVisitor::VisitExpr_(op);
    VisitBufferVar(op->buffer_var);
  }

  void VisitStmt_(const StoreNode* op) final {
    StmtExprVisitor::VisitStmt_(op);
    VisitBufferVar(op->buffer_var);
  }

  void VisitStmt_(const ForNode* op) final {
    ancestor_loops_.push_back(op);
    StmtExprVisitor::VisitStmt_(op);
    ancestor_loops_.pop_back();
    // The iter_dom_map is updated by post DFS order.
    // If the union point is under the for node, the loop var will not be relaxed.
    // If the union point is outer of the for loop, the loop var should be relaxed.
    iter_dom_map_on_post_order_[op->loop_var.get()] =
        arith::IntSet::FromMinExtent(op->min, op->extent);
  }

  void VisitStmt_(const BlockNode* op) final {
    // Step 0. Check there is no init part.
    ICHECK(!op->init.defined());
    // Step 1. Update outer buffer access info using buffer region
    for (const BufferRegion& region : op->reads) {
      VisitBufferAccess(region);
    }
    for (const BufferRegion& region : op->writes) {
      VisitBufferAccess(region);
    }

    // Step 2. Update inner buffer
    // Step 2.1. rebuild map buffer_var_in_scope
    std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_var_in_scope;
    for (const Buffer& buffer : op->alloc_buffers) {
      buffer_var_in_scope.emplace(buffer->data, buffer);
    }
    // Step 2.2 Record top stack element before recursive visiting.
    size_t stack_top = buffer_access_stack_.size();

    // Step 2.3. Update the buffer_var_in_scope_ of visitor and visit recursively
    std::swap(buffer_var_in_scope, buffer_var_in_scope_);
    StmtExprVisitor::VisitStmt_(op);
    std::swap(buffer_var_in_scope, buffer_var_in_scope_);

    // Step 2.4. Combine and relax access
    std::unordered_map<Buffer, NDIntSet, ObjectPtrHash, ObjectPtrEqual> relaxed_region =
        CombineAndRelax(stack_top);

    // Step 2.5. Visit ancestor_loops and try to relax outer thread loops.
    for (const Buffer& buffer : op->alloc_buffers) {
      auto it = relaxed_region.find(buffer);
      ICHECK(it != relaxed_region.end());
      const NDIntSet& nd_int_set = it->second;
      std::unordered_map<const VarNode*, arith::IntSet> dom_map;
      for (const ForNode* loop : ancestor_loops_) {
        const VarNode* loop_var = loop->loop_var.get();
        if (NeedRelaxThread(GetRef<For>(loop), runtime::StorageScope::Create(buffer.scope()))) {
          dom_map[loop_var] = arith::IntSet::FromMinExtent(loop->min, loop->extent);
        }
      }
      NDIntSet int_set = support::NDIntSetEval(nd_int_set, dom_map);
      buffer_access_region_[buffer] = NarrowBufferRegionFromNDIntSet(int_set, buffer->shape);
    }
  }

  /**************** Helper functions ****************/

  void VisitBufferAccess(const BufferRegion& buffer_region) {
    const BufferNode* buffer = buffer_region->buffer.get();
    auto it = buffer_var_in_scope_.find(buffer->data);
    if (it != buffer_var_in_scope_.end()) {
      const Buffer& buffer = it->second;
      const BufferAccessInfo* info =
          arena_.make<BufferAccessInfo>(buffer, support::NDIntSetFromRegion(buffer_region->region));
      buffer_access_stack_.push(info);
    }
  }

  void VisitBufferVar(const Var& var) {
    auto it = buffer_var_in_scope_.find(var);
    if (it != buffer_var_in_scope_.end()) {
      const Buffer& buffer = it->second;
      VisitBufferAccess(BufferRegion::FullRegion(buffer));
    }
  }

  /*!
   * \brief Combine buffer accesses in the sub-tree.
   * \details The access info is stored in a stack by DFS order, so that the accesses in the
   *          sub-tree are top-n elements in the stack.
   * \param stack_top compact the access information in `stack[stack_top:end]`.
   */
  std::unordered_map<Buffer, NDIntSet, ObjectPtrHash, ObjectPtrEqual> CombineAndRelax(
      size_t stack_top) {
    std::unordered_map<Buffer, NDIntSet, ObjectPtrHash, ObjectPtrEqual> accesses;
    while (buffer_access_stack_.size() > stack_top) {
      const BufferAccessInfo* info = buffer_access_stack_.top();
      buffer_access_stack_.pop();
      NDIntSet nd_int_set =
          support::NDIntSetEval(info->accessed_region, iter_dom_map_on_post_order_);
      auto it = accesses.find(info->buffer);
      if (it != accesses.end()) {
        support::NDIntSetUnionWith(&it->second, nd_int_set);
      } else {
        accesses[info->buffer] = nd_int_set;
      }
    }
    return accesses;
  }

  /*!
   * \brief Combine buffer accesses in the sub-tree and push the combined result into the stack.
   * \details The access info is stored in a stack by DFS order, so that the accesses in the
   *          sub-tree are top-n elements in the stack.
   * \param stack_top The top element of the stack before visiting the sub-tree.
   */
  std::unordered_map<Buffer, NDIntSet, ObjectPtrHash, ObjectPtrEqual> CombineRelaxAndPushStack(
      size_t stack_top) {
    std::unordered_map<Buffer, NDIntSet, ObjectPtrHash, ObjectPtrEqual> accesses =
        CombineAndRelax(stack_top);
    for (const auto& kv : accesses) {
      const Buffer& buffer = kv.first;
      const NDIntSet& int_set = kv.second;
      buffer_access_stack_.push(arena_.make<BufferAccessInfo>(buffer, int_set));
    }
    return accesses;
  }

  /*! \brief Check whether the thread binding loop should be relaxed with given storage scope. */
  static bool NeedRelaxThread(const For& loop, const runtime::StorageScope& scope) {
    if (loop->kind != ForKind::kThreadBinding) {
      return false;
    }
    ICHECK(loop->thread_binding.defined());
    const String& thread_tag = loop->thread_binding.value()->thread_tag;
    // When there is warp memory
    // threadIdx.x must be set to be warp index.
    return CanRelaxStorageUnderThread(scope, runtime::ThreadScope::Create(thread_tag));
  }

  /**************** Class members ****************/

  /*! \brief Buffer access in DFS order. */
  std::stack<const BufferAccessInfo*> buffer_access_stack_;
  /*! \brief The loops from the current node up to the root. */
  std::vector<const ForNode*> ancestor_loops_;
  /*! \brief The vars of the buffer allocated under the current block. */
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_var_in_scope_;
  /*! \brief The map from loop vars to their iter range. */
  std::unordered_map<const VarNode*, arith::IntSet> iter_dom_map_on_post_order_;
  /*! \brief The map from Buffer to it entire access region, used for returning. */
  std::unordered_map<Buffer, Region, ObjectPtrHash, ObjectPtrEqual> buffer_access_region_;
  /*! \brief Internal arena. */
  support::Arena arena_;
};

/*! \brief Collect storage alignment information from block annotations. */
class StorageAlignCollector : public StmtVisitor {
 public:
  static std::unordered_map<Buffer, StorageAlignAnnotation, ObjectPtrHash, ObjectPtrEqual> Collect(
      const PrimFunc& f) {
    StorageAlignCollector collector;
    collector(f->body);
    return std::move(collector.storage_align_);
  }

 private:
  void VisitStmt_(const BlockNode* op) final {
    auto it = op->annotations.find(attr::buffer_dim_align);
    if (it != op->annotations.end()) {
      auto storage_align_annotation = Downcast<StorageAlignAnnotation>((*it).second);
      for (const auto& storage_align_tuple : storage_align_annotation) {
        int buffer_index = storage_align_tuple[0]->value;
        const Buffer& buffer = op->writes[buffer_index]->buffer;
        storage_align_[buffer].push_back(storage_align_tuple);
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  /*! \brief The map from Buffer to its storage alignment information. */
  std::unordered_map<Buffer, StorageAlignAnnotation, ObjectPtrHash, ObjectPtrEqual> storage_align_;
};

/*! \brief Reallocate the buffers with minimal region. */
class BufferCompactor : public StmtExprMutator {
 public:
  static Stmt Compact(
      const PrimFunc& f,
      const std::unordered_map<Buffer, Region, ObjectPtrHash, ObjectPtrEqual>& regions,
      const std::unordered_map<Buffer, StorageAlignAnnotation, ObjectPtrHash, ObjectPtrEqual>&
          storage_align) {
    std::unordered_map<Buffer, BufferAllocInfo, ObjectPtrHash, ObjectPtrEqual> buffer_info;

    for (const auto& kv : regions) {
      const Buffer& buffer = kv.first;
      Region region = kv.second;
      BufferAllocInfo buffer_alloc_info(std::move(region));
      auto it = storage_align.find(buffer);
      if (it != storage_align.end()) {
        std::vector<DimAlignInfo> dim_aligns(buffer->shape.size());
        for (const StorageAlignTuple& dim_align : (*it).second) {
          ICHECK(dim_align.size() == 4);
          int dim = dim_align[1]->value;
          int factor = dim_align[2]->value;
          int offset = dim_align[3]->value;
          dim_aligns.at(dim) = {factor, offset};
        }
        buffer_alloc_info.dim_aligns = std::move(dim_aligns);
      }
      buffer_info.emplace(buffer, std::move(buffer_alloc_info));
    }
    BufferCompactor compactor(std::move(buffer_info));
    Stmt stmt = compactor(f->body);
    return stmt;
  }

 private:
  /*! \brief The storage alignment for a dimension */
  struct DimAlignInfo {
    /*! \brief The factor of the alignment */
    int align_factor{0};
    /*! \brief The offset of the alignment */
    int align_offset{0};
  };

  struct BufferAllocInfo {
    /*! \brief The buffer access region. */
    Region region;
    /*! \brief The storage alignment information. */
    std::vector<DimAlignInfo> dim_aligns;
    /*!
     * \brief The reallocated buffer with minimal size.
     * \note The value if NullOpt if the buffer do not need reallocate (e.g parameter buffer).
     */
    Buffer new_buffer;

    explicit BufferAllocInfo(Region region) : region(std::move(region)) {}
  };

  explicit BufferCompactor(
      std::unordered_map<Buffer, BufferAllocInfo, ObjectPtrHash, ObjectPtrEqual> buffer_info)
      : buffer_info_(std::move(buffer_info)) {}

  Stmt VisitStmt_(const BufferStoreNode* _op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_op));
    BufferStoreNode* op = store.CopyOnWrite();
    RewriteBufferAccess(&op->buffer, &op->indices);
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* _op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_op));
    BufferLoadNode* op = load.CopyOnWrite();
    RewriteBufferAccess(&op->buffer, &op->indices);
    return std::move(load);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    // Step 0. Check there is no Init part.
    ICHECK(!op->init.defined());
    // Step 1. Reallocate and rewrite alloc_buffers, also update BufferAllocInfo.
    Array<Buffer> alloc_buffers = RewriteAllocBuffer(op->alloc_buffers);
    // Step 2. Recursively rewrite BufferLoad/BufferStore.
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    // Step 3. Update block signature.
    BlockNode* n = block.CopyOnWrite();
    RewriteBufferRegions(&n->reads);
    RewriteBufferRegions(&n->writes);
    RewriteMatchBuffers(&n->match_buffers);
    n->alloc_buffers = std::move(alloc_buffers);
    return std::move(block);
  }

  Array<Buffer> RewriteAllocBuffer(const Array<Buffer>& buffers) {
    Array<Buffer> result;
    result.reserve(buffers.size());
    for (const Buffer& buffer : buffers) {
      auto it = buffer_info_.find(buffer);
      ICHECK(it != buffer_info_.end());
      BufferAllocInfo& info = it->second;
      Array<PrimExpr> shape;
      shape.reserve(info.region.size());
      for (const Range& range : info.region) {
        shape.push_back(range->extent);
      }
      Array<PrimExpr> strides;
      if (info.dim_aligns.size()) {
        ICHECK(info.dim_aligns.size() == shape.size());
        strides.resize(shape.size());
        PrimExpr stride = make_const(shape[0].dtype(), 1);
        for (size_t i = shape.size(); i != 0; --i) {
          size_t dim = i - 1;
          if (info.dim_aligns[dim].align_factor != 0) {
            PrimExpr factor = make_const(stride.dtype(), info.dim_aligns[dim].align_factor);
            PrimExpr offset = make_const(stride.dtype(), info.dim_aligns[dim].align_offset);
            stride = stride + indexmod(factor + offset - indexmod(stride, factor), factor);
          }
          strides.Set(dim, stride);
          stride = stride * shape[dim];
        }
      }
      ObjectPtr<BufferNode> n = make_object<BufferNode>(*buffer.get());
      n->shape = std::move(shape);
      n->strides = std::move(strides);
      info.new_buffer = Buffer(std::move(n));
      result.push_back(info.new_buffer);
    }
    return result;
  }

  void RewriteBufferAccess(Buffer* buffer, Array<PrimExpr>* indices) const {
    auto it = buffer_info_.find(*buffer);
    if (it == buffer_info_.end()) {
      // Skip if the buffer is parameter
      return;
    }
    const BufferAllocInfo& info = it->second;
    ICHECK_EQ(indices->size(), info.region.size());
    int ndim = info.region.size();
    Array<PrimExpr> new_indices;
    new_indices.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      new_indices.push_back((*indices)[i] - info.region[i]->min);
    }
    *buffer = info.new_buffer;
    *indices = std::move(new_indices);
  }

  void RewriteBufferRegion(Buffer* buffer, Region* region) const {
    auto it = buffer_info_.find(*buffer);
    if (it == buffer_info_.end()) {
      // Skip if the buffer is parameter
      return;
    }
    const BufferAllocInfo& info = it->second;
    ICHECK_EQ(region->size(), info.region.size());
    Region new_region;
    new_region.reserve(info.region.size());
    for (size_t i = 0; i < info.region.size(); ++i) {
      const Range& range = (*region)[i];
      new_region.push_back(Range::FromMinExtent(range->min - info.region[i]->min, range->extent));
    }
    *buffer = info.new_buffer;
    *region = std::move(new_region);
  }

  void RewriteBufferRegions(Array<BufferRegion>* regions) const {
    Array<BufferRegion> new_regions;
    new_regions.reserve(regions->size());
    for (const auto& region : *regions) {
      BufferRegion buffer_region = region;
      BufferRegionNode* p = buffer_region.CopyOnWrite();
      RewriteBufferRegion(&p->buffer, &p->region);
      new_regions.push_back(buffer_region);
    }
    *regions = std::move(new_regions);
  }

  void RewriteMatchBuffers(Array<MatchBufferRegion>* match_buffers) const {
    Array<MatchBufferRegion> result;
    result.reserve(match_buffers->size());
    for (const auto& match_buffer : *match_buffers) {
      const BufferRegion& buffer_region = match_buffer->source;
      auto p = make_object<BufferRegionNode>(*buffer_region.get());
      RewriteBufferRegion(&p->buffer, &p->region);
      result.push_back(MatchBufferRegion(match_buffer->buffer, BufferRegion(p)));
    }
    *match_buffers = std::move(result);
  }

  /*! \brief The allocation information about each buffer. */
  std::unordered_map<Buffer, BufferAllocInfo, ObjectPtrHash, ObjectPtrEqual> buffer_info_;
};

PrimFunc CompactBufferAllocation(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    std::unordered_map<Buffer, Region, ObjectPtrHash, ObjectPtrEqual> region =
        BufferAccessRegionCollector::Collect(f);
    std::unordered_map<Buffer, StorageAlignAnnotation, ObjectPtrHash, ObjectPtrEqual>
        storage_align = StorageAlignCollector::Collect(f);
    fptr->body = BufferCompactor::Compact(f, region, storage_align);
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass CompactBufferAllocation() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return CompactBufferAllocation(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.CompactBufferAllocation", {});
}

TVM_REGISTER_GLOBAL("tir.transform.CompactBufferAllocation")
    .set_body_typed(CompactBufferAllocation);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
