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
#include <tvm/arith/int_solver.h>
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
 * \brief simplify and return the region collected by NDIntSet. return the original
 * buffer shape if the int_set is empty.
 */
Region SimplifyAndNarrowBufferRegionFromNDIntSet(
    const NDIntSet& nd_int_set, const Array<PrimExpr>& original_shape, arith::Analyzer* analyzer,
    const std::vector<const ForNode*>& ancestor_loops) {
  Array<Range> result;
  result.reserve(nd_int_set.size());
  for (size_t i = 0; i < nd_int_set.size(); ++i) {
    const arith::IntSet& int_set = nd_int_set[i];
    Range range = int_set.CoverRange(Range(/*begin=*/0, /*end=*/original_shape[i]));
    PrimExpr min = analyzer->Simplify(tvm::max(0, range->min));
    PrimExpr extent = analyzer->Simplify(tvm::min(original_shape[i], range->extent));

    // Check the buffer region is not loop dependent, since loop dependent
    // allocation is not supported yet.
    auto is_loop_var = [&ancestor_loops](const VarNode* v) {
      return std::any_of(ancestor_loops.begin(), ancestor_loops.end(),
                         [v](const ForNode* n) { return n->loop_var.get() == v; });
    };
    if (UsesVar(extent, is_loop_var)) {
      // try estimate a constant upperbound on region's extent
      int64_t upperbound = analyzer->const_int_bound(extent)->max_value;
      if (upperbound != arith::ConstIntBound::kPosInf) {
        extent = make_const(extent->dtype, upperbound);
      } else {
        // or else we have to fallback to full region
        min = make_zero(original_shape[i]->dtype);
        extent = original_shape[i];
      }
    }

    result.push_back(Range::FromMinExtent(min, extent));
  }
  return result;
}

/*! \brief a more constrained bound estimate for n-dimentional int set */
NDIntSet NDIntSetEval(Region region, PrimExpr predicate,
                      const std::unordered_map<const VarNode*, arith::IntSet>& dom_map,
                      arith::Analyzer* analyzer) {
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectEqual> var_dom;
  for (const auto& it : dom_map) {
    var_dom[GetRef<Var>(it.first)] = it.second.CoverRange(Range::FromMinExtent(0, 0));
  }
  Optional<Array<arith::IntSet>> eval_res =
      arith::EstimateRegionUpperBound(region, var_dom, predicate, analyzer);
  if (eval_res.defined()) {
    return NDIntSet(eval_res.value().begin(), eval_res.value().end());
  }
  return support::NDIntSetEval(support::NDIntSetFromRegion(region), dom_map);
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
    VisitExpr(op->value);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    VisitBufferAccess(BufferRegion::FromPoint(op->buffer, op->indices));
  }

  void VisitExpr_(const VarNode* op) final { VisitBufferVar(GetRef<Var>(op)); }

  void VisitExpr_(const LoadNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
  }

  void VisitStmt_(const StoreNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
  }

  void VisitStmt_(const ForNode* op) final {
    ancestor_loops_.push_back(op);
    Range loop_range = Range::FromMinExtent(op->min, op->extent);
    dom_analyzer_.Bind(op->loop_var, loop_range);
    dom_map_.emplace(op->loop_var.get(), arith::IntSet::FromRange(loop_range));
    StmtExprVisitor::VisitStmt_(op);
    dom_map_.erase(op->loop_var.get());
    ancestor_loops_.pop_back();
  }

  void VisitStmt_(const LetStmtNode* op) final {
    StmtExprVisitor::VisitExpr(op->value);
    if (arith::IsIndexType(op->value->dtype)) {
      dom_analyzer_.Bind(op->var, op->value);
      dom_map_.emplace(op->var.get(), arith::IntSet::SinglePoint(op->value));
    }
    StmtExprVisitor::VisitStmt(op->body);
    if (arith::IsIndexType(op->value->dtype)) {
      dom_map_.erase(op->var.get());
    }
  }

  void VisitExpr_(const LetNode* op) final {
    StmtExprVisitor::VisitExpr(op->value);
    if (arith::IsIndexType(op->value->dtype)) {
      dom_analyzer_.Bind(op->var, op->value);
      dom_map_.emplace(op->var.get(), arith::IntSet::SinglePoint(op->value));
    }
    StmtExprVisitor::VisitExpr(op->body);
    if (arith::IsIndexType(op->value->dtype)) {
      dom_map_.erase(op->var.get());
    }
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    // Visit condition
    StmtExprVisitor::VisitExpr(op->condition);
    {
      // Visit then branch
      With<ConditionalBoundsContext> ctx(op->condition, &dom_map_, &hint_map_, true);
      StmtExprVisitor::VisitStmt(op->then_case);
    }
    if (op->else_case) {
      // Visit else branch
      With<ConditionalBoundsContext> ctx(op->condition, &dom_map_, &hint_map_, false);
      StmtExprVisitor::VisitStmt(op->else_case.value());
    }
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::if_then_else())) {
      // Visit condition
      StmtExprVisitor::VisitExpr(op->args[0]);
      {
        // Visit then branch
        With<ConditionalBoundsContext> ctx(op->args[0], &dom_map_, &hint_map_, true);
        StmtExprVisitor::VisitExpr(op->args[1]);
      }
      {
        // Visit else branch
        With<ConditionalBoundsContext> ctx(op->args[0], &dom_map_, &hint_map_, false);
        StmtExprVisitor::VisitExpr(op->args[2]);
      }
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode* op) final {
    // Step 0. Check there is no init part.
    ICHECK(!op->init.defined());
    // Step 1. Record and update current read/write region annotations
    std::unordered_map<Buffer, std::vector<BufferRegion>, ObjectPtrHash, ObjectPtrEqual>
        cur_access_annotations;
    for (const BufferRegion& region : op->reads) {
      cur_access_annotations[region->buffer].push_back(region);
    }
    for (const BufferRegion& region : op->writes) {
      cur_access_annotations[region->buffer].push_back(region);
    }
    for (auto& p : cur_access_annotations) {
      auto& regions = access_annotations_[p.first];
      p.second.swap(regions);
    }
    // Step 2. Record relax position of ancestor_loops_ into buffer_var_in_scope_
    for (const Buffer& buffer : op->alloc_buffers) {
      buffer_var_in_scope_.emplace(buffer->data, std::make_pair(buffer, ancestor_loops_.size()));
    }
    // Step 3. Visit match buffers
    for (const MatchBufferRegion& region : op->match_buffers) {
      VisitBufferAccess(region->source);
    }
    // Step 4. Visit block body recursively
    StmtExprVisitor::VisitStmt_(op);
    // Step 5. Recover read/write region annotations
    for (auto& p : cur_access_annotations) {
      auto& regions = access_annotations_[p.first];
      if (p.second.empty()) {
        access_annotations_.erase(p.first);
      } else {
        regions.swap(p.second);
      }
    }
    // Step 6. Update buffer_access_region_ from relaxed_accesses_ for inner buffers.
    for (const Buffer& buffer : op->alloc_buffers) {
      auto it = relaxed_accesses_.find(buffer);
      ICHECK(it != relaxed_accesses_.end())
          << buffer << " is allocated but not accessed within block scope";
      const NDIntSet& nd_int_set = it->second;
      buffer_access_region_[buffer] = SimplifyAndNarrowBufferRegionFromNDIntSet(
          nd_int_set, buffer->shape, &dom_analyzer_, ancestor_loops_);
    }
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    PrimExpr cur_predicate = predicate_in_scope;
    predicate_in_scope = op->predicate;
    StmtExprVisitor::VisitStmt_(op);
    predicate_in_scope = cur_predicate;
  }

  /**************** Helper functions ****************/

  void VisitBufferAccess(const BufferRegion& buffer_region) {
    const BufferNode* buffer = buffer_region->buffer.get();
    auto it = buffer_var_in_scope_.find(buffer->data);
    if (it != buffer_var_in_scope_.end()) {
      const Buffer& buffer = it->second.first;
      size_t n_ancestor_loops = it->second.second;
      // Step 1. Stop ancestor loop vars out of the allocation block from
      // being relaxed unless NeedRelaxThread() is true.
      std::vector<arith::IntSet> non_relaxed(n_ancestor_loops);
      for (size_t i = 0; i < n_ancestor_loops; ++i) {
        const ForNode* loop = ancestor_loops_[i];
        const VarNode* v = loop->loop_var.get();
        if (NeedRelaxThread(GetRef<For>(loop), runtime::StorageScope::Create(buffer.scope()))) {
          continue;
        }
        auto dom_it = dom_map_.find(v);
        ICHECK(dom_it != dom_map_.end())
            << "Could not find domain for loop variable " << v->name_hint;
        non_relaxed[i] = dom_it->second;
        dom_map_.erase(dom_it);
      }
      // Step 2. Relax the access region
      NDIntSet nd_int_set =
          NDIntSetEval(buffer_region->region, predicate_in_scope, dom_map_, &dom_analyzer_);
      // Step 3. Restore the non-relaxed ancestor loops domain
      for (size_t i = 0; i < n_ancestor_loops; ++i) {
        const VarNode* v = ancestor_loops_[i]->loop_var.get();
        dom_map_.emplace(v, non_relaxed[i]);
      }
      // Step 4. Update relaxed_accesses_ dict
      auto access_it = relaxed_accesses_.find(buffer);
      if (access_it != relaxed_accesses_.end()) {
        support::NDIntSetUnionWith(&access_it->second, nd_int_set);
      } else {
        relaxed_accesses_.insert(access_it, {buffer, nd_int_set});
      }
    }
  }

  void VisitBufferVar(const Var& var) {
    auto it = buffer_var_in_scope_.find(var);
    if (it != buffer_var_in_scope_.end()) {
      const Buffer& buffer = it->second.first;
      auto annotation_it = access_annotations_.find(buffer);
      if (annotation_it != access_annotations_.end()) {
        // opaque buffer has explicit accessed region annotations
        for (const BufferRegion& region : annotation_it->second) {
          VisitBufferAccess(region);
        }
      } else {
        VisitBufferAccess(BufferRegion::FullRegion(buffer));
      }
    }
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
  /*! \brief The loops from the current node up to the root. */
  std::vector<const ForNode*> ancestor_loops_;

  /*!
   * \brief The vars of the buffer allocated under the current block.
   * Map each buffer var to (buffer_obj, n_ancester_loop) pair, where
   * n_ancester_loop is the loop num out of the current block.
   * Tancestor_loops_[0: n_ancester_loop] should not be relaxed when
   * we evaluate this buffer's access regions.
   */
  std::unordered_map<Var, std::pair<Buffer, size_t>, ObjectPtrHash, ObjectPtrEqual>
      buffer_var_in_scope_;
  /*! \brief The block predicate of current scope */
  PrimExpr predicate_in_scope{true};

  /*! \brief The map from loop vars to their iter range. */
  std::unordered_map<const VarNode*, arith::IntSet> dom_map_;
  /*! \brief Extra map from free vars to their iter range hints. */
  std::unordered_map<const VarNode*, arith::IntSet> hint_map_;
  /*! \brief The analyzer aware of loop domains. */
  arith::Analyzer dom_analyzer_;
  /*! \brief The map from Buffer to it's relaxed access set. */
  std::unordered_map<Buffer, NDIntSet, ObjectPtrHash, ObjectPtrEqual> relaxed_accesses_;
  /*! \brief The map from Buffer to it entire access region, used for returning. */
  std::unordered_map<Buffer, Region, ObjectPtrHash, ObjectPtrEqual> buffer_access_region_;
  /*! \brief The map from Buffer to it's access regions annotated by current block. */
  std::unordered_map<Buffer, std::vector<BufferRegion>, ObjectPtrHash, ObjectPtrEqual>
      access_annotations_;
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
