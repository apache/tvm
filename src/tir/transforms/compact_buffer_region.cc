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

#include <numeric>
#include <stack>

#include "../../support/arena.h"
#include "../../support/nd_int_set.h"
#include "../../support/utils.h"
#include "../schedule/utils.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

using support::NDIntSet;

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
 * \brief Collect buffer aliasing information.
 */
class Var2BufferCollector : public StmtExprVisitor {
 public:
  /*! \brief Map the buffer var to all aliased buffers. */
  std::unordered_map<Var, std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>> var2buffer_;

 private:
  void VisitStmt_(const BufferStoreNode* op) final {
    var2buffer_[op->buffer->data].insert(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    var2buffer_[op->buffer->data].insert(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode* op) final {
    for (const Buffer& buffer : op->alloc_buffers) {
      var2buffer_[buffer->data].insert(buffer);
    }
    for (const MatchBufferRegion& region : op->match_buffers) {
      var2buffer_[region->buffer->data].insert(region->buffer);
      var2buffer_[region->source->buffer->data].insert(region->source->buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode* op) final {
    var2buffer_[op->buffer->data].insert(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
};

/*!
 * \brief Collect the access region of each buffer.
 * \note The param buffer regions will not be collected.
 */
class BufferAccessRegionCollector : public StmtExprVisitor {
 public:
  static std::unordered_map<Buffer, Region, ObjectPtrHash, ObjectPtrEqual> Collect(
      const PrimFunc& f, bool collect_inbound) {
    BufferAccessRegionCollector region_collector(collect_inbound);

    // collect buffer var to aliased buffer mapping
    Var2BufferCollector var2buffer_collector;
    var2buffer_collector(f->body);
    std::swap(region_collector.var2buffer_, var2buffer_collector.var2buffer_);

    // collect buffer access regions
    region_collector(f->body);
    return std::move(region_collector.buffer_access_region_);
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

  explicit BufferAccessRegionCollector(bool collect_inbound) : collect_inbound_(collect_inbound) {}

  /**************** Visitor overload ****************/

  void VisitStmt_(const BufferStoreNode* op) final {
    VisitBufferAccess(BufferRegion::FromPoint(op->buffer, op->indices));
    VisitExpr(op->value);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    auto explicit_it = explicit_access_annotations_.find(op->buffer);
    if (explicit_it != explicit_access_annotations_.end()) {
      VisitBufferAccess(explicit_it->second);
    } else {
      VisitBufferAccess(BufferRegion::FromPoint(op->buffer, op->indices));
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode* op) final { VisitBufferVar(GetRef<Var>(op)); }

  void VisitStmt_(const ForNode* op) final {
    Range loop_range = Range::FromMinExtent(op->min, op->extent);
    IterVar iter = op->kind == ForKind::kThreadBinding
                       ? IterVar(Range(), op->loop_var, IterVarType::kThreadIndex,
                                 op->thread_binding.value()->thread_tag)
                       : IterVar(Range(), op->loop_var, IterVarType::kDataPar);
    ancestor_iters_.push_back(iter);
    dom_analyzer_.Bind(op->loop_var, loop_range);
    dom_map_.emplace(op->loop_var.get(), arith::IntSet::FromRange(loop_range));
    StmtExprVisitor::VisitStmt_(op);
    dom_map_.erase(op->loop_var.get());
    ancestor_iters_.pop_back();
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
      With<ConditionalBoundsContext> ctx(op->condition, &dom_map_, &hint_map_,
                                         &pending_conditions_);
      StmtExprVisitor::VisitStmt(op->then_case);
    }
    if (op->else_case) {
      // Visit else branch
      With<ConditionalBoundsContext> ctx(!op->condition, &dom_map_, &hint_map_,
                                         &pending_conditions_);
      StmtExprVisitor::VisitStmt(op->else_case.value());
    }
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::if_then_else())) {
      // Visit condition
      StmtExprVisitor::VisitExpr(op->args[0]);
      {
        // Visit then branch
        With<ConditionalBoundsContext> ctx(op->args[0], &dom_map_, &hint_map_,
                                           &pending_conditions_);
        StmtExprVisitor::VisitExpr(op->args[1]);
      }
      {
        // Visit else branch
        With<ConditionalBoundsContext> ctx(!op->args[0], &dom_map_, &hint_map_,
                                           &pending_conditions_);
        StmtExprVisitor::VisitExpr(op->args[2]);
      }
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode* op) final {
    // Step 0. Check there is no init part and block is opaque
    ICHECK(!op->init.defined());
    ICHECK_EQ(op->iter_vars.size(), 0) << "CompactBufferRegion only works on opaque blocks";
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

    // Step 2. Record explicit read/write region annotations
    auto record_explicit_region = [&](const String& attr_key, BufferIndexType index_type) {
      auto it = op->annotations.find(attr_key);
      if (it != op->annotations.end()) {
        Array<Integer> buffer_indices = Downcast<Array<Integer>>((*it).second);
        for (const auto& index : buffer_indices) {
          int buffer_index = index->value;
          if (buffer_index >= 0 && buffer_index < static_cast<int>(op->reads.size())) {
            const BufferRegion& explicit_region = index_type == BufferIndexType::kRead
                                                      ? op->reads[buffer_index]
                                                      : op->writes[buffer_index];
            explicit_access_annotations_[explicit_region->buffer] = explicit_region;
          }
        }
      }
    };

    record_explicit_region(attr::explicit_read_region, BufferIndexType::kRead);
    record_explicit_region(attr::explicit_write_region, BufferIndexType::kWrite);

    // Step 3. Record relax position of ancestor_loops_
    for (const Buffer& buffer : op->alloc_buffers) {
      VisitBufferDef(buffer->data);
    }
    // Step 4. Visit match buffers
    for (const MatchBufferRegion& region : op->match_buffers) {
      VisitBufferAccess(region->source);
    }
    // Step 5. Visit block body recursively
    StmtExprVisitor::VisitStmt_(op);
    // Step 6. Recover read/write region annotations
    for (auto& p : cur_access_annotations) {
      auto& regions = access_annotations_[p.first];
      if (p.second.empty()) {
        access_annotations_.erase(p.first);
      } else {
        regions.swap(p.second);
      }
    }
    // Step 7. Clear explicit access annotations
    explicit_access_annotations_.clear();
    // Step 8. Update buffer_access_region_ from relaxed_accesses_ for inner buffers.
    for (const Buffer& buffer : op->alloc_buffers) {
      ICHECK_EQ(var2buffer_[buffer->data].size(), 1)
          << "Block allocation buffer shoud not be alised";
      SimplifyAndNarrowBufferRegionFromNDIntSet(buffer);
    }
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    With<ConditionalBoundsContext> ctx(op->predicate, &dom_map_, &hint_map_, &pending_conditions_);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocateNode* op) final {
    auto it = var2buffer_.find(op->buffer_var);

    // Do not make compaction when the buffer def and
    // the allocation is not one-to-one with the same dtype.
    if (it == var2buffer_.end() || it->second.size() > 1) {
      return StmtExprVisitor::VisitStmt_(op);
    }
    const Buffer& buffer = *it->second.begin();
    if (buffer->dtype != op->dtype) {
      return StmtExprVisitor::VisitStmt_(op);
    }

    // Step 0. Record relax position of ancestor_loops_
    VisitBufferDef(op->buffer_var);
    // Step 1. Visit block body recursively
    StmtExprVisitor::VisitStmt(op->body);
    // Step 2. Update buffer_access_region_ from relaxed_accesses_ for inner buffers.
    SimplifyAndNarrowBufferRegionFromNDIntSet(buffer);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread) {
      IterVar iter = Downcast<IterVar>(op->node);
      ancestor_iters_.push_back(iter);
      Range dom = iter->dom;
      if (!dom.defined()) {  // dom is empty for legacy te schedule
        dom = Range::FromMinExtent(make_zero(op->value->dtype), op->value);
      }
      dom_analyzer_.Bind(iter->var, dom);
      dom_map_.emplace(iter->var.get(), arith::IntSet::FromRange(dom));
      StmtExprVisitor::VisitStmt_(op);
      dom_map_.erase(iter->var.get());
      ancestor_iters_.pop_back();
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  /**************** Helper functions ****************/

  /*! \brief Record information on the buffer defining point. */
  void VisitBufferDef(const Var& buffer_data) {
    auto it = buffer_scope_depth_.find(buffer_data);
    ICHECK(it == buffer_scope_depth_.end()) << buffer_data << " has duplicate definitions";
    buffer_scope_depth_.insert(it, {buffer_data, ancestor_iters_.size()});
  }

  void VisitBufferAccess(const BufferRegion& buffer_region) {
    const Buffer& buffer = buffer_region->buffer;
    auto it = buffer_scope_depth_.find(buffer->data);
    if (it != buffer_scope_depth_.end()) {
      size_t n_ancestor_loops = it->second;
      // Step 1. Stop ancestor loop vars out of the allocation block from
      // being relaxed unless NeedRelaxThread() is true.
      std::vector<arith::IntSet> non_relaxed(n_ancestor_loops);
      for (size_t i = 0; i < n_ancestor_loops; ++i) {
        const IterVar& iter = ancestor_iters_[i];
        const VarNode* v = iter->var.get();
        if (NeedRelaxThread(iter, runtime::StorageScope::Create(buffer.scope()))) {
          continue;
        }
        auto dom_it = dom_map_.find(v);
        ICHECK(dom_it != dom_map_.end())
            << "Could not find domain for loop variable " << v->name_hint;
        non_relaxed[i] = dom_it->second;
        dom_map_.erase(dom_it);
      }
      // Step 2. Relax the access region
      auto normalize_pred = [](const PrimExpr& pred) {
        if (pred->dtype.is_bool()) return pred;
        return pred != make_zero(pred->dtype);
      };
      PrimExpr predicate = dom_analyzer_.Simplify(
          std::accumulate(pending_conditions_.begin(), pending_conditions_.end(), const_true(),
                          [normalize_pred](const PrimExpr& x, const PrimExpr& y) {
                            return normalize_pred(x) && normalize_pred(y);
                          }));
      NDIntSet nd_int_set =
          NDIntSetEval(buffer_region->region, predicate, dom_map_, &dom_analyzer_);

      // Step 3. Restore the non-relaxed ancestor loops domain
      for (size_t i = 0; i < n_ancestor_loops; ++i) {
        const VarNode* v = ancestor_iters_[i]->var.get();
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
    auto it = var2buffer_.find(var);
    if (it == var2buffer_.end()) {
      return;
    }
    for (const Buffer& buffer : it->second) {
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

  /*! \brief Check whether the thread binding iter should be relaxed with given storage scope. */
  static bool NeedRelaxThread(const IterVar& iter, const runtime::StorageScope& scope) {
    if (iter->iter_type != IterVarType::kThreadIndex) {
      return false;
    }
    ICHECK(iter->thread_tag.defined());
    // When there is warp memory
    // threadIdx.x must be set to be warp index.
    return CanRelaxStorageUnderThread(scope, runtime::ThreadScope::Create((iter->thread_tag)));
  }

  /*!
   * \brief simplify and narrow down the region collected by NDIntSet.
   * Update the `relaxed_accesses_` dict. If `collect_inbound_` is true,
   * the result region would never exceed the original buffer shape.
   */
  void SimplifyAndNarrowBufferRegionFromNDIntSet(const Buffer& buffer) {
    auto it = relaxed_accesses_.find(buffer);
    ICHECK(it != relaxed_accesses_.end())
        << buffer << " is allocated but not accessed within block scope";

    const Array<PrimExpr>& original_shape = buffer->shape;
    const NDIntSet& nd_int_set = it->second;
    Array<Range>& result_region = buffer_access_region_[buffer];
    result_region.resize(nd_int_set.size());

    for (size_t i = 0; i < nd_int_set.size(); ++i) {
      const arith::IntSet& int_set = nd_int_set[i];
      Range original =
          Range(/*begin=*/make_zero(original_shape[i]->dtype), /*end=*/original_shape[i]);
      Range range = int_set.CoverRange(original);
      PrimExpr min, extent;
      if (collect_inbound_) {
        min = dom_analyzer_.Simplify(tvm::max(0, range->min));
        extent = range->extent;
        // Apply stronger symbolic proof to help us remove symbolic min here.
        if (!dom_analyzer_.CanProveLessEqualThanSymbolicShapeValue(extent, original_shape[i])) {
          extent = tvm::min(original_shape[i], range->extent);
        }
        extent = dom_analyzer_.Simplify(extent);
      } else {
        min = dom_analyzer_.Simplify(range->min);
        extent = dom_analyzer_.Simplify(range->extent);
      }

      // We check the buffer extent is pure and not loop dependent, since loop dependent
      // or data dependent allocation is not supported yet. Otherwise we should
      // fallback to use original buffer shape.
      if (SideEffect(extent) > CallEffectKind::kPure) {
        result_region.Set(i, original);
        continue;
      }
      auto is_loop_var = [this](const VarNode* v) {
        return std::any_of(ancestor_iters_.begin(), ancestor_iters_.end(),
                           [v](const IterVar& n) { return n->var.get() == v; });
      };
      if (UsesVar(extent, is_loop_var)) {
        // try estimate a constant upperbound on region's extent
        int64_t upperbound = dom_analyzer_.const_int_bound(extent)->max_value;
        if (upperbound != arith::ConstIntBound::kPosInf) {
          extent = make_const(extent->dtype, upperbound);
        } else {
          result_region.Set(i, original);
          continue;
        }
      }
      result_region.Set(i, Range::FromMinExtent(min, extent));
    }
  }

  /**************** Class members ****************/
  /*! \brief Only collect accessed region within original buffer shape bound. */
  bool collect_inbound_{true};

  /*! \brief The iteration scopes from the current node up to the root. */
  std::vector<IterVar> ancestor_iters_;

  /*!
   * \brief Map each buffer var to the n_ancester_loop. which is the loop depth at the
   * define point. ancestor_loops_[0: n_ancester_loop] should not be relaxed when
   * we evaluate this buffer's access regions.
   */
  std::unordered_map<Var, size_t> buffer_scope_depth_;

  /*! \brief Map the buffer var to all aliased buffers. */
  std::unordered_map<Var, std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>> var2buffer_;

  /*! \brief The map from loop vars to their iter range. */
  std::unordered_map<const VarNode*, arith::IntSet> dom_map_;
  /*! \brief Extra map from free vars to their iter range hints. */
  std::unordered_map<const VarNode*, arith::IntSet> hint_map_;
  /*! \brief Unresolved conditions within current scope. */
  std::vector<PrimExpr> pending_conditions_;
  /*! \brief The analyzer aware of loop domains. */
  arith::Analyzer dom_analyzer_;
  /*! \brief The map from Buffer to it's relaxed access set. */
  std::unordered_map<Buffer, NDIntSet, ObjectPtrHash, ObjectPtrEqual> relaxed_accesses_;

  /*!
   * \brief The map from Buffer to it entire access region, used for returning.
   * The entire access region should get updated on the buffer's define point
   * and we sanity check that every buffer is defined only once.
   */
  std::unordered_map<Buffer, Region, ObjectPtrHash, ObjectPtrEqual> buffer_access_region_;

  /*! \brief The map from Buffer to it's access regions annotated by current block. */
  std::unordered_map<Buffer, std::vector<BufferRegion>, ObjectPtrHash, ObjectPtrEqual>
      access_annotations_;
  /*! \brief The map from Buffer to its explicit access region annotated by the block. */
  std::unordered_map<Buffer, BufferRegion, ObjectPtrHash, ObjectPtrEqual>
      explicit_access_annotations_;
};

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
};

/*! \brief Reallocate the buffers with minimal region. */
class BufferCompactor : public StmtExprMutator {
 public:
  explicit BufferCompactor(std::unordered_map<Var, BufferAllocInfo> buffer_info)
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
    Array<Buffer> alloc_buffers =
        op->alloc_buffers.Map([this](const Buffer& buf) { return RewriteAllocBuffer(buf); });
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

  Stmt VisitStmt_(const DeclBufferNode* op) final {
    Buffer new_buffer = RewriteAllocBuffer(op->buffer);
    auto n = CopyOnWrite(op);
    n->buffer = std::move(new_buffer);
    n->body = VisitStmt(op->body);
    return DeclBuffer(n);
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    Allocate allocate = Downcast<Allocate>(StmtExprMutator::VisitStmt_(op));
    auto it = buffer_info_.find(allocate->buffer_var);
    if (it == buffer_info_.end()) {
      return std::move(allocate);
    }
    // Rewrite allocation shape if the corresponding buffer is in the buffer_info_
    // dict and the dtype is consistent, which denotes there are no buffer aliasing
    // and the compaction is safe.
    const Buffer& new_buffer = it->second.new_buffer;
    if (op->dtype != new_buffer->dtype) {
      return std::move(allocate);
    }
    Array<PrimExpr> new_shape = GetBufferAllocationShape(new_buffer);
    auto n = allocate.CopyOnWrite();
    ICHECK(n->buffer_var.same_as(new_buffer->data));
    n->extents = new_shape;
    return std::move(allocate);
  }

  Buffer RewriteAllocBuffer(const Buffer& buffer) {
    auto it = buffer_info_.find(buffer->data);
    if (it != buffer_info_.end()) {
      return it->second.new_buffer;
    }
    return buffer;
  }

  void RewriteBufferAccess(Buffer* buffer, Array<PrimExpr>* indices) const {
    auto it = buffer_info_.find((*buffer)->data);
    if (it == buffer_info_.end()) {
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
    auto it = buffer_info_.find((*buffer)->data);
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

  /*! \brief Map buffer var to the allocation information about each buffer. */
  std::unordered_map<Var, BufferAllocInfo> buffer_info_;
};

Array<PrimExpr> CalcStrides(const BufferAllocInfo& alloc_info, const Array<PrimExpr>& shape) {
  std::vector<PrimExpr> strides;
  if (alloc_info.dim_aligns.size()) {
    ICHECK(alloc_info.dim_aligns.size() == shape.size());
    strides.resize(shape.size());
    PrimExpr stride = make_const(shape[0].dtype(), 1);
    for (size_t i = shape.size(); i != 0; --i) {
      size_t dim = i - 1;
      DimAlignInfo info = alloc_info.dim_aligns[dim];
      int align_factor = info.align_factor;
      int align_offset = info.align_offset;
      if (align_factor != 0) {
        PrimExpr factor = make_const(stride.dtype(), align_factor);
        PrimExpr offset = make_const(stride.dtype(), align_offset);
        stride = stride + indexmod(factor + offset - indexmod(stride, factor), factor);
      }
      strides[dim] = stride;
      stride = stride * shape[dim];
    }
  }
  return strides;
}

Stmt BufferCompactorCompact(
    const PrimFunc& f,
    const std::unordered_map<Buffer, Region, ObjectPtrHash, ObjectPtrEqual>& regions,
    const std::unordered_map<Var, StorageAlignAnnotation>& storage_align) {
  // collect buffer allocation info for no-alias buffers
  std::unordered_map<Var, BufferAllocInfo> buffer_info;
  for (const auto& kv : regions) {
    const Buffer& buffer = kv.first;
    // set dim alignment info
    Region region = kv.second;
    BufferAllocInfo alloc_info;
    auto it = storage_align.find(buffer->data);
    if (it != storage_align.end()) {
      std::vector<DimAlignInfo> dim_aligns(buffer->shape.size());
      for (const StorageAlignTuple& dim_align : (*it).second) {
        ICHECK(dim_align.size() == 4);
        int dim = dim_align[1]->value;
        int factor = dim_align[2]->value;
        int offset = dim_align[3]->value;
        dim_aligns.at(dim) = {factor, offset};
      }
      alloc_info.dim_aligns = std::move(dim_aligns);
    }

    // prepare new buffer
    Array<PrimExpr> shape = region.Map([](const Range& range) { return range->extent; });
    Array<PrimExpr> strides = CalcStrides(alloc_info, shape);
    ObjectPtr<BufferNode> n = make_object<BufferNode>(*buffer.get());
    n->shape = std::move(shape);
    n->strides = std::move(strides);
    alloc_info.new_buffer = Buffer(std::move(n));
    alloc_info.region = region;
    buffer_info.emplace(buffer->data, std::move(alloc_info));
  }
  BufferCompactor compactor(std::move(buffer_info));
  Stmt stmt = compactor(f->body);
  return stmt;
}

PrimFunc CompactBufferAllocation(PrimFunc f, bool is_strict) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    auto region = BufferAccessRegionCollector::Collect(f, /*collect_inbound=*/is_strict);
    auto storage_align = CollectStorageAlignAnnotation(f->body);
    fptr->body = BufferCompactorCompact(f, region, storage_align);
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass CompactBufferAllocation(bool is_strict) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return CompactBufferAllocation(std::move(f), is_strict);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.CompactBufferAllocation", {});
}

TVM_REGISTER_GLOBAL("tir.transform.CompactBufferAllocation")
    .set_body_typed(CompactBufferAllocation);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
