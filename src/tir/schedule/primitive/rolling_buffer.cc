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
#include <functional>

#include "../ir_comparator.h"
#include "../utils.h"

namespace tvm {
namespace tir {

namespace {

struct RollingBufferInfo {
  Buffer old_buffer;
  Buffer new_buffer;
  int rolling_axis;
  PrimExpr rolling_extent;
  std::vector<int> axis_overlaps;
  std::vector<Optional<Var>> axis_iter_vars;
  /*! \brief The map used for ScheduleStateNode::Replace. */
  Map<Block, Block> block_reuse;
};

BufferRegion GetRelaxedBufferRegion(const BlockRealize& realize, const BufferRegion& buffer_region,
                                    const Map<Var, arith::IntSet>& dom_map) {
  Array<arith::IntSet> relaxed_intsets =
      arith::EvalSet(Substitute(buffer_region->region, GetBindings(realize)), dom_map);
  Region relaxed_region;
  relaxed_region.reserve(relaxed_intsets.size());
  for (size_t i = 0; i < relaxed_intsets.size(); ++i) {
    relaxed_region.push_back(
        relaxed_intsets[i].CoverRange(Range::FromMinExtent(0, buffer_region->buffer->shape[i])));
  }
  return BufferRegion(buffer_region->buffer, relaxed_region);
}

class RollingBufferDependencyError : public ScheduleError {
 public:
  explicit RollingBufferDependencyError(IRModule mod, Block block)
      : mod_(mod), block_(std::move(block)) {}

  String FastErrorString() const final {
    return "ScheduleError: The target block is required to have only RAW dependencies";
  }

  String DetailRenderTemplate() const final {
    return "The target block {0} is required to have only RAW dependencies";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

  /*!
   * \brief Check if the block has only RAW dependencies.
   * \param self The schedule state
   * \param block_sref The sref of the block to be checked
   * \param scope_root_sref The sref of the scope root
   * \throw ScheduleError if the block has WAW or WAR dependency.
   */
  static void Check(const ScheduleState& self, const StmtSRef& block_sref,
                    const StmtSRef& scope_root_sref) {
    BlockScope scope = self->GetBlockScope(scope_root_sref);
    for (const Dependency& producers : scope->GetDepsByDst(block_sref)) {
      if (!(producers->kind == DepKind::kRAW)) {
        const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
        throw RollingBufferDependencyError(self->mod, GetRef<Block>(block));
      }
    }
    for (const Dependency& consumers : scope->GetDepsBySrc(block_sref)) {
      if (!(consumers->kind == DepKind::kRAW)) {
        const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
        throw RollingBufferDependencyError(self->mod, GetRef<Block>(block));
      }
    }
  }

 private:
  IRModule mod_;
  Block block_;
};

class RollingBufferMatchError : public ScheduleError {
 public:
  RollingBufferMatchError(IRModule mod, Block block, BufferRegion buffer_region)
      : mod_(mod), block_(block), buffer_region_(buffer_region) {}
  String FastErrorString() const final {
    return "ScheduleError: rolling_buffer expect the buffer region to have at least one dimention"
           "matching the rolling pattern such as: hh.outer * stride + hh.inner";
  }
  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The target buffer " << buffer_region_->buffer->name << " with region "
       << buffer_region_->region
       << " should have at least one dimension range that matches a rolling pattern "
          "such as hh.outer * stride + hh.inner. ";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  Block block_;
  BufferRegion buffer_region_;
};

class RollingBufferInsertionError : public ScheduleError {
 public:
  RollingBufferInsertionError(IRModule mod, Buffer buffer, Block block)
      : mod_(mod), buffer_(std::move(buffer)), block_(block) {}
  String FastErrorString() const final {
    return "ScheduleError: rolling_buffer injection is invalid, the lca of the access "
           "location of the target buffer is not a for loop. ";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "rolling_buffer injection is invalid. The block {0} should be tiled so that "
       << "the lca of the access location of the target buffer " << buffer_->name
       << " is a for loop. ";
    return os.str();
  }
  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  Buffer buffer_;
  Block block_;
};

class RollingBufferInfoCollector {
 public:
  static RollingBufferInfo CheckAndGetRollingBufferInfo(const IRModule& mod,
                                                        const StmtSRef& block_sref,
                                                        const BufferRegion& buffer_region) {
    RollingBufferInfoCollector collector;
    if (!collector.MatchRollingBuffer(block_sref, buffer_region)) {
      const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
      throw RollingBufferMatchError(mod, GetRef<Block>(block), buffer_region);
    }
    return collector.info_;
  }

 private:
  bool MatchRollingBuffer(const StmtSRef& block_sref, const BufferRegion& buffer_region) {
    const Buffer& buffer = buffer_region->buffer;
    const Region& region = buffer_region->region;

    std::vector<Optional<Var>> bound_iter_vars;
    std::vector<int> bound_overlaps;

    arith::PVar<Var> p_var;
    arith::PVar<IntImm> p_stride, p_divisor;
    for (auto bound : region) {
      auto stride = 0;
      auto divisor = 1;

      Optional<Var> iter_var;
      if (floordiv((p_var * p_stride), p_divisor).Match(bound->min)) {
        // Handle the case of fractional strides
        // They take this form: floordiv(hh.outer, 2)
        // Strip the floordiv and keep track of the divisor
        iter_var = p_var.Eval();
        divisor = p_divisor.Eval()->value;
        stride = std::ceil(static_cast<float>(p_stride.Eval()->value) / divisor);
      } else if ((p_var * p_stride).Match(bound->min)) {
        // The bound is the iter var multiplied by the stride
        iter_var = p_var.Eval();
        stride = p_stride.Eval()->value;
      } else if (p_var.Match(bound->min)) {
        // If the bound is just a Var, that implies the stride is 1
        iter_var = p_var.Eval();
        stride = 1;
      } else if (is_const_int(bound->min)) {
        // If the bound is an int, we can't roll over it
        iter_var = NullOpt;
      } else {
        // If all of the above matches fail, we're in unknown behaviour
        return false;
      }
      auto bound_overlap = 0;
      if (iter_var.defined()) {
        auto extent = Downcast<IntImm>(bound->extent)->value;
        bound_overlap = extent - stride;
        // Since Pass CompactBufferAllocation will be responsible for compacting the buffer
        // allocation region, there is no need to roll over the axis where the overlap is not
        // positive, so reset iter_var to NullOpt.
        if (bound_overlap <= 0) {
          iter_var = NullOpt;
        }
      }
      bound_iter_vars.push_back(iter_var);
      bound_overlaps.push_back(bound_overlap);
    }

    Array<StmtSRef> loop_srefs = GetLoops(block_sref);
    // Pick the outermost iter_var that's mentioned in the bounds
    // to be the rolling axis
    Optional<Var> roll_iter_var;
    int roll_axis;
    for (const tir::StmtSRef& loop_sref : loop_srefs) {
      auto loop_var = loop_sref->StmtAs<ForNode>()->loop_var;

      auto it{std::find_if(bound_iter_vars.begin(), bound_iter_vars.end(), [&](Optional<Var> var) {
        return var && (var.get() == loop_var.get());
      })};
      if (it != bound_iter_vars.end()) {
        auto i = std::distance(bound_iter_vars.begin(), it);
        roll_iter_var = loop_var;
        roll_axis = i;
        break;
      }
    }

    if (!roll_iter_var.defined()) {
      return false;
    }
    Array<PrimExpr> new_shape = buffer->shape;
    new_shape.Set(roll_axis, region[roll_axis]->extent);
    Buffer new_buffer = buffer;
    new_buffer.CopyOnWrite()->shape = new_shape;

    info_.old_buffer = buffer;
    info_.new_buffer = new_buffer;
    info_.rolling_axis = roll_axis;
    info_.rolling_extent = region[roll_axis]->extent;
    info_.axis_overlaps = bound_overlaps;
    info_.axis_iter_vars = bound_iter_vars;

    return true;
  }

  RollingBufferInfo info_;
};

class RollingBufferRewriter : public StmtExprMutator {
 public:
  static Stmt Rewrite(const StmtSRef& scope_sref, RollingBufferInfo* info) {
    RollingBufferRewriter rewriter(scope_sref, info);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }

 private:
  explicit RollingBufferRewriter(const StmtSRef& scope_sref, RollingBufferInfo* info)
      : scope_sref_(scope_sref), info_(info) {}

  void RewriteAccessRegion(Array<BufferRegion>* old_access_regions,
                           const Array<BufferRegion>& infered_access_regions) {
    auto fmutate = [this, &infered_access_regions](const BufferRegion& buffer_region) {
      if (buffer_region->buffer.same_as(info_->old_buffer)) {
        ICHECK(infered_access_regions.size() == 1);
        return infered_access_regions[0];
      }
      return buffer_region;
    };
    (*old_access_regions).MutateByApply(fmutate);
  }

  void RewriteBufferAccess(Buffer* buffer, Array<PrimExpr>* indices) const {
    Array<PrimExpr> new_indices;
    new_indices.reserve(indices->size());
    // First modify the access indices to use modulo arithmetic
    // for the rolling axis
    for (size_t i = 0; i < indices->size(); ++i) {
      if (static_cast<int>(i) == info_->rolling_axis) {
        new_indices.push_back(FloorMod((*indices)[i], info_->rolling_extent));
      } else {
        new_indices.push_back((*indices)[i]);
      }
    }
    // Replace the accessed buffer with the new buffer.
    *buffer = info_->new_buffer;
    *indices = std::move(new_indices);
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    Block old_stmt = GetRef<Block>(block);
    Block stmt = Downcast<Block>(StmtExprMutator::VisitStmt_(block));
    BlockNode* n = stmt.CopyOnWrite();
    if (block == scope_sref_->stmt) {
      Array<Buffer> new_alloc_buffers;
      for (const Buffer& buffer : stmt->alloc_buffers) {
        if (buffer != info_->old_buffer) {
          new_alloc_buffers.push_back(buffer);
        } else {
          new_alloc_buffers.push_back(info_->new_buffer);
        }
      }
      n->alloc_buffers = std::move(new_alloc_buffers);
    } else {
      Array<IterVar> new_iter_vars;
      for (size_t i = 0; i < stmt->iter_vars.size(); ++i) {
        auto old_iter_var = stmt->iter_vars[i];
        if (static_cast<int>(i) == info_->rolling_axis) {
          // All inner loops of the rolling axis has a loop carried dependency
          // (i.e. each iteration calculation of the rolling axis depends on
          // the calculation results of all the historical iterations of inner loops),
          // so annotate the iteration type of the rolling axis as 'opaque',
          // avoid the iterative range of its inner loop from being compressed
          // during lowering phase.
          IterVar new_iter_var =
              IterVar(old_iter_var->dom, old_iter_var->var, IterVarType::kOpaque);
          new_iter_vars.push_back(new_iter_var);
        } else {
          new_iter_vars.push_back(old_iter_var);
        }
      }
      Map<Var, Buffer> buffer_data_to_buffer = {{info_->new_buffer->data, info_->new_buffer}};
      auto infered_access_regions = GetBlockReadWriteRegion(stmt, buffer_data_to_buffer);

      n->iter_vars = std::move(new_iter_vars);
      RewriteAccessRegion(&n->reads, infered_access_regions[0]);
      RewriteAccessRegion(&n->writes, infered_access_regions[1]);
    }
    info_->block_reuse.Set(old_stmt, stmt);
    return std::move(stmt);
  }

  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    BlockRealize stmt = Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(realize));
    // Append block predicate to avoid recomputing elements.
    if (rewrite_block_predicate_) {
      rewrite_block_predicate_ = false;
      PrimExpr condition = stmt->predicate;
      for (size_t i = 0; i < info_->axis_iter_vars.size(); ++i) {
        auto iter_var = info_->axis_iter_vars[i];
        if (iter_var && info_->axis_overlaps[i] > 0) {
          Var var = iter_var.value();
          const Map<Var, arith::IntSet> dmap = {std::make_pair(var, arith::IntSet::Interval(0, 0))};
          auto iter_value = realize->iter_values[i];
          arith::Analyzer analyzer;
          auto term_2 = analyzer.int_set(iter_value, dmap).min();
          condition = analyzer.Simplify(
              And(condition, Or(LT(var, 1), GE(term_2, info_->axis_overlaps[i]))));
        }
      }
      BlockRealizeNode* n = stmt.CopyOnWrite();
      n->predicate = condition;
    }
    return std::move(stmt);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore stmt = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (stmt->buffer.same_as(info_->old_buffer)) {
      BufferStoreNode* n = stmt.CopyOnWrite();
      RewriteBufferAccess(&n->buffer, &n->indices);
      // Need to add predicate to the current block to avoid recomputing elements.
      rewrite_block_predicate_ = true;
    }
    return std::move(stmt);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad stmt = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (stmt->buffer.same_as(info_->old_buffer)) {
      BufferLoadNode* n = stmt.CopyOnWrite();
      RewriteBufferAccess(&n->buffer, &n->indices);
    }
    return std::move(stmt);
  }

 private:
  const StmtSRef& scope_sref_;
  RollingBufferInfo* info_;
  bool rewrite_block_predicate_ = false;
};

}  // namespace

void RollingBuffer(ScheduleState self, const StmtSRef& block_sref, int write_buffer_index) {
  /*!
   *  Check
   *    - The block is not an output block.
   *    - The block has only RAW dependencies.
   *    - The block is tiled and there is access overlap between adjacent tiles.
   *  Mutate
   *    - Select the outermost rollable axis appeared in the block's loop nest
   *      as the 'rolling axis', trim the target buffer from the rolling axis.
   *    - Use modulo arithmetic to modify the target buffer's read and load
   *      indices to circularize the buffer along the rolling dimension.
   *    - Append block predicate to avoid recomputing overlapping elements.
   */
  Map<Var, arith::IntSet> dom_map;
  const BlockRealize& realize = GetBlockRealize(self, block_sref);
  const Block& block = realize->block;

  // Step 1. Checking index, getting the target buffer region and the parent scope.
  const BufferRegion& buffer_region =
      GetNthAccessBufferRegion(self, block, write_buffer_index, BufferIndexType::kWrite);
  StmtSRef scope_root_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  // Step 2. Check if the target block is not an output block and has only RAW dependencies.
  CheckNotOutputBlock(self, block_sref, scope_root_sref);
  RollingBufferDependencyError::Check(self, block_sref, scope_root_sref);

  // Step 3. Find the lca of the access location of the target buffer and relax the buffer
  Array<StmtSRef> loop_srefs = GetLoops(block_sref);
  Array<StmtSRef> consumers_sref = GetConsumers(self, block_sref);
  consumers_sref.push_back(block_sref);
  StmtSRef lca = GetSRefLowestCommonAncestor(consumers_sref);
  if (!lca->StmtAs<ForNode>()) {
    throw RollingBufferInsertionError(self->mod, buffer_region->buffer, block);
  }

  for (auto it = loop_srefs.rbegin(); it != loop_srefs.rend(); ++it) {
    auto stmt = *it;
    // Stop at the lca of all the rolling_buffer access points;
    if (stmt == lca) {
      break;
    }
    For cur_loop = GetRef<For>(stmt->StmtAs<ForNode>());
    Range range = Range::FromMinExtent(cur_loop->min, cur_loop->extent);
    dom_map.Set(cur_loop->loop_var, arith::IntSet::FromRange(range));
  }
  BufferRegion relaxed_region = GetRelaxedBufferRegion(realize, buffer_region, dom_map);

  // Step 4. Find a valid rolling axis and collect bound overlaps on the target buffer.
  RollingBufferInfo info = RollingBufferInfoCollector::CheckAndGetRollingBufferInfo(
      self->mod, block_sref, relaxed_region);
  // Step 5. Mutate IR to apply rolling access pattern.
  Stmt new_scope_root = RollingBufferRewriter::Rewrite(scope_root_sref, &info);

  // Step 6. Update schedule states
  self->Replace(scope_root_sref, new_scope_root, info.block_reuse);
  // Step 7. Regenerate block info from the root block, because `region_cover` for the target block
  // and `stage_pipeline` for the root block are no longer satisfied after rolling buffer injection.
  self->UpdateScopeBlockInfo(tir::GetBlockRealize(self, self->stmt2ref.at(new_scope_root.get())));
}

struct RollingBufferTraits : public UnpackedInstTraits<RollingBufferTraits> {
  static constexpr const char* kName = "RollingBuffer";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer write_buffer_index) {
    return sch->RollingBuffer(block, write_buffer_index.IntValue());
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Integer write_buffer_index) {
    PythonAPICall py("rolling_buffer");
    py.Input("block", block);
    py.Input("write_buffer_index", write_buffer_index);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(RollingBufferTraits);
}  // namespace tir
}  // namespace tvm
