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

#include <unordered_set>

#include "../../analysis/var_use_def_analysis.h"
#include "../../transforms/ir_utils.h"
#include "../utils.h"

namespace tvm {
namespace tir {

/******** Error Classes ********/

class NotSingleWriteBlock : public ScheduleError {
 public:
  explicit NotSingleWriteBlock(IRModule mod, Buffer buffer, Array<StmtSRef> write_blocks)
      : mod_(std::move(mod)), buffer_(std::move(buffer)) {
    ICHECK_GT(write_blocks.size(), 1);
    write_blocks_.reserve(write_blocks.size());
    for (const StmtSRef& block_sref : write_blocks) {
      const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
      write_blocks_.push_back(GetRef<Block>(block));
    }
  }

  String FastErrorString() const final {
    return "ScheduleError: The buffer is allowed to be written by single block.";
  }

  String DetailRenderTemplate() const final {
    size_t k = write_blocks_.size();
    return "The buffer " + buffer_->name + " is expected to be written by single block, but got " +
           std::to_string(k) + " blocks who write it.";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final {
    return {write_blocks_.begin(), write_blocks_.end()};
  }

 private:
  IRModule mod_;
  Buffer buffer_;
  Array<Block> write_blocks_;
};

/******** Helper Functions/Classes ********/

/*! \brief The auxiliary info used for the insertion point and content of the cache stage. */
struct CacheStageInfo {
  /*! \brief The buffer to be read. */
  Buffer read_buffer;
  /*! \brief The buffer to be written. */
  Buffer write_buffer;
  /*! \brief The buffer allocation to be inserted into the block signature. */
  Optional<Buffer> alloc;
  /*! \brief The AST node whose body is where the cache stage should be inserted. */
  StmtSRef loc_sref;
  /*! \brief The index to insert the cache_read/cache_write stage. */
  size_t loc_pos;
  /*! \brief The cache_read/cache_write stage to be inserted. */
  Stmt cache_stage;
  /*! \brief The map used for ScheduleStateNode::Replace. */
  Map<Block, Block> block_reuse;
  /*! \brief A set of blocks that will consume the new cache. */
  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> consumer_blocks;
  /*! \brief cache region for the buffer to be cached */
  BufferRegion cache_region;
};

/*! \brief Return the buffer region related with the buffer */
Optional<BufferRegion> GetBufferRegionFromBuffer(const Array<BufferRegion>& buffer_regions,
                                                 const Buffer& buffer) {
  Optional<BufferRegion> res = NullOpt;
  for (const auto& region : buffer_regions) {
    if (region->buffer.same_as(buffer)) {
      ICHECK(!res.defined());
      res = region;
    }
  }
  return res;
}

struct ReindexCacheStageInfo : CacheStageInfo {
  /* Indices used to access the allocated cache buffer. */
  Array<PrimExpr> indices;
  /* Touched loop variable related information. */
  Array<Var> loop_vars;
  Array<Range> loop_ranges;
  /* Touched block variable related information. */
  Array<IterVar> block_iter_vars;
  Array<PrimExpr> block_iter_values;
};

/* \brief The schedule error that accessed buffer region is not a single point for
 * reindex_cache_read/write. */
class NotSinglePointAccess : public ScheduleError {
 public:
  explicit NotSinglePointAccess(IRModule mod, Block block, BufferRegion cache_region,
                                bool is_cache_read)
      : mod_(std::move(mod)), block_(std::move(block)), cache_region_(cache_region) {
    primitive_name_ = is_cache_read ? "reindex_cache_read" : "reindex_cache_write";
  }

  String FastErrorString() const final {
    return "ScheduleError: The buffer region accessed inside the block is not a single point.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The buffer region " << cache_region_
       << " accessed inside block {0} is not a single point, which violates"
       << " the prerequisite of " << primitive_name_ << " primitive.";
    return String(os.str());
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  Block block_;
  BufferRegion cache_region_;
  String primitive_name_;
};

/*!
 * \brief Create a loop nest that represents reindex cache copy (reindex_cache_read /
 * reindex_cache_write) from read buffer to write buffer.
 * \param cache_region The cached copy region.
 * \param info The cache stage information, which will be updated in the function.
 * \param storage_scope The storage scope of the cached buffer (only used in naming here)
 * \returns A block indicating the body of the loop nesting.
 */
template <bool is_cache_read>
Block MakeReindexCacheStage(const BufferRegion& cache_region, ReindexCacheStageInfo* info,
                            const String& storage_scope) {
  // loop variables
  std::vector<Var> loop_vars;
  // block variables
  Array<IterVar> block_vars;
  // bindings in block realize
  std::vector<PrimExpr> iter_values;
  // Create loop vars and block vars' binding_value
  Map<Var, Var> var_map;
  for (size_t i = 0; i < info->loop_vars.size(); ++i) {
    Var original_var = info->loop_vars[i];
    Var loop_var(original_var->name_hint, original_var.dtype());
    var_map.Set(original_var, loop_var);
    loop_vars.push_back(loop_var);
  }
  for (size_t i = 0; i < info->block_iter_vars.size(); ++i) {
    IterVar original_block_var = info->block_iter_vars[i];
    PrimExpr original_iter_value = info->block_iter_values[i];
    IterVar block_var = IterVar(
        /*dom=*/original_block_var->dom,
        /*var=*/Var(original_block_var->var->name_hint, original_block_var->var.dtype()),
        /*IterVarType=*/kDataPar);
    var_map.Set(original_block_var->var, block_var->var);
    block_vars.push_back(block_var);
    iter_values.push_back(Substitute(original_iter_value, var_map));
  }

  // block access region for read/write buffers
  Region read_access_region, write_access_region;
  Array<PrimExpr> read_access_indices, write_access_indices;
  // Compute read/write region and read/write access indices.
  Array<PrimExpr>& old_indices = (is_cache_read) ? read_access_indices : write_access_indices;
  Region& old_region = (is_cache_read) ? read_access_region : write_access_region;
  for (const Range& range : cache_region->region) {
    old_indices.push_back(Substitute(range->min, var_map));
    old_region.push_back(Range::FromMinExtent(old_indices.back(), Integer(1)));
  }
  Array<PrimExpr>& new_indices = (is_cache_read) ? write_access_indices : read_access_indices;
  Region& new_region = (is_cache_read) ? write_access_region : read_access_region;
  for (const PrimExpr& idx : info->indices) {
    new_indices.push_back(Substitute((idx), var_map));
    new_region.push_back(Range::FromMinExtent(new_indices.back(), Integer(1)));
  }

  // Create New Block
  Block block(
      /*iter_vars*/ std::move(block_vars),
      /*reads=*/{BufferRegion(info->read_buffer, read_access_region)},
      /*writes=*/{BufferRegion(info->write_buffer, write_access_region)},
      /*name_hint*/ cache_region->buffer->name + "_" + storage_scope,
      /*body=*/
      BufferStore(info->write_buffer, BufferLoad(info->read_buffer, read_access_indices),
                  write_access_indices),
      /*init=*/NullOpt,
      /*alloc_buffers=*/{},
      /*match_buffers=*/{},
      /*buf_doms=*/{});
  // Create Block Realize node
  Stmt body = BlockRealize(/*values=*/iter_values,
                           /*predicate=*/const_true(),
                           /*block=*/block);
  // Create surrounding loops
  for (size_t i = loop_vars.size(); i >= 1; --i) {
    body = For(/*loop_var=*/loop_vars[i - 1],
               /*min=*/info->loop_ranges[i - 1]->min,
               /*extent=*/info->loop_ranges[i - 1]->extent,
               /*kind=*/ForKind::kSerial,
               /*body=*/body);
  }
  info->cache_stage = std::move(body);
  return block;
}

/*!
 * \brief Create a loop nest that represents cache copy (cache_read / cache_write) from read buffer
 *        to write buffer.
 * \note This function will store the stmt with loop nesting to the CacheStageInfo, but only return
 *        the inside block.
 * \param cache_region The cached copy region.
 * \param info The cache stage information, which will be updated in the function.
 * \param storage_scope The storage scope of the cached buffer (only used in naming here)
 * \param cache_full_region A boolean indicating if the cache buffer is allocated with
 *        full region or compact region.
 * \returns A block indicating the body of the loop nesting.
 */
Block MakeCacheStage(const BufferRegion& cache_region, CacheStageInfo* info,
                     const String& storage_scope, bool cache_full_region = true) {
  // loop variables
  std::vector<Var> loop_vars;
  // bindings in block realize
  std::vector<PrimExpr> iter_values;
  // Create loop vars and block vars' binding_value
  for (const Range& axis_range : cache_region->region) {
    Var loop_var("ax" + std::to_string(loop_vars.size()), axis_range->extent.dtype());
    loop_vars.push_back(loop_var);
    iter_values.push_back(cache_full_region ? (axis_range->min + loop_var) : loop_var);
  }
  // block variables
  Array<IterVar> block_vars;
  // block access region for read/write buffers
  Region read_access_region;
  Region write_access_region;
  // indices used in block body
  Array<PrimExpr> read_access_indices;
  Array<PrimExpr> write_access_indices;
  // Create block vars, block's accessed region and accessing indices
  for (int i = 0; i < static_cast<int>(cache_region->buffer->shape.size()); ++i) {
    Range axis_range = cache_region->region[i];
    Var var("v" + std::to_string(read_access_indices.size()), axis_range->extent.dtype());
    if (cache_full_region) {
      PrimExpr dim = cache_region->buffer->shape[i];
      block_vars.push_back(IterVar(/*dom=*/Range::FromMinExtent(make_zero(dim->dtype), dim),
                                   /*var=*/var,
                                   /*IterVarType=*/kDataPar));
      read_access_indices.push_back(var);
      write_access_indices.push_back(var);
      read_access_region.push_back(Range::FromMinExtent(var, make_const(var.dtype(), 1)));
      write_access_region.push_back(Range::FromMinExtent(var, make_const(var.dtype(), 1)));
    } else {
      block_vars.push_back(IterVar(
          /*dom=*/Range::FromMinExtent(make_zero(axis_range->extent.dtype()), axis_range->extent),
          /*var=*/var,
          /*IterVarType=*/kDataPar));
      if (cache_region->buffer.same_as(info->read_buffer)) {
        // cache_read
        read_access_indices.push_back(axis_range->min + var);
        read_access_region.push_back(
            Range::FromMinExtent(axis_range->min + var, make_const(var.dtype(), 1)));
        write_access_indices.push_back(var);
        write_access_region.push_back(Range::FromMinExtent(var, make_const(var.dtype(), 1)));
      } else {
        // cache_write
        write_access_indices.push_back(axis_range->min + var);
        write_access_region.push_back(
            Range::FromMinExtent(axis_range->min + var, make_const(var.dtype(), 1)));
        read_access_indices.push_back(var);
        read_access_region.push_back(Range::FromMinExtent(var, make_const(var.dtype(), 1)));
      }
    }
  }

  // Create the body block:
  //   reads = [read_buffer[access_region]]
  //   writes = [write_buffer[access_region]]
  //     write_buffer[access_indices] = read_buffer[access_indices]
  Block block(
      /*iter_vars=*/std::move(block_vars),
      /*reads=*/{BufferRegion(info->read_buffer, read_access_region)},
      /*writes=*/{BufferRegion(info->write_buffer, write_access_region)},
      /*name_hint=*/cache_region->buffer->name + "_" + storage_scope,
      /*body=*/
      BufferStore(info->write_buffer, BufferLoad(info->read_buffer, read_access_indices),
                  write_access_indices),
      /*init=*/NullOpt,
      /*alloc_buffers=*/{},
      /*match_buffers=*/{},
      /*annotations=*/{});
  // Create the block realize node
  Stmt body = BlockRealize(/*values=*/iter_values,
                           /*predicate=*/const_true(),
                           /*block=*/block);
  // Create surrounding loops
  for (size_t i = loop_vars.size(); i >= 1; --i) {
    body = For(/*loop_var=*/loop_vars[i - 1],
               /*min=*/0,
               /*extent=*/cache_region->region[i - 1]->extent,
               /*kind=*/ForKind::kSerial,
               /*body=*/body);
  }
  info->cache_stage = std::move(body);
  return block;
}

/*!
 * \brief Create the reindex block and generate the corresponding outer loops.
 * \details The reindex block is a data copy block between the reindex buffer (the intermediate
 * buffer), and the target buffer.
    If buffer_index_type == kWrite, copy from the reindex buffer to the target buffer.
    If buffer_index_type == kRead, copy from the target buffer to the reindex buffer.
    The reindex block has the same block iters and the surrounding loops as the input block.
 However, if a block iter is not used in the indices of the target buffer being reindexed, the
 domain of the block iter, and the corresponding outer loop, will become constant value one, making
 it a trivial iter.
 * \param block The block to be reindexed
 * \param info The cache info
 * \param covered The set of block iter vars covered in the buffer access indices
 * \param original_indices The original buffer access indices
 * \param buffer_index The index of the target buffer
 * \param buffer_index_type The type of buffer index
 * \return The reindex block.
 */
Block MakeReIndexStage(const Block& block, CacheStageInfo* info,
                       const std::unordered_set<Var>& covered,
                       const Array<PrimExpr>& original_indices, int buffer_index,
                       BufferIndexType buffer_index_type) {
  // iters of the reindex block
  Array<IterVar> new_block_iters;
  // the substitution map from the original block iter to the iters of the reindex block
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectEqual> block_var_replace_map;
  // indices to access the reindex buffer and the target buffer
  Array<PrimExpr> reindex_indices, target_indices;

  // Step 1: Create block iters, access regions of the reindex block, and accessing indices to the
  // reindex buffer.
  std::unordered_set<int> skipped_block_iters;
  for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
    const IterVar& iter = block->iter_vars[i];
    Var var("v" + std::to_string(new_block_iters.size()), iter->var->dtype);
    bool used = covered.count(iter->var);
    if (used) {
      new_block_iters.push_back(IterVar(/*dom=*/iter->dom,
                                        /*var=*/var,
                                        /*IterVarType=*/kDataPar));
    } else {
      skipped_block_iters.insert(i);
    }
    if (used) {
      reindex_indices.push_back(var);
    }
    block_var_replace_map[iter->var] = var;
  }

  // Step 2: Replace the original block iters with the new block iters
  for (const PrimExpr& index : original_indices) {
    target_indices.push_back(Substitute(index, block_var_replace_map));
  }

  // Step 3: Create the reindex block

  // The src and the dst region and indices of the data copy
  Region src_region{nullptr};
  Region dst_region{nullptr};
  Array<PrimExpr> src_indices{nullptr};
  Array<PrimExpr> dst_indices{nullptr};

  if (buffer_index_type == BufferIndexType::kWrite) {
    src_indices = reindex_indices;
    dst_indices = target_indices;
  } else {
    src_indices = target_indices;
    dst_indices = reindex_indices;
  }

  // Create the body block
  Block new_block(
      /*iter_vars=*/new_block_iters,
      /*reads=*/{BufferRegion::FromPoint(info->read_buffer, src_indices)},
      /*writes=*/{BufferRegion::FromPoint(info->write_buffer, dst_indices)},
      /*name_hint=*/info->write_buffer->name + "_reindex",
      /*body=*/
      BufferStore(info->write_buffer, BufferLoad(info->read_buffer, src_indices), dst_indices));

  // Step 4: Create surrounding loops

  // Create loop vars and bindings for block iters
  std::vector<Var> loop_vars;         // loop variables
  std::vector<PrimExpr> iter_values;  // bindings in block realize
  for (int i = 0; i < static_cast<int>(block->iter_vars.size()); ++i) {
    if (skipped_block_iters.count(i)) {
      continue;
    }
    Var loop_var("ax" + std::to_string(loop_vars.size()), block->iter_vars[i]->var->dtype);
    loop_vars.push_back(loop_var);
    iter_values.push_back(loop_var);
  }

  // Create the block realize node
  Stmt body = BlockRealize(/*values=*/iter_values,
                           /*predicate=*/const_true(),
                           /*block=*/new_block);

  // Create the chain of loops
  for (int i = static_cast<int>(new_block_iters.size()) - 1; i >= 0; --i) {
    body = For(/*loop_var=*/loop_vars[i],
               /*min=*/new_block_iters[i]->dom->min,
               /*extent=*/new_block_iters[i]->dom->extent,
               /*kind=*/ForKind::kSerial,
               /*body=*/std::move(body));
  }
  // Update cache info, which will be used in the later rewriting.
  info->cache_stage = std::move(body);
  return new_block;
}

/*!
 * \brief Recalculate the `affine_binding` flag of a specific block
 * \param block_sref The sref to the specific block
 */
bool CalculateAffineFlag(const ScheduleState& self, const StmtSRef& block_sref) {
  if (block_sref->parent == nullptr) {
    return true;
  }
  arith::Analyzer analyzer;
  StmtSRef parent_sref = GetRef<StmtSRef>(block_sref->parent);
  return IsAffineBinding(/*realize=*/GetBlockRealize(self, block_sref),
                         /*loop_var_ranges=*/LoopDomainOfSRefTreePath(parent_sref),
                         /*analyzer=*/&analyzer);
}

/*!
 * \brief Insert the cache_read/cache_write stage into the specific position
 * \param stmt A sequence of statements or a single statement that the new stage is inserted in
 * \param pos The position where the cache stage is inserted
 * \param stage The stage to be inserted
 * \return A SeqStmt, the result after insertion
 */
Stmt InsertCacheStage(const Stmt& stmt, int pos, const Stmt& stage) {
  std::vector<Stmt> nest;
  Stmt body = stmt;
  while (true) {
    if (auto opt = body.as<AllocateConst>()) {
      auto alloc = opt.value();
      body = alloc->body;
      alloc.CopyOnWrite()->body = Evaluate(0);
      nest.push_back(alloc);
    } else if (auto opt = body.as<DeclBuffer>()) {
      auto decl_buffer = opt.value();
      body = decl_buffer->body;
      decl_buffer.CopyOnWrite()->body = Evaluate(0);
      nest.push_back(decl_buffer);
    } else {
      break;
    }
  }

  if (const auto* seq_stmt = body.as<SeqStmtNode>()) {
    Array<Stmt> seq = seq_stmt->seq;
    ICHECK_LE(pos, seq.size()) << "Cannot insert at position " << pos << " into sequence of length "
                               << seq.size();
    seq.insert(seq.begin() + pos, stage);
    body = SeqStmt(seq);
  } else if (pos == 0) {
    body = SeqStmt({stage, body});
  } else if (pos == 1) {
    body = SeqStmt({body, stage});
  } else {
    LOG(FATAL) << "Cannot insert at position " << pos
               << ".  When inserting adjacent to non-SeqStmt, "
               << "only positions 0 and 1 are valid.";
  }

  body = MergeNest(nest, body);

  return body;
}

/*!
 * \brief Get the only writer block of the input buffer in a given scope block.
 * \param self The state of the schedule
 * \param scope_sref The scope block where the write is considered
 * \param buffer The queried buffer
 * \return The sref of the only writer of the input buffer in the given scope,
 *         or `NullOpt` if no block writes it in the scope.
 * \throw NotSingleWriteBlock if there are more than one interested block.
 */
Optional<StmtSRef> GetOnlyWriteBlock(ScheduleState self, const StmtSRef& scope_sref,
                                     const Buffer& buffer) {
  BlockScope scope = self->GetBlockScope(scope_sref);
  auto it = scope->buffer_writers.find(buffer);
  if (it == scope->buffer_writers.end()) {
    return NullOpt;
  } else {
    const Array<StmtSRef>& block_srefs = it->second;
    ICHECK(!block_srefs.empty());
    if (block_srefs.size() > 1) {
      throw NotSingleWriteBlock(self->mod, buffer, block_srefs);
    }
    return block_srefs[0];
  }
}

/*!
 * \brief Check if all the consumer blocks of the given buffer in the given
 *        block scope are the children block of the given target stmt.
 * \param self The state of the schedule .
 * \param buffer The buffer whose consumer blocks are to be check.
 * \param scope_sref The scope block of the check.
 * \param stmt_sref The target stmt
 * \return A boolean indicating if all the consumer blocks of the input buffer
 *         meet the requirement.
 */
bool AllConsumersUnderStmt(ScheduleState self, Buffer buffer, StmtSRef scope_sref,
                           StmtSRef stmt_sref) {
  // Collect all children blocks of the target stmt.
  std::unordered_set<const BlockNode*> blocks_under_target;
  for (const StmtSRef& block_sref : GetChildBlocks(self, stmt_sref)) {
    const auto* block = block_sref->StmtAs<BlockNode>();
    ICHECK(block != nullptr);
    blocks_under_target.insert(block);
  }

  // For each block in the scope, if it is a consumer of the
  // input buffer, check if it is also a child block of the
  // target stmt.
  for (const StmtSRef& block_sref : GetChildBlocks(self, scope_sref)) {
    const auto* block = block_sref->StmtAs<BlockNode>();
    ICHECK(block != nullptr);
    if (GetBufferRegionFromBuffer(block->reads, buffer).defined()) {
      if (blocks_under_target.find(block) == blocks_under_target.end()) {
        return false;
      }
    }
  }
  return true;
}

/*!
 * \brief Get the buffer region under the sref tree path [dom_low_inclusive, dom_high_exclusive)
 * \param self The state of the schedule.
 * \param buffer_region The buffer region to be analyzed.
 * \param block_sref The sref of the block related to the region.
 * \param dom_low_inclusive The lowest node in the sref tree path.
 * \param dom_high_exclusive The highest node in the sref tree path.
 * \return The relaxed buffer region.
 */
BufferRegion RelaxBufferRegion(ScheduleState self, const BufferRegion& buffer_region,
                               const StmtSRef& block_sref, const StmtSRef& dom_low_inclusive,
                               const StmtSRef& dom_high_exclusive) {
  BlockRealize realize = GetBlockRealize(self, block_sref);
  Map<Var, PrimExpr> binding = GetBindings(realize);
  const Buffer& buffer = buffer_region->buffer;
  arith::Analyzer analyzer;
  BufferRegion subst_region = BufferRegion(buffer, Substitute(buffer_region->region, binding));
  Array<arith::IntSet> int_sets = AnalyzeRegionUpperBound(
      /*region=*/subst_region,
      /*predicate=*/realize->predicate,
      /*dom_low_inclusive=*/dom_low_inclusive,
      /*dom_high_exclusive=*/dom_high_exclusive,
      /*analyzer=*/&analyzer);
  ICHECK_EQ(buffer_region->region.size(), int_sets.size());

  Region region;
  region.reserve(int_sets.size());
  for (size_t i = 0; i < int_sets.size(); ++i) {
    region.push_back(int_sets[i].CoverRange(Range::FromMinExtent(0, buffer->shape[i])));
  }
  return BufferRegion(buffer, region);
}

/*! \brief Detect the insertion position of the new cache stage */
class CacheLocDetector : public StmtVisitor {
 public:
  /*!
   * \brief Detect the insertion position of the cache stage, and write the position into the
   * CacheStageInfo
   * \param self The state of the schedule
   * \param block_sref The sref of the unique writer block of the buffer being applied cache_read or
   * cache_write
   * \param scope_sref The sref of the scope block of the cached block
   * \param info The cache stage info.
   */
  template <bool is_cache_read>
  static void Detect(const ScheduleState& self, const StmtSRef& block_sref,
                     const StmtSRef& scope_sref, CacheStageInfo* info) {
    std::vector<StmtSRef> related_blocks;
    // If consumer is specified, skip detecting the others
    if (is_cache_read) {
      if (info->consumer_blocks.size() > 0) {
        for (StmtSRef consumer : info->consumer_blocks) {
          related_blocks.emplace_back(consumer);
        }
      } else {
        for (const Dependency& def : self->GetBlockScope(scope_sref)->GetDepsBySrc(block_sref)) {
          if (def->kind == DepKind::kRAW) {
            related_blocks.push_back(def->dst);
          }
        }
      }
    } else {
      for (const Dependency& def : self->GetBlockScope(scope_sref)->GetDepsBySrc(block_sref)) {
        if (def->kind == DepKind::kRAW) {
          if (info->consumer_blocks.count(def->dst)) {
            continue;
          }
          related_blocks.push_back(def->dst);
        }
      }
    }

    if (!related_blocks.empty()) {
      CacheLocDetector detector(self, block_sref, scope_sref, related_blocks);
      detector(GetRef<Stmt>(scope_sref->stmt));
      info->loc_sref = detector.loc_sref_;
      info->loc_pos = detector.loc_pos_;
    } else {
      info->loc_sref = scope_sref;

      auto block_body = scope_sref->StmtAs<BlockNode>()->body;
      // Find the SeqStmtNode within (potentially nested) AllocateConstNodes
      while (true) {
        if (auto* ptr = block_body.as<AllocateConstNode>()) {
          block_body = ptr->body;
        } else if (auto* ptr = block_body.as<DeclBufferNode>()) {
          block_body = ptr->body;
        } else {
          break;
        }
      }
      const auto* body = block_body.as<SeqStmtNode>();
      info->loc_pos = body == nullptr ? 1 : body->size();
    }
  }

 private:
  /*!
   * \brief Constructor
   * \param self The state of the schedule
   * \param block_sref The sref of the unique writer block of the buffer being applied cache_read or
   * cache_write
   * \param scope_sref The sref of the scope block of the cached block
   * \param related_blocks Producer blocks for cache_write, or consumer blocks for cache_read
   */
  CacheLocDetector(const ScheduleState self, const StmtSRef& block_sref, const StmtSRef& scope_sref,
                   const std::vector<StmtSRef>& related_blocks)
      : self_(self),
        block_sref_(block_sref),
        scope_sref_(scope_sref),
        related_blocks_(related_blocks) {}

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    bool previous_visited_block = visited_block_;
    visited_block_ = false;

    for (size_t i = 0; i < seq_stmt->size(); ++i) {
      if (loc_pos_ != -1) {
        break;
      }
      VisitStmt(seq_stmt->seq[i]);
      // `pos` can be assigned only once when we visited `block_sref`
      if (visited_block_ && visited_related_ && loc_pos_ == -1) {
        // The offset of insert position from the block
        loc_pos_ = i;
        break;
      } else if (visited_related_) {
        // If meet the target consumer, stop searching
        break;
      }
    }
    visited_block_ = visited_block_ || previous_visited_block;
  }

  void VisitStmt_(const BlockNode* block) final {
    // Only visit the current scope under buffer writer's parent block
    if (block == scope_sref_->stmt) {
      // The block visited is the current parent scope
      StmtVisitor::VisitStmt_(block);
      // Handling cases when insert outside any loop or cache_read for input buffer
      if (visited_related_ && !loc_sref_.defined()) {
        loc_sref_ = self_->stmt2ref.at(block);
        // Handling cache_read for input buffer
        if (visited_block_ == false && loc_pos_ == -1) {
          loc_pos_ = 0;
        }
      }
      return;
    }
    // Update `visited_block`
    if (block_sref_->stmt == block) {
      visited_block_ = true;
      return;
    }
    // Update `visited_related`
    for (const StmtSRef& related_block : related_blocks_) {
      if (related_block->stmt == block) {
        visited_related_ = true;
        return;
      }
    }
  }

  void VisitStmt_(const ForNode* loop) final {
    StmtVisitor::VisitStmt_(loop);
    if (visited_block_ && visited_related_ && !loc_sref_.defined() && loc_pos_ != -1) {
      loc_sref_ = self_->stmt2ref.at(loop);
    }
  }

 private:
  /*! \brief The schedule class */
  const ScheduleState self_;
  /*! \brief The dominate block which write the buffer */
  const StmtSRef& block_sref_;
  /*! \brief The parent scope of the dominate block */
  const StmtSRef& scope_sref_;
  /*! \brief Producer blocks for cache_write and consumer blocks for cache_read */
  const std::vector<StmtSRef>& related_blocks_;
  /*! \brief The flag whether we have visited the dominate block */
  bool visited_block_{false};
  /*! \brief The flag whether we have visited at least one related blocks */
  bool visited_related_{false};
  /*! \brief The AST node whose body is where the cache stage should be inserted */
  StmtSRef loc_sref_{nullptr};
  /*! \brief The index to insert the cache_read/cache_write stage */
  int loc_pos_{-1};
};

/*! \brief Detect the insertion position of the new cache stage */
class CacheInplaceLocDetector : public StmtVisitor {
 public:
  /*!
   * \brief Detect the insertion position of the cache stage, and write the position into the
   * CacheStageInfo
   * \param self The state of the schedule
   * \param block_sref The sref of the unique block of the buffer being applied cache_inplace
   * \param scope_sref The sref of the scope block of the cached block
   * \param info The cache stage info.
   */
  static void Detect(const ScheduleState& self, const StmtSRef& block_sref,
                     const StmtSRef& scope_sref, CacheStageInfo* info) {
    CacheInplaceLocDetector detector(self, block_sref, scope_sref);
    detector(GetRef<Stmt>(scope_sref->stmt));
    info->loc_sref = detector.loc_sref_;
    info->loc_pos = detector.loc_pos_;
  }

 private:
  /*!
   * \brief Constructor
   * \param self The state of the schedule
   * \param block_sref The sref of the unique writer block of the buffer being applied cache_inplace
   * \param scope_sref The sref of the scope block of the cached block
   */
  CacheInplaceLocDetector(const ScheduleState self, const StmtSRef& block_sref,
                          const StmtSRef& scope_sref)
      : self_(self), block_sref_(block_sref), scope_sref_(scope_sref) {}

  void VisitStmt_(const SeqStmtNode* seq_stmt) final {
    for (size_t i = 0; i < seq_stmt->size(); ++i) {
      if (loc_pos_ != -1) {
        break;
      }
      VisitStmt(seq_stmt->seq[i]);
      // `pos` can be assigned only once when we visited `block_sref`
      if (visited_block_ && loc_pos_ == -1) {
        // The offset of insert position from the block
        loc_pos_ = i;
        return;
      }
    }
  }

  void VisitStmt_(const BlockNode* block) final {
    // Only visit the current scope under buffer writer's parent block
    if (block == scope_sref_->stmt) {
      // The block visited is the current parent scope
      StmtVisitor::VisitStmt_(block);
      // Handling cases when insert outside any loop
      if (visited_block_ && !loc_sref_.defined()) {
        loc_sref_ = self_->stmt2ref.at(block);
        // Handling for input buffer
        if (loc_pos_ == -1) {
          loc_pos_ = 0;
        }
      }
    } else if (block_sref_->stmt == block) {
      visited_block_ = true;
    }
  }

  void VisitStmt_(const ForNode* loop) final {
    StmtVisitor::VisitStmt_(loop);
    if (visited_block_ && !loc_sref_.defined()) {
      loc_sref_ = self_->stmt2ref.at(loop);
      if (loc_pos_ == -1) {
        loc_pos_ = 0;
      }
    }
  }

 private:
  /*! \brief The schedule class */
  const ScheduleState self_;
  /*! \brief The dominate block which write the buffer */
  const StmtSRef& block_sref_;
  /*! \brief The parent scope of the dominate block */
  const StmtSRef& scope_sref_;
  /*! \brief The flag whether we have visited the target block */
  bool visited_block_{false};
  /*! \brief The AST node whose body is where the cache stage should be inserted */
  StmtSRef loc_sref_{nullptr};
  /*! \brief The index to insert the cache_read/cache_write stage */
  int loc_pos_{-1};
};

class ReindexCacheReadRewriter;

/*! \brief Mutator for CacheRead. */
class CacheReadRewriter : public StmtExprMutator {
 public:
  /*!
   * \brief Rewrite the AST and add a cache_read stage with the information provided
   * \param scope_sref The parent scope of this mutation
   * \param info The cache stage information
   * \param cache_full_region A boolean indicating if the cache buffer is allocated with
   *        full region or compact region.
   * \return The new AST rooting at the original parent scope
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, CacheStageInfo* info,
                      bool cache_full_region = true) {
    CacheReadRewriter rewriter(scope_sref, info, cache_full_region);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }

 private:
  explicit CacheReadRewriter(const StmtSRef& scope_sref, CacheStageInfo* info,
                             bool cache_full_region = true)
      : scope_sref_(scope_sref), info_(info), cache_full_region_(cache_full_region) {
    auto update_region = [this](const Region& region, const Region& offset) -> Region {
      ICHECK_EQ(region.size(), offset.size());
      std::vector<Range> ret;
      for (size_t i = 0; i < region.size(); ++i) {
        ret.push_back(Range::FromMinExtent(ana_.Simplify(region[i]->min - offset[i]->min),
                                           region[i]->extent));
      }
      return ret;
    };

    update_access_regions = [this, update_region](Array<BufferRegion> regions) {
      if (cache_full_region_) {
        return ReplaceBuffer(std::move(regions), info_->read_buffer, info_->write_buffer);
      }

      Array<BufferRegion> ret;
      for (const BufferRegion& region : regions) {
        if (region->buffer.same_as(info_->read_buffer)) {
          ret.push_back(BufferRegion(info_->write_buffer,
                                     update_region(region->region, info_->cache_region->region)));
        } else {
          ret.push_back(region);
        }
      }
      return ret;
    };
    update_match_buffers = [this, update_region](Array<MatchBufferRegion> match_buffers) {
      if (cache_full_region_) {
        return ReplaceBuffer(std::move(match_buffers), info_->read_buffer, info_->write_buffer);
      }

      Array<MatchBufferRegion> ret;
      for (const MatchBufferRegion& match_buffer : match_buffers) {
        if (match_buffer->source->buffer.same_as(info_->read_buffer)) {
          ret.push_back(MatchBufferRegion(
              match_buffer->buffer,
              BufferRegion(info_->write_buffer, update_region(match_buffer->source->region,
                                                              info_->cache_region->region))));
        } else {
          ret.push_back(match_buffer);
        }
      }
      return ret;
    };
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    Stmt stmt = StmtMutator::VisitStmt_(loop);
    // Check the insertion point
    if (loop == info_->loc_sref->stmt) {
      // Insert cache stage into the loop if it is the right place
      ObjectPtr<ForNode> n = make_object<ForNode>(*stmt.as<ForNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  Stmt VisitStmt_(const BlockNode* block) override {
    Block old_stmt = GetRef<Block>(block);
    // Check if this block is one of the specified consumers.
    // If no consumer blocks are specified, all blocks should be considered consumers.
    bool is_consumer = info_->consumer_blocks.empty();
    // Otherwise check if this is one of the specified blocks.
    for (StmtSRef consumer_sref : info_->consumer_blocks) {
      const BlockNode* consumer_node = TVM_SREF_TO_BLOCK(consumer_sref);
      Block consumer_block = GetRef<Block>(consumer_node);
      if (old_stmt.same_as(consumer_block)) {
        is_consumer = true;
      }
    }
    // Keep track of this blocks status. We'll use this when rewriting loads.
    current_block_consumes = is_consumer;
    // We don't mutate the block which generates info->read_buffer.
    if (block != scope_sref_->stmt &&
        GetBufferRegionFromBuffer(block->writes, info_->read_buffer).defined()) {
      return std::move(old_stmt);
    }
    // Mutate the body
    Block stmt = Downcast<Block>(StmtMutator::VisitStmt_(block));
    // Check the insertion point
    if (block == info_->loc_sref->stmt) {
      // Insert cache stage into the block if it is the right place
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Block(n);
    }
    // Check if it is the block corresponding to the parent scope
    if (block == scope_sref_->stmt) {
      // If so, put buffer allocation on the parent scope
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      // In cache_inplace case, alloc_buffer may be already exits.
      if (info_->alloc.defined()) {
        n->alloc_buffers.push_back(info_->alloc.value());
        stmt = Block(n);
      }
    } else {
      // Otherwise, update read regions and match_buffers
      // Only make this change if the block is one of the specified consumers.
      if (is_consumer) {
        // Use the updated block stmt
        Array<BufferRegion> reads = update_access_regions(stmt->reads);
        Array<MatchBufferRegion> match_buffers = update_match_buffers(stmt->match_buffers);
        if (!reads.same_as(stmt->reads) || !match_buffers.same_as(stmt->match_buffers)) {
          ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
          n->reads = std::move(reads);
          n->match_buffers = std::move(match_buffers);
          stmt = Block(n);
        }
      }
    }
    info_->block_reuse.Set(old_stmt, stmt);
    return std::move(stmt);
  }

  Array<PrimExpr> RewriteIndices(const Array<PrimExpr>& indices) {
    std::vector<PrimExpr> ret;
    for (size_t i = 0; i < indices.size(); ++i) {
      ret.push_back(ana_.Simplify(indices[i] - info_->cache_region->region[i]->min));
    }
    return ret;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) override {
    if (load->buffer.same_as(info_->read_buffer) && current_block_consumes) {
      ObjectPtr<BufferLoadNode> n = make_object<BufferLoadNode>(*load);
      n->buffer = info_->write_buffer;
      if (!cache_full_region_) {
        n->indices = std::move(RewriteIndices(load->indices));
      }
      return PrimExpr(n);
    }
    return ExprMutator::VisitExpr_(load);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (op == info_->read_buffer->data.get()) {
      return info_->write_buffer->data;
    }
    return GetRef<PrimExpr>(op);
  }

 private:
  /*! \brief The parent scope of the insertion */
  const StmtSRef& scope_sref_;
  /*! \brief The info for inserting cache stage */
  CacheStageInfo* info_;
  /*! \brief Whether the most recently visited block is a specified consumer. */
  bool current_block_consumes;
  /*! \brief function to update read/write region of block being cache read.*/
  std::function<Array<BufferRegion>(Array<BufferRegion>)> update_access_regions;
  /*! \brief function to update match buffers of block being cache read.*/
  std::function<Array<MatchBufferRegion>(Array<MatchBufferRegion>)> update_match_buffers;
  /*!
   * \brief A boolean indicating if the cache buffer is allocated with
   *        full region or compact region.
   */
  bool cache_full_region_;
  /*! \brief Arithmetic analyzer. */
  arith::Analyzer ana_;

  friend ReindexCacheReadRewriter;
};

/*! \brief Mutator for ReindexCacheRead. */
class ReindexCacheReadRewriter : public CacheReadRewriter {
 public:
  /*!
   * \brief Rewrite the AST and add a cache_read stage with the information provided.
   * \param scope_sref The parent scope of this mutation.
   * \param info The cache stage information.
   * \return The new AST rooting at the original parent scope.
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, ReindexCacheStageInfo* info) {
    ReindexCacheReadRewriter rewriter(scope_sref, info);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }

 private:
  explicit ReindexCacheReadRewriter(const StmtSRef& scope_sref, ReindexCacheStageInfo* info)
      : CacheReadRewriter(scope_sref, info) {
    new_indices_ = info->indices;
    update_access_regions = [&](Array<BufferRegion> reads) {
      Array<BufferRegion> new_reads;
      for (const BufferRegion& buf_region : reads) {
        if (buf_region->buffer.same_as(info_->read_buffer)) {
          Array<Range> region;
          for (const PrimExpr index : new_indices_) {
            region.push_back(Range::FromMinExtent(index, Integer(1)));
          }
          new_reads.push_back(BufferRegion(info_->write_buffer, region));
        } else {
          new_reads.push_back(buf_region);
        }
      }
      return new_reads;
    };
    update_match_buffers = [&](const Array<MatchBufferRegion> match_buffers) {
      Array<MatchBufferRegion> new_match_buffers;
      for (const MatchBufferRegion& match_buffer_region : match_buffers) {
        BufferRegion source = match_buffer_region->source;
        if (source->buffer.same_as(info_->read_buffer)) {
          Array<Range> region;
          for (const PrimExpr index : new_indices_) {
            region.push_back(Range::FromMinExtent(index, Integer(1)));
          }
          new_match_buffers.push_back(MatchBufferRegion(match_buffer_region->buffer,
                                                        BufferRegion(info_->write_buffer, region)));
        } else {
          new_match_buffers.push_back(match_buffer_region);
        }
      }
      return new_match_buffers;
    };
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) final {
    if (load->buffer.same_as(info_->read_buffer) && current_block_consumes) {
      ObjectPtr<BufferLoadNode> n = make_object<BufferLoadNode>(*load);
      n->buffer = info_->write_buffer;
      n->indices = new_indices_;
      return PrimExpr(n);
    }
    return ExprMutator::VisitExpr_(load);
  }

  /*! \brief The indices to use for new buffer. */
  Array<PrimExpr> new_indices_;
};

class ReindexCacheWriteRewriter;

/*! \brief Mutator for CacheWrite */
class CacheWriteRewriter : public StmtExprMutator {
 public:
  /*!
   * \brief Rewrite the AST and add a cache_write stage with the information provided.
   * \param scope_sref The parent scope of this mutation.
   * \param writer_block_sref The only writer block in the scope.
   * \param info The cache stage information.
   * \param cache_full_region A boolean indicating if the cache buffer is allocated with
   *        full region or compact region.
   * \return The new AST rooting at the original parent scope.
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                      CacheStageInfo* info, bool cache_full_region = true) {
    CacheWriteRewriter rewriter(scope_sref, writer_block_sref, info, cache_full_region);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }

 private:
  explicit CacheWriteRewriter(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                              CacheStageInfo* info, bool cache_full_region = true)
      : scope_sref_(scope_sref),
        writer_block_sref_(writer_block_sref),
        info_(info),
        cache_full_region_(cache_full_region) {
    auto update_region = [this](const Region& region, const Region& offset) -> Region {
      ICHECK_EQ(region.size(), offset.size());
      std::vector<Range> ret;
      for (size_t i = 0; i < region.size(); ++i) {
        ret.push_back(Range::FromMinExtent(ana_.Simplify(region[i]->min - offset[i]->min),
                                           region[i]->extent));
      }
      return ret;
    };

    update_access_regions = [this, update_region](Array<BufferRegion> regions) {
      if (cache_full_region_) {
        return ReplaceBuffer(regions, info_->write_buffer, info_->read_buffer);
      }

      Array<BufferRegion> ret;
      for (const BufferRegion& region : regions) {
        if (region->buffer.same_as(info_->write_buffer)) {
          ret.push_back(BufferRegion(info_->read_buffer,
                                     update_region(region->region, info_->cache_region->region)));
        } else {
          ret.push_back(region);
        }
      }
      return ret;
    };
    update_match_buffers = [this, update_region](Array<MatchBufferRegion> match_buffers) {
      if (cache_full_region_) {
        return ReplaceBuffer(match_buffers, info_->write_buffer, info_->read_buffer);
      }

      Array<MatchBufferRegion> ret;
      for (const MatchBufferRegion& match_buffer : match_buffers) {
        if (match_buffer->source->buffer.same_as(info_->write_buffer)) {
          ret.push_back(MatchBufferRegion(
              match_buffer->buffer,
              BufferRegion(info_->read_buffer, update_region(match_buffer->source->region,
                                                             info_->cache_region->region))));
        } else {
          ret.push_back(match_buffer);
        }
      }
      return ret;
    };
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    Stmt stmt = StmtMutator::VisitStmt_(loop);
    // Check the insertion point
    if (loop == info_->loc_sref->stmt) {
      // Insert cache stage into the loop if it is the right place
      ObjectPtr<ForNode> n = make_object<ForNode>(*stmt.as<ForNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  Stmt VisitStmt_(const BlockNode* block) override {
    Block old_stmt = GetRef<Block>(block);

    // Check if this block is one of the specified cache consumers.
    // update the read buffer to the cache.
    for (StmtSRef consumer_sref : info_->consumer_blocks) {
      const BlockNode* consumer_node = TVM_SREF_TO_BLOCK(consumer_sref);
      Block consumer_block = GetRef<Block>(consumer_node);
      if (old_stmt.same_as(consumer_block)) {
        Array<BufferRegion> writes = update_access_regions(block->writes);
        Array<BufferRegion> reads = update_access_regions(block->reads);
        Array<MatchBufferRegion> match_buffers = update_match_buffers(block->match_buffers);
        if (!writes.same_as(block->writes) || !reads.same_as(block->reads) ||
            !match_buffers.same_as(block->match_buffers)) {
          auto n = CopyOnWrite(block);
          n->writes = std::move(writes);
          n->reads = std::move(reads);
          n->match_buffers = std::move(match_buffers);
          n->body = VisitStmt(block->body);
          Block new_consumer = Block(n);
          info_->block_reuse.Set(old_stmt, new_consumer);
          return std::move(new_consumer);
        }
        return std::move(old_stmt);
      }
    }

    // We only mutate the block which generates info->write_buffer
    if (block != writer_block_sref_->stmt && block != scope_sref_->stmt && !under_writer_block_) {
      return std::move(old_stmt);
    }

    // Mutate the body
    bool under_scope = under_writer_block_ || block == writer_block_sref_->stmt;
    std::swap(under_scope, under_writer_block_);
    Block stmt = Downcast<Block>(StmtMutator::VisitStmt_(block));
    std::swap(under_scope, under_writer_block_);

    // Find the insertion point
    if (block == info_->loc_sref->stmt) {
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Block(n);
    }
    // Put buffer allocation on the parent scope
    if (block == scope_sref_->stmt) {
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      // In cache_inplace case, alloc_buffer may be already exits.
      if (info_->alloc.defined()) {
        n->alloc_buffers.push_back(info_->alloc.value());
        stmt = Block(n);
      }
    } else {
      // Since cache_write changes the block, we need to update the buffer it writes
      auto writes = update_access_regions(block->writes);
      auto reads = update_access_regions(block->reads);
      auto match_buffers = update_match_buffers(block->match_buffers);
      if (!writes.same_as(block->writes) || !reads.same_as(block->reads) ||
          !match_buffers.same_as(block->match_buffers)) {
        ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
        n->writes = std::move(writes);
        n->reads = std::move(reads);
        n->match_buffers = std::move(match_buffers);
        stmt = Block(n);
      }
    }
    info_->block_reuse.Set(old_stmt, stmt);
    return std::move(stmt);
  }

  Array<PrimExpr> RewriteIndices(const Array<PrimExpr>& indices) {
    std::vector<PrimExpr> ret;
    for (size_t i = 0; i < indices.size(); ++i) {
      ret.push_back(ana_.Simplify(indices[i] - info_->cache_region->region[i]->min));
    }
    return ret;
  }

  Stmt VisitStmt_(const BufferStoreNode* store) override {
    BufferStore stmt = Downcast<BufferStore>(StmtMutator::VisitStmt_(store));
    if (stmt->buffer.same_as(info_->write_buffer)) {
      auto n = CopyOnWrite(stmt.get());
      n->buffer = info_->read_buffer;
      if (!cache_full_region_) {
        n->indices = std::move(RewriteIndices(n->indices));
      }
      return Stmt(n);
    } else {
      return std::move(stmt);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) override {
    if (load->buffer.same_as(info_->write_buffer)) {
      ObjectPtr<BufferLoadNode> n = make_object<BufferLoadNode>(*load);
      n->buffer = info_->read_buffer;
      if (!cache_full_region_) {
        n->indices = std::move(RewriteIndices(n->indices));
      }
      return PrimExpr(n);
    }
    return ExprMutator::VisitExpr_(load);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (op == info_->write_buffer->data.get()) {
      return info_->read_buffer->data;
    }
    return GetRef<PrimExpr>(op);
  }

 private:
  /*! \brief The parent scope of the insertion. */
  const StmtSRef& scope_sref_;
  /*! \brief The parent scope of the insertion. */
  const StmtSRef& writer_block_sref_;
  /*! \brief The info for inserting cache stage. */
  CacheStageInfo* info_;
  /*! \brief Whether the current node is under the given block. */
  bool under_writer_block_{false};
  /*! \brief function to update read/write region of block being cache write.*/
  std::function<Array<BufferRegion>(Array<BufferRegion>)> update_access_regions;
  /*! \brief function to update match buffers of block being cache write.*/
  std::function<Array<MatchBufferRegion>(Array<MatchBufferRegion>)> update_match_buffers;
  /*!
   * \brief A boolean indicating if the cache buffer is allocated with
   *        full region or compact region.
   */
  bool cache_full_region_;
  /*! \brief Arithmetic analyzer. */
  arith::Analyzer ana_;

  friend ReindexCacheWriteRewriter;
};

/*! \brief Mutator for ReindexCacheWrite. */
class ReindexCacheWriteRewriter : public CacheWriteRewriter {
 public:
  /*!
   * \brief Rewrite the AST and add a cache_write stage with the information provided.
   * \param scope_sref The parent scope of this mutation.
   * \param writer_block_sref The only writer block in the scope.
   * \param info The cache stage information.
   * \return The new AST rooting at the original parent scope.
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                      ReindexCacheStageInfo* info) {
    ReindexCacheWriteRewriter rewriter(scope_sref, writer_block_sref, info);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }

 private:
  explicit ReindexCacheWriteRewriter(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                                     ReindexCacheStageInfo* info)
      : CacheWriteRewriter(scope_sref, writer_block_sref, info) {
    new_indices_ = info->indices;
    update_access_regions = [&](Array<BufferRegion> reads) {
      Array<BufferRegion> new_reads;
      for (const BufferRegion& buf_region : reads) {
        if (buf_region->buffer.same_as(info_->write_buffer)) {
          Array<Range> region;
          for (const PrimExpr index : new_indices_) {
            region.push_back(Range::FromMinExtent(index, Integer(1)));
          }
          new_reads.push_back(BufferRegion(info_->read_buffer, region));
        } else {
          new_reads.push_back(buf_region);
        }
      }
      return new_reads;
    };
    update_match_buffers = [&](const Array<MatchBufferRegion> match_buffers) {
      Array<MatchBufferRegion> new_match_buffers;
      for (const MatchBufferRegion& match_buffer_region : match_buffers) {
        BufferRegion source = match_buffer_region->source;
        if (source->buffer.same_as(info_->write_buffer)) {
          Array<Range> region;
          for (const PrimExpr index : new_indices_) {
            region.push_back(Range::FromMinExtent(index, Integer(1)));
          }
          new_match_buffers.push_back(MatchBufferRegion(match_buffer_region->buffer,
                                                        BufferRegion(info_->read_buffer, region)));
        } else {
          new_match_buffers.push_back(match_buffer_region);
        }
      }
      return new_match_buffers;
    };
  }

  Stmt VisitStmt_(const BufferStoreNode* store) final {
    BufferStore stmt = Downcast<BufferStore>(StmtMutator::VisitStmt_(store));
    if (stmt->buffer.same_as(info_->write_buffer)) {
      auto n = CopyOnWrite(stmt.get());
      n->buffer = info_->read_buffer;
      n->indices = new_indices_;
      return Stmt(n);
    } else {
      return std::move(stmt);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) final {
    if (load->buffer.same_as(info_->write_buffer)) {
      ObjectPtr<BufferLoadNode> n = make_object<BufferLoadNode>(*load);
      n->buffer = info_->read_buffer;
      n->indices = new_indices_;
      return PrimExpr(n);
    }
    return ExprMutator::VisitExpr_(load);
  }

  /*! \brief The indices to use for new buffer. */
  Array<PrimExpr> new_indices_;
};

/*!
 * \brief Create a new buffer by change the shape with block iters to be used as the reindex buffer
 * \param buffer The given buffer.
 * \param block_iters The block iters.
 * \param covered Set of block iter vars covered by the buffer access indices
 * \return The new buffer with target shape.
 */
Buffer CreateReindexBuffer(const Buffer& buffer, const Array<IterVar>& block_iters,
                           const std::unordered_set<Var>& covered) {
  ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*buffer.get());
  ObjectPtr<VarNode> new_var = make_object<VarNode>(*buffer->data.get());
  std::vector<PrimExpr> new_shape;
  std::vector<PrimExpr> new_strides;
  for (const auto& iter : block_iters) {
    if (covered.count(iter->var)) {
      new_shape.push_back(iter->dom->min + iter->dom->extent);
    }
  }
  new_strides.clear();
  new_buffer->shape = new_shape;
  new_buffer->strides = new_strides;
  new_buffer->data = buffer->data.copy_with_suffix("_reindex");
  new_buffer->name = buffer->name + "_reindex";
  return Buffer(new_buffer);
}

/*!
 * \brief The schedule error that the target is not a leaf block.
 */
class NotLeafBlockError : public ScheduleError {
 public:
  NotLeafBlockError(IRModule mod, Block block) : mod_(std::move(mod)), block_(std::move(block)) {}
  String FastErrorString() const final {
    return "ScheduleError: The target block is not a leaf block.";
  }

  String DetailRenderTemplate() const final { return "The target block {0} is not a leaf block."; }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
  IRModule mod_;
  Block block_;
};

/*! \brief The schedule error that the buffer access is invalid for reindex. */
class InvalidBufferAccessError : public ScheduleError {
 public:
  enum class ErrorKind {
    kNoAccess,         // buffer access not found
    kNonUniqueAccess,  // multiple buffer accesses with different indices
    kOpaqueAccess,     // opaque access to the buffer
  };

  InvalidBufferAccessError(IRModule mod, Buffer buffer, Block block, ErrorKind kind)
      : mod_(std::move(mod)), buffer_(std::move(buffer)), block_(std::move(block)), kind_(kind) {}
  String FastErrorString() const final {
    return "ScheduleError: The target buffer should be accessed via BufferLoad or BufferStore. The "
           "indices should be the same if there are multiple accesses to the target buffer.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The target buffer " << buffer_->name
       << " should be accessed in the leaf block {0} via BufferLoad or BufferStore. The indices "
          "should be the same if there are multiple accesses to the target buffer. ";
    if (kind_ == ErrorKind::kNoAccess) {
      os << "No buffer accesses found.";
    } else if (kind_ == ErrorKind::kNonUniqueAccess) {
      os << "Multiple buffer accesses have non-unique indices.";
    } else if (kind_ == ErrorKind::kOpaqueAccess) {
      os << "Opaque buffer accesses found.";
    }
    return os.str();
  }
  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  Buffer buffer_;
  Block block_;
  ErrorKind kind_;
};

/*! \brief Collect the related Load/Store to reindex */
class ReIndexCollector : public StmtExprVisitor {
 public:
  static Array<PrimExpr> Collect(const IRModule& mod, const Buffer& buffer, const Block& block) {
    ReIndexCollector collector(mod, buffer, block);
    collector(block->body);
    if (!collector.buffer_access_indices_.defined()) {
      throw InvalidBufferAccessError(mod, buffer, block,
                                     InvalidBufferAccessError::ErrorKind::kNoAccess);
    }
    return collector.buffer_access_indices_.value();
  }

 private:
  explicit ReIndexCollector(const IRModule& mod, const Buffer& buffer, const Block& block)
      : mod_(mod), buffer_(buffer), block_(block) {}

  void VisitExpr_(const BufferLoadNode* load) final {
    StmtExprVisitor::VisitExpr_(load);
    if (load->buffer.same_as(buffer_)) {
      CheckAndUpdateBufferAccessIndices(load->indices);
    }
  }

  void VisitStmt_(const BlockNode* block) final {
    // no sub-blocks under this block
    throw NotLeafBlockError(mod_, block_);
  }

  void VisitStmt_(const BufferStoreNode* store) final {
    StmtExprVisitor::VisitStmt_(store);
    if (store->buffer.same_as(buffer_)) {
      CheckAndUpdateBufferAccessIndices(store->indices);
    }
  }

  void CheckAndUpdateBufferAccessIndices(const Array<PrimExpr> indices) {
    if (!buffer_access_indices_.defined()) {
      buffer_access_indices_ = indices;
      return;
    } else if (!std::equal(buffer_access_indices_.value().begin(),
                           buffer_access_indices_.value().end(), indices.begin(), indices.end(),
                           ExprDeepEqual())) {
      throw InvalidBufferAccessError(mod_, buffer_, block_,
                                     InvalidBufferAccessError::ErrorKind::kNonUniqueAccess);
    }
  }

  void VisitExpr_(const VarNode* var) final {
    if (var == buffer_->data.get()) {
      throw InvalidBufferAccessError(mod_, buffer_, block_,
                                     InvalidBufferAccessError::ErrorKind::kOpaqueAccess);
    }
  }
  /*! \brief The IR module */
  IRModule mod_;
  /*! \brief The buffer to rewrite */
  Buffer buffer_;
  /*! \brief The block to visit */
  Block block_;
  /*! \brief The indices of buffer acess to rewrite */
  Optional<Array<PrimExpr>> buffer_access_indices_;
};

/*! \brief Mutator of ReIndex */
class ReIndexRewriter : public StmtExprMutator {
 public:
  static Stmt Rewrite(const StmtSRef& scope_sref, const StmtSRef& block_sref, CacheStageInfo* info,
                      const std::unordered_set<Var>& covered) {
    ReIndexRewriter rewriter(block_sref, info, covered);
    return rewriter(GetRef<Stmt>(scope_sref->stmt));
  }

 private:
  explicit ReIndexRewriter(const StmtSRef& block_sref, CacheStageInfo* info,
                           const std::unordered_set<Var>& covered)
      : block_sref_(block_sref), info_(info), covered_(covered) {
    new_buffer_ = info->alloc.value();
    old_buffer_ = info->read_buffer.same_as(new_buffer_) ? info->write_buffer : info->read_buffer;
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    Block old_stmt = GetRef<Block>(block);
    if (is_scope_) {
      is_scope_ = false;
      Block stmt = Downcast<Block>(StmtExprMutator::VisitStmt_(block));
      // Insert cache stage into the loop
      ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      n->alloc_buffers.push_back(info_->alloc.value());
      stmt = Block(n);
      info_->block_reuse.Set(old_stmt, stmt);
      return std::move(stmt);
    }

    // Visiting the blokc being reindexed
    if (block == block_sref_->stmt) {
      // Collect the updated indices and regions
      for (const IterVar& iter : block->iter_vars) {
        if (covered_.count(iter->var)) {
          indices_.push_back(iter->var);
          region_.push_back(Range::FromMinExtent(iter->var, IntImm(iter->var->dtype, 1)));
        }
      }
      Block stmt = Downcast<Block>(StmtExprMutator::VisitStmt_(block));
      // Update block reads/writes to use the intermediate reindex buffer
      auto writes =
          ReplaceBufferRegion(block->writes, old_buffer_, BufferRegion{new_buffer_, region_});
      auto reads =
          ReplaceBufferRegion(block->reads, old_buffer_, BufferRegion{new_buffer_, region_});
      auto match_buffers = ReplaceBufferRegion(block->match_buffers, old_buffer_,
                                               BufferRegion{new_buffer_, region_});
      if (!writes.same_as(block->writes) || !reads.same_as(block->reads) ||
          !match_buffers.same_as(block->match_buffers)) {
        ObjectPtr<BlockNode> n = make_object<BlockNode>(*stmt.as<BlockNode>());
        n->writes = std::move(writes);
        n->reads = std::move(reads);
        n->match_buffers = std::move(match_buffers);
        stmt = Block(n);
      }
      info_->block_reuse.Set(old_stmt, stmt);
      return std::move(stmt);
    }
    return std::move(old_stmt);
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    if (node->buffer.same_as(old_buffer_)) {
      auto* n = node.CopyOnWrite();
      n->buffer = new_buffer_;
      n->indices = indices_;
    }
    return node;
  }
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore buffer_store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    return VisitBufferAccess(std::move(buffer_store));
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad buffer_load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return VisitBufferAccess(std::move(buffer_load));
  }

 private:
  /*! \brief The parent scope of the insertion. */
  const StmtSRef& block_sref_;
  /*! \brief The info for inserting reindex stage. */
  CacheStageInfo* info_;
  /*! \brief Whether old block var is covered in the indices */
  const std::unordered_set<Var>& covered_;
  /*! \brief Whether the current block is scope block */
  bool is_scope_{true};
  /*! \brief The  buffer to be replaced */
  Buffer old_buffer_;
  /*! \brief The reindex buffer */
  Buffer new_buffer_;
  /*! \brief The new indices */
  Array<PrimExpr> indices_;
  /*! \brief The new region */
  Region region_;
};

void CheckRegionCover(const ScheduleState& self, StmtSRef scope_root, Buffer read_buffer) {
  class NotRegionCoverError : public ScheduleError {
   public:
    explicit NotRegionCoverError(IRModule mod, Block block) : mod_(mod), block_(block) {}
    IRModule mod() const final { return mod_; }
    String FastErrorString() const final {
      return "ScheduleError: The scope root's region cover is not complete.";
    }
    String DetailRenderTemplate() const final {
      return R"(The scope {0} 's region cover is not complete.
The region cover property require to hold for every of its child blocks
)";
    }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    Block block_;
  };

  for (const auto& child_block_sref : tir::GetChildBlocks(self, scope_root)) {
    const BlockNode* child_block = TVM_SREF_TO_BLOCK(child_block_sref);
    for (const BufferRegion& region : child_block->reads) {
      if (region->buffer.same_as(read_buffer)) {
        if (!self->block_info.at(child_block_sref).region_cover) {
          const BlockNode* block = TVM_SREF_TO_BLOCK(scope_root);
          throw NotRegionCoverError(self->mod, GetRef<Block>(block));
        }
      }
    }
  }
}

/******** Implementation ********/

StmtSRef CacheRead(ScheduleState self, const StmtSRef& block_sref, int read_buffer_index,
                   const String& storage_scope, const Array<StmtSRef> consumer_blocks) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *   - There is at most one block who write the buffer in the scope
   *
   * Mutate:
   *   - Allocate new cache buffer under the current scope.
   *   - Find the lowest ancestor of the block and ANY ONE of the consumers blocks.
   *   - Copy the buffer with the consumed region.
   */

  // Step 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Step 1. Check index, getting the target buffer and the parent scope
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  Buffer read_buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block), read_buffer_index, BufferIndexType::kRead);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  // Check required region cover for cache_read
  CheckRegionCover(self, scope_sref, read_buffer);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_sref);

  // Step 2. Create CacheStageInfo
  CacheStageInfo info;
  info.read_buffer = read_buffer;

  // info.consumer_blocks indicates which buffers should consume the cache.
  for (auto consumer : consumer_blocks) {
    info.consumer_blocks.insert(consumer);
    for (auto child : tir::GetChildBlocks(self, consumer)) {
      info.consumer_blocks.insert(child);
    }
  }

  // Step 3. Update cache stage info.
  BufferRegion cache_region{nullptr};
  if (Optional<StmtSRef> _write_block_sref = GetOnlyWriteBlock(self, scope_sref, read_buffer)) {
    // Case 1. The buffer is written inside the block.
    StmtSRef write_block_sref = _write_block_sref.value();
    const BlockNode* write_block = TVM_SREF_TO_BLOCK(write_block_sref);
    // Find the producing region
    BufferRegion region = GetBufferRegionFromBuffer(write_block->writes, read_buffer).value();
    StmtSRef parent_sref = GetRef<StmtSRef>(write_block_sref->parent);

    // Detect insert position
    CacheLocDetector::Detect</*is_cache_read=*/true>(self, write_block_sref, scope_sref, &info);
    cache_region = RelaxBufferRegion(self, region, write_block_sref, parent_sref, info.loc_sref);
  } else {
    // Case 2. The buffer is the input block for the scope.
    info.loc_sref = scope_sref;
    info.loc_pos = 0;
    if (Optional<BufferRegion> region =
            GetBufferRegionFromBuffer(scope_block->reads, read_buffer)) {
      cache_region = region.value();
    } else {
      cache_region = BufferRegion::FullRegion(read_buffer);
    }
  }

  // Step 4. Making new cache stage block and rewrite readers.
  bool cache_full_region = info.loc_sref->StmtAs<BlockNode>() == nullptr ||
                           !AllConsumersUnderStmt(self, read_buffer, scope_sref, info.loc_sref);
  info.cache_region = cache_region;
  info.write_buffer = WithScope(read_buffer, storage_scope);
  if (!cache_full_region) {
    auto* write_buffer = info.write_buffer.CopyOnWrite();
    std::vector<PrimExpr> shape;
    for (auto cache_range : info.cache_region->region) {
      shape.push_back(cache_range->extent);
    }
    write_buffer->shape = std::move(shape);
  }
  info.alloc = info.write_buffer;

  Block cache_read_stage =
      MakeCacheStage(/*cache_region=*/cache_region, /*info=*/&info,
                     /*storage_scope=*/storage_scope, /*cache_full_region=*/cache_full_region);
  Stmt new_scope = CacheReadRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info,
                                              /*cache_full_region=*/cache_full_region);

  // Step 5. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_read_stage.get());
  BlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.stage_pipeline = true;
  return result_block_sref;
}

StmtSRef CacheWrite(ScheduleState self, const StmtSRef& block_sref, int write_buffer_index,
                    const String& storage_scope, const Array<StmtSRef> consumer_blocks) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *   - There is only one block who write the buffer in the scope
   *
   * Mutate:
   *   - Allocate new cache buffer under the current scope.
   *   - Find the lowest ancestor of the block and ANY ONE of the producer blocks.
   *   - Copy the buffer with the consumed region.
   */

  // Step 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Step 1. Checking index, getting the target buffer and the parent scope
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  Buffer write_buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block), write_buffer_index, BufferIndexType::kWrite);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);

  // Step 2. Creating CacheStageInfo
  CacheStageInfo info;
  // Create the corresponding buffer to be written, i.e. result of cache_write
  info.write_buffer = write_buffer;

  // info.consumer_blocks indicates which buffers should consume the cache.
  for (auto consumer : consumer_blocks) {
    info.consumer_blocks.insert(consumer);
    for (auto child : tir::GetChildBlocks(self, consumer)) {
      info.consumer_blocks.insert(child);
    }
  }

  // Step 3. Check the only writer block.
  ICHECK_EQ(block_sref.get(), GetOnlyWriteBlock(self, scope_sref, write_buffer).get());

  // Step 4. Find the producing region and insert position
  BufferRegion region = GetBufferRegionFromBuffer(block->writes, write_buffer).value();
  StmtSRef parent_sref = GetRef<StmtSRef>(block_sref->parent);
  // Detect insert position
  CacheLocDetector::Detect</*is_cache_read=*/false>(self, block_sref, scope_sref, &info);
  BufferRegion cache_region =
      RelaxBufferRegion(self, region, block_sref, parent_sref, info.loc_sref);

  bool cache_full_region = info.loc_sref->StmtAs<BlockNode>() == nullptr ||
                           !AllConsumersUnderStmt(self, write_buffer, scope_sref, info.loc_sref);
  info.cache_region = cache_region;
  info.read_buffer = WithScope(write_buffer, storage_scope);
  if (!cache_full_region) {
    auto* read_buffer = info.read_buffer.CopyOnWrite();
    std::vector<PrimExpr> shape;
    for (auto cache_range : info.cache_region->region) {
      shape.push_back(cache_range->extent);
    }
    read_buffer->shape = std::move(shape);
  }
  info.alloc = info.read_buffer;

  // Step 5. Making new cache stage block and rewrite readers.
  Block cache_write_stage =
      MakeCacheStage(/*cache_region=*/cache_region, /*info=*/&info,
                     /*storage_scope=*/storage_scope, /*cache_full_region=*/cache_full_region);
  Stmt new_scope = CacheWriteRewriter::Rewrite(/*scope_sref=*/scope_sref,
                                               /*writer_block_sref=*/block_sref, /*info=*/&info,
                                               /*cache_full_region=*/cache_full_region);

  // Step 6. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_write_stage.get());
  BlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.stage_pipeline = true;
  return result_block_sref;
}

Array<StmtSRef> GetLoopsUnderScope(const StmtSRef& block_sref, const StmtSRef& top_sref) {
  std::vector<StmtSRef> result;
  for (StmtSRefNode* parent = block_sref->parent; parent && parent->stmt->IsInstance<ForNode>();
       parent = parent->parent) {
    if (parent == top_sref.get()) break;
    result.push_back(GetRef<StmtSRef>(parent));
  }
  return {result.rbegin(), result.rend()};
}

/*!
 * \brief The schedule error that block iter vars appears in old buffer and new
 * allocated cache buffer does not match.
 */
class ReindexCacheReadWriteNotMatchError : public ScheduleError {
 public:
  ReindexCacheReadWriteNotMatchError(IRModule mod, Block block, Var var,
                                     Array<PrimExpr> old_indices, Array<PrimExpr> new_indices,
                                     bool is_cache_read, bool appears_in_old)
      : mod_(std::move(mod)), block_(std::move(block)), var_(std::move(var)) {
    primitive_name_ = is_cache_read ? "reindex_cache_read" : "reindex_cache_write";
    if (appears_in_old) {
      appears_indices_ = std::move(old_indices);
      other_indices_ = std::move(new_indices);
    } else {
      appears_indices_ = std::move(new_indices);
      other_indices_ = std::move(old_indices);
    }
  }
  String FastErrorString() const final {
    return "ScheduleError: the block itervars appeared in lhs and rhs of reindex cache stage do "
           "not match.";
  }

  String DetailRenderTemplate() const final {
    std::stringstream s;
    s << "Error when applying " << primitive_name_ << " on block {0}, the block itervar " << var_
      << " appears in " << appears_indices_ << ", but not in " << other_indices_ << ".";
    return String(s.str());
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }
  IRModule mod_;
  String primitive_name_;
  Block block_;
  Var var_;
  Array<PrimExpr> appears_indices_;
  Array<PrimExpr> other_indices_;
};

/*!
 * \brief Update ReindexCacheStageInfo and create new cache buffer, used in
 * both ReindexCacheRead and ReindexCacheWrite.
 * \param info Pointer to ReindexCacheStageInfo
 * \param mod The IRModule.
 * \param block_sref The StmtSRef to the block we are working on.
 * \param storage_scope The storage scope of cache buffer (e.g. "shared"/"local").
 * \param index_map The user defined indices.
 * \param blok The block we are working on.
 * \param realize The BlockRealize this block belongs to.
 * \param old_buffer The buffer whose buffer access need to be rewriten.
 * \param cache_region The old buffer access region.
 */
template <bool is_cache_read>
void CollectReindexCacheStageInfoAndCreateBuffer(
    ReindexCacheStageInfo* info, const IRModule& mod, const StmtSRef& block_sref,
    const String& storage_scope, const IndexMap& index_map, const Block& block,
    const BlockRealize& realize, const Buffer& old_buffer, const BufferRegion& cache_region) {
  arith::Analyzer analyzer;
  Array<PrimExpr> block_iter_vars, block_shape;
  for (const IterVar& iter_var : block->iter_vars) {
    block_iter_vars.push_back(iter_var);
    block_shape.push_back(iter_var->dom->extent);
  }
  Array<PrimExpr> new_indices = index_map->MapIndices(block_iter_vars, &analyzer);
  Array<PrimExpr> new_shape = index_map->MapShape(block_shape, &analyzer);
  info->indices = new_indices;

  // Step 5. Update CacheTouchedInfo
  VarUseDefAnalyzer collector_old(/*defined_vars=*/{});
  Array<PrimExpr> old_indices;
  for (const Range& range : cache_region->region) {
    collector_old(range->min);
    old_indices.push_back(range->min);
  }

  VarUseDefAnalyzer collector_new(/*defined_vars=*/{});
  for (const PrimExpr& idx : new_indices) {
    collector_new(idx);
  }

  VarUseDefAnalyzer collector_iter_values(/*defined_vars=*/{});
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    const IterVar& block_iter_var = block->iter_vars[i];
    const PrimExpr& block_iter_value = realize->iter_values[i];
    bool appears_in_new = collector_new.use_count_.count(block_iter_var->var.get());
    bool appears_in_old = collector_old.use_count_.count(block_iter_var->var.get());
    if (appears_in_new != appears_in_old) {
      throw ReindexCacheReadWriteNotMatchError(mod, block, block_iter_var->var, old_indices,
                                               new_indices, is_cache_read, appears_in_old);
    }
    if (appears_in_new) {
      info->block_iter_vars.push_back(block_iter_var);
      info->block_iter_values.push_back(block_iter_value);
      collector_iter_values(block_iter_value);
    }
  }

  for (const StmtSRef& loop_sref : GetLoopsUnderScope(block_sref, info->loc_sref)) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
    if (collector_iter_values.use_count_.count(loop->loop_var.get())) {
      info->loop_vars.push_back(loop->loop_var);
      info->loop_ranges.push_back(Range::FromMinExtent(loop->min, loop->extent));
    }
  }

  // Create new buffer
  ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*old_buffer.get());
  ObjectPtr<VarNode> new_var = make_object<VarNode>(*old_buffer->data.get());
  const auto* ptr_type = TVM_TYPE_AS(old_buffer->data->type_annotation, PointerTypeNode);
  new_var->type_annotation = PointerType(ptr_type->element_type, storage_scope);
  new_buffer->data = Var(new_var->name_hint + "_" + storage_scope, new_var->type_annotation);
  new_buffer->name = old_buffer->name + "_" + storage_scope;
  new_buffer->shape = new_shape;

  if (is_cache_read) {
    info->write_buffer = Buffer(new_buffer);
    info->alloc = info->write_buffer;
  } else {
    info->read_buffer = Buffer(new_buffer);
    info->alloc = info->read_buffer;
  }
}

/*! \brief Check whether given cache_region is a single point access. */
template <bool is_cache_read>
void CheckSinglePoint(ScheduleState self, const Block& block, const BufferRegion& cache_region) {
  bool single_point = true;
  for (const Range& range : cache_region->region) {
    const auto* ext_int = range->extent.as<IntImmNode>();
    if (!ext_int || ext_int->value != 1) {
      single_point = false;
    }
  }
  if (!single_point) {
    throw NotSinglePointAccess(self->mod, block, cache_region, is_cache_read);
  }
}

StmtSRef ReindexCacheRead(ScheduleState self, const StmtSRef& block_sref, int read_buffer_index,
                          const String& storage_scope, const IndexMap& index_map) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *   - There is at most one block who write the buffer in the scope
   *
   * Mutate:
   *   - Allocate new cache buffer under the current scope.
   *   - Find the lowest ancestor of the block and ANY ONE of the consumers blocks.
   *   - Copy the buffer with the consumed region.
   */

  // Step 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Step 1. Check index, getting the target buffer and the parent scope
  Block block = GetRef<Block>(TVM_SREF_TO_BLOCK(block_sref));
  BlockRealize realize = GetBlockRealize(self, block_sref);
  Buffer read_buffer = GetNthAccessBuffer(self, block, read_buffer_index, BufferIndexType::kRead);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);

  // Step 2. Create CacheStageInfo
  ReindexCacheStageInfo info;
  info.read_buffer = read_buffer;
  info.consumer_blocks.insert(block_sref);

  // Step 3. Update cache stage info.
  Optional<BufferRegion> maybe_region = GetBufferRegionFromBuffer(block->reads, read_buffer);
  ICHECK(maybe_region.defined()) << read_buffer
                                 << " should appear in the block's read region: " << block->reads;
  BufferRegion cache_region = maybe_region.value();
  if (Optional<StmtSRef> _write_block_sref = GetOnlyWriteBlock(self, scope_sref, read_buffer)) {
    // Case 1. The buffer is written inside the block.
    StmtSRef write_block_sref = _write_block_sref.value();
    // Find the producing region
    StmtSRef parent_sref = GetRef<StmtSRef>(write_block_sref->parent);
    // Detect insert position
    CacheLocDetector::Detect</*is_cache_read=*/true>(self, write_block_sref, scope_sref, &info);
  } else {
    // Case 2. The buffer is the input block for the scope.
    info.loc_sref = scope_sref;
    info.loc_pos = 0;
  }

  // Step 4. Check whether cache region is a single point.
  CheckSinglePoint</*is_cache_read=*/true>(self, block, cache_region);

  // Step 5. Collect ReindexCacheStageInfo and create new buffer.
  CollectReindexCacheStageInfoAndCreateBuffer</*is_cache_read=*/true>(
      &info, self->mod, block_sref, storage_scope, index_map, block, realize, read_buffer,
      cache_region);

  // Step 6. Making new cache stage block and rewrite readers.
  Block cache_read_stage =
      MakeReindexCacheStage</*is_cache_read=*/true>(/*cache_region=*/cache_region,
                                                    /*info=*/&info,
                                                    /*storage_scope=*/storage_scope);
  Stmt new_scope = ReindexCacheReadRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info);

  // Step 7. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_read_stage.get());
  BlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.stage_pipeline = true;
  return result_block_sref;
}

StmtSRef ReindexCacheWrite(ScheduleState self, const StmtSRef& block_sref, int write_buffer_index,
                           const String& storage_scope, const IndexMap& index_map) {
  /*!
   * Check:
   *   - The index is in the array of block reading region
   *   - There is only one block who write the buffer in the scope
   *
   * Mutate:
   *   - Allocate new cache buffer under the current scope.
   *   - Find the lowest ancestor of the block and ANY ONE of the producer blocks.
   *   - Copy the buffer with the consumed region.
   */

  // Step 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Step 1. Checking index, getting the target buffer and the parent scope
  Block block = GetRef<Block>(TVM_SREF_TO_BLOCK(block_sref));
  BlockRealize realize = GetBlockRealize(self, block_sref);
  Buffer write_buffer =
      GetNthAccessBuffer(self, block, write_buffer_index, BufferIndexType::kWrite);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);

  // Step 2. Creating CacheStageInfo
  ReindexCacheStageInfo info;
  info.write_buffer = write_buffer;

  // Step 3. Check the only writer block.
  ICHECK_EQ(block_sref.get(), GetOnlyWriteBlock(self, scope_sref, write_buffer).get());

  // Step 4. Find the producing region and insert position
  Optional<BufferRegion> maybe_region = GetBufferRegionFromBuffer(block->writes, write_buffer);
  ICHECK(maybe_region.defined()) << write_buffer << " should appear in the block's write region";
  StmtSRef parent_sref = GetRef<StmtSRef>(block_sref->parent);
  // Detect insert position
  CacheLocDetector::Detect</*is_cache_read=*/false>(self, block_sref, scope_sref, &info);
  BufferRegion cache_region = maybe_region.value();

  CollectReindexCacheStageInfoAndCreateBuffer</*is_cache_read=*/false>(
      &info, self->mod, block_sref, storage_scope, index_map, block, realize, write_buffer,
      cache_region);

  // Step 5. Check whether cache region is a single point.
  CheckSinglePoint</*is_cache_read=*/false>(self, block, cache_region);

  // Step 6. Making new cache stage block and rewrite readers.
  Block cache_write_stage =
      MakeReindexCacheStage</*is_cache_read=*/false>(/*cache_region=*/cache_region,
                                                     /*info=*/&info,
                                                     /*storage_scope=*/storage_scope);
  Stmt new_scope = ReindexCacheWriteRewriter::Rewrite(
      /*scope_sref=*/scope_sref,
      /*writer_block_sref=*/block_sref, /*info=*/&info);

  // Step 7. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_write_stage.get());
  BlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.stage_pipeline = true;
  return result_block_sref;
}

/*! \brief The schedule error that the target block doesn't both read&write target buffer. */
class NotReadWriteError : public ScheduleError {
 public:
  NotReadWriteError(IRModule mod, Block block, Buffer buffer)
      : mod_(std::move(mod)), block_(std::move(block)), buffer_(std::move(buffer)) {}
  String FastErrorString() const final {
    return "ScheduleError: The target block does not both read & write target buffer.";
  }

  String DetailRenderTemplate() const final {
    return "The target block {0} does not both read & write target buffer {1}.";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_, buffer_}; }
  IRModule mod_;
  Block block_;
  Buffer buffer_;
};

Array<StmtSRef> CacheInplace(ScheduleState self, const StmtSRef& block_sref, int read_buffer_index,
                             const String& storage_scope) {
  /*!
   * Do cache read then cache write
   */

  // Check 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Check 1. Check index, get the target buffer and the parent scope
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  Buffer buffer =
      GetNthAccessBuffer(self, GetRef<Block>(block), read_buffer_index, BufferIndexType::kRead);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);

  // Check 3. Check required region cover for cache_read
  CheckRegionCover(self, scope_sref, buffer);

  // Check 4. Check if target block both read & write target buffer.
  const BlockNode* rw_block = TVM_SREF_TO_BLOCK(block_sref);
  Optional<BufferRegion> read_region = GetBufferRegionFromBuffer(rw_block->reads, buffer);
  Optional<BufferRegion> write_region = GetBufferRegionFromBuffer(rw_block->writes, buffer);
  if (!read_region.defined() || !write_region.defined()) {
    throw NotReadWriteError(self->mod, GetRef<Block>(rw_block), buffer);
  }

  Array<StmtSRef> results_block_sref;
  Buffer new_buffer = WithScope(buffer, storage_scope);

  // Do cache read
  // Cache read step 0. Create CacheStageInfo
  CacheStageInfo info;
  info.read_buffer = buffer;
  // Create the corresponding buffer to be written for cache_read
  info.write_buffer = new_buffer;
  // Create the corresponding buffer allocation
  info.alloc = info.write_buffer;
  // Indicate which buffers should consume the cache.
  info.consumer_blocks.insert(block_sref);

  // Cache read step 1. Detect insert position
  CacheInplaceLocDetector::Detect(self, block_sref, scope_sref, &info);

  // Cache read step 2. Making new cache stage block and rewrite readers.
  Block cache_read_stage = MakeCacheStage(/*cache_region=*/read_region.value(), /*info=*/&info,
                                          /*storage_scope=*/storage_scope);
  Stmt new_scope = CacheReadRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info);

  // Cache read step 3. Replacing and updating flags for cache read.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_read_stage.get());
  BlockInfo& block_info_read = self->block_info[result_block_sref];
  block_info_read.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info_read.region_cover = true;
  block_info_read.stage_pipeline = false;
  results_block_sref.push_back(result_block_sref);

  // Do cache write
  // Cache write step 0. Update cache stage info for cache_read.
  info.read_buffer = new_buffer;
  // Create the corresponding buffer to be written, i.e. result of cache_write
  info.write_buffer = buffer;
  // Create the corresponding buffer allocation
  info.alloc = nullptr;
  info.consumer_blocks.clear();

  // Cache write step 1. Detect insert position
  CacheInplaceLocDetector::Detect(self, block_sref, scope_sref, &info);
  // insert after target block for cache write
  info.loc_pos += 1;

  // Cache write step 2. Making new cache stage block and rewrite readers.
  Block cache_write_stage = MakeCacheStage(/*cache_region=*/write_region.value(), /*info=*/&info,
                                           /*storage_scope=*/storage_scope);
  new_scope = CacheWriteRewriter::Rewrite(/*scope_sref=*/scope_sref,
                                          /*writer_block_sref=*/block_sref, /*info=*/&info);

  // Cache write step 4. Replacing and updating flags for cache write.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  result_block_sref = self->stmt2ref.at(cache_write_stage.get());
  BlockInfo& block_info_write = self->block_info[result_block_sref];
  block_info_write.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info_write.region_cover = true;
  block_info_write.stage_pipeline = false;
  results_block_sref.push_back(result_block_sref);

  return results_block_sref;
}

StmtSRef ReIndex(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                 BufferIndexType buffer_index_type) {
  const BlockNode* block_ptr = TVM_SREF_TO_BLOCK(block_sref);
  Block block = GetRef<Block>(block_ptr);
  Buffer buffer = GetNthAccessBuffer(self, block, buffer_index, buffer_index_type);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);
  arith::Analyzer analyzer;

  // Step 1. Collect the original indices and check there's only single pattern of related
  // Load/Store and the buffer is not accessed opaquely
  Array<PrimExpr> original_indices = ReIndexCollector::Collect(self->mod, buffer, block);
  // Simplify the indices if possible
  for (const IterVar& iter : block->iter_vars) {
    analyzer.Bind(iter->var, iter->dom);
  }
  original_indices.MutateByApply(
      [&analyzer](const PrimExpr& expr) { return SimplifyNonTrivialExpr(expr, &analyzer); });

  // Collect block iters appearing in the original_indices
  std::unordered_set<Var> covered;
  for (const PrimExpr& index : original_indices) {
    PreOrderVisit(index, [&](const ObjectRef& obj) -> bool {
      if (auto var = obj.as<Var>()) {
        covered.insert(var.value());
      }
      return true;
    });
  }

  // Step 2. Creating CacheStageInfo
  CacheStageInfo info;
  // Create the corresponding buffer to be read(write), i.e. the result of reindex read(write)
  if (buffer_index_type == BufferIndexType::kWrite) {
    info.read_buffer = CreateReindexBuffer(buffer, block->iter_vars, covered);
    info.write_buffer = buffer;
    info.alloc = info.read_buffer;
  } else {
    info.read_buffer = buffer;
    info.write_buffer = CreateReindexBuffer(buffer, block->iter_vars, covered);
    info.alloc = info.write_buffer;
  }

  // Step 3. Check the block belongs to a chain loop nesting under the scope,
  //         and get the insert location
  const StmtSRefNode* loop;
  for (loop = block_sref->parent; loop->parent != scope_sref.get();) {
    const ForNode* outer = loop->parent->StmtAs<ForNode>();
    const ForNode* inner = loop->StmtAs<ForNode>();
    ICHECK(outer != nullptr && inner != nullptr);
    ICHECK(outer->body.get() == inner);
    loop = loop->parent;
  }

  info.loc_pos = loop->seq_index == -1 ? 0 : loop->seq_index;
  if (buffer_index_type == BufferIndexType::kWrite) {
    info.loc_pos++;
  }

  // Step 4. Making new reindex stage block and rewrite
  Block reindex_stage =
      MakeReIndexStage(block, &info, covered, original_indices, buffer_index, buffer_index_type);
  Stmt new_scope = ReIndexRewriter::Rewrite(scope_sref, block_sref, &info, covered);

  // Step 5. Replacing and updating flags
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(reindex_stage.get());
  BlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.stage_pipeline = true;
  return result_block_sref;
}

/******** Instruction Registration ********/

struct CacheReadTraits : public UnpackedInstTraits<CacheReadTraits> {
  static constexpr const char* kName = "CacheRead";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block,
                                         Array<BlockRV> consumer_blocks, Integer read_buffer_index,
                                         String storage_scope) {
    return sch->CacheRead(block, read_buffer_index->value, storage_scope, consumer_blocks);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Array<String> consumer_blocks,
                                 Integer read_buffer_index, String storage_scope) {
    PythonAPICall py("cache_read");
    py.Input("block", block);
    py.Input("read_buffer_index", read_buffer_index->value);
    py.Input("storage_scope", storage_scope);
    // Only write out consumer blocks if provided.
    if (!consumer_blocks.empty()) {
      py.Input("consumer_blocks", consumer_blocks);
    }
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct CacheWriteTraits : public UnpackedInstTraits<CacheWriteTraits> {
  static constexpr const char* kName = "CacheWrite";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block,
                                         Array<BlockRV> consumer_blocks, Integer write_buffer_index,
                                         String storage_scope) {
    return sch->CacheWrite(block, write_buffer_index->value, storage_scope, consumer_blocks);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Array<String> consumer_blocks,
                                 Integer write_buffer_index, String storage_scope) {
    PythonAPICall py("cache_write");
    py.Input("block", block);
    py.Input("write_buffer_index", write_buffer_index->value);
    py.Input("storage_scope", storage_scope);
    // Only write out consumer blocks if provided.
    if (!consumer_blocks.empty()) {
      py.Input("consumer_blocks", consumer_blocks);
    }
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct CacheInplaceTraits : public UnpackedInstTraits<CacheInplaceTraits> {
  static constexpr const char* kName = "CacheInplace";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static Array<BlockRV> UnpackedApplyToSchedule(Schedule sch, BlockRV block,
                                                Integer read_buffer_index, String storage_scope) {
    return sch->CacheInplace(block, read_buffer_index->value, storage_scope);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Integer read_buffer_index,
                                 String storage_scope) {
    PythonAPICall py("cache_inplace");
    py.Input("block", block);
    py.Input("read_buffer_index", read_buffer_index->value);
    py.Input("storage_scope", storage_scope);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct ReIndexTraits : public UnpackedInstTraits<ReIndexTraits> {
  static constexpr const char* kName = "ReIndex";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block, Integer buffer_index,
                                         Integer buffer_index_type) {
    return sch->ReIndex(block, buffer_index.IntValue(),
                        static_cast<BufferIndexType>(buffer_index_type->value));
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Integer buffer_index,
                                 Integer buffer_index_type) {
    PythonAPICall py("reindex");
    py.Input("block", block);
    std::ostringstream os;
    os << "(\"" << BufferIndexType2Str(static_cast<BufferIndexType>(buffer_index_type->value))
       << "\", " << buffer_index << ")";
    py.Input("buffer", os.str());
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct ReindexCacheReadTraits : public UnpackedInstTraits<ReindexCacheReadTraits> {
  static constexpr const char* kName = "ReindexCacheRead";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block, IndexMap index_map,
                                         Integer read_buffer_index, String storage_scope) {
    return sch->ReindexCacheRead(block, read_buffer_index->value, storage_scope, index_map);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, IndexMap index_map,
                                 Integer read_buffer_index, String storage_scope) {
    PythonAPICall py("reindex_cache_read");
    py.Input("block", block);
    py.Input("read_buffer_index", read_buffer_index->value);
    py.Input("storage_scope", storage_scope);
    py.Input("index_map", index_map->ToPythonString());
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct ReindexCacheWriteTraits : public UnpackedInstTraits<ReindexCacheWriteTraits> {
  static constexpr const char* kName = "ReindexCacheWrite";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block, IndexMap index_map,
                                         Integer write_buffer_index, String storage_scope) {
    return sch->ReindexCacheWrite(block, write_buffer_index->value, storage_scope, index_map);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, IndexMap index_map,
                                 Integer write_buffer_index, String storage_scope) {
    PythonAPICall py("reindex_cache_write");
    py.Input("block", block);
    py.Input("write_buffer_index", write_buffer_index->value);
    py.Input("storage_scope", storage_scope);
    py.Input("index_map", index_map->ToPythonString());
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(CacheReadTraits);
TVM_REGISTER_INST_KIND_TRAITS(CacheWriteTraits);
TVM_REGISTER_INST_KIND_TRAITS(CacheInplaceTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReIndexTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReindexCacheReadTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReindexCacheWriteTraits);

}  // namespace tir
}  // namespace tvm
