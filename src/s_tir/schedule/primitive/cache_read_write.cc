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

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/builtin.h>

#include <unordered_set>

#include "../../../tirx/analysis/var_use_def_analysis.h"
#include "../../../tirx/transform/ir_utils.h"
#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

/******** Error Classes ********/

class NotSingleWriteBlock : public ScheduleErrorContextObj {
 public:
  explicit NotSingleWriteBlock(IRModule mod, BufferVar buffer, ffi::Array<StmtSRef> write_blocks)
      : mod_(std::move(mod)), buffer_(std::move(buffer)) {
    TVM_FFI_ICHECK_GT(write_blocks.size(), 1);
    write_blocks_.reserve(write_blocks.size());
    for (const StmtSRef& block_sref : write_blocks) {
      const SBlockNode* block = TVM_SREF_TO_SBLOCK(block_sref);
      write_blocks_.push_back(ffi::GetRef<SBlock>(block));
    }
  }

  ffi::String FastErrorString() const final {
    return "ScheduleError: The buffer is allowed to be written by single block.";
  }

  ffi::String DetailRenderTemplate() const final {
    size_t k = write_blocks_.size();
    return "The buffer " + buffer_.name() + " is expected to be written by single block, but got " +
           std::to_string(k) + " blocks who write it.";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final {
    return {write_blocks_.begin(), write_blocks_.end()};
  }

 private:
  IRModule mod_;
  BufferVar buffer_;
  ffi::Array<SBlock> write_blocks_;
};

/******** Helper Functions/Classes ********/

/*! \brief The auxiliary info used for the insertion point and content of the cache stage. */
struct CacheStageInfo {
  /*! \brief The buffer to be read. */
  BufferVar read_buffer;
  /*! \brief The buffer to be written. */
  BufferVar write_buffer;
  /*! \brief The buffer allocation to be inserted into the block signature. */
  ffi::Optional<BufferVar> alloc;
  /*! \brief The AST node whose body is where the cache stage should be inserted. */
  StmtSRef loc_sref;
  /*! \brief The index to insert the cache_read/cache_write stage. */
  size_t loc_pos;
  /*! \brief The cache_read/cache_write stage to be inserted. */
  Stmt cache_stage;
  /*! \brief The map used for ScheduleStateNode::Replace. */
  ffi::Map<SBlock, SBlock> block_reuse;
  /*! \brief A set of blocks that will consume the new cache. */
  std::unordered_set<StmtSRef, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> consumer_blocks;
  /*! \brief cache region for the buffer to be cached */
  TensorRegion cache_region;
};

/*! \brief Return the buffer region related with the buffer */
ffi::Optional<TensorRegion> GetBufferRegionFromBuffer(
    const ffi::Array<TensorRegion>& buffer_regions, const BufferVar& buffer) {
  ffi::Optional<TensorRegion> res = std::nullopt;
  for (const auto& region : buffer_regions) {
    if (region->source.as_or_throw<tvm::tirx::BufferVar>().same_as(buffer)) {
      TVM_FFI_ICHECK(!res.has_value());
      res = region;
    }
  }
  return res;
}

struct ReindexCacheStageInfo : CacheStageInfo {
  /* Indices used to access the allocated cache buffer. */
  ffi::Array<PrimExpr> indices;
  /* Touched loop variable related information. */
  ffi::Array<Var> loop_vars;
  ffi::Array<Range> loop_ranges;
  /* Touched block variable related information. */
  ffi::Array<IterVar> block_iter_vars;
  ffi::Array<PrimExpr> block_iter_values;
};

/* \brief The schedule error that accessed buffer region is not a single point for
 * reindex_cache_read/write. */
class NotSinglePointAccess : public ScheduleErrorContextObj {
 public:
  explicit NotSinglePointAccess(IRModule mod, SBlock block, TensorRegion cache_region,
                                bool is_cache_read)
      : mod_(std::move(mod)), block_(std::move(block)), cache_region_(cache_region) {
    primitive_name_ = is_cache_read ? "reindex_cache_read" : "reindex_cache_write";
  }

  ffi::String FastErrorString() const final {
    return "ScheduleError: The buffer region accessed inside the block is not a single point.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The buffer region " << cache_region_
       << " accessed inside block {0} is not a single point, which violates"
       << " the prerequisite of " << primitive_name_ << " primitive.";
    return ffi::String(os.str());
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  SBlock block_;
  TensorRegion cache_region_;
  ffi::String primitive_name_;
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
SBlock MakeReindexCacheStage(const TensorRegion& cache_region, ReindexCacheStageInfo* info,
                             const ffi::String& storage_scope) {
  // loop variables
  std::vector<PrimVar> loop_vars;
  // block variables
  ffi::Array<IterVar> block_vars;
  // bindings in block realize
  std::vector<PrimExpr> iter_values;
  // Create loop vars and block vars' binding_value
  ffi::Map<Var, Var> var_map;
  for (size_t i = 0; i < info->loop_vars.size(); ++i) {
    Var original_var = info->loop_vars[i];
    PrimVar loop_var(original_var->name, original_var->ty.as_or_throw<PrimType>());
    var_map.Set(original_var, loop_var);
    loop_vars.push_back(loop_var);
  }
  auto f_substitute = [&var_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = var_map.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  for (size_t i = 0; i < info->block_iter_vars.size(); ++i) {
    IterVar original_block_var = info->block_iter_vars[i];
    PrimExpr original_iter_value = info->block_iter_values[i];
    IterVar block_var = IterVar(
        /*dom=*/original_block_var->dom,
        /*var=*/PrimVar(original_block_var->var->name, original_block_var->var.ty()),
        /*IterVarType=*/kDataPar);
    var_map.Set(original_block_var->var, block_var->var);
    block_vars.push_back(block_var);
    iter_values.push_back(
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(original_iter_value, f_substitute)
            .template as_or_throw<PrimExpr>());
  }

  // block access region for read/write buffers
  Region read_access_region, write_access_region;
  ffi::Array<PrimExpr> read_access_indices, write_access_indices;
  // Compute read/write region and read/write access indices.
  ffi::Array<PrimExpr>& old_indices = (is_cache_read) ? read_access_indices : write_access_indices;
  Region& old_region = (is_cache_read) ? read_access_region : write_access_region;
  for (const Range& range : cache_region->region) {
    old_indices.push_back(ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->min, f_substitute)
                              .template as_or_throw<PrimExpr>());
    old_region.push_back(Range::FromMinExtent(old_indices.back(), IntImm::Int32(1)));
  }
  ffi::Array<PrimExpr>& new_indices = (is_cache_read) ? write_access_indices : read_access_indices;
  Region& new_region = (is_cache_read) ? write_access_region : read_access_region;
  for (const PrimExpr& idx : info->indices) {
    new_indices.push_back(ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(idx, f_substitute)
                              .template as_or_throw<PrimExpr>());
    new_region.push_back(Range::FromMinExtent(new_indices.back(), IntImm::Int32(1)));
  }

  // Create New Block
  SBlock block(
      /*iter_vars*/ std::move(block_vars),
      /*reads=*/{BufferRegion(info->read_buffer, read_access_region)},
      /*writes=*/{BufferRegion(info->write_buffer, write_access_region)},
      /*name_hint*/ cache_region->source.as_or_throw<tvm::tirx::BufferVar>().name() + "_" +
          storage_scope,
      /*body=*/
      BufferStore(info->write_buffer, BufferLoad(info->read_buffer, read_access_indices),
                  write_access_indices),
      /*init=*/std::nullopt,
      /*alloc_buffers=*/{},
      /*match_buffers=*/{},
      /*buf_doms=*/{});
  // Create SBlock Realize node
  Stmt body = SBlockRealize(/*values=*/iter_values,
                            /*predicate=*/IntImm::Bool(true),
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
SBlock MakeCacheStage(const TensorRegion& cache_region, CacheStageInfo* info,
                      const ffi::String& storage_scope, bool cache_full_region = true) {
  // loop variables
  std::vector<PrimVar> loop_vars;
  // bindings in block realize
  std::vector<PrimExpr> iter_values;
  // Create loop vars and block vars' binding_value
  for (const Range& axis_range : cache_region->region) {
    PrimVar loop_var("ax" + std::to_string(loop_vars.size()), axis_range->extent.ty());
    loop_vars.push_back(loop_var);
    iter_values.push_back(cache_full_region ? (axis_range->min + loop_var) : loop_var);
  }
  // block variables
  ffi::Array<IterVar> block_vars;
  // block access region for read/write buffers
  Region read_access_region;
  Region write_access_region;
  // indices used in block body
  ffi::Array<PrimExpr> read_access_indices;
  ffi::Array<PrimExpr> write_access_indices;
  // Create block vars, block's accessed region and accessing indices
  for (int i = 0;
       i < static_cast<int>(cache_region->source.as_or_throw<tvm::tirx::BufferVar>()->shape.size());
       ++i) {
    Range axis_range = cache_region->region[i];
    PrimVar var("v" + std::to_string(read_access_indices.size()), axis_range->extent.ty());
    if (cache_full_region) {
      PrimExpr dim = cache_region->source.as_or_throw<tvm::tirx::BufferVar>()->shape[i];
      block_vars.push_back(IterVar(/*dom=*/Range::FromMinExtent(IntImm(dim.ty(), 0), dim),
                                   /*var=*/var,
                                   /*IterVarType=*/kDataPar));
      read_access_indices.push_back(var);
      write_access_indices.push_back(var);
      read_access_region.push_back(Range::FromMinExtent(var, IntImm(var.ty(), 1)));
      write_access_region.push_back(Range::FromMinExtent(var, IntImm(var.ty(), 1)));
    } else {
      block_vars.push_back(IterVar(
          /*dom=*/Range::FromMinExtent(IntImm(axis_range->extent.ty(), 0), axis_range->extent),
          /*var=*/var,
          /*IterVarType=*/kDataPar));
      if (cache_region->source.as_or_throw<tvm::tirx::BufferVar>().same_as(info->read_buffer)) {
        // cache_read
        read_access_indices.push_back(axis_range->min + var);
        read_access_region.push_back(
            Range::FromMinExtent(axis_range->min + var, IntImm(var.ty(), 1)));
        write_access_indices.push_back(var);
        write_access_region.push_back(Range::FromMinExtent(var, IntImm(var.ty(), 1)));
      } else {
        // cache_write
        write_access_indices.push_back(axis_range->min + var);
        write_access_region.push_back(
            Range::FromMinExtent(axis_range->min + var, IntImm(var.ty(), 1)));
        read_access_indices.push_back(var);
        read_access_region.push_back(Range::FromMinExtent(var, IntImm(var.ty(), 1)));
      }
    }
  }

  // Create the body block:
  //   reads = [read_buffer[access_region]]
  //   writes = [write_buffer[access_region]]
  //     write_buffer[access_indices] = read_buffer[access_indices]
  SBlock block(
      /*iter_vars=*/std::move(block_vars),
      /*reads=*/{BufferRegion(info->read_buffer, read_access_region)},
      /*writes=*/{BufferRegion(info->write_buffer, write_access_region)},
      /*name_hint=*/cache_region->source.as_or_throw<tvm::tirx::BufferVar>().name() + "_" +
          storage_scope,
      /*body=*/
      BufferStore(info->write_buffer, BufferLoad(info->read_buffer, read_access_indices),
                  write_access_indices),
      /*init=*/std::nullopt,
      /*alloc_buffers=*/{},
      /*match_buffers=*/{},
      /*annotations=*/{});
  // Create the block realize node
  Stmt body = SBlockRealize(/*values=*/iter_values,
                            /*predicate=*/IntImm::Bool(true),
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
SBlock MakeReIndexStage(const SBlock& block, CacheStageInfo* info,
                        const std::unordered_set<Var>& covered,
                        const ffi::Array<PrimExpr>& original_indices, int buffer_index,
                        BufferIndexType buffer_index_type) {
  // iters of the reindex block
  ffi::Array<IterVar> new_block_iters;
  // the substitution map from the original block iter to the iters of the reindex block
  std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> block_var_replace_map;
  // indices to access the reindex buffer and the target buffer
  ffi::Array<PrimExpr> reindex_indices, target_indices;

  // Step 1: Create block iters, access regions of the reindex block, and accessing indices to the
  // reindex buffer.
  std::unordered_set<int> skipped_block_iters;
  for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
    const IterVar& iter = block->iter_vars[i];
    PrimVar var("v" + std::to_string(new_block_iters.size()), iter->var.ty());
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
  auto f_substitute =
      [&block_var_replace_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto it = block_var_replace_map.find(var); it != block_var_replace_map.end()) {
      return ffi::Any(it->second);
    }
    return ffi::Unchanged();
  };
  for (const PrimExpr& index : original_indices) {
    target_indices.push_back(
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(index, f_substitute).as_or_throw<PrimExpr>());
  }

  // Step 3: Create the reindex block

  // The src and the dst region and indices of the data copy
  Region src_region{nullptr};
  Region dst_region{nullptr};
  ffi::Array<PrimExpr> src_indices{nullptr};
  ffi::Array<PrimExpr> dst_indices{nullptr};

  if (buffer_index_type == BufferIndexType::kWrite) {
    src_indices = reindex_indices;
    dst_indices = target_indices;
  } else {
    src_indices = target_indices;
    dst_indices = reindex_indices;
  }

  // Create the body block
  SBlock new_block(
      /*iter_vars=*/new_block_iters,
      /*reads=*/{BufferRegionFromPoint(info->read_buffer, src_indices)},
      /*writes=*/{BufferRegionFromPoint(info->write_buffer, dst_indices)},
      /*name_hint=*/info->write_buffer.name() + "_reindex",
      /*body=*/
      BufferStore(info->write_buffer, BufferLoad(info->read_buffer, src_indices), dst_indices));

  // Step 4: Create surrounding loops

  // Create loop vars and bindings for block iters
  std::vector<PrimVar> loop_vars;     // loop variables
  std::vector<PrimExpr> iter_values;  // bindings in block realize
  for (int i = 0; i < static_cast<int>(block->iter_vars.size()); ++i) {
    if (skipped_block_iters.count(i)) {
      continue;
    }
    PrimVar loop_var("ax" + std::to_string(loop_vars.size()), block->iter_vars[i]->var.ty());
    loop_vars.push_back(loop_var);
    iter_values.push_back(loop_var);
  }

  // Create the block realize node
  Stmt body = SBlockRealize(/*values=*/iter_values,
                            /*predicate=*/IntImm::Bool(true),
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
  StmtSRef parent_sref = ffi::GetRef<StmtSRef>(block_sref->parent);
  return IsAffineBinding(/*realize=*/GetSBlockRealize(self, block_sref),
                         /*loop_var_ranges=*/LoopDomainOfSRefTreePath(parent_sref),
                         /*analyzer=*/analyzer.get());
}

/*!
 * \brief Insert the cache_read/cache_write stage into the specific position
 * \param stmt A sequence of statements or a single statement that the new stage is inserted in
 * \param pos The position where the cache stage is inserted
 * \param stage The stage to be inserted
 * \return A SeqStmt, the result after insertion
 */
Stmt InsertCacheStage(const Stmt& stmt, int pos, const Stmt& stage) {
  if (const auto* seq_stmt = stmt.as<SeqStmtNode>()) {
    ffi::Array<Stmt> seq = seq_stmt->seq;
    TVM_FFI_ICHECK_LE(pos, seq.size())
        << "Cannot insert at position " << pos << " into sequence of length " << seq.size();
    seq.insert(seq.begin() + pos, stage);
    return SeqStmt(seq);
  } else if (pos == 0) {
    ffi::Array<Stmt> seq;
    seq.push_back(stage);
    seq.push_back(stmt);
    return SeqStmt(seq);
  } else if (pos == 1) {
    ffi::Array<Stmt> seq;
    seq.push_back(stmt);
    seq.push_back(stage);
    return SeqStmt(seq);
  } else {
    TVM_FFI_THROW(InternalError) << "Cannot insert at position " << pos
                                 << ".  When inserting adjacent to non-SeqStmt, "
                                 << "only positions 0 and 1 are valid.";
  }
}

/*!
 * \brief Get the only writer block of the input buffer in a given scope block.
 * \param self The state of the schedule
 * \param scope_sref The scope block where the write is considered
 * \param buffer The queried buffer
 * \return The sref of the only writer of the input buffer in the given scope,
 *         or `std::nullopt` if no block writes it in the scope.
 * \throw NotSingleWriteBlock if there are more than one interested block.
 */
ffi::Optional<StmtSRef> GetOnlyWriteBlock(ScheduleState self, const StmtSRef& scope_sref,
                                          const BufferVar& buffer) {
  SBlockScope scope = self->GetSBlockScope(scope_sref);
  auto it = scope->buffer_writers.find(buffer);
  if (it == scope->buffer_writers.end()) {
    return std::nullopt;
  } else {
    const ffi::Array<StmtSRef>& block_srefs = it->second;
    TVM_FFI_ICHECK(!block_srefs.empty());
    if (block_srefs.size() > 1) {
      throw MakeScheduleError<NotSingleWriteBlock>(self->mod, buffer, block_srefs);
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
bool AllConsumersUnderStmt(ScheduleState self, BufferVar buffer, StmtSRef scope_sref,
                           StmtSRef stmt_sref) {
  // Collect all children blocks of the target stmt.
  std::unordered_set<const SBlockNode*> blocks_under_target;
  for (const StmtSRef& block_sref : GetChildBlocks(self, stmt_sref)) {
    const auto* block = block_sref->StmtAs<SBlockNode>();
    TVM_FFI_ICHECK(block != nullptr);
    blocks_under_target.insert(block);
  }

  // For each block in the scope, if it is a consumer of the
  // input buffer, check if it is also a child block of the
  // target stmt.
  for (const StmtSRef& block_sref : GetChildBlocks(self, scope_sref)) {
    const auto* block = block_sref->StmtAs<SBlockNode>();
    TVM_FFI_ICHECK(block != nullptr);
    if (GetBufferRegionFromBuffer(block->reads, buffer).has_value()) {
      if (blocks_under_target.find(block) == blocks_under_target.end()) {
        return false;
      }
    }
  }
  return true;
}

/*!
 * \brief Collect OR-combined predicates from all nested BlockRealize nodes within
 * the given statement that access the specified buffer (read or write, controlled by
 * \p index_type). Each nested block's predicate is expressed in the enclosing block's
 * scope by substituting the nested block's iter var bindings. This is needed when the
 * actual access is gated by a predicate (T.where) on a nested block while the outer
 * block has a trivially-true predicate. Sibling blocks that each access the buffer under
 * different predicates are OR-ed together so the result covers the union of their access
 * regions.
 * \param body The body statement of the outer block to search within.
 * \param buffer The buffer being accessed.
 * \param index_type Whether to look for reads (kRead) or writes (kWrite).
 * \return The OR-combination of all nested block predicates found.
 */
static PrimExpr CollectNestedBlockPredicates(const Stmt& body, const BufferVar& buffer,
                                             BufferIndexType index_type) {
  struct Collector : public StmtExprVisitor {
    using StmtExprVisitor::Visit_;

    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
      if (value.as<ExprNode>()) return std::nullopt;
      return StmtExprVisitor::Visit(value);
    }

    Collector(const BufferVar& buf, BufferIndexType idx_type)
        : buffer_(buf), index_type_(idx_type), result_(IntImm::Bool(false)), found_(false) {}

    ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeNode* realize) final {
      const SBlockNode* block = realize->block.get();
      const auto& regions = (index_type_ == BufferIndexType::kRead) ? block->reads : block->writes;
      bool accesses_buffer = false;
      for (const TensorRegion& region : regions) {
        if (region->source.as_or_throw<tvm::tirx::BufferVar>().same_as(buffer_)) {
          accesses_buffer = true;
          break;
        }
      }
      if (accesses_buffer) {
        // Build substitution: nested block iter vars -> their binding values
        // (which are already expressed in terms of the outer scope).
        ffi::Map<Var, PrimExpr> subst;
        for (size_t i = 0; i < block->iter_vars.size(); ++i) {
          subst.Set(block->iter_vars[i]->var, realize->iter_values[i]);
        }
        auto f_substitute = [&subst](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
          if (auto repl = subst.Get(var)) return ffi::Any(*std::move(repl));
          return ffi::Unchanged();
        };
        PrimExpr pred = subst.empty() ? realize->predicate
                                      : ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
                                            realize->predicate, f_substitute)
                                            .as_or_throw<PrimExpr>();
        // OR the predicates across all accessing nested blocks: each such block is an
        // independent alternative access path (sibling blocks in a SeqStmt), so the
        // cache must cover the *union* of their access regions, not the intersection.
        // Using AND (the previous behaviour) underestimates the required region when
        // sibling blocks have non-overlapping predicates.
        result_ = found_ ? (result_ || pred) : pred;
        found_ = true;
      }
      // Continue recursing into deeper nested blocks.
      return StmtExprVisitor::Visit_(realize);
    }

    const BufferVar& buffer_;
    BufferIndexType index_type_;
    PrimExpr result_;
    bool found_;
  };

  auto collector = ffi::make_object<Collector>(buffer, index_type);
  collector->Visit(body);
  // If no nested block accessed the buffer, return true (no restriction — the caller
  // will fall back to the original scope-block reads / FullRegion path).
  return collector->found_ ? collector->result_ : IntImm::Bool(true);
}

/*!
 * \brief Get the buffer region under the sref tree path [dom_low_inclusive, dom_high_exclusive)
 * \param self The state of the schedule.
 * \param buffer_region The buffer region to be analyzed.
 * \param block_sref The sref of the block related to the region.
 * \param dom_low_inclusive The lowest node in the sref tree path.
 * \param dom_high_exclusive The highest node in the sref tree path.
 * \param extra_predicate An additional predicate (e.g. collected from nested blocks) to AND
 *        with the block's own predicate before relaxation. Defaults to true (no effect).
 * \return The relaxed buffer region.
 */
TensorRegion RelaxBufferRegion(ScheduleState self, const TensorRegion& buffer_region,
                               const StmtSRef& block_sref, const StmtSRef& dom_low_inclusive,
                               const StmtSRef& dom_high_exclusive,
                               PrimExpr extra_predicate = IntImm::Bool(true)) {
  SBlockRealize realize = GetSBlockRealize(self, block_sref);
  ffi::Map<Var, PrimExpr> binding = GetBindings(realize);
  const BufferVar& buffer = buffer_region->source.as_or_throw<tvm::tirx::BufferVar>();
  arith::Analyzer analyzer;
  auto f_substitute = [&binding](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = binding.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  ffi::Array<Range> mapped_region = buffer_region->region.Map([&f_substitute](const Range& range) {
    PrimExpr min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->min, f_substitute)
                       .as_or_throw<PrimExpr>();
    PrimExpr extent = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->extent, f_substitute)
                          .as_or_throw<PrimExpr>();
    return Range::FromMinExtent(min, extent);
  });
  TensorRegion subst_region = BufferRegion(buffer, mapped_region);
  ffi::Array<arith::IntSet> int_sets = AnalyzeRegionUpperBound(
      /*region=*/subst_region,
      /*predicate=*/
      ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(realize->predicate && extra_predicate,
                                                    f_substitute)
          .as_or_throw<PrimExpr>(),
      /*dom_low_inclusive=*/dom_low_inclusive,
      /*dom_high_exclusive=*/dom_high_exclusive,
      /*analyzer=*/analyzer.get());
  TVM_FFI_ICHECK_EQ(buffer_region->region.size(), int_sets.size());

  Region region;
  region.reserve(int_sets.size());
  for (size_t i = 0; i < int_sets.size(); ++i) {
    region.push_back(int_sets[i].CoverRange(Range::FromMinExtent(0, buffer->shape[i])));
  }
  return BufferRegion(buffer, region);
}

/*! \brief Detect the insertion position of the new cache stage */
class CacheLocDetector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprNode>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }

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
        for (const Dependency& def : self->GetSBlockScope(scope_sref)->GetDepsBySrc(block_sref)) {
          if (def->kind == DepKind::kRAW) {
            related_blocks.push_back(def->dst);
          }
        }
      }
    } else {
      for (const Dependency& def : self->GetSBlockScope(scope_sref)->GetDepsBySrc(block_sref)) {
        if (def->kind == DepKind::kRAW) {
          if (info->consumer_blocks.count(def->dst)) {
            continue;
          }
          related_blocks.push_back(def->dst);
        }
      }
    }

    if (!related_blocks.empty()) {
      auto detector =
          ffi::make_object<CacheLocDetector>(self, block_sref, scope_sref, related_blocks);
      detector->Visit(ffi::GetRef<Stmt>(scope_sref->stmt));
      info->loc_sref = detector->loc_sref_;
      info->loc_pos = detector->loc_pos_;
    } else {
      info->loc_sref = scope_sref;

      auto block_body = scope_sref->StmtAs<SBlockNode>()->body;
      const auto* body = block_body.as<SeqStmtNode>();
      info->loc_pos = body == nullptr ? 1 : body->size();
    }
  }

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

 private:
  ffi::Optional<VisitInterrupt> Visit_(const SeqStmtNode* seq_stmt) final {
    bool previous_visited_block = visited_block_;
    visited_block_ = false;

    for (size_t i = 0; i < seq_stmt->size(); ++i) {
      if (loc_pos_ != -1) {
        break;
      }
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(seq_stmt->seq[i]));
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
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockNode* block) final {
    // Only visit the current scope under buffer writer's parent block
    if (block == scope_sref_->stmt) {
      // The block visited is the current parent scope
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(block));
      // Handling cases when insert outside any loop or cache_read for input buffer
      if (visited_related_ && !loc_sref_.defined()) {
        loc_sref_ = self_->stmt2ref.at(block);
        // Handling cache_read for input buffer
        if (visited_block_ == false && loc_pos_ == -1) {
          loc_pos_ = 0;
        }
      }
      return std::nullopt;
    }
    // Update `visited_block`
    if (block_sref_->stmt == block) {
      visited_block_ = true;
      return std::nullopt;
    }
    // Update `visited_related`
    for (const StmtSRef& related_block : related_blocks_) {
      if (related_block->stmt == block) {
        visited_related_ = true;
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForNode* loop) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(loop));
    if (visited_block_ && visited_related_ && !loc_sref_.defined() && loc_pos_ != -1) {
      loc_sref_ = self_->stmt2ref.at(loop);
    }
    return std::nullopt;
  }

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
class CacheInplaceLocDetector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprNode>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }

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
    auto detector = ffi::make_object<CacheInplaceLocDetector>(self, block_sref, scope_sref);
    detector->Visit(ffi::GetRef<Stmt>(scope_sref->stmt));
    info->loc_sref = detector->loc_sref_;
    info->loc_pos = detector->loc_pos_;
  }

  /*!
   * \brief Constructor
   * \param self The state of the schedule
   * \param block_sref The sref of the unique writer block of the buffer being applied cache_inplace
   * \param scope_sref The sref of the scope block of the cached block
   */
  CacheInplaceLocDetector(const ScheduleState self, const StmtSRef& block_sref,
                          const StmtSRef& scope_sref)
      : self_(self), block_sref_(block_sref), scope_sref_(scope_sref) {}

 private:
  ffi::Optional<VisitInterrupt> Visit_(const SeqStmtNode* seq_stmt) final {
    for (size_t i = 0; i < seq_stmt->size(); ++i) {
      if (loc_pos_ != -1) {
        break;
      }
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(seq_stmt->seq[i]));
      // `pos` can be assigned only once when we visited `block_sref`
      if (visited_block_ && loc_pos_ == -1) {
        // The offset of insert position from the block
        loc_pos_ = i;
        return std::nullopt;
      }
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockNode* block) final {
    // Only visit the current scope under buffer writer's parent block
    if (block == scope_sref_->stmt) {
      // The block visited is the current parent scope
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(block));
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
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForNode* loop) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(loop));
    if (visited_block_ && !loc_sref_.defined()) {
      loc_sref_ = self_->stmt2ref.at(loop);
      if (loc_pos_ == -1) {
        loc_pos_ = 0;
      }
    }
    return std::nullopt;
  }

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
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

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
    auto rewriter = ffi::make_object<CacheReadRewriter>(scope_sref, info, cache_full_region);
    return rewriter->Mutate(ffi::GetRef<Stmt>(scope_sref->stmt))
        .ValueOrUnchanged(ffi::GetRef<Stmt>(scope_sref->stmt));
  }

  explicit CacheReadRewriter(const StmtSRef& scope_sref, CacheStageInfo* info,
                             bool cache_full_region = true)
      : scope_sref_(scope_sref), info_(info), cache_full_region_(cache_full_region) {
    VarRemapSet(info_->read_buffer, info_->write_buffer);
    auto update_region = [this](const Region& region, const Region& offset) -> Region {
      TVM_FFI_ICHECK_EQ(region.size(), offset.size());
      std::vector<Range> ret;
      for (size_t i = 0; i < region.size(); ++i) {
        ret.push_back(Range::FromMinExtent(ana_->Simplify(region[i]->min - offset[i]->min),
                                           region[i]->extent));
      }
      return ret;
    };

    update_access_regions = [this, update_region](ffi::Array<TensorRegion> regions) {
      if (cache_full_region_) {
        return ReplaceBuffer(std::move(regions), info_->read_buffer, info_->write_buffer);
      }

      ffi::Array<TensorRegion> ret;
      for (const TensorRegion& region : regions) {
        if (region->source.as_or_throw<tvm::tirx::BufferVar>().same_as(info_->read_buffer)) {
          ret.push_back(BufferRegion(info_->write_buffer,
                                     update_region(region->region, info_->cache_region->region)));
        } else {
          ret.push_back(region);
        }
      }
      return ret;
    };
    update_match_buffers = [this, update_region](ffi::Array<MatchBufferRegion> match_buffers) {
      if (cache_full_region_) {
        return ReplaceBuffer(std::move(match_buffers), info_->read_buffer, info_->write_buffer);
      }

      ffi::Array<MatchBufferRegion> ret;
      for (const MatchBufferRegion& match_buffer : match_buffers) {
        if (match_buffer->source->source.as_or_throw<tvm::tirx::BufferVar>().same_as(
                info_->read_buffer)) {
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

 private:
  UnchangedOr<Stmt> Mutate_(const ForNode* loop, InplaceMode inplace_mode) final {
    Stmt stmt =
        StmtExprMutator::Mutate_(loop, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(loop));
    // Check the insertion point
    if (loop == info_->loc_sref->stmt) {
      // Insert cache stage into the loop if it is the right place
      ffi::ObjectPtr<ForNode> n = ffi::make_object<ForNode>(*stmt.as<ForNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const SBlockNode* block, InplaceMode inplace_mode) override {
    SBlock old_stmt = ffi::GetRef<SBlock>(block);
    // Check if this block is one of the specified consumers.
    // If no consumer blocks are specified, all blocks should be considered consumers.
    bool is_consumer = info_->consumer_blocks.empty();
    // Otherwise check if this is one of the specified blocks.
    for (StmtSRef consumer_sref : info_->consumer_blocks) {
      const SBlockNode* consumer_node = TVM_SREF_TO_SBLOCK(consumer_sref);
      SBlock consumer_block = ffi::GetRef<SBlock>(consumer_node);
      if (old_stmt.same_as(consumer_block)) {
        is_consumer = true;
      }
    }
    // Keep track of this blocks status. We'll use this when rewriting loads.
    current_block_consumes = is_consumer;
    // We don't mutate the block which generates info->read_buffer.
    if (block != scope_sref_->stmt &&
        GetBufferRegionFromBuffer(block->writes, info_->read_buffer).has_value()) {
      return old_stmt;
    }
    // Mutate the body
    // Cache accesses change storage; the original allocation still owns its source.
    auto input_node = ffi::make_object<SBlockNode>(*block);
    input_node->alloc_buffers.clear();
    SBlock input(std::move(input_node));
    SBlock stmt = StmtExprMutator::Mutate_(input.get(), InplaceMode::kDisallow)
                      .ValueOrUnchanged(input)
                      .as_or_throw<SBlock>();
    stmt.CopyOnWrite()->alloc_buffers = block->alloc_buffers;
    // Check the insertion point
    if (block == info_->loc_sref->stmt) {
      // Insert cache stage into the block if it is the right place
      ffi::ObjectPtr<SBlockNode> n = ffi::make_object<SBlockNode>(*stmt.as<SBlockNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = SBlock(n);
    }
    // Check if it is the block corresponding to the parent scope
    if (block == scope_sref_->stmt) {
      // If so, put buffer allocation on the parent scope
      ffi::ObjectPtr<SBlockNode> n = ffi::make_object<SBlockNode>(*stmt.as<SBlockNode>());
      // In cache_inplace case, alloc_buffer may be already exits.
      if (info_->alloc.has_value()) {
        n->alloc_buffers.push_back(info_->alloc.value());
        stmt = SBlock(n);
      }
    } else {
      // Otherwise, update read regions and match_buffers
      // Only make this change if the block is one of the specified consumers.
      if (is_consumer) {
        // Use the updated block stmt
        ffi::Array<TensorRegion> reads = update_access_regions(stmt->reads);
        ffi::Array<MatchBufferRegion> match_buffers = update_match_buffers(stmt->match_buffers);
        if (!reads.same_as(stmt->reads) || !match_buffers.same_as(stmt->match_buffers)) {
          ffi::ObjectPtr<SBlockNode> n = ffi::make_object<SBlockNode>(*stmt.as<SBlockNode>());
          n->reads = std::move(reads);
          n->match_buffers = std::move(match_buffers);
          stmt = SBlock(n);
        }
      }
    }
    info_->block_reuse.Set(old_stmt, stmt);
    return stmt;
  }

  ffi::Array<PrimExpr> RewriteIndices(const ffi::Array<PrimExpr>& indices) {
    std::vector<PrimExpr> ret;
    for (size_t i = 0; i < indices.size(); ++i) {
      ret.push_back(ana_->Simplify(indices[i] - info_->cache_region->region[i]->min));
    }
    return ret;
  }

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) override {
    auto result = StmtExprMutator::Mutate_(op, inplace_mode);
    if (!result.IsUnchanged()) {
      op = ffi::AnyView(result).as<CallNode>();
      if (!op->unique()) inplace_mode = InplaceMode::kDisallow;
    }
    // Cache remapping can change pointer storage scope; the base Call hook preserves its type.
    if (!op->op.same_as(tirx::builtin::buffer_data()) || op->args.size() != 1) return result;
    PointerType type = op->args[0].as_or_throw<BufferVar>().DataPointerType();
    if (ffi::StructuralEqual()(op->ty, type)) return result;
    if (inplace_mode == InplaceMode::kAllow) {
      const_cast<CallNode*>(op)->ty = std::move(type);
      return result;
    }
    auto copy = ffi::make_object<CallNode>(*op);
    copy->ty = std::move(type);
    return Expr(std::move(copy));
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* load, InplaceMode inplace_mode) override {
    if (load->source.as_or_throw<tvm::tirx::BufferVar>().same_as(info_->read_buffer) &&
        current_block_consumes) {
      ffi::Array<PrimExpr> indices = load->indices;
      if (!cache_full_region_) {
        indices = RewriteIndices(load->indices);
      }
      TensorLoad node = ffi::GetRef<TensorLoad>(load);
      auto* n = node.CopyOnWrite();
      n->source = info_->write_buffer;
      n->indices = indices;
      return node;
    }
    auto indices = Mutate(load->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad node = ffi::GetRef<TensorLoad>(load);
    if (!indices.UnchangedOrSameAs(load->indices)) {
      node.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return node;
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionNode* op, InplaceMode inplace_mode) final {
    if (!op->source.as<BufferVar>()) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    auto region = Mutate(op->region).as_or_throw<UnchangedOr<ffi::Array<Range>>>();
    if (region.UnchangedOrSameAs(op->region)) return ffi::Unchanged();
    TensorRegion node = ffi::GetRef<TensorRegion>(op);
    node.CopyOnWrite()->region = std::move(region).ValueUnchecked();
    return node;
  }

  /*! \brief The parent scope of the insertion */
  const StmtSRef& scope_sref_;
  /*! \brief The info for inserting cache stage */
  CacheStageInfo* info_;
  /*! \brief Whether the most recently visited block is a specified consumer. */
  bool current_block_consumes;
  /*! \brief function to update read/write region of block being cache read.*/
  std::function<ffi::Array<TensorRegion>(ffi::Array<TensorRegion>)> update_access_regions;
  /*! \brief function to update match buffers of block being cache read.*/
  std::function<ffi::Array<MatchBufferRegion>(ffi::Array<MatchBufferRegion>)> update_match_buffers;
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
  using CacheReadRewriter::Mutate;
  using CacheReadRewriter::Mutate_;

  /*!
   * \brief Rewrite the AST and add a cache_read stage with the information provided.
   * \param scope_sref The parent scope of this mutation.
   * \param info The cache stage information.
   * \return The new AST rooting at the original parent scope.
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, ReindexCacheStageInfo* info) {
    auto rewriter = ffi::make_object<ReindexCacheReadRewriter>(scope_sref, info);
    return rewriter->Mutate(ffi::GetRef<Stmt>(scope_sref->stmt))
        .ValueOrUnchanged(ffi::GetRef<Stmt>(scope_sref->stmt));
  }

  explicit ReindexCacheReadRewriter(const StmtSRef& scope_sref, ReindexCacheStageInfo* info)
      : CacheReadRewriter(scope_sref, info) {
    new_indices_ = info->indices;
    update_access_regions = [&](ffi::Array<TensorRegion> reads) {
      ffi::Array<TensorRegion> new_reads;
      for (const TensorRegion& buf_region : reads) {
        if (buf_region->source.as_or_throw<tvm::tirx::BufferVar>().same_as(info_->read_buffer)) {
          Region region;
          for (const PrimExpr index : new_indices_) {
            region.push_back(Range::FromMinExtent(index, IntImm::Int32(1)));
          }
          new_reads.push_back(BufferRegion(info_->write_buffer, region));
        } else {
          new_reads.push_back(buf_region);
        }
      }
      return new_reads;
    };
    update_match_buffers = [&](const ffi::Array<MatchBufferRegion> match_buffers) {
      ffi::Array<MatchBufferRegion> new_match_buffers;
      for (const MatchBufferRegion& match_buffer_region : match_buffers) {
        TensorRegion source = match_buffer_region->source;
        if (source->source.as_or_throw<tvm::tirx::BufferVar>().same_as(info_->read_buffer)) {
          Region region;
          for (const PrimExpr index : new_indices_) {
            region.push_back(Range::FromMinExtent(index, IntImm::Int32(1)));
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

 private:
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* load, InplaceMode inplace_mode) final {
    if (load->source.as_or_throw<tvm::tirx::BufferVar>().same_as(info_->read_buffer) &&
        current_block_consumes) {
      TensorLoad node = ffi::GetRef<TensorLoad>(load);
      auto* n = node.CopyOnWrite();
      n->source = info_->write_buffer;
      n->indices = new_indices_;
      return node;
    }
    return CacheReadRewriter::Mutate_(load, inplace_mode);
  }

  /*! \brief The indices to use for new buffer. */
  ffi::Array<PrimExpr> new_indices_;
};

class ReindexCacheWriteRewriter;

/*! \brief Mutator for CacheWrite */
class CacheWriteRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

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
    auto rewriter = ffi::make_object<CacheWriteRewriter>(scope_sref, writer_block_sref, info,
                                                         cache_full_region);
    return rewriter->Mutate(ffi::GetRef<Stmt>(scope_sref->stmt))
        .ValueOrUnchanged(ffi::GetRef<Stmt>(scope_sref->stmt));
  }

  explicit CacheWriteRewriter(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                              CacheStageInfo* info, bool cache_full_region = true)
      : scope_sref_(scope_sref),
        writer_block_sref_(writer_block_sref),
        info_(info),
        cache_full_region_(cache_full_region) {
    VarRemapSet(info_->write_buffer, info_->read_buffer);
    auto update_region = [this](const Region& region, const Region& offset) -> Region {
      TVM_FFI_ICHECK_EQ(region.size(), offset.size());
      std::vector<Range> ret;
      for (size_t i = 0; i < region.size(); ++i) {
        ret.push_back(Range::FromMinExtent(ana_->Simplify(region[i]->min - offset[i]->min),
                                           region[i]->extent));
      }
      return ret;
    };

    update_access_regions = [this, update_region](ffi::Array<TensorRegion> regions) {
      if (cache_full_region_) {
        return ReplaceBuffer(regions, info_->write_buffer, info_->read_buffer);
      }

      ffi::Array<TensorRegion> ret;
      for (const TensorRegion& region : regions) {
        if (region->source.as_or_throw<tvm::tirx::BufferVar>().same_as(info_->write_buffer)) {
          ret.push_back(BufferRegion(info_->read_buffer,
                                     update_region(region->region, info_->cache_region->region)));
        } else {
          ret.push_back(region);
        }
      }
      return ret;
    };
    update_match_buffers = [this, update_region](ffi::Array<MatchBufferRegion> match_buffers) {
      if (cache_full_region_) {
        return ReplaceBuffer(match_buffers, info_->write_buffer, info_->read_buffer);
      }

      ffi::Array<MatchBufferRegion> ret;
      for (const MatchBufferRegion& match_buffer : match_buffers) {
        if (match_buffer->source->source.as_or_throw<tvm::tirx::BufferVar>().same_as(
                info_->write_buffer)) {
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

 private:
  UnchangedOr<Stmt> Mutate_(const ForNode* loop, InplaceMode inplace_mode) final {
    Stmt stmt =
        StmtExprMutator::Mutate_(loop, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(loop));
    // Check the insertion point
    if (loop == info_->loc_sref->stmt) {
      // Insert cache stage into the loop if it is the right place
      ffi::ObjectPtr<ForNode> n = ffi::make_object<ForNode>(*stmt.as<ForNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = Stmt(n);
    }
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const SBlockNode* block, InplaceMode inplace_mode) override {
    SBlock old_stmt = ffi::GetRef<SBlock>(block);

    // Check if this block is one of the specified cache consumers.
    // update the read buffer to the cache.
    for (StmtSRef consumer_sref : info_->consumer_blocks) {
      const SBlockNode* consumer_node = TVM_SREF_TO_SBLOCK(consumer_sref);
      SBlock consumer_block = ffi::GetRef<SBlock>(consumer_node);
      if (old_stmt.same_as(consumer_block)) {
        ffi::Array<TensorRegion> writes = update_access_regions(block->writes);
        ffi::Array<TensorRegion> reads = update_access_regions(block->reads);
        ffi::Array<MatchBufferRegion> match_buffers = update_match_buffers(block->match_buffers);
        if (!writes.same_as(block->writes) || !reads.same_as(block->reads) ||
            !match_buffers.same_as(block->match_buffers)) {
          SBlock new_consumer = old_stmt;
          SBlockNode* n = new_consumer.CopyOnWrite();
          n->writes = std::move(writes);
          n->reads = std::move(reads);
          n->match_buffers = std::move(match_buffers);
          n->body = Mutate(block->body, inplace_mode).ValueOrUnchanged(block->body);
          info_->block_reuse.Set(old_stmt, new_consumer);
          return new_consumer;
        }
        return old_stmt;
      }
    }

    // We only mutate the block which generates info->write_buffer
    if (block != writer_block_sref_->stmt && block != scope_sref_->stmt && !under_writer_block_) {
      return old_stmt;
    }

    // Mutate the body
    bool under_scope = under_writer_block_ || block == writer_block_sref_->stmt;
    std::swap(under_scope, under_writer_block_);
    // Cache accesses change storage; the original allocation still owns its source.
    auto input_node = ffi::make_object<SBlockNode>(*block);
    input_node->alloc_buffers.clear();
    SBlock input(std::move(input_node));
    SBlock stmt = StmtExprMutator::Mutate_(input.get(), InplaceMode::kDisallow)
                      .ValueOrUnchanged(input)
                      .as_or_throw<SBlock>();
    stmt.CopyOnWrite()->alloc_buffers = block->alloc_buffers;
    std::swap(under_scope, under_writer_block_);

    // Find the insertion point
    if (block == info_->loc_sref->stmt) {
      ffi::ObjectPtr<SBlockNode> n = ffi::make_object<SBlockNode>(*stmt.as<SBlockNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      stmt = SBlock(n);
    }
    // Put buffer allocation on the parent scope
    if (block == scope_sref_->stmt) {
      ffi::ObjectPtr<SBlockNode> n = ffi::make_object<SBlockNode>(*stmt.as<SBlockNode>());
      // In cache_inplace case, alloc_buffer may be already exits.
      if (info_->alloc.has_value()) {
        n->alloc_buffers.push_back(info_->alloc.value());
        stmt = SBlock(n);
      }
    } else {
      // Since cache_write changes the block, we need to update the buffer it writes
      auto writes = update_access_regions(block->writes);
      auto reads = update_access_regions(block->reads);
      auto match_buffers = update_match_buffers(block->match_buffers);
      if (!writes.same_as(block->writes) || !reads.same_as(block->reads) ||
          !match_buffers.same_as(block->match_buffers)) {
        ffi::ObjectPtr<SBlockNode> n = ffi::make_object<SBlockNode>(*stmt.as<SBlockNode>());
        n->writes = std::move(writes);
        n->reads = std::move(reads);
        n->match_buffers = std::move(match_buffers);
        stmt = SBlock(n);
      }
    }
    info_->block_reuse.Set(old_stmt, stmt);
    return stmt;
  }

  ffi::Array<PrimExpr> RewriteIndices(const ffi::Array<PrimExpr>& indices) {
    std::vector<PrimExpr> ret;
    for (size_t i = 0; i < indices.size(); ++i) {
      ret.push_back(ana_->Simplify(indices[i] - info_->cache_region->region[i]->min));
    }
    return ret;
  }

  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* store, InplaceMode inplace_mode) override {
    bool rewrite_buffer = store->buffer.same_as(info_->write_buffer);
    auto value = Mutate(store->value);
    auto indices = Mutate(store->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    BufferStore stmt = ffi::GetRef<BufferStore>(store);
    if (!value.UnchangedOrSameAs(store->value) || !indices.UnchangedOrSameAs(store->indices)) {
      auto* n = stmt.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(store->value);
      n->indices = std::move(indices).ValueOrUnchanged(store->indices);
    }
    if (rewrite_buffer) {
      BufferStoreNode* n = stmt.CopyOnWrite();
      n->buffer = info_->read_buffer;
      if (!cache_full_region_) {
        n->indices = RewriteIndices(n->indices);
      }
      return stmt;
    } else {
      return stmt;
    }
  }

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) override {
    auto result = StmtExprMutator::Mutate_(op, inplace_mode);
    if (!result.IsUnchanged()) {
      op = ffi::AnyView(result).as<CallNode>();
      if (!op->unique()) inplace_mode = InplaceMode::kDisallow;
    }
    // Cache remapping can change pointer storage scope; the base Call hook preserves its type.
    if (!op->op.same_as(tirx::builtin::buffer_data()) || op->args.size() != 1) return result;
    PointerType type = op->args[0].as_or_throw<BufferVar>().DataPointerType();
    if (ffi::StructuralEqual()(op->ty, type)) return result;
    if (inplace_mode == InplaceMode::kAllow) {
      const_cast<CallNode*>(op)->ty = std::move(type);
      return result;
    }
    auto copy = ffi::make_object<CallNode>(*op);
    copy->ty = std::move(type);
    return Expr(std::move(copy));
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* load, InplaceMode inplace_mode) override {
    if (load->source.as_or_throw<tvm::tirx::BufferVar>().same_as(info_->write_buffer)) {
      ffi::Array<PrimExpr> indices = load->indices;
      if (!cache_full_region_) {
        indices = RewriteIndices(indices);
      }
      TensorLoad node = ffi::GetRef<TensorLoad>(load);
      auto* n = node.CopyOnWrite();
      n->source = info_->read_buffer;
      n->indices = indices;
      return node;
    }
    auto indices = Mutate(load->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad node = ffi::GetRef<TensorLoad>(load);
    if (!indices.UnchangedOrSameAs(load->indices)) {
      node.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return node;
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionNode* op, InplaceMode inplace_mode) final {
    if (!op->source.as<BufferVar>()) {
      return StmtExprMutator::Mutate_(op, inplace_mode);
    }
    auto region = Mutate(op->region).as_or_throw<UnchangedOr<ffi::Array<Range>>>();
    if (region.UnchangedOrSameAs(op->region)) return ffi::Unchanged();
    TensorRegion node = ffi::GetRef<TensorRegion>(op);
    node.CopyOnWrite()->region = std::move(region).ValueUnchecked();
    return node;
  }

  /*! \brief The parent scope of the insertion. */
  const StmtSRef& scope_sref_;
  /*! \brief The parent scope of the insertion. */
  const StmtSRef& writer_block_sref_;
  /*! \brief The info for inserting cache stage. */
  CacheStageInfo* info_;
  /*! \brief Whether the current node is under the given block. */
  bool under_writer_block_{false};
  /*! \brief function to update read/write region of block being cache write.*/
  std::function<ffi::Array<TensorRegion>(ffi::Array<TensorRegion>)> update_access_regions;
  /*! \brief function to update match buffers of block being cache write.*/
  std::function<ffi::Array<MatchBufferRegion>(ffi::Array<MatchBufferRegion>)> update_match_buffers;
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
  using CacheWriteRewriter::Mutate;
  using CacheWriteRewriter::Mutate_;

  /*!
   * \brief Rewrite the AST and add a cache_write stage with the information provided.
   * \param scope_sref The parent scope of this mutation.
   * \param writer_block_sref The only writer block in the scope.
   * \param info The cache stage information.
   * \return The new AST rooting at the original parent scope.
   */
  static Stmt Rewrite(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                      ReindexCacheStageInfo* info) {
    auto rewriter =
        ffi::make_object<ReindexCacheWriteRewriter>(scope_sref, writer_block_sref, info);
    return rewriter->Mutate(ffi::GetRef<Stmt>(scope_sref->stmt))
        .ValueOrUnchanged(ffi::GetRef<Stmt>(scope_sref->stmt));
  }

  explicit ReindexCacheWriteRewriter(const StmtSRef& scope_sref, const StmtSRef& writer_block_sref,
                                     ReindexCacheStageInfo* info)
      : CacheWriteRewriter(scope_sref, writer_block_sref, info) {
    new_indices_ = info->indices;
    update_access_regions = [&](ffi::Array<TensorRegion> reads) {
      ffi::Array<TensorRegion> new_reads;
      for (const TensorRegion& buf_region : reads) {
        if (buf_region->source.as_or_throw<tvm::tirx::BufferVar>().same_as(info_->write_buffer)) {
          Region region;
          for (const PrimExpr index : new_indices_) {
            region.push_back(Range::FromMinExtent(index, IntImm::Int32(1)));
          }
          new_reads.push_back(BufferRegion(info_->read_buffer, region));
        } else {
          new_reads.push_back(buf_region);
        }
      }
      return new_reads;
    };
    update_match_buffers = [&](const ffi::Array<MatchBufferRegion> match_buffers) {
      ffi::Array<MatchBufferRegion> new_match_buffers;
      for (const MatchBufferRegion& match_buffer_region : match_buffers) {
        TensorRegion source = match_buffer_region->source;
        if (source->source.as_or_throw<tvm::tirx::BufferVar>().same_as(info_->write_buffer)) {
          Region region;
          for (const PrimExpr index : new_indices_) {
            region.push_back(Range::FromMinExtent(index, IntImm::Int32(1)));
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

 private:
  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* store, InplaceMode inplace_mode) final {
    bool rewrite_buffer = store->buffer.same_as(info_->write_buffer);
    auto value = Mutate(store->value);
    auto indices = Mutate(store->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    BufferStore stmt = ffi::GetRef<BufferStore>(store);
    if (!value.UnchangedOrSameAs(store->value) || !indices.UnchangedOrSameAs(store->indices)) {
      auto* n = stmt.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(store->value);
      n->indices = std::move(indices).ValueOrUnchanged(store->indices);
    }
    if (rewrite_buffer) {
      BufferStoreNode* n = stmt.CopyOnWrite();
      n->buffer = info_->read_buffer;
      n->indices = new_indices_;
      return stmt;
    } else {
      return stmt;
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* load, InplaceMode inplace_mode) final {
    if (load->source.as_or_throw<tvm::tirx::BufferVar>().same_as(info_->write_buffer)) {
      TensorLoad node = ffi::GetRef<TensorLoad>(load);
      auto* n = node.CopyOnWrite();
      n->source = info_->read_buffer;
      n->indices = new_indices_;
      return node;
    }
    auto indices = Mutate(load->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad node = ffi::GetRef<TensorLoad>(load);
    if (!indices.UnchangedOrSameAs(load->indices)) {
      node.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return node;
  }

  /*! \brief The indices to use for new buffer. */
  ffi::Array<PrimExpr> new_indices_;
};

/*!
 * \brief Create a new buffer by change the shape with block iters to be used as the reindex buffer
 * \param buffer The given buffer.
 * \param block_iters The block iters.
 * \param covered Set of block iter vars covered by the buffer access indices
 * \return The new buffer with target shape.
 */
BufferVar CreateReindexBuffer(const BufferVar& buffer, const ffi::Array<IterVar>& block_iters,
                              const std::unordered_set<Var>& covered) {
  ffi::ObjectPtr<BufferTypeNode> new_buffer = CopyBufferType(buffer);
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
  return RebuildBufferVar(buffer, std::move(new_buffer), buffer.name() + "_reindex");
}

/*!
 * \brief The schedule error that the target is not a leaf block.
 */
class NotLeafBlockError : public ScheduleErrorContextObj {
 public:
  NotLeafBlockError(IRModule mod, SBlock block) : mod_(std::move(mod)), block_(std::move(block)) {}
  ffi::String FastErrorString() const final {
    return "ScheduleError: The target block is not a leaf block.";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The target block {0} is not a leaf block.";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }
  IRModule mod_;
  SBlock block_;
};

/*! \brief The schedule error that the buffer access is invalid for reindex. */
class InvalidBufferAccessError : public ScheduleErrorContextObj {
 public:
  enum class ErrorKind {
    kNoAccess,         // buffer access not found
    kNonUniqueAccess,  // multiple buffer accesses with different indices
    kOpaqueAccess,     // opaque access to the buffer
  };

  InvalidBufferAccessError(IRModule mod, BufferVar buffer, SBlock block, ErrorKind kind)
      : mod_(std::move(mod)), buffer_(std::move(buffer)), block_(std::move(block)), kind_(kind) {}
  ffi::String FastErrorString() const final {
    return "ScheduleError: The target buffer should be accessed via TensorLoad or BufferStore. The "
           "indices should be the same if there are multiple accesses to the target buffer.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "The target buffer " << buffer_.name()
       << " should be accessed in the leaf block {0} via TensorLoad or BufferStore. The indices "
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
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }

 private:
  IRModule mod_;
  BufferVar buffer_;
  SBlock block_;
  ErrorKind kind_;
};

/*! \brief Collect the related Load/Store to reindex */
class ReIndexCollector : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  static ffi::Array<PrimExpr> Collect(const IRModule& mod, const BufferVar& buffer,
                                      const SBlock& block) {
    auto collector = ffi::make_object<ReIndexCollector>(mod, buffer, block);
    collector->Visit(block->body);
    if (!collector->buffer_access_indices_.has_value()) {
      throw MakeScheduleError<InvalidBufferAccessError>(
          mod, buffer, block, InvalidBufferAccessError::ErrorKind::kNoAccess);
    }
    return collector->buffer_access_indices_.value();
  }

  explicit ReIndexCollector(const IRModule& mod, const BufferVar& buffer, const SBlock& block)
      : mod_(mod), buffer_(buffer), block_(block) {}

 private:
  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadNode* load) final {
    for (const PrimExpr& index : load->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
    }
    if (load->source.as_or_throw<tvm::tirx::BufferVar>().same_as(buffer_)) {
      CheckAndUpdateBufferAccessIndices(load->indices);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockNode* block) final {
    // no sub-blocks under this block
    throw MakeScheduleError<NotLeafBlockError>(mod_, block_);
  }

  ffi::Optional<VisitInterrupt> Visit_(const BufferStoreNode* store) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(store->value));
    for (const PrimExpr& index : store->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
    }
    if (store->buffer.same_as(buffer_)) {
      CheckAndUpdateBufferAccessIndices(store->indices);
    }
    return std::nullopt;
  }

  void CheckAndUpdateBufferAccessIndices(const ffi::Array<PrimExpr> indices) {
    if (!buffer_access_indices_.has_value()) {
      buffer_access_indices_ = indices;
      return;
    } else if (!std::equal(buffer_access_indices_.value().begin(),
                           buffer_access_indices_.value().end(), indices.begin(), indices.end(),
                           prim::ExprDeepEqual())) {
      throw MakeScheduleError<InvalidBufferAccessError>(
          mod_, buffer_, block_, InvalidBufferAccessError::ErrorKind::kNonUniqueAccess);
    }
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarNode* var) final {
    if (def_region_kind() != kTVMFFIDefRegionKindNone) return std::nullopt;
    if (var == buffer_.get()) {
      throw MakeScheduleError<InvalidBufferAccessError>(
          mod_, buffer_, block_, InvalidBufferAccessError::ErrorKind::kOpaqueAccess);
    }
    return std::nullopt;
  }
  /*! \brief The IR module */
  IRModule mod_;
  /*! \brief The buffer to rewrite */
  BufferVar buffer_;
  /*! \brief The block to visit */
  SBlock block_;
  /*! \brief The indices of buffer acess to rewrite */
  ffi::Optional<ffi::Array<PrimExpr>> buffer_access_indices_;
};

/*! \brief Mutator of ReIndex */
class ReIndexRewriter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  static Stmt Rewrite(const StmtSRef& scope_sref, const StmtSRef& block_sref, CacheStageInfo* info,
                      const std::unordered_set<Var>& covered) {
    auto rewriter = ffi::make_object<ReIndexRewriter>(block_sref, info, covered);
    return rewriter->Mutate(ffi::GetRef<Stmt>(scope_sref->stmt))
        .ValueOrUnchanged(ffi::GetRef<Stmt>(scope_sref->stmt));
  }

  explicit ReIndexRewriter(const StmtSRef& block_sref, CacheStageInfo* info,
                           const std::unordered_set<Var>& covered)
      : block_sref_(block_sref), info_(info), covered_(covered) {
    new_buffer_ = info->alloc.value();
    old_buffer_ = info->read_buffer.same_as(new_buffer_) ? info->write_buffer : info->read_buffer;
  }

 private:
  UnchangedOr<Stmt> Mutate_(const SBlockNode* block, InplaceMode inplace_mode) final {
    SBlock old_stmt = ffi::GetRef<SBlock>(block);
    if (is_scope_) {
      is_scope_ = false;
      SBlock stmt = StmtExprMutator::Mutate_(block, inplace_mode)
                        .ValueOrUnchanged(ffi::GetRef<Stmt>(block))
                        .as_or_throw<SBlock>();
      // Insert cache stage into the loop
      ffi::ObjectPtr<SBlockNode> n = ffi::make_object<SBlockNode>(*stmt.as<SBlockNode>());
      n->body = InsertCacheStage(n->body, info_->loc_pos, info_->cache_stage);
      n->alloc_buffers.push_back(info_->alloc.value());
      stmt = SBlock(n);
      info_->block_reuse.Set(old_stmt, stmt);
      return stmt;
    }

    // Visiting the blokc being reindexed
    if (block == block_sref_->stmt) {
      // Collect the updated indices and regions
      for (const IterVar& iter : block->iter_vars) {
        if (covered_.count(iter->var)) {
          indices_.push_back(iter->var);
          region_.push_back(Range::FromMinExtent(iter->var, IntImm(iter->var.ty(), 1)));
        }
      }
      SBlock stmt = StmtExprMutator::Mutate_(block, inplace_mode)
                        .ValueOrUnchanged(ffi::GetRef<Stmt>(block))
                        .as_or_throw<SBlock>();
      // Update block reads/writes to use the intermediate reindex buffer
      auto writes =
          ReplaceBufferRegion(block->writes, old_buffer_, BufferRegion(new_buffer_, region_));
      auto reads =
          ReplaceBufferRegion(block->reads, old_buffer_, BufferRegion(new_buffer_, region_));
      auto match_buffers = ReplaceBufferRegion(block->match_buffers, old_buffer_,
                                               BufferRegion(new_buffer_, region_));
      if (!writes.same_as(block->writes) || !reads.same_as(block->reads) ||
          !match_buffers.same_as(block->match_buffers)) {
        ffi::ObjectPtr<SBlockNode> n = ffi::make_object<SBlockNode>(*stmt.as<SBlockNode>());
        n->writes = std::move(writes);
        n->reads = std::move(reads);
        n->match_buffers = std::move(match_buffers);
        stmt = SBlock(n);
      }
      info_->block_reuse.Set(old_stmt, stmt);
      return stmt;
    }
    return old_stmt;
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
  TensorLoad VisitBufferAccess(TensorLoad node) {
    if (node->source.as_or_throw<tvm::tirx::BufferVar>().same_as(old_buffer_)) {
      auto* n = node.CopyOnWrite();
      n->source = new_buffer_;
      n->indices = indices_;
    }
    return node;
  }
  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) final {
    auto value = Mutate(op->value);
    auto indices = Mutate(op->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    BufferStore buffer_store = ffi::GetRef<BufferStore>(op);
    if (!value.UnchangedOrSameAs(op->value) || !indices.UnchangedOrSameAs(op->indices)) {
      auto* n = buffer_store.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(op->value);
      n->indices = std::move(indices).ValueOrUnchanged(op->indices);
    }
    return VisitBufferAccess(std::move(buffer_store));
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    auto indices = Mutate(op->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad buffer_load = ffi::GetRef<TensorLoad>(op);
    if (!indices.UnchangedOrSameAs(op->indices)) {
      buffer_load.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return VisitBufferAccess(std::move(buffer_load));
  }

  /*! \brief The parent scope of the insertion. */
  const StmtSRef& block_sref_;
  /*! \brief The info for inserting reindex stage. */
  CacheStageInfo* info_;
  /*! \brief Whether old block var is covered in the indices */
  const std::unordered_set<Var>& covered_;
  /*! \brief Whether the current block is scope block */
  bool is_scope_{true};
  /*! \brief The  buffer to be replaced */
  BufferVar old_buffer_;
  /*! \brief The reindex buffer */
  BufferVar new_buffer_;
  /*! \brief The new indices */
  ffi::Array<PrimExpr> indices_;
  /*! \brief The new region */
  Region region_;
};

void CheckRegionCover(const ScheduleState& self, StmtSRef scope_root, BufferVar read_buffer) {
  class NotRegionCoverError : public ScheduleErrorContextObj {
   public:
    explicit NotRegionCoverError(IRModule mod, SBlock block) : mod_(mod), block_(block) {}
    IRModule mod() const final { return mod_; }
    ffi::String FastErrorString() const final {
      return "ScheduleError: The scope root's region cover is not complete.";
    }
    ffi::String DetailRenderTemplate() const final {
      return R"(The scope {0} 's region cover is not complete.
The region cover property require to hold for every of its child blocks
)";
    }
    ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }
    IRModule mod_;
    SBlock block_;
  };

  for (const auto& child_block_sref : GetChildBlocks(self, scope_root)) {
    const SBlockNode* child_block = TVM_SREF_TO_SBLOCK(child_block_sref);
    for (const TensorRegion& region : child_block->reads) {
      if (region->source.as_or_throw<tvm::tirx::BufferVar>().same_as(read_buffer)) {
        if (!self->block_info.at(child_block_sref).region_cover) {
          const SBlockNode* block = TVM_SREF_TO_SBLOCK(scope_root);
          throw MakeScheduleError<NotRegionCoverError>(self->mod, ffi::GetRef<SBlock>(block));
        }
      }
    }
  }
}

/******** Implementation ********/

StmtSRef CacheRead(ScheduleState self, const StmtSRef& block_sref, int read_buffer_index,
                   const ffi::String& storage_scope, const ffi::Array<StmtSRef> consumer_blocks) {
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
  const SBlockNode* block = TVM_SREF_TO_SBLOCK(block_sref);
  BufferVar read_buffer = GetNthAccessBuffer(self, ffi::GetRef<SBlock>(block), read_buffer_index,
                                             BufferIndexType::kRead);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);
  // Check required region cover for cache_read
  CheckRegionCover(self, scope_sref, read_buffer);
  const SBlockNode* scope_block = TVM_SREF_TO_SBLOCK(scope_sref);

  // Step 2. Create CacheStageInfo
  CacheStageInfo info;
  info.read_buffer = read_buffer;

  // info.consumer_blocks indicates which buffers should consume the cache.
  for (auto consumer : consumer_blocks) {
    info.consumer_blocks.insert(consumer);
    for (auto child : GetChildBlocks(self, consumer)) {
      info.consumer_blocks.insert(child);
    }
  }

  // Step 3. Update cache stage info.
  TensorRegion cache_region{nullptr};
  if (ffi::Optional<StmtSRef> _write_block_sref =
          GetOnlyWriteBlock(self, scope_sref, read_buffer)) {
    // Case 1. The buffer is written inside the block.
    StmtSRef write_block_sref = _write_block_sref.value();
    const SBlockNode* write_block = TVM_SREF_TO_SBLOCK(write_block_sref);
    // Find the producing region
    TensorRegion region = GetBufferRegionFromBuffer(write_block->writes, read_buffer).value();
    StmtSRef parent_sref = ffi::GetRef<StmtSRef>(write_block_sref->parent);

    // Detect insert position
    CacheLocDetector::Detect</*is_cache_read=*/true>(self, write_block_sref, scope_sref, &info);
    cache_region = RelaxBufferRegion(self, region, write_block_sref, parent_sref, info.loc_sref);
  } else {
    // Case 2. The buffer is the input block for the scope.
    info.loc_sref = scope_sref;
    info.loc_pos = 0;
    // When a nested block gates the actual read with T.where, the consumer block's own
    // predicate is trivially true, so the scope-block read annotation covers the full loop
    // range. Collect nested-read predicates and, if any are non-trivial, relax the consumer
    // block's read region under that predicate to get a tighter cache allocation.
    // Without a nested predicate we fall back to scope_block->reads (which preserves the
    // original buffer's dtype in its extents, e.g. int64 shapes).
    ffi::Optional<TensorRegion> read_region_opt =
        GetBufferRegionFromBuffer(block->reads, read_buffer);
    PrimExpr nested_pred = read_region_opt ? CollectNestedBlockPredicates(block->body, read_buffer,
                                                                          BufferIndexType::kRead)
                                           : IntImm::Bool(true);
    if (read_region_opt && !is_one(nested_pred) && block_sref->parent != nullptr) {
      StmtSRef parent_sref = ffi::GetRef<StmtSRef>(block_sref->parent);
      cache_region = RelaxBufferRegion(self, read_region_opt.value(), block_sref, parent_sref,
                                       scope_sref, nested_pred);
    } else if (ffi::Optional<TensorRegion> scope_region =
                   GetBufferRegionFromBuffer(scope_block->reads, read_buffer)) {
      cache_region = scope_region.value();
    } else {
      cache_region = FullBufferRegion(read_buffer);
    }
  }

  // Step 4. Making new cache stage block and rewrite readers.
  bool cache_full_region = info.loc_sref->StmtAs<SBlockNode>() == nullptr ||
                           !AllConsumersUnderStmt(self, read_buffer, scope_sref, info.loc_sref);
  info.cache_region = cache_region;
  info.write_buffer = WithScope(read_buffer, storage_scope);
  if (!cache_full_region) {
    auto write_buffer = CopyBufferType(info.write_buffer);
    std::vector<PrimExpr> shape;
    for (auto cache_range : info.cache_region->region) {
      shape.push_back(cache_range->extent);
    }
    write_buffer->shape = std::move(shape);
    info.write_buffer = RebuildBufferVar(info.write_buffer, std::move(write_buffer));
  }
  info.alloc = info.write_buffer;

  SBlock cache_read_stage =
      MakeCacheStage(/*cache_region=*/cache_region, /*info=*/&info,
                     /*storage_scope=*/storage_scope, /*cache_full_region=*/cache_full_region);
  Stmt new_scope = CacheReadRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info,
                                              /*cache_full_region=*/cache_full_region);

  // Step 5. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_read_stage.get());
  SBlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.stage_pipeline = true;
  return result_block_sref;
}

StmtSRef CacheWrite(ScheduleState self, const StmtSRef& block_sref, int write_buffer_index,
                    const ffi::String& storage_scope, const ffi::Array<StmtSRef> consumer_blocks) {
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
  const SBlockNode* block = TVM_SREF_TO_SBLOCK(block_sref);
  BufferVar write_buffer = GetNthAccessBuffer(self, ffi::GetRef<SBlock>(block), write_buffer_index,
                                              BufferIndexType::kWrite);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);

  // Step 2. Creating CacheStageInfo
  CacheStageInfo info;
  // Create the corresponding buffer to be written, i.e. result of cache_write
  info.write_buffer = write_buffer;

  // info.consumer_blocks indicates which buffers should consume the cache.
  for (auto consumer : consumer_blocks) {
    info.consumer_blocks.insert(consumer);
    for (auto child : GetChildBlocks(self, consumer)) {
      info.consumer_blocks.insert(child);
    }
  }

  // Step 3. Check the only writer block.
  ffi::Optional<StmtSRef> only_write_block = GetOnlyWriteBlock(self, scope_sref, write_buffer);
  TVM_FFI_ICHECK(only_write_block.has_value());
  TVM_FFI_ICHECK_EQ(block_sref.get(), only_write_block.value().get());

  // Step 4. Find the producing region and insert position
  TensorRegion region = GetBufferRegionFromBuffer(block->writes, write_buffer).value();
  // Detect insert position
  CacheLocDetector::Detect</*is_cache_read=*/false>(self, block_sref, scope_sref, &info);
  // Collect predicates from any nested blocks that gate the actual write (e.g. T.where on an
  // inner block). The outer block's own predicate may be trivially true even though the write
  // is restricted by a nested predicate, so we OR them together for a tighter region estimate.
  PrimExpr nested_write_pred =
      CollectNestedBlockPredicates(block->body, write_buffer, BufferIndexType::kWrite);
  TensorRegion cache_region;
  if (block_sref->parent != nullptr) {
    StmtSRef parent_sref = ffi::GetRef<StmtSRef>(block_sref->parent);
    cache_region =
        RelaxBufferRegion(self, region, block_sref, parent_sref, info.loc_sref, nested_write_pred);
  } else {
    // Root block: no enclosing loops to relax over, use the write region directly.
    cache_region = region;
  }

  bool cache_full_region = info.loc_sref->StmtAs<SBlockNode>() == nullptr ||
                           !AllConsumersUnderStmt(self, write_buffer, scope_sref, info.loc_sref);
  info.cache_region = cache_region;
  info.read_buffer = WithScope(write_buffer, storage_scope);
  if (!cache_full_region) {
    auto read_buffer_type = CopyBufferType(info.read_buffer);
    std::vector<PrimExpr> shape;
    for (auto cache_range : info.cache_region->region) {
      shape.push_back(cache_range->extent);
    }
    read_buffer_type->shape = std::move(shape);
    info.read_buffer = RebuildBufferVar(info.read_buffer, std::move(read_buffer_type));
  }
  info.alloc = info.read_buffer;

  // Step 5. Making new cache stage block and rewrite readers.
  SBlock cache_write_stage =
      MakeCacheStage(/*cache_region=*/cache_region, /*info=*/&info,
                     /*storage_scope=*/storage_scope, /*cache_full_region=*/cache_full_region);
  Stmt new_scope = CacheWriteRewriter::Rewrite(/*scope_sref=*/scope_sref,
                                               /*writer_block_sref=*/block_sref, /*info=*/&info,
                                               /*cache_full_region=*/cache_full_region);

  // Step 6. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_write_stage.get());
  SBlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.stage_pipeline = true;
  return result_block_sref;
}

ffi::Array<StmtSRef> GetLoopsUnderScope(const StmtSRef& block_sref, const StmtSRef& top_sref) {
  std::vector<StmtSRef> result;
  for (StmtSRefNode* parent = block_sref->parent; parent && parent->stmt->IsInstance<ForNode>();
       parent = parent->parent) {
    if (parent == top_sref.get()) break;
    result.push_back(ffi::GetRef<StmtSRef>(parent));
  }
  return {result.rbegin(), result.rend()};
}

/*!
 * \brief The schedule error that block iter vars appears in old buffer and new
 * allocated cache buffer does not match.
 */
class ReindexCacheReadWriteNotMatchError : public ScheduleErrorContextObj {
 public:
  ReindexCacheReadWriteNotMatchError(IRModule mod, SBlock block, Var var,
                                     ffi::Array<PrimExpr> old_indices,
                                     ffi::Array<PrimExpr> new_indices, bool is_cache_read,
                                     bool appears_in_old)
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
  ffi::String FastErrorString() const final {
    return "ScheduleError: the block itervars appeared in lhs and rhs of reindex cache stage do "
           "not match.";
  }

  ffi::String DetailRenderTemplate() const final {
    std::stringstream s;
    s << "Error when applying " << primitive_name_ << " on block {0}, the block itervar " << var_
      << " appears in " << appears_indices_ << ", but not in " << other_indices_ << ".";
    return ffi::String(s.str());
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_}; }
  IRModule mod_;
  ffi::String primitive_name_;
  SBlock block_;
  Var var_;
  ffi::Array<PrimExpr> appears_indices_;
  ffi::Array<PrimExpr> other_indices_;
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
    const ffi::String& storage_scope, const IndexMap& index_map, const SBlock& block,
    const SBlockRealize& realize, const BufferVar& old_buffer, const TensorRegion& cache_region) {
  arith::Analyzer analyzer;
  ffi::Array<PrimExpr> block_iter_vars, block_shape;
  for (const IterVar& iter_var : block->iter_vars) {
    block_iter_vars.push_back(iter_var);
    block_shape.push_back(iter_var->dom->extent);
  }
  ffi::Array<PrimExpr> new_indices = index_map->MapIndices(block_iter_vars, analyzer);
  ffi::Array<PrimExpr> new_shape = index_map->MapShape(block_shape, analyzer);
  info->indices = new_indices;

  // Step 5. Update CacheTouchedInfo
  auto collector_old = ffi::make_object<VarUseDefAnalyzer>(ffi::Array<Var>{});
  ffi::Array<PrimExpr> old_indices;
  for (const Range& range : cache_region->region) {
    collector_old->Visit(range->min);
    old_indices.push_back(range->min);
  }

  auto collector_new = ffi::make_object<VarUseDefAnalyzer>(ffi::Array<Var>{});
  for (const PrimExpr& idx : new_indices) {
    collector_new->Visit(idx);
  }

  auto collector_iter_values = ffi::make_object<VarUseDefAnalyzer>(ffi::Array<Var>{});
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    const IterVar& block_iter_var = block->iter_vars[i];
    const PrimExpr& block_iter_value = realize->iter_values[i];
    bool appears_in_new = collector_new->use_count_.count(block_iter_var->var.get());
    bool appears_in_old = collector_old->use_count_.count(block_iter_var->var.get());
    if (appears_in_new != appears_in_old) {
      throw MakeScheduleError<ReindexCacheReadWriteNotMatchError>(
          mod, block, block_iter_var->var, old_indices, new_indices, is_cache_read, appears_in_old);
    }
    if (appears_in_new) {
      info->block_iter_vars.push_back(block_iter_var);
      info->block_iter_values.push_back(block_iter_value);
      collector_iter_values->Visit(block_iter_value);
    }
  }

  for (const StmtSRef& loop_sref : GetLoopsUnderScope(block_sref, info->loc_sref)) {
    const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
    if (collector_iter_values->use_count_.count(loop->loop_var.get())) {
      info->loop_vars.push_back(loop->loop_var);
      info->loop_ranges.push_back(Range::FromMinExtent(loop->min, loop->extent));
    }
  }

  // Create new buffer
  ffi::ObjectPtr<BufferTypeNode> new_buffer = CopyBufferType(old_buffer);
  new_buffer->storage_scope = storage_scope;
  new_buffer->shape = new_shape;
  BufferVar rebuilt =
      RebuildBufferVar(old_buffer, std::move(new_buffer), old_buffer.name() + "_" + storage_scope);

  if (is_cache_read) {
    info->write_buffer = rebuilt;
    info->alloc = info->write_buffer;
  } else {
    info->read_buffer = rebuilt;
    info->alloc = info->read_buffer;
  }
}

/*! \brief Check whether given cache_region is a single point access. */
template <bool is_cache_read>
void CheckSinglePoint(ScheduleState self, const SBlock& block, const TensorRegion& cache_region) {
  bool single_point = true;
  for (const Range& range : cache_region->region) {
    const auto* ext_int = range->extent.as<IntImmNode>();
    if (!ext_int || ext_int->value != 1) {
      single_point = false;
    }
  }
  if (!single_point) {
    throw MakeScheduleError<NotSinglePointAccess>(self->mod, block, cache_region, is_cache_read);
  }
}

StmtSRef ReindexCacheRead(ScheduleState self, const StmtSRef& block_sref, int read_buffer_index,
                          const ffi::String& storage_scope, const IndexMap& index_map) {
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
  SBlock block = ffi::GetRef<SBlock>(TVM_SREF_TO_SBLOCK(block_sref));
  SBlockRealize realize = GetSBlockRealize(self, block_sref);
  BufferVar read_buffer =
      GetNthAccessBuffer(self, block, read_buffer_index, BufferIndexType::kRead);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);

  // Step 2. Create CacheStageInfo
  ReindexCacheStageInfo info;
  info.read_buffer = read_buffer;
  info.consumer_blocks.insert(block_sref);

  // Step 3. Update cache stage info.
  ffi::Optional<TensorRegion> maybe_region = GetBufferRegionFromBuffer(block->reads, read_buffer);
  TVM_FFI_ICHECK(maybe_region.has_value())
      << read_buffer << " should appear in the block's read region: " << block->reads;
  TensorRegion cache_region = maybe_region.value();
  if (ffi::Optional<StmtSRef> _write_block_sref =
          GetOnlyWriteBlock(self, scope_sref, read_buffer)) {
    // Case 1. The buffer is written inside the block.
    StmtSRef write_block_sref = _write_block_sref.value();
    // Find the producing region
    StmtSRef parent_sref = ffi::GetRef<StmtSRef>(write_block_sref->parent);
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
  SBlock cache_read_stage =
      MakeReindexCacheStage</*is_cache_read=*/true>(/*cache_region=*/cache_region,
                                                    /*info=*/&info,
                                                    /*storage_scope=*/storage_scope);
  Stmt new_scope = ReindexCacheReadRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info);

  // Step 7. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_read_stage.get());
  SBlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.stage_pipeline = true;
  return result_block_sref;
}

StmtSRef ReindexCacheWrite(ScheduleState self, const StmtSRef& block_sref, int write_buffer_index,
                           const ffi::String& storage_scope, const IndexMap& index_map) {
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
  SBlock block = ffi::GetRef<SBlock>(TVM_SREF_TO_SBLOCK(block_sref));
  SBlockRealize realize = GetSBlockRealize(self, block_sref);
  BufferVar write_buffer =
      GetNthAccessBuffer(self, block, write_buffer_index, BufferIndexType::kWrite);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);

  // Step 2. Creating CacheStageInfo
  ReindexCacheStageInfo info;
  info.write_buffer = write_buffer;

  // Step 3. Check the only writer block.
  ffi::Optional<StmtSRef> only_write_block = GetOnlyWriteBlock(self, scope_sref, write_buffer);
  TVM_FFI_ICHECK(only_write_block.has_value());
  TVM_FFI_ICHECK_EQ(block_sref.get(), only_write_block.value().get());

  // Step 4. Find the producing region and insert position
  ffi::Optional<TensorRegion> maybe_region = GetBufferRegionFromBuffer(block->writes, write_buffer);
  TVM_FFI_ICHECK(maybe_region.has_value())
      << write_buffer << " should appear in the block's write region";
  StmtSRef parent_sref = ffi::GetRef<StmtSRef>(block_sref->parent);
  // Detect insert position
  CacheLocDetector::Detect</*is_cache_read=*/false>(self, block_sref, scope_sref, &info);
  TensorRegion cache_region = maybe_region.value();

  CollectReindexCacheStageInfoAndCreateBuffer</*is_cache_read=*/false>(
      &info, self->mod, block_sref, storage_scope, index_map, block, realize, write_buffer,
      cache_region);

  // Step 5. Check whether cache region is a single point.
  CheckSinglePoint</*is_cache_read=*/false>(self, block, cache_region);

  // Step 6. Making new cache stage block and rewrite readers.
  SBlock cache_write_stage =
      MakeReindexCacheStage</*is_cache_read=*/false>(/*cache_region=*/cache_region,
                                                     /*info=*/&info,
                                                     /*storage_scope=*/storage_scope);
  Stmt new_scope = ReindexCacheWriteRewriter::Rewrite(
      /*scope_sref=*/scope_sref,
      /*writer_block_sref=*/block_sref, /*info=*/&info);

  // Step 7. Replacing and updating flags.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_write_stage.get());
  SBlockInfo& block_info = self->block_info[result_block_sref];
  block_info.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info.region_cover = true;
  block_info.stage_pipeline = true;
  return result_block_sref;
}

/*! \brief The schedule error that the target block doesn't both read&write target buffer. */
class NotReadWriteError : public ScheduleErrorContextObj {
 public:
  NotReadWriteError(IRModule mod, SBlock block, BufferVar buffer)
      : mod_(std::move(mod)), block_(std::move(block)), buffer_(std::move(buffer)) {}
  ffi::String FastErrorString() const final {
    return "ScheduleError: The target block does not both read & write target buffer.";
  }

  ffi::String DetailRenderTemplate() const final {
    return "The target block {0} does not both read & write target buffer {1}.";
  }

  IRModule mod() const final { return mod_; }
  ffi::Array<ffi::ObjectRef> LocationsOfInterest() const final { return {block_, buffer_}; }
  IRModule mod_;
  SBlock block_;
  BufferVar buffer_;
};

ffi::Array<StmtSRef> CacheInplace(ScheduleState self, const StmtSRef& block_sref,
                                  int read_buffer_index, const ffi::String& storage_scope) {
  /*!
   * Do cache read then cache write
   */

  // Check 0. Check the input storage scope.
  CheckStorageScope(self, storage_scope);

  // Check 1. Check index, get the target buffer and the parent scope
  const SBlockNode* block = TVM_SREF_TO_SBLOCK(block_sref);
  BufferVar buffer = GetNthAccessBuffer(self, ffi::GetRef<SBlock>(block), read_buffer_index,
                                        BufferIndexType::kRead);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/false);

  // Check 3. Check required region cover for cache_read
  CheckRegionCover(self, scope_sref, buffer);

  // Check 4. Check if target block both read & write target buffer.
  const SBlockNode* rw_block = TVM_SREF_TO_SBLOCK(block_sref);
  ffi::Optional<TensorRegion> read_region = GetBufferRegionFromBuffer(rw_block->reads, buffer);
  ffi::Optional<TensorRegion> write_region = GetBufferRegionFromBuffer(rw_block->writes, buffer);
  if (!read_region.has_value() || !write_region.has_value()) {
    throw MakeScheduleError<NotReadWriteError>(self->mod, ffi::GetRef<SBlock>(rw_block), buffer);
  }

  ffi::Array<StmtSRef> results_block_sref;
  BufferVar new_buffer = WithScope(buffer, storage_scope);

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
  SBlock cache_read_stage = MakeCacheStage(/*cache_region=*/read_region.value(), /*info=*/&info,
                                           /*storage_scope=*/storage_scope);
  Stmt new_scope = CacheReadRewriter::Rewrite(/*scope_sref=*/scope_sref, /*info=*/&info);

  // Cache read step 3. Replacing and updating flags for cache read.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(cache_read_stage.get());
  SBlockInfo& block_info_read = self->block_info[result_block_sref];
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
  info.alloc = std::nullopt;
  info.consumer_blocks.clear();

  // Cache write step 1. Detect insert position
  CacheInplaceLocDetector::Detect(self, block_sref, scope_sref, &info);
  // insert after target block for cache write
  info.loc_pos += 1;

  // Cache write step 2. Making new cache stage block and rewrite readers.
  SBlock cache_write_stage = MakeCacheStage(/*cache_region=*/write_region.value(), /*info=*/&info,
                                            /*storage_scope=*/storage_scope);
  new_scope = CacheWriteRewriter::Rewrite(/*scope_sref=*/scope_sref,
                                          /*writer_block_sref=*/block_sref, /*info=*/&info);

  // Cache write step 4. Replacing and updating flags for cache write.
  self->Replace(scope_sref, new_scope, info.block_reuse);
  result_block_sref = self->stmt2ref.at(cache_write_stage.get());
  SBlockInfo& block_info_write = self->block_info[result_block_sref];
  block_info_write.affine_binding = CalculateAffineFlag(self, result_block_sref);
  block_info_write.region_cover = true;
  block_info_write.stage_pipeline = false;
  results_block_sref.push_back(result_block_sref);

  return results_block_sref;
}

StmtSRef ReIndex(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                 BufferIndexType buffer_index_type) {
  const SBlockNode* block_ptr = TVM_SREF_TO_SBLOCK(block_sref);
  SBlock block = ffi::GetRef<SBlock>(block_ptr);
  BufferVar buffer = GetNthAccessBuffer(self, block, buffer_index, buffer_index_type);
  StmtSRef scope_sref = GetScopeRoot(self, block_sref, /*require_stage_pipeline=*/true);
  arith::Analyzer analyzer;

  // Step 1. Collect the original indices and check there's only single pattern of related
  // Load/Store and the buffer is not accessed opaquely
  ffi::Array<PrimExpr> original_indices = ReIndexCollector::Collect(self->mod, buffer, block);
  // Simplify the indices if possible
  for (const IterVar& iter : block->iter_vars) {
    analyzer->Bind(iter->var, iter->dom);
  }
  original_indices.MutateByApply(
      [&analyzer](const PrimExpr& expr) { return SimplifyNonTrivialExpr(expr, analyzer.get()); });

  // Collect block iters appearing in the original_indices
  std::unordered_set<Var> covered;
  auto walk_fn = [&covered](const Var& var) -> ffi::Expected<ffi::WalkResult> {
    covered.insert(var);
    return ffi::WalkResult::Advance();
  };
  for (const PrimExpr& index : original_indices) {
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(index, walk_fn);
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
    TVM_FFI_ICHECK(outer != nullptr && inner != nullptr);
    TVM_FFI_ICHECK(outer->body.get() == inner);
    loop = loop->parent;
  }

  info.loc_pos = loop->seq_index == -1 ? 0 : loop->seq_index;
  if (buffer_index_type == BufferIndexType::kWrite) {
    info.loc_pos++;
  }

  // Step 4. Making new reindex stage block and rewrite
  SBlock reindex_stage =
      MakeReIndexStage(block, &info, covered, original_indices, buffer_index, buffer_index_type);
  Stmt new_scope = ReIndexRewriter::Rewrite(scope_sref, block_sref, &info, covered);

  // Step 5. Replacing and updating flags
  self->Replace(scope_sref, new_scope, info.block_reuse);
  StmtSRef result_block_sref = self->stmt2ref.at(reindex_stage.get());
  SBlockInfo& block_info = self->block_info[result_block_sref];
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

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, SBlockRV block,
                                          ffi::Array<SBlockRV> consumer_blocks,
                                          IntImm read_buffer_index, ffi::String storage_scope) {
    return sch->CacheRead(block, read_buffer_index->value.as<int>().value(), storage_scope,
                          consumer_blocks);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      ffi::Array<ffi::String> consumer_blocks,
                                      IntImm read_buffer_index, ffi::String storage_scope) {
    PythonAPICall py("cache_read");
    py.Input("block", block);
    py.Input("read_buffer_index", read_buffer_index->value.as<int>().value());
    py.Input("storage_scope", storage_scope);
    // Only write out consumer blocks if provided.
    if (!consumer_blocks.empty()) {
      py.Input("consumer_blocks", consumer_blocks);
    }
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct CacheWriteTraits : public UnpackedInstTraits<CacheWriteTraits> {
  static constexpr const char* kName = "CacheWrite";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, SBlockRV block,
                                          ffi::Array<SBlockRV> consumer_blocks,
                                          IntImm write_buffer_index, ffi::String storage_scope) {
    return sch->CacheWrite(block, write_buffer_index->value.as<int>().value(), storage_scope,
                           consumer_blocks);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      ffi::Array<ffi::String> consumer_blocks,
                                      IntImm write_buffer_index, ffi::String storage_scope) {
    PythonAPICall py("cache_write");
    py.Input("block", block);
    py.Input("write_buffer_index", write_buffer_index->value.as<int>().value());
    py.Input("storage_scope", storage_scope);
    // Only write out consumer blocks if provided.
    if (!consumer_blocks.empty()) {
      py.Input("consumer_blocks", consumer_blocks);
    }
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct CacheInplaceTraits : public UnpackedInstTraits<CacheInplaceTraits> {
  static constexpr const char* kName = "CacheInplace";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static ffi::Array<SBlockRV> UnpackedApplyToSchedule(Schedule sch, SBlockRV block,
                                                      IntImm read_buffer_index,
                                                      ffi::String storage_scope) {
    return sch->CacheInplace(block, read_buffer_index->value.as<int>().value(), storage_scope);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      IntImm read_buffer_index, ffi::String storage_scope) {
    PythonAPICall py("cache_inplace");
    py.Input("block", block);
    py.Input("read_buffer_index", read_buffer_index->value.as<int>().value());
    py.Input("storage_scope", storage_scope);
    py.OutputList(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct ReIndexTraits : public UnpackedInstTraits<ReIndexTraits> {
  static constexpr const char* kName = "ReIndex";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, SBlockRV block, IntImm buffer_index,
                                          IntImm buffer_index_type) {
    return sch->ReIndex(block, buffer_index->value.as<int>().value(),
                        static_cast<BufferIndexType>(buffer_index_type->value.as<int>().value()));
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      IntImm buffer_index, IntImm buffer_index_type) {
    PythonAPICall py("reindex");
    py.Input("block", block);
    std::ostringstream os;
    os << "(\""
       << BufferIndexType2Str(
              static_cast<BufferIndexType>(buffer_index_type->value.as<int>().value()))
       << "\", " << buffer_index << ")";
    py.Input("buffer", ffi::String(os.str()));
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct ReindexCacheReadTraits : public UnpackedInstTraits<ReindexCacheReadTraits> {
  static constexpr const char* kName = "ReindexCacheRead";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, SBlockRV block, IndexMap index_map,
                                          IntImm read_buffer_index, ffi::String storage_scope) {
    return sch->ReindexCacheRead(block, read_buffer_index->value.as<int>().value(), storage_scope,
                                 index_map);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      IndexMap index_map, IntImm read_buffer_index,
                                      ffi::String storage_scope) {
    PythonAPICall py("reindex_cache_read");
    py.Input("block", block);
    py.Input("read_buffer_index", read_buffer_index->value.as<int>().value());
    py.Input("storage_scope", storage_scope);
    py.Input("index_map", index_map->ToPythonString());
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

struct ReindexCacheWriteTraits : public UnpackedInstTraits<ReindexCacheWriteTraits> {
  static constexpr const char* kName = "ReindexCacheWrite";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static SBlockRV UnpackedApplyToSchedule(Schedule sch, SBlockRV block, IndexMap index_map,
                                          IntImm write_buffer_index, ffi::String storage_scope) {
    return sch->ReindexCacheWrite(block, write_buffer_index->value.as<int>().value(), storage_scope,
                                  index_map);
  }

  static ffi::String UnpackedAsPython(ffi::Array<ffi::String> outputs, ffi::String block,
                                      IndexMap index_map, IntImm write_buffer_index,
                                      ffi::String storage_scope) {
    PythonAPICall py("reindex_cache_write");
    py.Input("block", block);
    py.Input("write_buffer_index", write_buffer_index->value.as<int>().value());
    py.Input("storage_scope", storage_scope);
    py.Input("index_map", index_map->ToPythonString());
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::s_tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(CacheReadTraits);
TVM_REGISTER_INST_KIND_TRAITS(CacheWriteTraits);
TVM_REGISTER_INST_KIND_TRAITS(CacheInplaceTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReIndexTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReindexCacheReadTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReindexCacheWriteTraits);

}  // namespace s_tir
}  // namespace tvm
