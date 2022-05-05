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
#ifndef TVM_TIR_SCHEDULE_TRANSFORM_H_
#define TVM_TIR_SCHEDULE_TRANSFORM_H_

#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/schedule/state.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <utility>

#include "../ir/functor_common.h"

namespace tvm {
namespace tir {

/******** Annotation ********/

/*!
 * \brief Create a new block with the given annotation added
 * \param block The block with original annotation
 * \param attr_key The annotation key to be added
 * \param attr_value The annotation value to be added
 * \return A new block with the given annotation as its last annotation
 */
Block WithAnnotation(const BlockNode* block, const String& attr_key, const ObjectRef& attr_value);

/******** Buffer Related ********/

/*!
 * \brief Create a new buffer by changing the storage scope.
 * \param buffer The given buffer.
 * \param scope The target storage scope.
 * \return The new buffer with target storage scope.
 */
Buffer WithScope(const Buffer& buffer, const String& scope);

/*!
 * \brief Replaces the buffer within the specific sequence of regions
 * \param regions The regions whose buffers are to be replaced
 * \param source The buffer to be replaced
 * \param target The buffer to be replaced to
 * \return The new sequence of regions after replacement
 */
Array<BufferRegion> ReplaceBuffer(Array<BufferRegion> regions, const Buffer& source,
                                  const Buffer& target);

/*!
 * \brief Replaces the buffer within the specific sequence of match_buffers
 * \param match_buffers The match_buffers whose buffers are to be replaced
 * \param source The buffer to be replaced
 * \param target The buffer to be replaced to
 * \return The new sequence of match_buffers after replacement
 */
Array<MatchBufferRegion> ReplaceBuffer(Array<MatchBufferRegion> match_buffers, const Buffer& source,
                                       const Buffer& target);

/*!
 * \brief A helper mutator which recursively replaces the old buffer with the new buffer and
 * collects the block sref reuse information for the following replacement.
 *
 * If the buffer to be replaced in used as the source in `match_buffers`, depending the specific
 * use cases, the target buffers in `match_buffers` may also need to be mutated. In this
 * case, this class should be subclassed to explicitly handle `match_buffers`.
 */
class ReplaceBufferMutator : public StmtExprMutator {
 public:
  ReplaceBufferMutator(const Buffer& old_buffer, Buffer new_buffer,
                       Map<Block, Block>* block_sref_reuse)
      : block_sref_reuse_(block_sref_reuse) {
    buffer_var_map_[old_buffer->data.get()] = std::move(new_buffer);
  }

 protected:
  PrimExpr VisitExpr_(const VarNode* var) final {
    auto it = buffer_var_map_.find(var);
    return it != buffer_var_map_.end() ? it->second->data : GetRef<Var>(var);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* load) final {
    BufferLoad res = Downcast<BufferLoad>(ExprMutator::VisitExpr_(load));

    auto it = buffer_var_map_.find(res->buffer->data.get());
    if (it != buffer_var_map_.end()) {
      ObjectPtr<BufferLoadNode> ptr = make_object<BufferLoadNode>(*res.get());
      ptr->buffer = it->second;
      return PrimExpr(ptr);
    } else {
      return std::move(res);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* store) final {
    BufferStore res = Downcast<BufferStore>(StmtMutator::VisitStmt_(store));

    auto it = buffer_var_map_.find(res->buffer->data.get());
    if (it != buffer_var_map_.end()) {
      ObjectPtr<BufferStoreNode> ptr = make_object<BufferStoreNode>(*res.get());
      ptr->buffer = it->second;
      return Stmt(ptr);
    } else {
      return std::move(res);
    }
  }

  virtual MatchBufferRegion VisitMatchBufferRegion(const MatchBufferRegion& match_buffer) {
    auto it = buffer_var_map_.find(match_buffer->source->buffer->data.get());
    if (it != buffer_var_map_.end()) {
      return MatchBufferRegion(match_buffer->buffer,
                               BufferRegion(it->second, match_buffer->source->region));
    } else {
      return match_buffer;
    }
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    // To reduce the number of blocks in block sref reuse map, we check whether the block is really
    // mutated (i.e., the old buffer appears in the block). If so, we return the block after
    // mutation. Otherwise we just return the original block.

    auto f_mutate_match_buffer = [this](const MatchBufferRegion& match_buffer) {
      return this->VisitMatchBufferRegion(match_buffer);
    };
    auto f_mutate_read_write_region = [this](const BufferRegion& buffer_region) {
      auto it = buffer_var_map_.find(buffer_region->buffer->data.get());
      return it == buffer_var_map_.end() ? buffer_region
                                         : BufferRegion(it->second, buffer_region->region);
    };
    auto f_mutate_alloc_buffers = [this](const Buffer& buffer) {
      auto it = buffer_var_map_.find(buffer->data.get());
      return it == buffer_var_map_.end() ? buffer : it->second;
    };

    // Step 1. Mutate `match_buffers`. If an old buffer appears as a source of MatchBufferRegion,
    Array<MatchBufferRegion> match_buffers =
        MutateArray(block->match_buffers, f_mutate_match_buffer);
    // Step 2. Mutate the read/write region.
    Array<BufferRegion> reads = MutateArray(block->reads, f_mutate_read_write_region);
    Array<BufferRegion> writes = MutateArray(block->writes, f_mutate_read_write_region);
    // Step 3. Mutate `alloc_buffers` for the old buffer allocated in this block.
    Array<Buffer> alloc_buffers = MutateArray(block->alloc_buffers, f_mutate_alloc_buffers);
    // Step 4. Recursively mutate the block.
    Block mutated_block = Downcast<Block>(StmtMutator::VisitStmt_(block));

    if (mutated_block.get() == block && reads.same_as(mutated_block->reads) &&
        writes.same_as(mutated_block->writes) &&
        alloc_buffers.same_as(mutated_block->alloc_buffers) &&
        match_buffers.same_as(mutated_block->match_buffers)) {
      return GetRef<Block>(block);
    } else {
      ObjectPtr<BlockNode> n = CopyOnWrite(mutated_block.get());
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      n->alloc_buffers = std::move(alloc_buffers);
      n->match_buffers = std::move(match_buffers);

      Block new_block(n);
      block_sref_reuse_->Set(GetRef<Block>(block), new_block);
      return std::move(new_block);
    }
  }

  /*! \brief The storage scope to be set. */
  String storage_scope_;
  /*!
   * \brief A mapping which maps old buffer vars to new buffers, including the buffers defined in
   * MatchBufferRegion.
   */
  std::unordered_map<const VarNode*, Buffer> buffer_var_map_;
  /*! \brief The block sref reuse map for the following replacement */
  Map<Block, Block>* block_sref_reuse_;
};

/******** Block Removal ********/

/*!
 * \brief Construct a new AST, with a specific sref tree leaf removed.
 * The leaf's ancestors who have only a single child will be removed too.
 * \param leaf_block_sref The block/loop sref to the sref tree leaf to be removed
 * \param src_stmt The root of the subtree where the replacement begins
 * \param tgt_stmt The root of the subtree after the replacement
 * \return A boolean indicating if the leaf can be removed successfully
 * \note Read before use:
 * 1) Removal is not conducted beyond scope-level.
 * 2) This method only works properly when the scope root is a stage pipeline.
 *
 * An example of the removal plan, say we are removing the leaf block "B" from the AST.
 *
 *  \code
 *    with block([], "scope_root"):
 *        ...
 *        with block([128, 128], "B") as [vi, vj]:
 *            B[vi, vj] = A[vi, vj] + 1.0
 *        with block([128, 128], "C") as [vi, vj]:
 *            C[vi, vj] = B[vi, vj] * 2.0
 *  \endcode
 *
 * Ths method does not mutate the AST, instead it returns the a `(src_stmt, tgt_stmt)` pair as a
 * plan to substitute certain pieces of the IR.
 *
 * In our example, it returns block "scope_root" as `src_stmt`, and the result `tgt_stmt` is:
 *
 *  \code
 *    with block([], "scope_root"):
 *        ...
 *        with block([128, 128], "C") as [vi, vj]:
 *            C[vi, vj] = B[vi, vj] * 2.0
 *  \endcode
 */
void LeafBlockRemovalPlan(const ScheduleState& self, const StmtSRef& leaf_block_sref,
                          Stmt* src_stmt, Stmt* tgt_stmt);

/*!
 * \brief Tile a subset of loops in the block according to the given tensor intrinsic.
 * \param self The schedule to which tiling is applied
 * \param block_rv The block whose subset of loops will be tiled
 * \param intrin_name The name of a tensor intrinsic, must be registerd via
 * TensorIntrin.register(...) beforehand
 * \return LoopRV corresponding to the outermost loop of a
 * block tiled according to the given intrin, NullOpt if a valid loop mapping is not found
 */
Optional<tir::LoopRV> TileWithTensorIntrin(const tir::Schedule& sch, const tir::BlockRV& block_rv,
                                           const String& intrin_name);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_TRANSFORM_H_
