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
#ifndef TVM_TIR_SCHEDULE_PRIMITIVE_H_
#define TVM_TIR_SCHEDULE_PRIMITIVE_H_

#include <tvm/support/random_engine.h>
#include <tvm/tir/schedule/state.h>

namespace tvm {
namespace tir {

/******** Schedule: Sampling ********/
/*!
 * \brief Sample once category from candidates according to the probability weights.
 * \param self The schedule to update
 * \param rand_state The pointer to schedule's random state
 * \param candidates The candidates
 * \param probs The probability distribution of the candidates
 * \param decision The sampling decision, if any
 * \return The random variable sampled from candidates
 */
TVM_DLL int64_t SampleCategorical(support::LinearCongruentialEngine::TRandState* rand_state,
                                  const Array<Integer>& candidates, const Array<FloatImm>& probs,
                                  Optional<Integer>* decision);

/******** Schedule: Get blocks & loops ********/
/*!
 * \brief Retrieves blocks in a specific function with its name
 * \param self The schedule state
 * \param name The name of the blocks to be retrieved
 * \param func_name The name of the function
 * \return A list of blocks with the specific name
 */
Array<StmtSRef> GetBlocks(const ScheduleState& self, const String& name, const String& func_name);
/*!
 * \brief Gets the parent loops of the block in its scope, from outer to inner
 * \param self The schedule state
 * \param block_sref The query block
 * \return A list of loops above the given block in its scope, from outer to inner
 */
Array<StmtSRef> GetLoops(const StmtSRef& block_sref);
/******** Schedule: Transform loops ********/
/*!
 * Split a loop into a list of consecutive loops. It requires:
 * 1) The loop can't have annotation or thread binding.
 * 2) The loop must start with 0.
 * \param self The state of the schedule
 * \param loop_sref The sref to the loop being split
 * \param factors The splitting factors
 * \return An array of srefs to the loops after splitting
 */
TVM_DLL Array<StmtSRef> Split(ScheduleState self, const StmtSRef& loop_sref,
                              const Array<PrimExpr>& factors);
/*!
 * \brief Fuse a list of consecutive loops into one. It requires:
 * 1) The loops can't have annotations or thread bindings.
 * 2) The inner loop must be the only child of the outer loop.
 * 3) All loops must start with 0.
 * \param self The state of the schedule
 * \param loop_srefs An array of srefs to the loops to be fused
 * \return The sref to the fused loop
 */
TVM_DLL StmtSRef Fuse(ScheduleState self, const Array<StmtSRef>& loop_srefs);
/*!
 * \brief Reorder a list of loops. It doesn't require the loops to be consecutive.
 * It requires:
 * 1) The loops are in the same chain. That means: the loops can be ordered to [l_1, l_2, ... ,
 *     l_n] where l_i is an ancestor of l_{i+1} and there are only single-branch loops between
 *     l_1 and l_n (which also indicates they are under the same scope).
 * 2) After reordering, the domain of an outer loop cannot depend on any of the inner loops.
 * 3) For every block under the loop nests, its block binding must be affine, and the block
 *    variables must be either data parallel or reduction.
 * 4) No duplicated loops are allowed in the arguments.
 * \param self The state of the schedule
 * \param ordered_loop_srefs An array of srefs which indicates the new order of loops
 */
TVM_DLL void Reorder(ScheduleState self, const Array<StmtSRef>& ordered_loop_srefs);

/******** Schedule: Manipulate ForKind ********/
/*!
 * \brief Parallelize the input loop. It requires:
 * 1) The scope block that the loop is in should have stage-pipeline property
 * 2) All the blocks under the loop are complete blocks or reduction blocks, and have affine
 * bindings
 * 3) For each block under the loop, the loop can only be contained in data-parallel block iters'
 * bindings
 * \param self The state of the schedule
 * \param loop_sref The sref of the loop to be parallelized
 */
TVM_DLL void Parallel(ScheduleState self, const StmtSRef& loop_sref);
/*!
 * \brief Vectorize the input loop. It requires:
 * 1) The scope block that the loop is in should have stage-pipeline property
 * 2) All the blocks under the loop are complete blocks or reduction blocks, and have affine
 * bindings
 * 3) For each block under the loop, the loop can only be contained in data-parallel block iters'
 * bindings
 * \param self The state of the schedule
 * \param loop_sref The sref of the loop to be vectorized
 */
TVM_DLL void Vectorize(ScheduleState self, const StmtSRef& loop_sref);
/*!
 * \brief Bind the input loop to the given thread axis. It requires:
 * 1) The scope block that the loop is in should have stage-pipeline property
 * 2) All the blocks under the loop are complete blocks or reduction blocks, and have affine
 * bindings
 * 3) For each block under the loop, if the thread axis starts with "threadIdx`, the loop can only
 * be contained in data-parallel block iter and reduction block iters' bindings. Otherwise the
 * loop can only be contained in data-parallel block iters' bindings
 * \param self The state of the schedule
 * \param loop_sref The sref of the loop to be bound to the thread axis
 * \param thread_axis The thread axis to be bound to the loop
 */
TVM_DLL void Bind(ScheduleState self, const StmtSRef& loop_sref, const IterVar& thread_axis);
/*!
 * \brief Unroll the input loop. It requires nothing
 * \param self The state of the schedule
 * \param loop_sref The loop to be unrolled
 */
TVM_DLL void Unroll(ScheduleState self, const StmtSRef& loop_sref);
/******** Schedule: Insert cache stages ********/
/*!
 * \brief Create a block that reads a buffer region into a read cache. It requires:
 * 1) There is at most one block who writes the buffer in the scope.
 * 2) The scope block have stage-pipeline property.
 * \param self The state of the schedule
 * \param block_sref The consumer block of the target buffer.
 * \param read_buffer_index The index of the buffer in block's read region.
 * \param storage_scope The target storage scope.
 * \return The cache stage block.
 */
TVM_DLL StmtSRef CacheRead(ScheduleState self, const StmtSRef& block_sref, int read_buffer_index,
                           const String& storage_scope);
/*!
 * \brief Create a block that writes a buffer region into a write cache. It requires:
 * 1) There is only one block that writes the target buffer.
 * 2) The scope block have stage-pipeline property.
 * \param self The state of the schedule
 * \param block_sref The producer of the buffer
 * \param write_buffer_index The index of the buffer in block's write region
 * \param storage_scope The target storage scope
 * \return The cache stage block.
 */
TVM_DLL StmtSRef CacheWrite(ScheduleState self, const StmtSRef& block_sref, int write_buffer_index,
                            const String& storage_scope);
/******** Schedule: Compute location ********/
/*!
 * \brief Move a producer block under the specific loop, and regenerate the loops induced by the
 * block so that the buffer region generated by the producer block could cover those regions
 * written by the producers. It requires:
 * 1) `block` and `loop` are under the same scope, `loop` is not the ancestor of `block`
 * 2) The scope block has stage-pipeline property
 * 3) The given block's subtree of the scope block satisfies compact dataflow condition. i.e. all
 * the blocks in the scope block's subtree must be either complete block or reduction block
 * 4) The block is not an output block, i.e. the buffer regions written by the block are allocated
 * under the current scope
 * 5) All the consumers of the block are under the given loop
 *
 * \param self The schedule state
 * \param block_sref The block to be moved
 * \param loop_sref The loop where the block to be moved to
 * \param preserve_unit_loops Whether to keep the trivial loops whose extents are 1
 */
TVM_DLL void ComputeAt(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                       bool preserve_unit_loops);
/*!
 * \brief Move a consumer block under the specific loop, and regenerate the loops induced by the
 * block so that the buffer region generated by the consumer block could cover those regions read
 * by the consumers. It requires:
 * 1) `block` and `loop` are under the same scope, `loop` is not the ancestor of `block`
 * 2) The scope block has stage-pipeline property
 * 3) The given block's subtree of the scope block satisfies compact dataflow condition.
 * i.e. all the blocks in the scope's subtree must be either complete block or reduction block
 * 4) All the producers of the block are under the given loop
 *
 * \param self The schedule state
 * \param block_sref The block to be moved
 * \param loop_sref The loop where the block to be moved to
 * \param preserve_unit_loops Whether to keep the trivial loops whose extents are 1
 */
TVM_DLL void ReverseComputeAt(ScheduleState self, const StmtSRef& block_sref,
                              const StmtSRef& loop_sref, bool preserve_unit_loops);
/*!
 * \brief Inline a block into its consumer(s). It requires:
 * 1) The block is a complete non-root block, which only produces one buffer
 * 2) The block must not be the only leaf in the scope.
 * 3) The body of the block must be a BufferStore statement in the form of,
 *    A[i, j, k, ...] = ...
 * where the indices of the LHS are all distinct atomic variables,
 * and no variables other than those indexing variables are allowed in the statement.
 * \param self The state of the schedule
 * \param block_sref The sref to the block to be inlined to its consumer(s)
 */
TVM_DLL void ComputeInline(ScheduleState self, const StmtSRef& block_sref);
/*!
 * \brief Inline a block into its only producer. It requires:
 * 1) The block is a complete non-root block, which only produces and consumers one buffer
 * 2) The block must not be the only leaf in the scope.
 * 3) The only producer of the block is a read-after-write producer and a complete non-root block
 * 4) The body of the block must be a BufferStore statement in the form of,
 *    B[f(i, j, k, ...)] = g(i, j, k, A[i, j, k, ...] ...)
 * where the indices of each `BufferLoad` on the RHS are all distinct atomic variables,
 * and no variables other than those indexing variables are allowed in the statement.
 * \param self The state of the schedule
 * \param block_sref The sref to the block to be inlined to its producer
 */
TVM_DLL void ReverseComputeInline(ScheduleState self, const StmtSRef& block_sref);
/******** Schedule: Reduction ********/
/*!
 * \brief Factor a reduction block by the specified loop
 * \details See python/tvm/tir/schedule/schedule.py
 * \param self The state of the schedule
 * \param loop_sref The loop outside block for which we want to do rfactor
 * \param factor_axis The position where the new dimension is placed in the new introduced rfactor
 *                    buffer. Suppose the original reduction block writes to buffer `B` with
 *                    ndim(B) dimensions, then `factor_axis` should be in range `[-ndim(B) - 1,
 *                    ndim(B)]`, and the negative index will be normalized to a non-negative one
 * \return The sref of the rfactor block
 */
TVM_DLL StmtSRef RFactor(ScheduleState self, const StmtSRef& loop_sref, int factor_axis);
/******** Schedule: Block annotation ********/
/*! \brief The quad used by StorageAlign for (buffer_idx, axis, factor, offset) */
using StorageAlignTuple = Array<Integer>;
/*! \brief A list of StorageAlignTuple, used by StorageAlign */
using StorageAlignAnnotation = Array<StorageAlignTuple>;
/*!
 * \brief Set alignment requirement for specific dimension such that
 *        stride[axis] == k * factor + offset for some k. This is useful to set memory layout for
 *        more friendly memory access pattern. For example, we can set alignment to be factor=2,
 *        offset=1 to avoid bank conflict for thread access on higher dimension in GPU shared
 *        memory.
 * \param block_sref The producer block of the buffer
 * \param buffer_index The index of the buffer in block's write region
 * \param axis The dimension to be specified for alignment
 * \param factor The factor multiple of alignment
 * \param offset The required offset factor
 */
TVM_DLL void StorageAlign(ScheduleState self, const StmtSRef& block_sref, int buffer_index,
                          int axis, int factor, int offset);

/******** Schedule: Blockize & Tensorize ********/
/******** Schedule: Annotation ********/
/******** Schedule: Misc ********/

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_PRIMITIVE_H_
