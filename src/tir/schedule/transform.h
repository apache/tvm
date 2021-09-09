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

#include <tvm/tir/schedule/state.h>

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

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_TRANSFORM_H_
