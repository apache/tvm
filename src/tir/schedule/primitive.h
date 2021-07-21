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

#include <tvm/tir/schedule/state.h>

namespace tvm {
namespace tir {

/******** Schedule: loops manipulation ********/
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
/******** Schedule: compute location ********/
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

/******** Schedule: loop binding/annotation ********/

/******** Schedule: cache read/write ********/

/******** Schedule: reduction ********/

/******** Schedule: blockize & tensorize ********/

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_PRIMITIVE_H_
