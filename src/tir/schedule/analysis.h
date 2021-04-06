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
#ifndef TVM_TIR_SCHEDULE_ANALYSIS_H_
#define TVM_TIR_SCHEDULE_ANALYSIS_H_

#include <tvm/tir/schedule/state.h>

namespace tvm {
namespace tir {

/******** Verification ********/
/*!
 * \brief Verify the sref tree state is consistent with the IR
 * \param self The schedule state containing the sref to be verified
 * \throw An exception will be thrown if the sref tree is not valid
 */
void VerifySRefTree(const ScheduleState& self);

/******** Block-loop relation ********/
/*!
 * \brief Get the leaf blocks of a scope where a specific block/loop is in
 * \param self The schedule state
 * \param parent_sref The StmtSRef that points to the parent block/loop
 * \return A list of leaf blocks
 */
Array<StmtSRef> GetChildBlocks(const ScheduleState& self, const StmtSRef& parent_sref);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_ANALYSIS_H_
