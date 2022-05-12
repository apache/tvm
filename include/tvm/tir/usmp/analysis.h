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

/*!
 * \file tir/usmp/analysis.h
 * \brief The analysis passes for TIR-based Unified Static Memory Planner
 */

#ifndef TVM_TIR_USMP_ANALYSIS_H_
#define TVM_TIR_USMP_ANALYSIS_H_

#include <tvm/tir/function.h>
#include <tvm/tir/usmp/utils.h>

namespace tvm {
namespace tir {
namespace usmp {

/*!
 * \brief Extract BufferInfo objects from a TIR IRModule
 *
 * This pass would extract the buffer information of allocate nodes
 * including liveness conflict with other buffer info objects.
 *
 * \return A Map of BufferInfo objects and their associated Stmts
 */
BufferInfoAnalysis ExtractBufferInfo(const PrimFunc& main_func, const IRModule& mod);

}  // namespace usmp
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_USMP_ANALYSIS_H_
