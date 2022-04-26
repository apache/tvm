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
 * \file tir/usmp/transform.h
 * \brief The transform passes for TIR-based Unified Static Memory Planner
 */

#ifndef TVM_TIR_USMP_TRANSFORM_H_
#define TVM_TIR_USMP_TRANSFORM_H_

#include <tvm/tir/usmp/utils.h>

namespace tvm {
namespace tir {
namespace usmp {
namespace transform {

using Pass = tvm::transform::Pass;

/*!
 * \brief Convert the analyzed PoolAllocation to offsets from pool variables
 *
 * This pass would convert the main function to accept pool variables as an input
 * that get passed onto the operator PrimFuncs. Furthermore, the static allocations
 * will be converted to offsets within the pool variable.
 *
 * \return the pass
 */
TVM_DLL Pass ConvertPoolAllocationsToOffsets(const Map<tir::Stmt, PoolAllocation>& pool_allocations,
                                             Bool emit_tvmscript_printable = Bool(false));

/*!
 * \brief Assign PoolInfo objects to tir.allocate nodes depending on the PrimFunc's target
 *
 * This pass would assign default PoolInfo objects to allocate nodes that are not otherwise
 * annotated, depending on pool info supplied for each target.
 *
 * \return the pass
 */
TVM_DLL Pass AssignPoolInfo();

/*!
 * \brief This pass creates Allocate nodes for I/O tensors
 *
 * If the user wants to place the I/O tensors in the workspace, this pass is required to be
 * run. In doing so, it will create Allocate nodes for I/O tensors to be planned, and be removed
 * from function arguments.
 *
 * \return the pass
 */
TVM_DLL Pass CreateAllocatesForIO();

}  // namespace transform
}  // namespace usmp
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_USMP_TRANSFORM_H_
