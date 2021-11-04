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
 * This pass would convert the IRModule that contains all PrimFuncs that contains
 * the associated PoolAllocations to be read from being offset from the input var
 * of the PrimFunc.
 *
 * \return the pass
 */
TVM_DLL Pass ConvertPoolAllocationsToOffsets(const Map<tir::Stmt, PoolAllocation>& pool_allocations,
                                             Bool emit_tvmscript_printable = Bool(false));

/*!
 * \brief Assign PoolInfo objects to tir.allocate nodes depending on the PrimFunc's target
 *
 * This pass would assign PoolInfo objects to tir.allocate nodes depending on the each target
 * that each PrimFunc would belong to. If there are not any pools provided in the IRModule,
 * this pass would create a global workspace pool that every target could access for as the
 * default behaviour.
 *
 * \return the pass
 */
TVM_DLL Pass AssignPoolInfo();

}  // namespace transform
}  // namespace usmp
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_USMP_TRANSFORM_H_
