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
 * \file tvm/tir/transform.h
 * \brief TIR specific transformation passes.
 */
#ifndef TVM_TIR_TRANSFORM_H_
#define TVM_TIR_TRANSFORM_H_

#include <tvm/ir/transform.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>

#include <string>

namespace tvm {
namespace tir {
namespace transform {

using tvm::transform::Pass;
using tvm::transform::PassNode;
using tvm::transform::PassInfo;
using tvm::transform::PassInfoNode;
using tvm::transform::PassContext;
using tvm::transform::PassContextNode;
using tvm::transform::Sequential;

/*
 * \brief Create a function pass that optimizes PrimFuncs.
 *
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the function pass.
 * \param name The name of the function pass.
 * \param required The list of the passes that the function pass is dependent on.
 *
 * \return The created function pass.
 */
TVM_DLL Pass CreatePrimFuncPass(const runtime::TypedPackedFunc<
                                PrimFunc(PrimFunc, IRModule, PassContext)>& pass_func,
                                int opt_level,
                                const std::string& name,
                                const tvm::Array<tvm::PrimExpr>& required);

/*!
 * \brief Bind the device type ofthe function to be
 *        the device_type specified in the target attribute.
 *
 * \return The pass.
 */
TVM_DLL Pass BindDeviceType();

/*!
 * \brief Split the function into a host function and device functions.
 *
 * \return The pass.
 */
TVM_DLL Pass SplitHostDevice();

/*!
 * \brief skip assert stmt.
 *
 * \return The pass.
 */
TVM_DLL Pass SkipAssert();

/*!
 * \brief Insert sync between parallel read/write of shared buffers.
 *
 * \param storage_scope The storage scope considered.
 * \return The pass.
 */
TVM_DLL Pass ThreadSync(std::string storage_scope);


/*!
 * \brief Lower cross thread alleduce.
 *
 * \return The pass.
 */
TVM_DLL Pass LowerThreadAllreduce();

/*!
 * \brief Infer the TensorCore fragment infomation using tensor intrinsics
 *
 * \return The pass.
 */
TVM_DLL Pass InferFragment();

/*!
 * \brief Lower builtin intrinsics.
 * \return The pass.
 */
TVM_DLL Pass LowerTVMBuiltin();

/*!
 * \brief Lower the target specific function intrinsics in each of the function.
 *
 * \return The pass.
 */
TVM_DLL Pass LowerIntrin();

/*!
 * \brief Lower warp memory access to low-level device related function calls.
 * \return The pass.
 */
TVM_DLL Pass LowerWarpMemory();

/*!
 * \brief Lower attached storage access information on device.
 *
 * \note Run this pass after all storage access analysis finish.
 *
 * \return The pass.
 */
TVM_DLL Pass LowerDeviceStorageAccessInfo();

/*!
 * \brief Combine context calls in the host function.
 *
 * \return The pass.
 */
TVM_DLL Pass CombineContextCall();


/*!
 * \brief Narrow down PrimExpr datatype in stmt to target_bits.
 *
 * \note Run this pass after StorageFlatten.
 *
 * \return The pass.
 */
TVM_DLL Pass NarrowDataType();

}  // namespace transform
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_TRANSFORM_H_
