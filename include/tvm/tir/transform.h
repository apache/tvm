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
 * \brief Create PrimFuncPass to combine context calls in the host function.
 *
 * \return The pass.
 */
Pass CombineContextCall();

}  // namespace transform
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_TRANSFORM_H_
