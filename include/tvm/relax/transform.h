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
 * \file tvm/relax/transform.h
 * \brief Relax specific transformation passes.
 */
#ifndef TVM_RELAX_TRANSFORM_H_
#define TVM_RELAX_TRANSFORM_H_

#include <tvm/ir/transform.h>
#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {
namespace transform {

using Pass = tvm::transform::Pass;
using PassInfo = tvm::transform::PassInfo;
using PassContext = tvm::transform::PassContext;
using Function = tvm::relax::Function;
using DataflowBlock = tvm::relax::DataflowBlock;

/*!
 * \brief Create a function pass.
 *
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the function pass.
 * \param name The name of the function pass.
 * \param required The list of the passes that the function pass is dependent on.
 * \param traceable Boolean variable whether the dataflowblock pass is traceable.
 *
 * \return The created function pass.
 */
TVM_DLL Pass CreateFunctionPass(
    const runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>& pass_func,
    int opt_level, String name, tvm::Array<String> required, bool traceable = false);

/*!
 * \brief Create a dataflowblock pass.
 *
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the dataflowblock pass.
 * \param name The name of the dataflowblock pass.
 * \param required The list of the passes that the dataflowblock pass is dependent on.
 * \param traceable Boolean variable whether the dataflowblock pass is traceable.
 *
 * \return The created dataflowblock pass.
 */
TVM_DLL Pass CreateDataflowBlockPass(
    const runtime::TypedPackedFunc<DataflowBlock(DataflowBlock, IRModule, PassContext)>& pass_func,
    int opt_level, String name, tvm::Array<String> required, bool traceable = false);

/*!
 * \brief Transform all dataflow structure to non-dataflow version.
 *
 * \return The Pass.
 */
TVM_DLL Pass ToNonDataflow();

/*!
 * \brief Perform explicit tensor allocation for call_tir.
 *
 * \return The Pass.
 */
TVM_DLL Pass CallTIRRewrite();

/*!
 * \brief Convert all reshape-like call_tir whose corresponding binding
 * vars are DataflowVars to relax.reshape operator calls. The relax.reshape
 * calls will be lowered an external builtin function call in a subsequent
 * pass, where the external builtin function does a CreateView operation
 * at runtime, instead of doing real data copy.
 * Here "reshape-like" includes reshape, expand_dims, flatten, etc.
 *
 * \return The Pass.
 * \note The pass is applied at the first stage of Relax VM build, before
 * rewriting call_tir, as this pass requires dataflow information.
 */
TVM_DLL Pass RewriteDataflowReshape();

/*!
 * \brief The static memory planning pass on BindingBlock level.
 * The pass will reuse allocated memory to its best effort, in order to
 * reduce the total amount of allocated memory size.
 *
 * \return The pass.
 */
TVM_DLL Pass StaticPlanBlockMemory();

/*!
 * \brief Bind params of function of the module to constant tensors.
 *
 * \param func_name The name of the function to bind parameters.
 * \param params The parameters to bind.
 *
 * \return The Pass.
 */
TVM_DLL Pass BindParams(String func_name, Map<String, runtime::NDArray> params);

/*!
 * \brief Fold constant expressions.
 *
 * \return The Pass.
 */
TVM_DLL Pass FoldConstant();
/*!
 * \brief Attach global_symbol to Relax functions and TIR Primfuncs for codegen.
 *
 * \return The Pass.
 */
TVM_DLL Pass AttachGlobalSymbol();
/*!
 * \brief Bind params of function of the module to constant tensors.
 *
 * \param func_name The name of the function to bind parameters.
 * \param params The parameters to bind.
 *
 * \return The Pass.
 */
TVM_DLL Pass BindParams(String func_name, Map<String, runtime::NDArray> params);

/*!
 * \brief Fold constant expressions.
 *
 * \return The Pass.
 */
TVM_DLL Pass FoldConstant();

/*!
 * \brief Transform Relax IR to normal form: transform AST to A-normal form, and fill the
 * checked_type_ and shape_ of expressions.
 *
 * \return The Pass.
 */
TVM_DLL Pass Normalize();

/*!
 * \brief Legalize high-level operator calls in Relax functions to call_tir
 * with corresponding low-level TIR PrimFuncs.
 *
 * For each high-level operator, we register the way of legalizing it as a
 * function, which takes a context BlockBuilder and the Call being legalized
 * as input, and returns the legalized call. Here the input BlockBuilder is
 * mainly used for adding the PrimFunc created by call_te into the context
 * IRModule.
 *
 * The legalization function for each operator is registered as an attribute (with
 * attribute key `FLegalize`) of the operator.
 *
 * For customizability, the user can pass their own legalization by an optional customized map,
 * with the key to be the operator name and value to be the legalization function.
 * The default legalization function will be overridden by the customized one.
 *
 * \param cmap The customized operator legalization function map. The customized function
 * will override the default one.
 * \return The Pass.
 */
TVM_DLL Pass LegalizeOps(Optional<Map<String, PackedFunc>> cmap);

/*
 * \brief Lift transformation of the parameters of a function.
 *
 * When some inputs of the function is marked as 'parameters' (the model weights), this pass
 * identifies the transformation of the parameters and lifts them to a separate function called
 * `transform_params`. `transform_params` takes a tuple of the original parameters as input and
 * returns a tuple of the transformed parameters. The original function will be rewritten to accept
 * a tuple of transformed parameters as input.
 *
 * Users are expected to invoke the `transform_params` function in runtime and pass the transformed
 * parameters to the original function as input.
 *
 * \return The Pass.
 */
TVM_DLL Pass LiftTransformParams();

}  // namespace transform
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_H_
