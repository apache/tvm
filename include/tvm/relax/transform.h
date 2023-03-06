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
#include <tvm/relax/dataflow_pattern.h>
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
 * \brief Perform lambda lifting to lift functions from nested into global.
 *
 * \return The Pass.
 */
TVM_DLL Pass LambdaLift();

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
 * \brief Attach global_symbol to Relax functions and TIR Primfuncs for codegen.
 *
 * \return The Pass.
 */
TVM_DLL Pass AttachGlobalSymbol();

/*!
 * \brief Transform Relax IR to normal form: transform AST to A-normal form, and fill the
 * checked_type_ and shape_ of expressions.
 *
 * \return The Pass.
 */
TVM_DLL Pass Normalize();

/*!
 * \brief Simplify a Relax module by folding var bindings and match shape nodes.
 * May include other forms of expression simplification in the future.
 * Best used alongside constant folding and eliminating unused bindings.
 *
 * \return The Pass.
 */
TVM_DLL Pass CanonicalizeBindings();

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

/*!
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

/*!
 * \brief Annotate Op Pattern Kind for TIR functions, which is used in FuseOps.
 * \note It is an auto-detect pass for "unscheduled prim_funcs", the op_pattern will be
 *       "opaque" of we can't detect it. Users can manually annotate the attr `op_pattern`
 *       to prim_func.
 * \return The Pass.
 */
TVM_DLL Pass AnnotateTIROpPattern();

/*!
 * \brief This pass groups bindings in a dataflow block of Relax functions and generates a new
 * grouped Relax function for each group, according to the fusion algorithm described in the pass
 * implementation. By grouping bindings into new Relax functions, we substitute the bindings in the
 * function being manipulated into function calls to the new grouped function.
 *
 * A follow-up pass named "FuseTIR" will generate a TIR PrimFunc for each grouped function.
 * \param fuse_opt_level The level of fuse optimization.
 *        -1 indicates that the level will be inferred from pass context.
 * \return The Pass.
 */
TVM_DLL Pass FuseOps(int fuse_opt_level = -1);

/*!
 * \brief Apply pattern matching to each function in the given module, and group matched
 * expressions into a new function. The end result is similar to FuseOps, but fusion is driven
 * completely by the provided patterns.
 *
 * \param pattern_names The name of each pattern. It becomes the value of the kComposite attribute
 * of a fused function after successful matching.
 * \param patterns The patterns to detect. The order of the patterns determines the order
 * of priority in which they are matched. Higher-priority patterns should come earlier in the list.
 * \param checks The callback functions with type (Map<DFPattern, Expr>, Expr) -> bool. It takes a
 * match result and returns a boolean value to indicate whether the match result is accepted.
 * \param bind_constants Whether or not to keep bound constants of the grouped function.
 * \param annotate_codegen If true, wrap each created composite function with another function,
 * whose body consists only of a call to the composite function, and annotate the outer function
 * with kCodegen and kGlobalSymbol attributes. The kCodegen attribute is set as the prefix of the
 * corresponding pattern name. For example, "dnnl" if the pattern name is "dnnl.conv2d_relu".
 * This must be True if the created composite functions are intended to be offloaded to
 * an external backend without using the MergeCompositeFunctions pass.
 * \return The Pass.
 */
TVM_DLL Pass FuseOpsByPattern(const tvm::Array<runtime::String>& pattern_names,
                              const tvm::Array<DFPattern>& patterns,
                              const tvm::Array<PackedFunc>& checks, bool bind_constants = true,
                              bool annotate_codegen = false);

/*!
 * \brief Group one or multiple composite functions created by FuseOpsByPattern into a new
 *  function. The new function will be annotated with kCodegen and GlobalSymbol attributes,
 *  and it is intented to be offloaded to an external backend.
 *
 * \return The Pass.
 */
TVM_DLL Pass MergeCompositeFunctions();

/*!
 * \brief Fuse relax sub-function into a larger TIR function if possible.
    this pass works together with FuseOps to perform operator fusion.

 * \return The Pass.
 */
TVM_DLL Pass FuseTIR();

/*!
 * \brief Remove unused global relax functions in an IRModule.
 * \param entry_functions list of entry functions
 * \return The Pass.
 */
TVM_DLL Pass RemoveUnusedFunctions(Array<runtime::String> entry_functions);

/*!
 * \brief Run codegen.
 * \param target_options pairs of target name and compilation options
 * \param entry_functions list of entry functions
 * \return The Pass.
 */
TVM_DLL Pass RunCodegen(Optional<Map<String, Map<String, ObjectRef>>> target_options,
                        Array<runtime::String> entry_functions);

}  // namespace transform
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_H_
