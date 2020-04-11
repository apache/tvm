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
 * \file tvm/relay/transform.h
 * \brief Relay specific transformation passes.
 */
#ifndef TVM_RELAY_TRANSFORM_H_
#define TVM_RELAY_TRANSFORM_H_

#include <tvm/runtime/container.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/ir/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/op.h>

#include <string>

namespace tvm {
namespace relay {
namespace transform {

using Pass = tvm::transform::Pass;
using PassNode = tvm::transform::PassNode;
using PassInfo = tvm::transform::PassInfo;
using PassInfoNode = tvm::transform::PassInfoNode;
using PassContext = tvm::transform::PassContext;
using PassContextNode = tvm::transform::PassContextNode;
using Sequential = tvm::transform::Sequential;

/*
 * \brief Create a function pass.
 *
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the function pass.
 * \param name The name of the function pass.
 * \param required The list of the passes that the function pass is dependent on.
 *
 * \return The created function pass.
 */
TVM_DLL Pass CreateFunctionPass(const runtime::TypedPackedFunc<
                                Function(Function, IRModule, PassContext)>& pass_func,
                                int opt_level,
                                const std::string& name,
                                const tvm::Array<runtime::String>& required);

/*! \brief Remove expressions which does not effect the program result.
 *
 * It will remove let bindings which are not referenced,
 * and inline let bindings that are only used once.
 *
 * For example, this pass should turn `let a = 1 in 2` into `2`,
 * as the value of the expression does not depend on a.
 *
 * As another example, `let a = 1 in a` will be optimized into 1.
 *
 * \param inline_once whether or not to inline binding used one.
 *
 * \return the pass.
 */
TVM_DLL Pass DeadCodeElimination(bool inline_once = false);

/*!
* \brief Convert all expressions of TensorType into GradCell,
* an algebraic data type defined in gradient.rly.
*
* This will delay or decrease memory usage. All calls to
* ones, ones_like, zeros, zeros_like will not immediately instantiate a tensor in memory,
* rather only instantiate if needed. It also defines + and * operation
* between GradCell types which can increase performance when using
* zero-filled or one-filled tensors, which is the case in reverse mode ad.
*
* \return the pass
*/
TVM_DLL Pass LazyGradientInit();

/*!
 * \brief Fold constant expressions.
 *
 * \return The pass.
 */
TVM_DLL Pass FoldConstant();

/*!
 * \brief Fuse operations into expr into seperate functions.
 *
 * \param fuse_opt_level Optimization level. If it is -1 it will be inferred from pass context.
 *
 * \return The pass.
 */
TVM_DLL Pass FuseOps(int fuse_opt_level = -1);

/*!
 * \brief Rewrite the annotated program.
 *
 * \param fallback_device The fallback device which is the default device for
 *                        operators without annotation.
 *
 * \return The pass.
 */
TVM_DLL Pass RewriteAnnotatedOps(int fallback_device);

/*!
 * \brief turn a dataflow graph into Administrative Normal Form, or A-Normal Form (ANF).
 *
 * It will turn an expression that is in a graph form (with sharing implicit),
 * to an expression with explicit sharing (A-Normal Form).
 *
 * The scope of the root expression is the global scope.
 *
 * The scope of any non root expression is the least common ancestor of all it's scope.
 *
 * Values are ordered by post-DFS order in each scope.
 *
 * \return The pass.
 */
TVM_DLL Pass ToANormalForm();

/*!
 * \brief Turn an expression into continuation passing style(CPS).
 *
 * CPS mean that every function will, instead of returning the result directly,
 * be passed down an extra function (called the continuation) as argument,
 * and pass the result to the continuation instead.
 *
 * Thus, every function call has to be passed an extra argument
 * that represent the rest of the computation (Hence the name of continuation).
 *
 * Similarly, all other compute will be wrapped and call the continuation as well.
 *
 * \return the pass.
 */
TVM_DLL Pass ToCPS();

/*!
 * \brief Remove let binding and directly share via pointer instead.
 *
 * It will remove all let binding,
 * and turn all of the variable bound by let into direct pointer reference.
 *
 * \return the expression in graph normal form.
 */
TVM_DLL Pass ToGraphNormalForm();

/*!
 * \brief Aggressive constant propagation/constant folding/inlining.
 *
 * It will do as much computation in compile time as possible.
 * It has two benefit: remove runtime overhead, and allow more optimization (typically fusion).
 * As a side effect, code size will explode.
 *
 * \return the optimized expression.
 */
TVM_DLL Pass PartialEval();

/*!
 * \brief Simplify certain operators during inference. For example, batch norm
 * will be unpacked into a number of simplified operators.
 *
 * \return The Pass.
 */
TVM_DLL Pass SimplifyInference();

/*!
 * \brief Replaces non linear activation functions with their fast but approximate counterparts.
 *
 * \return The Pass.
 */
TVM_DLL Pass FastMath();

/*!
 * \brief Infer the type of an expression.
 *
 * The result of type checking is a new expression with unambigous
 * type information filled in, as well as it's checked type field
 * populated with the result type.
 *
 * \return The pass.
 */
TVM_DLL Pass InferType();

/*!
 * \brief Search and eliminate common subexpression. For example, if there are
 * two expressions evaluated to an identical value, a single variable is created
 * and these two expressions are replaced by this variable.
 *
 * \param fskip The callback argument that allows to skip certain expressions.
 *
 * \return The pass.
 */
TVM_DLL Pass EliminateCommonSubexpr(runtime::PackedFunc fskip = nullptr);

/*!
 * \brief Combine parallel 2d convolutions into a single convolution if the
 * number of branches of this conv2d operator is not less than
 * `min_num_branch`.
 *
 * \param min_num_branches The minimun number of branches.
 *
 * \return The pass.
 */
TVM_DLL Pass CombineParallelConv2D(uint64_t min_num_branches = 3);

/*!
 * \brief Combine parallel dense ops into a single batch_matmul if the
 * number of branches of this dense operator is not less than
 * `min_num_branch`.
 *
 * \param min_num_branches The minimun number of branches.
 *
 * \return The pass.
 */
TVM_DLL Pass CombineParallelDense(uint64_t min_num_branches = 3);

/*!
 * \brief Backward fold axis scaling into weights of conv/dense operators.
 *
 * \return The pass.
 */
TVM_DLL Pass BackwardFoldScaleAxis();

/*!
 * \brief Forward fold axis scaling into weights of conv/dense operators.
 *
 * \return The pass.
 */
TVM_DLL Pass ForwardFoldScaleAxis();

/*!
 * \brief A sequential pass that executes ForwardFoldScaleAxis and
 * BackwardFoldScaleAxis passes.
 *
 * \return The pass.
 */
TVM_DLL Pass FoldScaleAxis();

/*!
 * \brief Canonicalize some operators to the simplified operators. For example,
 * bias_add can be canonicalized to expand_dims and broadcast_add.
 *
 * \return The pass.
 */
TVM_DLL Pass CanonicalizeOps();

/*!
 * \brief Alternate the layouts of operators or replace primitive operators
 * with other expressions.
 *
 * \return The pass.
 */
TVM_DLL Pass AlterOpLayout();

/*!
 * \brief Given a dest layout, this pass transforms the expr such that most of the ops input data
 * layout is changed to the dest layout. In ideal situation, there are only 2 layout transforms, one
 * at the start and one at the end.
 *
 * This pass is not a part of relay.build and is expected to be called between framework-relay
 * parser and relay.build call. This is very helpful for hardware backends that support/prefer only
 * type of data layout.
 *
 * RFC - https://discuss.tvm.ai/t/layout-conversion-pass/4009
 *
 * This pass uses most of the AlterOpLayout and InferCorrectLayout infrastructure. We can define new
 * layouts for conv2d ops for now. Most of the other operators try to adapt to their input layout
 * using the InferCorrectLayout infrastructure.
 *
 * \param desired_layout The desired layout.
 * \return The pass.
 */
TVM_DLL Pass ConvertLayout(const std::string& desired_layout);

/*!
 * \brief Legalizes an expr with another expression.
 * \param legalize_map_attr_name The Op's attr name which corresponds to the legalize rule function.
 * One can collect and isolate similar type of legalize transformations using this param. For
 * example, transformations that only apply to Dialects can be isolated into a FTVMDialectLegalize
 * string. This pass calls only those transformations that have been registered using the supplied
 * legalize_map_attr_name.
 *
 * \return The pass.
 */
TVM_DLL Pass Legalize(const std::string& legalize_map_attr_name = "FTVMLegalize");

/*!
 * \brief Canonicalize cast expressions to make operator fusion more efficient.
 *
 * \return The pass.
 */
TVM_DLL Pass CanonicalizeCast();

/*!
 * \brief Add abstraction over a constructor or global variable bound to a function.
 *
 * For example: `square` is transformed to
 * `fn (%x: int32) -> int32 { square(x) }`.
 *
 * See https://en.wikipedia.org/wiki/Lambda_calculus#%CE%B7-conversion
 * for more details.
 *
 * \param expand_constructor Whether to expand constructors.
 * \param expand_global_var Whether to expand global variables.
 *
 * \return The pass.
 */
TVM_DLL Pass EtaExpand(bool expand_constructor, bool expand_global_var);

/*!
 * \brief Print the IR for a module to help debugging.
 *
 * \param show_meta_data The flag to control if meta data needs to be printed.
 *
 * \return the pass.
 */
TVM_DLL Pass PrintIR(bool show_meta_data = true);

/*!
 * \brief Partition a Relay program into regions that can be executed on
 * different backends.
 *
 * \return The pass.
 */
TVM_DLL Pass PartitionGraph();

/*!
 * \brief Inline the global functions marked as `inline` in a given Relay
 * IRModule.
 *
 * \return The pass.
 */
TVM_DLL Pass Inline();

/*!
 * \brief Remove the unused functions in the Relay IRModule.
 *
 * \param entry_functions The entry functions used to search the functions that
 *        are being used.
 *
 * \return The pass.
 */
TVM_DLL Pass RemoveUnusedFunctions(Array<runtime::String> entry_functions);

}  // namespace transform

/*!
 * \brief Bind the free variables to a Relay expression. This is a helper
 * function usually called by other pass functions to help optimizations.
 *
 * \param expr The input expression.
 * \param binds The variable to expression map that will be used to help the
 *        binding.
 *
 * \return The updated expression.
 */
TVM_DLL Expr Bind(const Expr& expr, const tvm::Map<Var, Expr>& binds);

/*!
 * \brief Infer the type of a function as if it is mapped to var in the mod.
 *
 * \param f the function.
 * \param mod The module used for referencing global functions.
 * \param var The global variable corresponding to the function.
 *
 * \return A type checked Function with its checked_type field populated.
 * \note this function mutates mod and is not thread-safe.
 */
TVM_DLL Function InferType(const Function& f,
                           const IRModule& mod,
                           const GlobalVar& var);

/*!
 * \brief Apply rewrite rules to rewrite the expr in post DFS order. This
 * function is used as a helper function to rewrtie an expression in a pass.
 *
 * \param expr The expression.
 * \param rewrite_map_attr_name The Op's attr name which corresponds to the rewrite
 *                              rule function.
 * \param fcontext Additional callback to provide context argument for each call node.
 * \param fmulti_ref_trigger Transformation function to be called when
 *                           an Expr consumed by multiple callers.
 * \return The rewritten expression.
 */
TVM_DLL Expr ForwardRewrite(const Expr& expr,
                            const std::string& rewrite_map_attr_name,
                            std::function<ObjectRef(const Call&)> fcontext = nullptr,
                            std::function<Expr(const Expr&)> fmulti_ref_trigger = nullptr);

/*!
 * \brief Apply rewrite rules to rewrite the expr in post DFS order. This
 * function is used as a helper function to rewrtie an expression in a pass.
 *
 * \param expr The expression.
 * \param rewrite_func The rewrite func that will apply to all operators.
 * \param fcontext Additional callback to provide context argument for each call node.
 * \param fmulti_ref_trigger Transformation function to be called when
 *                           an Expr consumed by multiple callers.
 *
 * \return The rewritten expression.
 */
TVM_DLL Expr ForwardRewrite(const Expr& expr,
                            const FForwardRewrite& rewrite_func,
                            std::function<ObjectRef(const Call&)> fcontext = nullptr,
                            std::function<Expr(const Expr&)> fmulti_ref_trigger = nullptr);

/*!
 * \brief Rewrite the annotated program.
 *
 * \param expr The expression.
 * \param fallback_device The fallback device which is the default device for
 *                        operators without annotation.
 *
 * \return The updated program.
 */
TVM_DLL Expr RewriteAnnotatedOps(const Expr& expr, int fallback_device);

/*!
 * \brief Turn an expression into continuation passing style(CPS).
 *
 * CPS mean that every function will, instead of returning the result directly,
 * be passed down an extra function (called the continuation) as argument,
 * and pass the result to the continuation instead.
 *
 * Thus, every function call has to be passed an extra argument
 * that represent the rest of the computation (Hence the name of continuation).
 *
 * Similarly, all other compute will be wrapped and call the continuation as well.
 *
 * \param f the function.
 * \param mod the module.
 *
 * \return the converted Function.
 */
TVM_DLL Function ToCPS(const Function& f, const IRModule& mod);

/*!
 * \brief Remove the continuation argument of a CPS function.
 *
 * Note that this only transform the type back into un-CPS form
 * when there is no higher order input/output.
 *
 * \param f the function.
 *
 * \return the converted Function.
 */
TVM_DLL Function UnCPS(const Function& f);

/*!
 * \brief Deduplicate the bound variables and type variables in the expression.
 *
 * \param e the expression.
 *
 * \return the deduplicated expression.
 */
TVM_DLL Expr DeDup(const Expr& e);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_TRANSFORM_H_
