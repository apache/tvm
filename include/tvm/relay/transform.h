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

#include <tvm/ir/transform.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/target/compilation_config.h>
#include <tvm/target/target.h>
#include <tvm/target/virtual_device.h>

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
using FTVMRelayToTIR = tvm::transform::Pass;
/*!
 * \brief TIRToRuntime conversion specific to a TargetKind
 *
 * This function is responsible for scanning an IRModule for appropriate Target-specific functions
 and generating a Runtime module representing the compiled output
 *
 * \param ir_module Unified IRModule
 * \param target Target to filter on or retrieve arguments from
 * \return Runtime Module containing compiled functions
 */
using FTVMTIRToRuntime = tvm::runtime::TypedPackedFunc<runtime::Module(IRModule, Target)>;

/*!
 * \brief RelayToTIR tvm::transform::Pass specific to a TargetKind
 *
 * Called before the default lowering passes.
 *
 * \param mod The module that an optimization pass runs on.
 * \param pass_ctx The pass context that can provide information for the optimization.
 *
 * \return The transformed module.
 */
using FTVMRelayToTIR = tvm::transform::Pass;

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
TVM_DLL Pass CreateFunctionPass(
    const runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>& pass_func,
    int opt_level, String name, tvm::Array<String> required, bool traceable = false);

/*! \brief Remove let-bound expressions which do not effect the program result.
 *
 * This pass will remove let bindings which are not referenced. If inline_once is True,
 * let bindings which are only referenced once will also be inlined.
 *
 * For example, this pass should turn `let a = 1; 2` into `2`,
 * as the value of the expression does not depend on a.
 *
 * As another example, `let a = 1; a` will be optimized into 1 if inline_once is True.
 *
 * If ignore_purity is False, possibly side-effecting expressions (such as memory allocation,
 * random number generation, reading/writing references, or calls to primitive or external
 * functions) are never elided or inlined. This is sound, but ignore_purity can be set to True
 * to suppress this check.
 *
 * The analysis is fairly conservative, for example it assumes all local functions
 * may be called more than once, any functions passed as arguments have side effects,
 * and so on.
 *
 * \param inline_once whether or not to inline bindings used exactly once.
 * \param ignore_purity whether to ignore whether expressions have side-effects
 *
 * \return the pass.
 */
TVM_DLL Pass DeadCodeElimination(bool inline_once = false, bool ignore_purity = false);

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
 *  Because of backward compatibility reason it skips QNN primitives from folding by default.
 *  There are some transformation passes like FakeQuantizationToInteger, which requires to keep QNN
 *  primitives for constant subgraphs. Uncontrolled constant folding of QNN primitives may break
 *  applicability of FakeQuantizationToInteger. We suggest to use FoldConstant pass with none
 *  default fold_qnn=True value only when all other QNN sensitive passes were already applied.
 *
 * \param fold_qnn Whether to fold constants for QNN operations.
 *
 * \return The pass.
 */
TVM_DLL Pass FoldConstant(bool fold_qnn = false);

/*!
 * \brief Split function with huge number of arguments to smaller pieces.
 *
 * \param max_function_args Maximum number of function arguments. If it equals 0 then SplitArgs
 *                          shouldn't split the function.
 *
 * \return The pass.
 */
TVM_DLL Pass SplitArgs(uint64_t max_function_args);

/*!
 * \brief Fuse operations into expr into separate functions.
 *
 * \param fuse_opt_level Optimization level. If it is -1 it will be inferred from pass context.
 *
 * \return The pass.
 */
TVM_DLL Pass FuseOps(int fuse_opt_level = -1);

/*!
 * \brief The inverse operation of FuseOps. It transforms a fused program returned by
 * FuseOps into the program before FuseOps. (i.e. x == DefuseOps(FuseOps(x)))
 *
 * \return The pass.
 */
TVM_DLL Pass DefuseOps();

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
 * \brief Turn an expression to Basic Block Normal Form.
 *
 * We define a block as a group of expressions implied by the scope structure.
 *
 * Each graph node can only belong to a single block.
 *
 * For any value that is being used in multiple blocks, it has to be referred
 * by a Var which is defined in a block, whose scope is the least common ancestor
 * of blocks this value is used.
 *
 * \return The pass.
 */
TVM_DLL Pass ToBasicBlockNormalForm();

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
 * \brief ToANormalForm but on incomplete graph.
 *
 * \param expr the graph.
 *
 * \return The transformed program.
 */
TVM_DLL Expr ToANormalForm(const Expr& expr);

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
 * \brief Simplify certain operators during inference. For example, the result
 * of a batch norm which is indexed at tuple index 0 will be unpacked into a
 * number of simplified operators.
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
 * \brief Find Dynamic ops and make them static
 *
 * Searches the graph for dynamic ops. If the dynamic inputs to those ops are constants, it replaces
 * them with static ops and re-performs type inference and constant folding. The pass repeats
 * itself until the graph stops changing or we run too many iterations.
 *
 * \return The pass.
 */
TVM_DLL Pass DynamicToStatic();

/*!
 * \brief Infer the type of an expression.
 *
 * The result of type checking is a new expression with unambiguous
 * type information filled in, as well as it's checked type field
 * populated with the result type.
 *
 * \return The pass.
 */
TVM_DLL Pass InferType();

/*!
 * \brief Infer the type of an expression, reusing existing type information.
 *
 * The result of type checking is a new expression with unambiguous
 * type information filled in for the given node only. The local
 * version can use existing type information populated throughout
 * the expression and assumes this information is correct. The local
 * version also avoids examining large amounts of the graph assuming
 * type information is filled in properly which makes it much faster if we
 * iteratively call type inference.
 *
 * \return The type of the expression.
 */
TVM_DLL Type InferTypeLocal(const Expr& expr);

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
 * \param to_batch_matmul Whether to combine parallel dense ops to batch matmul.
 *                        If set false, combine dense ops to single dense op.
 *
 * \return The pass.
 */
TVM_DLL Pass CombineParallelDense(uint64_t min_num_branches = 3, bool to_batch_matmul = true);

/*!
 * \brief Combine parallel batch_matmul ops into a single batch_matmul
 *  if the number of branches of this dense operator is not less than
 * `min_num_branch`.
 *
 * \param min_num_branches The minimun number of branches.
 *
 * \return The pass.
 */
TVM_DLL Pass CombineParallelBatchMatmul(uint64_t min_num_branches = 3);

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
 * \brief Do layout rewrite according to the tile structure created by auto-scheduler.
 * \return The pass
 */
TVM_DLL Pass AutoSchedulerLayoutRewrite();

/*!
 * \brief Do layout rewrite according to the tile structure created by meta-schedule.
 * \return The pass
 */
TVM_DLL Pass MetaScheduleLayoutRewrite();

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
 * \param desired_layouts Specify mapping of op_name to array of desired layouts for each input.
 *                        For example: Map("nn.conv2d", Array("NHWC", "OHWI")),
 *                        this specifies the desired layout for data then kernel for nn.conv2d.
 * \return The pass.
 */
TVM_DLL Pass ConvertLayout(const Map<String, Array<String>>& desired_layouts);

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
TVM_DLL Pass Legalize(const String& legalize_map_attr_name = "FTVMLegalize");

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

/*!
 * \brief Simplify the Relay expression.
 *
 * \return The pass.
 */
TVM_DLL Pass SimplifyExpr();

/*!
 * \brief Stripped down version of SimplifyExpr which is run after AlterOpLayout.
 *
 * \return The pass.
 */
TVM_DLL Pass SimplifyExprPostAlterOp();

/*!
 * \brief Run any custom passes registered under "RelayToTIR" attributes on TargetKinds.
 *
 * This pass looks for inline, let-bound or global functions which have a "Compiler" attribute.
 * If the attribute value corresponds to a TargetKind with a "RelayToTIR" attribute, then the
 * 'custom' pass bound to that attribute is run (at most once) on the IRModule as a whole.
 *
 * If, in addition, the \p config has a Target with a matching TargetKind, that Target is set
 * as the 'current' target before the custom pass is executed. In this way it is possible
 * for custom passes to pick up target options which may guide how they transform the IRModule.
 * (Those targets are referred to as 'extern codegen targets' elsewhere).
 *
 * A typical custom pass will:
 *  - Find calls to "Compiler" attributes functions with matching compiler name.
 *  - Lower those function to TIR PrimFuncs.
 *  - Bind those functions into the IRModule under the functions' "global_symbol" attribute.
 *  - Replace all calls to those functions with 'call_lowered' to the matching global.
 * Care should be taken to handle multiple calls to the same function.
 * See src/relay/backend/contrib/example_target_hooks/relay_to_tir.cc for an example custom pass.
 *
 * It is also possible (despite the pass and attribute names!) for the custom pass to proceed
 * directly to a runtime::Module, which can be attached to the output IRModules "external_mods"
 * attribute (taking care not to clobber any existing modules). In this case the flow is as above,
 * except:
 *  - The runtime::Module must contain a binding for each compiled function under their
 *    "global_symbol" (ie runtime::Module::ImplementsFunction should return true).
 *  - A Relay Function must be bound (or re-bound) into the result IRModule, again with the same
 *    "global_symbol", but with only the "Extern" attribute set to Integer(1). The function body
 *    should be the original function body. In this way we always have a TVM definition matching
 *    every global function name.
 *
 * There are many existing runtime::Modules, ranging from source to object to dynamic libaries to
 * entirely custom implementations. Some of those may require additional compilation using
 * 'export_library' on the final build artifact.
 *
 * The OutlineCompilerFunctionsWithExistingGlobalSymbols and MarkCompilerFunctionsAsExtern utility
 * passes can be used by custom passes to take care of some of the boilerplate.
 *
 * TODO(mbs): Rename PreLoweringTargetHooks?
 *
 * \param config All available targets.
 *
 * \return The pass.
 */
TVM_DLL Pass RelayToTIRTargetHook(CompilationConfig config);

/*!
 * \brief A pass for manifesting explicit memory allocations and rewriting
 * specific dialects.
 *
 * \param cpu_virtual_device VirtualDevice for computations and data which must reside on a CPU,
 * such as shapes and shape functions.
 *
 * \return The pass.
 */
TVM_DLL Pass ManifestAlloc(VirtualDevice cpu_virtual_device);

/*!
 * \brief A pass for manifesting variable lifetimes by inserting kill operations when variables
 * become dead. This pass should be run after ManifestAlloc, and should not be run more than once.
 *
 * \return The pass.
 */
TVM_DLL Pass ManifestLifetimes();

/*!
 * \brief Uses existing "on_device" and "device_copy" CallNodes to infer the \p VirtualDevice on
 * which every Relay sub-expression should run and the result stored. Captures the result of that
 * analysis using new "on_device" and "device_copy" CallNodes.
 *
 * See tvm::relay::transform::{LexicalOnDeviceMixin,DeviceAwareExprVisitor,DeviceAwareExprMutator}
 * for help recovering the device for an arbitrary sub-expression in downstream transformations.
 *
 * \param config Describes the targets and default \p VirtualDevice for all primitive operators and
 * host sub-expressions.
 *
 * \return The pass.
 */
TVM_DLL Pass PlanDevices(CompilationConfig config);

/*!
 * \brief This transform flattens atrous convolution, which corresponds to the sequence of
 * operations: "space_to_batch_nd"->"conv2d"->"batch_to_space_nd" and convert them into subgraphs
 * with a convolution with the modified "dilation" and recalculated "padding" parameters.
 *
 * \return The pass.
 */
TVM_DLL Pass FlattenAtrousConv();

/*!
 * \brief Annotates the minimum required memory of each primitive function callsite by analyzing
 * the liveness of the input/output tensors at each function callsite and calculating the total
 * amount of memory these tensors require. This is added as a "used_memory" annotation to the
 * function in question as a list of the number of bytes for each callsite. In addition, the
 * containing function is annotated with an "io_used_memory" annotation which refers to the total
 * memory required for the IO tensors.
 *
 * Note: This pass does not support dynamic shapes, it is the users responsibility to check this
 * pass isn't applied where dynamic shapes may be input.
 */
TVM_DLL Pass AnnotateUsedMemory();

/*!
 * \brief Captures the post-dfs index and dominator post-dfs index of (most) expression nodes in
 * their span, in the form "index:<post-dfs index>:<dominator post-dfs index>". This is useful for
 * debugging since a) it helps identify pretty-printed sub-expressions within the overall model
 * and b) the indexes are heavily used by Collage for its compact representation of sub-graphs.
 *
 * Note that Op and Constructor nodes are not changed even though they are assigned an
 * post-dfs index.
 */
TVM_DLL Pass CapturePostDfsIndexInSpans();

/*!
 * \brief Calls device dependent memory scope analysis pass, collects mapping of desirable
 * expr->memory_scope and annotates expressions by VirtualDevice with required memory_scope
 */
TVM_DLL Pass AnnotateMemoryScope();

/*!
 * \brief Removes non-fused reshapes after lowering the graph.
 * InferType() cannot be invoked after calling this pass as it removes reshapes from the call
 * graph. Many targets only need buffer addresses irrespective of the shapes of them. This makes
 * reshapes symbolic once the graph has been lowered. Reshape removal results into smaller code
 * size and reduced buffer allocations. It opens up opportunities of operator fusion in the target
 * backend. Thus, consequently, it improves the performance of the inference.
 */
TVM_DLL Pass RemoveStandaloneReshapes();

}  // namespace transform

/*!
 * \brief Bind the free variables to a Relay expression. This is a helper
 * function usually called by other pass functions to help optimizations.
 * If any free variables are introduced into a function, those are added
 * to the functoin parameters.
 * Additionally this may change the order of parameters if you map a variable
 * to a variable.
 *
 * \param expr The input expression.
 * \param binds The variable to expression map that will be used to help the
 *        binding.
 *
 * \return The updated expression.
 */
TVM_DLL Expr Bind(const Expr& expr, const tvm::Map<Var, Expr>& binds);

/*!
 * \brief Substitute variables with new variables (including function parameters) in a function.
 * This is a helper function usually called by other pass functions to help optimizations.
 * Expects all values in the bind map to be Vars.
 *
 * \param func The input function.
 * \param binds The variable to expression map that will be used to help the
 *        binding.
 *
 * \return The updated expression.
 */
TVM_DLL Function SubstituteBoundVars(const Function& func, const tvm::Map<Var, Expr>& binds);

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
TVM_DLL Expr ForwardRewrite(const Expr& expr, const String& rewrite_map_attr_name,
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
TVM_DLL Expr ForwardRewrite(const Expr& expr, const FForwardRewrite& rewrite_func,
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

namespace legalize {
TVM_DLL Expr Legalize(const Expr& expr, const std::string& legalize_map_attr_name);
}  // namespace legalize

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_TRANSFORM_H_
