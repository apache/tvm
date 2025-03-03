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
#include <tvm/tir/function.h>
#include <tvm/tir/index_map.h>
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
 * \brief Activate force_pure on all pure functions in the module
 * and unwrap all pure override ops into the normal versions.
 *
 * This effectively means that there will be no more purity tracking,
 * useful for low-level code generation.
 *
 * \return The Pass.
 *
 * \note Should be used after ToNonDataflow()
 */
TVM_DLL Pass RemovePurityChecking();

/*!
 * \brief Perform explicit tensor allocation for call_tir and call_dps_packed.
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
 * The pass "supports" dynamic shape in the way of TIR variable upper bound
 * annotation. We can optionally annotate the attribute "tir_var_upper_bound"
 * to Relax functions. The attribute value is a dict from strings to integers,
 * denoting the name of TIR variables to the upper bound values of the TIR vars.
 * Note: The annotated upper bound attribute only applies to TIR vars in the
 * function signature for clarity.
 *
 * For example, we can annotate a Relax function with
 *   `R.func_attr({"tir_var_upper_bound": {"n": 1024}})`.
 * It means the maximum value of variable that names "n" in the function
 * signature will have upper bound 1024. And we will use 1024 as its value
 * during memory planning.
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
 * \brief Possibly rename the GlobalVar in an IRModule to ensure these properties:
 * 1. (Invariant) First ensure every public function has the same name as its "global_symbol"
 *    attribute;
 * 2. To ensure 1., we may need to rename private functions with conflicting names;
 * 3. Finally, the name of every GlobalVar is unique in the IRModule.
 */
TVM_DLL Pass NormalizeGlobalVar();

/*!
 * \brief Simplify a Relax module by folding var bindings and match shape nodes,
 * as well as tuple indices.
 * Best used alongside constant folding and eliminating unused bindings.
 *
 * \note If a dataflow var is used only in a binding to the dataflow block
 * output var (i.e., a non-dataflow var), this pass will also remove the dataflow var
 * and replaces the output var's binding with the dataflow var's direct definition.
 *
 * \return The Pass.
 */
TVM_DLL Pass CanonicalizeBindings();

/*!
 * Eliminate common subexpressions within functions.
 * \return The pass that eliminates common subexpressions.
 *
 * \note For nested functions, this pass performs CSE *within* those functions.
 * \param call_only If true, enable eliminating only call nodes.
 */
TVM_DLL Pass EliminateCommonSubexpr(bool call_only = false);

/*!
 * \brief Bind params of function of the module to constant tensors.
 *
 * \param func_name The name of the function to bind parameters.
 * \param params The parameters to bind.
 *
 * \return The Pass.
 */
TVM_DLL Pass BindParams(String func_name, Map<ObjectRef, ObjectRef> params);

/*!
 * \brief Bind symbolic vars to constant shape values.
 *
 * \param binding_map The dictionary of symbolic variables and their
 *      constant shape values.  Dictionary keys may be either a
 *      `tir.Var` or a string name of the variable.  If the variables
 *      are referred to by name, the name must uniquely identify a
 *      symbolic variable in each function where it is used.
 *
 * \param func_name The name of the function in which to bind shape
 *      values.  If NullOpt, all functions in the module will be
 *      updated.
 *
 * \return The Pass.
 */
TVM_DLL Pass BindSymbolicVars(Map<ObjectRef, PrimExpr> binding_map,
                              Optional<String> func_name = NullOpt);

/*!
 * \brief Fold constant expressions within dataflow blocks.
 *
 * \note ConvertToDataflow may need to be called first to provide dataflow blocks.
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
 * \param enable_warning A boolean value indicating if to print warnings for TIR functions not
 * showing up in the database.
 * \return The Pass.
 */
TVM_DLL Pass LegalizeOps(Optional<Map<String, PackedFunc>> cmap, bool enable_warning = false);

/*!
 * \brief Propagate virtual device information.
 * \return The Pass.
 */
TVM_DLL Pass RealizeVDevice();

/*!
 * \brief Attach layout free buffers to the tir::PrimFunc.
 *
 * This pass is used to attach layout free buffers to the tir::PrimFunc according to
 * the function usage in the relax function. Currently, the layout free buffers are the model
 * weights and relax constants.
 *
 * \note We recommend applying CanonicalizeBindings before this pass.
 * \return The Pass.
 */
TVM_DLL Pass AttachAttrLayoutFreeBuffers();

/*!
 * \brief Split the layout rewrite preproc block to a separate tir::PrimFunc.
 *
 * This pass is used in the prepack weight after meta_schedule tuning.
 *
 * \return The Pass.
 */
TVM_DLL Pass SplitLayoutRewritePreproc();

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
 * \param shared_transform Indicates how the parameter transformation function will be produced.
 *    - `False` (default): A separate parameter transformation function will be produced for each
 *      function with the `"num_input"` attribute.
 *
 *    - `True`: A single parameter transformation function will be produced, containing the
 *      preprocessing steps common across all functions with the `"num_input"` attribute.
 *
 *    - List[str]: A single parameter transformation function will be produced, containing the
 *      preprocessing steps common across each function whose name is in the list. Passing a list of
 *      all functions with the `"num_input"` attribute or an empty list is equivalent to passing
 *      `True`.
 *
 * \return The Pass.
 */
TVM_DLL Pass LiftTransformParams(Variant<Bool, Array<String>> shared_transform = Bool(false));

/*!
 * \brief Update virtual device.
 * \param new_vdevice The new virtual device.
 * \param index The device index indicates the device on which the update will be performed.
 * \return The Pass.
 */
TVM_DLL Pass UpdateVDevice(VDevice new_vdevice, int64_t index);

/*! \brief Expand tuple arguments to internal functions
 *
 * \return The Pass
 */
TVM_DLL Pass ExpandTupleArguments();

/*! \brief Remove unused parameters to internal functions
 *
 * \return The Pass
 */
TVM_DLL Pass RemoveUnusedParameters();

/*! \brief Remove unused outputs from internal functions
 *
 * \return The Pass
 */
TVM_DLL Pass RemoveUnusedOutputs();

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
 * \brief The pattern object used as the input of FuseOpsByPattern. For bindings to be
 * fused, it needs to be matched with `pattern` and the `check` function needs to return
 * true.
 */
class FusionPatternNode : public Object {
 public:
  /*!
   * \brief The name of pattern. It becomes the value of the kComposite attribute
   * of a fused function after successful matching
   */
  String name;

  /*!
   * \brief The dataflow pattern that will be used to match expression in the DataflowBlock.
   * All the call nodes covered by the pattern will be extracted into the fused function.
   */
  DFPattern pattern;

  /*!
   * \brief The map which is used to extract important expressions from the pattern match
   * result. All DFPattern in this map should be part of the `pattern`.
   */
  Map<String, DFPattern> annotation_patterns;

  /*!
   * \brief The function to determine whether the match result is accepted. This can be
   * NullOpt if check function is not necessary for this pattern.
   *
   * It should have signature
   * bool(const PatternCheckContext& context)
   */
  Optional<PackedFunc> check;

  /*!
   * \brief The function to get attributes for fused function
   *
   * It should have signature
   * Map<String, ObjectRef>(const Map<String, Expr>& context)
   */
  Optional<PackedFunc> attrs_getter;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("pattern", &pattern);
    v->Visit("annotation_patterns", &annotation_patterns);
    v->Visit("check", &check);
    v->Visit("attrs_getter", &attrs_getter);
  }

  static constexpr const char* _type_key = "relax.transform.FusionPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(FusionPatternNode, Object);
};

class FusionPattern : public ObjectRef {
 public:
  FusionPattern(String name, DFPattern pattern, Map<String, DFPattern> annotation_patterns,
                Optional<PackedFunc> check, Optional<PackedFunc> attrs_getter);

  FusionPattern(String name, DFPattern pattern)
      : FusionPattern(name, pattern, {}, NullOpt, NullOpt) {}

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(FusionPattern, ObjectRef, FusionPatternNode);
};

/*!
 * \brief The input of FusionPattern::check.
 */
class PatternCheckContextNode : public Object {
 public:
  /*!
   * \brief The expression that's matched with the FusionPattern::pattern.
   */
  Expr matched_expr;

  /*!
   * \brief A map which contains all expressions matched by the sub patterns in
   * FusionPattern::annotation_patterns.
   */
  Map<String, Expr> annotated_expr;

  /*!
   * \brief Map from variable to its value. It contains variables from bindings that
   * is being fused by FuseOpsByPattern.
   */
  Map<Var, Expr> matched_bindings;

  /*!
   * \brief A map mapping variable definitions to a set of uses. It has all variables
   * used in the function.
   */
  Map<Var, Array<Var>> var_usages;

  /*!
   * \brief Map from value to its bound variable. It doesn't have variables after the
   * matched expression.
   */
  Map<Expr, Var> value_to_bound_var;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("matched_expr", &matched_expr);
    v->Visit("annotated_expr", &annotated_expr);
    v->Visit("matched_bindings", &matched_bindings);
    v->Visit("var_usages", &var_usages);
    v->Visit("value_to_bound_var", &value_to_bound_var);
  }

  static constexpr const char* _type_key = "relax.transform.PatternCheckContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(PatternCheckContextNode, Object);
};

class PatternCheckContext : public ObjectRef {
 public:
  PatternCheckContext(Expr matched_expr, Map<String, Expr> annotated_expr,
                      Map<Var, Expr> matched_bindings, Map<Var, Array<Var>> var_usages,
                      Map<Expr, Var> value_to_bound_var);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(PatternCheckContext, ObjectRef,
                                            PatternCheckContextNode);
};

/*!
 * \brief Reverse-mode automatic differentiation.
 *
 * This pass will differentiate one function in the IRModule. Now the input function must have only
 * one dataflow block.
 *
 * For a given function specified by `func_name`, it generates a new function with the name
 * `func_name + "_adjoint"`. The new function computes the gradient of the **differentiation
 * target** with respect to the arguments specified by `require_grads` of the original function.
 *
 * If the function has only one return value, the return value will be specified as target. If the
 * function has more than one return values, the target will be specified as the target_index-th
 * return value. The target must be a scalar (0-dim tensor).
 *
 * \param func_name The name of the specified function.
 * \param require_grads The relax variables whose adjoints is needed. Must be parameters of the
 * given function and should not be duplicate. If it is not specified, adjoints of all parameters
 * would be computed.
 * \param target_index If the specified function has more than one return values, specify the index
 * of the return value as the target. If it is not specified, the first return value will be the
 * target.
 * \return The Pass.
 *
 * \note ConvertToDataflow may need to be called first to provide dataflow blocks.
 */
TVM_DLL Pass Gradient(String func_name, Optional<Array<Var>> require_grads = NullOpt,
                      int target_index = 0);

/*!
 * \brief Apply pattern matching to each function in the given module, and group matched
 * expressions into a new function. The end result is similar to FuseOps, but fusion is driven
 * completely by the provided patterns.
 *
 * \param patterns The patterns to detect. The order of the patterns determines the order
 * of priority in which they are matched. Higher-priority patterns should come earlier in the list.
 * \param bind_constants Whether or not to keep bound constants of the grouped function.
 * \param annotate_codegen If true, wrap each created composite function with another function,
 * whose body consists only of a call to the composite function, and annotate the outer function
 * with kCodegen and kGlobalSymbol attributes. The kCodegen attribute is set as the prefix of the
 * corresponding pattern name. For example, "dnnl" if the pattern name is "dnnl.conv2d_relu".
 * This must be True if the created composite functions are intended to be offloaded to
 * an external backend without using the MergeCompositeFunctions pass.
 * \param entry_function_names The names of functions that should be considered as entry points. If
 * not specified, all externally exposed functions will be considered as entry points.
 * \return The Pass.
 *
 * \note Only operates within dataflow blocks. ConvertToDataflow may need to be called first.
 */
TVM_DLL Pass FuseOpsByPattern(const tvm::Array<FusionPattern>& patterns, bool bind_constants = true,
                              bool annotate_codegen = false,
                              const tvm::Array<String>& entry_function_names = {});

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
 * \brief Run codegen.
 * \param target_options pairs of target name and compilation options
 * \param entry_functions list of entry functions
 * \return The Pass.
 */
TVM_DLL Pass RunCodegen(Optional<Map<String, Map<String, ObjectRef>>> target_options,
                        Array<runtime::String> entry_functions);

/*!
 * \brief Decompose composite operators during inference. For example, The result of batch norm (a
 * triple) will be simplified. Operators like Attention, Erf, etc. can be also simplified into
 * several operators as well.
 *
 * \param func_name The name of the specified function. If not specified, the pass will run in
 * all functions.
 */
TVM_DLL Pass DecomposeOpsForInference(Optional<String> func_name);

/*!
 * \brief Decompose composite operators during training. For example, The result of batch norm (a
 * triple) will be simplified. Operators like Attention, Erf, etc. can be also simplified into
 * several operators as well.
 *
 * \param func_name The name of the specified function. If not specified, the pass will run in
 * all functions.
 */
TVM_DLL Pass DecomposeOpsForTraining(Optional<String> func_name);

/*!
 * \brief Returns a pass which replaces PrimFuncs which have matching kOperatorName attribute in \p
 * op_impl_map, with replacement PrimFunc that could possibly have different layouts on i/o
 * buffers. The layout transformations on i/o buffers is present in the \p op_buffer_transforms. The
 * pass inserts the layout transformations in the call sites of PrimFuncs being replaced to
 * transform i/o buffers into expected layout.
 *
 * \param op_impl_map Map from kOperatorName attr (e.g., relax.conv2d) to replacement PrimFunc
 * \param op_buffer_transforms Map from kOperatorName attr to layout transformations on each of the
 * PrimFunc i/o buffers.
 * \param axis_separators Map from kOperatorName attr to axis_separators of each buffer_transforms
 * \param input_axis_separators Map from kOperatorName attr to axis_separator for input buffer
 * \return The Pass.
 */
TVM_DLL Pass AlterOpImpl(const Map<String, tir::PrimFunc>& op_impl_map,
                         const Map<String, Array<tir::IndexMap>>& op_buffer_transforms,
                         const Map<String, Array<Array<IntImm>>>& axis_separators,
                         const Map<String, Array<Array<IntImm>>>& input_axis_separators);

/*!
 * \brief Layout conversion pass.
 * \param desired_layouts The desired layouts for some operators.
 * \return The Pass.
 * \note Operates only on dataflow blocks. ConvertToDataflow may need to be called first.
 */
TVM_DLL Pass ConvertLayout(Map<String, Array<String>> desired_layouts);

/*!
 * \brief A pass that converts consecutive dataflow operations
 *   inside binding blocks into dataflow blocks.
 * \param min_size The minimum number of consecutive dataflow bindings
 *   required for the pass to create a new dataflow block
 * \return The Pass.
 */
TVM_DLL Pass ConvertToDataflow(int min_size = 2);

/*!
 * \brief Dead code elimination.
 * \sa RemoveAllUnused
 * Currently it removes:
 *   1. Unused local VarBindings
 *      (those where the bound var is unused and no impure operation is used).
 *   2. Unused Relax functions in the module.
 *      We detect the call chain from the entry function, and remove all unused functions.
 *
 * Any binding blocks that are left empty will be removed by the normalizer.
 *
 * \param entry_functions Names of functions that should be considered
 *     as entry points, in addition to any externally exposed functions.
 *
 * \return The Pass.
 */
TVM_DLL Pass DeadCodeElimination(Array<runtime::String> entry_functions = {});

/*!
 * \brief Pass that changes calls to operators that can be done in-place
 * (generally, these are elementwise operations) in dataflow blocks into in-place implementations.
 * Supported operators will be replaced by calls to `call_tir_inplace` that invoke in-place
 * PrimFunc implementations of those operators (which are based on the legalizations of those
 * operators).
 * \note ConvertToDataflow may need to be called first to provide dataflow blocks.
 * \return The pass.
 */
TVM_DLL Pass DataflowUseInplaceCalls();

/*!
 * \brief Automatic mixed precision pass. Currently the pass assumes the input module to be fp32
 * only, and will automatically cast fp32 to fp16 for certain ops.
 * \param out_dtype The output data type of gemm/conv, which is the data type of the accumulator.
 * \param fp16_input_names The names of function parameters whose dtype should become fp16. The
 * function signature would change accordingly.
 * \return The Pass.
 *
 * \note Mainly operates within dataflow blocks. ConvertToDataflow may need to be called first.
 */
TVM_DLL Pass ToMixedPrecision(const DataType& out_dtype,
                              Optional<Array<String>> fp16_input_names = NullOpt);

/*!
 * \brief Rewrite a Relax module for executing with CUDA graph. This pass identifies
 * the regions that can be executed with CUDA graph and lifts them into new functions for runtime
 * graph capturing.
 */
TVM_DLL Pass RewriteCUDAGraph();

/*!
 * \brief The pass is designed for few shot tuning for static shape PrimFuncs. It examines all the
 *  blocks within the PrimFunc and conducts loop fusion, splitting, and other transformations based
 *  on MetaSchedule schedule rules but directly samples from the search space instead of using the
 *  tuning algorithm. User can specify the number of valid counts to try and whether to use runner
 *  for benchmarking.
 * \param valid_count The number of valid counts to try.
 * \param benchmark Whether to use runner for benchmarking.
 * \return The Pass.
 */
TVM_DLL Pass FewShotTuning(int valid_count, bool benchmark);

}  // namespace transform
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_H_
