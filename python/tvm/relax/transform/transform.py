# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Relax transformation passes."""
import functools
import inspect
import types
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np  # type: ignore

import tvm.ir
from tvm.relax import Expr, Var
from tvm.relax.dpl import DFPattern
from tvm.runtime import NDArray, Object
from tvm.tir import IndexMap, PrimFunc

from . import _ffi_api
from .legalize_ops.common import LegalizeFunc


@tvm._ffi.register_object("relax.FunctionPass")
class FunctionPass(tvm.ir.transform.Pass):
    """A pass that works on each tvm.relax.Function in a module. A function
    pass class should be created through `function_pass`.
    """


@tvm._ffi.register_object("relax.DataflowBlockPass")
class DataflowBlockPass(tvm.ir.transform.Pass):
    """A pass that works on each tvm.relax.DataflowBlock in a module."""


def ToNonDataflow() -> tvm.ir.transform.Pass:
    """Transform all dataflow structure to non-dataflow version.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.ToNonDataflow()  # type: ignore


def LambdaLift():
    """A pass that lifts local functions into global.

    Returns
    -------
    ret : tvm.ir.transform.Pass
    """
    return _ffi_api.LambdaLift()


def CallTIRRewrite() -> tvm.ir.transform.Pass:
    """Perform explicit tensor allocation for call_tir and call_dps_packed.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.CallTIRRewrite()  # type: ignore


def Normalize() -> tvm.ir.transform.Pass:
    """Transforming Relax IR to normal form, i.e., the expressions are normalized(no nesting
    and hence the AST is in ANF), and all checked_type_ and shape_ of expressions are available.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.Normalize()  # type: ignore


def CanonicalizeBindings() -> tvm.ir.transform.Pass:
    """
    Canonicalizes variable definitions
    (e.g., if there is y = x and z = y, it replaces uses of y and z with x).

    Best combined with constant folding and the elimination of unused definitions.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.CanonicalizeBindings()  # type: ignore


def EliminateCommonSubexpr(fskip=None):
    """Eliminate common subexpressions within dataflow blocks.

    Parameters
    ----------
    fskip: Callable
        The callback function that decides whether an expression should be
        skipped.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that eliminates common subexpressions.
    """
    return _ffi_api.EliminateCommonSubexpr(fskip)


def RewriteDataflowReshape() -> tvm.ir.transform.Pass:
    """Convert all reshape-like call_tir to VM reshape operator call.
    The VM reshape operator calls will be further lowered to a CreateView
    operation at runtime, instead of doing real data copy.
    Here "reshape-like" includes reshape, expand_dims, flatten, etc.

    Returns
    -------
    ret : tvm.ir.transform.Pass
    """
    return _ffi_api.RewriteDataflowReshape()  # type: ignore


def StaticPlanBlockMemory() -> tvm.ir.transform.Pass:
    """The static memory planning pass on BindingBlock level.
    The pass will reuse allocated memory to its best effort, in order to
    reduce the total amount of allocated memory size.
    Returns
    -------
    ret : tvm.ir.transform.Pass
    """
    return _ffi_api.StaticPlanBlockMemory()  # type: ignore


def VMBuiltinLower() -> tvm.ir.transform.Pass:
    """Lowering generic intrinsic to VM intrinsics.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.VMBuiltinLower()  # type: ignore


def VMShapeLower(*, emit_err_ctx: bool = True) -> tvm.ir.transform.Pass:
    """Lower the symbolic shape and argument and match-cast structinfo matching.

    Parameters
    ----------
    emit_err_ctx: Optional[bool]
        Whether emit err context string, can be turned off for testing purposes.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.VMShapeLower(emit_err_ctx)  # type: ignore


def AttachGlobalSymbol() -> tvm.ir.transform.Pass:
    """Attach global_symbol to Relax functions and TIR Primfuncs for codegen.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.AttachGlobalSymbol()  # type: ignore


def BindParams(
    func_name: str,
    params: Dict[str, Union[tvm.runtime.NDArray, np.ndarray]],
) -> tvm.ir.transform.Pass:
    """Bind params of function of the module to constant tensors.

    Parameters
    ----------

    func_name: str
        The function name to be bound

    params : Dict[str, Union[tvm.runtime.NDArray, np.ndarray]]
        The map from param name to constant tensors.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    tvm_params = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            v = tvm.nd.array(v)
        assert isinstance(
            v, tvm.runtime.NDArray
        ), f"param values are expected to be TVM.NDArray or numpy.ndarray, but got {type(v)}"
        tvm_params[k] = v

    return _ffi_api.BindParams(func_name, tvm_params)  # type: ignore


def RunCodegen(
    target_options: Optional[dict] = None,
    entry_functions: Optional[List[str]] = None,
) -> tvm.ir.transform.Pass:
    """Produce the runtime::Module with an annotated codegen and global symbol.

    Parameters
    ----------
    target_options: Optional[dict]
        Pairs of a target name and compilation options
    entry_functions: Optional[List[str]]
        The set of entry functions to start from.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass to remove unused functions.
    """
    if entry_functions is None:
        entry_functions = ["main"]
    # enable cutlass byoc registries
    # pylint: disable=unused-import,import-outside-toplevel
    from tvm.contrib import cutlass as _cutlass

    return _ffi_api.RunCodegen(target_options, entry_functions)  # type: ignore


def FoldConstant() -> tvm.ir.transform.Pass:
    """Fold constant expressions.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.FoldConstant()  # type: ignore


def AnnotateTIROpPattern() -> tvm.ir.transform.Pass:
    """Annotate Op Pattern Kind for TIR functions

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.AnnotateTIROpPattern()  # type: ignore


def FuseOps(fuse_opt_level=-1) -> tvm.ir.transform.Pass:
    """This pass groups bindings in a dataflow block of Relax functions and generate a new grouped
    Relax function for each group, according to the fusion algorithm described in the pass
    implementation. By grouping bindings into new Relax functions, we substitute the bindings in
    the function being manipulated into function calls to the new grouped function.

    A follow-up pass named "FuseTIR" will generate a TIR PrimFunc for each grouped function.

    Parameters
    ----------
    fuse_opt_level : int
        The level of fuse optimization. -1 indicates that the level will be
        inferred from pass context.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for operator fusion.
    """
    return _ffi_api.FuseOps(fuse_opt_level)  # type: ignore


def FuseTIR() -> tvm.ir.transform.Pass:
    """Fuse primitive relax function into a larger TIR function if possible

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for tir fusion.
    """
    return _ffi_api.FuseTIR()  # type: ignore


@tvm._ffi.register_object("relax.transform.PatternCheckContext")
class PatternCheckContext(Object):
    """
    The input of check function `FusionPattern.check`.

    Parameters
    ----------
    annotated_expr: Mapping[str, Expr]
        A map which contains all expressions matched by the sub patterns in
        FusionPattern.annotation_patterns.

    var_usages: Mapping[Var, Sequence[Var]]
        A map mapping variable definitions to a set of uses.

    value_to_bound_var: Mapping[Expr, Var]
        Map from value to its bound variable.
    """

    annotated_expr: Mapping[str, Expr]
    var_usages: Mapping[Var, Sequence[Var]]
    value_to_bound_var: Mapping[Expr, Var]


@tvm._ffi.register_object("relax.transform.FusionPattern")
class FusionPattern(Object):
    """
    The pattern used by `FuseOpsByPattern`. It's mainly DFPattern but with other
    information to help during the fusion pass.

    Parameters
    ----------
    name: str
        The name of pattern. Usually it starts with the name of backend, like 'cutlass.matmul'.

    pattern: DFPattern
        The dataflow pattern that will be used to match expressions that can be handled
        by external backends.

    annotation_patterns: Mapping[str, DFPattern]
        The map which is used to extract important expressions from the pattern match
        result. All DFPattern in this map should be part of the `pattern`.

    check: Callable[[PatternCheckContext], bool]
        The function to check whether the match result is accepted.
    """

    name: str
    pattern: DFPattern
    annotation_patterns: Mapping[str, DFPattern]
    check: Callable[[PatternCheckContext], bool]

    def __init__(
        self,
        name: str,
        pattern: DFPattern,
        annotation_patterns: Optional[Mapping[str, DFPattern]] = None,
        check: Optional[Callable[[Mapping[str, Expr]], bool]] = None,
    ):
        if annotation_patterns is None:
            annotation_patterns = {}
        self.__init_handle_by_constructor__(
            _ffi_api.FusionPattern, name, pattern, annotation_patterns, check  # type: ignore
        )


def FuseOpsByPattern(
    patterns: List[Union[FusionPattern, Tuple]],
    bind_constants: bool = True,
    annotate_codegen: bool = False,
) -> tvm.ir.transform.Pass:
    """Apply pattern matching to each function in the given module, and group matched expressions
    into a new function.

    The end result is similar to FuseOps, but fusion is driven completely by the provided patterns.

    Parameters
    ----------
    patterns : List[Union[FusionPattern, Tuple]]
        A list of patterns to be matched. The order of the patterns determines the order of priority
        in which they are matched. Higher-priority patterns should come earlier in the list.

        In addition to FusionPattern, a tuple can be passed as item of this list. The pattern
        will be constructed through FusionPattern(*item)

    bind_constants : bool
        Whether or not to keep bound constants in the grouped function.

    annotate_codegen : bool
        If True, wrap each created composite function with another function, whose body consists
        only of a call to the composite function, and annotate the outer function with "Codegen"
        and "global_symbol" attributes. The "Codegen" attribute is set as the prefix of the
        corresponding pattern name. For example, "dnnl" if the pattern name is "dnnl.conv2d_relu".

        This must be True if the created composite functions are intended to be offloaded to
        an external backend without using the MergeCompositeFunctions pass.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for pattern-based fusion.

    """
    converted_patterns = []
    for pattern in patterns:
        if isinstance(pattern, tuple):
            converted_patterns.append(FusionPattern(*pattern))
        elif isinstance(pattern, FusionPattern):
            converted_patterns.append(pattern)
        else:
            raise ValueError(f"Invalid pattern: {pattern}")

    return _ffi_api.FuseOpsByPattern(
        converted_patterns,
        bind_constants,
        annotate_codegen,
    )  # type: ignore


def MergeCompositeFunctions() -> tvm.ir.transform.Pass:
    """Group one or multiple composite functions created by FuseOpsByPattern into a new function.
    The new function will be annotated with "Codegen" and "global_symbol" attributes, and it
    is intented to be offloaded to an external backend.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for merging composite functions.
    """
    return _ffi_api.MergeCompositeFunctions()  # type: ignore


def LiftTransformParams() -> tvm.ir.transform.Pass:
    """Lift transformation of the parameters of a function.

    When some inputs of the function is marked as 'parameters' (the model weights), this pass
    identifies the transformation of the parameters and lifts them to a separate function called
    `transform_params`. `transform_params` takes a tuple of the original parameters as input and
    returns a tuple of the transformed parameters. The original function will be rewritten to accept
    a tuple of transformed parameters as input.

    Users are expected to invoke the `transform_params` function in runtime and pass the transformed
    parameters to the original function as input.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for lifting transformation of parameters.
    """
    return _ffi_api.LiftTransformParams()  # type: ignore


def LegalizeOps(customize_legalize_map: Optional[Dict[str, LegalizeFunc]] = None):
    """Legalize high-level operator calls in Relax functions to call_tir
    with corresponding low-level TIR PrimFuncs.

    For each high-level operator, we register the way of legalizing it as a
    function, which takes a context BlockBuilder and the Call being legalized
    as input, and returns the legalized call. Here the input BlockBuilder is
    mainly used for adding the PrimFunc created by call_te into the context
    IRModule.

    The legalization function for each operator is registered as an attribute (with
    attribute key `FLegalize`) of the operator.

    This pass provides customizability for users to use their own legalization
    function for operators. The pass takes an optional customized map,
    with the key to be the operator name (`str`) and value to be the function
    (`LegalizeFunc`). The default legalization function will be overridden by the customized
    one.

    Parameters
    ----------
    customize_legalize_map : Optional[Dict[str, LegalizeFunc]]
        The customized operator legalization function map. The customized function will override
        the default one.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass

    Examples
    --------
    The following code shows how to use this pass:

    .. code-block:: python

        # Define the pass input IRModule
        @tvm.script.ir_module
        class Module:
            @R.function
            def main(
                x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
            ) -> R.Tensor((2, 3), "float32"):
                z: R.Tensor((2, 3), "float32") = R.add(x, y)
                r: R.Tensor((2, 3), "float32") = R.multiply(y, z)
                return r

        # Define the customized legalization function for "relax.add"
        def customize_legalize_add(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
            from tvm import topi
            return bb.call_te(topi.add, call.args[1], call.args[0])

        # Apply the pass with the customized function to the module.
        mod = LegalizeOps({"relax.add": customize_legalize_add})(Module)

    Print out the result by `mod.show()`, we can see the IRModule after
    legalization becomes

    .. code-block:: python

        @tvm.script.ir_module
        class Module:
            @R.function
            def main(
                x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
            ) -> R.Tensor((2, 3), "float32"):
                z = R.call_tir(add, (y, x), (2, 3), dtype="float32")
                r = R.call_tir(multiply, (y, z), (2, 3), dtype="float32")
                return r

            @T.prim_func
            def add(
                A: T.Buffer((2, 3), "float32"),
                B: T.Buffer((2, 3), "float32"),
                T_add: T.Buffer((2, 3), "float32"),
            ):
                T.func_attr({"tir.noalias": True})
                for ax0, ax1 in T.grid(2, 3):
                    with T.block("T_add"):
                        v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                        T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                        T.writes(T_add[v_ax0, v_ax1])
                        T_add[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[v_ax0, v_ax1]

            @T.prim_func
            def multiply(
                A: T.Buffer((2, 3), "float32"),
                B: T.Buffer((2, 3), "float32"),
                T_multiply: T.Buffer((2, 3), "float32"),
            ):
                T.func_attr({"tir.noalias": True})
                for ax0, ax1 in T.grid(2, 3):
                    with T.block("T_multiply"):
                        v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                        T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                        T.writes(T_multiply[v_ax0, v_ax1])
                        T_multiply[v_ax0, v_ax1] = A[v_ax0, v_ax1] * B[v_ax0, v_ax1]
    """

    return _ffi_api.LegalizeOps(customize_legalize_map)  # type: ignore


def MetaScheduleApplyDatabase(
    work_dir: Optional[str] = None,
) -> tvm.ir.transform.Pass:
    """Apply the best schedule from tuning database.
    work_dir : Optional[str]
       work directory to deduce default database if database is not provided
       (it will be ignored when an user passes database)
    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass
    """
    return _ffi_api.MetaScheduleApplyDatabase(work_dir)  # type: ignore


def MetaScheduleTuneTIR(
    work_dir: str,
    max_trials_global: int,
) -> tvm.ir.transform.Pass:
    """Tune TIR with MetaSchedule.
    Parameters
    ----------
    work_dir: str
       work directory
    max_trials_gloabl: int
       maximum number of total trials allowed for tuning
    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.MetaScheduleTuneTIR(work_dir, max_trials_global)  # type: ignore


def MetaScheduleTuneIRMod(
    params: Dict[str, NDArray],
    work_dir: str,
    max_trials_global: int,
) -> tvm.ir.transform.Pass:
    """Tune Relax IRModule with MetaSchedule.
    Parameters
    ----------
    params: Dict[str, NDArray]
       model params
    work_dir: str
       work directory
    max_trials_gloabl: int
       maximum number of total trials allowed for tuning
    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.MetaScheduleTuneIRMod(params, work_dir, max_trials_global)  # type: ignore


def SimplifyNormInference() -> tvm.ir.transform.Pass:
    """Simplify normalization operators during inference. For example, the result
    of a batch norm which is indexed at tuple index 0 will be unpacked into a
    number of simplified operators.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass
    """

    return _ffi_api.SimplifyNormInference()  # type: ignore


def AlterOpImpl(
    op_impl_map: Dict[str, PrimFunc],
    op_buffer_transforms: Dict[str, List[Union[IndexMap, Callable]]],
):
    """Replace all PrimFunc's which have matching 'operator_name' attribute, with replacement
    PrimFunc that could possibly have different layouts on i/o buffers. The layout
    transformations on i/o buffers is present in the op_buffer_transforms map. Inserts the layout
    transformations in the call sites of PrimFuncs being replaced to transform i/o
    tensors into expected layout by new PrimFunc.

    Parameters
    ----------
    op_impl_map: Dict[str, PrimFunc]
        op_kind to PrimFunc map
    op_buffer_transforms: Dict[str, List[Union[IndexMap, Callable]]
        op_kind to layout transformation map for each of the buffers
    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    for operator_name, transform_list in op_buffer_transforms.items():
        l = []
        for transform in transform_list:
            if isinstance(transform, Callable):
                transform = IndexMap.from_func(transform)
            l.append(transform)
        op_buffer_transforms[operator_name] = l

    return _ffi_api.AlterOpImpl(op_impl_map, op_buffer_transforms)  # type: ignore


def ConvertLayout(desired_layouts: Dict[str, List[str]]) -> tvm.ir.transform.Pass:
    """Automatic layout conversion pass.
    Parameters
    ----------
    desired_layouts : Dict[str, List[str]]
        The desired layout of conv2d ops is a map from the name of the op to the desired layout
        of the desired feature map, weight and output. For example, if we want to convert the
        layout of conv2d from NCHW to NHWC, we can set the desired layout of conv2d to be
        {"conv2d": ["NHWC", "OHWI"]}.
    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for layout conversion.
    """
    return _ffi_api.ConvertLayout(desired_layouts)  # type: ignore


def DeadCodeElimination(entry_functions: Optional[List[str]] = None) -> tvm.ir.transform.Pass:
    """Remove dead code in the IRModule.
       Currently it removes:
       1. Unused local VarBindings in a DataflowBlock.
       2. Unused DataflowBlocks in a function.
       3. Unused Relax functions in the module.
          We detect the call chain from the entry function, and remove all unused functions.

    Parameters
    ----------
    entry_functions: Optional[List[str]]
        The set of entry functions to start from.

    Notes
    -----
    For function-wise DCE, use py:func:`tvm.relax.analysis.remove_all_unused`.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass.
    """
    if entry_functions is None:
        entry_functions = ["main"]
    return _ffi_api.DeadCodeElimination(entry_functions)  # type: ignore


def ToMixedPrecision(out_dtype="float32") -> tvm.ir.transform.Pass:
    """Automatic mixed precision pass. Currently the pass assumes the input module to be fp32
    only, and will automatically cast fp32 to fp16 for certain ops.
    Parameters
    ----------
    out_dtype : str
        The output data type of gemm/conv, which is the data type of the accumulator.
    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for mixed precision.
    """
    return _ffi_api.ToMixedPrecision(out_dtype)  # type: ignore


def _wrap_class_function_pass(pass_cls, pass_info):
    """Wrap a python class as function pass."""

    class PyFunctionPass(FunctionPass):
        """Internal wrapper class to create a class instance."""

        def __init__(self, *args, **kwargs):
            # initialize handle in case pass_cls creation failed.
            self.handle = None
            inst = pass_cls(*args, **kwargs)

            # it is important not to capture self to
            # avoid a cyclic dependency
            def _pass_func(func, mod, ctx):
                return inst.transform_function(func, mod, ctx)

            self.__init_handle_by_constructor__(
                _ffi_api.MakeFunctionPass, _pass_func, pass_info  # type: ignore
            )
            self._inst = inst

        def __getattr__(self, name):
            # fall back to instance attribute if there is not any
            return self._inst.__getattribute__(name)

    functools.update_wrapper(PyFunctionPass.__init__, pass_cls.__init__)
    PyFunctionPass.__name__ = pass_cls.__name__
    PyFunctionPass.__doc__ = pass_cls.__doc__
    PyFunctionPass.__module__ = pass_cls.__module__
    return PyFunctionPass


def function_pass(
    pass_func=None,
    opt_level=None,
    name=None,
    required=None,
    traceable=False,
) -> Union[Callable, FunctionPass]:
    """Decorate a function pass.

    This function returns a callback when pass_func
    is provided. Otherwise, it returns the created function pass using the
    given optimization function.

    Parameters
    ----------
    pass_func : Optional[Callable[(Function, Module, PassContext) -> Function]]
        The transformation function or class.

    opt_level : int
        The optimization level of this function pass.

    name : Optional[str]
        The name of the function pass. The name could be empty. In this case, the
        name of the optimization function will be used as the pass name.

    required : Optional[List[str]]
        The list of passes that the function pass is dependent on.

    traceable: Boolean
        Boolean variable whether the function pass is traceable

    Returns
    -------
    create_function_pass : Union[Callable, FunctionPass]

        A decorator will be returned if pass_func is not provided,
        otherwise return the decorated result.
        The returned decorator has two behaviors depending on the input:
        A new FunctionPass will be returned when we decorate a pass function.
        A new FunctionPass class will be returned when we decorate a class type.

    Examples
    --------
    The following code block decorates a function pass class.

    .. code-block:: python

        @relax.transform.function_pass(opt_level=1)
        class TestReplaceFunc:
            def __init__(self, new_func):
                self.new_func = new_func

            def transform_function(self, func, mod, ctx):
                # just for demo purposes
                # transform func to new_func
                return self.new_func

        @R.function
        def f1(x: Tensor[(m, n), "float32"]):
            return x

        @tvm.script.ir_module
        class InputMod:
            @R.function
            def f2(x: Tensor[(m, n), "float32"]):
                gv0 = relax.add(x, x)
                return gv0
        # fpass is now a special pass that replaces every
        # function to f1
        fpass = TestReplaceFunc(f1)
        # now every function in InputMod is replaced by f1
        updated_mod = fpass(InputMod)


    The following code creates a function pass by decorating
    a user defined transform function.

    .. code-block:: python

        @relax.transform.function_pass(opt_level=2)
        def transform(func, mod, ctx):
            # my transformations here.
            return func

        function_pass = transform
        assert isinstance(function_pass, relax.transform.FunctionPass)
        assert function_pass.info.opt_level == 2

        # Given a module m, the optimization could be invoked as the follwoing:
        updated_mod = function_pass(m)
        # Now transform should have been applied to every function in
        # the provided module m. And the updated module will be returned.
    """

    if opt_level is None:
        raise ValueError("Please provide opt_level for the function pass.")

    required = required if required else []
    if not isinstance(required, (list, tuple)):
        raise TypeError("Required is expected to be the type of " + "list/tuple.")

    def create_function_pass(pass_arg):
        """Internal function that creates a function pass"""
        fname = name if name else pass_arg.__name__
        info = tvm.transform.PassInfo(opt_level, fname, required, traceable)
        if inspect.isclass(pass_arg):
            return _wrap_class_function_pass(pass_arg, info)
        if not isinstance(pass_arg, (types.FunctionType, types.LambdaType)):
            raise TypeError("pass_func must be a callable for Function pass")
        return _ffi_api.MakeFunctionPass(pass_arg, info)  # type: ignore

    if pass_func:
        return create_function_pass(pass_func)
    return create_function_pass


def _wrap_class_dataflowblock_pass(pass_cls, pass_info):
    """Wrap a python class as dataflowblock pass"""

    class PyDataflowBlockPass(DataflowBlockPass):
        """Internal wrapper class to create a class instance."""

        def __init__(self, *args, **kwargs):
            # initialize handle in case pass_cls creation failed.
            self.handle = None
            inst = pass_cls(*args, **kwargs)

            # it is important not to capture self to
            # avoid a cyclic dependency
            def _pass_func(func, mod, ctx):
                return inst.transform_dataflowblock(func, mod, ctx)

            self.__init_handle_by_constructor__(
                _ffi_api.MakeDataflowBlockPass, _pass_func, pass_info  # type: ignore
            )
            self._inst = inst

        def __getattr__(self, name):
            # fall back to instance attribute if there is not any
            return self._inst.__getattribute__(name)

    functools.update_wrapper(PyDataflowBlockPass.__init__, pass_cls.__init__)
    PyDataflowBlockPass.__name__ = pass_cls.__name__
    PyDataflowBlockPass.__doc__ = pass_cls.__doc__
    PyDataflowBlockPass.__module__ = pass_cls.__module__
    return PyDataflowBlockPass


def dataflowblock_pass(
    pass_func=None, opt_level=None, name=None, required=None, traceable=False
) -> Union[Callable, DataflowBlockPass]:
    """Decorate a dataflowblock pass.

    This function returns a callback when pass_func
    is provided. Otherwise, it returns the created dataflowblock pass using the
    given optimization function.

    Parameters
    ----------
    pass_func : Optional[Callable[(DataflowBlock, Module, PassContext) -> DataflowBlock]]
        The transformation function or class.

    opt_level : int
        The optimization level of this dataflowblock pass.

    name : Optional[str]
        The name of the dataflowblock pass. The name could be empty. In this case, the
        name of the optimization function will be used as the pass name.

    required : Optional[List[str]]
        The list of passes that the dataflowblock pass is dependent on.

    traceable: Boolean
        Boolean variable whether the dataflowblock pass is traceable

    Returns
    -------
    create_dataflowblock_pass : Union[Callable, DataflowBlockPass]

        A decorator will be returned if pass_func is not provided,
        otherwise return the decorated result.
        The returned decorator has two behaviors depending on the input:
        A new DataflowBlockPass will be returned when we decorate a pass function.
        A new DataflowBlockPass class will be returned when we decorate a class type.

    Examples
    --------
    The following code block decorates a dataflowblock pass class.

    .. code-block:: python

        @relax.transform.dataflowblock_pass(opt_level=1)
        class TestReplaceBinding:
            # Simple test function to replace the first VarBinding to another.

            def __init__(self):
                # create a new VarBinding
                m, n = tir.Var("m", "int64"), tir.Var("n", "int64")
                lv0 = relax.Var("lv1", relax.TensorStructInfo([m, n], "float32"))
                val = relax.const(np.random.rand(24, 56))
                self.new_binding = relax.VarBinding(lv0, val)

            def transform_dataflowblock(self, block, mod, ctx):
                # just for demo purposes
                # Replace the first binding in the DataflowBlock
                new_bindings = [self.new_binding, block.bindings[1]]
                new_block = relax.expr.DataflowBlock(new_bindings, block.span)
                return new_block

        @tvm.script.ir_module
        class InputMod:
            @R.function
            def f1(x: Tensor[(m, n), "float32"]):
                with relax.dataflow():
                    lv0 = relax.multiply(x, x)
                    gv0 = relax.add(x, x)
                    relax.output(gv0)
                return gv0
        # block_pass is now a special pass that replaces every
        # first binding to the constant value binding
        block_pass = TestReplaceBinding()
        # now every first binding in DataflowBlock of InputMod
        # is replaced by new_binding
        updated_mod = block_pass(InputMod)


    The following code creates a dataflowblock pass by decorating
    a user defined transform function.

    .. code-block:: python

        @relax.transform.dataflowblock_pass(opt_level=2)
        def transform(block, mod, ctx):
            # my transformations here.
            return block

        block_pass = transform
        assert isinstance(block_pass, relax.transform.DataflowBlockPass)
        assert block_pass.info.opt_level == 2

        # Given a module m, the optimization could be invoked as the follwoing:
        updated_mod = block_pass(m)
        # Now transform should have been applied to every DataflowBlock in
        # the provided module m. And the updated module will be returned.
    """

    if opt_level is None:
        raise ValueError("Please provide opt_level for the dataflowblock pass.")

    required = required if required else []
    if not isinstance(required, (list, tuple)):
        raise TypeError("Required is expected to be the type of " + "list/tuple.")

    def create_dataflowblock_pass(pass_arg):
        """Internal function that creates a dataflowblock pass"""
        fname = name if name else pass_arg.__name__
        info = tvm.transform.PassInfo(opt_level, fname, required, traceable)
        if inspect.isclass(pass_arg):
            return _wrap_class_dataflowblock_pass(pass_arg, info)
        if not isinstance(pass_arg, (types.FunctionType, types.LambdaType)):
            raise TypeError("pass_func must be a callable for DataflowBlock pass")
        return _ffi_api.MakeDataflowBlockPass(pass_arg, info)  # type: ignore

    if pass_func:
        return create_dataflowblock_pass(pass_func)
    return create_dataflowblock_pass
