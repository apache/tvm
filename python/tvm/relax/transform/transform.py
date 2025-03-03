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
import warnings
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np  # type: ignore

import tvm.ir
from tvm.ir.container import Array
from tvm.relax import Expr, Var, StructInfo
from tvm.relax.dpl import DFPattern
from tvm.runtime import NDArray, Object
from tvm.tir import IndexMap, PrimFunc

from . import _ffi_api
from .legalize_ops.common import LegalizeFunc
from ..expr import Var


@tvm._ffi.register_object("relax.FunctionPass")
class FunctionPass(tvm.ir.transform.Pass):
    """A pass that works on each tvm.relax.Function in a module. A function
    pass class should be created through `function_pass`.
    """


@tvm._ffi.register_object("relax.DataflowBlockPass")
class DataflowBlockPass(tvm.ir.transform.Pass):
    """A pass that works on each tvm.relax.DataflowBlock in a module."""


def Gradient(
    func_name: str, require_grads: Optional[Union[Var, List[Var]]] = None, target_index: int = 0
) -> tvm.ir.transform.Pass:
    """Reverse-mode automatic differentiation.

    This pass will differentiate one function in the IRModule. Now the input function must have only
    one dataflow block (ConvertToDataflow may need to be called first).

    For a given function specified by `func_name`, it generates a new function with the name
    `func_name + "_adjoint"`. The new function computes the gradient of the **differentiation
    target** with respect to the arguments specified by `require_grads` of the original function.

    If the function has only one return value, the return value will be specified as target. If the
    function has more than one return values, the target will be specified as the target_index-th
    return value. The target must be a scalar (0-dim tensor).

    The new function will be like:

    .. code-block:: python

        @R.function
        def main_adjoint(original_parameters):
            with R.dataflow():
                # the bindings of the original function
                ...
                # calculating the gradients
                ...
                R.output(original_outputs, grad_1, grad_2, ...)
            return (original_return_value, (grad_1, grad_2, ...))

    This AD pass also supports checkpointing as described in
    "Training deep nets with sublinear memory cost." - Chen, Tianqi, et al. (2016).
    See tvm.relax.testing.nn.checkpoint for more details.

    Parameters
    ----------
    func_name : str
        The name of the specific function.

    require_grads : Optional[Union[relax.Var, List[relax.Var]]]
        The relax variables whose adjoints is needed. Must be parameters of the given function and
        should not be duplicate. If it is not specified, adjoints of all parameters would be
        computed.

    target_index : int
        If the specified function has more than one return values, specify the index of the return
        value as the target. If it is not specified, the first return value will be the target.

    Returns
    -------
    ret : tvm.ir.transform.Pass
        The Pass.

    Examples
    --------
    The following code shows how to use this pass:

    .. code-block:: python

        @I.ir_module
        class Module:
            @R.function
            def main(
                x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
            ) -> R.Tensor((), dtype="float32"):
                with R.dataflow():
                    lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                    # use R.sum to reduce the tensor to a scalar
                    lv2: R.Tensor((), dtype="float32") = R.sum(lv1, axis=None, keepdims=False)
                    R.output(lv2)
                return lv2

        After = relax.transform.Gradient("main")(Module)

    The module after the Gradient pass will be:

    .. code-block:: python

        @I.ir_module
        class After:
            @R.function
            def main(
                x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
            ) -> R.Tensor((), dtype="float32"):
                with R.dataflow():
                    lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                    lv2: R.Tensor((), dtype="float32") = R.sum(lv1, axis=None, keepdims=False)
                    R.output(lv2)
                return lv2

            @R.function
            def main_adjoint(
                x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
            ) -> R.Tuple(
                R.Tensor((), dtype="float32"),
                R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")),
            ):
                with R.dataflow():
                    # original bindings
                    lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                    lv2: R.Tensor((), dtype="float32") = R.sum(lv1, axis=None, keepdims=False)
                    # bindings w.r.t. intermediate variables
                    lv2_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                    lv1_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(
                        lv2_adjoint, (3, 3)
                    )
                    # bindings w.r.t. parameters
                    x_adjoint: R.Tensor((3, 3), dtype="float32") = lv1_adjoint
                    y_adjoint: R.Tensor((3, 3), dtype="float32") = lv1_adjoint
                    R.output(lv2, x_adjoint, y_adjoint)
                # return value: (orig_return_values, tuple(adjoints))
                return (lv2, (x_adjoint, y_adjoint))

    The second example is returning multiple values and specifying the target with `target_index`:

    .. code-block:: python

        @I.ir_module
        class Module:
            @R.function
            def main(
                x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
            ) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32")):
                with R.dataflow():
                    lv1: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
                    lv2: R.Tensor((), dtype="float32") = R.sum(y, axis=None, keepdims=False)
                    R.output(lv1, lv2)
                return (lv1, lv2)

        After = relax.transform.Gradient("main", target_index=1)(Module)

    The module after the Gradient pass will be:

    .. code-block:: python

        @I.ir_module
        class Module:
            @R.function
            def main(
                x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
            ) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32")):
                with R.dataflow():
                    lv1: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
                    lv2: R.Tensor((), dtype="float32") = R.sum(y, axis=None, keepdims=False)
                    R.output(lv1, lv2)
                return (lv1, lv2)

            @R.function
            def main_adjoint(
                x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
            ) -> R.Tuple(
                R.Tuple(R.Tensor((), dtype="float32"), R.Tensor((), dtype="float32")),
                R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")),
            ):
                with R.dataflow():
                    # original bindings
                    lv1: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
                    lv2: R.Tensor((), dtype="float32") = R.sum(y, axis=None, keepdims=False)
                    # bindings w.r.t. intermediate variables
                    # gradient of intermediate variables that is not related to the target will not
                    # be calculated
                    lv2_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                    # bindings w.r.t. parameters
                    x_adjoint: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                    y_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(
                        lv2_adjoint, (3, 3)
                    )
                    R.output(lv1, lv2, x_adjoint, y_adjoint)
                # return value: (orig_return_values, tuple(adjoints))
                return ((lv1, lv2), (x_adjoint, y_adjoint))
    """
    if require_grads is not None and not isinstance(require_grads, list):
        require_grads = [require_grads]

    return _ffi_api.Gradient(func_name, require_grads, target_index)  # type: ignore


def ToNonDataflow() -> tvm.ir.transform.Pass:
    """Transform all dataflow structure to non-dataflow version.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.ToNonDataflow()  # type: ignore


def TopologicalSort(order="depth-first", direction="from-inputs") -> tvm.ir.transform.Pass:
    """Sort bindings in relax.Dataflow blocks in the order specified

    Parameters
    ----------
    order: str

        The order in which bindings should be emitted.  Allowed values
        are "depth-first" and "breadth-first".

    direciton: str

        The direction in which the sort should be performed.  Allowed
        values are "from-inputs" and "from-outputs".

    Returns
    -------
    ret: tvm.ir.transform.Pass

    """
    return _ffi_api.TopologicalSort(order, direction)  # type: ignore


def RemovePurityChecking() -> tvm.ir.transform.Pass:
    """Activate relax.force_pure on all pure functions in the module
    and unwrap all pure override ops into the normal versions.

    This effectively means that there will be no more purity tracking,
    useful for low-level code generation.

    Returns
    -------
    ret: tvm.ir.transform.Pass
        The Pass.

    Note
    ----
    Should be used after ToNonDataflow()
    """
    return _ffi_api.RemovePurityChecking()  # type: ignore


def DataflowUseInplaceCalls() -> tvm.ir.transform.Pass:
    """
    Pass that changes calls to operators that can be done in-place
    (generally, these are elementwise operations) into in-place implementations.
    Supported operators will be replaced by calls to `call_tir_inplace` that invoke
    in-place PrimFunc implementations of those operators (which are based on the legalizations of
    those operators).

    Note: ConvertToDataflow may need to be called first to provide dataflow blocks.

    Returns
    -------
    ret: tvm.ir.transform.Pass
        The pass
    """
    return _ffi_api.DataflowUseInplaceCalls()


def LambdaLift() -> tvm.ir.transform.Pass:
    """A pass that lifts local functions into global.

    Returns
    -------
    ret : tvm.ir.transform.Pass
    """
    return _ffi_api.LambdaLift()


def LazyGetInput() -> tvm.ir.transform.Pass:
    """A pass that requests inputs lazily.

    In many cases, the size of the model weights exceeds the available
    memory on a GPU.  In these cases, a function that accepts all
    model weights as arguments would not be able to be called.  In
    these cases, parameters must be loaded as they are required by the
    function, and unloaded once they are no longer needed.

    This pass mutates a function such that all model weights
    (arguments after the first `func.attrs["num_input"]` arguments)
    are loaded on demand.  Rather than accepting the weights as
    function arguments, the function accepts a callback argument,
    which can load each parameter as needed.  The callback accepts two
    arguments, first the index of the model weight, and second the
    name of the parameter.  The callback should return the parameter
    as specified.

    .. code-block:: python

        @R.function
        def before(A: R.Tensor([16,32],"float32")):
            ...

        @R.function
        def after(fget_param: R.Callable([R.Prim('int64'), R.Object], R.Object)):
            A_untyped = fget_param(0, R.str('A'))
            A = R.match_cast(A_untyped, R.Tensor([16,32], "float32")
            ...

    Returns
    -------
    ret : tvm.ir.transform.Pass

    """
    return _ffi_api.LazyGetInput()


def LazySetOutput() -> tvm.ir.transform.Pass:
    """A pass that sets function outputs when available

    In many cases, the size of the model weights exceeds the available
    memory on a GPU.  In these cases, a function that produces all
    model weights as a single return value would not be able to be
    called.  In these cases, parameters must be returned as they are
    produced, unloaded from the GPU (or saved to disk), before
    producing additional outputs.

    This pass mutates a function such that all outputs from a function
    are returned when they are available.  The function accepts an
    additional callback argument, which is called with each output of
    the function.  The callback accepts two arguments, first the index
    of the output tuple that was produced (or zero if the output is
    not a tuple), and second the value itself.

    .. code-block:: python

        @R.function
        def before(args):
            ...
            return (A, B)

        @R.function
        def after(args, fset_param: R.Callable([R.Prim('int64'), R.Object])):
            ...
            fset_param(0, A)
            ...
            fset_param(1, B)
            ...
            return ()


    Returns
    -------
    ret : tvm.ir.transform.Pass

    """
    return _ffi_api.LazySetOutput()


def ConvertToDataflow(min_size: int = 2) -> tvm.ir.transform.Pass:
    """A pass that converts consecutive dataflow operations
    inside binding blocks into dataflow blocks.

    Note: ConvertToDataflow may need to be called first.

    Parameters
    ----------
    min_size: int
        The minimum number of consecutive dataflow bindings
        the pass needs to extract a new block.

    Returns
    -------
    ret: tvm.ir.transform.Pass
        The pass.
    """
    return _ffi_api.ConvertToDataflow(min_size)


def CallTIRRewrite() -> tvm.ir.transform.Pass:
    """Perform explicit tensor allocation for call_tir and call_dps_packed.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.CallTIRRewrite()  # type: ignore


def Normalize() -> tvm.ir.transform.Pass:
    """Transforming Relax IR to normal form, i.e., the expressions are normalized(no nesting
    and hence the AST is in ANF), and all ``checked_type_`` and ``shape_`` of expressions are
    available.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.Normalize()  # type: ignore


def NormalizeGlobalVar() -> tvm.ir.transform.Pass:
    """Possibly rename the GlobalVar in an IRModule to ensure these properties:

    1. (Invariant) First ensure every public function has the same name as its "global_symbol"
    attribute
    2. To ensure 1., we may need to rename private functions with conflicting names;
    3. Finally, the name of every GlobalVar is unique in the IRModule.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.NormalizeGlobalVar()  # type: ignore


def CanonicalizeBindings() -> tvm.ir.transform.Pass:
    """
    Canonicalizes variable definitions
    (e.g., if there is y = x and z = y, it replaces uses of y and z with x).
    Also simplifies match cast nodes (eliminating redundant checks)
    and tuple indices.

    Best combined with constant folding and the elimination of unused definitions.

    Note: If a dataflow var is used only in a binding to the dataflow block
    output var (i.e., a non-dataflow var), this pass will also remove the dataflow var
    and replaces the output var's binding with the dataflow var's direct definition.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.CanonicalizeBindings()  # type: ignore


def EliminateCommonSubexpr(call_only=False) -> FunctionPass:
    """Eliminate common subexpressions within functions.

    Note: For nested functions, this pass performs CSE *within* those functions

    Parameters
    ----------
    call_only : bool
        If True, enable eliminating only call nodes.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass that eliminates common subexpressions.
    """
    return _ffi_api.EliminateCommonSubexpr(call_only)  # type: ignore


def UpdateVDevice(new_vdevice: tvm.ir.VDevice, index: int) -> tvm.ir.transform.Pass:
    """Update virtual device.

    Parameters
    ----------
    new_vdevice : tvm.ir.VDevice
        The new virtual device.
    index : int
        The device index indicates the device on which the update will be performed.

    Returns
    -------
    ret : tvm.ir.transform.Pass
        The registered pass that modifies the virtual device.
    """
    return _ffi_api.UpdateVDevice(new_vdevice, index)  # type: ignore


def RewriteDataflowReshape() -> tvm.ir.transform.Pass:
    """Convert all reshape-like call_tir to VM reshape operator call.
    The VM reshape operator calls will be further lowered to a CreateView
    operation at runtime, instead of doing real data copy.
    Here "reshape-like" includes reshape, expand_dims, flatten, etc.

    Note: Operates only in dataflow blocks. ConvertToDataflow may need to be called first.

    Returns
    -------
    ret : tvm.ir.transform.Pass
    """
    return _ffi_api.RewriteDataflowReshape()  # type: ignore


def StaticPlanBlockMemory() -> tvm.ir.transform.Pass:
    """The static memory planning pass on BindingBlock level.
    The pass will reuse allocated memory to its best effort, in order to
    reduce the total amount of allocated memory size.

    The pass "supports" dynamic shape in the way of TIR variable upper bound
    annotation. We can optionally annotate the attribute "tir_var_upper_bound"
    to Relax functions. The attribute value is a dict from strings to integers,
    denoting the name of TIR variables to the upper bound values of the TIR vars.
    Note: The annotated upper bound attribute only applies to TIR vars in the
    function signature for clarity.

    For example, we can annotate a Relax function with
    :code:`R.func_attr({"tir_var_upper_bound": {"n": 1024}})`.
    It means the maximum value of variable that names "n" in the function
    signature will have upper bound 1024. And we will use 1024 as its value
    during memory planning.

    Returns
    -------
    ret : tvm.ir.transform.Pass
    """
    return _ffi_api.StaticPlanBlockMemory()  # type: ignore


def LowerAllocTensor() -> tvm.ir.transform.Pass:
    """Lower remaining instances of R.builtin.alloc_tensor

    The static memory planner removes static instances of
    `R.builtin.alloc_tensor`, replacing with `R.memory.alloc_storage`
    and `R.memory.alloc_tensor`.  However, `R.builtin.alloc_tensor`
    still remains for any dynamic allocations.

    This transform replaces any remaining `R.builtin.alloc_tensor`
    instances with `R.memory.alloc_storage` and
    `R.memory.alloc_tensor`.  If no `R.builtin.alloc_tensor` are
    present, this pass has no effect.

    Returns
    -------
    ret : tvm.ir.transform.Pass
    """
    return _ffi_api.LowerAllocTensor()  # type: ignore


def KillAfterLastUse() -> tvm.ir.transform.Pass:
    """Drop all tensor/storage objects after last use

    Returns
    -------
    ret : tvm.ir.transform.Pass
    """
    return _ffi_api.KillAfterLastUse()  # type: ignore


def ComputePrimValue() -> tvm.ir.transform.Pass:
    """Compute all R.prim_value instances

    While high-level relax can include expressions in terms of its
    symbolic variables, these expressions cannot natively be computed
    within relax.  In order to provide values for symbolic expressions
    (e.g. `R.prim_value(N*N)`, where `N` is a symbolic variable), this
    pass generates a PrimFunc in which the expression can be computed.
    The relax graph is then updated to include a call to that
    PrimFunc, in place of the original `R.prim_value(expr)`.

    Returns
    -------
    ret : tvm.ir.transform.Pass

    """
    return _ffi_api.ComputePrimValue()  # type: ignore


def LowerRuntimeBuiltin() -> tvm.ir.transform.Pass:
    """Lowering generic intrinsic to VM intrinsics.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.LowerRuntimeBuiltin()  # type: ignore


def VMBuiltinLower() -> tvm.ir.transform.Pass:
    """Lowering generic intrinsic to VM intrinsics.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    warnings.warn(
        "tvm.relax.transform.VMBuiltinLower has been renamed to 'LowerRuntimeBuiltin'.  "
        "This wrapper is for backwards compatibility, and will be removed in a later update."
    )
    return _ffi_api.LowerRuntimeBuiltin()  # type: ignore


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
    params: Dict[Union[str, Var], Union[tvm.runtime.NDArray, np.ndarray]],
) -> tvm.ir.transform.Pass:
    """Bind params of function of the module to constant tensors.

    Parameters
    ----------
    func_name: str
        The function name to be bound

    params: Dict[Union[str,relax.Var], Union[tvm.runtime.NDArray, np.ndarray]]
        The map from parameter or parameter name to constant tensors.

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


def BindSymbolicVars(
    binding_map: Mapping[Union[str, tvm.tir.Var], tvm.tir.PrimExpr],
    func_name: Optional[str] = None,
) -> tvm.ir.transform.Pass:
    """Bind params of function of the module to constant tensors.

    Parameters
    ----------
    binding_map : Mapping[Union[str, tvm.tir.Var], tvm.tir.PrimExpr]
        The map from symbolic varname to integer.

    func_name : Optional[str]
        The function name to be bound. If None (default), all
        functions within the module will be updated.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    # Relax uses int64 for symbolic variables, but the FFI
    # converts python integers into int32.
    binding_map = {
        key: tvm.tir.const(value, "int64") if isinstance(value, int) else value
        for key, value in binding_map.items()
    }
    return _ffi_api.BindSymbolicVars(binding_map, func_name)  # type: ignore


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
        entry_functions = []

    # enable cutlass byoc registries
    # pylint: disable=unused-import,import-outside-toplevel
    from tvm.contrib import cutlass as _cutlass

    return _ffi_api.RunCodegen(target_options, entry_functions)  # type: ignore


def FoldConstant() -> tvm.ir.transform.Pass:
    """Fold constant expressions within dataflow blocks.

    Note: ConvertToDataflow may need to be called first to provide dataflow blocks.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.FoldConstant()  # type: ignore


def ExpandTupleArguments() -> tvm.ir.transform.Pass:
    """Expand tuple arguments to internal functions

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.ExpandTupleArguments()  # type: ignore


def RemoveUnusedParameters() -> tvm.ir.transform.Pass:
    """Remove unused arguments to internal functions

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.RemoveUnusedParameters()  # type: ignore


def RemoveUnusedOutputs() -> tvm.ir.transform.Pass:
    """Remove unused outputs from internal functions

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.RemoveUnusedOutputs()  # type: ignore


def InlinePrivateFunctions() -> tvm.ir.transform.Pass:
    """Inline all private relax functions

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.InlinePrivateFunctions()  # type: ignore


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

    Note: ConvertToDataflow may need to be called first to provide dataflow blocks.

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
    matched_expr: Expr
        The expression that's matched with the FusionPattern.pattern.

    annotated_expr: Mapping[str, Expr]
        A map which contains all expressions matched by the sub patterns in
        FusionPattern.annotation_patterns.

    matched_bindings: Mapping[Var, Expr]
        Map from variable to its value. It contains variables from bindings that is
        being fused by FuseOpsByPattern.

    var_usages: Mapping[Var, Sequence[Var]]
        A map mapping variable definitions to a set of uses. It has all variables
        used in the function.

    value_to_bound_var: Mapping[Expr, Var]
        Map from value to its bound variable. It doesn't have variables after the
        matched expression.
    """

    matched_expr: Expr
    annotated_expr: Mapping[str, Expr]
    matched_bindings: Mapping[Var, Expr]
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
    attrs_getter: Callable[[Dict[str, Expr]], Dict[str, str]]

    def __init__(
        self,
        name: str,
        pattern: DFPattern,
        annotation_patterns: Optional[Mapping[str, DFPattern]] = None,
        check: Optional[Callable[[PatternCheckContext], bool]] = None,
        attrs_getter: Optional[Callable[[Dict[str, Expr]], Dict[str, str]]] = None,
    ):
        if annotation_patterns is None:
            annotation_patterns = {}
        self.__init_handle_by_constructor__(
            _ffi_api.FusionPattern, name, pattern, annotation_patterns, check, attrs_getter
        )  # type: ignore


def FuseOpsByPattern(
    patterns: List[Union[FusionPattern, Tuple]],
    bind_constants: bool = True,
    annotate_codegen: bool = False,
    entry_functions: Optional[List[str]] = None,
) -> tvm.ir.transform.Pass:
    """Apply pattern matching to each function in the given module, and group matched expressions
    into a new function.

    The end result is similar to FuseOps, but fusion is driven completely by the provided patterns.

    Note: Only operates within dataflow blocks. ConvertToDataflow may need to be called first.

    Parameters
    ----------
    patterns : List[Union[FusionPattern, Tuple]]
        A list of patterns to be matched. The order of the patterns determines the order of priority
        in which they are matched. Higher-priority patterns should come earlier in the list.

        In addition to FusionPattern, a tuple can be passed as item of this list. The pattern
        will be constructed through :code:`FusionPattern(*item)`

    bind_constants : bool
        Whether or not to keep bound constants in the grouped function.

    annotate_codegen : bool
        If True, wrap each created composite function with another function, whose body consists
        only of a call to the composite function, and annotate the outer function with "Codegen"
        and "global_symbol" attributes. The "Codegen" attribute is set as the prefix of the
        corresponding pattern name. For example, "dnnl" if the pattern name is "dnnl.conv2d_relu".

        This must be True if the created composite functions are intended to be offloaded to
        an external backend without using the MergeCompositeFunctions pass.

    entry_functions : Optional[List[str]]
        The set of entry functions to start from.

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
        entry_functions or [],
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


def AttachAttrLayoutFreeBuffers() -> tvm.ir.transform.Pass:
    """Attach layout free buffers to the tir::PrimFunc.

    This pass is used to attach layout free buffers to the tir::PrimFunc according to
    the function usage in the relax function. Currently, the layout free buffers are the model
    weights and relax constants.

    Note that we recommend applying CanonicalizeBindings before this pass.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for attaching layout free buffers.
    """
    return _ffi_api.AttachAttrLayoutFreeBuffers()  # type: ignore


def SplitLayoutRewritePreproc() -> tvm.ir.transform.Pass:
    """Split the TIR layout rewrite into multiple TIR functions.
    This pass is used in the prepack weight after meta_schedule tuning.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for splitting TIR layout rewrite.
    """
    return _ffi_api.SplitLayoutRewritePreproc()  # type: ignore


def LiftTransformParams(shared_transform: Union[bool, List[str]] = False) -> tvm.ir.transform.Pass:
    """Lift transformation of the parameters of a function.

    When some inputs of the function is marked as 'parameters' (the model weights), this pass
    identifies the transformation of the parameters and lifts them to a separate function called
    `transform_params`. `transform_params` takes a tuple of the original parameters as input and
    returns a tuple of the transformed parameters. The original function will be rewritten to accept
    a tuple of transformed parameters as input.

    Users are expected to invoke the `transform_params` function in runtime and pass the transformed
    parameters to the original function as input.

    Parameters
    ----------
    shared_transform: Union[bool, List[str]]

        Indicates how the parameter transformation function will be produced

        - `False` (default): A separate parameter transformation function will be
          produced for each function with the `"num_input"` attribute.

        - `True`: A single parameter transformation function will be produced,
          containing the preprocessing steps common across all functions with
          the `"num_input"` attribute.

        - List[str]: A single parameter transformation function will be produced,
          containing the preprocessing steps common across each function whose
          name is in the list.  Passing a list of all functions with the `"num_input"`
          attribute or an empty list is equivalent to passing `True`.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for lifting transformation of parameters.
    """
    return _ffi_api.LiftTransformParams(shared_transform)  # type: ignore


def BundleModelParams(param_tuple_name: Optional[str] = None) -> tvm.ir.transform.Pass:
    """Bundle several model parameters into a single tuple paramters

    For each function, if the function has the attribute "num_input",
    separate between run-time parameters and compile-time weights.
    Run-time parameters (e.g. activations) are the first `num_input`
    parameters, and the remainder are compile-time weights.

    Parameters
    ----------
    param_tuple_name: Optional[str]

        The name of the tuple parameter. If unspecified, defaults to
        "model_params".

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for bundling model parameters.
    """
    return _ffi_api.BundleModelParams(param_tuple_name)  # type: ignore


def LegalizeOps(
    customize_legalize_map: Optional[Dict[str, LegalizeFunc]] = None, enable_warning: bool = False
):
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

    enable_warning : bool
        A boolean value indicating if to print warnings for CallNode whose op's
        legalization function is not registered. By default we don't print
        warnings.

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

    return _ffi_api.LegalizeOps(customize_legalize_map, enable_warning)  # type: ignore


def RealizeVDevice() -> tvm.ir.transform.Pass:
    """Propagate virtual device information.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass
    """

    return _ffi_api.RealizeVDevice()  # type: ignore


def MetaScheduleApplyDatabase(
    work_dir: Optional[str] = None, enable_warning: bool = False
) -> tvm.ir.transform.Pass:
    """Apply the best schedule from tuning database.

    Parameters
    ----------
    work_dir : Optional[str]
       work directory to deduce default database if database is not provided
       (it will be ignored when an user passes database)
    enable_warning : bool
        A boolean value indicating if to print warnings for TIR functions not
        showing up in the database. By default we don't print warning.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass
    """
    return _ffi_api.MetaScheduleApplyDatabase(work_dir, enable_warning)  # type: ignore


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
    max_trials_per_task: Optional[int] = None,
    op_names: Optional[List[str]] = None,
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
    max_trials_per_task: int
       maximum number of trials per task
    op_names: Optional[List[str]]
       A list of operator names to specify which op to tune. When it is None, all operators
       are tuned.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.MetaScheduleTuneIRMod(
        params, work_dir, max_trials_global, max_trials_per_task, op_names
    )  # type: ignore


def FewShotTuning(
    valid_count: int = 1,
    benchmark: bool = False,
) -> tvm.ir.transform.Pass:
    """The pass is designed for few shot tuning for static shape PrimFuncs. It examines all the
    blocks within the PrimFunc and conducts loop fusion, splitting, and other transformations based
    on MetaSchedule schedule rules but directly samples from the search space instead of using the
    tuning algorithm. User can specify the number of valid counts to try and whether to use runner
    for benchmarking.

    Parameters
    ----------
    valid_count: int
        The number of valid counts to try.
    benchmark: bool
        Whether to use runner for benchmarking.

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    return _ffi_api.FewShotTuning(valid_count, benchmark)  # type: ignore


def DecomposeOpsForInference(func_name: Optional[str] = None) -> tvm.ir.transform.Pass:
    """Decompose composite operators that are composed by other operators during inference.
    For example, the result of batch norm (a triple) will be simplified. Attention, tensor_to_shape,
    etc. can be also decomposed into a number of simplified operators as well.

    Parameters
    ----------
    func_name: Optional[str]
        The name of the specified function. If not specified, the pass will run in
        all functions.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass
    """
    return _ffi_api.DecomposeOpsForInference(func_name)  # type: ignore


def DecomposeOpsForTraining(func_name: Optional[str] = None) -> tvm.ir.transform.Pass:
    """Decompose composite operators that are composed by other operators during training.
    For example, the result of batch norm (a triple) will be simplified. Attention, tensor_to_shape,
    etc. can be also decomposed into a number of simplified operators as well.

    Parameters
    ----------
    func_name: Optional[str]
        The name of the specified function. If not specified, the pass will run in
        all functions.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass
    """
    return _ffi_api.DecomposeOpsForTraining(func_name)  # type: ignore


def AlterOpImpl(
    op_impl_map: Dict[str, PrimFunc],
    op_buffer_transforms: Dict[str, List[Union[IndexMap, Callable]]],
    op_buffer_axis_separators: Dict[str, List[Union[IndexMap.AXIS_SEPARATOR, Callable]]],
    op_buffer_input_axis_separators: Dict[str, List[Union[IndexMap.AXIS_SEPARATOR, Callable]]],
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
    op_buffer_axis_separators: Dict[str, List[Union[IndexMap.AXIS_SEPARATOR, Callable]]]
        op_kind to axis_separator for each index_map
    op_buffer_input_axis_separators: Dict[str, List[Union[IndexMap.AXIS_SEPARATOR, Callable]]]
        op_kind to axis_separator for input index_map

    Returns
    -------
    ret: tvm.ir.transform.Pass
    """
    for operator_name, transform_list in op_buffer_transforms.items():
        l = []
        for transform in transform_list:
            # Extract the index_map
            if isinstance(transform, Callable):
                transform = IndexMap.from_func_with_separators(transform)[0]
            elif isinstance(transform, (Array, tuple)) and isinstance(transform[0], IndexMap):
                transform = transform[0]
            l.append(transform)
        op_buffer_transforms[operator_name] = l

    return _ffi_api.AlterOpImpl(
        op_impl_map,
        op_buffer_transforms,
        op_buffer_axis_separators,
        op_buffer_input_axis_separators,
    )  # type: ignore


def ConvertLayout(desired_layouts: Dict[str, List[str]]) -> tvm.ir.transform.Pass:
    """Automatic layout conversion pass.

    Parameters
    ----------
    desired_layouts : Dict[str, List[str]]
        The desired layout of conv2d ops is a map from the name of the op to the desired layout
        of the desired feature map, weight and output. For example, if we want to convert the
        layout of conv2d from NCHW to NHWC, we can set the desired layout of conv2d to be
        ``{"relax.nn.conv2d": ["NHWC", "OHWI"]}``.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for layout conversion.
    """
    return _ffi_api.ConvertLayout(desired_layouts)  # type: ignore


def DeadCodeElimination(entry_functions: Optional[List[str]] = None) -> tvm.ir.transform.Pass:
    """Remove dead code in the IRModule.
    Currently it removes:

       1. Unused local VarBindings
          (those where the bound var is unused and no impure operation is used).
       2. Unused Relax functions in the module.
          We detect the call chain from the entry function, and remove all unused functions.

    Any binding blocks that are left empty will be removed by the normalizer.

    Notes
    -----
    For function-wise DCE, use py:func:`tvm.relax.analysis.remove_all_unused`.

    Parameters
    ----------
    entry_functions: Optional[List[str]]
        The set of entry functions to start from.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass.
    """
    if entry_functions is None:
        entry_functions = []
    return _ffi_api.DeadCodeElimination(entry_functions)  # type: ignore


def ToMixedPrecision(
    out_dtype="float32", fp16_input_names: Optional[List[str]] = None
) -> tvm.ir.transform.Pass:
    """Automatic mixed precision pass. Currently the pass assumes the input module to be fp32
    only, and will automatically cast fp32 to fp16 for certain ops.

    Note: Mainly operates within dataflow blocks. ConvertToDataflow may need to be called first.

    Parameters
    ----------
    out_dtype : str
        The output data type of gemm/conv, which is the data type of the accumulator.
    fp16_input_names : List[str]
        The names of function parameters whose dtype should become fp16. The  function signature
        would change accordingly.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for mixed precision.
    """
    return _ffi_api.ToMixedPrecision(out_dtype, fp16_input_names)  # type: ignore


def SplitCallTIRByPattern(patterns: List[PrimFunc], fcodegen: Callable) -> tvm.ir.transform.Pass:
    """Split a PrimFunc into 2 parts: the first part is a TIR PrimFunc which is
       matched with some pattern, and the second part is the rest of the original
       PrimFunc. It will call fcodegen to generate the code for the matched pattern
       to replace it with a ExternFunc call.

    Parameters
    ----------
    patterns : List[PrimFunc]
        The list of patterns to match.

    fcodegen: Callable[[List[MatchResult]], List[Object]]
        The function to generate the code for the matched patterns.

    Returns
    -------
    ret : tvm.transform.Pass
        The registered pass for splitting call_tir.
    """
    return _ffi_api.SplitCallTIRByPattern(patterns, fcodegen)  # type: ignore


def UpdateParamStructInfo(sinfo_func: Callable[[Var], Optional[StructInfo]]):
    """Update struct info of parameters

    Update struct info of parameters.  Internal bindings and function
    return type will be updated using relax's struct inference rules.
    Errors resulting from struct inference will be propagated to the
    user.

    Parameters
    ----------
    sinfo_func: Callable[[Var], Optional[StructInfo]]

        A function that is called once for each function parameter,
        and returns the updated struct info to be used for it.  If the
        function returns `None`, the parameter is not modified.

    Returns
    -------
    ret : tvm.transform.Pass
        The corresponding pass.

    """
    return _ffi_api.UpdateParamStructInfo(sinfo_func)  # type: ignore


def AdjustMatmulOrder():
    """Reorder `x*(A*B)` to `(x*A)*B`

    Useful for optimizing LoRA computations, where `matmul(x,
    LoraA*LoraB)` may be computed as `matmul(matmul(x, LoraA),
    LoraB)`, reducing the total memory usage.


    Returns
    -------
    ret : tvm.transform.Pass
        The corresponding pass.
    """

    return _ffi_api.AdjustMatmulOrder()  # type: ignore


def ExpandMatmulOfSum():
    """Expand `matmul(x, A+B)` to `matmul(x,A) + matmul(x,B)`

    If either operand can be fully computed at compile-time (only
    depends on function parameters after kNumInput), this expansion is
    suppressed.

    Useful for optimizing LoRA computations, where `matmul(x, Base +
    LoraA*LoraB)` may be expanded to `matmul(x, Base) + matmul(x,
    LoraA*LoraB)`, allowing it to optimized with  `CombineParallelMatmul`.

    Returns
    -------
    ret : tvm.transform.Pass
        The corresponding pass.
    """

    return _ffi_api.ExpandMatmulOfSum()  # type: ignore


def ReorderPermuteDimsAfterConcat():
    """Reorder `concat(permute_dims(A), permute_dims(B))` into `permute_dims(concat(A,B))`

    Useful for optimizing computations after `CombineParallelMatmul`.
    The patterns for optimized `nn.Linear` implementations look for
    `matmul(activations, permute_dims(weights))`.  After
    `CombineParallelMatmul`, the `matmul(activations,
    concat(permute_dims(A), permute_dims(B)))` no longer matches this
    pattern.  Rearranging into `matmul(activations,
    permute_dims(concat(A,B)))` restores the pattern match.

    Returns
    -------
    ret : tvm.transform.Pass
        The corresponding pass.
    """

    return _ffi_api.ReorderPermuteDimsAfterConcat()  # type: ignore


def ReorderTakeAfterMatmul():
    """Reorder `matmul(x, take(weights, indices))` to `take(matmul(x,weights),indices)`

    Useful for optimizing LoRA computations, where several LoRAs may
    be batched together.

    Returns
    -------
    ret : tvm.transform.Pass
        The corresponding pass.
    """

    return _ffi_api.ReorderTakeAfterMatmul()  # type: ignore


def CombineParallelMatmul(check=None):
    """Combine multiple matmul operators sharing the same LHS matrix into one,
    followed by slicing. When all matmul branches in a tree have the same set of fused ops,
    the fused ops are applied to the combined matmul output before slicing.

    Currently, only a limited set of fused ops is supported. It includes bias add,
    relu, gelu, gelu_tanh and silu activation.

    Parameters
    ----------
    check : Callable[[Var, List[Var], List[Var], Dict[Var, Expr]], bool]
        A function to filter out unwanted branches, with the signature
        (input, [rhs], [bias], binding) -> bool.

    Returns
    -------
    ret : tvm.transform.Pass
        The corresponding pass.
    """
    if check is None:
        check = lambda *_: True
    return _ffi_api.CombineParallelMatmul(check)  # type: ignore


def RewriteCUDAGraph() -> tvm.ir.transform.Pass:
    """Rewrite a Relax module for executing with CUDA graph. This pass identifies the regions that
    can be executed with CUDA graph and lifts them into new functions for runtime graph capturing.

    Returns
    -------
    ret: tvm.ir.transform.Pass
        The registered pass for rewriting cuda graph
    """
    return _ffi_api.RewriteCUDAGraph()  # type: ignore


def AllocateWorkspace() -> tvm.ir.transform.Pass:
    """Allocate a workspace, represented by a tensor of size big enough for all external
    functions that require a temporary storage, and append it to the arguments of external
    functions.

    An external function can specify its workspace requirement by the kWorkspaceSize attribute.

    Returns
    -------
    ret: tvm.ir.transform.Pass
        The registered pass for allocating workspace.
    """
    return _ffi_api.AllocateWorkspace()  # type: ignore


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
