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
# pylint: disable=redefined-builtin
"""The base Relax operators."""

from typing import Dict, Union, List, Tuple, Optional, Callable


import tvm
import tvm.runtime
from tvm.runtime.object import Object
from tvm.runtime import ObjectGeneric

from . import _ffi_api
from ..expr import Expr, StringImm, ShapeExpr, Call, ExternFunc, GlobalVar, Var
from ..struct_info import StructInfo, TensorStructInfo
from ...ir import PrimExpr
from ..utils import args_converter


py_print = print  # pylint: disable=invalid-name


def register_gradient(
    op_name: str,
    fgradient: Callable[[Var, Call, Var, "BlockBuilder"], List[Expr]] = None,
    level: int = 10,
):
    """Register operator gradient function for a relax operator.

    Parameters
    ----------
    op_name: str
        The name of the op.

    fgradient: function (orig_var: Var, orig_call: Call, output_grad: Var, ctx: BlockBuilder)
         -> partials: List[Expr]
        The gradient function being used.

    level: int
        The priority level
    """
    return tvm.ir.register_op_attr(op_name, "FPrimalGradient", fgradient, level)


def null_value() -> Call:
    """Create a call node that represents a null value object.

    Returns
    -------
    ret: Call
        The created call node.
    """
    return _ffi_api.null_value()  # type: ignore


def _wrap_inline_arg_tuple(args) -> Expr:
    """Helper function to wrap argument tuple

    Normalize the arguments provided the functions that accept a tuple
    of arguments, and require the tuple of arguments to be written
    in-line.  If the arguments provided are a single relax expression,
    and are not a reference to a relax tuple, then wrap them into an
    in-line relax Tuple.

    """
    if (
        isinstance(args, Expr)
        and not isinstance(args, tvm.relax.Tuple)
        and (
            args.struct_info_ is None
            or not isinstance(args.struct_info_, tvm.relax.TupleStructInfo)
        )
    ):
        return tvm.relax.Tuple([args])
    else:
        return args


@args_converter.auto
def call_tir(
    gvar: GlobalVar,
    args: Expr,
    out_sinfo: Union[TensorStructInfo, List[TensorStructInfo]],
    tir_vars: Optional[Union[ShapeExpr, Tuple[PrimExpr], List[PrimExpr]]] = None,
) -> Call:
    """
    Call a tir.prim_func and return the output.

    Parameters
    ----------
    gvar : GlobalVar
        The GlobalVar referring to a tir PrimFunc.

    args : Expr
        The input arguments.

    out_sinfo : Union[TensorStructInfo, List[TensorStructInfo]]
        The structure info of the call_tir output.
        It should be a single or a list of TensorStructInfo. Each one denotes the
        structure info of a returned tensor.

    tir_vars : Optional[Union[ShapeExpr, Tuple[PrimExpr], List[PrimExpr]]]
        ShapeExpr representing a tuple of integers to unpack when calling func. Is null if not used

    Returns
    -------
    ret: Call
        A call node for the call_tir operator.
    """
    args = _wrap_inline_arg_tuple(args)

    if not isinstance(out_sinfo, list):
        out_sinfo = [out_sinfo]

    if isinstance(tir_vars, (list, tuple)):
        tir_vars = ShapeExpr(tir_vars)

    return _ffi_api.call_tir(gvar, args, out_sinfo, tir_vars)  # type: ignore


@args_converter.auto
def call_tir_with_grad(
    gvar: GlobalVar,
    args: Expr,
    out_sinfo: Union[TensorStructInfo, List[TensorStructInfo]],
    te_grad_name: str,
    te_grad_kwargs: Dict[str, Object] = None,
    tir_vars: Optional[Union[ShapeExpr, Tuple[PrimExpr], List[PrimExpr]]] = None,
) -> Call:
    """
    Call a tir.prim_func and return the output. This intrinsic will bind a te gradient function
    (refered by te_grad_name) to the call_tir_with_grad node. The te gradient function will be
    called by the Gradient pass.

    Parameters
    ----------
    gvar : GlobalVar
        The GlobalVar referring to a tir PrimFunc.

    args : Expr
        The input arguments.

    out_sinfo : Union[TensorStructInfo, List[TensorStructInfo]]
        The structure info of the call_tir_with_grad output.
        It should be a single or a list of TensorStructInfo. Each one denotes the
        structure info of a returned tensor.

    te_grad_name : str
        The registered name of the te gradient function associated with the call_tir_with_grad
        node. Must be provided as a keyword argument.

    te_grad_kwargs : Dict[str, Object], optional
        The keyword arguments passed to the te gradient function.
        Optionally provided as a keyword argument. Default: {}.

    tir_vars : Optional[Union[ShapeExpr, Tuple[PrimExpr], List[PrimExpr]]]
        ShapeExpr representing a tuple of integers to unpack when calling func. Is null if not used

    Returns
    -------
    ret: Call
        A call node for the call_tir_with_grad operator.
    """
    args = _wrap_inline_arg_tuple(args)

    if not isinstance(out_sinfo, list):
        out_sinfo = [out_sinfo]

    if isinstance(tir_vars, (list, tuple)):
        tir_vars = ShapeExpr(tir_vars)

    if te_grad_kwargs is None:
        te_grad_kwargs = {}

    return _ffi_api.call_tir_with_grad(  # type: ignore
        gvar, args, out_sinfo, te_grad_name, te_grad_kwargs, tir_vars
    )


@args_converter.auto
def call_tir_inplace(
    gvar: GlobalVar,
    args: Expr,
    inplace_indices: Union[int, List[int]],
    out_sinfo: Union[TensorStructInfo, List[TensorStructInfo]],
    tir_vars: Optional[Union[ShapeExpr, Tuple[PrimExpr], List[PrimExpr]]] = None,
) -> Call:
    """
    Call a TIR PrimFunc and return the result, doing the specified computations in-place
    (based on the `inplace_indices` argument; outputs will alias the inputs
    selected by in-place indices).

    Warning: This operator is considered pure by the type system but actually mutates
    the arguments specified by `inplace_indices`. This operator should not be used directly,
    but rather should be inserted by passes that have checked whether it is safe to perform
    operations in-place (i.e., none of the arguments specified as an output is aliased or is
    live after calling call_tir_inplace).

    Direct calls to this operator should be done for testing purposes only.

    Parameters
    ----------
    gvar : GlobalVar
        The GlobalVar referring to a TIR PrimFunc.

    args : Expr
        The input arguments.

    inplace_indices : Union[int, List[int]]
        Specify which arguments should be used for in-place computations.
        If `inplace_indices` is a single integer, it will be made into a singleton list.
        Suppose `inplace_indices[i] = j`, where `j >= 0`. Then the `i`th output
        will be an alias of `args[j]`.
        If `inplace_indices[i] = -1`, then the `i`th output will be a freshly allocated tensor.
        At least one member of `inplace_indices` must not be -1.

    out_sinfo : Union[TensorStructInfo, List[TensorStructInfo]]
        The structure info of the call_tir_inplace output.
        It should be a single `TensorStructInfo` or a list of `TensorStructInfo`.
        Each one denotes the structure info of a returned tensor.
        If a list of `TensorStructInfo` is given, the result will be a tuple of `TensorStructInfo`.

    tir_vars : Optional[Union[ShapeExpr, Tuple[PrimExpr], List[PrimExpr]]]
        ShapeExpr representing a tuple of integers to unpack when calling func. Is null if not used

    Returns
    -------
    ret: Call
        A call node for the call_tir operator.
    """
    args = _wrap_inline_arg_tuple(args)

    if not isinstance(inplace_indices, list):
        inplace_indices = [inplace_indices]

    if not isinstance(out_sinfo, list):
        out_sinfo = [out_sinfo]

    if isinstance(tir_vars, (list, tuple)):
        tir_vars = ShapeExpr(tir_vars)

    return _ffi_api.call_tir_inplace(  # type: ignore
        gvar,
        args,
        inplace_indices,
        out_sinfo,
        tir_vars,
    )


@args_converter.auto
def call_dps_packed(
    func: Union[str, Expr],
    args: Expr,
    out_sinfo: Union[TensorStructInfo, List[TensorStructInfo]],
) -> Call:
    """
    Call a destination-passing-style packed function and return the output.

    Note: The called function is assumed to be _pure_ (other than modifying the designated
    output arguments). If the function _does_ result in other side effects, then the compiler
    may end up removing, reordering, or repeating those effects--no guarantees can be made.

    Parameters
    ----------
    func : Union[str, Expr]
        The destination-passing-style function, can be ExternFunc.

    args : Expr
        The input arguments.

    out_sinfo : Union[TensorStructInfo, List[TensorStructInfo]]
        The structure info of the call_dps_packed output.
        It should be a single or a list of TensorStructInfo. Each one denotes the
        structure info of a returned tensor.

    Returns
    -------
    ret: Call
        A call node for the call_dps_packed operator.
    """
    if isinstance(func, str):
        func = ExternFunc(func)

    args = _wrap_inline_arg_tuple(args)

    if not isinstance(out_sinfo, list):
        out_sinfo = [out_sinfo]

    return _ffi_api.call_dps_packed(func, args, out_sinfo)  # type: ignore


@args_converter.auto
def call_builtin_with_ctx(
    func: Union[str, Expr],
    args: Expr,
    *,
    sinfo_args: Optional[Union[StructInfo, List[StructInfo]]] = None,
) -> Call:
    """Call a builtin function func.

    Parameters
    ----------
    func : Expr
        The builtin function to be called.

    args : Expr
        The input arguments.

    sinfo_args: Optional[Union[StructInfo, List[StructInfo]]]
        The struct info arguments to the call node.

    Returns
    -------
    ret: Call
        The created call node.
    """
    if isinstance(func, str):
        func = ExternFunc(func)

    if sinfo_args is not None and not isinstance(sinfo_args, (list, tuple)):
        sinfo_args = [sinfo_args]

    return _ffi_api.call_builtin_with_ctx(  # type: ignore
        func,
        args,
        sinfo_args,  # type: ignore
    )


@args_converter.auto
def make_closure(
    func: Expr,
    args: Expr,
) -> Object:
    """
    Create a closure with free variables and return the closure.

    Parameters
    ----------
    func : Expr
        The closure, can be ExternFunc or PrimFunc.

    args : Expr
        The input arguments.


    Returns
    -------
    ret: Object
        The VMClosure.
    """

    return _ffi_api.make_closure(func, args)  # type: ignore


@args_converter.auto
def invoke_closure(
    closure: Expr,
    args: Expr,
    sinfo_args: Union[List[StructInfo], StructInfo],
) -> Call:
    """
    Invoke a closure.

    Parameters
    ----------
    closure : Expr
        The VMClosure object.

    args : Expr
        The input arguments.

    type_args: Union[List[StructInfo], StructInfo]
        The structure info arguments of the CallNode

    Returns
    -------
    ret: Call
        A call to `invoke_closure`.
    """

    if not isinstance(sinfo_args, (list, tuple)):
        sinfo_args = [sinfo_args]

    return _ffi_api.invoke_closure(closure, args, sinfo_args)  # type: ignore


def render_object(val: tvm.Object) -> str:
    """
    Given a TVM Object, renders it in string form. Used for Relax printing and assertions.

    Parameters
    ----------
    val: tvm.Object
        An object to render

    Returns
    -------
    ret: str
        A string representing the value, ideally human-readable
    """
    if isinstance(val, tvm.nd.NDArray):
        return str(val)
    # no pretty-printer by default, so if we don't handle this,
    # then we can't look inside tuples
    if isinstance(val, tvm.runtime.container.ADT):
        # the fields array of an ADT cannot be directly accessed in Python
        # so we have to get the length and index into the fields separately
        fields = ", ".join([render_object(val[i]) for i in range(len(val))])
        # special case: tag = 0 is a tuple
        if val.tag == 0:
            return f"({fields})"
        return f"ADT(tag={val.tag}, fields=[{fields}])"
    if isinstance(val, tvm.ir.Array):
        fields = ", ".join([render_object(val[i]) for i in range(len(val))])
        return f"({fields})"
    return str(val)


@tvm.register_func("relax.run.shape_to_tensor")
def relax_shape_to_tensor(shape_tuple: tvm.runtime.ShapeTuple) -> tvm.nd.NDArray:
    """
    Takes a ShapeTuple and convert it to NDArray.

    Parameters
    ----------
    shape_tuple: tvm.runtime.ShapeTuple
        Shape tuple that we want to convert to NDArray at runtime
    """
    return tvm.nd.array([int(v) for v in shape_tuple])


@tvm.register_func("relax.run.print")
def relax_print(format_str: str, *format_args: tvm.Object) -> None:
    """
    Takes a list of values to print, formats with the given format string.
    If the format string is empty, simply prints.

    Call from TVM script like this:
    `relax.print(value1, value2, ..., valueN, format=format_str)`
    or
    `relax.print(value1, value2, ..., valueN) # format_str defaults to ""`

    Parameters
    ----------
    format_str: str
        The last argument is a Python-style format string for printing the value

    format_args: List[Object]
        The values to print.
    """
    val_strs = map(render_object, format_args)
    if format_str == "":
        py_print(*val_strs)
    else:
        py_print(format_str.format(*val_strs))


def print(*values: List[Expr], format: Union[str, Expr] = "") -> Expr:
    """Print op to print the values

    Parameters
    ----------
    values : List[Expr]
        The values to print.

    format: Union[str, Expr]
        The format string or StringImm.

    Returns
    -------
    result : Expr
        A relax Call, which will print the value during runtime.
    """
    if isinstance(format, str):
        format = StringImm(format)

    return _ffi_api.print(values, format)  # type: ignore # pylint: disable=no-member


@tvm.register_func("relax.run.assert_op")
def relax_assert_op(condition: tvm.Object, format_str: str, *format_args: tvm.Object) -> None:
    """
    A variadic function. The first value serves as the assertion condition:
    If the condition is true, then the operator does nothing.
    If the condition is false, then the operator raises an assertion error.

    Arguments after the first value serve as format arguments for the error message;
    the last argument must be a format string for the error message (empty by default).
    If the format string is the empty string, then the error message will simply include
    a comma-separated list of the format arguments.
    The condition argument is not included in the format string.

    Parameters
    ----------
    condition: tvm.Object
        The assertion condition. Must be a boolean scalar.

    format_str: str
        The last argument is a Python-style format string for printing the value

    format_args: List[tvm.Object]
        Values used for formatting the string.
    """
    if not isinstance(format_str, str):
        raise ValueError(
            f"The format string argument to assert must be a string, given {type(format_str)})"
        )

    if isinstance(condition, (bool, int)):
        val = condition
    elif isinstance(condition, tvm.nd.NDArray):
        # may happen if the original program had unknown shape or dtype for the tensor's type
        dtype = condition.dtype
        if dtype != "bool":
            raise ValueError(f"The condition must be a bool scalar, but given a {dtype} tensor")
        shape = condition.shape
        if len(shape) != 0:
            raise ValueError(f"The condition must be a scalar, but it has a shape of {shape}")

        val = condition.numpy()

    else:
        # should be guaranteed by the type system
        raise ValueError(
            f"The condition for relax assert must be a bool, int, or NDArray, "
            f"but received a {type(condition)}."
        )

    if not val:
        error_message = "Assertion Failed"
        if format_args or format_str != "":
            rendered = map(render_object, format_args)
            if format_str != "":
                error_message = format_str.format(*rendered)
            else:
                error_message = ", ".join(rendered)
        raise AssertionError(error_message)


def assert_op(
    condition: Union[Expr, PrimExpr],
    format_args: Optional[Union[Expr, List[Expr]]] = None,
    format: Union[str, Expr] = "",
) -> Expr:
    """
    Create a call to Relax's assert_op operation (`assert` is reserved in Python,
    so the name must be distinct).

    Parameters
    ----------
    condition: Union[Expr, PrimExpr]
        The assertion condition.

    format_args: Optional[Union[Expr, List[Expr]]]
        Format arguments for the error message if the condition fails.

    format: Union[str, Expr]
        The format string or StringImm for the error message.

    Returns
    -------
    result : Expr
        A Call to the Relax assert operation.
    """
    if not isinstance(condition, Expr):
        condition = tvm.relax.PrimValue(condition)

    if format_args is None:
        format_args = []
    elif isinstance(format_args, Expr):
        format_args = [format_args]

    if isinstance(format, str):
        format = StringImm(format)

    return _ffi_api.assert_op(condition, format_args, format)  # type: ignore


def shape_of(expr: Expr) -> Expr:
    """Get shape of a tensor.

    Parameters
    ----------
    expr : Expr
        The input Expr.

    Returns
    -------
    result : Expr
        A relax Call, which gets the shape of the input
    """
    return _ffi_api.shape_of(expr)  # type: ignore # pylint: disable=no-member


def tensor_to_shape(expr: Expr) -> Expr:
    """Convert tensor to shape expr.
    Parameters
    ----------
    expr : Expr
        The input Expr
    Returns
    -------
    result : Expr
        A relax Call, which transforms the tensor values to the shape
    """
    return _ffi_api.tensor_to_shape(expr)  # type: ignore # pylint: disable=no-member


def shape_to_tensor(expr: Expr) -> Expr:
    """Convert shape to tensor expr.
    Parameters
    ----------
    expr : Expr
        The input Expr
    Returns
    -------
    result : Expr
        A relax Call, which transforms the shape values to the tensor
    """
    return _ffi_api.shape_to_tensor(expr)  # type: ignore # pylint: disable=no-member


@args_converter.auto
def call_inplace_packed(
    func: Union[str, ExternFunc, GlobalVar],
    *args: Expr,
    inplace_indices: Union[int, List[int]],
    sinfo_args: Union[StructInfo, List[StructInfo]],
) -> Expr:
    """
    Construct a call to a packed function that consumes some of its arguments "in-place"
    and returns the mutated arguments (aliased), but should be considered to be otherwise pure.
    The `inplace_indices` argument indicates which of the outputs are mutated arguments.

    The resulting call will have the same semantics as calling the packed function directly.

    Note: This should be used for cases when the user knows that calling the packed function
    with these arguments will **in reality** not cause any other side effects.
    If it is used for a call that **does** result in other side effects, then the compiler
    may end up removing, reordering, or repeating that call, with no guarantees
    made about any side effects from the callee.

    Warning: This operator as treated as pure by the type system even though it *is* performing
    side effects (mutating some arguments). It is therefore incumbent upon the user to ensure
    that it is being used safely (viz., that mutated arguments are not live after the mutation,
    that they do not alias values live after the mutation).

    Parameters
    ----------
    func : Union[str, ExternFunc]
      The name (global symbol) for a PackedFunc or an ExternFunc node.

    args: Expr
      The arguments for the PackedFunc.

    inplace_indices : Union[int, List[int]]
      Specify which arguments should be used for in-place computations.
      If `inplace_indices` is a single integer, it will be made into a singleton list.
      Suppose `inplace_indices[i] = j`, where `j >= 0`. Then the `i`th output
      will be an alias of `args[j]`.
      If `inplace_indices[i] = -1`, then the `i`th output will be a freshly allocated tensor.
      At least one member of `inplace_indices` must not be -1.

    sinfo_args: Union[StructInfo, List[StructInfo]]
        The list of structure info arguments (giving the structural info for the returned value).

    Returns
    -------
    result : Expr
      A Relax call, corresponding to
      `call_pure_packed(ExternFunc(func), args, DictAttrs(kwargs), sinfo_args)`
    """
    if isinstance(func, ExternFunc):
        func = func.global_symbol

    op = ExternFunc(func)
    if sinfo_args is None:
        raise ValueError("R.call_pure_packed is required to have type_args")
    if isinstance(sinfo_args, tuple):  # type: ignore
        sinfo_args = list(sinfo_args)
    elif not isinstance(sinfo_args, list):
        sinfo_args = [sinfo_args]
    if not isinstance(inplace_indices, list):
        inplace_indices = [inplace_indices]

    return _ffi_api.call_inplace_packed(op, args, inplace_indices, sinfo_args)  # type: ignore # pylint: disable=no-member


@args_converter.auto
def call_pure_packed(
    func: Union[str, ExternFunc, GlobalVar],
    *args: Expr,
    sinfo_args: Union[StructInfo, List[StructInfo]],
) -> Expr:
    """
    Construct a call to a packed function that should be treated as pure,
    even though packed calls are normally not treated as pure.

    The resulting call will have the same semantics as calling the packed function directly.

    Note: This should be used for cases when the user knows that calling the packed function
    with these arguments will **in reality** not cause any side effects.
    If it is used for a call that **does** result in side effects, then the compiler
    may end up removing, reordering, or repeating that call, with no guarantees
    made about any side effects from the callee.

    Parameters
    ----------
    func : Union[str, ExternFunc]
      The name (global symbol) for a PackedFunc or an ExternFunc node.

    args: Expr
      The arguments for the PackedFunc.

    sinfo_args: Union[StructInfo, List[StructInfo]]
        The list of structure info arguments (giving the structural info for the returned value).

    Returns
    -------
    result : Expr
      A Relax call, corresponding to
      `call_pure_packed(ExternFunc(func), args, DictAttrs(kwargs), sinfo_args)`
    """
    if isinstance(func, ExternFunc):
        func = func.global_symbol

    op = ExternFunc(func)

    if sinfo_args is None:
        raise ValueError("R.call_pure_packed is required to have type_args")

    if isinstance(sinfo_args, tuple):  # type: ignore
        sinfo_args = list(sinfo_args)
    elif not isinstance(sinfo_args, list):
        sinfo_args = [sinfo_args]

    sinfo_args = [
        sinfo()
        if callable(sinfo)
        else sinfo.asobject()
        if isinstance(sinfo, ObjectGeneric)
        else sinfo
        for sinfo in sinfo_args
    ]

    # note: if we need attributes, we can also take them here

    return _ffi_api.call_pure_packed(op, args, None, sinfo_args)  # type: ignore # pylint: disable=no-member


@args_converter.auto
def invoke_pure_closure(
    closure: Expr,
    args: Expr,
    sinfo_args: Union[List[StructInfo], StructInfo],
) -> Call:
    """
    Invoke a closure and indicate to the compiler that it is pure.

    Note: This should be used for cases when the user knows that calling the closure
    with these arguments will **in reality** not cause any side effects.
    If it is used for a call that _does_ result in side effects, then the compiler
    may end up removing, reordering, or repeating that call, with no guarantees
    made about any side effects from the callee.

    Parameters
    ----------
    closure : Expr
        The VMClosure object.

    args : Expr
        The input arguments.

    type_args: Union[List[StructInfo], StructInfo]
        The structure info arguments of the CallNode

    Returns
    -------
    ret: Call
        A call to `invoke_pure_closure`.
    """

    if not isinstance(sinfo_args, (list, tuple)):
        sinfo_args = [sinfo_args]

    return _ffi_api.invoke_pure_closure(closure, args, sinfo_args)  # type: ignore


def to_vdevice(data, dst_vdevice) -> Expr:
    """Copy data to the destination device. This
    operator helps data transferring between difference devices for
    heterogeneous execution.

    Parameters
    ----------
    data : Expr
        The tensor to be copied.

    dst_device : VDevice
        The destination device where the data is copied to.

    Returns
    -------
    result : Expr
        The copied result.
    """
    return _ffi_api.to_vdevice(data, dst_vdevice)  # type: ignore


def hint_on_device(data, dst_vdevice) -> Expr:
    """It provides a hint specifying the device on which the input data should be executed.
    This hint is utilized by RealizeVDevice to propagate the virtual device."

    Parameters
    ----------
    data : Expr
        The tensor to be copied.

    dst_device : VDevice
        The destination device where the data is supposed to be executed.

    Returns
    -------
    result : Expr
        The result.
    """
    return _ffi_api.hint_on_device(data, dst_vdevice)  # type: ignore
