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
from typing import Union, List, Tuple, Optional


import tvm
from tvm.runtime.object import Object

from . import _ffi_api
from ..expr import Expr, StringImm, ShapeExpr, Call, ExternFunc, GlobalVar
from ..expr import Tuple as RxTuple
from ..struct_info import StructInfo, TensorStructInfo
from ...ir import PrimExpr
from ..utils import args_converter


py_print = print  # pylint: disable=invalid-name


def null_value() -> Call:
    """Create a call node that represents a null value object.

    Returns
    -------
    ret: Call
        The created call node.
    """
    return _ffi_api.null_value()  # type: ignore


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
    if isinstance(args, Expr) and not isinstance(args, RxTuple):  # type: ignore
        args = RxTuple((args,))

    if not isinstance(out_sinfo, list):
        out_sinfo = [out_sinfo]

    if isinstance(tir_vars, (list, tuple)):
        tir_vars = ShapeExpr(tir_vars)

    return _ffi_api.call_tir(gvar, args, out_sinfo, tir_vars)  # type: ignore


@args_converter.auto
def call_dps_packed(
    func: Union[str, Expr],
    args: Expr,
    out_sinfo: Union[TensorStructInfo, List[TensorStructInfo]],
) -> Call:
    """
    Call a destination-passing-style packed function and return the output.

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

    if isinstance(args, Expr) and not isinstance(args, RxTuple):  # type: ignore
        args = RxTuple((args,))

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
) -> Object:
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
    ret: Object
        The result.
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

    # should be guaranteed by the type system
    if not isinstance(condition, tvm.nd.NDArray):
        raise ValueError(f"The condition must be an NDArray, but given a {type(condition)}.")

    # may happen if the original program had unknown shape or dtype for the tensor's type
    dtype = condition.dtype
    if dtype != "bool":
        raise ValueError(f"The condition must be a bool scalar, but given a {dtype} tensor")
    shape = condition.shape
    if len(shape) != 0:
        raise ValueError(f"The condition must be a scalar, but it has a shape of {shape}")

    val = condition.numpy()
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
    condition: Expr,
    format_args: Optional[Union[Expr, List[Expr]]] = None,
    format: Union[str, Expr] = "",
) -> Expr:
    """
    Create a call to Relax's assert_op operation (`assert` is reserved in Python,
    so the name must be distinct).

    Parameters
    ----------
    condition: Expr
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
    if format_args is None:
        format_args = []
    if isinstance(format_args, Expr):  # type: ignore
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
