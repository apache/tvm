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
# pylint: disable=redefined-builtin, wrong-import-order, no-member, invalid-name
"""IRBuilder for Relax dialect"""

import builtins
import functools
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import tvm
from tvm import DataType, relax
from tvm.ir import PrimExpr
from tvm.relax import Call, Expr, ExternFunc, TupleGetItem, Var, const

############################### Operators ###############################
from tvm.relax.op import (
    add,
    assert_op,
    builtin,
    call_builtin_with_ctx,
    call_tir,
    invoke_closure,
    make_closure,
    memory,
    multiply,
    null_value,
    print,
    reshape,
    shape_of,
)
from tvm.relax.struct_info import StructInfo
from tvm.relax.utils import args_converter
from tvm.runtime import Object as tvm_Object
from tvm.runtime import ObjectGeneric

from . import _ffi_api, frame

##################### Python Native Function Alias ######################

py_print = builtins.print
py_tuple = tuple
py_str = str


############################### Function ################################


def function() -> frame.FunctionFrame:
    """Start a function frame.
    Returns
    -------
    frame: FunctionFrame
        The constructed function frame.
    """
    return _ffi_api.Function()  # type: ignore[attr-defined] # pylint: disable=no-member


def arg(name: py_str, struct_info: StructInfo) -> Var:
    """Add a parameter to the last function frame.
    Parameters
    ----------
    name: str
        The name of the parameter.
    struct_info: StructInfo
        The Struct Info of the parameter

    Returns
    -------
    var: Var
        The created function parameter var.
    """

    return _ffi_api.Arg(name, struct_info)  # type: ignore[attr-defined] # pylint: disable=no-member


def func_name(name: py_str) -> None:
    """Specify the name of the last function frame.
    Parameters
    ----------
    name: str
        The function name.
    """
    return _ffi_api.FuncName(name)  # type: ignore[attr-defined] # pylint: disable=no-member


def func_attr(attrs: Dict[py_str, tvm_Object]) -> None:
    """Specify the attrs of the last function frame.
    Parameters
    ----------
    attrs: Dict[str, Object]
        The function attrs.
    """
    return _ffi_api.FuncAttrs(attrs)  # type: ignore[attr-defined] # pylint: disable=no-member


def func_ret_struct_info(ret_sinfo: StructInfo) -> None:
    """Specify the return struct info of the last function frame.
    Parameters
    ----------
    ret_type: StructInfo
        The function return struct info.
    """
    return _ffi_api.FuncRetStructInfo(ret_sinfo)  # type: ignore[attr-defined] # pylint: disable=no-member


def func_ret_value(value: Expr) -> None:
    """Specify the return value of the last function frame.
    Parameters
    ----------
    value: Expr
        The function return value.
    """
    return _ffi_api.FuncRetValue(value)  # type: ignore[attr-defined] # pylint: disable=no-member


############################# BindingBlock ##############################


def dataflow() -> frame.BlockFrame:
    """Start a dataflow binding block frame.
    Returns
    -------
    frame: frame.BlockFrame
        The created ir_builder Block frame.
    """
    return _ffi_api.Dataflow()  # type: ignore[attr-defined] # pylint: disable=no-member


def output(*vars: Tuple[Var]) -> None:
    """Expose the dataflow block output variables as global ones.
    Parameters
    ----------
    vars: Tuple[Var]
        The output variables of a dataflow block.
    """
    return _ffi_api.DataflowBlockOutput(vars)  # type: ignore[attr-defined] # pylint: disable=no-member


################################## Ops #################################


@args_converter.auto
def call_packed(
    func: py_str,
    *args: Expr,
    sinfo_args: Union[StructInfo, List[StructInfo]],
    **kwargs: Any,
) -> Call:
    """Create a relax Call, which calls a packed function.
    Parameters
    ----------
    func: str
        The name of extern function.
    *args : Expr
        The arguments.
    sinfo_args: Union[StructInfo, List[StructInfo]]
        The list of structure info arguments.
    kwargs: Expr
        The keyword arguments.

    Returns
    -------
    call: Call
        The created Relax Call
    """
    op = ExternFunc(func)
    if sinfo_args is None:
        raise ValueError("R.call_packed is required to have type_args")
    if isinstance(sinfo_args, py_tuple):  # type: ignore
        sinfo_args = list(sinfo_args)
    elif not isinstance(sinfo_args, list):
        sinfo_args = [sinfo_args]
    for i, sinfo_arg in enumerate(sinfo_args):
        if callable(sinfo_arg):
            sinfo_arg = sinfo_arg()
        # Convert possible StructInfoProxy to StructInfo
        if isinstance(sinfo_arg, ObjectGeneric):
            sinfo_arg = sinfo_arg.asobject()
        sinfo_args[i] = sinfo_arg

    is_default = False
    if "attrs_type_key" in kwargs:
        attrs_type_key = kwargs["attrs_type_key"]
        kwargs.pop("attrs_type_key")
    else:
        attrs_type_key = "DictAttrs"
        is_default = True
    attrs = None
    if kwargs or not is_default:
        attrs = tvm.ir.attrs.make_node(attrs_type_key, **kwargs)

    return Call(op, args, attrs=attrs, sinfo_args=sinfo_args)


def _sinfo_arg_wrapper(func):
    """A wrapper to convert StructInfoProxies to StructInfo for builtin operators with sinfo_args"""

    def _convert_tensor_type(args):
        if isinstance(args, (list, py_tuple)):  # type: ignore
            new_args = [_convert_tensor_type(x) for x in args]
            return type(args)(new_args)
        if isinstance(args, dict):
            return {_convert_tensor_type(k): _convert_tensor_type(v) for k, v in args.items()}
        if inspect.isfunction(args):
            args = args()
        if isinstance(args, ObjectGeneric):
            args = args.asobject()
        return args

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return func(*_convert_tensor_type(args), **_convert_tensor_type(kwargs))

    return wrapped  # type: ignore


invoke_closure = _sinfo_arg_wrapper(invoke_closure)  # pylint: disable=invalid-name

call_builtin_with_ctx = _sinfo_arg_wrapper(call_builtin_with_ctx)  # pylint: disable=invalid-name

############################### Bindings ###############################


def emit(value: Expr, annotate_struct_info: Optional[StructInfo] = None) -> Var:
    """Emit a binding to the last binding block frame.
    Parameters
    ----------
    value: Expr
        The right side value of the bindings to be emitted.

    annotate_struct_info: Optional[StructInfo]
        The optional struct info annotation for the emitted value.

    Returns
    -------
    var: Var
        The left side var of the emitted binding.
    """
    return _ffi_api.Emit(value, annotate_struct_info)  # type: ignore[attr-defined] # pylint: disable=no-member


def emit_match_cast(value: Expr, struct_info: StructInfo) -> Var:
    """Emit a match_cast binding to the last binding block frame.
    Parameters
    ----------
    value: Expr
        The value of the MatchCast to be emitted.
    struct_info: StructInfo
        The struct_info of the MatchCast to be emitted.

    Returns
    -------
    var: Var
        The left side var of the emitted binding.
    """
    return _ffi_api.EmitMatchCast(value, struct_info)  # type: ignore


############################# If Then Else #############################


def If(condition: Expr) -> frame.IfFrame:  # pylint: disable=invalid-name
    """Create an if frame.
    Parameters
    ----------
    condition : Expr
        The condition of if statement, executes the true branch if the condition is true,
        otherwise jump into the false branch.
    Returns
    -------
    res : frame.IfFrame
        The result IfFrame.
    """
    return _ffi_api.If(condition)  # type: ignore[attr-defined] # pylint: disable=no-member


def Then() -> frame.ThenFrame:  # pylint: disable=invalid-name
    """Create a then frame.
    Returns
    -------
    res : frame.ThenFrame
        The result ThenFrame.
    """
    return _ffi_api.Then()  # type: ignore[attr-defined] # pylint: disable=no-member


def Else() -> frame.ElseFrame:  # pylint: disable=invalid-name
    """Create an else frame.
    Returns
    -------
    res : frame.ElseFrame
        The result ElseFrame.
    """
    return _ffi_api.Else()  # type: ignore[attr-defined] # pylint: disable=no-member


############################### R.tuple ################################


def tuple(*fields: Expr) -> Expr:
    """Create a tuple expression.
    Parameters
    ----------
    *fields : Expr
        The fields of the tuple.
    Returns
    -------
    res : Expr
        The result tuple.
    """
    if len(fields) == 0:
        fields = py_tuple()

    return relax.Tuple(fields)  # type: ignore[attr-defined] # pylint: disable=no-member


############################### PrimValue ##############################


def prim_value(value: PrimExpr) -> Expr:
    """Create a prim value expression.
    Parameters
    ----------
    value : PrimExpr
        The value of the prim value.
    Returns
    -------
    res : Expr
        The result prim value.
    """
    return relax.PrimValue(value)  # type: ignore[attr-defined] # pylint: disable=no-member


def str(value: py_str) -> Expr:
    """Create a string imm expression.
    Parameters
    ----------
    value : str
        The value of the str.
    Returns
    -------
    res : Expr
        The result str.
    """
    return relax.StringImm(value)  # type: ignore[attr-defined] # pylint: disable=no-member


def dtype(value: Union[py_str, DataType]) -> Expr:
    """Create a dtype imm expression.
    Parameters
    ----------
    value : dtype
        The value of the dtype.
    Returns
    -------
    res : Expr
        The result dtype.
    """
    return relax.DataTypeImm(value)  # type: ignore[attr-defined] # pylint: disable=no-member


############################### Importer ###############################

__all__ = [
    "Else",
    "If",
    "Then",
    "TupleGetItem",
    "add",
    "arg",
    "assert_op",
    "builtin",
    "call_packed",
    "call_tir",
    "call_builtin_with_ctx",
    "const",
    "dataflow",
    "dtype",
    "emit",
    "emit_match_cast",
    "func_attr",
    "func_name",
    "func_ret_struct_info",
    "func_ret_value",
    "function",
    "invoke_closure",
    "make_closure",
    "memory",
    "multiply",
    "null_value",
    "output",
    "prim_value",
    "print",
    "reshape",
    "shape_of",
    "str",
    "tuple",
]
