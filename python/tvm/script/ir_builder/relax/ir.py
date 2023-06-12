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
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import tvm
from tvm import DataType, relax
from tvm.ir import PrimExpr
from ..ir import decl_function
from tvm.relax import Call, Expr, ExternFunc, TupleGetItem, ShapeExpr, Var, VarBinding, const
from tvm.relax.utils import gen_call_tir_inputs


############################### Operators ###############################
from tvm.relax.op import (
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    add,
    arange,
    argmax,
    argmin,
    assert_op,
    astype,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    broadcast_to,
    builtin,
    call_builtin_with_ctx,
    call_pure_packed,
    call_tir,
    call_dps_packed,
    ceil,
    clip,
    collapse_sum_like,
    collapse_sum_to,
    concat,
    cos,
    cosh,
    cumsum,
    einsum,
    scatter_elements,
    divide,
    equal,
    ewise_fma,
    exp,
    expand_dims,
    flatten,
    flip,
    floor,
    floor_divide,
    full,
    full_like,
    grad,
    greater,
    greater_equal,
    image,
    invoke_closure,
    invoke_pure_closure,
    isfinite,
    isinf,
    isnan,
    layout_transform,
    less,
    less_equal,
    linear,
    log,
    logical_and,
    logical_or,
    logical_xor,
    make_closure,
    matmul,
    max,
    maximum,
    mean,
    memory,
    min,
    minimum,
    multiply,
    negative,
    not_equal,
    null_value,
    ones,
    ones_like,
    permute_dims,
    power,
    print,
    prod,
    repeat,
    reshape,
    tensor_to_shape,
    shape_to_tensor,
    round,
    rsqrt,
    shape_of,
    std,
    strided_slice,
    dynamic_strided_slice,
    sum,
    take,
    variance,
    sigmoid,
    sign,
    sin,
    sinh,
    split,
    square,
    squeeze,
    sqrt,
    subtract,
    tan,
    tanh,
    tile,
    tril,
    triu,
    unique,
    vm,
    where,
    wrap_param,
    zeros,
    zeros_like,
    nn,
)
from tvm.relax.op.builtin import stop_lift_params
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


def is_pure(purity: bool = True) -> None:
    """Specify the purity of the last function frame.

    Parameters
    ----------
    purity: bool
        The annotated purity.
    """
    return _ffi_api.FuncIsPure(purity)  # type: ignore[attr-defined] # pylint: disable=no-member


def is_impure() -> None:
    """Specify that the last function frame is annotated as impure.
    (Syntactic sugar for R.is_pure(False))"""
    return _ffi_api.FuncIsPure(False)  # type: ignore[attr-defined] # pylint: disable=no-member


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


############################### Emits ###############################


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


def emit_te(func: Callable, *args: Any, **kwargs: Any) -> Call:
    """Emit a call node according to the te function.
    This function converts arguments from relax expression to te tensor,
    The callback func should return a te tensor or a list of te tensors.

    Parameters
    ----------
    func : Callable
        A function that returns a te tensor or a list of te tensors.

    args : Any, optional
        arguments passed to the function.

    kwargs : Any, optional
        The keyword arguments passed to the function.
        Note that the following keyword args are reserved:

            - 'primfunc_name_hint' for passing name hint to the PrimFunc
                that gets generated.
            - 'primfunc_attrs' is reserved for passing func attributes to
                be added to the PrimFunc that gets created.

    Returns
    -------
    call : Call
        A newly created call that calls into a tir function.
    """
    primfunc_name_hint = kwargs.pop("primfunc_name_hint", None)
    tir_func, call_args, out_sinfo, tir_vars = gen_call_tir_inputs(func, *args, **kwargs)
    if not primfunc_name_hint:
        primfunc_name_hint = func.__name__
    gvar = decl_function(primfunc_name_hint, tir_func)  # type: ignore
    return call_tir(gvar, call_args, out_sinfo, tir_vars)


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


def emit_var_binding(value: VarBinding) -> Var:
    """Emit a binding to the last binding block frame.
    Parameters
    ----------
    value: VarBinding
        The binding to be emitted.
    Returns
    -------
    var: Var
        The left side var of the emitted binding.
    """
    return _ffi_api.EmitVarBinding(value)  # type: ignore


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


############################### R.shape ################################


def shape(value: List[PrimExpr]) -> Expr:
    """Create a ShapeExpr.
    Parameters
    ----------
    value : List[PrimExpr]
        The fields of the tuple.
    Returns
    -------
    res : Expr
        The result tuple.
    """
    return relax.ShapeExpr(value)  # pylint: disable=no-member # type: ignore


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
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "add",
    "arange",
    "arg",
    "argmax",
    "argmin",
    "assert_op",
    "astype",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "broadcast_to",
    "builtin",
    "call_packed",
    "call_pure_packed",
    "call_tir",
    "call_dps_packed",
    "call_builtin_with_ctx",
    "ceil",
    "clip",
    "collapse_sum_like",
    "collapse_sum_to",
    "concat",
    "cos",
    "cosh",
    "const",
    "cumsum",
    "einsum",
    "scatter_elements",
    "dataflow",
    "divide",
    "dtype",
    "emit",
    "emit_te",
    "emit_var_binding",
    "emit_match_cast",
    "equal",
    "ewise_fma",
    "exp",
    "expand_dims",
    "flatten",
    "flip",
    "floor",
    "floor_divide",
    "full",
    "full_like",
    "func_attr",
    "func_name",
    "func_ret_struct_info",
    "func_ret_value",
    "function",
    "grad",
    "greater",
    "greater_equal",
    "image",
    "invoke_closure",
    "invoke_pure_closure",
    "is_impure",
    "is_pure",
    "isfinite",
    "isinf",
    "isnan",
    "layout_transform",
    "less",
    "less_equal",
    "linear",
    "log",
    "logical_and",
    "logical_or",
    "logical_xor",
    "make_closure",
    "matmul",
    "max",
    "maximum",
    "mean",
    "memory",
    "min",
    "minimum",
    "multiply",
    "negative",
    "not_equal",
    "null_value",
    "ones",
    "ones_like",
    "output",
    "permute_dims",
    "power",
    "prim_value",
    "print",
    "prod",
    "repeat",
    "reshape",
    "tensor_to_shape",
    "shape_to_tensor",
    "round",
    "rsqrt",
    "shape",
    "shape_of",
    "ShapeExpr",
    "std",
    "str",
    "sum",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "split",
    "square",
    "squeeze",
    "sqrt",
    "stop_lift_params",
    "str",
    "strided_slice",
    "dynamic_strided_slice",
    "subtract",
    "take",
    "tan",
    "tanh",
    "tile",
    "tril",
    "triu",
    "tuple",
    "unique",
    "variance",
    "vm",
    "where",
    "wrap_param",
    "zeros",
    "zeros_like",
    "nn",
]
