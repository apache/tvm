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
"""Relax expression operators and explicit call construction."""

from __future__ import annotations

import builtins
import functools
import inspect
import numbers as _numbers
from typing import Any

import tvm_ffi as _ffi

import tvm
from tvm import ir as _ir
from tvm import relax as _relax
from tvm import tirx as _tir
from tvm.ir import StringImm
from tvm.ir.prim import _ffi_api as _prim_ffi
from tvm.relax import Call, Expr, ExternFunc
from tvm.relax.global_info import VDevice
from tvm.relax.op import (
    abs,
    acos,
    acosh,
    add,
    arange,
    argmax,
    argmin,
    argsort,
    asin,
    asinh,
    assert_op,
    astype,
    atan,
    atan2,
    atanh,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    broadcast_to,
    bucketize,
    builtin,
    call_builtin_with_ctx,
    call_dps_packed,
    call_inplace_packed,
    call_pure_packed,
    call_tir,
    call_tir_inplace,
    call_tir_with_grad,
    ccl,
    ceil,
    clip,
    collapse_sum_like,
    collapse_sum_to,
    concat,
    cos,
    cosh,
    cumprod,
    cumsum,
    dequantize,
    divide,
    dynamic_strided_slice,
    einsum,
    equal,
    erf,
    ewise_fma,
    exp,
    expand_dims,
    eye,
    eye_like,
    flatten,
    flip,
    floor,
    floor_divide,
    floor_mod,
    full,
    full_like,
    gather_elements,
    gather_nd,
    grad,
    greater,
    greater_equal,
    hamming_window,
    hint_on_device,
    image,
    index_put,
    index_tensor,
    invoke_closure,
    invoke_pure_closure,
    isfinite,
    isinf,
    isnan,
    layout_transform,
    left_shift,
    less,
    less_equal,
    linear,
    log,
    log_add_exp,
    logical_xor,
    make_closure,
    matmul,
    max,
    maximum,
    mean,
    median,
    memory,
    meshgrid,
    min,
    minimum,
    mod,
    multinomial_from_uniform,
    multiply,
    negative,
    nn,
    nonzero,
    not_equal,
    null_value,
    one_hot,
    ones,
    ones_like,
    outer,
    permute_dims,
    power,
    print,
    prod,
    quantize,
    repeat,
    reshape,
    reverse_sequence,
    right_shift,
    round,
    rsqrt,
    scatter_elements,
    scatter_nd,
    shape_of,
    shape_to_tensor,
    sigmoid,
    sign,
    sin,
    sinh,
    size,
    slice_scatter,
    sort,
    split,
    sqrt,
    square,
    squeeze,
    stack,
    std,
    strided_slice,
    subtract,
    sum,
    take,
    tan,
    tanh,
    tensor_to_shape,
    tile,
    topk,
    tril,
    triu,
    trunc,
    unique,
    variance,
    vision,
    vm,
    where,
    wrap_param,
    zeros,
    zeros_like,
)
from tvm.relax.op import (
    call_py_func as _call_py_func,
)
from tvm.relax.op.builtin import stop_lift_params
from tvm.relax.type import Type
from tvm.relax.utils import convert_to_expr
from tvm.runtime import ObjectConvertible
from tvm.script.ir_builder import base as _base

from .ir import _value, lookup_vdevice

py_print = builtins.print
py_tuple = tuple
py_str = str
_Span = _base.SpanEntry | _ir.Span | None


def to_vdevice(data: Expr, dst_vdevice: py_str | VDevice) -> Expr:
    """Copy data to the destination device.

    Parameters
    ----------
    data : Expr
        The tensor to be copied.

    dst_vdevice : Union[py_str, VDevice]
        The destination device where the data is copied to.

    Returns
    -------
    result : Expr
        The copied result.
    """
    if isinstance(dst_vdevice, py_str):
        if ":" in dst_vdevice:
            split_vdev = dst_vdevice.split(":")
            dst_vdevice = lookup_vdevice(split_vdev[0], int(split_vdev[1]))
        else:
            dst_vdevice = lookup_vdevice(dst_vdevice, 0)

    return tvm.relax.op.to_vdevice(data, dst_vdevice)


def call_packed(
    func: py_str,
    *args: Expr,
    ty_args: Type | list[Type] | None = None,
    **kwargs: Any,
) -> Call:
    """Create a relax Call, which calls a packed function.
    Parameters
    ----------
    func: str
        The name of extern function.
    *args : Expr
        The arguments.
    ty_args: Optional[Union[Type, List[Type]]]
        The list of type information arguments.
    kwargs: Expr
        The keyword arguments.

    Returns
    -------
    call: Call
        The created Relax Call
    """
    op = ExternFunc(func)
    args = py_tuple(convert_to_expr(a) for a in args)
    if ty_args is None:
        ty_args = []
    if isinstance(ty_args, py_tuple):  # type: ignore
        ty_args = list(ty_args)
    elif not isinstance(ty_args, list):
        ty_args = [ty_args]

    ty_args = [
        (ty() if callable(ty) else ty.asobject() if isinstance(ty, ObjectConvertible) else ty)
        for ty in ty_args
    ]

    is_default = False
    if "attrs_type_key" in kwargs:
        attrs_type_key = kwargs["attrs_type_key"]
        kwargs.pop("attrs_type_key")
    else:
        attrs_type_key = "ir.DictAttrs"
        is_default = True
    attrs = None
    if kwargs or not is_default:
        attrs = tvm.ir.attrs.make_node(attrs_type_key, **kwargs)

    return Call(op, args, attrs=attrs, ty_args=ty_args)


def call_py_func(
    py_func_name: py_str,
    *args: Expr,
    out_ty: Type | list[Type],
) -> Call:
    """Create a relax Call, which calls a Python function.

    Parameters
    ----------
    py_func_name: str
        The name of the Python function to call. This should correspond to a function
        in the IRModule's pyfuncs attribute.
    *args : Expr
        The arguments.
    out_ty: Union[Type, List[Type]]
        The type information of the call_py_func output.
        It should be a single or a list of TensorType. Each one denotes the
        type information of a returned tensor.

    Returns
    -------
    call: Call
        The created Relax Call for call_py_func operator.
    """
    args = py_tuple(convert_to_expr(a) for a in args)
    if isinstance(out_ty, py_tuple):  # type: ignore
        out_ty = list(out_ty)
    elif not isinstance(out_ty, list):
        out_ty = [out_ty]

    out_ty = [
        (ty() if callable(ty) else ty.asobject() if isinstance(ty, ObjectConvertible) else ty)
        for ty in out_ty
    ]

    # Convert string to StringImm
    try:
        func_name_imm = (
            StringImm(py_func_name) if isinstance(py_func_name, py_str) else py_func_name
        )
    except (TypeError, ValueError, AttributeError):
        func_name_imm = StringImm(py_func_name)
    return _call_py_func(func_name_imm, args, out_ty)


def _ty_arg_wrapper(func):
    """Normalize callable and object-convertible type arguments for operators."""

    def _convert_tensor_type(args):
        if isinstance(args, list | py_tuple):  # type: ignore
            new_args = [_convert_tensor_type(x) for x in args]
            return type(args)(new_args)
        if isinstance(args, dict):
            return {_convert_tensor_type(k): _convert_tensor_type(v) for k, v in args.items()}
        if inspect.isfunction(args):
            args = args()
        if isinstance(args, ObjectConvertible):
            args = args.asobject()
        return args

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return func(*_convert_tensor_type(args), **_convert_tensor_type(kwargs))

    return wrapped  # type: ignore


def emit_with_type(
    op: str,
    args: Expr,
    ty_args: Type | list[Type] | None = None,
) -> Call:
    """Create a Relax Call with type arguments.
    Parameters
    ----------
    op: Expr
        The relax op for which type args are to be appended
    args : Expr
        The arguments.
    ty_args: Optional[Union[Type, List[Type]]]
        The list of type arguments.

    Returns
    -------
    call: Call
        The created Relax Call
    """
    builtin_call = tvm.ir.Op.get(op)
    return Call(builtin_call, args, attrs=None, ty_args=ty_args)


def _logical_pair(lhs, rhs, operation, primitive, python_operation):
    if not isinstance(lhs, _ir.Expr) and not isinstance(rhs, _ir.Expr):
        return python_operation(lhs, rhs)
    if _ir.is_prim_expr(lhs) or _ir.is_prim_expr(rhs):
        return primitive(lhs, rhs)
    return operation(_value(lhs), _value(rhs))


def logical_and(*values):
    """Construct conjunction of host, primitive, or tensor values.

    Parameters
    ----------
    values : Expr or Python value
        One or more eagerly evaluated operands. Host pairs follow Python
        logical operations; primitive pairs use primitive IR and tensor pairs
        use the corresponding Relax operation.

    Returns
    -------
    result : Expr or Python value
        The conjunction reduced from left to right.

    Notes
    -----
    All arguments are evaluated before this call; it does not provide Python
    short-circuit evaluation of the argument expressions.
    """
    if not values:
        raise TypeError("logical_and requires at least one operand")
    result = values[0]
    for value in values[1:]:
        result = _logical_pair(result, value, _relax.op.logical_and, _tir.And, lambda a, b: a and b)
    return result


def logical_or(*values):
    """Construct disjunction of host, primitive, or tensor values.

    Parameters
    ----------
    values : Expr or Python value
        One or more eagerly evaluated operands. Host pairs follow Python
        logical operations; primitive pairs use primitive IR and tensor pairs
        use the corresponding Relax operation.

    Returns
    -------
    result : Expr or Python value
        The disjunction reduced from left to right.

    Notes
    -----
    All arguments are evaluated before this call; it does not provide Python
    short-circuit evaluation of the argument expressions.
    """
    if not values:
        raise TypeError("logical_or requires at least one operand")
    result = values[0]
    for value in values[1:]:
        result = _logical_pair(result, value, _relax.op.logical_or, _tir.Or, lambda a, b: a or b)
    return result


def logical_not(value):
    """Negate a host, primitive, or tensor condition.

    Parameters
    ----------
    value : Expr or Python value
        Operand to negate. Primitive and tensor expressions use their IR
        logical operation; host values use Python truth testing.

    Returns
    -------
    result : Expr or bool
        The logical negation without testing an IR expression as a Python bool.
    """
    if _ir.is_prim_expr(value):
        return _tir.Not(value)
    if isinstance(value, _ir.Expr):
        return _relax.op.logical_not(value)
    return not value


def select(condition, true_value, false_value):
    """Select between already-constructed values.

    Parameters
    ----------
    condition : Expr or Python value
        Primitive condition, tensor condition, or host truth value.
    true_value : Expr or Python value
        Value selected when the condition is true.
    false_value : Expr or Python value
        Value selected when the condition is false.

    Returns
    -------
    result : Expr or Python value
        A primitive Select, Relax elementwise where, or the selected host value.

    Notes
    -----
    Both value arguments are evaluated before this call. Host selection returns
    the selected object; tensor selection converts Python numbers and tuples.
    """
    if _ir.is_prim_expr(condition):
        return _tir.Select(condition, true_value, false_value)
    if isinstance(condition, _ir.Expr):
        return _relax.op.where(condition, _value(true_value), _value(false_value))
    return true_value if condition else false_value


def if_then_else_(condition: Any, true_value: Any, false_value: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.if_then_else_`."""
    if isinstance(condition, _ffi.ObjectConvertible):
        condition = condition.asobject()
    if not isinstance(condition, _ir.Expr):
        return true_value if condition else false_value
    true_value = (
        true_value.asobject() if isinstance(true_value, _ffi.ObjectConvertible) else true_value
    )
    false_value = (
        false_value.asobject() if isinstance(false_value, _ffi.ObjectConvertible) else false_value
    )
    if _ir.is_prim_expr(condition) and all(
        _ir.is_prim_expr(value)
        if isinstance(value, _ir.Expr)
        else isinstance(value, _numbers.Number)
        for value in (true_value, false_value)
    ):
        return _tir.if_then_else(condition, true_value, false_value)
    return _relax.If(condition, _value(true_value), _value(false_value))


def and_(*values: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.and_`."""
    return logical_and(*values)


def or_(*values: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.or_`."""
    return logical_or(*values)


def not_(value: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.not_`."""
    return logical_not(value)


def lt_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.lt_`."""
    if any(isinstance(value, _ir.Expr) and not _ir.is_prim_expr(value) for value in (lhs, rhs)):
        lhs = _relax.const(lhs) if isinstance(lhs, _numbers.Number) else lhs
        rhs = _relax.const(rhs) if isinstance(rhs, _numbers.Number) else rhs
        return _base.at_(span, _relax.op.less(lhs, rhs))
    return _prim_ffi._OpLT(lhs, rhs, span.span if isinstance(span, _base.SpanEntry) else span)


def le_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.le_`."""
    if any(isinstance(value, _ir.Expr) and not _ir.is_prim_expr(value) for value in (lhs, rhs)):
        lhs = _relax.const(lhs) if isinstance(lhs, _numbers.Number) else lhs
        rhs = _relax.const(rhs) if isinstance(rhs, _numbers.Number) else rhs
        return _base.at_(span, _relax.op.less_equal(lhs, rhs))
    return _prim_ffi._OpLE(lhs, rhs, span.span if isinstance(span, _base.SpanEntry) else span)


def gt_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.gt_`."""
    if any(isinstance(value, _ir.Expr) and not _ir.is_prim_expr(value) for value in (lhs, rhs)):
        lhs = _relax.const(lhs) if isinstance(lhs, _numbers.Number) else lhs
        rhs = _relax.const(rhs) if isinstance(rhs, _numbers.Number) else rhs
        return _base.at_(span, _relax.op.greater(lhs, rhs))
    return _prim_ffi._OpGT(lhs, rhs, span.span if isinstance(span, _base.SpanEntry) else span)


def ge_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.ge_`."""
    if any(isinstance(value, _ir.Expr) and not _ir.is_prim_expr(value) for value in (lhs, rhs)):
        lhs = _relax.const(lhs) if isinstance(lhs, _numbers.Number) else lhs
        rhs = _relax.const(rhs) if isinstance(rhs, _numbers.Number) else rhs
        return _base.at_(span, _relax.op.greater_equal(lhs, rhs))
    return _prim_ffi._OpGE(lhs, rhs, span.span if isinstance(span, _base.SpanEntry) else span)


def eq_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.eq_`."""
    if any(isinstance(value, _ir.Expr) and not _ir.is_prim_expr(value) for value in (lhs, rhs)):
        lhs = _relax.const(lhs) if isinstance(lhs, _numbers.Number) else lhs
        rhs = _relax.const(rhs) if isinstance(rhs, _numbers.Number) else rhs
        return _base.at_(span, _relax.op.equal(lhs, rhs))
    return _prim_ffi._OpEQ(lhs, rhs, span.span if isinstance(span, _base.SpanEntry) else span)


def ne_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.ne_`."""
    if any(isinstance(value, _ir.Expr) and not _ir.is_prim_expr(value) for value in (lhs, rhs)):
        lhs = _relax.const(lhs) if isinstance(lhs, _numbers.Number) else lhs
        rhs = _relax.const(rhs) if isinstance(rhs, _numbers.Number) else rhs
        return _base.at_(span, _relax.op.not_equal(lhs, rhs))
    return _prim_ffi._OpNE(lhs, rhs, span.span if isinstance(span, _base.SpanEntry) else span)


invoke_closure = _ty_arg_wrapper(invoke_closure)
call_builtin_with_ctx = _ty_arg_wrapper(call_builtin_with_ctx)

__all__ = [
    "abs",
    "acos",
    "acosh",
    "add",
    "and_",
    "arange",
    "argmax",
    "argmin",
    "argsort",
    "asin",
    "asinh",
    "assert_op",
    "astype",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "broadcast_to",
    "bucketize",
    "builtin",
    "call_builtin_with_ctx",
    "call_dps_packed",
    "call_inplace_packed",
    "call_packed",
    "call_pure_packed",
    "call_py_func",
    "call_tir",
    "call_tir_inplace",
    "call_tir_with_grad",
    "ccl",
    "ceil",
    "clip",
    "collapse_sum_like",
    "collapse_sum_to",
    "concat",
    "cos",
    "cosh",
    "cumprod",
    "cumsum",
    "dequantize",
    "divide",
    "dynamic_strided_slice",
    "einsum",
    "emit_with_type",
    "eq_",
    "equal",
    "erf",
    "ewise_fma",
    "exp",
    "expand_dims",
    "eye",
    "eye_like",
    "flatten",
    "flip",
    "floor",
    "floor_divide",
    "floor_mod",
    "full",
    "full_like",
    "gather_elements",
    "gather_nd",
    "ge_",
    "grad",
    "greater",
    "greater_equal",
    "gt_",
    "hamming_window",
    "hint_on_device",
    "if_then_else_",
    "image",
    "index_put",
    "index_tensor",
    "invoke_closure",
    "invoke_pure_closure",
    "isfinite",
    "isinf",
    "isnan",
    "layout_transform",
    "le_",
    "left_shift",
    "less",
    "less_equal",
    "linear",
    "log",
    "log_add_exp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "lt_",
    "make_closure",
    "matmul",
    "max",
    "maximum",
    "mean",
    "median",
    "memory",
    "meshgrid",
    "min",
    "minimum",
    "mod",
    "multinomial_from_uniform",
    "multiply",
    "ne_",
    "negative",
    "nn",
    "nonzero",
    "not_",
    "not_equal",
    "null_value",
    "one_hot",
    "ones",
    "ones_like",
    "or_",
    "outer",
    "permute_dims",
    "power",
    "print",
    "prod",
    "quantize",
    "repeat",
    "reshape",
    "reverse_sequence",
    "right_shift",
    "round",
    "rsqrt",
    "scatter_elements",
    "scatter_nd",
    "select",
    "shape_of",
    "shape_to_tensor",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "size",
    "slice_scatter",
    "sort",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "stop_lift_params",
    "strided_slice",
    "subtract",
    "sum",
    "take",
    "tan",
    "tanh",
    "tensor_to_shape",
    "tile",
    "to_vdevice",
    "topk",
    "tril",
    "triu",
    "trunc",
    "unique",
    "variance",
    "vision",
    "vm",
    "where",
    "wrap_param",
    "zeros",
    "zeros_like",
]
