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
"""Concrete Relax construction operations over the shared native builder stack."""

# pylint: disable=wildcard-import,redefined-builtin,invalid-name
import builtins as _python
import numbers as _numbers

from tvm import ir as _ir
from tvm import relax as _relax
from tvm import tirx as _tir
from tvm.relax.distributed import DeviceMesh as _DeviceMesh
from tvm.relax.distributed import DTensorType as _DTensorType
from tvm.relax.distributed import Placement as _Placement
from tvm.relax.distributed import device_mesh as device_mesh
from tvm.script.ir_builder import resolve_global_info_args as _resolve_global_info_args
from tvm.script.ir_builder.base import at as _at
from tvm.script.ir_builder.base import source_span as _source_span
from tvm.script.parser.protocol_registry import constexpr as constexpr

from . import distributed as dist
from . import ir as _native
from .distributed.ir import _lookup_device_mesh
from .ir import *
from .parser_protocol import (
    and_,
    arg,
    assert_,
    bind_,
    break_,
    call_global_var_,
    check_well_formed_,
    continue_,
    decl_mutable_cell_,
    else_,
    emit_,
    eq_,
    for_,
    func_name,
    func_ret_type,
    function_,
    ge_,
    gt_,
    if_,
    if_then_else_,
    le_,
    lt_,
    ne_,
    not_,
    or_,
    range_,
    resolve_global_info_,
    resolve_type_var_,
    return_,
    set_mutable_cell_,
    setattr_,
    setitem_,
    then_,
    unpack,
    while_,
)

If = if_
For = for_

# Syntax capability: mutable declaration policies apply only in this dialect.
supports_mutable_declarations = False


@_resolve_global_info_args("vdevice", resolver=resolve_global_info_)
def Tensor(shape=None, dtype=None, vdevice=None, ndim=-1, *, span=None):
    """Construct a Relax tensor type.

    Parameters
    ----------
    shape : Expr or sequence of Expr, optional
        Tensor shape, or None when unknown. A string supplied without dtype
        is shorthand for the dtype. Symbolic dimensions are expressions over
        explicit variables, such as those created with I.dynamic.
    dtype : str or PrimType, optional
        Element type; None leaves the element type unknown.
    vdevice : VDevice or str, optional
        Concrete virtual device or a module metadata selector, such as "cuda:0".
        None leaves the virtual device unspecified. Strings require an active module
        builder; use a quoted whole annotation or postponed annotations when defining
        a Python function before its module builder opens.
    ndim : int, optional
        Rank when shape is unknown; -1 means unknown rank. Do not supply
        an explicit rank together with a known shape.
    span : Span or source location, optional
        Source location attached to the constructed IR; None leaves it unspecified.

    Returns
    -------
    result : TensorType
        The constructed tensor type.
        String selectors outside an active module always raise ValueError.
    """
    if isinstance(shape, _python.str) and dtype is None:
        dtype, shape = shape, None
    return _relax.TensorType(shape, dtype, vdevice, ndim, _source_span(span))


@_resolve_global_info_args("device_mesh", resolver=resolve_global_info_)
def DTensor(shape=None, dtype=None, device_mesh=None, placement="", *, ndim=-1, span=None):
    """Construct a Relax distributed tensor type.

    Parameters
    ----------
    shape : Expr or sequence of Expr, optional
        Global tensor shape, or None when unknown. Symbolic dimensions are
        expressions over explicit variables, such as those created with I.dynamic.
    dtype : str or PrimType, optional
        Element type; None leaves the element type unknown.
    device_mesh : DeviceMesh or str, optional
        Concrete mesh or module metadata selector. None creates an empty mesh
        placeholder. A string selector requires an active module builder; use a quoted
        whole annotation or postponed annotations before the builder opens.
    placement : Placement or str, optional
        Distribution placement. Text, including the default empty string, is
        parsed with Placement.from_text.
    ndim : int, optional
        Global rank when shape is unknown; -1 means unknown rank.
    span : Span or source location, optional
        Source location attached to the constructed IR; None leaves it unspecified.

    Returns
    -------
    result : DTensorType
        The constructed distributed type.
        String selectors outside an active module always raise ValueError.
    """
    if device_mesh is None:
        device_mesh = _DeviceMesh([], _ir.Range(0, 1))
    if isinstance(placement, _python.str):
        placement = _Placement.from_text(placement)
    return _DTensorType(Tensor(shape, dtype, ndim=ndim), device_mesh, placement, _source_span(span))


# The distributed source spelling shares the decorated concrete constructor.
dist.DTensor = DTensor
dist.device_mesh = device_mesh

Range = _ir.Range

# Syntax metadata: Relax statements return one same-named branch output.
__tvm_value_if__ = True


def Shape(values=None, ndim=-1, *, span=None):
    """Construct a Relax shape type.

    Parameters
    ----------
    values : sequence of Expr, optional
        Known dimensions, or None for an unknown shape value. Symbolic dimensions
        are expressions over explicit variables, such as those created with I.dynamic.
    ndim : int, optional
        Number of dimensions when values is None; -1 leaves it unknown.
        Do not supply an explicit count together with known values.
    span : Span or source location, optional
        Source location attached to the constructed IR; None leaves it unspecified.

    Returns
    -------
    result : ShapeType
        The constructed shape type.
    """
    return _relax.ShapeType(values, ndim, _source_span(span))


def _type(value):
    if value is None:
        return _ir.TupleType([])
    if callable(value):
        value = value()
    if _ir.is_prim_expr(value):
        value = value.ty
    if not isinstance(value, _ir.Type):
        raise TypeError(f"Expected a concrete type, got {type(value).__name__}")
    return value


def Callable(params=None, ret=None, purity=None, derive_func=None, *, span=None):
    """Construct a concrete or opaque Relax function type.

    Parameters
    ----------
    params : Type, callable, or sequence of annotations, optional
        Parameter annotations. A single annotation is accepted; None creates
        an opaque callable with an unspecified parameter list.
    ret : Type or callable, optional
        Return annotation. None means an empty tuple for a concrete callable
        and an unspecified result for an opaque callable.
    purity : bool, optional
        Whether the callable is pure. None selects True for a concrete
        parameter list and False for an opaque callable.
    derive_func : str or EnvFunc, optional
        Custom result-type derivation for an opaque callable. It is not
        accepted when params supplies a concrete parameter list.
    span : Span or source location, optional
        Source location attached to the constructed IR; None leaves it unspecified.

    Returns
    -------
    result : FuncType
        The constructed function type.

    Notes
    -----
    Annotations may be types, primitive expressions supplying their types, or
    zero-argument factories returning either. Opaque result and derivation rules
    follow :meth:`tvm.relax.FuncType.opaque_func`.
    """
    if purity is None:
        purity = params is not None
    if params is None:
        return _relax.FuncType.opaque_func(
            ret=None if ret is None else _type(ret),
            derive_func=derive_func,
            purity=purity,
            span=_source_span(span),
        )
    if derive_func is not None:
        raise ValueError("A derivation function requires an opaque callable")
    if not isinstance(params, list | _python.tuple):
        params = [params]
    return _relax.FuncType(
        [_type(param) for param in params], _type(ret), purity, _source_span(span)
    )


def Tuple(*fields, span=None):
    """Construct a Relax tuple type.

    Parameters
    ----------
    fields : Type or callable
        Field annotations as positional arguments, or one list or tuple.
        Each annotation is normalized to a type; None denotes an empty tuple.
    span : Span or source location, optional
        Source location attached to the constructed IR; None leaves it unspecified.

    Returns
    -------
    result : TupleType
        The tuple type with fields in the supplied order.
    """
    if len(fields) == 1 and isinstance(fields[0], list | _python.tuple):
        fields = fields[0]
    return _ir.TupleType([_type(field) for field in fields], _source_span(span))


def Object(*, span=None):
    """Construct the unconstrained Relax value type.

    Parameters
    ----------
    span : Span or source location, optional
        Source location attached to the constructed IR; None leaves it unspecified.

    Returns
    -------
    result : AnyType
        A type accepting any Relax value.
    """
    return _relax.AnyType(_source_span(span))


Any = Object


is_type_var = _ir.is_prim_var


def type_var(name, *, dtype=None, span=None):
    """Construct a fresh standalone primitive symbol.

    Parameters
    ----------
    name : str
        Name of the symbol.
    dtype : str or PrimType, optional
        Primitive type of the symbol; None selects "int64".
    span : Span or source location, optional
        Source location attached to the constructed IR; None leaves it unspecified.

    Returns
    -------
    result : Var
        The newly constructed primitive variable.

    Notes
    -----
    This constructor creates a new symbol on each call. Use the language variant
    resolver for symbols shared by name within a function signature.
    """
    return _ir.Var(name, "int64" if dtype is None else dtype, _source_span(span))


func_ret_ty = func_ret_type


def dataflow(*, span=None):
    """Create a dataflow context with explicit finalized exports.

    Parameters
    ----------
    span : Span or source location, optional
        Source location attached to the constructed IR.

    Returns
    -------
    res : frame.BindingBlockFrame
        The constructed frame, retaining source metadata.
    """
    return _at(span, _native.dataflow())


def _value(value, ty=None):
    if isinstance(value, _python.tuple):
        return _relax.utils.convert_to_expr(value)
    if isinstance(value, _numbers.Number):
        if isinstance(ty, _ir.PrimType):
            return _relax.prim_value(value, dtype=ty.dtype)
        return _relax.const(value)
    return value


def match_cast(value, ty, *, span=None):
    """Construct a match-cast descriptor for the binding hook.

    Parameters
    ----------
    value : Expr or Python value
        Value to match against the asserted type. Numbers and Python tuples
        are converted to Relax expressions; None is not accepted.
    ty : Type or callable
        Asserted type, or a zero-argument factory producing its annotation.
    span : Span or source location, optional
        Source location attached to the constructed IR; None leaves it unspecified.

    Returns
    -------
    result : MatchCast
        An unbound match-cast descriptor, consumed by the language variant
        binding hook to emit and name the result.
    """
    if value is None:
        raise ValueError("The match-cast value cannot be None")
    ty = _type(ty)
    return _relax.MatchCast(_ir.Var("", ty), _value(value), ty, _source_span(span))


__all__ = [
    "then_",
    "else_",
    "while_",
    "setattr_",
    *_native.__all__,
    "Any",
    "Callable",
    "DTensor",
    "For",
    "for_",
    "function_",
    "if_",
    "check_well_formed_",
    "resolve_type_var_",
    "resolve_global_info_",
    "call_global_var_",
    "decl_mutable_cell_",
    "set_mutable_cell_",
    "Object",
    "Range",
    "Shape",
    "Tensor",
    "Tuple",
    "assert_",
    "break_",
    "continue_",
    "bind_",
    "device_mesh",
    "dist",
    "emit_",
    "eq_",
    "ge_",
    "gt_",
    "le_",
    "lt_",
    "ne_",
    "is_type_var",
    "match_cast",
    "return_",
    "setitem_",
    "type_var",
    "unpack",
]


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


__all__ += ["logical_and", "logical_not", "logical_or", "select"]


__all__ += ["and_", "if_then_else_", "not_", "or_"]
