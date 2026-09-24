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
from tvm.script.ir_builder import IRBuilder as _IRBuilder
from tvm.script.ir_builder.base import at as _at
from tvm.script.ir_builder.base import source_span as _source_span
from tvm.script.parser.protocol_registry import ARGS_POLICIES as _ARGS_POLICIES
from tvm.script.parser.protocol_registry import args_policy as _args_policy
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


@_args_policy("R.Tensor", {"shape": "expr_str", "vdevice": "global_info"}, scalar_strings=False)
def Tensor(shape=None, dtype=None, vdevice=None, ndim=-1, *, span=None):
    """Construct a tensor type from concrete shape dimensions."""
    if isinstance(shape, _python.str) and dtype is None:
        dtype, shape = shape, None
    if isinstance(vdevice, _python.str) and not _IRBuilder.is_in_scope():
        # Python evaluates annotations before the module decorator opens its
        # frame. The generated declaration resolves this module-owned reference.
        return _ir.Type.missing()
    vdevice = resolve_global_info_(vdevice)
    return _relax.TensorType(shape, dtype, vdevice, ndim, _source_span(span))


@_args_policy(
    "R.DTensor", {"shape": "expr_str", "device_mesh": "global_info"}, scalar_strings=False
)
def DTensor(shape=None, dtype=None, device_mesh=None, placement="", *, ndim=-1, span=None):
    """Construct a distributed tensor type from concrete dimensions."""
    if isinstance(device_mesh, _python.str) and not _IRBuilder.is_in_scope():
        return _ir.Type.missing()
    if device_mesh is None:
        device_mesh = _DeviceMesh([], _ir.Range(0, 1))
    else:
        device_mesh = resolve_global_info_(device_mesh)
    if isinstance(placement, _python.str):
        placement = _Placement.from_text(placement)
    return _DTensorType(Tensor(shape, dtype, ndim=ndim), device_mesh, placement, _source_span(span))


# The distributed source spelling shares concrete constructors and argument policy.
dist.DTensor = DTensor
_ARGS_POLICIES["R.dist.DTensor"] = _ARGS_POLICIES["R.DTensor"]
dist.device_mesh = device_mesh

Range = _ir.Range

# Syntax metadata: Relax statements return one same-named branch output.
__tvm_value_if__ = True


@_args_policy("R.Shape", {"values": "expr_str"}, dtype="int64")
def Shape(values=None, ndim=-1, *, span=None):
    """Construct a shape type from concrete dimensions."""
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
    """Construct a concrete or opaque Relax function type."""
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
    """Construct a tuple type while preserving missing component types."""
    if len(fields) == 1 and isinstance(fields[0], list | _python.tuple):
        fields = fields[0]
    return _ir.TupleType([_type(field) for field in fields], _source_span(span))


def Object(*, span=None):
    """Construct the unconstrained Relax value type."""
    return _relax.AnyType(_source_span(span))


Any = Object


is_type_var = _ir.is_prim_var


def type_var(name, *, dtype=None, span=None):
    """Construct a fresh standalone primitive symbol."""
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
    """Construct a match-cast descriptor for ``bind_`` to consume."""
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
    """Construct conjunction of concrete host, primitive, or tensor values."""
    if not values:
        raise TypeError("logical_and requires at least one operand")
    result = values[0]
    for value in values[1:]:
        result = _logical_pair(result, value, _relax.op.logical_and, _tir.And, lambda a, b: a and b)
    return result


def logical_or(*values):
    """Construct disjunction of concrete host, primitive, or tensor values."""
    if not values:
        raise TypeError("logical_or requires at least one operand")
    result = values[0]
    for value in values[1:]:
        result = _logical_pair(result, value, _relax.op.logical_or, _tir.Or, lambda a, b: a or b)
    return result


def logical_not(value):
    """Negate a host or IR boolean without coercing IR to Python bool."""
    if _ir.is_prim_expr(value):
        return _tir.Not(value)
    if isinstance(value, _ir.Expr):
        return _relax.op.logical_not(value)
    return not value


def select(condition, true_value, false_value):
    """Construct elementwise selection or select already-built host values."""
    if _ir.is_prim_expr(condition):
        return _tir.Select(condition, true_value, false_value)
    if isinstance(condition, _ir.Expr):
        return _relax.op.where(condition, _value(true_value), _value(false_value))
    return true_value if condition else false_value


__all__ += ["logical_and", "logical_not", "logical_or", "select"]


__all__ += ["and_", "if_then_else_", "not_", "or_"]
