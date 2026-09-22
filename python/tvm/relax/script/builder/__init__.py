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

import tvm_ffi as _ffi

from tvm import ir as _ir
from tvm import relax as _relax
from tvm import tirx as _tir
from tvm.relax.distributed import DeviceMesh as _DeviceMesh
from tvm.relax.distributed import DTensorType as _DTensorType
from tvm.relax.distributed import Placement as _Placement
from tvm.relax.distributed import device_mesh as device_mesh
from tvm.script.ir_builder import IRBuilder as _IRBuilder
from tvm.script.ir_builder import ir as _I
from tvm.script.ir_builder.base import BypassBind as _BypassBind
from tvm.script.ir_builder.base import _construction_span
from tvm.script.ir_builder.base import _frame_result as _named_frame_result
from tvm.script.ir_builder.base import at as _at
from tvm.script.ir_builder.base import source_span as _source_span
from tvm.script.ir_builder.parser_support import lookup_global_info as _lookup_global_info
from tvm.script.ir_builder.type_var_frame import TypeVarDecl as _TypeVarDecl
from tvm.script.ir_builder.type_var_frame import resolve_type_var
from tvm.script.parser.protocol import args_policy as _args_policy
from tvm.script.parser.protocol import constexpr as constexpr
from tvm.script.parser.protocol import expr_str_args as _expr_str_args

from . import _ffi_api
from . import distributed as dist
from . import frame as _frame
from . import ir as _native
from .comparison import eq as eq
from .comparison import ge as ge
from .comparison import gt as gt
from .comparison import le as le
from .comparison import lt as lt
from .comparison import ne as ne
from .distributed.ir import _lookup_device_mesh
from .ir import *
from .protocol import bind_ as bind_
from .protocol import emit_ as emit_


@_args_policy({"shape": "expr_str", "vdevice": "global_info"}, scalar_strings=False)
def Tensor(shape=None, dtype=None, vdevice=None, ndim=-1, *, span=None):
    """Construct a tensor type from concrete shape dimensions."""
    if isinstance(shape, _python.str) and dtype is None:
        dtype, shape = shape, None
    if isinstance(vdevice, _python.str) and not _IRBuilder.is_in_scope():
        # Python evaluates annotations before the module decorator opens its
        # frame. The generated declaration resolves this module-owned reference.
        return _ir.Type.missing()
    vdevice = _lookup_global_info(vdevice)
    return _relax.TensorType(shape, dtype, vdevice, ndim, _source_span(span))


@_args_policy({"shape": "expr_str", "device_mesh": "global_info"}, scalar_strings=False)
def DTensor(shape=None, dtype=None, device_mesh=None, placement="", *, ndim=-1, span=None):
    """Construct a distributed tensor type from concrete dimensions."""
    if isinstance(device_mesh, _python.str) and not _IRBuilder.is_in_scope():
        return _ir.Type.missing()
    if device_mesh is None:
        device_mesh = _DeviceMesh([], _ir.Range(0, 1))
    else:
        device_mesh = _lookup_global_info(device_mesh)
    if isinstance(placement, _python.str):
        placement = _Placement.from_text(placement)
    return _DTensorType(Tensor(shape, dtype, ndim=ndim), device_mesh, placement, _source_span(span))


# The distributed source spelling shares concrete constructors and argument policy.
dist.DTensor = DTensor
dist.device_mesh = device_mesh

Range = _ir.Range


@_expr_str_args("values", introduce=True, dtype="int64")
def Shape(values=None, ndim=-1, *, span=None):
    """Construct a shape type from concrete dimensions."""
    return _relax.ShapeType(values, ndim, _source_span(span))


def _type(value):
    if value is None:
        return _ir.TupleType([])
    if callable(value):
        value = value()
    if _ir.is_prim_expr(value) or isinstance(value, _TypeVarDecl):
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


def Prim(dtype, *, span=None):
    """Construct a primitive scalar type."""
    return _ir.PrimType(dtype)


Prim.__tvm_parameter_dtype__ = "dtype"


def Object(*, span=None):
    """Construct the unconstrained Relax value type."""
    return _relax.AnyType(_source_span(span))


Any = Object


is_type_var = _ir.is_prim_var


def type_var(name, *, dtype=None, span=None):
    """Construct a fresh standalone primitive symbol."""
    return _ir.Var(name, "int64" if dtype is None else dtype, _source_span(span))


class _Frame:
    """Retain source metadata and exports around an existing native frame."""

    def __init__(self, native, span=None):
        # Each wrapper owns one native construction frame and source location.
        # result starts empty, records finalized lexical exports on exit, and
        # never outlives its construction region or stores another function's
        # symbols (those belong to the separate TypeVarFrame).
        self.native = native
        self.span = span
        self.result = {}

    def __getattr__(self, name):
        return getattr(self.native, name)

    @property
    def reference(self):
        """Return the stable module or local function reference after declaration."""
        if isinstance(self.native, _frame.FunctionFrame):
            local_var = self.native.local_var
            return local_var if local_var is not None else self.native.global_var
        raise AttributeError("This frame does not declare a function")

    def _set_source_span(self, span):
        _at(span, self.native)

    def __enter__(self):
        with _construction_span(self.span):
            self.native.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with _construction_span(self.span):
            self.native.__exit__(exc_type, exc_value, traceback)
        if exc_type is None:
            if isinstance(self.native, _frame.BindingBlockFrame):
                self.result = {var.name: var for var in self.native.output_vars}
            elif (
                isinstance(self.native, _frame.FunctionFrame) and self.native.local_var is not None
            ):
                self.result = {self.native.name: self.native.local_var}
            elif isinstance(self.native, _frame.IfFrame):
                self.result = {self.native.var_name: self.native.var}
        return False


def frame_result(completed_frame, name):
    """Read one named export from an explicitly supplied completed region."""
    return _named_frame_result(completed_frame, name)


def function(is_pure=True, is_private=False, *, local=False, reference=None, span=None):
    """Start a function frame.

    Parameters
    ----------
    is_pure: bool
        Whether the function is annotated as pure.

    is_private : bool
        Whether the function is annotated as private.

    local : bool, optional
        Whether to define a local function instead of a module function.
    reference : Var, optional
        The declared function variable required when local is True.
    span : Span or source location, optional
        Source location attached to the function frame.

    Returns
    -------
    frame : context manager
        Construction context for the native frame, retaining source metadata.
    """
    if local:
        if reference is None:
            raise ValueError("A local function requires its declared reference")
        return _Frame(_ffi_api.LocalFunction(is_pure, reference), span)
    return _Frame(_native.function(is_pure, is_private), span)


def decl_function(is_pure=True, is_private=False, *, local=False, span=None):
    """Create a bodyless Relax function declaration context."""
    return _Frame(_ffi_api.DeclFunction(is_pure, is_private, local), span)


def arg(name, ty, *, span=None):
    """Add a parameter to the last function frame.

    Parameters
    ----------
    name: str
        The name of the parameter.
    ty : Type, Var or callable
        The parameter type or an existing variable. An existing variable
        retains its native identity; callable type annotations are resolved
        before creating the parameter.

    span : Span or source location, optional
        Source location attached to the constructed IR.

    Returns
    -------
    var: Var
        The created or retained function parameter variable.
    """
    with _construction_span(span):
        if isinstance(ty, _ir.Var):
            return _ffi_api.ArgVar(name, ty)
        return _at(span, _native.arg(name, _type(ty)))


def func_ret_type(ret_ty):
    """Set the active function signature return type."""
    return _native.func_ret_type(_type(ret_ty))


func_ret_ty = func_ret_type


def dataflow(*, span=None):
    """Create a dataflow context with explicit finalized exports."""
    return _Frame(_native.dataflow(), span)


def If(condition, *, span=None):
    """Create an if frame.

    Parameters
    ----------
    condition : Expr

        The condition of if statement, executes the true branch if the
        condition is true, otherwise jump into the false branch.

    span : Span or source location, optional
        Source location attached to the constructed IR.

    Returns
    -------
    frame : context manager
        Construction context for the native frame, retaining source metadata.
    """
    return _Frame(_native.If(condition), span)


def Then(*, span=None):
    """Create the true branch of the active conditional."""
    return _Frame(_native.Then(), span)


def Else(*, span=None):
    """Create the false branch of the active conditional."""
    return _Frame(_native.Else(), span)


def _value(value, ty=None):
    if isinstance(value, _python.tuple):
        return _relax.utils.convert_to_expr(value)
    if isinstance(value, _numbers.Number):
        if isinstance(ty, _ir.PrimType):
            return _relax.prim_value(value, dtype=ty.dtype)
        return _relax.const(value)
    return value


def return_(value=None, *, span=None):
    """Record a function result without exiting Python construction."""
    with _construction_span(span):
        if value is None:
            value = _relax.Tuple([])
        _native.func_ret_value(_value(value))


def match_cast(value, ty, *, span=None):
    """Construct a match-cast descriptor for bind_ to consume."""
    if value is None:
        raise ValueError("The match-cast value cannot be None")
    ty = _type(ty)
    return _relax.MatchCast(_ir.Var("", ty), _value(value), ty, _source_span(span))


def unpack(value):
    """Project an IR tuple with known arity or preserve host iteration."""
    if isinstance(value, _BypassBind):
        return _python.tuple(_BypassBind(item) for item in unpack(value.value))
    if isinstance(value, _relax.Tuple):
        return _python.tuple(value.fields)
    if isinstance(value, _relax.Expr) and isinstance(value.ty, _ir.TupleType):
        return _python.tuple(_relax.TupleGetItem(value, i) for i in range(len(value.ty.fields)))
    return value


def assert_(condition, message="", *, span=None):
    """Emit a runtime assertion with construction-time diagnostic text."""
    if not isinstance(message, _python.str):
        raise TypeError("An assertion message must be construction-time text")
    with _construction_span(span):
        emit_(_at(span, _native.assert_op(condition, format=message)), span=span)


def For(*args, span=None, **kwargs):
    """Reject imperative for loops in the Relax expression dialect."""
    raise TypeError("Relax does not support imperative for loops")


def break_(*, span=None):
    """Reject break in the Relax expression dialect."""
    raise TypeError("Relax does not support break")


def continue_(*, span=None):
    """Reject continue in the Relax expression dialect."""
    raise TypeError("Relax does not support continue")


def setitem(target, index, value, *, span=None):
    """Reject indexed assignment in the Relax expression dialect."""
    raise TypeError("Relax does not support indexed assignment")


__all__ = [
    *_native.__all__,
    "Any",
    "Callable",
    "DTensor",
    "For",
    "for_",
    "frame_result",
    "resolve_type_var",
    "Object",
    "Prim",
    "Range",
    "Shape",
    "Tensor",
    "Tuple",
    "assert_",
    "break_",
    "continue_",
    "bind_",
    "decl_function",
    "device_mesh",
    "dist",
    "emit_",
    "eq",
    "ge",
    "gt",
    "le",
    "lt",
    "ne",
    "is_type_var",
    "match_cast",
    "return_",
    "setitem",
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


for_ = For


def if_then_else_(condition, true_value, false_value):
    """Construct scalar conditional evaluation from eagerly constructed operands."""
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


def _chain_binding(variable, value, body):
    if _ir.is_prim_expr(value) and _ir.is_prim_expr(body):
        return _tir.Let(variable, value, body)
    return _relax.SeqExpr([_relax.BindingBlock([_relax.VarBinding(variable, value)])], body)


def and_(*values, chain=None):
    """Construct the dialect's logical conjunction from evaluated values."""
    if chain is not None:
        from tvm.tirx.script.builder.comparison import _comparison_chain

        return _comparison_chain(values, chain, and_, _chain_binding)
    return logical_and(*values)


def or_(*values):
    """Construct the dialect's logical disjunction from evaluated values."""
    return logical_or(*values)


def not_(value):
    """Negate a host or IR boolean without coercing IR to Python bool."""
    return logical_not(value)


__all__ += ["and_", "if_then_else_", "not_", "or_"]
