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
"""Relax implementation of the shared source-to-builder protocol.

Hooks delegate construction to this language variant's native builder and IR APIs.
For example, generated ``X.if_(condition)`` creates the native conditional frame;
``X.then_()`` and ``X.else_()`` enter its branches. See the corresponding shared
``tvm.script.ir_builder.parser_protocol`` hooks for operand and span contracts.
"""

from __future__ import annotations

import builtins as _python
import numbers as _numbers
import re as _re
from collections.abc import Sequence
from typing import Any, NoReturn

import tvm_ffi as _ffi

import tvm.relax.script.ir_builder as _builder
from tvm import ir as _ir
from tvm import relax as _relax
from tvm import tirx as _tir
from tvm.ir.prim import _ffi_api as _prim_ffi
from tvm.script.ir_builder import IRBuilder as _IRBuilder
from tvm.script.ir_builder import base as _base
from tvm.script.ir_builder.frame import IRModuleFrame as _IRModuleFrame

from . import _ffi_api
from . import frame as _frame
from . import ir as _native

_Span = _base.SpanEntry | _ir.Span | None


# --------------------------------------
# Section: control flow
# --------------------------------------
#
# ``with X.if_(cond):`` opens a conditional frame.
# ``with X.then_():`` enters its true branch.
# ``with X.else_():`` enters its false branch.
# ``with X.for_(loop) as i:`` binds a loop iterator.
# ``with X.while_(cond):`` builds a while loop.
# ``X.range_(start, stop)`` constructs a loop descriptor.
# ``X.break_()`` emits a loop exit.
# ``X.continue_()`` emits a loop continuation.


def if_(condition: Any, *, span: _Span = None) -> _frame.IfFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.if_`."""
    return _base.at_(span, _native.If(condition))


def then_(*, span: _Span = None) -> _frame.ThenFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.then_`."""
    return _base.at_(span, _native.Then())


def else_(*, span: _Span = None) -> _frame.ElseFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.else_`."""
    return _base.at_(span, _native.Else())


def for_(
    iterable: Any, *, names: str | Sequence[str] | None = None, span: _Span = None
) -> NoReturn:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.for_`.

    Relax rejects source loops; use supported functional control flow.
    """
    raise TypeError("Relax does not support imperative for loops")


def while_(condition: Any, *, span: _Span = None) -> NoReturn:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.while_`."""
    raise TypeError("Relax does not support imperative while loops")


def range_(*args: Any, annotations: dict[str, Any] | None = None) -> NoReturn:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.range_`."""
    raise TypeError("Relax does not support imperative for loops")


def break_(*, span: _Span = None) -> NoReturn:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.break_`.

    Relax rejects loop-control statements.
    """
    raise TypeError("Relax does not support break")


def continue_(*, span: _Span = None) -> NoReturn:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.continue_`.

    Relax rejects loop-control statements.
    """
    raise TypeError("Relax does not support continue")


# --------------------------------------
# Section: operator overloading
# --------------------------------------
#
# ``X.if_then_else_(c, a, b)`` selects an expression.
# ``X.and_(a, b)`` constructs conjunction.
# ``X.or_(a, b)`` constructs disjunction.
# ``X.not_(a)`` negates a condition.
# ``X.lt_(a, b)`` lowers ``a < b``.
# ``X.le_(a, b)`` lowers ``a <= b``.
# ``X.gt_(a, b)`` lowers ``a > b``.
# ``X.ge_(a, b)`` lowers ``a >= b``.
# ``X.eq_(a, b)`` lowers ``a == b``.
# ``X.ne_(a, b)`` lowers ``a != b``.


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
    return _relax.If(condition, _builder._value(true_value), _builder._value(false_value))


def and_(*values: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.and_`."""
    return _builder.logical_and(*values)


def or_(*values: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.or_`."""
    return _builder.logical_or(*values)


def not_(value: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.not_`."""
    return _builder.logical_not(value)


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


# --------------------------------------
# Section: context lookup and resolution
# --------------------------------------
#
# ``X.resolve_global_info_(key)`` resolves module metadata.
# ``X.resolve_type_var_("n")`` resolves a symbolic dimension.
# ``X.call_global_var_(f, args)`` calls a module function.


def resolve_global_info_(content: Any) -> Any:
    """Resolve a module-owned selector or retain a concrete object.

    Parameters
    ----------
    content : str or Any
        Original global-info selector or concrete value. "mesh[0]" indexes a named list;
        "cuda:1" selects the second CUDA vdevice; "vdevice:0" selects by absolute index.
        A trailing memory-scope suffix is accepted without changing device selection.

    Returns
    -------
    Any
        The exact registered global-info object, or the unchanged non-string input.

    Notes
    -----
    String lookup requires the nearest active native module frame and creates no metadata.
    Missing context, malformed selectors or unmatched devices raise ValueError; missing map
    entries or out-of-range indices propagate KeyError/IndexError. Non-string values require
    no frame. No source span is attached to an existing metadata object.

    .. code:: python

        # The constructor decorator calls this resolver for string selectors.
        R.Tensor((n,), "float32", vdevice="cuda:0")
        # Direct resolution requires the same active module frame.
        device = R.resolve_global_info_("cuda:0")
    """
    if not isinstance(content, str):
        return content
    if not _IRBuilder.is_in_scope():
        raise ValueError("Global-info lookup requires an enclosing module frame")
    for frame in reversed(_IRBuilder.current().frames):
        if isinstance(frame, _IRModuleFrame):
            break
    else:
        raise ValueError("Global-info lookup requires an enclosing module frame")
    match = _re.fullmatch(r"([^\[\]]+)\[(\d+)\]", content)
    if match:
        name, index = match.groups()
        return frame.global_infos[name][int(index)]
    selector = _re.fullmatch(r"([^:\[\]]+)(?::(\d+)(?::([^:]+))?)?", content)
    if selector is None:
        raise ValueError(f"Invalid global-info reference: {content!r}")
    target, index, _scope = selector.groups()
    ordinal = int(index) if index is not None else 0
    devices = frame.global_infos.get("vdevice", ())
    if target == "vdevice":
        return devices[ordinal]
    for device in devices:
        if device.target.kind.name == target:
            if ordinal == 0:
                return device
            ordinal -= 1
    raise ValueError(f"Global-info device reference was not found: {content!r}")


def resolve_type_var_(
    name: str,
    dtype: str | _ir.Type | _ir.Var | None = None,
    *,
    value: _ir.Var | None = None,
    span: _Span = None,
) -> _ir.Var:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.resolve_type_var_`."""
    return _base._current_function_frame().resolve_type_var(name, dtype, value=value, span=span)


def call_global_var_(function: _ir.GlobalVar, args: Sequence[Any]) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.call_global_var_`."""
    return _relax.Call(function, [_relax.utils.convert_to_expr(value) for value in args])


# --------------------------------------
# Section: special protocol
# --------------------------------------
#
# Syntax markers live in tvm.script.parser.protocol_registry.
# ``with X.function_(...) as fn:`` builds a function frame.
# ``a = X.arg("a", ty)`` declares a parameter.
# ``X.func_name("main")`` sets the function name.
# ``X.func_ret_type(ty)`` sets the result annotation.
# ``X.check_well_formed_(result)`` validates completed IR.


def function_(
    pure: bool = True,
    private: bool = False,
    *,
    decl: bool = False,
    local: bool = False,
    reference: _ir.Var | None = None,
    span: _Span = None,
) -> _frame.FunctionFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.function_`.

    Public pure/private options pass to native purity and visibility controls.
    Local function bodies require their previously declared reference.
    """
    if decl:
        return _base.at_(span, _ffi_api.DeclFunction(pure, private, local))
    if local:
        if reference is None:
            raise ValueError("A local function requires its declared reference")
        return _base.at_(span, _ffi_api.LocalFunction(pure, reference))
    return _base.at_(span, _native.function(pure, private))


def arg(name: str, ty: Any, *, span: _Span = None) -> _ir.Var:
    """Declare a Relax parameter using the shared argument-hook contract.

    Parameters
    ----------
    name : str
        Source parameter name.
    ty : Type, Var, or callable
        Parameter annotation; corresponds to ``annotation`` in the shared
        :func:`tvm.script.ir_builder.parser_protocol.arg` contract. Primitive
        annotations resolve through the active signature symbol context; an
        existing variable retains its identity.
    span : SpanEntry, Span or None, optional
        Source location attached to the constructed IR; None leaves it unspecified.

    Returns
    -------
    result : Var
        The parameter registered in the current function frame.
    """
    if not isinstance(ty, _ir.Var):
        ty = _builder._type(ty)
    if isinstance(ty, _ir.PrimType) or _ir.is_prim_var(ty):
        ty = resolve_type_var_(name, ty, span=span)
    if isinstance(ty, _ir.Var):
        return _ffi_api.ArgVar(name, ty)
    return _base.at_(span, _native.arg(name, _builder._type(ty)))


def func_name(name: str) -> None:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.func_name`."""
    return _native.func_name(name)


def func_ret_type(annotation: Any, *, span: _Span = None) -> None:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.func_ret_type`."""
    return _native.func_ret_type(_builder._type(_base._return_annotation(annotation)))


def check_well_formed_(function: _relax.Function) -> None:
    """Validate a completed Relax function.

    Parameters
    ----------
    function : tvm.relax.Function
        Completed function to validate, without changing its IR.

    Raises
    ------
    ValueError
        If the function fails language variant validation.

    Notes
    -----
    See :func:`tvm.script.ir_builder.parser_protocol.check_well_formed_`
    for the shared whole-module validation coordinator.

    Validate the completed Relax function. Whole-module cross-function
    validation is registered separately with the root coordinator.
    """
    from tvm import s_tir

    message = (
        "Program is not well-formed. If this is deliberate, set "
        "check_well_formed=False in the top-level decorator."
    )
    if not _relax.analysis.check_well_formed(function):
        raise ValueError(message)
    try:
        s_tir.analysis.verify_well_formed(_ir.IRModule.from_expr(function))
    except Exception as error:
        raise ValueError(f"{message}\n{error}") from error


def _check_module_well_formed(module: _ir.IRModule) -> None:
    """Own Relax whole-module validation, including cross-function references."""
    if not _relax.analysis.check_well_formed(module):
        raise ValueError(
            "Program is not well-formed. If this is deliberate, set "
            "check_well_formed=False in the top-level decorator."
        )


# --------------------------------------
# Section: binding
# --------------------------------------
#
# ``a = X.bind_(value, name="a")`` binds a source name.
# ``X.decl_mutable_cell_(value, ty=ty)`` declares mutable storage.
# ``X.set_mutable_cell_(cell, value)`` updates mutable storage.
# ``a, b = X.unpack(value)`` destructures a binding.


def bind_(
    value: Any = _base.MISSING,
    *,
    ty: Any = None,
    name: str | None = None,
    span: _Span = None,
    value_span: _Span = None,
    name_span: _Span = None,
    frame_value: bool = False,
) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.bind_`.

    Relax values emit a binding or MatchCast; value_span belongs to the actual
    RHS and span to the binding target. Metadata passes through unchanged.
    """
    if isinstance(span, _base.SpanEntry):
        span = span.span
    name_span = span if name_span is None else name_span
    if isinstance(name_span, _base.SpanEntry):
        name_span = name_span.span
    if frame_value:
        if isinstance(value, _python.list | _python.tuple | _ir.Array):
            for index, item in enumerate(value):
                bind_(
                    item,
                    name=None if name is None else f"{name}_{index}",
                    span=span,
                    name_span=name_span,
                    frame_value=True,
                )
        elif isinstance(value, _ir.Var):
            if name is not None:
                _IRBuilder.name(name, value)
            _base.at_(name_span if name_span is not None else span, value)
        return value
    if value is _base.MISSING:
        raise ValueError("Relax bindings require an initializer")
    ty = None if ty is None else _builder._type(ty)
    if isinstance(value, _base.AlreadyEmitted):
        return _base.at_(value_span, value)
    value = _builder._value(value, ty)
    if isinstance(value, _relax.MatchCast):
        _base.at_(value_span, value.value)
        if ty is not None and not _ffi.structural_equal(ty, value.ty):
            raise TypeError("The binding annotation differs from the match-cast type")
        result = _ffi_api.EmitMatchCastWithSpan(value.value, value.ty, name_span, span)
    elif isinstance(value, _relax.Expr):
        _base.at_(value_span, value)
        result = _ffi_api.EmitWithSpan(value, ty, name_span, span)
    else:
        return value
    if name is not None:
        _IRBuilder.name(name, result)
    return _base.at_(name_span if name_span is not None else span, result)


def decl_mutable_cell_(
    value: Any = _base.MISSING,
    *,
    ty: Any = None,
    name: str | None = None,
    span: _Span = None,
    name_span: _Span = None,
) -> NoReturn:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.decl_mutable_cell_`.

    Relax has immutable bindings and rejects mutable storage declarations.
    """
    raise TypeError("Relax does not support mutable local storage")


def set_mutable_cell_(target: Any, value: Any, *, span: _Span = None) -> NoReturn:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.set_mutable_cell_`.

    Relax has immutable bindings and rejects mutable storage updates.
    """
    raise TypeError("Relax does not support mutable local storage")


def unpack(value: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.unpack`."""
    if isinstance(value, _relax.Tuple):
        return _python.tuple(value.fields)
    if isinstance(value, _relax.Expr) and isinstance(value.ty, _ir.TupleType):
        return _python.tuple(_relax.TupleGetItem(value, i) for i in range(len(value.ty.fields)))
    return value


# --------------------------------------
# Section: statement
# --------------------------------------
#
# ``X.emit_(value)`` emits an expression statement.
# ``X.return_(value)`` emits a function return.
# ``X.setitem_(a, i, value)`` emits an indexed store.
# ``X.setattr_(a, "field", value)`` emits an attribute store.
# ``X.assert_(condition, message)`` emits an assertion.


def emit_(value: Any, *, span: _Span = None) -> None:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.emit_`.

    Only void expressions may be discarded. Receipts and None emit nothing;
    non-void expressions raise ValueError and unsupported host values raise TypeError.
    """
    if isinstance(value, _base.AlreadyEmitted):
        _base.at_(span, value)
        return None
    if value is None:
        return
    if not isinstance(value, _relax.Expr):
        raise TypeError(f"Unsupported expression statement value: {type(value).__name__}")
    result = bind_(_base.at_(span, value), name="_", span=span)
    if not isinstance(result.ty, _ir.TupleType) or len(result.ty.fields) != 0:
        raise ValueError(
            "Non-void expressions must be bound to a variable; "
            f"expression of type {result.ty} was used as a statement"
        )


def return_(value: Any = None, *, span: _Span = None) -> None:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.return_`."""
    if value is None:
        value = _relax.Tuple([])
    # Normalization may emit bindings, but an existing result keeps its own span.
    _base.with_at_group_(span, lambda: _native.func_ret_value(_builder._value(value)))


def setitem_(target: Any, key: Any, value: Any, *, span: _Span = None) -> NoReturn:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.setitem_`."""
    raise TypeError("Relax does not support indexed assignment")


def setattr_(target: Any, name: str, value: Any, *, span: _Span = None) -> NoReturn:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.setattr_`."""
    raise TypeError("Relax does not support attribute assignment")


def assert_(
    condition: Any,
    message: str | tuple[str, Sequence[Any]] | Sequence[Any] = "",
    *,
    span: _Span = None,
) -> None:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.assert_`."""
    if not isinstance(message, _python.str):
        raise TypeError("An assertion message must be construction-time text")
    emit_(_base.at_(span, _native.assert_op(condition, format=message)), span=span)
