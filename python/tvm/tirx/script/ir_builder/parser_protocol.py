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
"""TIRx implementation of the shared source-to-builder protocol.

Hooks delegate construction to this language variant's native builder and IR APIs.
For example, generated ``X.if_(condition)`` creates the native conditional frame;
``X.then_()`` and ``X.else_()`` enter its branches. See the corresponding shared
``tvm.script.ir_builder.parser_protocol`` hooks for operand and span contracts.
"""

from __future__ import annotations

import builtins as _python
from collections.abc import Sequence
from functools import partial as _partial
from typing import Any

import tvm.tirx.script.ir_builder as _builder
from tvm import ir as _ir
from tvm import tirx as _tir
from tvm.ir.prim import _ffi_api as _prim_ffi
from tvm.script.ir_builder import IRBuilder as _IRBuilder
from tvm.script.ir_builder import base as _base

from . import _ffi_api
from . import frame as _frame
from . import ir as _native

_Span = _base.SpanEntry | _ir.Span | tuple[_ir.SourceName, int, int, int, int] | None


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
) -> _frame.ForFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.for_`.

    A single native loop returns its scalar Var; multiple loops return their
    sequence. frame.vars remains the stable sequence for source unpacking.
    """
    if isinstance(iterable, _python.range):
        iterable = _native.serial(iterable.start, iterable.stop, step=iterable.step)
    if not isinstance(iterable, _frame.ForFrame):
        raise TypeError("A primitive for loop requires an iteration specification")
    iterable.set_names(names)
    return _base.at_(span, iterable)


def while_(condition: Any, *, span: _Span = None) -> _frame.WhileFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.while_`."""
    return _base.at_(span, _native.While(condition))


def range_(*args: Any, annotations: dict[str, Any] | None = None) -> _frame.ForFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.range_`."""
    if len(args) == 1:
        args = (0, args[0], None)
    elif len(args) == 2:
        args = (*args, None)
    elif len(args) != 3:
        raise TypeError("range expects one to three arguments")
    if isinstance(args[2], _python.int) and args[2] == 0:
        raise ValueError("range step cannot be zero")
    return _native.serial(args[0], args[1], step=args[2], annotations=annotations)


def break_(*, span: _Span = None) -> _base.AlreadyEmitted[_tir.Stmt]:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.break_`.

    Legality is checked on the completed function, across loop and function boundaries.
    """
    return _base.at_(span, _native.evaluate(_native.break_loop()))


def continue_(*, span: _Span = None) -> _base.AlreadyEmitted[_tir.Stmt]:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.continue_`.

    Legality is checked on the completed function, across loop and function boundaries.
    """
    return _base.at_(span, _native.evaluate(_native.continue_loop()))


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
    return _builder.select(condition, true_value, false_value)


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
    return _prim_ffi._OpLT(lhs, rhs, _base.source_span(span))


def le_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.le_`."""
    return _prim_ffi._OpLE(lhs, rhs, _base.source_span(span))


def gt_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.gt_`."""
    return _prim_ffi._OpGT(lhs, rhs, _base.source_span(span))


def ge_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.ge_`."""
    return _prim_ffi._OpGE(lhs, rhs, _base.source_span(span))


def eq_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.eq_`."""
    return _prim_ffi._OpEQ(lhs, rhs, _base.source_span(span))


def ne_(lhs: Any, rhs: Any, *, span: _Span = None) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.ne_`."""
    return _prim_ffi._OpNE(lhs, rhs, _base.source_span(span))


# --------------------------------------
# Section: context lookup and resolution
# --------------------------------------
#
# ``X.resolve_global_info_(key)`` resolves module metadata.
# ``X.resolve_type_var_("n")`` resolves a symbolic dimension.
# ``X.call_global_var_(f, args)`` calls a module function.


def resolve_global_info_(content: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.resolve_global_info_`.

    TIRx does not define global-info selectors.
    """
    raise NotImplementedError("TIRx does not support global-info lookup")


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
    return _native._call_global(function, *args)


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
    *,
    private: bool = False,
    s_tir: bool = False,
    persistent: bool = False,
    is_stir: bool | None = None,
    decl: bool = False,
    span: _Span = None,
) -> _frame.PrimFuncFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.function_`.

    Private/persistent options pass to the native TIRx function frame.
    Legacy s_tir/is_stir flags may only be false; S-TIR uses its own
    Ts.prim_func entry. The same frame supports declaration and body entry.
    """
    if s_tir or is_stir:
        raise ValueError("T.prim_func only accepts TIRx; use Ts.prim_func for S-TIR")
    native = (
        _ffi_api.DeclFunction(private, s_tir, persistent)
        if decl
        else _native.prim_func(private=private, s_tir=s_tir, persistent=persistent)
    )
    return _base.at_(span, native)


def arg(name: str, annotation: Any, *, span: _Span = None) -> _ir.Var:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.arg`."""
    if getattr(annotation, "__tvm_optional_annotation__", None) is not None:
        raise TypeError("T.Optional is only supported by @T.jit")
    if callable(annotation) and not isinstance(annotation, _ir.Expr):
        annotation = annotation()
    if isinstance(annotation, _ir.PrimType) or _ir.is_prim_var(annotation):
        annotation = resolve_type_var_(name, annotation, span=span)
    elif isinstance(annotation, _ir.Type):
        annotation = _ir.Var(name, annotation)
    if _tir.is_buffer_var(annotation) and annotation.ty.layout is not None:
        frames = _IRBuilder.current().frames
        if _python.any(isinstance(frame, _frame.PrimFuncFrame) and frame.s_tir for frame in frames):
            ty = annotation.ty
            annotation = _native.buffer(
                ty.shape,
                ty.dtype,
                strides=ty.strides,
                elem_offset=ty.elem_offset,
                scope=ty.storage_scope,
                align=ty.data_alignment,
                offset_factor=ty.offset_factor,
                layout=None,
                allocated_addr=list(ty.allocated_addr),
                buffer_name=name,
            )
    return _native.arg(name, _base.at_(span, annotation))


def func_name(name: str) -> None:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.func_name`."""
    return _native.func_name(name)


def func_ret_type(annotation: Any, *, span: _Span = None) -> None:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.func_ret_type`."""
    annotation = _base._return_annotation(annotation)
    if callable(annotation) and not isinstance(annotation, _ir.Expr | _ir.Type):
        annotation = annotation()
    if isinstance(annotation, _ir.Expr):
        annotation = annotation.ty
    return _native.func_ret(annotation)


def check_well_formed_(function: _tir.PrimFunc) -> None:
    """Validate a completed TIRx function.

    Parameters
    ----------
    function : tvm.tirx.PrimFunc
        Completed function to validate, without changing its IR.

    Raises
    ------
    ValueError
        If the function fails language variant validation.

    Notes
    -----
    See :func:`tvm.script.ir_builder.parser_protocol.check_well_formed_`
    for the shared whole-module validation coordinator.

    The completed native function owns semantic loop and IR validation.
    Module-wide S-TIR and TIRx checks are supplied separately to the root coordinator.
    """
    from tvm import s_tir

    message = (
        "Program is not well-formed. If this is deliberate, set "
        "check_well_formed=False in the top-level decorator."
    )
    try:
        s_tir.analysis.verify_well_formed(_ir.IRModule.from_expr(function))
        if not function.attrs.get("s_tir", False):
            _tir.analysis.verify_tirx_well_formed(function)
    except Exception as error:
        raise ValueError(f"{message}\n{error}") from error


def _check_module_well_formed(module: _ir.IRModule) -> None:
    """Own completed-module S-TIR checks and primitive-function eligibility."""
    from tvm import s_tir

    message = (
        "Program is not well-formed. If this is deliberate, set "
        "check_well_formed=False in the top-level decorator."
    )
    try:
        s_tir.analysis.verify_well_formed(module)
        for function in module.functions.values():
            if isinstance(function, _tir.PrimFunc) and not function.attrs.get("s_tir", False):
                _tir.analysis.verify_tirx_well_formed(function)
    except Exception as error:
        raise ValueError(f"{message}\n{error}") from error


# --------------------------------------
# Section: binding
# --------------------------------------
#
# ``a = X.bind_(value, name="a")`` binds a source name.
# ``X.decl_mutable_cell_(value, ty=ty)`` declares mutable storage.
# ``X.set_mutable_cell_(cell, value)`` updates mutable storage.
# ``a, b = X.unpack(value)`` destructures a binding.


def _name(value: Any, name: str | None, span: _Span) -> Any:
    if name is not None:
        _IRBuilder.name(name, value)
    return _base.at_(span, value)


def _enter_concise(frame: _base.IRBuilderFrame) -> Any:
    # add_callback registers on the active parent before the child enters.
    # Later statements emit into the child; parent exit closes this scope.
    frame.add_callback(_partial(frame.__exit__, None, None, None))
    return frame.__enter__()


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

    Returned Vars, including buffers, and metadata retain identity, names and spans.
    Other expressions create native Bind nodes; value_span belongs to the RHS.
    Explicit typed bindings and frame targets retain their separate contracts.
    """
    name_span = span if name_span is None else name_span
    if frame_value:
        if isinstance(value, _frame.SBlockFrame):
            raise TypeError("A block does not introduce an as-target value")
        if isinstance(value, _python.list | _python.tuple | _ir.Array):
            for index, item in enumerate(value):
                bind_(
                    item,
                    name=None if name is None else f"{name}_{index}",
                    span=span,
                    name_span=name_span,
                    frame_value=True,
                )
        elif isinstance(value, _ir.Var | _tir.Layout):
            _name(value, name, name_span)
        elif isinstance(value, _ir.TensorLoad) and _tir.is_buffer_var(value.source):
            _name(value.source, name, name_span)
        return value
    if isinstance(ty, _native.LetAnnotation):
        if value is _base.MISSING:
            raise ValueError("An immutable binding requires an initializer")
        value = _builder._as_expr(value)
        if not isinstance(value, _ir.Var):
            _base.at_(value_span, value)
        variable = _name(ty.as_var(rhs_dtype=value.ty), name, name_span)
        _base.with_at_group_(span, lambda: _native.Bind(value, var=variable))
        return variable
    if ty is not None:
        annotation = ty() if callable(ty) and not isinstance(ty, _ir.Expr) else ty
        annotation = annotation.ty if isinstance(annotation, _ir.Expr) else annotation
        value = _builder._as_expr(value)
        if not isinstance(value, _ir.Var):
            _base.at_(value_span, value)
        variable = _ir.Var(name or "", annotation)
        return _name(
            _base.with_at_group_(span, lambda: _native.Bind(value, var=variable)), name, name_span
        )
    if value is _base.MISSING:
        raise ValueError("An uninitialized binding requires a scalar type annotation")
    if isinstance(value, _base.AlreadyEmitted):
        return _base.at_(value_span, value)
    if isinstance(value, _base.IRBuilderFrame):
        frame_span = value_span if value_span is not None else span
        return _name(_enter_concise(_base.at_(frame_span, value)), name, name_span)
    # a = existing_var and a = producer() share the same runtime value rule.
    # A Var already owns its declaration, including a newly constructed buffer view.
    if isinstance(value, _ir.Var):
        return value
    if isinstance(value, _ir.TensorRegion):
        return value
    if not isinstance(value, _ir.Expr | _python.int | _python.float | _python.bool | str):
        return value
    value = _base.at_(value_span, _builder._as_expr(value))
    return _name(_base.with_at_group_(span, lambda: _native.Bind(value)), name, name_span)


def decl_mutable_cell_(
    value: Any = _base.MISSING,
    *,
    ty: Any = None,
    name: str | None = None,
    span: _Span = None,
    name_span: _Span = None,
) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.decl_mutable_cell_`.

    Primitive annotations allocate scalar local storage; vector annotations
    allocate their declared shape. Buffer declaration producers retain their own effects.
    """
    name_span = span if name_span is None else name_span
    if isinstance(ty, _native.LocalVectorAnnotation):
        if value is not _base.MISSING:
            raise ValueError("Vector annotation does not support an initializer")
        return _name(
            _base.with_at_group_(span, lambda: _native.alloc_local(ty.shape, ty.dtype)),
            name,
            name_span,
        )
    if ty is not None:
        annotation = ty() if callable(ty) and not isinstance(ty, _ir.Expr) else ty
        annotation = annotation.ty if isinstance(annotation, _ir.Expr) else annotation
        if not isinstance(annotation, _ir.PrimType) or str(annotation) == "handle":
            raise TypeError("Mutable scalar annotations require a primitive scalar type")
        storage = _base.with_at_group_(span, lambda: _native.local_scalar(str(annotation))).scalar
        if value is not _base.MISSING:
            set_mutable_cell_(storage, value, span=span)
    else:
        storage = value.scalar if isinstance(value, _native.scalar_wrapper) else value
    if isinstance(storage, _ir.TensorLoad):
        _name(storage.source, name, name_span)
    elif _tir.is_buffer_var(storage):
        _name(storage, name, name_span)
    else:
        raise TypeError("A mutable declaration requires scalar or vector storage")
    return storage


def set_mutable_cell_(
    target: _ir.TensorLoad | _ir.Var | _native.scalar_wrapper, value: Any, *, span: _Span = None
) -> _base.AlreadyEmitted[_tir.Stmt]:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.set_mutable_cell_`.

    Updates emit a scalar buffer store. Targets must denote scalar storage.
    """
    if isinstance(target, _native.scalar_wrapper):
        target = target.scalar
    if isinstance(target, _ir.TensorLoad):
        return _base.at_(span, _builder.buffer_store(target.source, value, list(target.indices)))
    elif (
        _tir.is_buffer_var(target)
        and len(target.ty.shape) == 1
        and isinstance(target.ty.shape[0], _tir.IntImm)
        and target.ty.shape[0].value == 1
    ):
        return _base.at_(span, _builder.buffer_store(target, value, [0]))
    else:
        raise TypeError("A mutable assignment requires scalar storage")


def unpack(value: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.unpack`."""
    if isinstance(value, _ir.Tuple):
        return _python.tuple(value.fields)
    if isinstance(value, _ir.Expr) and isinstance(value.ty, _ir.TupleType):
        return _python.tuple(_ir.TupleGetItem(value, i) for i in range(len(value.ty.fields)))
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

    Native statements emit once; receipts are already emitted. Vars, layouts
    and meta_class instances are inert. Sequences are consumed
    elementwise; concise frames close with their owning parent. Other values use
    native expression conversion, retaining its errors for unsupported host values.
    """
    if isinstance(value, _base.AlreadyEmitted):
        _base.at_(span, value)
        return None
    if (
        value is None
        or isinstance(value, str | _ir.Var | _tir.Layout)
        or getattr(type(value), "_is_meta_class", False)
    ):
        return
    if isinstance(value, list | tuple | _ir.Array):
        for item in value:
            emit_(item, span=span)
        return
    if isinstance(value, _base.IRBuilderFrame):
        _enter_concise(_base.at_(span, value))
    elif hasattr(value, "frames"):
        for frame in value.frames:
            _enter_concise(_base.at_(span, frame))
    elif isinstance(value, _tir.Stmt):
        _native.add_to_parent(_base.at_(span, value))
    else:
        # Native conversion owns Python literals; annotate the exact expression
        # it stored, as well as the statement, without converting or emitting twice.
        emitted = _native.evaluate(value)
        _base.at_(span, emitted.value.value)
        _base.at_(span, emitted)


def return_(value: Any = None, *, span: _Span = None) -> _base.AlreadyEmitted[_tir.Stmt]:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.return_`."""
    if value is None:
        raise TypeError("A primitive function return requires an expression")
    return _base.with_at_group_(span, lambda: _native.Return(_builder._as_expr(value)))


def setitem_(
    target: Any, key: Any, value: Any, *, span: _Span = None
) -> _base.AlreadyEmitted[_tir.Stmt]:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.setitem_`."""
    return _base.at_(span, _builder.buffer_store(target, value, key))


def setattr_(
    target: Any, name: str, value: Any, *, span: _Span = None
) -> _base.AlreadyEmitted[_tir.Stmt] | None:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.setattr_`."""
    previous = getattr(target, name, _base.MISSING)
    if isinstance(previous, _native.scalar_wrapper):
        previous = previous.scalar
    buffer = previous.source if isinstance(previous, _ir.TensorLoad) else previous
    if _tir.is_buffer_var(buffer):
        shape = buffer.ty.shape
        if len(shape) == 1 and _python.bool(shape[0] == 1):
            return set_mutable_cell_(previous, value, span=span)
    _python.setattr(target, name, value)


def assert_(
    condition: Any,
    message: str | tuple[str, Sequence[Any]] | Sequence[Any] = "",
    *,
    span: _Span = None,
) -> None:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.assert_`."""
    kind = "RuntimeError"
    if isinstance(message, tuple):
        if len(message) != 2 or not isinstance(message[0], str):
            raise TypeError("Assertion metadata must be (error_kind, message_parts)")
        kind, message = message
    if isinstance(message, list | tuple):
        message = [str(part) for part in message]
    if not isinstance(message, list | tuple):
        message = [message]
    with _base.at_(span, _native.Assert(condition, message, error_kind=kind)):
        pass
