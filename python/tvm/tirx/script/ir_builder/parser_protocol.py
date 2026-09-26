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

Hooks construct native frames and statements using this dialect's IR and operations.
For example, generated ``X.if_(condition)`` creates the native conditional frame;
``X.then_()`` and ``X.else_()`` enter its branches. See the corresponding shared
``tvm.script.ir_builder.parser_protocol`` hooks for operand and span contracts.
"""

from __future__ import annotations

import builtins as _python
from collections.abc import Sequence
from functools import partial as _partial
from typing import Any

# isort: off
# isort: on
from tvm_ffi.core import String

from tvm import ir as _ir
from tvm import tirx as _tir
from tvm.ir import StringImm as _StringImm
from tvm.ir import TensorRegion, Type, is_prim_expr
from tvm.runtime import convert
from tvm.script.ir_builder import base as _base
from tvm.script.ir_builder.base import AlreadyEmitted
from tvm.script.ir_builder.base import IRBuilder as _IRBuilder
from tvm.tirx import Buffer, Expr
from tvm.tirx.exec_scope import Var
from tvm.tirx.expr import (
    IntImm,
)

from . import _ffi_api, frame, utils
from . import ir as _native
from . import op as _op
from .op import and_ as and_
from .op import eq_ as eq_
from .op import ge_ as ge_
from .op import gt_ as gt_
from .op import if_then_else_ as if_then_else_
from .op import le_ as le_
from .op import lt_ as lt_
from .op import ne_ as ne_
from .op import not_ as not_
from .op import or_ as or_

_Span = _base.SpanEntry | _ir.Span | None

# --------------------------------------
# Function
# --------------------------------------


def prim_func(
    is_private: bool = False,
    persistent: bool = False,
    *,
    private: bool | None = None,
) -> frame.PrimFuncFrame:
    """The primitive function statement.

    Parameters
    ----------
    is_private : bool
        Whether the PrimFunc is annotated as private.
    persistent : bool
        Whether this is a persistent kernel.
    private : bool
        Alias for ``is_private`` (used in decorator syntax).

    Returns
    -------
    res : frame.PrimFuncFrame
        The PrimFuncFrame.
    """
    if private is not None:
        is_private = private
    return _ffi_api.PrimFunc(is_private, persistent)  # type: ignore[attr-defined] # pylint: disable=no-member


def function_(
    *,
    private: bool = False,
    persistent: bool = False,
    decl: bool = False,
    span: _Span = None,
) -> frame.PrimFuncFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.function_`.

    Private/persistent options pass to the native TIRx function frame.
    The same frame supports declaration and body entry.
    """
    native = (
        _ffi_api.DeclFunction(private, persistent)
        if decl
        else _ffi_api.PrimFunc(private, persistent)
    )
    return _base.at_(span, native)


def arg_(name: str, annotation: Any, *, span: _Span = None) -> _ir.Var:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.arg_`."""
    if getattr(annotation, "__tvm_optional_annotation__", None) is not None:
        raise TypeError("T.Optional is only supported by @T.jit")
    if callable(annotation) and not isinstance(annotation, _ir.Expr):
        annotation = annotation()
    if isinstance(annotation, _ir.Type):
        annotation = _ir.Var(name, annotation)
    return _ffi_api.Arg(name, _base.at_(span, annotation))


def func_name_(name: str) -> None:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.func_name_`."""
    return _ffi_api.FuncName(name)


def func_ret_type_(annotation: Any, *, span: _Span = None) -> None:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.func_ret_type_`."""
    annotation = _base._return_annotation(annotation)
    if callable(annotation) and not isinstance(annotation, _ir.Expr | _ir.Type):
        annotation = annotation()
    if isinstance(annotation, _ir.Expr):
        annotation = annotation.ty
    return _ffi_api.FuncRet(_ir.Type.missing() if annotation is None else annotation)


def func_attr(attrs: dict[str, Any]) -> None:
    """The PrimFunc annotation statement.

    Parameters
    ----------
    attrs : Dict[str, Any]
        The annotations of the PrimFunc.
    """
    _ffi_api.FuncAttrs(attrs)  # type: ignore[attr-defined] # pylint: disable=no-member


def device_entry() -> None:
    """Mark the device-region entry within the enclosing PrimFunc body.

    Flat marker (no ``with``). Subsequent statements in the function body
    accumulate into an ``AttrStmt("tirx.device_entry", True, body=...)``;
    the wrapping is closed by the PrimFunc frame at function end.

    Anything written before this marker is host code (e.g. buffer layout setup);
    anything after is device code.

    Example::

        @T.prim_func
        def kernel(...):
            A = T.Buffer(...)
            T.device_entry()           # device region starts here
            bx = T.cta_id([SM_COUNT])  # standalone scope-id def
            ...
    """
    attr_frame = _ffi_api.DeviceEntry()  # type: ignore[attr-defined] # pylint: disable=no-member
    attr_frame.__enter__()


def check_well_formed_(function: _tir.PrimFunc) -> None:
    """Validate a completed TIRx function."""
    try:
        _tir.analysis.verify_well_formed(function)
        _tir.analysis.verify_tirx_well_formed(function)
    except Exception as error:
        raise ValueError(
            "Program is not well-formed. If this is deliberate, set "
            f"check_well_formed=False in the top-level decorator.\n{error}"
        ) from error


def _check_module_well_formed(module: _ir.IRModule) -> None:
    """Validate completed functions belonging to the TIRx dialect."""
    for function in module.functions.values():
        if isinstance(function, _tir.PrimFunc) and function.is_tirx:
            check_well_formed_(function)


# --------------------------------------
# Bindings
# --------------------------------------


def resolve_global_info_(content: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.resolve_global_info_`.

    TIRx does not define global-info selectors.
    """
    raise NotImplementedError("TIRx does not support global-info lookup")


def call_global_var_(function: _ir.GlobalVar, args: Sequence[Any]) -> _ir.Expr:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.call_global_var_`."""
    return _op._call_global(function, *args)


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
        value = _op._as_expr(value)
        if not isinstance(value, _ir.Var):
            _base.at_(value_span, value)
        variable = _name(ty.as_var(rhs_dtype=value.ty), name, name_span)
        _base.with_at_group_(span, lambda: bind(value, var=variable))
        return variable
    if ty is not None:
        annotation = ty() if callable(ty) and not isinstance(ty, _ir.Expr) else ty
        annotation = annotation.ty if isinstance(annotation, _ir.Expr) else annotation
        value = _op._as_expr(value)
        if not isinstance(value, _ir.Var):
            _base.at_(value_span, value)
        variable = _ir.Var(name or "", annotation)
        return _name(_base.with_at_group_(span, lambda: bind(value, var=variable)), name, name_span)
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
    value = _base.at_(value_span, _op._as_expr(value))
    return _name(_base.with_at_group_(span, lambda: bind(value)), name, name_span)


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
        storage = _base.with_at_group_(span, lambda: _native.local_scalar(str(annotation)))
        if value is not _base.MISSING:
            set_mutable_cell_(storage, value, span=span)
    else:
        storage = value
    if isinstance(storage, _ir.TensorLoad):
        _name(storage.source, name, name_span)
    elif _tir.is_buffer_var(storage):
        _name(storage, name, name_span)
    else:
        raise TypeError("A mutable declaration requires scalar or vector storage")
    return storage


def set_mutable_cell_(
    target: _ir.TensorLoad | _ir.Var, value: Any, *, span: _Span = None
) -> _base.AlreadyEmitted[_tir.Stmt]:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.set_mutable_cell_`.

    Updates emit a scalar buffer store. Targets must denote scalar storage.
    """
    if isinstance(target, _ir.TensorLoad):
        return _base.at_(span, buffer_store(target.source, value, list(target.indices)))
    elif (
        _tir.is_buffer_var(target)
        and len(target.ty.shape) == 1
        and isinstance(target.ty.shape[0], _tir.IntImm)
        and target.ty.shape[0].value == 1
    ):
        return _base.at_(span, buffer_store(target, value, [0]))
    else:
        raise TypeError("A mutable assignment requires scalar storage")


def unpack_(value: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.unpack_`."""
    if isinstance(value, _ir.Tuple):
        return _python.tuple(value.fields)
    if isinstance(value, _ir.Expr) and isinstance(value.ty, _ir.TupleType):
        return _python.tuple(_ir.TupleGetItem(value, i) for i in range(len(value.ty.fields)))
    return value


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
        add_to_parent(_base.at_(span, value))
    else:
        # Native conversion owns Python literals; annotate the exact expression
        # it stored, as well as the statement, without converting or emitting twice.
        emitted = evaluate(value)
        _base.at_(span, emitted.value.value)
        _base.at_(span, emitted)


def setitem_(
    target: Any, key: Any, value: Any, *, span: _Span = None
) -> _base.AlreadyEmitted[_tir.Stmt]:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.setitem_`."""
    return _base.at_(span, buffer_store(target, value, key))


def setattr_(
    target: Any, name: str, value: Any, *, span: _Span = None
) -> _base.AlreadyEmitted[_tir.Stmt] | None:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.setattr_`."""
    previous = getattr(target, name, _base.MISSING)
    buffer = previous.source if isinstance(previous, _ir.TensorLoad) else previous
    if _tir.is_buffer_var(buffer):
        shape = buffer.ty.shape
        if len(shape) == 1 and _python.bool(shape[0] == 1):
            return set_mutable_cell_(previous, value, span=span)
    _python.setattr(target, name, value)


def bind(  # pylint: disable=invalid-name
    value: Expr,
    type_annotation: Type | None = None,  # pylint: disable=redefined-outer-name
    *,
    var: Var | None = None,  # pylint: disable=redefined-outer-name
) -> Var:
    """Create a Bind (variable binding).

    Emits a flat Bind statement to the current frame and returns the bound variable.

    Parameters
    ----------
    value : Expr
        The value to be bound.
    type_annotation : Optional[Type] = None
        The type annotation of the binding. Usually it is used for fine-grained var typing,
        particularly, PointerType.
    var : Optional[Var] = None
        The variable to bind. If not specified, a new variable will be created.

    Returns
    -------
    var : Var
        The bound variable.
    """
    if type_annotation is not None:
        # Canonical Vars are callable when they denote functions.  Here a Var is
        # already a resolved type annotation, rather than a deferred annotation factory.
        if callable(type_annotation) and not isinstance(type_annotation, Expr):
            type_annotation = type_annotation()
        if isinstance(type_annotation, _ir.Var):
            type_annotation = type_annotation.ty
    return _ffi_api.Bind(value, type_annotation, var)  # type: ignore[attr-defined] # pylint: disable=no-member


def attr(
    node_or_dict: Any, attr_key: str | None = None, value: Expr | str | None = None
) -> frame.AttrFrame | utils._FrameScope:
    """Create an attribute node, or multiple attribute nodes from a dict.

    Usage 1 — single attr::

        with T.attr(node, key, value):
            ...

    Usage 2 — dict sugar (node defaults to ``0``)::

        with T.attr({"key1": value1, "key2": value2}):
            ...

    Parameters
    ----------
    node_or_dict : Any
        If a dict, each key-value pair becomes an AttrStmt with
        ``node=0``.  Otherwise the node to annotate.

    attr_key : str, optional
        Attribute type key (required when ``node_or_dict`` is not a dict).

    value : Union[Expr, str], optional
        The attribute value (required when ``node_or_dict`` is not a dict).

    Returns
    -------
    res : Union[frame.AttrFrame, _FrameScope]
        A single AttrFrame, or a _FrameScope wrapping multiple AttrFrames.
    """
    if isinstance(node_or_dict, dict):
        frames = []
        for k, v in node_or_dict.items():
            if isinstance(v, bool):
                v = IntImm("bool", v)
            frames.append(_ffi_api.Attr(0, k, convert(v)))  # type: ignore[attr-defined]
        if len(frames) == 1:
            return frames[0]
        return utils._FrameScope(frames)
    else:
        if attr_key is None or value is None:
            raise ValueError("T.attr(node, attr_key, value) requires all three arguments")
        node_or_dict = convert(node_or_dict)
        value = convert(value)
        return _ffi_api.Attr(node_or_dict, attr_key, value)  # type: ignore[attr-defined] # pylint: disable=no-member


def hint(message: str = "", **attrs) -> frame.HintFrame:
    """Universal directive primitive for the sketch language.

    Parameters
    ----------
    message : str
        Free-form directive string that the agent interprets.
    **attrs
        Optional structured key-value attributes for known patterns.

    Returns
    -------
    res : frame.HintFrame
        Usable as context manager (with T.hint("msg"):) or bare statement (T.hint("msg")).
    """
    return _ffi_api.Hint(message, attrs or {})  # type: ignore[attr-defined] # pylint: disable=no-member


def buffer_store(
    buffer: Buffer,  # pylint: disable=redefined-outer-name
    value: Expr,
    indices: list[Expr | slice],
) -> AlreadyEmitted[_tir.Stmt]:
    """Emit a buffer store and return a receipt for the stored statement.

    Parameters
    ----------
    buffer : Buffer
        The buffer.

    value : Expr
        The value to be stored.

    indices : List[Union[Expr, slice]]
        The indices location to be stored.

    Returns
    -------
    result : AlreadyEmitted[Stmt]
        Receipt for the exact stored statement; consuming it does not emit again.

    """
    from tvm.sym import Analyzer  # pylint: disable=import-outside-toplevel

    if not isinstance(indices, list | tuple | _ir.Array):
        indices = [indices]

    expr_indices = []
    for index in indices:
        if isinstance(index, slice):
            step = 1 if index.step is None else index.step
            lanes = Analyzer().simplify(  # pylint: disable=redefined-outer-name
                (index.stop - index.start + step - 1) // step
            )
            if lanes == 1:
                expr_indices.append(index.start)
            else:
                expr_indices.append(_op.ramp(index.start, step, lanes))
        else:
            expr_indices.append(index)
    if isinstance(value, bool) and buffer.ty.dtype == "bool":
        value = IntImm("bool", value)
    return AlreadyEmitted(_ffi_api.BufferStore(buffer, value, expr_indices))


def evaluate(value: Expr) -> AlreadyEmitted[_tir.Stmt]:
    """Emit an evaluation and return a reference to its stored statement.

    Parameters
    ----------
    value : Expr
        The input expression to evaluate.

    Returns
    -------
    result : AlreadyEmitted[Stmt]
        A receipt containing the emitted statement, so expression-statement
        handling does not emit it again.
    """
    if isinstance(value, str):
        value = _StringImm(value)
    if isinstance(value, bool):
        value = IntImm("bool", value)
    if isinstance(value, TensorRegion):
        raise TypeError(
            "T.evaluate does not accept TensorRegion values; "
            "construct a BufferLoad with explicit indices"
        )
    return AlreadyEmitted(_ffi_api.Evaluate(value))  # type: ignore[attr-defined] # pylint: disable=no-member


def add_to_parent(stmt: _tir.Stmt) -> None:
    """Add a statement to the parent frame."""
    _ffi_api.AddToParent(stmt)  # type: ignore[attr-defined] # pylint: disable=no-member


# --------------------------------------
# Special
# --------------------------------------
# Syntax markers and declaration policies are registered by the source namespace.
# They are consumed by the parser before runtime builder calls.

# --------------------------------------
# Control
# --------------------------------------


def if_(condition: Any, *, span: _Span = None) -> frame.IfFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.if_`."""
    if isinstance(condition, _python.bool):
        condition = IntImm("bool", condition)
    return _base.at_(span, _ffi_api.If(condition))


def then_(*, span: _Span = None) -> frame.ThenFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.then_`."""
    return _base.at_(span, _ffi_api.Then())


def else_(*, span: _Span = None) -> frame.ElseFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.else_`."""
    return _base.at_(span, _ffi_api.Else())


def for_(
    iterable: Any, *, names: str | Sequence[str] | None = None, span: _Span = None
) -> frame.ForFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.for_`.

    A single native loop returns its scalar Var; multiple loops return their
    sequence. frame.vars remains the stable sequence for source unpacking.
    """
    if isinstance(iterable, _python.range):
        iterable = serial(iterable.start, iterable.stop, step=iterable.step)
    if not isinstance(iterable, frame.ForFrame):
        raise TypeError("A primitive for loop requires an iteration specification")
    iterable.set_names(names)
    return _base.at_(span, iterable)


def while_(condition: Any, *, span: _Span = None) -> frame.WhileFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.while_`."""
    if isinstance(condition, _python.bool):
        condition = IntImm("bool", condition)
    return _base.at_(span, _ffi_api.While(condition))


def range_(*args: Any, annotations: dict[str, Any] | None = None) -> frame.ForFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.range_`."""
    if len(args) == 1:
        args = (0, args[0], None)
    elif len(args) == 2:
        args = (*args, None)
    elif len(args) != 3:
        raise TypeError("range expects one to three arguments")
    if isinstance(args[2], _python.int) and args[2] == 0:
        raise ValueError("range step cannot be zero")
    return serial(args[0], args[1], step=args[2], annotations=annotations)


def break_(*, span: _Span = None) -> _base.AlreadyEmitted[_tir.Stmt]:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.break_`.

    Legality is checked on the completed function, across loop and function boundaries.
    """
    return _base.with_at_group_(span, lambda: _base.AlreadyEmitted(_ffi_api.Break()))


def continue_(*, span: _Span = None) -> _base.AlreadyEmitted[_tir.Stmt]:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.continue_`.

    Legality is checked on the completed function, across loop and function boundaries.
    """
    return _base.with_at_group_(span, lambda: _base.AlreadyEmitted(_ffi_api.Continue()))


def return_(value: Any = None, *, span: _Span = None) -> _base.AlreadyEmitted[_tir.Stmt]:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.return_`."""
    if value is None:
        raise TypeError("A primitive function return requires an expression")
    return _base.with_at_group_(
        span, lambda: _base.AlreadyEmitted(_ffi_api.Return(_op._as_expr(value)))
    )


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
    if isinstance(condition, _python.bool):
        condition = IntImm("bool", condition)
    with _base.at_(span, _ffi_api.Assert(condition, kind, message)):
        pass


def serial(
    start: Expr,
    stop: Expr = None,
    *,
    annotations: dict[str, Any] | None = None,
    step: Expr | None = None,
    unroll: bool | int | None = None,
    dtype: str | None = None,
) -> frame.ForFrame:
    """The serial For statement.

    Parameters
    ----------
    start : Expr
        The minimum value of iteration.

    stop : Expr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    step : Expr
        The optional step value of iteration.

    unroll : bool or int, optional
        If True, adds ``{"pragma_unroll": True}`` annotation, which asks CUDA codegen
        to emit ``#pragma unroll`` while preserving the loop as a C++ ``for``.
        If False, adds ``{"disable_unroll": True}`` annotation.
        If a positive integer, emits ``#pragma unroll N``. Boolean values are
        handled separately from integers, so ``False`` keeps disabling unrolling.

    dtype : str, optional
        The dtype of the loop variable, either ``"int32"`` or ``"uint32"``. When
        omitted it is inferred from the bounds. Bounds that do not already have this
        dtype are converted (literals are retyped, other expressions get a Cast).
        Note ``T.thread_binding`` does not support this; its loop var is always int32.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if unroll is not None:
        annotations = dict(annotations) if annotations else {}
        if isinstance(unroll, bool):
            if unroll:
                annotations["pragma_unroll"] = True
            else:
                annotations["disable_unroll"] = True
        elif isinstance(unroll, int):
            if unroll < 1:
                raise ValueError("unroll must be a positive integer")
            annotations["pragma_unroll"] = unroll
        else:
            raise TypeError("unroll must be a bool, a positive integer, or None")
    if stop is None:
        stop = start
        if is_prim_expr(start):
            start = IntImm(start.ty, 0)
        else:
            start = 0
    return _ffi_api.Serial(start, stop, annotations, step, dtype)  # type: ignore[attr-defined] # pylint: disable=no-member


def parallel(
    start: Expr,
    stop: Expr = None,
    *,
    annotations: dict[str, Any] | None = None,
    step: Expr | None = None,
    dtype: str | None = None,
) -> frame.ForFrame:
    """The parallel For statement.

    Parameters
    ----------
    start : Expr
        The minimum value of iteration.

    stop : Expr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    step : Expr
        The optional step value of iteration.

    dtype : str, optional
        The dtype of the loop variable, either ``"int32"`` or ``"uint32"``. When
        omitted it is inferred from the bounds.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if stop is None:
        stop = start
        if is_prim_expr(start):
            start = IntImm(start.ty, 0)
        else:
            start = 0
    return _ffi_api.Parallel(start, stop, annotations, step, dtype)  # type: ignore[attr-defined] # pylint: disable=no-member


def vectorized(
    start: Expr,
    stop: Expr = None,
    *,
    annotations: dict[str, Any] | None = None,
    step: Expr | None = None,
    dtype: str | None = None,
) -> frame.ForFrame:
    """The vectorized For statement.

    Parameters
    ----------
    start : Expr
        The minimum value of iteration.

    stop : Expr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    step : Expr
        The optional step value of iteration.

    dtype : str, optional
        The dtype of the loop variable, either ``"int32"`` or ``"uint32"``. When
        omitted it is inferred from the bounds.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if stop is None:
        stop = start
        if is_prim_expr(start):
            start = IntImm(start.ty, 0)
        else:
            start = 0
    return _ffi_api.Vectorized(start, stop, annotations, step, dtype)  # type: ignore[attr-defined] # pylint: disable=no-member


def unroll(
    start: Expr,
    stop: Expr = None,
    *,
    annotations: dict[str, Any] | None = None,
    step: Expr | None = None,
    dtype: str | None = None,
) -> frame.ForFrame:
    """The unrolled For statement.

    Parameters
    ----------
    start : Expr
        The minimum value of iteration.

    stop : Expr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    step : Expr
        The optional step value of iteration.

    dtype : str, optional
        The dtype of the loop variable, either ``"int32"`` or ``"uint32"``. When
        omitted it is inferred from the bounds.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if stop is None:
        stop = start
        if is_prim_expr(start):
            start = IntImm(start.ty, 0)
        else:
            start = 0
    return _ffi_api.Unroll(start, stop, annotations, step, dtype)  # type: ignore[attr-defined] # pylint: disable=no-member


def thread_binding(
    start: Expr,
    stop: Expr = None,
    thread: str | None = None,
    *,
    annotations: dict[str, Any] | None = None,
) -> frame.ForFrame:
    """The thread-binding For statement.

    Parameters
    ----------
    start : Expr
        The minimum value of iteration.

    stop : Expr
        The maximum value of iteration.

    thread : str
        The thread for loop variable to bind.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if thread is None:
        if not isinstance(stop, str):
            raise ValueError("Thread cannot be None for thread_binding")
        thread = stop
        stop = start
        if is_prim_expr(start):
            start = IntImm(start.ty, 0)
        else:
            start = 0
    elif stop is None:
        stop = start
        if is_prim_expr(start):
            start = IntImm(start.ty, 0)
        else:
            start = 0
    return _ffi_api.ThreadBinding(  # type: ignore[attr-defined] # pylint: disable=no-member
        start, stop, thread, annotations
    )


def grid(*extents: tuple[Expr | tuple[Expr, Expr]], dtype: str | None = None) -> frame.ForFrame:
    """The grid For statement.

    Parameters
    ----------
    extents : Tuple[Union[Expr, Tuple[Expr, Expr]]]
        If a single Expr is provided, it is used as the extent of the iteration.
        If a tuple of two Expr is provided, the first is the start of the iteration,
        and the second is the extent of the iteration.

    dtype : str, optional
        The dtype of every loop variable, either ``"int32"`` or ``"uint32"``. When
        omitted each loop variable takes the dtype of its own extent.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    # Convert integer extents to IntImm
    # TODO(@bohan): fix this after FFI refactor
    imm_dtype = dtype if dtype is not None else "int32"
    processed_extents = []
    for extent in extents:
        if isinstance(extent, tuple):
            start, extent = extent
            start = IntImm(imm_dtype, start) if isinstance(start, int) else start
            extent = IntImm(imm_dtype, extent) if isinstance(extent, int) else extent
            processed_extents.append((start, extent))
        else:
            processed_extents.append(
                IntImm(imm_dtype, extent) if isinstance(extent, int) else extent
            )
    extents = tuple(processed_extents)
    return _ffi_api.Grid(extents, dtype)  # type: ignore[attr-defined] # pylint: disable=no-member


def launch_thread(
    thread: Var | str,  # pylint: disable=redefined-outer-name
    extent: Expr,
) -> frame.LaunchThreadFrame:
    """Launch a thread.

    Parameters
    ----------
    thread : Union[Var, str]
        The iteration variable.

    extent : Expr
        The extent of environment thread.

    Returns
    -------
    res : frame.LaunchThreadFrame
        The result LaunchThreadFrame.

    Examples
    --------

    .. code-block:: python

    from tvm.tirx.script import ir_builder as T
    brow = T.env_thread("blockIdx.y")
    T.launch_thread(brow, 1)

    """

    if isinstance(thread, str):
        thread = String(thread)
    return _ffi_api.LaunchThread(thread, extent)  # type: ignore[attr-defined] # pylint: disable=no-member


def env_thread(thread_tag: str, dtype: str = "int32") -> Var:
    """Bind a var to thread env

    Parameters
    ----------
    thread_tag : str
        The thread type tag.

    dtype : str
        The data type of the thread env.

    Returns
    -------
    res : Var
        The thread variable; native function state retains its iteration metadata.

    """
    return _ffi_api.EnvThread(thread_tag, dtype)  # type: ignore[attr-defined] # pylint: disable=no-member


# --------------------------------------
# Operators
# --------------------------------------
# Operator hooks are re-exported directly from the concrete op module above.

func_ret = func_ret_type_
emit = emit_

__all__ = [
    "add_to_parent",
    "and_",
    "arg_",
    "assert_",
    "attr",
    "bind",
    "bind_",
    "break_",
    "buffer_store",
    "call_global_var_",
    "check_well_formed_",
    "continue_",
    "decl_mutable_cell_",
    "device_entry",
    "else_",
    "emit",
    "emit_",
    "env_thread",
    "eq_",
    "evaluate",
    "for_",
    "func_attr",
    "func_name_",
    "func_ret",
    "func_ret_type_",
    "function_",
    "ge_",
    "grid",
    "gt_",
    "hint",
    "if_",
    "if_then_else_",
    "launch_thread",
    "le_",
    "lt_",
    "ne_",
    "not_",
    "or_",
    "parallel",
    "prim_func",
    "range_",
    "resolve_global_info_",
    "return_",
    "serial",
    "set_mutable_cell_",
    "setattr_",
    "setitem_",
    "then_",
    "thread_binding",
    "unpack_",
    "unroll",
    "vectorized",
    "while_",
]
