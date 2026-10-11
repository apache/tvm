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
"""Shared construction of core IR statements and their native frames.

Dialect namespaces re-export these operations and retain their own binding,
function, tensor, and execution-scope policies.
"""

from __future__ import annotations

import builtins as _python
import contextlib
from collections.abc import Sequence
from functools import partial as _partial
from typing import Any

import tvm_ffi as _ffi

import tvm
from tvm import ir as _ir
from tvm.ir import Expr, TensorRegion, Type, Var
from tvm.ir import StringImm as _StringImm
from tvm.ir.prim import IntImm
from tvm.script.ir_builder import base as _base
from tvm.script.ir_builder.base import AlreadyEmitted

from . import _ffi_api, frame

_Loc = _base.LocationEntry | _ir.Location | None


def _as_expr(value):
    if isinstance(value, _ffi.ObjectConvertible):
        value = value.asobject()
    if isinstance(value, _ir.Expr):
        return value
    if isinstance(value, str):
        return _ir.StringImm(value)
    if isinstance(value, list | tuple):
        return _ir.Tuple([_as_expr(item) for item in value])
    return _ir.const(value)


def _enter_concise(frame: _base.IRBuilderFrame) -> Any:
    # add_callback registers on the active parent before the child enters.
    # Later statements emit into the child; parent exit closes this scope.
    frame.add_callback(_partial(frame.__exit__, None, None, None))
    return frame.__enter__()


def unpack_(value: Any) -> Any:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.unpack_`."""
    if isinstance(value, _ir.Tuple):
        return _python.tuple(value.fields)
    if isinstance(value, _ir.Expr) and isinstance(value.ty, _ir.TupleType):
        return _python.tuple(_ir.TupleGetItem(value, i) for i in range(len(value.ty.fields)))
    return value


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
        particularly, PtrType.
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


def evaluate(value: Expr) -> AlreadyEmitted[tvm.ir.Stmt]:
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
            "construct a TensorLoad with explicit indices"
        )
    return AlreadyEmitted(_ffi_api.Evaluate(value))  # type: ignore[attr-defined] # pylint: disable=no-member


def add_to_parent(stmt: tvm.ir.Stmt) -> None:
    """Add a statement to the parent frame."""
    _ffi_api.AddToParent(stmt)  # type: ignore[attr-defined] # pylint: disable=no-member


def if_(condition: Any, *, loc: _Loc = None) -> frame.IfFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.if_`."""
    if isinstance(condition, _python.bool):
        condition = IntImm("bool", condition)
    return _base.at_(loc, _ffi_api.If(condition))


def then_(*, loc: _Loc = None) -> frame.ThenFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.then_`."""
    return _base.at_(loc, _ffi_api.Then())


def else_(*, loc: _Loc = None) -> frame.ElseFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.else_`."""
    return _base.at_(loc, _ffi_api.Else())


def while_(condition: Any, *, loc: _Loc = None) -> frame.WhileFrame:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.while_`."""
    if isinstance(condition, _python.bool):
        condition = IntImm("bool", condition)
    return _base.at_(loc, _ffi_api.While(condition))


def break_(*, loc: _Loc = None) -> _base.AlreadyEmitted[tvm.ir.Stmt]:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.break_`.

    Legality is checked on the completed function, across loop and function boundaries.
    """
    return _base.with_at_group_(loc, lambda: _base.AlreadyEmitted(_ffi_api.Break()))


def continue_(*, loc: _Loc = None) -> _base.AlreadyEmitted[tvm.ir.Stmt]:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.continue_`.

    Legality is checked on the completed function, across loop and function boundaries.
    """
    return _base.with_at_group_(loc, lambda: _base.AlreadyEmitted(_ffi_api.Continue()))


def return_(value: Any = None, *, loc: _Loc = None) -> _base.AlreadyEmitted[tvm.ir.Stmt]:
    """Implements :func:`tvm.script.ir_builder.parser_protocol.return_`."""
    if value is None:
        raise TypeError("A primitive function return requires an expression")
    return _base.with_at_group_(loc, lambda: _base.AlreadyEmitted(_ffi_api.Return(_as_expr(value))))


def assert_(
    condition: Any,
    message: str | tuple[str, Sequence[Any]] | Sequence[Any] = "",
    *,
    loc: _Loc = None,
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
    with _base.at_(loc, _ffi_api.Assert(condition, kind, message)):
        pass


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


def region(
    op: _ir.Op | str,
    args: Sequence[Expr],
    body_params: Sequence[Var] | None = None,
    attrs: _ir.DictAttrs | dict[str, Any] | None = None,
) -> frame.RegionFrame:
    """Construct a result-free region with operation-defined body parameters.

    When ``body_params`` is omitted, the operation's ``FRegionGetBodyParams``
    hook creates fresh typed variables. Every region operation must register
    the hook, returning an empty array for no body parameters. Missing hooks
    reject construction even with explicit parameters. Explicit parameters must
    match the hook's count and types and retain their identities.

    Operands and attributes belong to the enclosing scope. Entering the frame
    returns one parameter directly, or a sequence for zero or multiple parameters.
    Result variables are supported by direct ``tvm.ir.RegionStmt`` construction;
    outward-result script syntax is not supported.
    """
    if isinstance(op, str):
        op = _ir.Op.get(op)
    if attrs is None or isinstance(attrs, dict):
        attrs = _ir.make_node("ir.DictAttrs", **(attrs or {}))
    return _ffi_api.Region(op, args, body_params, attrs)


class _FrameScope:
    """Context manager to enter multiple IRBuilder frames without deep nesting.

    This class allows entering multiple frames in a single `with` statement,
    avoiding the pyramid of nested context managers.

    Parameters
    ----------
    frames : List[IRBuilderFrame]
        The list of frames to enter.
    """

    def __init__(self, frames):
        self.frames = frames if isinstance(frames, list | tuple) else [frames]
        self._stack = None

    def __enter__(self):
        self._stack = contextlib.ExitStack()
        self._stack.__enter__()
        results = [self._stack.enter_context(f) for f in self.frames]
        return tuple(results) if len(results) > 1 else results[0]

    def __exit__(self, *args):
        return self._stack.__exit__(*args)


def frame_scope(frames: list[frame.StmtFrame]) -> _FrameScope:
    """Enter multiple IRBuilder frames without deep nesting.

    This function provides a way to enter multiple frames in a single `with`
    statement, which is particularly useful when migrating from cases where
    allocations don't require nested scopes.

    Parameters
    ----------
    frames : List[frame.StmtFrame]
        The list of frames to enter. Each frame's `__enter__` return value
        will be collected and returned as a tuple.

    Returns
    -------
    _FrameScope
        A context manager that enters all frames and returns their values.
    """
    return _FrameScope(frames)
