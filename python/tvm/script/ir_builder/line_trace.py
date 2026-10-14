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
"""Executed Python line locations for imperative IR construction."""

import linecache
import sys
import sysconfig
import threading
from contextlib import AbstractContextManager
from pathlib import Path
from types import FrameType
from typing import Any

import tvm_ffi

from tvm.ir import SourceLoc, SourceName, UnknownLoc

from . import _ffi_api
from .base import IRBuilder

_STATE = threading.local()
_TVM_ROOT = Path(__file__).resolve().parents[2]
_FFI_ROOT = Path(tvm_ffi.__file__).resolve().parent
_STDLIB_ROOT = Path(sysconfig.get_path("stdlib")).resolve()
_SITE_ROOTS = tuple(Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib"))


class _TraceManager:
    """One callback per thread, shared by lexically nested trace scopes."""

    def __init__(self) -> None:
        self.previous_trace = sys.gettrace()
        self.callback = self.trace
        self.scopes: list[_LineTrace] = []
        self.frames: dict[FrameType, tuple[Any, bool, bool]] = {}
        self.sources: dict[str, bool] = {}
        self.locations: dict[tuple[str, int], SourceLoc] = {}

    def is_source(self, filename: str) -> bool:
        if filename not in self.sources:
            accepted = False
            if filename and not filename.startswith("<"):
                try:
                    path = Path(filename).resolve()
                    implementation = path.is_relative_to(_TVM_ROOT) or path.is_relative_to(
                        _FFI_ROOT
                    )
                    stdlib = path.is_relative_to(_STDLIB_ROOT) and not any(
                        path.is_relative_to(root) for root in _SITE_ROOTS
                    )
                    accepted = not implementation and not stdlib and path.is_file()
                except (OSError, RuntimeError):
                    pass
            self.sources[filename] = accepted
        return self.sources[filename]

    def activate(self, frame: FrameType) -> None:
        if frame not in self.frames:
            self.frames[frame] = (frame.f_trace, frame.f_trace_lines, frame.f_trace_opcodes)
        frame.f_trace = self.callback
        frame.f_trace_lines = True
        frame.f_trace_opcodes = False

    def set_line(self, scope: "_LineTrace", frame: FrameType) -> None:
        key = (frame.f_code.co_filename, frame.f_lineno)
        if key not in self.locations:
            filename, line = key
            text = linecache.getline(filename, line).rstrip("\r\n")
            self.locations[key] = SourceLoc(
                SourceName(filename), line, 1, line, max(2, len(text) + 1)
            )
        _ffi_api.IRBuilderUpdateTraceLoc(scope.builder, scope.slot, self.locations[key])

    def trace(self, frame: FrameType, event: str, arg: Any) -> Any:
        # CPython disables tracing while executing this callback, including FFI calls.
        if not self.scopes:
            return None
        scope = self.scopes[-1]
        if event == "call":
            if not self.is_source(frame.f_code.co_filename):
                return None
            self.activate(frame)
        elif event == "line":
            if (
                self.is_source(frame.f_code.co_filename)
                and IRBuilder.is_in_scope()
                and IRBuilder.current().same_as(scope.builder)
            ):
                self.set_line(scope, frame)
        elif event == "return":
            if IRBuilder.is_in_scope() and IRBuilder.current().same_as(scope.builder):
                caller = frame.f_back
                restored = False
                while caller is not None:
                    if self.is_source(caller.f_code.co_filename):
                        self.set_line(scope, caller)
                        restored = True
                        break
                    if caller is scope.caller:
                        break
                    caller = caller.f_back
                if not restored:
                    _ffi_api.IRBuilderUpdateTraceLoc(scope.builder, scope.slot, UnknownLoc())
            # Completed helper frames can own large IR graphs. Restore them now
            # instead of retaining every call until the enclosing scope exits.
            if frame in self.frames and all(frame is not item.caller for item in self.scopes):
                trace, lines, opcodes = self.frames.pop(frame)
                frame.f_trace = trace
                frame.f_trace_lines = lines
                frame.f_trace_opcodes = opcodes
                return trace
        return self.callback

    def restore(self) -> None:
        sys.settrace(self.previous_trace)
        for frame, (trace, lines, opcodes) in self.frames.items():
            frame.f_trace = trace
            frame.f_trace_lines = lines
            frame.f_trace_opcodes = opcodes
        self.frames.clear()
        self.locations.clear()
        self.sources.clear()


class _LineTrace(AbstractContextManager[None]):
    def __init__(self) -> None:
        self.manager: _TraceManager | None = None
        self.builder: IRBuilder | None = None
        self.caller: FrameType | None = None
        self.slot = -1

    def __enter__(self) -> None:
        if self.manager is not None:
            raise RuntimeError("A line_trace context cannot enter itself")
        if not IRBuilder.is_in_scope():
            raise RuntimeError("line_trace requires an active IRBuilder")
        manager = getattr(_STATE, "manager", None)
        outermost = manager is None
        if outermost:
            manager = _TraceManager()
        self.builder = IRBuilder.current()
        self.caller = sys._getframe(1)  # pylint: disable=protected-access
        self.slot = _ffi_api.IRBuilderPushTraceLoc(self.builder)
        self.manager = manager
        manager.scopes.append(self)
        try:
            manager.activate(self.caller)
            if manager.is_source(self.caller.f_code.co_filename):
                manager.set_line(self, self.caller)
            if outermost:
                _STATE.manager = manager
                sys.settrace(manager.callback)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        manager = self.manager
        if manager is None:
            return
        try:
            _ffi_api.IRBuilderPopTraceLoc(self.builder, self.slot)
        finally:
            manager.scopes.pop()
            if not manager.scopes:
                try:
                    manager.restore()
                finally:
                    _STATE.manager = None
            elif sys.gettrace() is not manager.callback:
                # A callback exception disables Python tracing. Resume the owned
                # frames when unwinding an inner scope so its caller can recover.
                for frame in manager.frames:
                    manager.activate(frame)
                sys.settrace(manager.callback)
            self.manager = None
            self.builder = None
            self.caller = None


def line_trace() -> AbstractContextManager[None]:
    """Attach executed Python source lines to imperative builder operations.

    Enter this context inside an active :class:`IRBuilder`. Direct statements
    in the with-body and calls to user helpers supply line locations to native
    builder emission and frame construction. Returning from a helper restores
    its caller's line, including further operations on the same Python line.
    Explicit ``at_`` locations and ``with_loc`` scopes compose with these lines
    through the builder's ordinary location normalization.

    Returns
    -------
    context : AbstractContextManager[None]
        A synchronous context manager that yields None. Nested contexts share
        one trace callback on the current thread and restore their enclosing
        location state on normal and exceptional exit.

    Raises
    ------
    RuntimeError
        If no IRBuilder is active when the context is entered.

    Notes
    -----
    This uses ``sys.settrace`` call, line, and return events, including local
    tracing of the already-running caller frame. Each location covers one
    executed source line; it does not identify operands, expression columns,
    or every physical line of a multiline expression. Only ordinary existing
    source files are traced. Synthetic filenames (including interactive code),
    TVM and tvm-ffi implementation files, and Python standard-library files
    are excluded. Other installed packages are treated as user helpers; this
    filter does not distinguish third-party DSL implementation from user code.

    Existing global and affected local trace hooks are suspended, then restored
    with their line/opcode settings on exit. They do not receive events during
    the scope; debugger and coverage interoperability is not provided. Do not
    replace trace hooks, yield or await from inside this context, or
    share it across threads. Tracing applies only to the builder active at
    entry; use another nested context to trace a different nested builder.
    Raw IR constructors keep their usual behavior: tracing supplies builder
    context and does not automatically annotate arbitrary Python return values.

    Examples
    --------
    .. code-block:: python

        from tvm.script import ir_builder
        from tvm.script.ir_builder import IRBuilder
        from tvm.tirx.script import ir_builder as T

        with IRBuilder() as builder:
            with ir_builder.line_trace():
                T.evaluate(1)
        statement = builder.get()
    """
    return _LineTrace()
