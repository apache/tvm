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
"""A generic IRBuilder across the TVM stack"""

from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import Any

from tvm_ffi import register_object as _register_object

from tvm import ir
from tvm.runtime import Object as _Object

from . import _ffi_api


@_register_object("script.ir_builder.IRBuilderFrame")
class IRBuilderFrame(_Object):
    """A stack frame of the IRBuilder used to keep track of the current scope.

    Furthermore, the information stored in each stack frame can be useful for context-dependent
    IR construction.

    Examples
    --------

    The `T.match_buffer` below matches a function parameter in the active `PrimFuncFrame`:

    .. code-block:: python

        from tvm.script.ir_builder import tirx as T
        from tvm.script.ir_builder import IRBuilder

        with IRBuilder() as builder:
            with T.prim_func(...):  # pushes a PrimFuncFrame (subclass of IRBuilderFrame)
                                    # to `builder`'s stack of frames
                buffer = T.match_buffer(...)


    The `T.match_buffer` below instead generates `MatchBufferRegion` in a TIR block:

    .. code-block:: python

        from tvm.script.ir_builder import tirx as T
        from tvm.script.ir_builder import IRBuilder

        with IRBuilder() as builder:
            with T.prim_func(...):  # pushes a PrimFuncFrame (subclass of IRBuilderFrame)
                                    # to `builder`'s stack of frames
                with T.sblock(...):  # pushes an SBlockFrame (subclass of IRBuilderFrame)
                                    # to `builder`'s stack of frames
                    buffer = T.match_buffer(...)
    """

    def __enter__(self) -> "IRBuilderFrame":
        with _construction_span(self.source_span):
            _ffi_api.IRBuilderFrameEnter(self)  # type: ignore[attr-defined] # pylint: disable=no-member
        return self

    def __exit__(self, exc_type, exc_value, trace) -> None:  # pylint: disable=unused-argument
        if exc_type is None and exc_value is None:
            # Do not execute `FrameExit` if the with scope exits because of exceptions
            with _construction_span(self.source_span):
                _ffi_api.IRBuilderFrameExit(self)  # type: ignore[attr-defined] # pylint: disable=no-member

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback method invoked when exiting the with-scope."""
        _ffi_api.IRBuilderFrameAddCallback(  # type: ignore[attr-defined] # pylint: disable=no-member
            self, callback
        )


@_register_object("script.ir_builder.IRBuilder")
class IRBuilder(_Object):
    """A dialect-agnostic IRBuilder that constructs any IR of TVM.

    Examples
    --------
    An idiomatic use of this class is to put this inside the with-scope,
    call dialect-specific methods accordingly. Upon exiting the scope.

    .. code-block:: python

        from tvm.script.ir_builder import tirx as T
        from tvm.script.ir_builder import IRBuilder

        with IRBuilder() as builder:
            with T.prim_func(...):  # pushes a PrimFuncFrame (subclass of IRBuilderFrame)
                                # to `builder`'s stack of frames
                buffer = T.match_buffer(...)

        return builder.get()        # returns the constructed IR, i.e. tirx.PrimFunc
    """

    def __init__(self) -> None:
        """Construct an IRBuilder."""
        self.__init_handle_by_constructor__(
            _ffi_api.IRBuilder  # type: ignore[attr-defined] # pylint: disable=no-member
        )

    def __enter__(self) -> "IRBuilder":
        """Enter the with-scope for IRBuilder, which allows the IRBuilder to be discoverable
        using `IRBuilder.current()`.

        Examples
        --------
        .. code-block:: python

            from tvm.script.ir_builder import IRBuilder

            with IRBuilder() as builder:
                assert IRBuilder.current() == builder

        """
        _ffi_api.IRBuilderEnter(self)  # type: ignore[attr-defined] # pylint: disable=no-member
        return self

    def __exit__(self, ptype, value, trace) -> None:  # pylint: disable=unused-argument
        _ffi_api.IRBuilderExit(self)  # type: ignore[attr-defined] # pylint: disable=no-member

    @staticmethod
    def current() -> "IRBuilder":
        """Get the current IRBuilder put in the with-scope."""
        return _ffi_api.IRBuilderCurrent()  # type: ignore[attr-defined] # pylint: disable=no-member

    @staticmethod
    def is_in_scope() -> bool:
        """See if the current thread-local scope has an IRBuilder."""
        return _ffi_api.IRBuilderIsInScope()  # type: ignore[attr-defined] # pylint: disable=no-member

    def get(self) -> _Object:
        """Get the constructed IR."""
        return _ffi_api.IRBuilderGet(self)  # type: ignore[attr-defined] # pylint: disable=no-member

    @contextmanager
    def with_source_span(self, span):
        """Attach ``span`` to IR nodes constructed in the nested scope.

        Nested scopes are retained as a ``SequentialSpan`` when they describe
        distinct source ranges, such as a TVMScript inline expansion.

        Parameters
        ----------
        span : tvm.ir.Span
            The frontend source range active in the nested scope.
        """
        _ffi_api.IRBuilderPushSourceSpan(  # type: ignore[attr-defined] # pylint: disable=no-member
            self, span
        )
        try:
            yield
        finally:
            _ffi_api.IRBuilderPopSourceSpan(  # type: ignore[attr-defined] # pylint: disable=no-member
                self
            )

    def _set_current_source_span(self, value):
        """Compose the active source span onto the same supported IR node or frame."""
        return _ffi_api.IRBuilderSetCurrentSourceSpan(  # type: ignore[attr-defined] # pylint: disable=no-member
            self, value
        )

    @staticmethod
    def name(s: str, v: Any) -> Any:
        """Set the name of an object.

        Parameters
        ----------
        s : str
            The name of the object.
        v : Any
            The object to name.

        Returns
        -------
        v : Any
            The same object with the name set.
        """
        return _ffi_api.IRBuilderName(s, v)  # type: ignore[attr-defined] # pylint: disable=no-member

    @staticmethod
    def name_many(  # pylint: disable=invalid-name
        s: list[str],
        vs: list[Any],
    ) -> list[Any]:
        """Set the name of a list of objects.

        Parameters
        ----------
        s : List[str]
            The names of the objects.
        vs : List[Any]
            The objects to name.

        Returns
        -------
        vs : List[Any]
            The same objects with the names set.
        """
        assert len(s) == len(vs)
        return [IRBuilder.name(i, v) for i, v in zip(s, vs)]


# Absence is an explicit value; no function data is retained.
class _Missing:
    def __repr__(self):
        return "MISSING"


MISSING = _Missing()


def source_span(location):
    """Materialize a source range without retaining source-unit state."""
    if location is None or isinstance(location, (ir.Span, ir.SequentialSpan)):
        return location
    source_name, line, end_line, column, end_column = location
    if isinstance(source_name, str):
        source_name = ir.SourceName(source_name)
    return ir.Span(source_name, line, end_line, column, end_column)


@contextmanager
def _construction_span(span):
    """Apply a builder operation's span through the existing native span stack."""
    if span is None:
        yield
        return
    span = source_span(span)
    context = (
        IRBuilder.current().with_source_span(span)
        if span is not None and IRBuilder.is_in_scope()
        else nullcontext()
    )
    try:
        with context:
            yield
    except Exception as error:
        if span is not None and not hasattr(error, "__tvm_script_location__"):
            diagnostic_span = span.spans[-1] if isinstance(span, ir.SequentialSpan) else span
            error.__tvm_script_location__ = (
                str(diagnostic_span.source_name.name),
                diagnostic_span.line,
                diagnostic_span.end_line,
                diagnostic_span.column,
                diagnostic_span.end_column,
            )
        raise


class BypassBind:
    """Carry an already-constructed value through assignment without binding it."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class BypassEmit:
    """Reference an already-emitted statement without emitting it again."""

    __slots__ = ("stmt",)

    def __init__(self, stmt):
        self.stmt = stmt


def at(span, value):
    """Attach source context to the same IR node, emission receipt, or frame.

    Native mutation annotates the statement held by the builder itself.  Keep
    the original Python facade as well, including callable objects and frames.
    Unsupported objects and ordinary Python values pass through unchanged.
    """
    if span is None or not IRBuilder.is_in_scope():
        return value
    target = value.stmt if isinstance(value, BypassEmit) else value
    if isinstance(target, BypassBind):
        if isinstance(target.value, (list, tuple)):
            for item in target.value:
                at(span, item)
        else:
            at(span, target.value)
        return value
    if isinstance(target, _Object):
        with _construction_span(span):
            IRBuilder.current()._set_current_source_span(target)
    elif callable(set_source_span := getattr(target, "_set_source_span", None)):
        set_source_span(span)
    return value


def _frame_result(frame, name):
    """Read one explicit export without consulting ambient construction state."""
    if not isinstance(name, str):
        raise TypeError("A frame result name must be a string")
    exports = frame if isinstance(frame, dict) else getattr(frame, "result", {})
    return exports.get(name, MISSING)
