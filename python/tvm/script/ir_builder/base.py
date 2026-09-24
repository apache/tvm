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
from functools import wraps
from typing import Any, Generic, TypeVar

from tvm_ffi import register_object as _register_object
from tvm_ffi.dataclasses import MISSING

from tvm import ir
from tvm.runtime import Object as _Object

from . import _ffi_api


@_register_object("script.ir_builder.IRBuilderFrame")
class IRBuilderFrame(_Object):
    """A stack frame of the IRBuilder used to keep track of the current scope.

    A language variant supplies frame subclasses that retain the information
    needed for context-dependent construction. Entering a frame pushes it onto
    the active builder's stack; nested operations can inspect that stack to
    find their enclosing scope. Normal exit finalizes the frame and runs its
    registered callbacks.
    """

    def __enter__(self) -> "IRBuilderFrame":
        _ffi_api.IRBuilderFrameEnter(self)  # type: ignore[attr-defined] # pylint: disable=no-member
        return self

    def __exit__(self, exc_type, exc_value, trace) -> None:  # pylint: disable=unused-argument
        if exc_type is None and exc_value is None:
            # Do not execute `FrameExit` if the with scope exits because of exceptions
            _ffi_api.IRBuilderFrameExit(self)  # type: ignore[attr-defined] # pylint: disable=no-member

    def add_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback method invoked when exiting the with-scope.

        Parameters
        ----------
        callback : Callable[[], None]
            The callback method to be invoked.
        """
        _ffi_api.IRBuilderFrameAddCallback(  # type: ignore[attr-defined] # pylint: disable=no-member
            self, callback
        )


@_register_object("script.ir_builder.IRBuilder")
class IRBuilder(_Object):
    """A shared construction context for any language variant's IR.

    Examples
    --------
    Enter the builder before using a language variant's construction frames.
    Each completed frame contributes to the result returned by ``get``.

    .. code-block:: python

        from tvm.script.ir_builder import IRBuilder

        def build(frame, emit_body):
            with IRBuilder() as builder:
                with frame:
                    emit_body()
            return builder.get()
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
        """Get the current IRBuilder put in the with-scope.

        Returns
        -------
        builder : IRBuilder
            The current IRBuilder.
        """
        return _ffi_api.IRBuilderCurrent()  # type: ignore[attr-defined] # pylint: disable=no-member

    @staticmethod
    def is_in_scope() -> bool:
        """See if the current thread-local scope has an IRBuilder.

        Returns
        -------
        bool
            Whether the current thread-local scope has an IRBuilder
        """
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


_T = TypeVar("_T")


class SpanEntry:
    """A materialized source range shared by generated builder operations.

    Entries retain only fixed source metadata. Calling an entry attaches its
    span to the same result; ``ctx(thunk)`` additionally supplies call provenance
    during evaluation. ``ctx(thunk, attach_result=False)`` supplies only the
    evaluation context, leaving result attachment to the binding operation.
    Both compose the active caller context at invocation.
    Builders accepting an explicit span normalize an entry with ``source_span``.
    """

    __slots__ = ("span",)

    def __init__(self, span: ir.Span) -> None:
        self.span = span

    def __call__(self, value: _T) -> _T:
        """Attach this range to the same value, receipt, or native frame."""
        return at(self.span, value)

    def ctx(self, thunk: Callable[[], _T], *, attach_result: bool = True) -> _T:
        """Evaluate once under this range, optionally attaching it to the result.

        Context is restored even on failure. Frames constructed during the call
        retain their native construction span regardless of result attachment.
        """
        return with_at_group_(self.span, thunk, attach_result=attach_result)


def source_span(
    location: SpanEntry | ir.Span | tuple[str | ir.SourceName, int, int, int, int] | None,
) -> ir.Span | None:
    """Normalize an entry or materialize a range without retaining source-unit state."""
    if isinstance(location, SpanEntry):
        return location.span
    if location is None or isinstance(location, ir.Span | ir.SequentialSpan):
        return location
    source_name, line, end_line, column, end_column = location
    if isinstance(source_name, str):
        source_name = ir.SourceName(source_name)
    return ir.Span(source_name, line, end_line, column, end_column)


class AlreadyEmitted(Generic[_T]):
    """Hold the exact emitted value; location handling preserves this receipt.

    Parameters
    ----------
    value : Any
        Object already emitted by a language variant's builder. The receipt
        retains this object without copying it. Binding or emitting the receipt
        must not emit the object again.

    Attributes
    ----------
    value : Any
        The same emitted object, available for identity checks and source-span
        attachment.
    """

    __slots__ = ("value",)

    def __init__(self, value: _T) -> None:
        self.value = value


def at(
    span: SpanEntry | ir.Span | tuple[str | ir.SourceName, int, int, int, int] | None, value: _T
) -> _T:
    """Attach source context to the same IR node, emission receipt, or frame.

    Parameters
    ----------
    span : SpanEntry, Span, tuple or None
        Source location to compose with the active construction context. A tuple
        contains ``(source_name, line, end_line, column, end_column)``; its source
        name may be a string or SourceName. None leaves the value unchanged.
    value : Any
        Native object, :class:`AlreadyEmitted` receipt, or list/tuple of native
        objects to annotate. A receipt's contained object receives the span.

    Returns
    -------
    Any
        The exact ``value`` object, including its original receipt or container.
        With no active builder, the value is returned without modification.

    Notes
    -----
    Native mutation annotates the statement held by the builder itself.  Keep
    the original Python facade as well, including callable objects and frames.
    Unsupported objects and ordinary Python values pass through unchanged.
    """
    if span is None or not IRBuilder.is_in_scope():
        return value
    target = value.value if isinstance(value, AlreadyEmitted) else value
    targets = target if isinstance(target, list | tuple) else (target,)
    for item in targets:
        if isinstance(item, _Object):
            _ffi_api.IRBuilderSetSourceSpan(IRBuilder.current(), item, source_span(span))
    return value


def require_defined(value, name):
    """Report a source name whose designated region output was not produced.

    Parameters
    ----------
    value : Any
        Candidate binding. Only the canonical ``MISSING`` singleton denotes an
        absent value; explicit None is a defined value.
    name : str
        Source identifier to include in the missing-name diagnostic.

    Returns
    -------
    Any
        The exact ``value`` when it is defined.

    Raises
    ------
    NameError
        If ``value`` is ``MISSING``.
    """
    if value is MISSING:
        raise NameError(f"name {name!r} is not defined")
    return value


def with_at_group_(
    location: SpanEntry | ir.Span | tuple[str | ir.SourceName, int, int, int, int] | None,
    thunk: Callable[[], _T],
    *,
    attach_result: bool = True,
) -> _T:
    """Evaluate once under a location, optionally attaching it to the same result.

    Parameters
    ----------
    location : SpanEntry, Span, tuple or None
        Source context for the call. A tuple contains
        ``(source_name, line, end_line, column, end_column)``. None, or the
        absence of an active builder, leaves construction context unchanged.
    thunk : Callable[[], Any]
        Zero-argument callable evaluated exactly once inside that context.
    attach_result : bool, optional
        Attach the location to the returned object with :func:`at_`. Defaults
        to True. False supplies construction context only, leaving explicit
        result attachment to a later operation.

    Returns
    -------
    Any
        The exact result of ``thunk``, with its receipt or container preserved.

    Notes
    -----
    The prior source context is restored even if the callable raises; its
    exception propagates unchanged. Frames created during the call retain their
    construction spans regardless of ``attach_result``.
    """
    span = source_span(location)
    context = (
        IRBuilder.current().with_source_span(span)
        if span is not None and IRBuilder.is_in_scope()
        else nullcontext()
    )
    with context:
        value = thunk()
        return at(location, value) if attach_result else value


at_ = at


def _resolve_type_var(frame, ffi_resolver, name, dtype=None, *, value=None, span=None):
    """Validate a native function resolver's inputs, leaving its map owned by C++."""
    if not isinstance(name, str) or not name:
        raise ValueError("A symbolic variable requires a nonempty string name")
    if isinstance(dtype, ir.Var):
        value, dtype = dtype, dtype.ty
    if isinstance(dtype, str):
        dtype = ir.PrimType(dtype)
    if dtype is not None and not isinstance(dtype, ir.PrimType):
        raise TypeError("A symbolic variable requires a primitive type")
    if value is not None and not ir.is_prim_var(value):
        raise TypeError("A symbolic binding requires a primitive Var")
    return ffi_resolver(frame, name, dtype, value, source_span(span))


def _current_function_frame():
    """Find the nearest function for eager shared annotation constructors."""
    if IRBuilder.is_in_scope():
        for frame in reversed(IRBuilder.current().frames):
            if callable(getattr(frame, "resolve_type_var", None)):
                return frame
    raise ValueError("Symbol resolution requires an active function frame")


def wrap_expression_constructor(constructor, call_signature, policy, *, as_type=False):
    """Adapt an eager constructor using parser-owned expression-string metadata."""
    fields = policy.fields

    def unresolved(value, nested=False):
        if isinstance(value, str):
            return nested or policy.scalar_strings
        if isinstance(value, TypeVar):
            return True
        if isinstance(value, tuple | list):
            return any(unresolved(item, True) for item in value)
        return False

    @wraps(constructor)
    def invoke(*args, **kwargs):
        bound = call_signature.bind(*args, **kwargs)
        if IRBuilder.is_in_scope():
            # typing.TypeVar is ordinary eager Python metadata. Resolve it
            # here, never in the syntax-only transpiler.
            def resolve(value):
                if isinstance(value, TypeVar):
                    if value.__bound__ is not None or value.__constraints__:
                        raise TypeError("A symbolic TypeVar cannot have constraints or a bound")
                    return _current_function_frame().resolve_type_var(value.__name__)
                if isinstance(value, tuple):
                    return tuple(resolve(item) for item in value)
                if isinstance(value, list):
                    return [resolve(item) for item in value]
                return value

            for field in fields:
                if field in bound.arguments:
                    bound.arguments[field] = resolve(bound.arguments[field])
        if any(unresolved(bound.arguments[field]) for field in fields if field in bound.arguments):
            if IRBuilder.is_in_scope():
                raise TypeError(
                    "Builder expression arguments require concrete symbols, not strings"
                )
            return ir.Type.missing()
        return constructor(*bound.args, **bound.kwargs)

    result = invoke
    if as_type:
        # The class is an annotation surface, not an IR or proxy type.
        # __new__ returns the concrete construction result (or MissingType).
        result = type(
            constructor.__name__,
            (),
            {
                "__new__": lambda cls, *args, **kwargs: invoke(*args, **kwargs),
                "__signature__": call_signature,
                "__doc__": constructor.__doc__,
                "__module__": constructor.__module__,
            },
        )
    return result


def _return_annotation(annotation):
    """Evaluate a return annotation without introducing return-only symbols."""
    if not callable(annotation) or isinstance(annotation, ir.Expr | ir.Type):
        return annotation
    frame = _current_function_frame()
    declared = set(frame.type_var_map)
    annotation = annotation()
    introduced = set(frame.type_var_map) - declared
    if introduced:
        raise ValueError(f"Return annotation introduces unbound symbol {sorted(introduced)[0]!r}")
    return annotation


def annotation_value_(name, value):
    """Adapt a real definition-context symbol using the native function map.

    Parameters
    ----------
    name : str
        Source spelling used to resolve the symbol in the nearest active
        function frame.
    value : TypeVar, Var or Any
        Captured definition-context value. An unconstrained ``typing.TypeVar``
        resolves a symbolic variable; a primitive IR Var supplies its existing
        value to the resolver. Other values pass through unchanged.

    Returns
    -------
    Any
        The function frame's resolved symbol, or the unchanged nonsymbolic value.

    Raises
    ------
    TypeError
        If a TypeVar has a bound or constraints.
    ValueError
        If a symbolic value requires resolution without an active function frame.
    """
    if isinstance(value, TypeVar):
        if value.__bound__ is not None or value.__constraints__:
            raise TypeError("A symbolic TypeVar cannot have constraints or a bound")
        return _current_function_frame().resolve_type_var(name)
    if ir.is_prim_var(value):
        return _current_function_frame().resolve_type_var(name, value=value)
    return value
