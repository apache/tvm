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
from inspect import signature
from typing import Any, Generic, TypeVar

from tvm_ffi import register_object as _register_object
from tvm_ffi.dataclasses import MISSING as MISSING

from tvm import ir
from tvm.runtime import Object as _Object

from . import _ffi_api


def resolve_global_info_args(
    *fields: str, resolver: Callable[[str], Any]
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Resolve selected string arguments before calling a builder operation.

    Parameters
    ----------
    *fields : str
        Names of positional-only, positional-or-keyword, or keyword-only parameters.
        Repeated names are resolved once. Omitted arguments use their declared defaults.
    resolver : Callable[[str], Any]
        Explicit callback receiving each selected string and returning its replacement.
        The callback owns selector syntax, lookup scope, and errors. Non-string values
        pass through with their identity preserved; containers are not decoded recursively.

    Returns
    -------
    Callable
        Decorator preserving the callable's signature, name, and documentation. The
        signature is inspected once when decorating, then reused for argument binding.

    Raises
    ------
    ValueError
        If a selected name is absent or names a variadic parameter.

    Notes
    -----
    All argument expressions are evaluated once in ordinary Python order before binding,
    resolution, and the callable body. Positional, keyword, unpacked, and aliased calls
    share this behavior. Selected string defaults are resolved on every call. Exceptions
    from binding, the resolver, and the callable propagate unchanged.

    .. code:: python

        @resolve_global_info_args("device", resolver=lookup_device)
        def tensor(shape, device="default"):
            return make_tensor(shape, device)
    """
    fields = tuple(dict.fromkeys(fields))

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        call_signature = signature(function)
        for field in fields:
            parameter = call_signature.parameters.get(field)
            if parameter is None or parameter.kind in (
                parameter.VAR_POSITIONAL,
                parameter.VAR_KEYWORD,
            ):
                raise ValueError(f"Unknown or variadic global-info argument: {field!r}")

        @wraps(function)
        def invoke(*args, **kwargs):
            bound = call_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            for field in fields:
                value = bound.arguments[field]
                if isinstance(value, str):
                    bound.arguments[field] = resolver(value)
            return function(*bound.args, **bound.kwargs)

        return invoke

    return decorate


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


class SpanEntry:
    """A materialized source range shared by generated builder operations.

    Entries retain only fixed source metadata. Calling an entry attaches its
    span to the same result; ``ctx(thunk)`` additionally supplies call provenance
    during evaluation. ``ctx(thunk, attach_result=False)`` supplies only the
    evaluation context, leaving result attachment to the binding operation.
    Both compose the active caller context at invocation.
    Builders accepting an explicit span unwrap the entry at native boundaries.
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


def at(span: SpanEntry | ir.Span | None, value: _T) -> _T:
    """Attach source context to the same IR node, emission receipt, or frame.

    Parameters
    ----------
    span : SpanEntry, Span or None
        Source location to compose with the active construction context.
        None leaves the value unchanged.
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
    if isinstance(span, SpanEntry):
        span = span.span
    target = value.value if isinstance(value, AlreadyEmitted) else value
    targets = target if isinstance(target, list | tuple) else (target,)
    for item in targets:
        if isinstance(item, _Object):
            _ffi_api.IRBuilderSetSourceSpan(IRBuilder.current(), item, span)
    return value


def with_at_group_(
    location: SpanEntry | ir.Span | None,
    thunk: Callable[[], _T],
    *,
    attach_result: bool = True,
) -> _T:
    """Evaluate once under a location, optionally attaching it to the same result.

    Parameters
    ----------
    location : SpanEntry, Span or None
        Source context for the call. None, or the absence of an active builder,
        leaves construction context unchanged.
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
    span = location.span if isinstance(location, SpanEntry) else location
    context = (
        IRBuilder.current().with_source_span(span)
        if span is not None and IRBuilder.is_in_scope()
        else nullcontext()
    )
    with context:
        value = thunk()
        return at(span, value) if attach_result else value


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
    return ffi_resolver(
        frame, name, dtype, value, span.span if isinstance(span, SpanEntry) else span
    )


def _current_function_frame():
    """Find the nearest native function frame for explicit symbol declarations."""
    if IRBuilder.is_in_scope():
        for frame in reversed(IRBuilder.current().frames):
            if callable(getattr(frame, "resolve_type_var", None)):
                return frame
    raise ValueError("Symbol resolution requires an active function frame")


def _return_annotation(annotation):
    """Evaluate the deferred return expression before normalizing its annotation value."""
    if callable(annotation) and not isinstance(annotation, ir.Expr | ir.Type):
        return annotation()
    return annotation


def annotation_constructor(constructor):
    """Expose a constructor as a real annotation class supporting Python unions.

    Calls construct ordinary native values directly. The class preserves the
    constructor's signature and documentation without adapting its arguments.
    """
    return type(
        constructor.__name__,
        (),
        {
            "__new__": lambda cls, *args, **kwargs: constructor(*args, **kwargs),
            "__signature__": signature(constructor),
            "__doc__": constructor.__doc__,
            "__module__": constructor.__module__,
        },
    )
