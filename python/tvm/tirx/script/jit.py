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
"""Lazy JIT specialization; validation and caches stay outside transpilation."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable, Mapping
from types import FunctionType
from typing import Any

from tvm.script.parser import protocol_registry as protocol
from tvm.tirx import PrimFunc


class OptionalAnnotation:
    """Retain the runtime annotation of a removable JIT parameter."""

    def __init__(self, annotation: Any) -> None:
        self.annotation = annotation

    def __tvm_optional_annotation__(self) -> Any:
        return self.annotation


def make_jit(builder: object) -> Callable[..., Any]:
    """Create a definition-site JIT decorator for a construction namespace.

    Parameters
    ----------
    builder : object
        Namespace associated with the public decorator. The decorated source
        must name its registered namespace; parsing selects construction hooks
        from that syntax rather than accepting a root namespace override.

    Returns
    -------
    Callable[..., Any]
        Decorator accepting a function at its definition site or keyword options
        for that application. Its result defers construction until specialization.
    """

    def jit(
        func: FunctionType | None = None,
        *,
        private: bool = False,
        check_well_formed: bool = True,
        persistent: bool = False,
    ) -> TIRJit | Callable[[FunctionType], TIRJit]:
        """Decorator: capture the kernel and defer parsing until ``.specialize()``.

        Use ``@T.jit`` (instead of ``@T.prim_func``) when the kernel takes
        compile-time parameters annotated with ``T.constexpr`` or runtime
        parameters that may be removed with ``T.Optional``. The resulting object
        exposes ``.specialize(**specialization_kwargs)``, which returns a
        ``tvm.tirx.PrimFunc``.

        Parameters
        ----------
        func : types.FunctionType or None, optional
            Function supplied by a definition-site ``@T.jit`` application. None,
            the default, returns a decorator for ``@T.jit(**options)``.
        private : bool, optional
            Omit the function's public global symbol when True. Default is False.
        check_well_formed : bool, optional
            Validate each constructed specialization. Default is True; this option
            controls the parser separately from the function builder kwargs.
        persistent : bool, optional
            Mark constructed specializations as persistent kernels. Default is False.

        Returns
        -------
        TIRJit or Callable[[types.FunctionType], TIRJit]
            Deferred kernel, or its definition-site decorator when func is None.

        Raises
        ------
        TypeError
            If the decorated value is not a Python function.
        SyntaxError
            If application occurs after definition or uses an unsupported bare or
            preconfigured callable alias instead of a qualified namespace decorator.
        OSError
            If the function's source cannot be recovered.

        Examples
        --------
        Specialize compile-time dimensions before compiling the kernel::

            from __future__ import annotations

            from tvm.script import tirx as T

            @T.jit
            def add(
                A: T.Buffer((N,), "float32"),
                B: T.Buffer((N,), "float32"),
                *,
                N: T.constexpr,
            ):
                for i in T.serial(N):
                    B[i] = A[i] + 1.0

            kernel = add.specialize(N=1024)  # returns a PrimFunc

            @T.jit
            def guarded(optional: T.Optional(T.handle), out: T.handle):
                output = T.match_buffer(out, (1,), "int32")
                if T.constexpr(optional is not None):
                    value = T.match_buffer(optional, (1,), "int32")
                    output[0] = value[0]
                else:
                    output[0] = 0

            present = guarded.specialize()
            absent = guarded.specialize(optional=None)
        """

        def apply(function: FunctionType) -> TIRJit:
            from tvm.script.parser.inspect_source import (
                capture_definition_scope,
                require_definition_site,
            )

            if not inspect.isfunction(function):
                raise TypeError(f"Expect a function, but got: {function}")
            frame = inspect.currentframe().f_back
            try:
                if frame.f_code is jit.__code__:
                    frame = frame.f_back
                require_definition_site(function, frame, jit)
                definition_scope = capture_definition_scope(frame)
            finally:
                del frame
            return TIRJit(
                function,
                check_well_formed,
                persistent,
                private,
                definition_scope=definition_scope,
                builder=builder,
            )

        return apply(func) if func is not None else apply

    return jit


class TIRJit:
    """Top-level kernel decorator with compile-time ``.specialize()`` params.

    Parses the function body lazily: parsing is deferred until ``.specialize()``
    supplies concrete values for the params annotated as ``T.constexpr``. The
    return type of ``.specialize()`` is a ``tvm.tirx.PrimFunc``, identical in
    type to what ``@T.prim_func`` produces today.

    Constexpr params are removed from the resulting PrimFunc's parameter list;
    their values are baked into the IR (e.g. into ``T.Buffer((M, K), ...)``
    shape annotations and into the body).
    """

    def __init__(
        self,
        func: FunctionType,
        check_well_formed: bool = True,
        persistent: bool = False,
        private: bool = False,
        *,
        definition_scope: Mapping[str, Any] | None = None,
        builder: object | None = None,
    ) -> None:
        """Capture the original kernel and its deferred construction options.

        Parameters
        ----------
        func : types.FunctionType
            Original inspectable Python function retained for each specialization.
        check_well_formed : bool, optional
            Validate constructed functions when True, the default.
        persistent : bool, optional
            Mark constructed functions as persistent kernels. Default is False.
        private : bool, optional
            Omit a public global symbol on constructed functions. Default is False.
        definition_scope : Mapping[str, Any] or None, optional
            Definition-site bindings for deferred annotations and decorator namespace
            lookup. None and an empty mapping both add no external bindings; actual
            source globals and closures are captured independently.
        builder : object or None, optional
            Namespace associated with the creating decorator. None is the default;
            construction itself is resolved from qualified source decorator syntax.

        Raises
        ------
        OSError
            If source inspection cannot recover the function.
        SyntaxError
            If the function's source or a quoted annotation cannot be parsed.
        """
        from tvm.script.parser.inspect_source import (
            capture_annotation_bindings,
            capture_lexical_bindings,
        )

        self.builder = builder
        self.func = func
        self.check_well_formed = check_well_formed
        self.persistent = persistent  # pylint: disable=unused-private-member
        self.private = private  # pylint: disable=unused-private-member
        # Resolved closure vars (computed once; the function itself is the
        # capture point, so this never changes between specializations).
        self._closure_vars: dict[str, Any] = capture_lexical_bindings(func)
        # JIT deliberately retains only names read by deferred annotations.
        # A new temporary builder is created for every uncached specialization.
        self._definition_scope: dict[str, Any] = capture_annotation_bindings(
            func, definition_scope or {}
        )
        # Detect which params are marked T.constexpr or T.Optional. With PEP 563
        # (``from __future__ import annotations``), each annotation is a
        # string. Resolve only marker names; annotation constructors execute
        # later in the generated declaration's builder context.
        raw_anns = getattr(func, "__annotations__", {}) or {}
        annotation_scope = {**func.__globals__, **self._closure_vars, **self._definition_scope}

        from tvm.script.parser.prescan import resolve_namespace_value

        sig = inspect.signature(func)
        constexpr_names: set[str] = set()
        constexpr_defaults: dict[str, Any] = {}
        optional_names: set[str] = set()
        for name, param in sig.parameters.items():
            raw_ann = raw_anns.get(name)
            ann = raw_ann
            if isinstance(ann, str):
                try:
                    node = ast.parse(ann, mode="eval").body
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        node = ast.parse(node.value, mode="eval").body
                    ann = resolve_namespace_value(
                        node.func if isinstance(node, ast.Call) else node, annotation_scope
                    )
                    if isinstance(node, ast.Call) and ann is not OptionalAnnotation:
                        ann = None
                except SyntaxError:
                    ann = None
            if ann is protocol.constexpr:
                constexpr_names.add(name)
                if param.default is not inspect.Parameter.empty:
                    constexpr_defaults[name] = param.default
            if ann is OptionalAnnotation or isinstance(ann, OptionalAnnotation):
                optional_names.add(name)
        self.constexpr_names: frozenset[str] = frozenset(constexpr_names)
        self.constexpr_defaults: dict[str, Any] = constexpr_defaults
        self.optional_names: frozenset[str] = frozenset(optional_names)
        self._cache: dict[tuple[tuple[str, type, Any], ...], PrimFunc] = {}

    def specialize(self, **specialization_kwargs: Any) -> PrimFunc:
        """Build a PrimFunc by binding constexprs and absent optional params.

        Parameters
        ----------
        **specialization_kwargs : Any
            One hashable value per ``T.constexpr``-annotated parameter.  A
            ``T.Optional`` parameter may additionally be supplied as ``None``
            to remove it from the resulting PrimFunc ABI.  Omitting an
            optional parameter keeps it as a normal runtime parameter.

        Returns
        -------
        PrimFunc
            A concrete TIRx PrimFunc, identical in type to the output of
            ``@T.prim_func``. Repeated selections reuse the cached function.

        Raises
        ------
        TypeError
            If a name is not specializable, a required constexpr value is absent,
            an optional parameter is assigned anything other than None, or a
            constexpr value is not hashable.

        Notes
        -----
        No supplied kwargs still selects JIT construction with an empty map when
        no constexpr defaults exist. Selected None optional values omit parameters
        and skip their annotations. Source and builder exceptions propagate unchanged.
        """
        from tvm.script.parser.entry import parse

        specializable_names = self.constexpr_names | self.optional_names
        extra = specialization_kwargs.keys() - specializable_names
        if extra:
            raise TypeError(
                f"{self.func.__name__}.specialize() got unexpected arg(s): "
                f"{sorted(extra)} (specializable params are: {sorted(specializable_names)})"
            )
        invalid_optional = {
            name: specialization_kwargs[name]
            for name in self.optional_names & specialization_kwargs.keys()
            if specialization_kwargs[name] is not None
        }
        if invalid_optional:
            raise TypeError(
                f"{self.func.__name__}.specialize(): T.Optional parameters only accept None; "
                f"pass actual tensors when calling the compiled kernel (got: {invalid_optional!r})"
            )

        # One selection contains constexprs and explicit optional None values.
        effective = {**self.constexpr_defaults, **specialization_kwargs}
        missing = self.constexpr_names - effective.keys()
        if missing:
            raise TypeError(
                f"{self.func.__name__}.specialize() missing constexpr arg(s) "
                f"(no default provided): {sorted(missing)}"
            )

        try:
            cache_key = tuple(
                (name, type(value), value) for name, value in sorted(effective.items())
            )
            cached = self._cache.get(cache_key)
        except TypeError as err:
            raise TypeError(
                f"{self.func.__name__}.specialize(): all constexpr values must "
                f"be hashable (got: {effective!r})"
            ) from err
        if cached is not None:
            return cached

        prim_func = parse(
            self.func,
            self._closure_vars,
            definition_scope=self._definition_scope,
            root_function_kwargs={
                "private": self.private,
                "persistent": self.persistent,
            },
            _specialization_bindings=effective,
            check_well_formed=self.check_well_formed,
        )
        setattr(prim_func, "__name__", self.func.__name__)
        self._cache[cache_key] = prim_func
        return prim_func
