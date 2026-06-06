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
"""The entry point of TVM parser for tirx."""

import inspect
from collections.abc import Callable
from typing import Any

from tvm.ir.base import deprecated
from tvm.script.parser._core import parse, scan_macro, utils
from tvm.script.parser.core.parser import Parser, ScriptMacro, VarTable
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.script.builder import block_name_suffix_context, buffer, ptr


def prim_func(
    func: Callable | None = None,
    private: bool = False,
    check_well_formed=True,
    s_tir: bool = False,
    persistent: bool = False,
) -> PrimFunc | Callable:
    """The parsing method for tirx prim func, by using `@prim_func` as decorator.

    Parameters
    ----------
    func : Callable
        The function to be parsed as prim func.
        (Listed as optional to allow the decorator to be used
        without arguments, like `@prim_func`,
        or with an argument, `@prim_func(private=True)`)

    private : bool, optional
        Whether the function should be treated as private.
        A private function has no global symbol attribute;
        if the function is not private, it will have a global symbol
        matching the function name.

    Returns
    -------
    res : Union[PrimFunc, Callable]
        The parsed tirx prim func.
    """
    # pylint: disable=unused-argument
    # (private will be used in the parser, but not immediately)

    # need to capture this var outside the wrapper because the wrapper
    # adds to the stack
    outer_stack = inspect.stack()

    def decorator_wrapper(func):
        if not inspect.isfunction(func):
            raise TypeError(f"Expect a function, but got: {func}")
        if utils.is_defined_in_class(outer_stack, func):
            return func
        extra_vars = utils.inspect_function_capture(func)
        utils.resolve_closure_vars(func, extra_vars, outer_stack)
        f = parse(func, extra_vars, check_well_formed=check_well_formed, s_tir=s_tir)
        setattr(f, "__name__", func.__name__)
        return f

    if func is not None:
        # no optional args given => use wrapper directly
        return decorator_wrapper(func)
    else:
        # if there is an optional arg given, return a new decorator
        # that will then be invoked
        setattr(decorator_wrapper, "dispatch_token", "tirx")
        return decorator_wrapper


setattr(prim_func, "dispatch_token", "tirx")


class TIRInline(ScriptMacro):
    """Specialization of ScriptMacro for TIR with Python LEGB scoping.

    Two definition paths:
    1. Outside @T.prim_func (standalone @T.inline): definition_depth is None,
       closure_vars captured at definition time are used (module globals are
       effectively late-bound since they don't change during parsing).
    2. Inside @T.prim_func (inline def in parsed body): definition_depth is set
       to the VarTable frame depth at definition time, and defining_var_table
       stores a reference to the VarTable that was active. At call time,
       defining_var_table.get_at_depth(definition_depth) reads current values
       from the lexically enclosing frames.

    Attributes
    ----------
    definition_depth : Optional[int]
        VarTable frame depth at definition time, or None for outside-prim_func.
    defining_var_table : Optional[VarTable]
        Reference to the VarTable that was active at definition time.
    call_count : int
        Counter for unique block name suffixes.
    """

    def __init__(
        self,
        source,
        closure_vars: dict[str, Any],
        func: Callable,
        definition_depth: int | None = None,
        defining_var_table: VarTable | None = None,
    ) -> None:
        # hygienic=True for the base class (field kept for compat but not used in dispatch)
        super().__init__(source, closure_vars, func, hygienic=True)
        self.definition_depth = definition_depth
        self.defining_var_table = defining_var_table
        self.call_count = 0

    def parse_macro(self, parser: Parser) -> None:
        macro_def = self.get_macro_def()
        suffix = f"_{self.call_count}" if self.call_count > 0 else ""
        self.call_count += 1
        with block_name_suffix_context(suffix):
            parser.visit_body(macro_def.body)

    def __call__(self, *args, **kwargs):
        param_binding = inspect.signature(self.func).bind(*args, **kwargs)
        param_binding.apply_defaults()
        local_vars = param_binding.arguments
        parser = self._find_parser_def()

        with parser.with_diag_source(self.source):
            if self.defining_var_table is not None:
                # Inside-prim_func path: LEGB late binding from the defining scope
                enclosing_vars = self.defining_var_table.get_at_depth(self.definition_depth)
            else:
                # Outside-prim_func path: use captured closure vars
                enclosing_vars = self.closure_vars

            saved_var_table = parser.var_table
            parser.var_table = VarTable()

            with parser.var_table.with_frame():
                for k, v in enclosing_vars.items():
                    parser.var_table.add(k, v)
                with parser.var_table.with_frame():
                    for k, v in local_vars.items():
                        parser.var_table.add(k, v)

                    parse_result = self.parse_macro(parser)

            parser.var_table = saved_var_table

        return parse_result


def inline(*args, definition_depth: int | None = None, defining_var_table=None) -> Callable:
    """Decorator for inline function definitions with Python LEGB scoping.

    @T.inline follows Python's lexical scoping with late binding:
    - At definition time, record which scopes are visible.
    - At call time, read current values from those scopes.

    Example::

        import tvm
        from tvm.script import tirx as T

        x_value = 128

        @T.inline
        def capture(A, B):
            B[()] = A[x_value]          # x_value resolved from enclosing scope

        @T.prim_func(s_tir=True)
        def use(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
            capture(A, B)               # Produces B[()] = A[128]
    """

    def _decorator(func: Callable) -> Callable:
        source, closure_vars = scan_macro(func, utils.inspect_function_capture(func))
        obj = TIRInline(
            source,
            closure_vars,
            func,
            definition_depth=definition_depth,
            defining_var_table=defining_var_table,
        )

        def wrapper(*args, **kwargs):
            return obj(*args, **kwargs)

        return wrapper

    if len(args) == 0:
        setattr(_decorator, "dispatch_token", "tir.inline")
        return _decorator
    if len(args) == 1 and inspect.isfunction(args[0]):
        return _decorator(args[0])

    raise ValueError("Invalid use of T.inline. Usage: @T.inline or @T.inline()")


setattr(inline, "dispatch_token", "tir.inline")


class TIRJit:
    """Top-level kernel decorator with constexpr params + ``.specialize()``.

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
        func: Callable,
        check_well_formed: bool = True,
        is_stir: bool = False,
        persistent: bool = False,
        private: bool = False,
    ) -> None:
        self.func = func
        self.check_well_formed = check_well_formed
        self.is_stir = is_stir
        self.persistent = persistent  # pylint: disable=unused-private-member
        self.private = private  # pylint: disable=unused-private-member
        # Resolved closure vars (computed once; the function itself is the
        # capture point, so this never changes between specializations).
        self._closure_vars: dict[str, Any] = utils.inspect_function_capture(func)
        # Detect which params are marked T.constexpr. With PEP 563
        # (``from __future__ import annotations``), each annotation is a
        # string; we eval them one-by-one so a constexpr probe is not
        # blocked by sibling annotations that reference yet-undefined names
        # (e.g. ``A: T.Buffer((N,), ...)`` referencing constexpr ``N``).
        raw_anns = getattr(func, "__annotations__", {}) or {}
        eval_globals = {**func.__globals__, **self._closure_vars}
        sig = inspect.signature(func)
        constexpr_names: set[str] = set()
        constexpr_defaults: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            ann = raw_anns.get(name)
            if isinstance(ann, str):
                try:
                    ann = eval(ann, eval_globals)  # pylint: disable=eval-used
                except Exception:  # pylint: disable=broad-except
                    ann = None
            if ann is constexpr:
                constexpr_names.add(name)
                if param.default is not inspect.Parameter.empty:
                    constexpr_defaults[name] = param.default
        self.constexpr_names: frozenset[str] = frozenset(constexpr_names)
        self.constexpr_defaults: dict[str, Any] = constexpr_defaults
        self._cache: dict[tuple, PrimFunc] = {}

    def specialize(self, **constexpr_kwargs) -> PrimFunc:
        """Build a concrete PrimFunc by binding the constexpr params.

        Parameters
        ----------
        **constexpr_kwargs
            One value per ``T.constexpr``-annotated parameter. All such
            parameters must be supplied; passing names that are not
            constexpr-annotated is an error.

        Returns
        -------
        PrimFunc
            A concrete TIRx PrimFunc, identical in type to the output of
            ``@T.prim_func``.
        """
        extra = constexpr_kwargs.keys() - self.constexpr_names
        if extra:
            raise TypeError(
                f"{self.func.__name__}.specialize() got unexpected arg(s): "
                f"{sorted(extra)} (constexpr params are: {sorted(self.constexpr_names)})"
            )
        effective = {**self.constexpr_defaults, **constexpr_kwargs}
        missing = self.constexpr_names - effective.keys()
        if missing:
            raise TypeError(
                f"{self.func.__name__}.specialize() missing constexpr arg(s) "
                f"(no default provided): {sorted(missing)}"
            )

        try:
            cache_key = tuple(sorted(effective.items()))
            cached = self._cache.get(cache_key)
        except TypeError as err:
            raise TypeError(
                f"{self.func.__name__}.specialize(): all constexpr values must "
                f"be hashable (got: {effective!r})"
            ) from err
        if cached is not None:
            return cached

        extra_vars = {**self._closure_vars, **effective}
        prim_func = parse(
            self.func,
            extra_vars,
            check_well_formed=self.check_well_formed,
            s_tir=self.is_stir,
        )
        setattr(prim_func, "__name__", self.func.__name__)
        self._cache[cache_key] = prim_func
        return prim_func


def jit(
    func: Callable | None = None,
    private: bool = False,
    check_well_formed: bool = True,
    is_stir: bool = False,
    persistent: bool = False,
) -> "TIRJit | Callable":
    """Decorator: capture the kernel and defer parsing until ``.specialize()``.

    Use ``@T.jit`` (instead of ``@T.prim_func``) when the kernel takes
    compile-time parameters annotated with ``T.constexpr``. The resulting
    object exposes ``.specialize(**constexpr_kwargs)``, which returns a
    ``tvm.tirx.PrimFunc``.

    Example::

        from tvm.script import tirx as T

        @T.jit
        def add(
            A: T.Buffer((N,), "float32"),
            B: T.Buffer((N,), "float32"),
            *,
            N: T.constexpr,
        ):
            ...

        kernel = add.specialize(N=1024)  # returns a PrimFunc
    """

    def decorator_wrapper(func: Callable) -> TIRJit:
        if not inspect.isfunction(func):
            raise TypeError(f"Expect a function, but got: {func}")
        return TIRJit(
            func,
            check_well_formed=check_well_formed,
            is_stir=is_stir,
            persistent=persistent,
            private=private,
        )

    if func is not None:
        return decorator_wrapper(func)
    setattr(decorator_wrapper, "dispatch_token", "tirx")
    return decorator_wrapper


setattr(jit, "dispatch_token", "tirx")


class TIRMacro(ScriptMacro):
    """Specialization of the ScriptMacro class for TIR.

    Apache-compatible hygienic macro. Distinct from ``TIRInline`` (which
    uses Python LEGB late binding) so upstream code that relies on
    capture-at-definition-time semantics keeps working.

    Attributes
    ----------
    call_count : int
        Counter for the number of times this macro has been invoked.
        Used to generate unique block name suffixes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0

    def parse_macro(self, parser: Parser) -> None:
        macro_def = self.get_macro_def()
        suffix = f"_{self.call_count}" if self.call_count > 0 else ""
        self.call_count += 1
        with block_name_suffix_context(suffix):
            parser.visit_body(macro_def.body)


def macro(*args, hygienic: bool = True) -> Callable:
    """Decorator for macro definitions with hygienic capture.

    Parameters
    ----------
    hygienic: bool
        Specifies whether the macro is hygienic or not. A hygienic macro
        resolves symbols at definition time; a non-hygienic macro at use
        time. Defaults to ``True``.
    """

    def _decorator(func: Callable) -> TIRMacro:
        source, closure_vars = scan_macro(func, utils.inspect_function_capture(func))
        obj = TIRMacro(source, closure_vars, func, hygienic)

        def wrapper(*args, **kwargs):
            return obj(*args, **kwargs)

        return wrapper

    if len(args) == 0:
        return _decorator
    if len(args) == 1 and inspect.isfunction(args[0]):
        return _decorator(args[0])

    raise ValueError("Invalid use of T.macro. Usage: @T.macro or @T.macro()")


setattr(macro, "dispatch_token", "tir.macro")


class BufferProxy:
    """Buffer proxy class for constructing tirx buffer."""

    def __or__(self, other):
        """Support ``T.Buffer | None`` union syntax in annotations."""
        return self

    def __ror__(self, other):
        """Support ``None | T.Buffer`` union syntax in annotations."""
        return self

    def __call__(
        self,
        shape,
        dtype="float32",
        data=None,
        strides=None,
        elem_offset=None,
        byte_offset=None,
        scope="global",
        align=0,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
        layout="default",
    ) -> Buffer:
        return buffer(
            shape,
            dtype=dtype,
            data=data,
            strides=strides,
            elem_offset=elem_offset,
            byte_offset=byte_offset,
            scope=scope,
            align=align,
            offset_factor=offset_factor,
            buffer_type=buffer_type,
            axis_separators=axis_separators,
            layout=layout,
        )

    @deprecated("T.Buffer[...]", "T.Buffer(...)")
    def __getitem__(self, keys) -> Buffer:
        if not isinstance(keys, tuple):
            return self(keys)
        if len(keys) >= 2 and not isinstance(keys[1], str):
            return self(keys)
        return self(*keys)  # type: ignore[attr-defined] # pylint: disable=no-member


class PtrProxy:
    """Ptr proxy class for constructing tirx pointer."""

    def __or__(self, other):
        """Support union syntax in annotations."""
        return self

    def __ror__(self, other):
        """Support union syntax in annotations."""
        return self

    @deprecated("T.Ptr(...)", "T.handle(...)")
    def __call__(self, dtype, storage_scope="global"):
        if callable(dtype):
            dtype = dtype().dtype
        return ptr(dtype, storage_scope)  # type: ignore[attr-defined] # pylint: disable=no-member

    @deprecated("T.Ptr[...]", "T.handle(...)")
    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            return self(keys)
        return self(*keys)


class _ConstexprProxy:
    """Sentinel marker for compile-time (specialization-time) parameters.

    Used as a parameter annotation in ``@T.jit`` decorated functions to mark
    a parameter as constexpr — its value is supplied to ``.specialize(**kwargs)``
    rather than at call time, and it is removed from the generated PrimFunc's
    runtime parameter list.
    """

    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self


Buffer = BufferProxy()  # pylint: disable=invalid-name
Ptr = PtrProxy()  # pylint: disable=invalid-name
constexpr = _ConstexprProxy()  # pylint: disable=invalid-name
