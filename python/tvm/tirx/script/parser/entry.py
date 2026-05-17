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


Buffer = BufferProxy()  # pylint: disable=invalid-name
Ptr = PtrProxy()  # pylint: disable=invalid-name
