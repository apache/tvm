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
"""The entry point of TVM parser for tir."""
import inspect
from typing import Any, Callable, Dict, Union

from tvm.ir.base import deprecated
from tvm.tir import Buffer, PrimFunc

from ...ir_builder.tir import buffer, ptr
from .._core import doc, parse, parse_macro, utils


def prim_func(func: Callable) -> Union[PrimFunc, Callable]:
    """The parsing method for tir prim func, by using `@prim_func` as decorator.

    Parameters
    ----------
    func : Callable
        The function to be parsed as prim func.

    Returns
    -------
    res : Union[PrimFunc, Callable]
        The parsed tir prim func.
    """
    if not inspect.isfunction(func):
        raise TypeError(f"Expect a function, but got: {func}")
    if utils.is_defined_in_class(inspect.stack(), func):
        return func
    f = parse(func, utils.inspect_function_capture(func))
    setattr(f, "__name__", func.__name__)
    return f


setattr(prim_func, "dispatch_token", "tir")


# Semantics of TIR macros:
# - Function that is decorated with @T.macro can have any parameters that
#   follow Python syntax, i.e. positional, keyword, etc. Type annotations
#   are not required, but are allowed.
# - Macro use follows the same syntax as a function call.
#   For `macro_name(arg1, arg2, arg3, ...)`, the values are substituted into
#   the body of the macro, and the body with the substituted values is then
#   inserted at the point where the call to the macro is located.


class TIRMacro:
    """Representation of T.macro."""

    def __init__(
        self,
        source_ast: doc.AST,
        source_txt: str,
        closure_vars: Dict[str, Any],
        func: Callable,
        hygienic: bool,
    ) -> None:
        self.source_ast = source_ast
        self.source_txt = source_txt
        self.closure_vars = closure_vars
        self.func = func
        self.hygienic = hygienic

    def __repr__(self):
        return self.source_txt


def macro(*args, hygienic: bool = True) -> Callable:
    """Decorator for macro definitions.

    Parameters
    ----------
    hygienic: bool
        Specifies whether the macro is hygienic or not.
        A macro is hygienic if all symbols used in the macro's body are resolved
        to values from the location of the macro definition. A non-hygienic macro
        will have its symbols resolved to values at the time of the macro's use.

        Example:
        ```
        import tvm
        from tvm.script import tir as T

        x_value = 128

        @T.macro(hygienic=True)
        def static_capture(A, B):
            B[()] = A[x_value]          ### x_value binds to 128

        @T.macro(hygienic=False)
        def dynamic_capture(A, B):
            B[()] = A[x_value]          ### x_value will bind at the time of use


        @T.prim_func
        def use1(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
            for x_value in T.serial(10):
                static_capture(A, B)    ### Produces B[()] = A[128]

        @T.prim_func
        def use2(A: T.Buffer((1024,), "int32"), B: T.Buffer((), "int32")) -> None:
            for x_value in T.serial(10):
                dynamic_capture(A, B)   ### Produces B[()] = A[x_value]
        ```
    """

    def _decorator(func: Callable) -> TIRMacro:
        source_ast, source_txt, closure_vars = parse_macro(
            func, utils.inspect_function_capture(func)
        )
        obj = TIRMacro(source_ast, source_txt, closure_vars, func, hygienic)
        obj.__name__ = func.__name__
        # We don't need to explicitly store the return value anywhere.
        # This function is a decorator, so the return value will replace
        # the function definition (to which the decorator it is applied)
        # in that function's name space.
        return obj

    if len(args) == 0:
        return _decorator
    if len(args) == 1 and inspect.isfunction(args[0]):
        return _decorator(args[0])

    raise ValueError(
        "Invalid use of T.macro. Usage: @T.macro, @T.macro(), @T.macro(hygienic=[True|False])"
    )


# There is no dispatch_token for macro, because macro doesn't invoke parser.


class BufferProxy:
    """Buffer proxy class for constructing tir buffer."""

    def __call__(
        self,
        shape,
        dtype="float32",
        data=None,
        strides=None,
        elem_offset=None,
        scope="global",
        align=0,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
    ) -> Buffer:
        return buffer(
            shape,
            dtype=dtype,
            data=data,
            strides=strides,
            elem_offset=elem_offset,
            scope=scope,
            align=align,
            offset_factor=offset_factor,
            buffer_type=buffer_type,
            axis_separators=axis_separators,
        )

    @deprecated("T.Buffer[...]", "T.Buffer(...)")
    def __getitem__(self, keys) -> Buffer:
        if not isinstance(keys, tuple):
            return self(keys)
        if len(keys) >= 2 and not isinstance(keys[1], str):
            return self(keys)
        return self(*keys)  # type: ignore[attr-defined] # pylint: disable=no-member


class PtrProxy:
    """Ptr proxy class for constructing tir pointer."""

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
