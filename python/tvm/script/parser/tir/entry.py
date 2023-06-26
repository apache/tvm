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
from typing import Callable, Optional, Union

from tvm.ir.base import deprecated
from tvm.tir import Buffer, PrimFunc

from ...ir_builder.tir import buffer, ptr
from .._core import parse, utils


def prim_func(func: Optional[Callable] = None, private: bool = False) -> Union[PrimFunc, Callable]:
    """The parsing method for tir prim func, by using `@prim_func` as decorator.

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
        The parsed tir prim func.
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
        f = parse(func, utils.inspect_function_capture(func))
        setattr(f, "__name__", func.__name__)
        return f

    if func is not None:
        # no optional args given => use wrapper directly
        return decorator_wrapper(func)
    else:
        # if there is an optional arg given, return a new decorator
        # that will then be invoked
        setattr(decorator_wrapper, "dispatch_token", "tir")
        return decorator_wrapper


setattr(prim_func, "dispatch_token", "tir")


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
