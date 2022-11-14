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
from typing import Callable, Union

from tvm.tir import Buffer, PrimFunc

from ...ir_builder.tir import buffer_decl, ptr
from .._core import parse, utils


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
    return parse(func, utils.inspect_function_capture(func))


setattr(prim_func, "dispatch_token", "tir")


class BufferProxy:
    """Buffer proxy class for constructing tir buffer.
    Overload __call__ and __getitem__ to support syntax as T.Buffer() and T.Buffer[].
    """

    def __call__(
        self,
        shape,
        dtype=None,
        data=None,
        strides=None,
        elem_offset=None,
        scope="global",
        align=0,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
    ) -> Buffer:
        if dtype is None:
            raise ValueError("Data type must be specified when constructing buffer")
        return buffer_decl(
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

    def __getitem__(self, keys) -> Buffer:
        if not isinstance(keys, tuple):
            return self(keys)
        if len(keys) >= 2 and not isinstance(keys[1], str):
            return self(keys)
        return self(*keys)  # pylint: disable=no-member # type: ignore


class PtrProxy:
    """Ptr proxy class for constructing tir pointer.
    Overload __call__ and __getitem__ to support syntax as T.Ptr() and T.Ptr[].
    """

    def __call__(self, dtype, storage_scope="global"):
        if callable(dtype):
            dtype = dtype().dtype
        return ptr(dtype, storage_scope)  # pylint: disable=no-member # type: ignore

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            return self(keys)
        return self(*keys)


Buffer = BufferProxy()  # pylint: disable=invalid-name
Ptr = PtrProxy()  # pylint: disable=invalid-name
