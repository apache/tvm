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
from typing import Any, Callable, List

from tvm._ffi import register_object as _register_object
from tvm.runtime import Object as _Object

from . import _ffi_api


@_register_object("script.ir_builder.IRBuilderFrame")
class IRBuilderFrame(_Object):
    """A stack frame of the IRBuilder used to keep track of the current scope.
    Furthermore, the information stored in each stack frame can be useful for context-dependent
    IR construction.

    Examples
    --------

    The `T.match_buffer` below instead an element in the buffer map of `PrimFuncFrame`:

    .. code-block:: python

    from tvm.script.ir_builder import tir as T
    from tvm.script.ir_builder import IRBuilder

    with IRBuilder() as builder:
        with T.prim_func(...):  # pushes a PrimFuncFrame (subclass of IRBuilderFrame)
                                # to `builder`'s stack of frames
            buffer = T.match_buffer(...)


    The `T.match_buffer` below instead generates `MatchBufferRegion` in a TIR block:

    .. code-block:: python

    from tvm.script.ir_builder import tir as T
    from tvm.script.ir_builder import IRBuilder

    with IRBuilder() as builder:
        with T.prim_func(...):  # pushes a PrimFuncFrame (subclass of IRBuilderFrame)
                                # to `builder`'s stack of frames
            with T.block(...):  # pushes a BlockFrame (subclass of IRBuilderFrame)
                                # to `builder`'s stack of frames
                buffer = T.match_buffer(...)
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
    """A dialect-agnostic IRBuilder that constructs any IR of TVM.

    Examples
    --------
    An idiomatic use of this class is to put this inside the with-scope,
    call dialect-specific methods accordingly. Upon exiting the scope.

    .. code-block:: python
    from tvm.script.ir_builder import tir as T
    from tvm.script.ir_builder import IRBuilder

    with IRBuilder() as builder:
        with T.prim_func(...):  # pushes a PrimFuncFrame (subclass of IRBuilderFrame)
                                # to `builder`'s stack of frames
            buffer = T.match_buffer(...)

    return builder.get()        # returns the constructed IR, i.e. tir.PrimFunc
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
        s: List[str],
        vs: List[Any],
    ) -> List[Any]:
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
