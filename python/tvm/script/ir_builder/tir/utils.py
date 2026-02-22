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
"""Utility helpers for TIR IRBuilder."""

import contextlib
from typing import List

from tvm import tir
from tvm.tir import Buffer

from . import frame
from . import ir as T


class _FrameScope:
    """Context manager to enter multiple IRBuilder frames without deep nesting.

    This class allows entering multiple frames (e.g., T.allocate) in a single
    `with` statement, avoiding the pyramid of nested context managers.

    Parameters
    ----------
    frames : List[IRBuilderFrame]
        The list of frames to enter.

    Examples
    --------
    Instead of deeply nested allocations:

    .. code-block:: python

        with T.allocate([1], "int32", scope="local") as lo:
            with T.allocate([1], "int32", scope="local") as hi:
                # code here

    Use frame_scope for flat structure:

    .. code-block:: python

        with frame_scope([
            T.allocate([1], "int32", scope="local"),
            T.allocate([1], "int32", scope="local"),
        ]) as (lo, hi):
            # code here
    """

    def __init__(self, frames):
        self.frames = frames if isinstance(frames, (list, tuple)) else [frames]
        self._stack = None

    def __enter__(self):
        self._stack = contextlib.ExitStack()
        self._stack.__enter__()
        results = [self._stack.enter_context(f) for f in self.frames]
        return tuple(results) if len(results) > 1 else results[0]

    def __exit__(self, *args):
        return self._stack.__exit__(*args)


def frame_scope(frames: List[frame.TIRFrame]) -> _FrameScope:
    """Enter multiple IRBuilder frames without deep nesting.

    This function provides a way to enter multiple frames in a single `with`
    statement, which is particularly useful when migrating from cases where
    allocations don't require nested scopes.

    Parameters
    ----------
    frames : List[frame.TIRFrame]
        The list of frames to enter. Each frame's `__enter__` return value
        will be collected and returned as a tuple.

    Returns
    -------
    _FrameScope
        A context manager that enters all frames and returns their values.

    Examples
    --------
    .. code-block:: python

        from tvm.script.ir_builder import IRBuilder
        from tvm.script.ir_builder import tir as T
        from tvm.script.ir_builder.tir.utils import frame_scope

        with IRBuilder() as ib:
            with frame_scope([
                T.allocate([1], "int32", scope="local"),
                T.allocate([1], "int32", scope="local"),
                T.allocate([size], dtype, scope="local"),
            ]) as (lo, hi, temp):
                # Use lo, hi, temp directly
                T.buffer_store(lo, T.int32(0), [0])
                ...
    """
    return _FrameScope(frames)


def seq_scope():
    """Create a scope that allows multiple consecutive statements.

    The IRBuilder requires a parent frame when having multiple consecutive
    top-level statements (e.g., multiple loops). This function creates a
    dummy attr frame that serves as a parent scope.

    Returns
    -------
    frame.AttrFrame
        A dummy attribute frame that wraps multiple statements.

    Examples
    --------
    Without seq_scope, multiple consecutive loops fail:

    .. code-block:: python

        with IRBuilder() as ib:
            with T.serial(0, 10) as i:
                T.evaluate(i)
            with T.serial(0, 5) as j:  # This would fail!
                T.evaluate(j)

    With seq_scope, multiple consecutive statements work:

    .. code-block:: python

        with IRBuilder() as ib:
            with seq_scope():
                with T.serial(0, 10) as i:
                    T.evaluate(i)
                with T.serial(0, 5) as j:
                    T.evaluate(j)
            result = ib.get()
    """
    return T.attr(tir.const(0, "int32"), "pragma_scope", tir.StringImm("seq"))


def _unravel_index(index, shape):
    """Convert a flat index to multi-dimensional indices.

    Parameters
    ----------
    index : PrimExpr
        The flat index.
    shape : Tuple
        The shape of the buffer.

    Returns
    -------
    List[PrimExpr]
        The multi-dimensional indices.
    """
    indices = []
    for i, dim in enumerate(reversed(shape)):
        if i == len(shape) - 1:
            # Outermost dimension: use remaining quotient directly (no modulo)
            indices.append(index)
        else:
            indices.append(index % dim)
            index = index // dim
    return list(reversed(indices))


class _BufferProxy:
    """Proxy for flat indexing on multi-dimensional buffers.

    This class wraps a TIR Buffer and provides flat indexing that gets
    automatically converted to multi-dimensional indices. It also supports
    assignment syntax via __setitem__.

    Parameters
    ----------
    buf : Buffer
        The TIR buffer to wrap.

    Examples
    --------
    .. code-block:: python

        buf = tvm.tir.decl_buffer([2, 3], "float32")
        ptr = buffer_proxy(buf)

        # Read with flat index (converted to [0, 1])
        val = ptr[1]

        # Write with flat index
        ptr[1] = 42.0

        # Multi-dimensional access still works
        val = ptr[0, 2]
    """

    def __init__(self, buf):
        self._buffer = buf
        self.dtype = buf.dtype
        self.shape = buf.shape
        self.name = buf.name
        self.data = buf.data

    def _normalize_index(self, index):
        """Convert flat index to multi-dimensional indices if needed."""
        try:
            index = [*index]
        except TypeError:
            index = [index]
        if len(index) == 1 and len(self._buffer.shape) != 1:
            index = _unravel_index(index[0], self._buffer.shape)
        return index

    def __getitem__(self, index):
        index = self._normalize_index(index)
        return tir.BufferLoad(self._buffer, index)

    def __setitem__(self, index, value):
        index = self._normalize_index(index)
        T.buffer_store(self._buffer, value, index)


def buffer_proxy(buf: Buffer) -> _BufferProxy:
    """Create a buffer proxy for flat indexing on multi-dimensional buffers.

    This provides flat indexing that gets converted to multi-dimensional indices.
    It also supports assignment syntax via __setitem__.

    Parameters
    ----------
    buf : Buffer
        The TIR buffer to wrap.

    Returns
    -------
    _BufferProxy
        A proxy object that supports flat indexing and assignment.

    Examples
    --------
    .. code-block:: python

        from tvm.script.ir_builder.tir.utils import buffer_proxy

        buf = tvm.tir.decl_buffer([2, 3], "float32")
        ptr = buffer_proxy(buf)

        # Flat indexing (index 1 -> indices [0, 1])
        val = ptr[1]

        # Assignment syntax
        ptr[1] = 42.0
    """
    return _BufferProxy(buf)
