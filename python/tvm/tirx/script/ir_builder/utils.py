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

from tvm.ir import StringImm
from tvm.tirx import Buffer

from . import _ffi_api, frame


class _FrameScope:
    """Context manager to enter multiple IRBuilder frames without deep nesting.

    This class allows entering multiple frames in a single `with` statement,
    avoiding the pyramid of nested context managers.

    Parameters
    ----------
    frames : List[IRBuilderFrame]
        The list of frames to enter.
    """

    def __init__(self, frames):
        self.frames = frames if isinstance(frames, list | tuple) else [frames]
        self._stack = None

    def __enter__(self):
        self._stack = contextlib.ExitStack()
        self._stack.__enter__()
        results = [self._stack.enter_context(f) for f in self.frames]
        return tuple(results) if len(results) > 1 else results[0]

    def __exit__(self, *args):
        return self._stack.__exit__(*args)


def frame_scope(frames: list[frame.TIRFrame]) -> _FrameScope:
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
    return _ffi_api.Attr(0, "pragma_scope", StringImm("seq"))


def buffer_indices(buffer: Buffer, index):
    """Translate logical flat or multidimensional indices for a concrete buffer.

    A single index is unraveled in row-major logical order, retaining the
    outermost quotient. Explicit multidimensional coordinates pass through.
    The result indexes the original buffer, preserving its strides, layout,
    element offset and aliases; this function never creates a buffer view.

    Parameters
    ----------
    buffer : Buffer
        The concrete buffer whose logical shape determines the coordinates.
    index : Expr or sequence of Expr
        A flat logical index or explicit multidimensional coordinates.

    Returns
    -------
    indices : list of Expr
        Coordinates for a buffer load or an explicitly emitted buffer store.
    """
    try:
        indices = list(index)
    except TypeError:
        indices = [index]
    shape = buffer.shape
    if len(indices) != 1 or len(shape) == 1:
        return indices
    index = indices[0]
    indices = []
    for axis, extent in enumerate(reversed(shape)):
        if axis == len(shape) - 1:
            indices.append(index)
        else:
            indices.append(index % extent)
            index = index // extent
    return list(reversed(indices))
