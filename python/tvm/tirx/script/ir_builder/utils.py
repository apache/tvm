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

from tvm.script.ir_builder.stmt import frame_scope as frame_scope
from tvm.tirx import Var


def tensor_indices(buffer: Var, index):
    """Translate logical flat or multidimensional indices for a concrete buffer.

    A single index is unraveled in row-major logical order, retaining the
    outermost quotient. Explicit multidimensional coordinates pass through.
    The result indexes the original buffer, preserving its strides, layout,
    element offset and aliases; this function never creates a buffer view.

    Parameters
    ----------
    buffer : Var
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
