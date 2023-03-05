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
# pylint: disable=invalid-name
"""Search operators."""
from typing import Optional

from . import _ffi_api
from ..expr import Expr


def where(condition: Expr, x1: Expr, x2: Expr) -> Expr:
    """Selecting elements from either the input tensors depending on the value of the
    condition.

    For a given position, return the corresponding value in `x1` if `condition` is True,
    and return the corresponding value in `x2` otherwise.

    Parameters
    ----------
    condition : relax.Expr
        When True, yield `x1`; otherwise, yield `x2`.
        Must be broadcasting compatible with `x1` and `x2`.
        Must have boolean dtype.

    x1 : relax.Expr
        The first input tensor.
        Must be broadcasting compatible with `condition` and `x2`.

    x2 : relax.Expr
        The second input tensor.
        Must be broadcasting compatible with `condition` and `x1`.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.where(condition, x1, x2)  # type: ignore


def argmax(x: Expr, axis: Optional[int] = None, keepdims: bool = False) -> Expr:
    """Computes the argmax of tensor elements over given axis.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[int]
        Axis along which an argmax operation is performed.
        The default, axis=None, will compute the argmax of all elements in the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axis being reduced is left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.argmax(x, axis, keepdims)  # type: ignore


def argmin(x: Expr, axis: Optional[int] = None, keepdims: bool = False) -> Expr:
    """Computes the argmin of tensor elements over given axis.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[int]
        Axis along which an argmin operation is performed.
        The default, axis=None, will compute the argmin of all elements in the
        input tensor. Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axis being reduced is left in the result as
        dimensions with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.argmin(x, axis, keepdims)  # type: ignore
