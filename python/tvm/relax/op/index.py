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
"""Indexing operators."""
from typing import List, Optional, Union

from tvm.ir.expr import PrimExpr

from . import _ffi_api
from ..expr import Expr

PrimExprLike = Union[int, PrimExpr]


def take(x: Expr, indices: Expr, axis: Optional[int] = None) -> Expr:
    """Take elements from a tensor along an axis.

    Parameters
    ----------
    x : relax.Expr
        The source tensor.

    indices : relax.Expr
        The indices of the values to extract.
        It is required to be a one-dimensional tensor which has integer dtype.

    axis : Optional[int]
        The axis over which to select values.
        If it is none, the input tensor is required to be one-dimensional.

    Returns
    -------
    ret : relax.Expr
        The taken result.
    """
    return _ffi_api.take(x, indices, axis)  # type: ignore


def strided_slice(
    x: Expr,
    axes: List[int],
    begin: List[PrimExprLike],
    end: List[PrimExprLike],
    strides: Optional[List[PrimExprLike]] = None,
) -> Expr:
    """Strided slice of a tensor.

    Parameters
    ----------
    x : relax.Expr
        The source tensor to be sliced.

    axes : List[int]
        Axes along which slicing is applied.

    begin : List[PrimExprLike]
        The indices to begin with in the slicing, inclusive.

    end : List[PrimExprLike]
        The indices indicating end of the slice, exclusive.

    strides : Optional[List[PrimExprLike]]
        Specifies the stride values, it can be negative in that case,
        the input tensor will be reversed in that particular axis.
        If not specified, it by default is an list of ones of the same length as `axes`.

    Returns
    -------
    ret : relax.Expr
        The sliced result.

    Note
    ----
    strided_slice require the input `begin`, `end` and `strides` to have the
    same length as `axes`.
    """
    return _ffi_api.strided_slice(x, axes, begin, end, strides)  # type: ignore
