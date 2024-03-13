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
    Its semantic is mostly similar to `numpy.take`
    (https://numpy.org/doc/stable/reference/generated/numpy.take.html),
    which can cover `torch.take` (https://pytorch.org/docs/stable/generated/torch.take.html) and
    `onnx.gather` (https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gather-13).

    Parameters
    ----------
    x : relax.Expr
        The source tensor.

    indices : relax.Expr
        The indices of the values to extract.

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
    assume_inbound: bool = False,
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

    assume_inbound : bool
        Whether to assume the indices are in bound. If it is set to false,
        out of bound indices will be clipped to the bound.
    Returns
    -------
    ret : relax.Expr
        The sliced result.

    Note
    ----
    strided_slice require the input `begin`, `end` and `strides` to have the
    same length as `axes`.
    """
    return _ffi_api.strided_slice(x, axes, begin, end, strides, assume_inbound)  # type: ignore


def dynamic_strided_slice(
    x: Expr,
    begin: Expr,
    end: Expr,
    strides: Expr,
) -> Expr:
    """Dynamic strided slice of a tensor. `begin`, `end`, `strides` can be computed at runtime.

    Parameters
    ----------
    x : Expr
        The source tensor to be sliced.

    begin : Expr
        The indices to begin with in the slicing, inclusive.

    end : Expr
        The indices indicating end of the slice, exclusive.

    strides : Expr
        Specifies the stride values, it can be negative in that case,
        the input tensor will be reversed in that particular axis.
        If not specified, it by default is an list of ones of the same length as `axes`.

    Returns
    -------
    ret : relax.Expr
        The sliced result.

    Note
    ----
    dyn_strided_slice require the input `begin`, `end` and `strides` to have the
    same length as rank of `data` tensor.
    """
    return _ffi_api.dynamic_strided_slice(x, begin, end, strides)  # type: ignore
