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
"""Creation operators."""
from typing import Optional, Tuple, Union

from tvm import DataType
from tvm.ir.expr import PrimExpr

from . import _ffi_api
from ..expr import Expr, ShapeExpr

PrimExprLike = Union[int, PrimExpr]


def full(
    shape: Union[Tuple[PrimExprLike], Expr],
    fill_value: Expr,
    dtype: Optional[Union[str, DataType]] = None,
) -> Expr:
    """Fill array with scalar value.

    Parameters
    ----------
    shape : Union[Tuple[PrimExprLike], Expr]
        The shape of the created tensor.

    fill_value : relax.Expr
        The value to fill. Must be a scalar tensor.

    dtype : Optional[Union[str, DataType]]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of fill_value.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.full(shape, fill_value, dtype)  # type: ignore


def full_like(x: Expr, fill_value: Expr, dtype: Optional[Union[str, DataType]] = None) -> Expr:
    """Construct a tensor such that
    - its shape is the same as the input data tensor's shape,
    - its value is filled with the input scalar fill value.

    Parameters
    ----------
    x : relax.Expr
        The input tensor, which provides the shape, and dtype
        when the `dtype` field is not specified.

    fill_value : relax.Expr
        The value to fill. Must be a scalar tensor.

    dtype : Optional[Union[str, DataType]]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of the input tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.full_like(x, fill_value, dtype)  # type: ignore


def ones(shape: Union[Tuple[PrimExprLike], Expr], dtype: Union[str, DataType]) -> Expr:
    """Construct a tensor of all ones, with the input shape and dtype.

    Parameters
    ----------
    shape : Union[Tuple[PrimExprLike], Expr]
        The shape of the created tensor.

    dtype : Union[str, DataType]
        The data type of the created tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    if isinstance(shape, (tuple, list)):
        shape = ShapeExpr(shape)
    return _ffi_api.ones(shape, dtype)  # type: ignore


def ones_like(x: Expr, dtype: Optional[Union[str, DataType]] = None) -> Expr:
    """Construct a tensor with all ones, with shape of the input tensor shape.

    Parameters
    ----------
    x : relax.Expr
        The input tensor, which provides the shape, and dtype
        when the `dtype` field is not specified.

    dtype : Optional[Union[str, DataType]]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of the input tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.ones_like(x, dtype)  # type: ignore


def zeros(shape: Union[Tuple[PrimExprLike], Expr], dtype: Union[str, DataType]) -> Expr:
    """Construct a tensor of all zeros, with the input shape and dtype.

    Parameters
    ----------
    shape : Union[Tuple[PrimExprLike], Expr]
        The shape of the created tensor.

    dtype : Union[str, DataType]
        The data type of the created tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    if isinstance(shape, (tuple, list)):
        shape = ShapeExpr(shape)
    return _ffi_api.zeros(shape, dtype)  # type: ignore


def zeros_like(x: Expr, dtype: Optional[Union[str, DataType]] = None) -> Expr:
    """Construct a tensor with all zeros, with shape of the input tensor shape.

    Parameters
    ----------
    x : relax.Expr
        The input tensor, which provides the shape, and dtype
        when the `dtype` field is not specified.

    dtype : Optional[Union[str, DataType]]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of the input tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.zeros_like(x, dtype)  # type: ignore


def tril(x: Expr, k: int = 0) -> Expr:
    """Return the lower triangular part of a matrix or a batch of matrices.

    Parameters
    ----------
    x : relax.Expr
        The tensor that tril will be applied to.
        It is required to have at least two dimensions.

    k : int
        The index indicating the diagonal above which to zero elements.
        If k = 0, the diagonal is the main diagonal.
        If k < 0, the diagonal is below the main diagonal.
        If k > 0, the diagonal is above the main diagonal.

    Returns
    -------
    ret : relax.Expr
        The result tensor.
    """
    return _ffi_api.tril(x, k)  # type: ignore


def triu(x: Expr, k: int = 0) -> Expr:
    """Return the upper triangular part of a matrix or a batch of matrices.

    Parameters
    ----------
    x : relax.Expr
        The tensor that triu will be applied to.
        It is required to have at least two dimensions.

    k : int
        The index indicating the diagonal below which to zero elements.
        If k = 0, the diagonal is the main diagonal.
        If k < 0, the diagonal is below the main diagonal.
        If k > 0, the diagonal is above the main diagonal.

    Returns
    -------
    ret : relax.Expr
        The result tensor.
    """
    return _ffi_api.triu(x, k)  # type: ignore
