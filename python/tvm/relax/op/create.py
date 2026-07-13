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

from tvm import DataType, DataTypeCode
from tvm.ir import PrimType, is_prim_expr

from ..expr import Expr, ShapeExpr, prim_value
from . import _ffi_api

PrimExprLike = int | Expr


def _raw_dtype(dtype):
    return dtype.dtype if isinstance(dtype, PrimType) else dtype


def _normalize_shape(shape):
    if isinstance(shape, tuple | list):
        return ShapeExpr(shape)
    if not isinstance(shape, Expr) or is_prim_expr(shape):
        raise TypeError("shape must be a tuple/list or a Relax shape expression")
    return shape


def full(
    shape: tuple[PrimExprLike] | Expr,
    fill_value: Expr,
    dtype: str | DataType | None = None,
) -> Expr:
    """Fill array with scalar value.

    Parameters
    ----------
    shape : Union[Tuple[PrimExprLike], Expr]
        The shape of the created tensor.

    fill_value : relax.Expr
        The value to fill. Must be a scalar tensor.

    dtype : Optional[str | DataType]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of fill_value.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    shape = _normalize_shape(shape)
    return _ffi_api.full(shape, fill_value, _raw_dtype(dtype))  # type: ignore


def full_like(x: Expr, fill_value: Expr, dtype: str | DataType | None = None) -> Expr:
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

    dtype : Optional[str | DataType]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of the input tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.full_like(x, fill_value, _raw_dtype(dtype))  # type: ignore


def ones(shape: tuple[PrimExprLike] | Expr, dtype: str | DataType) -> Expr:
    """Construct a tensor of all ones, with the input shape and dtype.

    Parameters
    ----------
    shape : Union[Tuple[PrimExprLike], Expr]
        The shape of the created tensor.

    dtype : str | DataType
        The data type of the created tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    shape = _normalize_shape(shape)
    return _ffi_api.ones(shape, _raw_dtype(dtype))  # type: ignore


def ones_like(x: Expr, dtype: str | DataType | None = None) -> Expr:
    """Construct a tensor with all ones, with shape of the input tensor shape.

    Parameters
    ----------
    x : relax.Expr
        The input tensor, which provides the shape, and dtype
        when the `dtype` field is not specified.

    dtype : Optional[str | DataType]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of the input tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.ones_like(x, _raw_dtype(dtype))  # type: ignore


def zeros(shape: tuple[PrimExprLike] | Expr, dtype: str | DataType) -> Expr:
    """Construct a tensor of all zeros, with the input shape and dtype.

    Parameters
    ----------
    shape : Union[Tuple[PrimExprLike], Expr]
        The shape of the created tensor.

    dtype : str | DataType
        The data type of the created tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    shape = _normalize_shape(shape)
    return _ffi_api.zeros(shape, _raw_dtype(dtype))  # type: ignore


def zeros_like(x: Expr, dtype: str | DataType | None = None) -> Expr:
    """Construct a tensor with all zeros, with shape of the input tensor shape.

    Parameters
    ----------
    x : relax.Expr
        The input tensor, which provides the shape, and dtype
        when the `dtype` field is not specified.

    dtype : Optional[str | DataType]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of the input tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _ffi_api.zeros_like(x, _raw_dtype(dtype))  # type: ignore


def eye(
    n: PrimExprLike,
    m: PrimExprLike | None = None,
    k: PrimExprLike = 0,
    dtype: str | DataType = "float32",
) -> Expr:
    """Construct a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n : PrimExprLike
        Number of rows in the output.

    m : Optional[PrimExprLike]
        Number of columns in the output. If None, defaults to n.

    k : PrimExprLike
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal, and a negative value
        to a lower diagonal.

    dtype : str | DataType
        The data type of the created tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    m = n if m is None else m
    n = prim_value(n)
    m = prim_value(m)
    k = prim_value(k)
    return _ffi_api.eye(n, m, k, _raw_dtype(dtype))  # type: ignore


def eye_like(
    x: Expr,
    k: PrimExprLike = 0,
    dtype: str | DataType | None = None,
) -> Expr:
    """Return a 2-D tensor with ones on the diagonal and zeros elsewhere,
    with the same shape as the input tensor.

    Parameters
    ----------
    x : relax.Expr
        The input tensor, which provides the shape, and dtype
        when the `dtype` field is not specified.

    k : PrimExprLike
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal, and a negative value
        to a lower diagonal.

    dtype : Optional[str | DataType]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of the input tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    k = prim_value(k)
    return _ffi_api.eye_like(x, k, _raw_dtype(dtype))  # type: ignore


def arange(
    start: PrimExprLike,
    end: PrimExprLike | None = None,
    step: PrimExprLike = 1,
    dtype: str | DataType | None = None,
) -> Expr:
    """Construct a tensor with evenly spaced elements.

    Parameters
    ----------
    start : PrimExprLike
        The start of the interval.

    end : Optional[PrimExprLike]
        The end of the interval. If not given, it will be set to start,
        and start will be set to 0.

    step : PrimExprLike
        The step size.

    dtype : Optional[str | DataType]
        The data type of the created tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    if end is None:
        end = start
        start = 0

    def is_int(expr):
        if isinstance(expr, int):
            return True
        if is_prim_expr(expr):
            return expr.ty.matches_code(DataTypeCode.INT)
        return False

    if dtype is None:
        args = (start, end, step)
        integer_args = all(is_int(arg) for arg in args)
        dtype = "int64" if integer_args else "float32"

    start = prim_value(start)
    end = prim_value(end)
    step = prim_value(step)
    return _ffi_api.arange(start, end, step, _raw_dtype(dtype))  # type: ignore


def hamming_window(window_size, periodic, alpha, beta, dtype):
    """Hamming window function.

    Parameters
    ----------
    window_size : Expr
        The size of returned window.

    periodic : Expr
        If True, returns a window to be used as periodic function.
        If False, return a symmetric window.

    alpha : Expr
        The co-efficient alpha.

    beta : Expr
        The co-efficient beta.

    Returns
    -------
    ret : relax.Expr
        The result tensor.
    """
    if not is_prim_expr(window_size):
        window_size = prim_value(window_size)
    if not is_prim_expr(periodic):
        periodic = prim_value(periodic)
    if not is_prim_expr(alpha):
        alpha = prim_value(alpha)
    if not is_prim_expr(beta):
        beta = prim_value(beta)

    return _ffi_api.hamming_window(window_size, periodic, alpha, beta, dtype)


def tril(x: Expr, k: int | Expr = 0) -> Expr:
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
    if not is_prim_expr(k):
        k = prim_value(k)

    return _ffi_api.tril(x, k)  # type: ignore


def triu(x: Expr, k: int | Expr = 0) -> Expr:
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
    if not is_prim_expr(k):
        k = prim_value(k)

    return _ffi_api.triu(x, k)  # type: ignore
