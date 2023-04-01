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
"""Manipulation operators."""
from typing import Callable, List, Optional, Tuple, Union

from tvm import DataType
from tvm.ir.expr import PrimExpr
from tvm.tir import FloatImm, IndexMap, IntImm

from ...expr import Expr, PrimValue, ShapeExpr
from ...expr import Tuple as RxTuple
from . import _ffi_api

PrimExprLike = Union[int, PrimExpr]


def concat(tensors: Union[Expr, List[Expr]], axis: Optional[int] = 0) -> Expr:
    """Concatenate the input tensors along the given axis.

    Parameters
    ----------
    tensors : Union[relax.Expr, List[relax.Expr]]
        An Expr in Tuple type, containing the tensors to be concatenated,
        or a list of Tensors.

    axis : Optional[int]
        The axis along which the tensors are concatenated.
        If `axis` is `None`, the input tensor is required to be flattened before concatenation.

    Returns
    -------
    result: relax.Expr
        The concatenated tensor.
    """
    if isinstance(tensors, (list, tuple)):
        tensors = RxTuple(tensors)
    return _ffi_api.concat(tensors, axis)  # type: ignore


def split(
    x: Expr,
    indices_or_sections: Union[int, List[PrimExprLike]],
    axis: int = 0,
) -> Expr:
    """Split input tensor along axis by sections or indices.

    If indices_or_sections is an integer, the input will be divided equally
    along given axis (if possible). Last section will be smaller if the tensor
    size along the given dimension is not divisible by the integer.

    If indices_or_sections is a tuple of mixture of int or PrimExpr,
    the entries indicate the indices where along axis the array is split.

    Parameters
    ----------
    x : relax.Expr
        The tensor to be split.

    indices_or_sections : Union[int, List[PrimExprLike]]
        Indices or sections to split into. Accepts an int or a list.

    axis : int
        The axis over which to split.

    Returns
    -------
    ret : relax.Expr
        The computed result.
    """
    if isinstance(indices_or_sections, int):
        indices_or_sections = IntImm("int64", indices_or_sections)
    return _ffi_api.split(x, indices_or_sections, axis)  # type: ignore


def collapse_sum_like(data: Expr, collapse_target: Expr) -> Expr:
    """Return a summation of data to the shape of collapse_target.

    For details, please see relax.op.collapse_sum_to.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    collapse_target : relax.Expr
        The tensor whose shape is the shape to collapse to.

    Returns
    -------
    result : relax.Expr
        The result tensor after summation.
    """
    return _ffi_api.collapse_sum_like(data, collapse_target)  # type: ignore


def collapse_sum_to(data: Expr, shape: Union[Tuple[PrimExprLike], Expr]) -> Expr:
    """Return a summation of data to the given shape.

    collapse_sum_to is intended as the backward operator of tvm.relax.op.broadcast_to and
    other broadcast operators in the automatic differentiation process.

    We expect that data is the result of broadcasting some tensor of the given shape in some
    broadcast operation. Thus the given `shape` and `data.shape` must follow broadcast rules.

    During computation, all axes of `data.shape` and `shape` are checked from right to left.
    For an axis, if it follows these rules, `data` will be summed over this axis:
    - the axis exists in `data.shape` but not in `shape`, or
    - the axis exists in `data.shape` and equals to 1 in `shape`.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    shape : Union[Tuple[PrimExprLike], relax.Expr]
        The shape to collapse to.

    Returns
    -------
    result : relax.Expr
        The result tensor of the given shape after summation.
    """
    if isinstance(shape, (tuple, list)):
        shape = ShapeExpr(shape)
    return _ffi_api.collapse_sum_to(data, shape)  # type: ignore


def cumsum(data: Expr, axis: Optional[int] = None, dtype: Optional[Union[str, DataType]] = None):
    """Numpy style cumsum op. Return the cumulative inclusive sum of the elements along
    a given axis.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    axis : Optional[int]
        Axis along which the cumulative sum is computed. The default (None) is to compute
        the cumsum over the flattened array.

    dtype : Optional[Union[str, DataType]]
        Type of the returned array and of the accumulator in which the elements are summed.
        If dtype is not specified, it defaults to the dtype of data.

    Returns
    -------
    result : relax.Expr
        The result has the same size as data, and the same shape as data if axis is not None.
        If axis is None, the result is a 1-d array.

    Examples
    --------
    .. code-block:: python

        a = [[1, 2, 3], [4, 5, 6]]

        cumsum(a)  # if axis is not provided, cumsum is done over the flattened input.
        -> [ 1,  3,  6, 10, 15, 21]

        cumsum(a, dtype="float32")
        -> [  1.,   3.,   6.,  10.,  15.,  21.]

        cumsum(a, axis=0)  # sum over rows for each of the 3 columns
        -> [[1, 2, 3],
            [5, 7, 9]]

        cumsum(a, axis=1)
        -> [[ 1,  3,  6],
            [ 4,  9, 15]]

        a = [1, 0, 1, 0, 1, 1, 0]  # a is a boolean array
        cumsum(a, dtype=int32)  # dtype should be provided to get the expected results
        -> [1, 1, 2, 2, 3, 4, 4]
    """
    return _ffi_api.cumsum(data, axis, dtype)  # type: ignore
