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
"""Sortings operators."""
from . import _ffi_api
from ..expr import Expr, Constant


def sort(x: Expr, axis: int = -1, descending: bool = False):
    """Performs sorting along the given axis and returns an array
    in sorted order.

    Parameters
    ----------
    x : relax.Expr
        The input tensor.

    axis : int
        Axis along which to sort the input tensor.
        By default the last axis of the input is used.

    descending : bool
        Whether to sort in descending order, the default is False

    Returns
    -------
    out : relax.Expr
        Sorted tensor.

    """
    return _ffi_api.sort(x, axis, descending)  # type: ignore


def argsort(data: Expr, axis: int = -1, descending: bool = False, dtype: str = "int32"):
    """Performs sorting along the given axis and returns an array of indices
    having same shape as an input array that index data in sorted order.

    Parameters
    ----------
    data : relax.Expr
        The input data tensor.

    axis : int
        Axis long which to sort the input tensor.

    descending : bool
        Whether to sort in descending order, the default is False

    dtype : str
        The data type of the output indices.

    Returns
    -------
    out : relax.Expr
        Tensor with same shape as data.
    """
    return _ffi_api.argsort(data, axis, descending, dtype)  # type: ignore


def topk(
    data: Expr,
    k: int = 1,
    axis: int = -1,
    ret_type: str = "both",
    largest: bool = True,
    dtype: str = "int32",
):
    """Get the top k elements in an input tensor along the given axis.

    ret_type specifies the return type, can be one of ("both", "values", "indices").

    Parameters
    ----------
    data : relax.Expr
        The input data tensor.

    k : int
        Number of top elements to select. Return all elements if k < 1.

    axis : int
        Axis long which to sort the input tensor.

    ret_type: str
        The return type [both, values, indices].
        "both": return both top k data and indices.
        "values": return top k data only.
        "indices": return top k indices only.

    largest : bool
        Whether to return largest or smallest elements.
        The k smallest elements are returned if largest is False.

    dtype : str
        The data type of the indices output.

    Returns
    -------
    out : relax.Expr or List[relax.Expr]
        The computed result.
    """
    if isinstance(k, Constant):
        k = k.data.numpy().item()
    return _ffi_api.topk(data, k, axis, ret_type, largest, dtype)  # type: ignore
