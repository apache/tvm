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

from tvm.ir import Call as _Call
from tvm.ir import GenericConst
from tvm.ir.attrs import make_node as _make_attrs

from ..expr import Expr


def sort(
    x: Expr,
    axis: int = -1,
    descending: bool = False,
    *,
    ty=None,
    span=None,
):
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
    return _Call(
        "relax.sort",
        [x],
        attrs=_make_attrs("relax.attrs.SortAttrs", axis=axis, descending=descending),
        ty=ty,
        span=span,
    )  # type: ignore


def argsort(
    data: Expr,
    axis: int = -1,
    descending: bool = False,
    dtype: str = "int32",
    *,
    ty=None,
    span=None,
):
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
    return _Call(
        "relax.argsort",
        [data],
        attrs=_make_attrs(
            "relax.attrs.ArgsortAttrs", axis=axis, descending=descending, dtype=dtype
        ),
        ty=ty,
        span=span,
    )  # type: ignore


def topk(
    data: Expr,
    k: int = 1,
    axis: int = -1,
    ret_type: str = "both",
    largest: bool = True,
    dtype: str = "int32",
    *,
    ty=None,
    span=None,
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
    if isinstance(k, GenericConst):
        k = k.value.numpy().item()
    return _Call(
        "relax.topk",
        [data],
        attrs=_make_attrs(
            "relax.attrs.TopKAttrs", k=k, axis=axis, ret_type=ret_type, largest=largest, dtype=dtype
        ),
        ty=ty,
        span=span,
    )  # type: ignore
