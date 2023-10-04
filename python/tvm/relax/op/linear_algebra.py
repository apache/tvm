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
"""Relax linear algebra operators"""
from typing import Optional, Union

from tvm import DataType

from . import _ffi_api
from ..expr import Expr, Tuple as RxTuple
from .manipulate import permute_dims


def matmul(x1: Expr, x2: Expr, out_dtype: Optional[Union[str, DataType]] = None) -> Expr:
    """General matrix multiplication of two tensors, with broadcasting on batched dimensions.

    The semantics and output shape deduction rule is specified as
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html.

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.

    x2 : relax.Expr
        The second input tensor.

    out_dtype: Optional[Union[str, DataType]]
        The data type of the matmul result.
        When it is not specified, the output dtype will be the same as input dtype.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _ffi_api.matmul(x1, x2, out_dtype)  # type: ignore


def linear(
    data: Expr,
    weight: Expr,
    bias: Optional[Expr] = None,
    out_dtype: Optional[Union[str, DataType]] = None,
) -> Expr:
    """Applies a linear transformation to the incoming data: y = xA^T + b

    Parameters
    ----------
    data : relax.Expr
        The input data.

    weight : relax.Expr
        The weight tensor.

    bias : Optional[Expr]
        The bias tensor.

    out_dtype: Optional[Union[str, DataType]]
        The data type of the matmul result.
        When it is not specified, the output dtype will be the same as input dtype.

    Notes
    -----
    Relax does not regard the Linear Op as a primitive Op,
    while combine the transpose, matmul and add op to implement it.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """

    # Since weight can be 1D or 2D, we use `axes=None` to support both cases.
    x = matmul(data, permute_dims(weight, axes=None), out_dtype=out_dtype)
    return x + bias if bias is not None else x


def einsum(operands, subscripts):
    """Evaluates the Einstein summation convention on data

    Parameters
    ----------
    operands : Union(List[relax.Expr], Tuple[relax.Expr])
        A list of expression.

    subscripts : str
        The einsum expression string.

    Returns
    -------
    result : relax.Expr
        The output from the einsum op.
    """
    if isinstance(operands, (list, tuple)):
        operands = RxTuple(operands)

    return _ffi_api.einsum(operands, subscripts)  # type: ignore
