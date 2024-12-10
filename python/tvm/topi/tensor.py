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
# pylint: disable=invalid-name,consider-using-enumerate,unused-argument,len-as-condition
"""Elementwise operators"""

from typing import Optional

from tvm import te

from . import cpp


def elemwise_sum(xs):
    """Perform element-wise sum on inputs

    Parameters
    ----------
    xs : list of tvm.te.Tensor
        Input arguments.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.elemwise_sum(xs)


def full(shape, dtype, fill_value):
    """Fill tensor with fill_value

    Parameters
    ----------
    shape : tuple
        Input tensor shape.
    dtype : str
        Data type
    fill_value : float
        Value to be filled

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.full(shape, dtype, fill_value)


def full_like(x, fill_value):
    """Construct a tensor with same shape as input tensor,
       then fill tensor with fill_value.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.
    fill_value : float
        Value to be filled

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.full_like(x, fill_value)


def eye(n: int, m: Optional[int] = None, k: int = 0, dtype: str = "float32") -> te.Tensor:
    """Generate an identity matrix or a matrix with ones on the k-th diagonal.

    Parameters
    ----------
    n : int
        Number of rows
    m : int, optional
        Number of columns. If None, defaults to n.
    k : int, optional
        Index of the diagonal. 0 (default) refers to the main diagonal.
        A positive value refers to an upper diagonal, and a negative value
        to a lower diagonal.
    dtype : str, optional
        Data type of the returned array.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    m = m if m is not None else n
    return te.compute(
        (n, m),
        lambda i, j: te.if_then_else(i == j - k, te.const(1, dtype), te.const(0, dtype)),
        name="eye",
    )
