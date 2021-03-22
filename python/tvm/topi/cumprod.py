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
"""Cumprod operator"""
from typing import Optional

import tvm

from ..tir import generic
from .cumsum import cumbinop


def cumprod(
    data: tvm.te.Tensor,
    axis: Optional[int] = None,
    dtype: Optional[int] = None,
    exclusive: Optional[bool] = None,
):
    """Numpy style cumprod op. Return the cumulative product of the elements along a given axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input data to the operator.

    axis : int, optional
        Axis along which the cumulative product is computed. The default (None) is to compute
        the cumprod over the flattened array.

    dtype : string, optional
        Type of the returned array and of the accumulator in which the elements are multiplied.
        If dtype is not specified, it defaults to the dtype of data.

    exclusive : bool, optional
        If True, will return exclusive product in which the first element is not
        included. In other terms, if True, the j-th output element would be
        the product of the first (j-1) elements. Otherwise, it would be the product of
        the first j elements.

    Returns
    -------
    result : tvm.te.Tensor
        The result has the same size as data, and the same shape as data if axis is not None.
        If axis is None, the result is a 1-d array.
    """
    return cumbinop(
        data=data,
        binop=generic.multiply,
        identity_value=1,
        op_name="cumprod_generic",
        axis=axis,
        dtype=dtype,
        exclusive=exclusive,
    )
