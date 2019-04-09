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
"""Binary Neural Network (BNN) Operators"""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import tvm
from ..util import get_const_tuple


def batch_matmul(x, y):
    """Computes batch matrix multiplication of `x` and `y` when `x` and `y` are
    data in batch.

    Parameters
    ----------
    x : tvm.Tensor
        3-D with shape [batch, M, K]

    y : tvm.TEnsor
        3-D with shape [batch, N, K]

    Returns
    -------
    output : tvm.Tensor
        3-D with shape [batch, M, N]
    """
    assert len(x.shape) == 3 and len(y.shape) == 3, "only support 3-dim batch_matmul"
    x_shape = get_const_tuple(x.shape)
    y_shape = get_const_tuple(y.shape)
    assert x_shape[0] == y_shape[0], "batch dimension doesn't match"
    assert x_shape[2] == y_shape[2], "shapes of x and y is inconsistant"
    batch, M, K = x.shape
    N = y.shape[1]
    k = tvm.reduce_axis((0, K), name='k')
    return tvm.compute((batch, M, N),
                       lambda b, i, j: tvm.sum(x[b, i, k] * y[b, j, k], axis=k),
                       tag='batch_matmul')
