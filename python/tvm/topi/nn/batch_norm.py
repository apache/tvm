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
"""Batch normalization."""
from tvm import te
from tvm import topi


def batch_norm(
    data: te.Tensor,
    gamma: te.Tensor,
    beta: te.Tensor,
    axis: int = 1,
    epsilon: float = 1e-5,
    center: bool = True,
    scale: bool = True,
):
    """Batch normalization layer (Ioffe and Szegedy, 2014).

    Normalizes the input at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation
    standard deviation close to 1.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input to be batch-normalized.

    gamma : tvm.te.Tensor
        Scale factor to be applied to the normalized tensor.

    beta : tvm.te.Tensor
        Offset to be applied to the normalized tensor.

    axis : Optional[int] = 1
        Specify along which shape axis the normalization should occur.

    epsilon : Optional[float] = 1e-5
        Small float added to variance to avoid dividing by zero.

    center : Optional[bool] = True
        If True, add offset of beta to normalized tensor, If False,
        beta is ignored.

    scale : Optional[bool] = True
        If True, scale normalized tensor by gamma. If False, gamma
        is ignored.

    Returns
    -------
    output : tvm.te.Tensor
        Normalized data with same shape as input
    """
    mean = topi.reduction.sum(data, axis=axis, keepdims=True) / data.shape[axis]
    var_summands = topi.broadcast.power(topi.broadcast.subtract(data, mean), 2.0)
    var = topi.reduction.sum(var_summands, axis=axis, keepdims=True) / data.shape[axis]
    std = topi.math.sqrt(var + epsilon)
    out = (data - mean) / std

    shape = [1] * len(data.shape)
    shape[axis] = data.shape[axis]

    if scale:
        out = out * topi.reshape(gamma, shape)
    if center:
        out = out + topi.reshape(beta, shape)

    return out
