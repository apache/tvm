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
import typing
from functools import reduce

from tvm import te
from tvm import topi


def batch_norm(
    data: te.Tensor,
    gamma: te.Tensor,
    beta: te.Tensor,
    moving_mean: te.Tensor,
    moving_var: te.Tensor,
    axis: typing.Optional[int] = None,
    epsilon: typing.Optional[float] = None,
    center: typing.Optional[bool] = None,
    scale: typing.Optional[bool] = None,
    training: typing.Optional[bool] = None,
    momentum: typing.Optional[float] = None,
) -> typing.List[te.Tensor]:
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

    moving_mean : tvm.te.Tensor
        Running mean of input.

    moving_var : tvm.te.Tensor
        Running variance of input.

    axis : int, optional, default=1
        Specify along which shape axis the normalization should occur.

    epsilon : float, optional, default=1e-5
        Small float added to variance to avoid dividing by zero.

    center : bool, optional, default=True
        If True, add offset of beta to normalized tensor, If False,
        beta is ignored.

    scale : bool, optional, defualt=True
        If True, scale normalized tensor by gamma. If False, gamma
        is ignored.

    training : bool, optional, defualt=False
        Indicating whether it is in training mode. If True, update
        moving_mean and moving_var.

    momentum : float, optional, default=0.1
        The value used for the moving_mean and moving_var update.

    Returns
    -------
    output : list of tvm.te.Tensor
        Normalized data with same shape as input

    moving_mean : tvm.te.Tensor
        Running mean of input.

    moving_var : tvm.te.Tensor
        Running variance of input.
    """
    if axis is None:
        axis = 1

    if epsilon is None:
        epsilon = 1e-5

    if center is None:
        center = True

    if scale is None:
        scale = True

    if training is None:
        training = False

    if momentum is None:
        momentum = 0.1

    shape = [1] * len(data.shape)
    shape[axis] = data.shape[axis]

    if training:
        reduce_axes = list(range(len(data.shape)))
        reduce_axes.remove(axis)
        shape_prod = reduce(lambda x, y: x * y, [data.shape[ax] for ax in reduce_axes], 1)
        data_mean = topi.sum(data, axis=reduce_axes) / shape_prod
        data_mean_rs = topi.reshape(data_mean, shape)
        data_var = (
            topi.sum((data - data_mean_rs) * (data - data_mean_rs), axis=reduce_axes) / shape_prod
        )
        data_var_rs = topi.reshape(data_var, shape)
        out = (data - data_mean_rs) / topi.math.sqrt(data_var_rs + epsilon)
    else:
        moving_mean_rs = topi.reshape(moving_mean, shape)
        moving_var_rs = topi.reshape(moving_var, shape)
        out = (data - moving_mean_rs) / topi.math.sqrt(moving_var_rs + epsilon)

    if scale:
        out = out * topi.reshape(gamma, shape)
    if center:
        out = out + topi.reshape(beta, shape)

    if training:
        assert 0 <= momentum <= 1, "the valid momentum range is [0, 1]."
        data_var = (
            topi.sum((data - data_mean_rs) * (data - data_mean_rs), axis=reduce_axes) / shape_prod
        )
        return [
            out,
            (1 - momentum) * moving_mean + momentum * data_mean,
            (1 - momentum) * moving_var + momentum * data_var,
        ]

    # Moving mean and var aren't updated during test. To avoid
    # placeholder reuse, we multiply by 1 and return them.
    return [out, moving_mean * 1, moving_var * 1]
