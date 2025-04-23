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
"""Instance normalization operator."""
from .. import cpp
from tvm import te
from tvm import topi
from functools import reduce
from typing import Union,List

# def instance_norm(data, gamma, beta, axis, epsilon=1e-5):
#     """Instance normalization operator.

#     Parameters
#     ----------
#     data : tvm.te.Tensor
#         N-D with shape (d_0, d_1, ..., d_{N-1})

#     gamma: tvm.te.Tensor
#         K-D with shape (r_0, r_1, ..., r_{K-1}) where K == len(axis) and d_{axis_k} == r_k

#     beta: tvm.te.Tensor
#         Optional, K-D with shape (r_0, r_1, ..., r_{K-1}) where K == len(axis) and d_{axis_k} == r_k

#     axis : list of int
#         Axis over the normalization applied (the axis along which the mean and variance are
#         computed)

#     epsilon : float
#         The epsilon value to avoid division by zero.

#     Returns
#     -------
#     result : tvm.te.Tensor
#         N-D with shape (d_0, d_1, ..., d_{N-1})
#     """
#     return cpp.nn.instance_norm(data, gamma, beta, axis, epsilon)

def instance_norm(
    data: te.Tensor,
    gamma: te.Tensor,
    beta: te.Tensor,
    axis: Union[int, List[int]] = [0, 1],
    epsilon: float = 1e-5,
) -> te.Tensor:
    """Instance normalization over spatial dimensions.

    Normalizes each instance in a batch independently per channel,
    typically used in style transfer and vision models.

    Parameters
    ----------
    data : te.Tensor
        Input tensor with shape [N, C, H, W].

    gamma : te.Tensor
        Scale tensor of shape [C].

    beta : te.Tensor
        Offset tensor of shape [C].

    axis : int or list of int, default=[0, 1]
        Axes to preserve (typically N and C). Reduction happens over the rest.

    epsilon : float
        Small value added to variance to avoid divide-by-zero.

    Returns
    -------
    out : te.Tensor
        Instance-normalized tensor with same shape as input.
    """
    if isinstance(axis, int):
        axis = [axis]

    shape = [1] * len(data.shape)
    for ax in axis:
        print(type(int(ax)))
        shape[int(ax)] = data.shape[int(ax)]

    reduce_axes = [i for i in range(len(data.shape)) if i not in axis]
    shape_prod = reduce(lambda x, y: x * y, [data.shape[ax] for ax in reduce_axes], 1)

    mean = topi.sum(data, axis=reduce_axes) / shape_prod
    mean_rs = topi.reshape(mean, shape)

    var = topi.sum(topi.power(data - mean_rs, 2), axis=reduce_axes) / shape_prod
    var_rs = topi.reshape(var, shape)

    gamma_rs = topi.reshape(gamma, shape)
    beta_rs = topi.reshape(beta, shape)

    normalized = (data - mean_rs) / topi.sqrt(var_rs + epsilon)
    out = normalized * gamma_rs + beta_rs

    return out