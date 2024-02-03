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
# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Group normalization in python"""
import numpy as np


def group_norm_python(data, gamma, beta, num_groups, channel_axis, axes, epsilon=1e-5):
    """Group normalization operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        N-D with shape (d_0, d_1, ..., d_{N-1})

    gamma: tvm.te.Tensor
        1-D with shape (r_0) where r_0 == d_{channel_axis}

    beta: tvm.te.Tensor
        Optional, 1-D with shape (r_0) where r_0 == d_{channel_axis}

    num_groups : int
        The number of groups

    channel_axis : int
        The channel axis

    axes : list of int
        Axis over the normalization applied, excluding the channel axis

    epsilon : float
        The epsilon value to avoid division by zero.

    Returns
    -------
    result : tvm.te.Tensor
        N-D with shape (d_0, d_1, ..., d_{N-1})
    """
    old_shape = data.shape
    old_dtype = data.dtype
    new_shape = list(old_shape)
    new_shape[channel_axis] = data.shape[channel_axis] // num_groups
    new_shape.insert(channel_axis, num_groups)
    data = np.reshape(data, new_shape).astype("float32")
    new_axes = [channel_axis + 1]
    for axis in axes:
        if axis < channel_axis:
            new_axes.append(axis)
        else:
            new_axes.append(axis + 1)
    mean = np.mean(data, axis=tuple(new_axes), keepdims=True)
    var = np.var(data, axis=tuple(new_axes), keepdims=True)
    data = (data - mean) / np.sqrt(var + epsilon)
    data = np.reshape(data, old_shape).astype(old_dtype)

    gamma_broadcast_shape = [1 for _ in range(len(old_shape))]
    gamma_broadcast_shape[channel_axis] = gamma.shape[0]
    gamma = np.reshape(gamma, gamma_broadcast_shape)

    beta_broadcast_shape = [1 for _ in range(len(old_shape))]
    beta_broadcast_shape[channel_axis] = beta.shape[0]
    if beta is not None:
        beta = np.reshape(beta, beta_broadcast_shape)

    data *= gamma
    if beta is not None:
        data += beta

    return data
