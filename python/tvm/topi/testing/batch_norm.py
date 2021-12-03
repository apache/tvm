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
"""Batch Normalization implemented in Numpy."""
import numpy as np


def batch_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    axis: int,
    epsilon: float,
    center: bool,
    scale: bool,
):
    """Batch Normalization operator implemented in Numpy.

    Parameters
    ----------
    data : np.ndarray
        Input to be batch-normalized.

    gamma : np.ndarray
        Scale factor to be applied to the normalized tensor.

    beta : np.ndarray
        Offset to be applied to the normalized tensor.

    axis : int
        Specify along which shape axis the normalization should occur.

    epsilon : float
        Small float added to variance to avoid dividing by zero.

    center : bool
        If True, add offset of beta to normalized tensor, If False,
        beta is ignored.

    scale : bool
        If True, scale normalized tensor by gamma. If False, gamma
        is ignored.

    Returns
    -------
    output : np.ndarray
        Normalized data with same shape as input
    """
    mean = x.mean(axis=axis, keepdims=True)
    var = x.var(axis=axis, keepdims=True)
    std = np.sqrt(var + epsilon)
    out = (x - mean) / std

    shape = [1] * len(x.shape)
    shape[axis] = x.shape[axis]

    if scale:
        out = out * gamma.reshape(shape)
    if center:
        out = out + beta.reshape(shape)

    return out
