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
"""Layer normalization operator."""
from .. import cpp


def group_norm(data, gamma, beta, num_groups, channel_axis, axes, epsilon=1e-5):
    """Group normalization operator.
    It accepts fp16 and fp32 as input data type. It will cast the input to fp32
    to perform the computation. The output will have the same data type as input.

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
    return cpp.nn.group_norm(data, gamma, beta, num_groups, channel_axis, axes, epsilon)
