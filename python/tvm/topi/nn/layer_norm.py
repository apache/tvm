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


def layer_norm(data, gamma, beta, axis, epsilon=1e-5):
    """Layer normalization operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        N-D with shape (d_0, d_1, ..., d_n)

    gamma: tvm.te.Tensor
        R-D with shape (r_0, r_1, ..., r_k) where R == len(axis) and d_{axis_i} == r_i

    beta: tvm.te.Tensor
        Optional, R-D with shape (r_0, r_1, ..., r_k) where R == len(axis) and d_{axis_i} == r_i

    axis : list of int
        Axis over the normalization applied

    epsilon : float
        The epsilon value to avoid division by zero.

    Returns
    -------
    result : tvm.te.Tensor
        N-D with shape (d_0, d_1, ..., d_n)
    """
    return cpp.nn.layer_norm(data, gamma, beta, axis, epsilon)
