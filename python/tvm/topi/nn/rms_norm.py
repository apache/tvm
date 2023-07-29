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
"""Root mean square normalization operator."""
from .. import cpp


def rms_norm(data, weight, axis, epsilon=1e-5):
    """Root mean square normalization operator. The output will have the same data type as input.

    Parameters
    ----------
    data : tvm.te.Tensor
        N-D with shape (d_0, d_1, ..., d_{N-1})

    weight: tvm.te.Tensor
        K-D with shape (r_0, r_1, ..., r_{K-1}) where K == len(axis) and d_{axis_k} == r_k

    axis : list of int
        Axis over the normalization applied

    epsilon : float
        The epsilon value to avoid division by zero.

    Returns
    -------
    result : tvm.te.Tensor
        N-D with shape (d_0, d_1, ..., d_{N-1})
    """
    return cpp.nn.rms_norm(data, weight, axis, epsilon)
