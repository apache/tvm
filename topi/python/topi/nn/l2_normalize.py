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
"""TVM operator for l2 normalize"""
from __future__ import absolute_import
import tvm
from .. import cpp

@tvm.target.generic_func
def l2_normalize(data, eps, axis=None):
    """Perform L2 normalization on the input data

    For axis=None, y(i, j) = x(i, j) / sqrt(max(sum(x^2), eps))

    Parameters
    ----------
    data : tvm.Tensor
        4-D with NCHW or NHWC layout

    eps : float
        epsilon value

    axis : list of int
        axis over the normalization applied

    Returns
    -------
    output : tvm.Tensor
        4-D output with same shape
    """
    return cpp.nn.l2_normalize(data, eps, axis)
