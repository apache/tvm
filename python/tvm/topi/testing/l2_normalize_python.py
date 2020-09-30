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
"""L2 normalize in python"""
import numpy as np


def l2_normalize_python(a_np, eps, axis=None):
    """L2 normalize operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    eps : float
        epsilon constant value
    axis : list of int
        axis over the normalization applied

    Returns
    -------
    l2_normalize_out : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    dot_value = np.power(a_np, 2.0)
    sqr_sum = np.sum(dot_value, axis, keepdims=True)
    sqrt_sum = np.sqrt(np.maximum(np.broadcast_to(sqr_sum, a_np.shape), eps))
    l2_normalize_out = np.divide(a_np, sqrt_sum)
    return l2_normalize_out
