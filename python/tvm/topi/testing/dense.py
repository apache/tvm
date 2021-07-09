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
"""Dense in python"""
import numpy as np


def dense(x, y, bias, use_bias=False, use_relu=False, out_dtype=None):
    """dense operator implemented in numpy.

    Parameters
    ----------
    x : numpy.ndarray
        2-D with shape [M, K]

    y : numpy.ndarray
        2-D with shape [N, K]

    bias: numpy.ndarray
        1-D with shape [M,]

    out_dtype: string, optional
        Specify the dtype of output

    Returns
    -------
    out : numpy.ndarray
        2-D with shape [M, N]
    """
    dtype = x.dtype if out_dtype is None else out_dtype
    if use_bias:
        out = np.dot(x.astype(dtype), y.T.astype(dtype)) + bias
    else:
        out = np.dot(x.astype(dtype), y.T.astype(dtype))

    if use_relu:
        out = np.maximum(out, 0)

    return out
