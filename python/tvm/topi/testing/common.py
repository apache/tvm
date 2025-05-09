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
"""Common utility for topi test"""

import numpy as np
import scipy.signal


def _convolve2d(data, weights):
    """2d convolution operator in HW layout.

    This is intended to be used as a replacement for
    scipy.signals.convolve2d, with wider support for different dtypes.
    scipy.signal.convolve2d does not support all TVM-supported
    dtypes (e.g. float16).  Where possible, this function uses
    scipy.signal.convolve2d to take advantage of compiled scipy
    routines, falling back to an explicit loop only where needed.

    Parameters
    ----------
    data : numpy.ndarray
        2-D with shape [in_height, in_width]

    weights : numpy.ndarray
        2-D with shape [filter_height, filter_width].

    Returns
    -------
    b_np : np.ndarray
        2-D with shape [out_height, out_width]

        Return value and layout conventions are matched to
        ``scipy.signal.convolve2d(data, weights, mode="valid")``
    """

    try:
        return scipy.signal.convolve2d(data, weights, mode="valid")
    except ValueError:
        pass

    weights = np.rot90(weights, k=2)

    assert len(data.shape) == len(weights.shape) == 2

    dtype = data.dtype
    kernel_h, kernel_w = weights.shape

    output_shape = [a_dim - w_dim + 1 for a_dim, w_dim in zip(data.shape, weights.shape)]
    output = np.zeros(output_shape, dtype=dtype)

    for y in range(output_shape[0]):
        for x in range(output_shape[1]):
            output[y][x] = np.sum(data[y : y + kernel_h, x : x + kernel_w] * weights)

    return output
