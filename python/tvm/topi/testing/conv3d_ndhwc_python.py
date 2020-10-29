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
"""Convolution 3D in python"""
import numpy as np
import scipy.signal
from tvm.topi.nn.utils import get_pad_tuple3d


def conv3d_ndhwc_python(a_np, w_np, stride, padding):
    """Convolution 3D operator in NDHWC layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    w_np : numpy.ndarray
        5-D with shape [num_filter, in_channel, filter_depth, filter_height, filter_width]

    stride : int or a list/tuple of three ints
        Stride size, or [stride_depth, stride_height, stride_width]

    padding : int or str or a list/tuple of three ints
        Padding size, or ['VALID', 'SAME'], or [pad_depth, pad_height, pad_width]
    groups : int
        Number of groups

    Returns
    -------
    b_np : np.ndarray
        5-D with shape [batch, out_channel, out_depth, out_height, out_width]
    """
    batch, in_depth, in_height, in_width, in_channel = a_np.shape
    kernel_d, kernel_h, kernel_w, _, num_filter = w_np.shape
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right = get_pad_tuple3d(
        padding, (kernel_d, kernel_h, kernel_w)
    )
    pad_d = pad_front + pad_back
    pad_h = pad_top + pad_bottom
    pad_w = pad_left + pad_right
    # compute the output shape
    out_channel = num_filter
    out_depth = (in_depth - kernel_d + pad_d) // stride_d + 1
    out_height = (in_height - kernel_h + pad_h) // stride_h + 1
    out_width = (in_width - kernel_w + pad_w) // stride_w + 1
    # change the layout from NHWC to NCHW
    at = a_np.transpose((0, 4, 1, 2, 3))
    wt = w_np.transpose((4, 3, 0, 1, 2))
    bt = np.zeros((batch, out_channel, out_depth, out_height, out_width))
    # computation
    for n in range(batch):
        for f in range(out_channel):
            for c in range(in_channel):
                if pad_d > 0 or pad_h > 0 or pad_w > 0:
                    apad = np.zeros((in_depth + pad_d, in_height + pad_h, in_width + pad_w))
                    apad[
                        pad_front : pad_front + in_depth,
                        pad_top : pad_top + in_height,
                        pad_left : pad_left + in_width,
                    ] = at[n, c]
                else:
                    apad = at[n, c]
                out = scipy.signal.convolve(apad, np.flip(wt[f, c]), mode="valid")
                bt[n, f] += out[::stride_d, ::stride_h, ::stride_w]
    return bt.transpose((0, 2, 3, 4, 1))
