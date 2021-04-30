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
"""Convolution in python"""
import numpy as np
import scipy.signal
from tvm.topi.nn.utils import get_pad_tuple


def conv2d_hwcn_python(a_np, w_np, stride, padding):
    """Convolution operator in HWCN layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [in_height, in_width, in_channel, batch]

    w_np : numpy.ndarray
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str or a list/tuple of 2 or 4 ints
        Padding size, or ['VALID', 'SAME'], or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 2 ints

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [out_height, out_width, out_channel, batch]
    """
    in_height, in_width, in_channel, batch = a_np.shape
    kernel_h, kernel_w, _, num_filter = w_np.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel_h, kernel_w))
    pad_h = pad_top + pad_bottom
    pad_w = pad_left + pad_right
    # compute the output shape
    out_channel = num_filter
    out_height = (in_height - kernel_h + pad_h) // stride_h + 1
    out_width = (in_width - kernel_w + pad_w) // stride_w + 1
    # change the layout from HWCN to NCHW
    at = a_np.transpose((3, 2, 0, 1))
    wt = w_np.transpose((3, 2, 0, 1))
    bt = np.zeros((batch, out_channel, out_height, out_width))
    # computation
    for n in range(batch):
        for f in range(out_channel):
            for c in range(in_channel):
                if pad_h > 0 or pad_w > 0:
                    apad = np.zeros((in_height + pad_h, in_width + pad_w))
                    apad[pad_top : pad_top + in_height, pad_left : pad_left + in_width] = at[n, c]
                else:
                    apad = at[n, c]
                out = scipy.signal.convolve2d(apad, np.rot90(np.rot90(wt[f, c])), mode="valid")
                bt[n, f] += out[::stride_h, ::stride_w]
    return bt.transpose((2, 3, 1, 0))
