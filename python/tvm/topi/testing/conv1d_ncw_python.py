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
# pylint: disable=unused-variable, invalid-name
"""1D convolution in python"""
import numpy as np
from tvm.topi.nn.utils import get_pad_tuple1d


def dilate_np(x, dilation):
    """1D dilation using numpy

    Parameters
    ----------
    x : numpy.ndarray
        Array to dilate with shape [batch, in_channel, in_width]

    dilation : int
        dilation rate of output

    Returns
    -------
    out : numpy.ndarray
        Dilated output with shape [batch, in_channel, (in_width - 1) * dilation + 1]
    """
    irange = range(len(x) - 1)
    for d in range(dilation - 1):
        indices = [(d + 1) * (i + 1) for i in irange]
        x = np.insert(x, indices, 0)
    return x


def conv1d_ncw_python(a_np, w_np, stride, padding, dilation):
    """1D convolution operator in NCW layout

    Parameters
    ----------
    a_np : numpy.ndarray
        3-D with shape [batch, in_channel, in_width]

    w_np : numpy.ndarray
        3-D with shape [num_filter, in_channel, filter_width]

    stride : int
        Stride size

    padding : int, tuple, or str
        Single int for padding size or tuple of (left, right) padding
        or a string in ['VALID', 'SAME']

    dilation : int
        Dilation rate of the kernel

    Returns
    -------
    b_np : numpy.ndarray
        3-D with shape [batch, out_channel, out_width]
    """
    batch, in_c, in_w = a_np.shape
    out_c, _, filter_w = w_np.shape
    if isinstance(stride, (tuple, list)):
        stride = stride[0]
    if isinstance(dilation, (tuple, list)):
        dilation = dilation[0]

    dilated_filter_w = (filter_w - 1) * dilation + 1
    pad_left, pad_right = get_pad_tuple1d(padding, (dilated_filter_w,))
    out_w = ((in_w - dilated_filter_w + pad_left + pad_right) // stride) + 1

    padded_a_np = np.zeros((batch, in_c, in_w + pad_left + pad_right))
    padded_a_np[:, :, pad_left : (in_w + pad_left)] = a_np

    b_np = np.zeros((batch, out_c, out_w))
    for n in range(batch):
        for f in range(out_c):
            for c in range(in_c):
                out = np.convolve(
                    padded_a_np[n, c], np.flip(dilate_np(w_np[f, c], dilation)), mode="valid"
                )
                b_np[n, f] += out[::stride]
    return b_np
