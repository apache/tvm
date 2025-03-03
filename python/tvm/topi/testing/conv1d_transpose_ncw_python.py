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
# pylint: disable=unused-variable
"""Transposed 1D convolution in python"""
import numpy as np
import scipy
import tvm.topi.testing
from tvm.topi.nn.utils import get_pad_tuple1d


def group_conv1d_transpose_ncw_python(a_np, w_np, stride, padding, output_padding, groups=1):
    "Grouped version of `conv1d_transpose_ncw_python`, see that for documentation"
    a_slices = np.array_split(a_np, groups, axis=1)
    w_slices = np.array_split(w_np, groups, axis=0)
    b_slices = [
        conv1d_transpose_ncw_python(a_slice, w_slice, stride, padding, output_padding)
        for a_slice, w_slice in zip(a_slices, w_slices)
    ]
    b_np = np.concatenate(b_slices, axis=1)
    return b_np


def conv1d_transpose_ncw_python(a_np, w_np, stride, padding, output_padding):
    """Transposed 1D convolution operator in NCW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        3-D with shape [batch, in_channel, in_width]

    w_np : numpy.ndarray
        3-D with shape [in_channel, num_filter, filter_width]

    stride : int or a list/tuple of one int
        Stride size, or [stride_width]

    padding : int, tuple, or str
        Single int for padding size, or
        tuple of 2 ints for left and right padding, or
        ['VALID', 'SAME']

    output_padding : tuple
        Used to recover the actual output shape in case more than one
        is possible

    Returns
    -------
    b_np : np.ndarray
        3-D with shape [batch, out_channel, out_width]

    """
    batch, in_c, in_w = a_np.shape
    _, out_c, filter_w = w_np.shape
    opad = output_padding[0]
    if isinstance(stride, int):
        stride_w = stride
    else:
        stride_w = stride[0]
    assert opad < stride_w
    fpad_left, fpad_right = get_pad_tuple1d(padding, filter_w)
    # dilate stage
    dilated_a_np = tvm.topi.testing.dilate_python(a_np, [1, 1, stride_w])
    # padding stage
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right + opad
    padded_a_np = np.zeros((batch, in_c, dilated_a_np.shape[2] + bpad_left + bpad_right))
    padded_a_np[:, :, bpad_left : dilated_a_np.shape[2] + bpad_left] = dilated_a_np
    # convolution stage
    out_w = (in_w - 1) * stride_w - fpad_left - fpad_right + filter_w + opad
    b_np = np.zeros((batch, out_c, out_w))
    for n in range(batch):
        for f in range(out_c):
            for c in range(in_c):
                out = scipy.signal.convolve(padded_a_np[n, c], w_np[c, f], mode="valid")
                b_np[n, f] += out
    return b_np
