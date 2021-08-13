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
# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals, too-many-branches
"""Convolution in python"""
import numpy as np
import scipy

from tvm.topi.nn.utils import get_pad_tuple


def _conv2d_nchw_python(a_np, w_np, stride, padding):
    """Convolution operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    w_np : numpy.ndarray
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str or a list/tuple of 2 or 4 ints
        Padding size, or ['VALID', 'SAME'], or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 2 ints

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, in_channel, in_height, in_width = a_np.shape
    num_filter, _, kernel_h, kernel_w = w_np.shape
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
    b_np = np.zeros((batch, out_channel, out_height, out_width), dtype=a_np.dtype)
    # computation
    for n in range(batch):
        for f in range(out_channel):
            for c in range(in_channel):
                if pad_h > 0 or pad_w > 0:
                    apad = np.zeros((in_height + pad_h, in_width + pad_w), dtype=a_np.dtype)
                    apad[pad_top : pad_top + in_height, pad_left : pad_left + in_width] = a_np[n, c]
                else:
                    apad = a_np[n, c]

                out = _conv2d_hw(apad, w_np[f, c])
                b_np[n, f] += out[::stride_h, ::stride_w]
    return b_np


def _conv2d_hw(apad, w_np_fc):
    """2d convolution operator in HW layout.

    This is intended to be used as a subroutine from
    _conv2d_nchw_python.  Using scipy.signal.convolve2d directly does
    not work for all dtypes (e.g. float16).  Where possible, this
    function uses scipy.signal.convolve2d to take advantage of
    compiled scipy routines, falling back to an explicit loop only
    where needed

    Parameters
    ----------
    a_np : numpy.ndarray
        2-D with shape [in_height, in_width]

    w_np : numpy.ndarray
        2-D with shape [filter_height, filter_width].

    Returns
    -------
    b_np : np.ndarray
        2-D with shape [out_height, out_width]
    """

    try:
        return scipy.signal.convolve2d(apad, np.rot90(np.rot90(w_np_fc)), mode="valid")
    except ValueError:
        pass

    assert len(apad.shape) == len(w_np_fc.shape) == 2

    dtype = apad.dtype
    in_height, in_width = apad.shape
    kernel_h, kernel_w = w_np_fc.shape

    output_shape = [a_dim - w_dim + 1 for a_dim, w_dim in zip(apad.shape, w_np_fc.shape)]
    output = np.zeros(output_shape, dtype=apad.dtype)

    for y in range(output_shape[0]):
        for x in range(output_shape[1]):
            output[y][x] = np.sum(apad[y : y + kernel_h, x : x + kernel_w] * w_np_fc)

    return output


def conv2d_nchw_python(a_np, w_np, stride, padding, groups=1):
    """Convolution operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    w_np : numpy.ndarray
        4-D with shape [num_filter, in_channel // groups, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str or a list/tuple of 2 or 4 ints
        Padding size, or ['VALID', 'SAME'], or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 2 ints

    groups : int
        Number of groups

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    a_slices = np.array_split(a_np, groups, axis=1)
    w_slices = np.array_split(w_np, groups, axis=0)
    b_slices = [
        _conv2d_nchw_python(a_slice, w_slice, stride, padding)
        for a_slice, w_slice in zip(a_slices, w_slices)
    ]
    b_np = np.concatenate(b_slices, axis=1)
    return b_np
