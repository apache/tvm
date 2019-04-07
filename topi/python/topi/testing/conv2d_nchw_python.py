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
import scipy.signal


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

    padding : int or str or a list/tuple of two ints
        Padding size, or ['VALID', 'SAME'], or [pad_height, pad_width]

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
    if isinstance(padding, int):
        pad_h = pad_w = padding * 2
    elif isinstance(padding, (list, tuple)):
        pad_h, pad_w = padding[0] * 2, padding[1] * 2
    else:
        pad_h = 0 if padding == 'VALID' else kernel_h - 1
        pad_w = 0 if padding == 'VALID' else kernel_w - 1
    pad_top = int(np.ceil(float(pad_h) / 2))
    pad_bottom = pad_h - pad_top
    pad_left = int(np.ceil(float(pad_w) / 2))
    pad_right = pad_w - pad_left
    # compute the output shape
    out_channel = num_filter
    out_height = (in_height - kernel_h + pad_h) // stride_h + 1
    out_width = (in_width - kernel_w + pad_w) // stride_w + 1
    b_np = np.zeros((batch, out_channel, out_height, out_width))
    # computation
    for n in range(batch):
        for f in range(out_channel):
            for c in range(in_channel):
                if pad_h > 0 or pad_w > 0:
                    apad = np.zeros((in_height + pad_h, in_width + pad_w))
                    if pad_h == 0:
                        apad[:, pad_left:-pad_right] = a_np[n, c]
                    elif pad_w == 0:
                        apad[pad_top:-pad_bottom, :] = a_np[n, c]
                    else:
                        apad[pad_top:-pad_bottom, pad_left:-pad_right] = a_np[n, c]
                else:
                    apad = a_np[n, c]
                out = scipy.signal.convolve2d(
                    apad, np.rot90(np.rot90(w_np[f, c])), mode='valid')
                b_np[n, f] += out[::stride_h, ::stride_w]
    return b_np


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

    padding : int or str or a list/tuple of two ints
        Padding size, or ['VALID', 'SAME'], or [pad_height, pad_width]

    groups : int
        Number of groups

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    a_slices = np.array_split(a_np, groups, axis=1)
    w_slices = np.array_split(w_np, groups, axis=0)
    b_slices = [_conv2d_nchw_python(a_slice, w_slice, stride, padding)
                for a_slice, w_slice in zip(a_slices, w_slices)]
    b_np = np.concatenate(b_slices, axis=1)
    return b_np
