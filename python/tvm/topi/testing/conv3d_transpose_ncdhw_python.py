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
"""Convolution 3D transpose in python"""
import numpy as np
import tvm.topi.testing
from tvm.topi.nn.utils import get_pad_tuple3d


def _conv3d_transpose_ncdhw_python(a_np, w_np, stride, padding, output_padding):
    """Transposed 3d convolution operator in NCDHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    w_np : numpy.ndarray
        5-D with shape [in_channel, num_filter, filter_depth, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_depth, stride_height, stride_width]

    padding : int or str
        Padding size

    output_padding : int or list/tuple of three ints
        Used to disambiguate output shape.

    Returns
    -------
    b_np : np.ndarray
        5-D with shape [batch, out_channel, out_depth, out_height, out_width]
    """
    batch, in_c, in_d, in_h, in_w = a_np.shape
    _, out_c, filter_d, filter_h, filter_w = w_np.shape
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride
    if isinstance(output_padding, int):
        opad_d = opad_h = opad_w = output_padding
    else:
        opad_d, opad_h, opad_w = output_padding
    assert opad_d < stride_d and opad_h < stride_h and opad_w < stride_w

    # dilate stage
    dilated_a_np = tvm.topi.testing.dilate_python(a_np, [1, 1, stride_d, stride_h, stride_w])

    # padding stage
    fpad_front, fpad_top, fpad_left, fpad_back, fpad_bottom, fpad_right = get_pad_tuple3d(
        padding, (filter_d, filter_h, filter_w)
    )

    bpad_front = filter_d - 1 - fpad_front
    bpad_back = filter_d - 1 - fpad_back + opad_d
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom + opad_h
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right + opad_w

    padded_a_np = np.zeros(
        (
            batch,
            in_c,
            dilated_a_np.shape[2] + bpad_front + bpad_back,
            dilated_a_np.shape[3] + bpad_top + bpad_bottom,
            dilated_a_np.shape[4] + bpad_left + bpad_right,
        )
    )

    padded_a_np[
        :,
        :,
        bpad_front : dilated_a_np.shape[2] + bpad_front,
        bpad_top : dilated_a_np.shape[3] + bpad_top,
        bpad_left : dilated_a_np.shape[4] + bpad_left,
    ] = dilated_a_np

    # convolution stage
    out_d = (in_d - 1) * stride_d - bpad_front - bpad_back + filter_d
    out_h = (in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (in_w - 1) * stride_w - fpad_left - fpad_right + filter_w

    w_np = np.flip(w_np, axis=[2, 3, 4]).transpose((1, 0, 2, 3, 4))
    b_np = tvm.topi.testing.conv3d_ncdhw_python(
        padded_a_np, w_np, stride=(1, 1, 1), padding=(0, 0, 0)
    )

    return b_np


def conv3d_transpose_ncdhw_python(a_np, w_np, stride, padding, output_padding, groups=1):
    """Transposed 3d convolution operator in NCDHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    w_np : numpy.ndarray
        5-D with shape [in_channel, num_filter, filter_depth, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_depth, stride_height, stride_width]

    padding : int or str
        Padding size

    output_padding : int or list/tuple of three ints
        Used to disambiguate output shape.

    groups : int
        Number of groups

    Returns
    -------
    b_np : np.ndarray
        5-D with shape [batch, out_channel, out_depth, out_height, out_width]
    """
    a_slices = np.array_split(a_np, groups, axis=1)
    w_slices = np.array_split(w_np, groups, axis=0)
    b_slices = [
        _conv3d_transpose_ncdhw_python(a_slice, w_slice, stride, padding, output_padding)
        for a_slice, w_slice in zip(a_slices, w_slices)
    ]
    b_np = np.concatenate(b_slices, axis=1)
    return b_np
