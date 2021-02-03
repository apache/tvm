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
# pylint: disable=invalid-name, too-many-locals, too-many-arguments
"""Deformable convolution in python"""
import itertools
import math
import numpy as np
from tvm.topi.nn.utils import get_pad_tuple


def deformable_conv2d_nchw_python(
    a_np, offset_np, w_np, stride, padding, dilation, deformable_groups, groups
):
    """Deformable convolution operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    offset_np : numpy.ndarray
        4-D with shape [batch, deformable_groups * filter_height * filter_width * 2,
                        out_height, out_width]

    w_np : numpy.ndarray
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str or a list/tuple of 2 or 4 ints
        Padding size, or ['VALID', 'SAME'], or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 2 ints

    dilation : int or a list/tuple of two ints
        Dilation size, or [dilate_height, dilate_width]

    deformable_groups : int
        Number of deformable groups

    groups : int
        Number of groups

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, in_channel, in_height, in_width = a_np.shape
    out_channel, _, kernel_h, kernel_w = w_np.shape
    out_height, out_width = offset_np.shape[-2:]
    dtype = a_np.dtype
    ic_per_dgroup = in_channel // deformable_groups
    assert groups == 1, "deformable_conv2d_nchw_python does not support groups > 1"

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    pad_top, pad_left, _, _ = get_pad_tuple(padding, (kernel_h, kernel_w))

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    def _bilinear(n, c, h, w):
        y_low = int(math.floor(h))
        x_low = int(math.floor(w))
        y_high = y_low + 1
        x_high = x_low + 1

        wy_h = h - y_low
        wx_h = w - x_low
        wy_l = 1 - wy_h
        wx_l = 1 - wx_h

        val = 0
        for wx, xp in zip((wx_l, wx_h), (x_low, x_high)):
            for wy, yp in zip((wy_l, wy_h), (y_low, y_high)):
                if 0 <= yp < in_height and 0 <= xp < in_width:
                    val += wx * wy * a_np[n, c, yp, xp]
        return val

    a_deform = np.zeros((batch, in_channel, out_height, out_width, kernel_h, kernel_w), dtype=dtype)
    for n, h, w in itertools.product(range(batch), range(out_height), range(out_width)):
        offset = offset_np[n, :, h, w].reshape(deformable_groups, kernel_h, kernel_w, 2)
        in_h = h * stride_h - pad_top
        in_w = w * stride_w - pad_left

        index_h_base, index_w_base = np.meshgrid(
            np.arange(in_h, in_h + kernel_h * dilation_h, dilation_h, dtype=offset_np.dtype),
            np.arange(in_w, in_w + kernel_w * dilation_w, dilation_w, dtype=offset_np.dtype),
            indexing="ij",
        )

        for c, kh, kw in itertools.product(range(in_channel), range(kernel_h), range(kernel_w)):
            dg = c // ic_per_dgroup
            index_h = index_h_base + offset[dg, ..., 0]
            index_w = index_w_base + offset[dg, ..., 1]

            y, x = index_h[kh, kw], index_w[kh, kw]
            if y < 0 or y >= in_height or x < 0 or x >= in_width:
                continue
            a_deform[n, c, h, w, kh, kw] = _bilinear(n, c, y, x)

    b_np = np.zeros((batch, out_channel, out_height, out_width), dtype=dtype)
    for n, c, f, h, w in itertools.product(
        range(batch), range(in_channel), range(out_channel), range(out_height), range(out_width)
    ):
        b_np[n, f, h, w] += np.tensordot(a_deform[n, c, h, w], w_np[f, c])

    return b_np


def deformable_conv2d_nhwc_python(
    a_np, offset_np, w_np, stride, padding, dilation, deformable_groups, groups
):
    """Deformable convolution operator in NHWC layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_height, in_width, in_channel]

    offset_np : numpy.ndarray
        4-D with shape [batch, out_height, out_width,
                        deformable_groups * filter_height * filter_width * 2]

    w_np : numpy.ndarray
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str or a list/tuple of 2 or 4 ints
        Padding size, or ['VALID', 'SAME'], or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 2 ints

    dilation : int or a list/tuple of two ints
        Dilation size, or [dilate_height, dilate_width]

    deformable_groups : int
        Number of deformable groups

    groups : int
        Number of groups

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    a_np = np.transpose(a_np, [0, 3, 1, 2])  # NHWC -> NCHW
    offset_np = np.transpose(offset_np, [0, 3, 1, 2])  # NHWC -> NCHW
    w_np = np.transpose(w_np, [3, 2, 0, 1])  # HWIO -> OIHW
    b_np = deformable_conv2d_nchw_python(
        a_np, offset_np, w_np, stride, padding, dilation, deformable_groups, groups
    )
    b_np = np.transpose(b_np, [0, 2, 3, 1])  # NCHW -> NHWC
    return b_np
