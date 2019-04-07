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
import numpy as np


def deformable_conv2d_nchw_python(a_np, offset_np, w_np, stride, padding, dilation,
                                  deformable_groups, groups):
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

    padding : int or str or a list/tuple of two ints
        Padding size, or ['VALID', 'SAME'], or [pad_height, pad_width]

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
    if isinstance(padding, int):
        pad_h = pad_w = padding * 2
    elif isinstance(padding, (list, tuple)):
        pad_h, pad_w = padding[0] * 2, padding[1] * 2
    else:
        pad_h = 0 if padding == 'VALID' else kernel_h - 1
        pad_w = 0 if padding == 'VALID' else kernel_w - 1
    pad_top = int(np.ceil(float(pad_h) / 2))
    pad_left = int(np.ceil(float(pad_w) / 2))
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation


    def _bilinear(n, c, h, w):
        low_h, low_w = int(h), int(w)
        high_h = min(low_h + 1, in_height - 1)
        high_w = min(low_w + 1, in_width - 1)
        y_lerp = h - low_h
        x_lerp = w - low_w

        bottom = (1 - x_lerp) * a_np[n, c, low_h, low_w] + x_lerp * a_np[n, c, low_h, high_w]
        top = (1 - x_lerp) * a_np[n, c, high_h, low_w] + x_lerp * a_np[n, c, high_h, high_w]
        return (1 - y_lerp) * bottom + y_lerp * top


    a_deform = np.zeros((batch, in_channel, out_height, out_width, kernel_h, kernel_w), dtype=dtype)
    for n, h, w in itertools.product(range(batch), range(out_height), range(out_width)):
        offset = offset_np[n, :, h, w].reshape(deformable_groups, kernel_h, kernel_w, 2)
        in_h = h * stride_h - pad_top
        in_w = w * stride_w - pad_left

        index_h_base, index_w_base = np.meshgrid(
            np.arange(in_h, in_h + kernel_h * dilation_h, dilation_h, dtype=offset_np.dtype),
            np.arange(in_w, in_w + kernel_w * dilation_w, dilation_w, dtype=offset_np.dtype),
            indexing='ij')

        for c, kh, kw in itertools.product(range(in_channel), range(kernel_h), range(kernel_w)):
            dg = c // ic_per_dgroup
            index_h = index_h_base + offset[dg, ..., 0]
            index_w = index_w_base + offset[dg, ..., 1]

            y, x = index_h[kh, kw], index_w[kh, kw]
            if y < 0 or y >= in_height or x < 0 or x >= in_width:
                continue
            a_deform[n, c, h, w, kh, kw] = _bilinear(n, c, y, x)

    b_np = np.zeros((batch, out_channel, out_height, out_width), dtype=dtype)
    for n, c, f, h, w in itertools.product(range(batch), range(in_channel), range(out_channel),
                                           range(out_height), range(out_width)):
        b_np[n, f, h, w] += np.tensordot(a_deform[n, c, h, w], w_np[f, c])

    return b_np
