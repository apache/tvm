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
# pylint: disable=invalid-name, too-many-nested-blocks
"""Gradient of conv2d with respect to weight in python"""
import numpy as np


# Reference: cutlass/tools/util/include/cutlass/util/reference/host/convolution.h
def conv2d_backward_weight_nchw_python(
    dy_np, x_np, kernel_size, stride, padding, groups=1, channels=None
):
    """Gradient of the conv2d op with respect to weight, in NCHW layout.

    Parameters
    ----------
    dy_np : numpy.ndarray
        4-D with shape [batch, in_channel, out_height, out_width]

    x_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel_size : tuple of two ints
        Height and width of the weight

    stride : tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : tuple of two ints
        Spatial padding, or [pad_h, pad_w]

    Returns
    -------
    dw_np : np.ndarray
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    """
    N, C, H, W = x_np.shape
    _, K, P, Q = dy_np.shape
    R, S = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    is_depth_wise = C == K and C == groups

    if is_depth_wise:
        assert channels == groups, "Only channel_mult == 1 supported for now."
        dw = np.zeros((K, 1, R, S)).astype(dy_np.dtype)
    else:
        assert groups == 1, "General grouped conv2d not supported for now."
        dw = np.zeros((K, C, R, S)).astype(dy_np.dtype)

    for k in range(K):
        for r in range(R):
            for s in range(S):
                for c in range(dw.shape[1]):
                    acc = 0
                    for n in range(N):
                        for p in range(P):
                            for q in range(Q):
                                if not is_depth_wise:
                                    in_c = c
                                else:
                                    in_c = k

                                coord = (
                                    n,
                                    in_c,
                                    p * stride_h - pad_h + r,
                                    q * stride_w - pad_w + s,
                                )

                                if (
                                    coord[2] < H
                                    and coord[2] >= 0
                                    and coord[3] < W
                                    and coord[3] >= 0
                                ):
                                    acc += dy_np[n, k, p, q] * x_np[coord]

                    dw[k, c, r, s] = acc

    return dw


def conv2d_backward_weight_python(
    dy_np, x_np, kernel_size, stride, padding, layout="NCHW", groups=1, channels=None
):
    """Gradient of the conv2d op with respect to weight, in NCHW or NHWC layout.

    Parameters
    ----------
    dy_np : numpy.ndarray
        4-D with shape [batch, in_channel, out_height, out_width] for NCHW layout

    x_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width] for NCHW layout

    kernel_size : tuple of two ints
        Height and width of the weight

    stride : tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : tuple of two ints
        Spatial padding, or [pad_h, pad_w]

    layout: string
        Layout of dy_np and x_np

    groups: int
        Number of groups for grouped convolution.

    channels : int
        Number of output channels of this convolution.

    Returns
    -------
    dw_np : np.ndarray
        Tensor of shape [num_filter, in_channel, filter_height, filter_width] for NCHW layout,
        [num_filter, filter_height, filter_width, in_channel] for NHWC layout.
    """
    if layout == "NCHW":
        return conv2d_backward_weight_nchw_python(
            dy_np, x_np, kernel_size, stride, padding, groups, channels
        )

    dw_np_oihw = conv2d_backward_weight_nchw_python(
        np.transpose(dy_np, [0, 3, 1, 2]),
        np.transpose(x_np, [0, 3, 1, 2]),
        kernel_size,
        stride,
        padding,
        groups,
        channels,
    )
    return np.transpose(dw_np_oihw, [0, 2, 3, 1])
