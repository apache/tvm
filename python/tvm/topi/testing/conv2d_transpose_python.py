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
"""Transposed convolution in python"""
import numpy as np
import scipy
import tvm.topi.testing
from tvm.topi.nn.utils import get_pad_tuple


def _conv2d_transpose_nchw_python(a_np, w_np, stride, padding, output_padding):
    """Transposed convolution operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    w_np : numpy.ndarray
        4-D with shape [in_channel, num_filter, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    output_padding : int or a list/tuple of two ints
        Use to disambiguate the output shape.

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, in_c, in_h, in_w = a_np.shape
    _, out_c, filter_h, filter_w = w_np.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    if isinstance(output_padding, int):
        opad_h = opad_w = output_padding
    else:
        opad_h, opad_w = output_padding
    assert opad_h < stride_h and opad_w < stride_w
    # dilate stage
    dilated_a_np = tvm.topi.testing.dilate_python(a_np, [1, 1, stride_h, stride_w])
    # padding stage
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(padding, (filter_h, filter_w))
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom + opad_h
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right + opad_w
    padded_a_np = np.zeros(
        (
            batch,
            in_c,
            dilated_a_np.shape[2] + bpad_top + bpad_bottom,
            dilated_a_np.shape[3] + bpad_left + bpad_right,
        )
    ).astype(a_np.dtype)
    padded_a_np[
        :,
        :,
        bpad_top : dilated_a_np.shape[2] + bpad_top,
        bpad_left : dilated_a_np.shape[3] + bpad_left,
    ] = dilated_a_np
    # convolution stage
    out_h = (in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h + opad_h
    out_w = (in_w - 1) * stride_w - fpad_left - fpad_right + filter_w + opad_w
    b_np = np.zeros((batch, out_c, out_h, out_w)).astype(a_np.dtype)
    for n in range(batch):
        for f in range(out_c):
            for c in range(in_c):
                out = scipy.signal.convolve2d(padded_a_np[n, c], w_np[c, f], mode="valid")
                b_np[n, f] += out
    return b_np


def conv2d_transpose_nhwc_python(
    a_nhwc, weight, weight_format, stride, padding, output_padding=(0, 0)
):
    """Transposed convolution operator in NHWC layout.

    Parameters
    ----------
    a_nhwc : numpy.ndarray
        4-D with shape [batch, in_height, in_width, in_channel]

    weight : numpy.ndarray
        4-D in formats HWIO, HWOI, OIHW or IOHW

    weight_format : str
        ['HWIO', 'HWOI', 'OIHW', 'IOHW']

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    assert a_nhwc.ndim == 4, "a_nhwc number of dimensions should be 4"
    assert weight.ndim == 4, "weight number of dimensions should be 4"

    a_nchw = np.transpose(a_nhwc, (0, 3, 1, 2))

    # conv2d_transpose_nchw_python needs kernel layout to be IOHW
    if weight_format == "HWIO":
        w_iohw = np.transpose(weight, (2, 3, 0, 1))
    elif weight_format == "HWOI":
        w_iohw = np.transpose(weight, (3, 2, 0, 1))
    elif weight_format == "OIHW":
        w_iohw = np.transpose(weight, (1, 0, 2, 3))
    elif weight_format == "IOHW":
        w_iohw = weight
    else:
        raise ValueError("Valid weight_formats are HWIO, HWOI, OIHW or IOHW")

    res_nchw = conv2d_transpose_nchw_python(
        a_nchw, w_iohw, stride, padding, output_padding=output_padding
    )
    res_nhwc = np.transpose(res_nchw, (0, 2, 3, 1))
    return res_nhwc


def conv2d_transpose_nchw_python(a_np, w_np, stride, padding, output_padding, groups=1):
    """Convolution operator in NCHW layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    w_np : numpy.ndarray
        4-D with shape [in_channel, num_filter // groups, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    output_padding : int or a list/tuple of two ints
        Use to disambiguate the output shape.

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
        _conv2d_transpose_nchw_python(a_slice, w_slice, stride, padding, output_padding)
        for a_slice, w_slice in zip(a_slices, w_slices)
    ]
    b_np = np.concatenate(b_slices, axis=1)
    return b_np
