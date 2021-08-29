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
# pylint: disable=invalid-name, unused-variable, line-too-long
"""Depthwise convolution in python"""
import numpy as np

from tvm.topi.nn.utils import get_pad_tuple
from .common import _convolve2d


def depthwise_conv2d_python_nchw(input_np, filter_np, stride, padding):
    """Depthwise convolution operator in NCHW layout.

    Parameters
    ----------
    input_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    filter_np : numpy.ndarray
        4-D with shape [in_channel, channel_multiplier, filter_height, filter_width]

    stride : list / tuple of 2 ints
        [stride_height, stride_width]

    padding : str
        'VALID' or 'SAME'

    Returns
    -------
    output_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, in_channel, in_height, in_width = input_np.shape
    _, channel_multiplier, filter_height, filter_width = filter_np.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (filter_height, filter_width))
    pad_h = pad_top + pad_bottom
    pad_w = pad_left + pad_right

    out_channel = in_channel * channel_multiplier
    out_height = (in_height - filter_height + pad_h) // stride_h + 1
    out_width = (in_width - filter_width + pad_w) // stride_w + 1
    output_np = np.zeros((batch, out_channel, out_height, out_width))

    for i in range(batch):
        for j in range(out_channel):
            apad = input_np[i, j // channel_multiplier, :, :]
            if pad_h or pad_w:
                apad = np.pad(apad, [(pad_top, pad_bottom), (pad_left, pad_right)])

            conv = _convolve2d(
                apad,
                np.rot90(filter_np[j // channel_multiplier, j % channel_multiplier, :, :], k=2),
            )
            output_np[i, j, :, :] = conv[
                ::stride_h,
                ::stride_w,
            ]

    return output_np


def depthwise_conv2d_python_nchwc(input_np, filter_np, stride, padding):
    """Depthwise convolution operator in NCHWc layout.

    Parameters
    ----------
    input_np : numpy.ndarray
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    filter_np : numpy.ndarray
        6-D with shape [out_channel_chunk, channel_multiplier_chunk,
                        filter_height, filter_width,
                        channel_multiplier_block, out_channel_block]

    stride : list / tuple of 2 ints
        [stride_height, stride_width]

    padding : str
        'VALID' or 'SAME'

    Returns
    -------
    output_np : np.ndarray
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """
    # Transform to NCHW
    batch_size, in_channel_chunk, in_height, in_width, in_channel_block = input_np.shape
    input_nchw = input_np.transpose(0, 1, 4, 2, 3).reshape(
        (batch_size, in_channel_chunk * in_channel_block, in_height, in_width)
    )

    (
        out_channel_chunk,
        channel_multiplier_chunk,
        filter_height,
        filter_width,
        channel_multiplier_block,
        out_channel_block,
    ) = filter_np.shape
    filter_nchw = filter_np.transpose(0, 5, 1, 4, 2, 3).reshape(
        (
            out_channel_chunk * out_channel_block,
            channel_multiplier_chunk * channel_multiplier_block,
            filter_height,
            filter_width,
        )
    )

    # Perform conv2d
    output_np = depthwise_conv2d_python_nchw(input_nchw, filter_nchw, stride, padding)

    # Transform back to NCHWc

    # pylint: disable=unpacking-non-sequence
    batch_size, out_channel, out_height, out_width = output_np.shape
    return output_np.reshape(
        (batch_size, out_channel_chunk, out_channel_block, out_height, out_width)
    ).transpose(0, 1, 3, 4, 2)


def depthwise_conv2d_python_nhwc(input_np, filter_np, stride, padding):
    """Depthwise convolution operator in nhwc layout.

    Parameters
    ----------
    input_np : numpy.ndarray
        4-D with shape [batch, in_height, in_width, in_channel]

    filter_np : numpy.ndarray
        4-D with shape [filter_height, filter_width, in_channel, channel_multiplier]

    stride : list / tuple of 2 ints
        [stride_height, stride_width]

    padding : str
        'VALID' or 'SAME'

    Returns
    -------
    output_np : np.ndarray
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    input_nchw = input_np.transpose(0, 3, 1, 2)
    filter_nchw = filter_np.transpose(2, 3, 0, 1)
    output_nchw = depthwise_conv2d_python_nchw(input_nchw, filter_nchw, stride, padding)
    return output_nchw.transpose(0, 2, 3, 1)
