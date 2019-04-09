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
from scipy import signal

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

    # calculate output shape
    if padding == 'VALID':
        out_channel = in_channel * channel_multiplier
        out_height = (in_height - filter_height) // stride_h + 1
        out_width = (in_width - filter_width) // stride_w + 1
        output_np = np.zeros((batch, out_channel, out_height, out_width))
        for i in range(batch):
            for j in range(out_channel):
                output_np[i, j, :, :] = signal.convolve2d(input_np[i, j//channel_multiplier, :, :], \
                    np.rot90(filter_np[j//channel_multiplier, j%channel_multiplier, :, :], 2), \
                    mode='valid')[0:(in_height - filter_height + 1):stride_h, 0:(in_width - filter_height + 1):stride_w]
    if padding == 'SAME':
        out_channel = in_channel * channel_multiplier
        out_height = np.int(np.ceil(float(in_height) / float(stride_h)))
        out_width = np.int(np.ceil(float(in_width) / float(stride_w)))
        output_np = np.zeros((batch, out_channel, out_height, out_width))
        pad_along_height = np.int(np.max((out_height - 1) * stride_h + filter_height - in_height, 0))
        pad_along_width = np.int(np.max((out_width - 1) * stride_w + filter_width - in_width, 0))
        pad_top_tvm = np.int(np.ceil(float(pad_along_height) / 2))
        pad_left_tvm = np.int(np.ceil(float(pad_along_width) / 2))
        pad_top_scipy = np.int(np.ceil(float(filter_height - 1) / 2))
        pad_left_scipy = np.int(np.ceil(float(filter_width - 1) / 2))
        index_h = pad_top_scipy - pad_top_tvm
        index_w = pad_left_scipy - pad_left_tvm
        for i in range(batch):
            for j in range(out_channel):
                output_np[i, j, :, :] = signal.convolve2d(input_np[i, j//channel_multiplier, :, :], \
                    np.rot90(filter_np[j//channel_multiplier, j%channel_multiplier, :, :], 2), \
                    mode='same')[index_h:in_height:stride_h, index_w:in_width:stride_w]

    return output_np

def depthwise_conv2d_python_nhwc(input_np, filter_np, stride, padding):
    """Depthwise convolution operator in nchw layout.

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
    batch, in_height, in_width, in_channel = input_np.shape
    filter_height, filter_width, _, channel_multiplier = filter_np.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    # calculate output shape
    if padding == 'VALID':
        out_channel = in_channel * channel_multiplier
        out_height = (in_height - filter_height) // stride_h + 1
        out_width = (in_width - filter_width) // stride_w + 1
        output_np = np.zeros((batch, out_height, out_width, out_channel))
        for i in range(batch):
            for j in range(out_channel):
                output_np[i, :, :, j] = signal.convolve2d(input_np[i, :, :, j//channel_multiplier], \
                    np.rot90(filter_np[:, :, j//channel_multiplier, j%channel_multiplier], 2), \
                    mode='valid')[0:(in_height - filter_height + 1):stride_h, 0:(in_width - filter_height + 1):stride_w]
    if padding == 'SAME':
        out_channel = in_channel * channel_multiplier
        out_height = np.int(np.ceil(float(in_height) / float(stride_h)))
        out_width = np.int(np.ceil(float(in_width) / float(stride_w)))
        output_np = np.zeros((batch, out_height, out_width, out_channel))
        pad_along_height = np.int(np.max((out_height - 1) * stride_h + filter_height - in_height, 0))
        pad_along_width = np.int(np.max((out_width - 1) * stride_w + filter_width - in_width, 0))
        pad_top_tvm = np.int(np.ceil(float(pad_along_height) / 2))
        pad_left_tvm = np.int(np.ceil(float(pad_along_width) / 2))
        pad_top_scipy = np.int(np.ceil(float(filter_height - 1) / 2))
        pad_left_scipy = np.int(np.ceil(float(filter_width - 1) / 2))
        index_h = pad_top_scipy - pad_top_tvm
        index_w = pad_left_scipy - pad_left_tvm
        for i in range(batch):
            for j in range(out_channel):
                output_np[i, :, :, j] = signal.convolve2d(input_np[i, :, :, j//channel_multiplier], \
                    np.rot90(filter_np[:, :, j//channel_multiplier, j%channel_multiplier], 2), \
                    mode='same')[index_h:in_height:stride_h, index_w:in_width:stride_w]

    return output_np
