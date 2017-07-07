# pylint: disable=invalid-name, line-too-long, unused-variable
"""Convolution operators"""
from __future__ import absolute_import as _abs
import tvm
import numpy as np
from .util import get_const_tuple

@tvm.tag_scope(tag="depthwise_conv2d")
def depthwise_conv2d(Input, Filter, Stride, padding):
    """Depthwise convolution operator, as depthwise_conv2d in tensorflow.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.Tensor
        4-D with shape [in_channel, channel_multiplier, filter_height, filter_width]

    Stride : tvm.Tensor
        1-D of size 2

    padding : str
        'VALID' or 'SAME'

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    in_shape = get_const_tuple(Input.shape)
    batch = in_shape[0]
    in_channel = in_shape[1]
    in_height = in_shape[2]
    in_width = in_shape[3]
    filter_shape = get_const_tuple(Filter.shape)
    filter_channel = filter_shape[0]
    channel_multiplier = filter_shape[1]
    filter_height = filter_shape[2]
    filter_width = filter_shape[3]
    stride_h = Stride.asnumpy()[0]
    stride_w = Stride.asnumpy()[1]
    # calculate output shape
    if padding == 'VALID':
        out_channel = in_channel * channel_multiplier
        out_height = (in_height - filter_height) / stride_h + 1
        out_width = (in_width - filter_width) / stride_w + 1
        pad_along_height = 0
        pad_along_width = 0
    if padding == 'SAME':
        out_channel = in_channel * channel_multiplier
        out_height = np.int(np.ceil(float(in_height) / float(stride_h)))
        out_width = np.int(np.ceil(float(in_width) / float(stride_w)))
        pad_along_height = np.int(np.max((out_height - 1) * stride_h + filter_height - in_height, 0))
        pad_along_width = np.int(np.max((out_width - 1) * stride_w + filter_width - in_width, 0))
    height_after_pad = in_height + pad_along_height
    width_after_pad = in_width + pad_along_width
    pad_top = np.int(np.ceil(float(pad_along_height) / 2))
    pad_left = np.int(np.ceil(float(pad_along_width) / 2))
    # padding stage
    PaddedInput = tvm.compute(
        (batch, in_channel, height_after_pad, width_after_pad),
        lambda b, c, i, j: tvm.select(
            tvm.all(i >= pad_top, i - pad_top < in_height, j >= pad_left, j - pad_left < in_width),
            Input[b, c, i - pad_top, j - pad_left], tvm.const(0.0)),
        name="PaddedInput")
    # depthconv stage
    di = tvm.reduce_axis((0, filter_height), name='di')
    dj = tvm.reduce_axis((0, filter_width), name='dj')
    Output = tvm.compute(
        (batch, out_channel, out_height, out_width),
        lambda b, c, i, j: tvm.sum(
            PaddedInput[b, c/channel_multiplier, i*stride_h + di, j*stride_w + dj] * Filter[c/channel_multiplier, c%channel_multiplier, di, dj],
            axis=[di, dj]),
        name='DepthwiseConv2d')
    return Output
