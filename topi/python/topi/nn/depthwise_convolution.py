# pylint: disable=invalid-name, unused-variable, too-many-locals
"""Depthwise Convolution operators"""
from __future__ import absolute_import as _abs
import tvm
from .pad import pad
from .util import get_pad_tuple
from ..util import simplify


def depthwise_conv2d_nchw(Input, Filter, stride, padding):
    """Depthwise convolution nchw forward operator.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.Tensor
        4-D with shape [in_channel, channel_multiplier, filter_height, filter_width]

    stride : tuple of two ints
        The spatial stride along height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, in_channel, in_height, in_width = Input.shape
    filter_channel, channel_multiplier, filter_height, filter_width = Filter.shape
    stride_h, stride_w = stride

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (filter_height, filter_width))
    out_channel = simplify(in_channel * channel_multiplier)
    out_height = simplify((in_height - filter_height + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - filter_width + pad_left + pad_right) // stride_w + 1)

    # padding stage
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    # depthconv stage
    di = tvm.reduce_axis((0, filter_height), name='di')
    dj = tvm.reduce_axis((0, filter_width), name='dj')
    Output = tvm.compute(
        (batch, out_channel, out_height, out_width),
        lambda b, c, i, j: tvm.sum(
            (PaddedInput[b, c/channel_multiplier, i*stride_h + di, j*stride_w + dj] *
             Filter[c/channel_multiplier, c%channel_multiplier, di, dj]),
            axis=[di, dj]),
        name='DepthwiseConv2d', tag="depthwise_conv2d_nchw")
    return Output

def depthwise_conv2d_nhwc(Input, Filter, stride, padding):
    """Depthwise convolution nhwc forward operator.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    Filter : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, channel_multiplier]

    Stride : tvm.Tensor
        1-D of size 2

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    batch, in_height, in_width, in_channel = Input.shape
    filter_height, filter_width, filter_channel, channel_multiplier = Filter.shape
    stride_h, stride_w = stride

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (filter_height, filter_width))
    out_channel = simplify(in_channel * channel_multiplier)
    out_height = simplify((in_height - filter_height + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - filter_width + pad_left + pad_right) // stride_w + 1)

    # padding stage
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    # depthconv stage
    di = tvm.reduce_axis((0, filter_height), name='di')
    dj = tvm.reduce_axis((0, filter_width), name='dj')
    Output = tvm.compute(
        (batch, out_height, out_width, out_channel),
        lambda b, i, j, c: tvm.sum(
            (PaddedInput[b, i*stride_h + di, j*stride_w + dj, c/channel_multiplier] *
             Filter[di, dj, c/channel_multiplier, c%channel_multiplier]),
            axis=[di, dj]),
        name='DepthwiseConv2d', tag="depthwise_conv2d_nhwc")
    return Output
