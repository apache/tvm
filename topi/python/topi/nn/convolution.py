# pylint: disable=invalid-name, unused-variable, too-many-locals
"""Convolution operators"""
from __future__ import absolute_import as _abs
import tvm
from ..util import simplify
from .pad import pad, _spatial2d_pad_option


def conv2d_nchw(Input, Filter, stride, padding):
    """Convolution operator in HWCN layout.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(padding, int) or padding in ['VALID', 'SAME']
    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    pad_top, pad_left, pad_down, pad_right = _spatial2d_pad_option(
        padding, (kernel_h, kernel_w))
    # compute the output shape
    out_channel = num_filter
    out_height = simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad(Input, pad_before, pad_after, name="pad_temp")
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')

    return tvm.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: tvm.sum(
            temp[nn, rc, yy * stride_h + ry, xx * stride_w + rx] * Filter[ff, rc, ry, rx],
            axis=[rc, ry, rx]), tag="conv2d_nchw")


def conv2d_hwcn(Input, Filter, stride, padding):
    """Convolution operator in HWCN layout.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [in_height, in_width, in_channel, batch]

    Filter : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [out_height, out_width, out_channel, batch]
    """
    assert isinstance(stride, int) or len(stride) == 2
    in_height, in_width, in_channel, batch = Input.shape
    kernel_h, kernel_w, channel, num_filter = Filter.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    pad_top, pad_left, pad_down, pad_right = _spatial2d_pad_option(
        padding, (kernel_h, kernel_w))
    # compute the output shape
    out_channel = num_filter
    out_height = simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [pad_top, pad_left, 0, 0]
    pad_after = [pad_down, pad_right, 0, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    Output = tvm.compute(
        (out_height, out_width, out_channel, batch),
        lambda yy, xx, ff, nn: tvm.sum(
            PaddedInput[yy * stride_h + ry, xx * stride_w + rx, rc, nn] * Filter[ry, rx, rc, ff],
            axis=[ry, rx, rc]),
        name="Conv2dOutput", tag="conv2d_hwcn")
    return Output

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

    padding : str
        'VALID' or 'SAME'

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    batch, in_channel, in_height, in_width = Input.shape
    filter_channel, channel_multiplier, filter_height, filter_width = Filter.shape
    stride_h, stride_w = stride

    pad_top, pad_left, pad_down, pad_right = _spatial2d_pad_option(
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

    padding : str
        'VALID' or 'SAME'

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    batch, in_height, in_width, in_channel = Input.shape
    filter_height, filter_width, filter_channel, channel_multiplier = Filter.shape
    stride_h, stride_w = stride

    pad_top, pad_left, pad_down, pad_right = _spatial2d_pad_option(
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
