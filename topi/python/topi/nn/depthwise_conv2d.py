# pylint: disable=invalid-name, unused-variable, too-many-locals
"""Depthwise convolution operators"""
from __future__ import absolute_import as _abs
import tvm

from .dilate import dilate
from .pad import pad
from .util import get_pad_tuple
from ..util import simplify


@tvm.target.generic_func
def depthwise_conv2d_nchw(Input, Filter, stride, padding, out_dtype=None):
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

    out_dtype: str, optional
        Output data type

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    out_dtype = Input.dtype if out_dtype is None else out_dtype

    batch, in_channel, in_height, in_width = Input.shape
    filter_channel, channel_multiplier, filter_height, filter_width = Filter.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
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
            (PaddedInput[b, c/channel_multiplier, i*stride_h+di, j*stride_w+dj].astype(out_dtype) *
             Filter[c/channel_multiplier, c%channel_multiplier, di, dj].astype(out_dtype)),
            axis=[di, dj]),
        name='DepthwiseConv2d', tag="depthwise_conv2d_nchw")
    return Output


@tvm.target.generic_func
def depthwise_conv2d_nhwc(Input, Filter, stride, padding, out_dtype=None):
    """Depthwise convolution nhwc forward operator.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    Filter : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, channel_multiplier]

    stride : tuple of two ints
        The spatial stride along height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    out_dtype: str, optional
        Output data type

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    out_dtype = Input.dtype if out_dtype is None else out_dtype

    batch, in_height, in_width, in_channel = Input.shape
    filter_height, filter_width, filter_channel, channel_multiplier = Filter.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
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
            (PaddedInput[b, i*stride_h + di, j*stride_w + dj, c/channel_multiplier].astype(
                out_dtype) *
             Filter[di, dj, c/channel_multiplier, c%channel_multiplier].astype(out_dtype)),
            axis=[di, dj]),
        name='DepthwiseConv2d', tag="depthwise_conv2d_nhwc")
    return Output

def depthwise_conv2d_backward_input_nhwc(Filter, Out_grad, oshape, ishape, stride, padding):
    """Depthwise convolution nhwc backward wrt input operator.

    Parameters
    ----------
    Filter : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, channel_multiplier]

    Out_grad : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]

    stride : tuple of two ints
        The spatial stride along height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]
    """
    batch, in_h, in_w, in_c = ishape
    _, out_h, out_w, out_c = oshape
    filter_h, filter_w, _, channel_multiplier = Filter.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    dilated_out_grad = dilate(Out_grad, [1, stride_h, stride_w, 1], name='dilated_out_grad')

    # padding params in forward propagation
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(padding, (filter_h, filter_w))
    # padding params in backward propagation
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = (filter_h - 1 - fpad_bottom) + (stride_h - 1)
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = (filter_w - 1 - fpad_right) + (stride_w - 1)

    padded_out_grad = pad(dilated_out_grad, \
                                  [0, bpad_top, bpad_left, 0], \
                                  [0, bpad_bottom, bpad_right, 0], \
                                  name='padded_out_grad')

    dh = tvm.reduce_axis((0, filter_h), name='dh')
    dw = tvm.reduce_axis((0, filter_w), name='dw')
    dc = tvm.reduce_axis((0, channel_multiplier), name='dc')

    In_grad = tvm.compute(
        (batch, in_h, in_w, in_c),
        lambda b, h, w, c: tvm.sum(padded_out_grad[b, h+dh, w+dw, c*channel_multiplier + dc] * \
                                   Filter[filter_h-1-dh, filter_w-1-dw, c, dc],
                                   axis=[dh, dw, dc]), tag='depthwise_conv2d_backward_input_nhwc')

    return In_grad


def depthwise_conv2d_backward_weight_nhwc(Input, Out_grad, oshape, fshape, stride, padding):
    """Depthwise convolution nhwc backward wrt weight operator.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    Out_grad : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]

    stride : tuple of two ints
        The spatial stride along height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, channel_multiplier]
    """
    batch, out_h, out_w, out_c = oshape
    filter_h, filter_w, _, channel_multiplier = fshape
    in_c = Input.shape[3].value
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (filter_h, filter_w))

    padded_in = pad(Input, \
                        [0, pad_top, pad_left, 0], \
                        [0, pad_bottom, pad_right, 0], \
                        name='padded_in')

    dh = tvm.reduce_axis((0, Out_grad.shape[1].value), name='dh')
    dw = tvm.reduce_axis((0, Out_grad.shape[2].value), name='dw')
    db = tvm.reduce_axis((0, batch), name='db')

    Weight_grad = tvm.compute(
        (filter_h, filter_w, in_c, channel_multiplier),
        lambda fh, fw, c, m: tvm.sum(
            Out_grad[db, dh, dw, c*channel_multiplier+m%channel_multiplier] *
            padded_in[db, fh+dh*stride_h, fw+dw*stride_w, c], axis=[db, dh, dw]),
        tag='depthwise_conv2d_backward_weight_nhwc')

    return Weight_grad
