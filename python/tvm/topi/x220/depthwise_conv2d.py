"""Depthwise convolution operators for x220"""
# Referenced python/tvm/topi/nn/depthwise_conv2d.py

from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm
from tvm import te

from .dilate import dilate
from .pad import pad
from .utils import get_pad_tuple
from ..utils import simplify, get_const_tuple

from math import prod

# workload description of depthwise-conv2d
Workload = namedtuple(
    "Workload",
    [
        "in_dtype",
        "out_dtype",
        "height",
        "width",
        "in_filter",
        "out_filter",
        "kernel_h",
        "kernel_w",
        "padt",
        "padl",
        "padb",
        "padr",
        "dilation_h",
        "dilation_w",
        "stride_h",
        "stride_w",
    ],
)


def _get_workload(data, kernel, stride, padding, dilation, out_dtype, data_layout="NCHW"):
    """Get the workload structure for a depthwise conv2d.

    Input data and filter should use NCHW layout.
    """
    if data_layout == "NCHW":
        _, in_channel, height, width = get_const_tuple(data.shape)
        filter_channel, channel_multiplier, kh, kw = get_const_tuple(kernel.shape)
    elif data_layout == "NHWC":
        _, height, width, in_channel = get_const_tuple(data.shape)
        kh, kw, filter_channel, channel_multiplier = get_const_tuple(kernel.shape)
    elif data_layout == "NCHWc":
        _, in_channel_chunk, height, width, in_channel_block = get_const_tuple(data.shape)
        in_channel = in_channel_chunk * in_channel_block
        (
            filter_channel_chunk,
            cm_chunk,
            kh,
            kw,
            cm_block,
            filter_channel_block,
        ) = get_const_tuple(kernel.shape)
        filter_channel = filter_channel_chunk * filter_channel_block
        channel_multiplier = cm_chunk * cm_block

        assert (
            in_channel_block == filter_channel_block
        ), "Incorrect dimensions, data has block size {}, but filter has block size {}".format(
            in_channel_block, filter_channel_block
        )

    else:
        raise ValueError("Data layout {} not supported".format(data_layout))

    assert (
        in_channel == filter_channel
    ), "Incorrect dimensions, data has {} channels but filter expects {} channels".format(
        in_channel, filter_channel
    )

    out_channel = filter_channel * channel_multiplier
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    )
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    assert (data.dtype == kernel.dtype) or (
        data.dtype == "uint8" and kernel.dtype == "int8"
    ), "Do not support inputs with different data types now. ' \
        '{} vs. {}".format(
        data.dtype, kernel.dtype
    )
    dilated_kernel_h = (kh - 1) * dilation_h + 1
    dilated_kernel_w = (kw - 1) * dilation_w + 1
    pt, pl, pb, pr = get_pad_tuple(padding, (dilated_kernel_h, dilated_kernel_w))
    return Workload(
        data.dtype,
        out_dtype,
        height,
        width,
        in_channel,
        out_channel,
        kh,
        kw,
        pt,
        pl,
        pb,
        pr,
        dilation_h,
        dilation_w,
        HSTR,
        WSTR,
    )


def depthwise_conv2d_nchw(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Depthwise convolution nchw forward operator.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.te.Tensor
        4-D with shape [in_channel, channel_multiplier, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        The spatial stride, or (stride_height, stride_width).

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype: str, optional
        Output data type

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    out_dtype = Input.dtype if out_dtype is None else out_dtype

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    # shape of dilated kernel
    filter_channel, channel_multiplier, filter_height, filter_width = Filter.shape

    dilated_kernel_h = (filter_height - 1) * dilation_h + 1
    dilated_kernel_w = (filter_width - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = simplify(in_channel * channel_multiplier)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    # depthconv stage
    idxdiv = tvm.tir.indexdiv # floor(a/b)
    idxmod = tvm.tir.indexmod # remainder of indexdiv
    di = te.reduce_axis((0, filter_height), name="di")
    dj = te.reduce_axis((0, filter_width), name="dj")

    # Filter = te.compute(
    #     Filter.shape,
    #     lambda o, i, h, w:
    #     Filter[o, i, h, w], name="Filter", tag="Initialize"
    # )
    Acc = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda b, c, i, j:
        te.sum(
            (Input[b, idxdiv(c, channel_multiplier), i * stride_h + di * dilation_h, j * stride_w + dj * dilation_w,].astype(out_dtype)
                * Filter[idxdiv(c, channel_multiplier), idxmod(c, channel_multiplier), di, dj].astype(out_dtype)
            ), axis=[di, dj],
        ), name="Acc", tag="accumulate",
    )
    Output = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda b, c, i, j:
            Acc[b, c, i, j]
        , name="Output", tag="depthwise_conv2d_nchw"
    )
    return Output


def depthwise_conv2d_nhwc(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Depthwise convolution nhwc forward operator.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    Filter : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, channel_multiplier]

    stride : tuple of two ints
        The spatial stride along height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype: str, optional
        Output data type

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    out_dtype = Input.dtype if out_dtype is None else out_dtype

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_height, in_width, in_channel = Input.shape
    # shape of dilated kernel
    filter_height, filter_width, filter_channel, channel_multiplier = Filter.shape

    dilated_kernel_h = (filter_height - 1) * dilation_h + 1
    dilated_kernel_w = (filter_width - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = simplify(in_channel * channel_multiplier)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    # padding stage
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    # depthconv stage
    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    di = te.reduce_axis((0, filter_height), name="di")
    dj = te.reduce_axis((0, filter_width), name="dj")
    Output = te.compute(
        (batch, out_height, out_width, out_channel),
        lambda b, i, j, c: te.sum(
            (
                PaddedInput[
                    b,
                    i * stride_h + di * dilation_h,
                    j * stride_w + dj * dilation_w,
                    idxdiv(c, channel_multiplier),
                ].astype(out_dtype)
                * Filter[
                    di, dj, idxdiv(c, channel_multiplier), idxmod(c, channel_multiplier)
                ].astype(out_dtype)
            ),
            axis=[di, dj],
        ),
        name="DepthwiseConv2d",
        tag="depthwise_conv2d_nhwc",
    )
    return Output