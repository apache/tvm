# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin
"""Conv2D operators"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import numpy as np
import tvm

from .pad import pad
from .util import get_pad_tuple
from ..util import simplify, const_matrix, get_const_tuple

# workload description of conv2d
Workload = namedtuple('Workload',
                      ['in_dtype', 'out_dtype', 'height', 'width', 'in_filter', 'out_filter',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

@tvm.target.generic_func
def conv2d(input, filter, strides, padding, dilation, layout='NCHW', out_dtype=None):
    """Conv2D operator.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        layout of data

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    # search platform specific declaration first
    # default declaration
    if layout == 'NCHW':
        return conv2d_nchw(input, filter, strides, padding, dilation, out_dtype)
    if layout == 'HWCN':
        return conv2d_hwcn(input, filter, strides, padding, dilation, out_dtype)
    if layout == 'NHWC':
        return conv2d_nhwc(input, filter, strides, padding, dilation, out_dtype)
    raise ValueError("not support this layout {} yet".format(layout))


@tvm.target.generic_func
def conv2d_alter_layout(attrs, inputs, tinfos, F):
    """Change Conv2D layout.

    Parameters
    ----------
    attrs : nnvm.top.AttrDict or tvm.attrs.Attrs
        Attributes of current convolution
    inputs : nnvm.symbol or tvm.relay.Expr
        Grouped input symbols
    tinfos : list
        Input shape and dtype
    F: symbol
        The context, can be either nnvm.sym or relay.op

    Note
    ----
    Unlike other TOPI functions, this function operates on both graph level and operator level,
    so we have to pass 'F' to make it support our two versions of graph IR, NNVM and Relay.
    """
    # not to change by default
    return None


def _get_workload(data, kernel, stride, padding, out_dtype):
    """ Get the workload structure. """
    _, CI, IH, IW = [x.value for x in data.shape]
    CO, _, KH, KW = [x.value for x in kernel.shape]
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    assert (data.dtype == kernel.dtype) or (data.dtype == 'uint8' and kernel.dtype == 'int8'), \
        "Do not support inputs with different data types now. ' \
        '{} vs. {}".format(data.dtype, kernel.dtype)
    return Workload(data.dtype, out_dtype, IH, IW, CI, CO, KH, KW, HPAD, WPAD, HSTR, WSTR)


def conv2d_nchw(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Convolution operator in NCHW layout.

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

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
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
            temp[nn, rc, yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w].astype(out_dtype) *
            Filter[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), tag="conv2d_nchw")


def conv2d_hwcn(Input, Filter, stride, padding, dilation, out_dtype=None):
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

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [out_height, out_width, out_channel, batch]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    in_height, in_width, in_channel, batch = Input.shape
    kernel_h, kernel_w, channel, num_filter = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [pad_top, pad_left, 0, 0]
    pad_after = [pad_down, pad_right, 0, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    Output = tvm.compute(
        (out_height, out_width, out_channel, batch),
        lambda yy, xx, ff, nn: tvm.sum(
            PaddedInput[yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w,
                        rc, nn].astype(out_dtype) *
            Filter[ry, rx, rc, ff].astype(out_dtype), axis=[ry, rx, rc]),
        name="Conv2dOutput", tag="conv2d_hwcn")
    return Output


def conv2d_nhwc(Input, Filter, stride, padding, dilation, out_dtype='float32'):
    """Convolution operator in NHWC layout.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    Filter : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_height, in_width, in_channel = Input.shape
    kernel_h, kernel_w, channel, num_filter = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    Output = tvm.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: tvm.sum(
            PaddedInput[nn, yy * stride_h + ry * dilation_h,
                        xx * stride_w + rx * dilation_w, rc].astype(out_dtype) *
            Filter[ry, rx, rc, ff].astype(out_dtype), axis=[ry, rx, rc]),
        name="Conv2dOutput", tag="conv2d_nhwc")
    return Output


@tvm.target.generic_func
def conv2d_NCHWc(data, kernel, stride, padding, dilation, layout, out_layout, out_dtype='float32'):
    """Conv2D operator for nChw[x]c layout.

    Parameters
    ----------
    data : tvm.Tensor
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.Tensor
        6-D with shape
        [num_filter_chunk, in_channel_chunk, filter_height, filter_width,
        in_channel_block, num_filter_block]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        Input data layout

    out_layout : str
        Output data layout

    out_dtype : str
        output data type

    Returns
    -------
    output : tvm.Tensor
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """
    # search platform specific declaration first
    # default declaration
    raise ValueError("missing register for topi.nn.conv2d_NCHWc")


def conv2d_winograd_weight_transform(kernel, tile_size):
    """Weight transformation for winograd

    Parameters
    ----------
    kernel: Tensor
        The raw kernel tensor with layout "NCHW". Only 3x3 kernel is supported for now
    tile_size: int
        Tile size of winograd transform. e.g. 2 for F(2x2, 3x3) and 4 for F(4x4, 3x3)

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [alpha, alpha, CO, CI]
    """
    K = 3

    shape = get_const_tuple(kernel.shape)
    assert shape[2:] == (K, K), "Only support 3x3 kernel"

    r = tile_size + K - 1
    shape = (r, r) + shape[:2]

    if tile_size == 2:
        G_data = np.array([
            [1, 0, 0],
            [1.0/2, 1.0/2, 1.0/2],
            [1.0/2, -1.0/2, 1.0/2],
            [0, 0, 1],
        ], dtype=kernel.dtype)
    elif tile_size == 4:
        G_data = np.array([
            [1 / 4.0, 0, 0],
            [-1 / 6.0, -1 / 6.0, -1 / 6.0],
            [-1 / 6.0, 1 / 6.0, -1 / 6.0],
            [1 / 24.0, 1 / 12.0, 1 / 6.0],
            [1 / 24.0, -1 / 12.0, 1 / 6.0],
            [0, 0, 1]
        ], dtype=kernel.dtype)
    else:
        raise ValueError("Unsupoorted tile size:" + tile_size)

    G = const_matrix(G_data, 'G')
    r_kh = tvm.reduce_axis((0, K), name='r_kh')
    r_kw = tvm.reduce_axis((0, K), name='r_kw')
    return tvm.compute(shape, lambda eps, nu, co, ci:
                       tvm.sum(kernel[co][ci][r_kh][r_kw] *
                               G[eps][r_kh] * G[nu][r_kw],
                               axis=[r_kh, r_kw]), name='transform_weight')


@tvm.target.generic_func
def conv2d_winograd_without_weight_transform(input, filter, strides, padding, dilation,
                                             layout, out_dtype, tile_size):
    """Compute convolution in winograd algorithm. The filter is supposed to be transformed
    in advance.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]
    filter : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]
    strides : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]
    padding : int or str
        Padding size, or ['VALID', 'SAME']
    tile_size: int
        Tile size of winograd transform. e.g. 2 for F(2x2, 3x3) and 4 for F(4x4, 3x3)

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    raise ValueError("missing register for topi.nn.conv2d_winograd_without_weight_transform")


@tvm.target.generic_func
def group_conv2d_nchw(Input, Filter, stride, padding, dilation, groups, out_dtype=None):
    """Group convolution operator in NCHW layout.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.Tensor
        4-D with shape [num_filter, in_channel // groups, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    groups : int
        number of groups

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = get_const_tuple(Input.shape)
    num_filter, _, kernel_h, kernel_w = get_const_tuple(Filter.shape)

    assert in_channel % groups == 0, "input channels must divide group size"
    assert num_filter % groups == 0, "output channels must divide group size"

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (kernel_h, kernel_w))
    # compute the output shape
    out_channel = num_filter
    out_height = simplify(
        (in_height - (kernel_h - 1) * dilation_h - 1 + pad_top + pad_down) // stride_h + 1)
    out_width = simplify(
        (in_width - (kernel_w - 1) * dilation_w - 1 + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad(Input, pad_before, pad_after, name="pad_temp")
    rc = tvm.reduce_axis((0, in_channel // groups), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    return tvm.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: tvm.sum(
            temp[nn, ff // (num_filter//groups) * (in_channel//groups) + rc,
                 yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w].astype(out_dtype) *
            Filter[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), tag="conv2d_nchw")
