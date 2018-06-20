# pylint: disable=invalid-name, unused-variable, too-many-locals, unused-argument
"""Conv2D operators"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm
from .pad import pad
from .util import get_pad_tuple
from ..util import simplify

# workload description of conv2d
Workload = namedtuple('Workload',
                      ['in_dtype', 'out_dtype', 'height', 'width', 'in_filter', 'out_filter',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

# schedule description of spatial
SpatialPack = namedtuple('SpatialPack',
                         ['vh', 'vw', 'vc', 'ba', 'bc', 'unroll'])

# schedule description of im2col
Im2ColPack = namedtuple('Im2ColPack',
                        ['vp', 'vq', 'ba', 'bc', 'unroll'])

_WORKLOADS = [
    # workloads of resnet18 on imagenet
    Workload('float32', 'float32', 224, 224, 3, 64, 7, 7, 3, 3, 2, 2),
    Workload('float32', 'float32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
    Workload('float32', 'float32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
    Workload('float32', 'float32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
    # workloads of mobile net on imagenet
    Workload('float32', 'float32', 224, 224, 3, 32, 3, 3, 1, 1, 2, 2),
    Workload('float32', 'float32', 112, 112, 32, 64, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 56, 56, 128, 128, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 28, 28, 256, 256, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 14, 14, 512, 512, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 512, 1024, 1, 1, 0, 0, 1, 1),
    Workload('float32', 'float32', 7, 7, 1024, 1024, 1, 1, 0, 0, 1, 1),
    # workloads of resnet18 on imagenet (int16->int32 version)
    Workload('int16', 'int32', 224, 224, 3, 64, 7, 7, 3, 3, 2, 2),
    Workload('int16', 'int32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
    Workload('int16', 'int32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
    Workload('int16', 'int32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
    Workload('int16', 'int32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
    Workload('int16', 'int32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
    Workload('int16', 'int32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
    Workload('int16', 'int32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
    Workload('int16', 'int32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
    Workload('int16', 'int32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
    Workload('int16', 'int32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
    Workload('int16', 'int32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
    # workloads of mobile net on imagenet (int16->int32 version)
    Workload('int16', 'int32', 224, 224, 3, 32, 3, 3, 1, 1, 2, 2),
    Workload('int16', 'int32', 112, 112, 32, 64, 1, 1, 0, 0, 1, 1),
    Workload('int16', 'int32', 56, 56, 64, 128, 1, 1, 0, 0, 1, 1),
    Workload('int16', 'int32', 56, 56, 128, 128, 1, 1, 0, 0, 1, 1),
    Workload('int16', 'int32', 28, 28, 128, 256, 1, 1, 0, 0, 1, 1),
    Workload('int16', 'int32', 28, 28, 256, 256, 1, 1, 0, 0, 1, 1),
    Workload('int16', 'int32', 14, 14, 256, 512, 1, 1, 0, 0, 1, 1),
    Workload('int16', 'int32', 14, 14, 512, 512, 1, 1, 0, 0, 1, 1),
    Workload('int16', 'int32', 7, 7, 512, 1024, 1, 1, 0, 0, 1, 1),
    Workload('int16', 'int32', 7, 7, 1024, 1024, 1, 1, 0, 0, 1, 1),
]

# platform specific schedule
_CONV_SCHEDULE = {}

@tvm.target.generic_func
def conv2d(data, kernel, stride, padding, layout='NCHW', out_dtype=None):
    """Conv2D operator.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

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
        return conv2d_nchw(data, kernel, stride, padding, out_dtype)
    elif layout == 'HWCN':
        return conv2d_hwcn(data, kernel, stride, padding, out_dtype)
    elif layout == 'NHWC':
        return conv2d_nhwc(data, kernel, stride, padding, out_dtype)
    else:
        raise ValueError("not support this layout {} yet".format(layout))


@tvm.target.generic_func
def conv2d_alter_layout(attrs, inputs, tinfos):
    """Change Conv2D layout.

    Parameters
    ----------
    attrs : nnvm.top.AttrDict
        Attributes of current convolution
    inputs : nnvm.symbol
        Grouped input symbols
    tinfos : list
        Input shape and dtype
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
    assert data.dtype == kernel.dtype, \
        "Do not support inputs with different data types now. ' \
        '{} vs. {}".format(data.dtype, kernel.dtype)
    return Workload(data.dtype, out_dtype, IH, IW, CI, CO, KH, KW, HPAD, WPAD, HSTR, WSTR)


@tvm.target.generic_func
def _get_schedule(wkl):
    # pylint: disable=unreachable
    """ Get the platform specific schedule. """
    target = tvm.target.current_target()
    raise RuntimeError(
        "No schedule for current target:{}".format(target))
    # This return has no use, merely to supress pylint warning
    return wkl


def _spatial_pack(data, kernel, stride, padding, out_dtype=None):
    """ Compute convolution with pack on spatial axes. """
    if out_dtype is None:
        out_dtype = data.dtype
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    wkl = _get_workload(data, kernel, stride, padding, out_dtype)
    sch = _get_schedule(wkl)

    H, W = wkl.height, wkl.width
    CI, CO = wkl.in_filter, wkl.out_filter
    KH, KW = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    HCAT, WCAT = KH-1, KW-1

    VH = sch.vh
    VW = sch.vw
    VC = sch.vc
    UNROLL = sch.unroll

    TH = H + 2*HPAD
    TW = W + 2*WPAD
    OH = (H + 2*HPAD - KH) // HSTR + 1
    OW = (W + 2*WPAD - KW) // WSTR + 1

    dshape = (1, CI, H, W)
    dpshape = (1, CI, TH, TW)
    dvshape = (1, TH//(VH*HSTR), TW//(VW*WSTR), CI, VH*HSTR+HCAT, VW*WSTR+WCAT)

    kshape = (CO, CI, KH, KW)
    kvshape = (CO/VC, CI, KH, KW, VC)

    ovshape = (1, CO // VC, OH // VH, OW // VW, VH, VW, VC)
    oshape = (1, CO, OH, OW)

    DOPAD = (HPAD != 0 and WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw: \
        data_pad[n][ci][h*VH*HSTR+vh][w*VW*WSTR+vw], name='data_vec')

    kernel_vec = tvm.compute(kvshape, lambda co, ci, dh, dw, vc: \
        kernel[co*VC+vc][ci][dh][dw], name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    dh = tvm.reduce_axis((0, KH), name='dh')
    dw = tvm.reduce_axis((0, KW), name='dw')

    conv = tvm.compute(ovshape, lambda n, co, h, w, vh, vw, vc: \
        tvm.sum(data_vec[n, h, w, ci, vh*HSTR+dh, vw*WSTR+dw].astype(out_dtype) *
                kernel_vec[co, ci, dh, dw, vc].astype(out_dtype),
                axis=[ci, dh, dw]), name='conv')

    output = tvm.compute(oshape, lambda n, co, h, w:
                         conv[n][co//VC][h/VH][w//VW][h%VH][w%VW][co%VC],
                         name='output_unpack', tag='spatial_conv_output')

    return output


def _im2col_pack(data, kernel, stride, padding, out_dtype=None):
    """ Compute convolution with im2col pack layout. """
    if out_dtype is None:
        out_dtype = data.dtype
    assert data.shape[0].value == 1, "im2col pack convolution only support batch size=1"
    wkl = _get_workload(data, kernel, stride, padding, out_dtype)
    sch = _get_schedule(wkl)

    N = 1
    H, W = wkl.height, wkl.width
    CI = wkl.in_filter
    CO = wkl.out_filter
    KH, KW = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.hpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    OH = (H + 2*HPAD - KH) // HSTR + 1
    OW = (W + 2*WPAD - KW) // WSTR + 1

    P = sch.vp
    Q = sch.vq
    UNROLL = sch.unroll

    dshape = (N, CI, H, W)
    dpshape = (N, CI, H+2*HPAD, W+2*WPAD)
    dcshape = (N, OH, OW, CI, KH, KW)
    dvshape = (N, OH * OW // P, CI, KH, KW, P)

    kshape = (CO, CI, KH, KW)
    kvshape = (CO // Q, CI, KH, KW, Q)

    ovshape = (N, CO // Q, OH * OW // P, P, Q)
    oshape = (N, CO, OH, OW)

    ############### declaration

    DO_PAD = (wkl.hpad != 0 and wkl.wpad != 0)
    if DO_PAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    data_col = tvm.compute(dcshape, lambda n, oh, ow, ci, hk, wk: \
        data_pad[n][ci][oh*HSTR+hk][ow*WSTR+wk], name='data_col')

    data_vec = tvm.compute(dvshape, lambda n, im, ci, hk, wk, vim: \
        data_col[n][(im*P+vim)//OW][(im*P+vim)%OW][ci][hk][wk], name='data_vec')


    kernel_vec = tvm.compute(kvshape, lambda co, ci, dh, dw, vc: \
        kernel[co*Q+vc][ci][dh][dw], name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    hk = tvm.reduce_axis((0, KH), name='hk')
    wk = tvm.reduce_axis((0, KW), name='wk')

    conv = tvm.compute(ovshape, lambda n, co, im, vim, vco: \
        tvm.sum(data_vec[n][im][ci][hk][wk][vim].astype(out_dtype) *
                kernel_vec[co][ci][hk][wk][vco].astype(out_dtype),
                axis=[ci, hk, wk]), name='conv')

    output = tvm.compute(oshape, lambda n, co, h, w: \
                         conv[n][co//Q][(h*OW+w)//P][(h*OW+w)%P][co%Q],
                         name='output_vec', tag='im2col_conv_output')

    return output


def conv2d_nchw(Input, Filter, stride, padding, out_dtype=None):
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

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
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
            temp[nn, rc, yy * stride_h + ry, xx * stride_w + rx].astype(out_dtype) *
            Filter[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), tag="conv2d_nchw")


def conv2d_hwcn(Input, Filter, stride, padding, out_dtype=None):
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
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    in_height, in_width, in_channel, batch = Input.shape
    kernel_h, kernel_w, channel, num_filter = Filter.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
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
            PaddedInput[yy * stride_h + ry, xx * stride_w + rx, rc, nn].astype(out_dtype) *
            Filter[ry, rx, rc, ff].astype(out_dtype), axis=[ry, rx, rc]),
        name="Conv2dOutput", tag="conv2d_hwcn")
    return Output


def conv2d_nhwc(Input, Filter, stride, padding, out_dtype='float32'):
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

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_height,  out_width, out_channel]
    """
    assert isinstance(stride, int) or len(stride) == 2
    batch, in_height, in_width, in_channel = Input.shape
    kernel_h, kernel_w, channel, num_filter = Filter.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (kernel_h, kernel_w))
    # compute the output shape
    out_channel = num_filter
    out_height = simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    Output = tvm.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: tvm.sum(
            PaddedInput[nn, yy * stride_h + ry, xx * stride_w + rx, rc].astype(out_dtype) *
            Filter[ry, rx, rc, ff].astype(out_dtype), axis=[ry, rx, rc]),
        name="Conv2dOutput", tag="conv2d_nhwc")
    return Output

@tvm.target.generic_func
def conv2d_NCHWc(data, kernel, num_filter, kernel_size, stride, padding, out_dtype='float32'):
    """Conv2D operator for nChw[x]c layout.

    Parameters
    ----------
    data : tvm.Tensor
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.Tensor
        6-D with shape
        [num_filter_chunk, in_channel_chunk, filter_height, filter_width,
        in_channel_block, num_filter_block]

    num_filter : int
        number of filters, i.e., output channel size

    kernel_size : tuple of two ints
       [kernel_height, kernel_width]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

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

# map from schedule type to declaration function
_SCH_TO_DECL_FUNC = {
    SpatialPack: _spatial_pack,
    Im2ColPack: _im2col_pack,
}
