# pylint: disable=invalid-name, unused-variable, too-many-locals, unused-argument
"""Conv2D operators"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm
from .pad import pad
from .util import get_pad_tuple, bitpack
from ..util import simplify, get_const_int, get_const_tuple
import numpy as np


# workload description of qconv2d
Workload = namedtuple('Workload',
                      ['in_dtype', 'out_dtype', 'height', 'width', 'in_filter', 'out_filter',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

QuantizedSpatialPackNCHW = namedtuple('SpatialPack', 
                        ['vh', 'vw', 'vc', 'ba', 'bc'])

QuantizedSpatialPackNHWC= namedtuple('SpatialPack', 
                        ['vh', 'vw', 'vc', 'ba', 'bc'])

# RPI version - broken right now
RaspQuantizedSpatialPack = namedtuple('SpatialPack', 
                        ['vh', 'vw', 'vc', 'ba', 'bc', 'split_ci', 'kfactor'])


_WORKLOADS = [
    # workloads of resnet18 on imagenet
    # input_size, input_size, ic, oc, kh, kw, pad, pad, stride, stride
    Workload('uint32', 'int32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
    Workload('uint32', 'int32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
    Workload('uint32', 'int32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
    Workload('uint32', 'int32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
    Workload('uint32', 'int32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
    Workload('uint32', 'int32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
    Workload('uint32', 'int32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
    Workload('uint32', 'int32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
    Workload('uint32', 'int32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
    Workload('uint32', 'int32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
    Workload('uint32', 'int32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
]

@tvm.target.generic_func
def qconv2d(data, kernel, stride, padding,  activation_bits, weight_bits, layout='NCHW', 
           pack_dtype='uint32', out_dtype='int32', dorefa=True):
    """Conv2D operator.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width] or 
                       [batch, in_height, in_width, in_channel]

    filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    layout : str
        layout of data

    activation_bits: int

    weight_bits: int

    out_dtype: str
        return type of convolution

    pack_dtype: str
        bit packing type
    
    dorefa: bool
        method of preforming popcount

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    # search platform specific declaration first
    # default declaration
    if layout == 'NCHW':
        return spatial_pack_nchw(data, kernel, stride, padding, activation_bits, weight_bits, pack_dtype=pack_dtype, 
                                 out_dtype=out_dtype, dorefa=dorefa)
    elif layout == 'NHWC':
        return spatial_pack_nhwc(data, kernel, stride, padding, activation_bits, weight_bits, pack_dtype=pack_dtype, 
                                 out_dtype=out_dtype, dorefa=dorefa)
    else:
        raise ValueError("not support this layout {} yet".format(layout))

def _get_workload(data, kernel, stride, padding, out_dtype, layout):
    """ Get the workload structure. """
    assert layout == "NCHW" or layout == "NHWC", \
        "Only support layouts NCHW and NHWC"
    if layout == "NCHW":
        _, CI, IH, IW = [x.value for x in data.shape]
        CO, _, KH, KW = [x.value for x in kernel.shape]
    else: # NHWC
        IH, IW = data.shape[1].value, data.shape[2].value
        KH, KW, CI, CO = [x for x in get_const_tuple(kernel.shape)]

    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    
    return Workload(data.dtype, out_dtype, IH, IW, CI, CO, KH, KW, HPAD, WPAD, HSTR, WSTR)

@tvm.target.generic_func
def _get_schedule(wkl, layout):
    # pylint: disable=unreachable
    """ Get the platform specific schedule. """
    target = tvm.target.current_target()
    raise RuntimeError(
        "No schedule for current target:{}".format(target))
    # This return has no use, merely to supress pylint warning
    return wkl


def qconv2d_nchw(Input, Filter, stride, padding, activation_bits, weight_bits, out_dtype='int32', pack_type='uint32'):
    assert isinstance(stride, int) or len(stride) == 2
    Input_q = bitpack(Input, activation_bits, pack_axis=1, bit_axis=2, pack_type=pack_type)
    Filter_q = bitpack(Filter, weight_bits, pack_axis=1, bit_axis=4, pack_type=pack_type)
    batch, in_channel, activation_bits, in_height, in_width = Input_q.shape
    num_filter, channel, kernel_h, kernel_w, weight_bits = Filter_q.shape

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (kernel_h, kernel_w))
    pad_before = [0, 0, 0, pad_top, pad_left]
    pad_after = [0, 0, 0, pad_down, pad_right]

    PadInput_q = pad(Input_q, pad_before, pad_after, name="pad_temp")
    # compute the output shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    out_channel = num_filter
    out_height = simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)

    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    b1 = tvm.reduce_axis((0, activation_bits), name='b1')
    b2 = tvm.reduce_axis((0, weight_bits), name='b2')

    def _conv(nn, ff, yy, xx):
        b1b2 = (b1+b2).astype(out_dtype)
        return tvm.sum( 
            (tvm.popcount(PadInput_q[nn, rc, b1, yy * stride_h + ry, xx * stride_w + rx] & 
                Filter_q[ff, rc, ry, rx, b2])<< (b1b2)).astype(out_dtype),
            axis=[rc, ry, rx, b2, b1]).astype(out_dtype)

    return tvm.compute((batch, out_channel, out_height, out_width), _conv, 
        name="QConv2dOutput", tag="qconv2d_nchw")


def qconv2d_nhwc(Input, Filter, stride, padding, activation_bits, weight_bits, out_dtype='int32', pack_type='uint32'):
    assert isinstance(stride, int) or len(stride) == 2
    Input_q = bitpack(Input, activation_bits, pack_axis=3, bit_axis=4, pack_type=pack_type)
    Filter_q = bitpack(Filter, weight_bits, pack_axis=2, bit_axis=4, pack_type=pack_type)
    batch, in_height, in_width, in_channel_q, _ = Input_q.shape
    kernel_h, kernel_w, _, num_filter, _ = Filter_q.shape

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(padding, (kernel_h, kernel_w))
    # compute the output shape
    out_channel = num_filter
    out_height = simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_top, pad_left, 0, 0]
    pad_after = [0, pad_down, pad_right, 0, 0]
    PadInput_q = pad(Input_q, pad_before, pad_after, name="PaddedInput")

    rc = tvm.reduce_axis((0, in_channel_q), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    b1 = tvm.reduce_axis((0, activation_bits), name='b1')
    b2 = tvm.reduce_axis((0, weight_bits), name='b2')

    def _conv(nn, yy, xx, ff):
        return tvm.sum( 
            (tvm.popcount(PadInput_q[nn, yy * stride_h + ry, xx * stride_w + rx, rc, b1] & 
                Filter_q[ry, rx, rc, ff, b2])<< b1b2).astype(out_dtype),
            axis=[rc, ry, rx, b2, b1])
    
    return tvm.compute( (batch, out_height, out_width, out_channel), _conv,
        name="QConv2dOutput", tag="qconv2d_nhwc")


def spatial_pack_nchw(data, kernel, stride, padding, in_bits, weight_bits, pack_dtype, out_dtype, dorefa=False):
    """ Compute convolution with pack on spatial axes. """
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    data_q = bitpack(data, in_bits, pack_axis=1, bit_axis=0, pack_type=pack_dtype)
    kernel_q = bitpack(kernel, weight_bits, pack_axis=1, bit_axis=0, pack_type=pack_dtype)
    IB, _, CI, H, W = data_q.shape
    KB, CO, _, KH, KW = kernel_q.shape
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)

    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    HCAT, WCAT = KH-1, KW-1

    wkl = _get_workload(data, kernel, stride, padding, out_dtype, "NCHW")
    sch = _get_schedule(wkl, "NCHW")
    VH = sch.vh
    VW = sch.vw
    VC = sch.vc

    TH = H + 2*HPAD
    TW = W + 2*WPAD
    OH = (H + 2*HPAD - KH) // HSTR + 1
    OW = (W + 2*WPAD - KW) // WSTR + 1

    dshape = (IB, 1, CI, H, W)
    dpshape = (IB, 1, CI, TH, TW)
    dvshape = (1, TH//(VH*HSTR), TW//(VW*WSTR), CI, VH*HSTR+HCAT, VW*WSTR+WCAT, IB)

    kshape = (KB, CO, CI, KH, KW)
    kvshape = (CO//VC, CI, KH, KW, KB, VC)

    ovshape = (1, CO//VC, OH//VH, OW//VW, VH, VW, VC)
    oshape = (1, CO, OH, OW)

    DOPAD = (HPAD != 0 and WPAD != 0)
    if DOPAD:
        data_pad = pad(data_q, (0, 0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data_q

    data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw, b: \
        data_pad[b][n][ci][h*VH*HSTR+vh][w*VW*WSTR+vw], name='data_vec')

    kernel_vec = tvm.compute(kvshape, lambda co, ci, dh, dw, b, vc: \
        kernel_q[b][co*VC+vc][ci][dh][dw], name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    dh = tvm.reduce_axis((0, KH), name='dh')
    dw = tvm.reduce_axis((0, KW), name='dw')
    b1 = tvm.reduce_axis((0, IB), name='ib')
    b2 = tvm.reduce_axis((0, KB), name='kb')
    
    def _conv(n, co, h, w, vh, vw, vc):
        b1b2 = (b1+b2).astype(out_dtype)
        if dorefa:
            return tvm.sum( 
                (tvm.popcount(data_vec[n, h, w, ci, vh*HSTR+dh, vw*WSTR+dw, b1] &
                    kernel_vec[co, ci, dh, dw, b2, vc])  -
                tvm.popcount(data_vec[n, h, w, ci, vh*HSTR+dh, vw*WSTR+dw, b1] &
                    ~kernel_vec[co, ci, dh, dw, b2, vc])).astype(out_dtype) << b1b2,
                axis=[ci, dh, dw, b1, b2])
        else:
            return tvm.sum( 
                (tvm.popcount(data_vec[n, h, w, ci, vh*HSTR+dh, vw*WSTR+dw, b1] &
                    kernel_vec[co, ci, dh, dw, b2, vc])).astype(out_dtype) << b1b2,
                axis=[ci, dh, dw, b1, b2])

    conv = tvm.compute(ovshape, _conv, name='conv_out')

    return tvm.compute(oshape, lambda n, co, h, w:
        conv[n][co//VC][h//VH][w//VW][h%VH][w%VW][co%VC],
        name='conv_vec', tag='spatial_qconv_nchw')
        


def spatial_pack_nhwc(data, kernel, stride, padding, in_bits, weight_bits, pack_dtype, out_dtype, dorefa=False):
    """ Compute convolution with pack on spatial axes. """
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    data_q = bitpack(data, in_bits, pack_axis=3, bit_axis=4, pack_type=pack_dtype)
    kernel_q = bitpack(kernel, weight_bits, pack_axis=2, bit_axis=4, pack_type=pack_dtype)
    _, H, W, CI, IB = data_q.shape
    KH, KW, _, CO, KB = kernel_q.shape
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)

    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    HCAT, WCAT = KH-1, KW-1

    wkl = _get_workload(data, kernel, stride, padding, out_dtype, "NHWC")
    sch = _get_schedule(wkl, "NHWC")
    VH = sch.vh
    VW = sch.vw
    VC = sch.vc

    PAD_H = H + 2*HPAD
    PAD_W = W + 2*WPAD
    OH = (H + 2*HPAD - KH) // HSTR + 1
    OW = (W + 2*WPAD - KW) // WSTR + 1

    dvshape = (1, PAD_H//(VH*HSTR), PAD_W//(VW*WSTR), VH*HSTR+HCAT, VW*WSTR+WCAT, CI, IB)
    kvshape = (CO, KH, KW, CI, VC, KB)
    ovshape = (1, OH, OW, CO, VH, VW, VC)
    oshape = (1, OH, OW, CO)

    if (HPAD != 0 and WPAD != 0):
        data_pad = pad(data_q, (0, HPAD, WPAD, 0, 0), name="data_pad")
    else:
        data_pad = data_q

    data_vec = tvm.compute(dvshape, lambda n, h, w, vh, vw, ci, b: \
        data_pad[n][h*VH*HSTR+vh][w*VW*WSTR+vw][ci][b], name='data_vec')

    kernel_vec = tvm.compute(kvshape, lambda co, dh, dw, ci, vc, b: \
        kernel_q[dh][dw][ci][co*VC+vc][b], name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    dh = tvm.reduce_axis((0, KH), name='dh')
    dw = tvm.reduce_axis((0, KW), name='dw')
    b1 = tvm.reduce_axis((0, IB), name='ib')
    b2 = tvm.reduce_axis((0, KB), name='kb')

    def _conv(n, h, w, co, vh, vw, vc):
        b1b2 = (b1+b2).astype(out_dtype)
        if dorefa:
            return tvm.sum( 
                (tvm.popcount(data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ci, b1] &
                    kernel_vec[co, dh, dw, ci, vc, b2]) -
                tvm.popcount(data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ci, b1] &
                    ~kernel_vec[co, dh, dw, ci, vc, b2])).astype(out_dtype) << b1b2,
                axis=[dh, dw, ci, b1, b2])
        else:
            return tvm.sum( 
                tvm.popcount(data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ci, b1] &
                    kernel_vec[co, dh, dw, ci, vc, b2]).astype(out_dtype) << b1b2,
                axis=[dh, dw, ci, b1, b2])

    conv = tvm.compute(ovshape, _conv, name='conv')

    return tvm.compute(oshape, lambda n, h, w, co:
        conv[n][h//VH][w//VW][co//VC][h%VH][w%VW][co%VC],
        name='output_unpack', tag='spatial_qconv_nhwc')

_SCH_TO_DECL_FUNC_QUANT = {
    QuantizedSpatialPackNCHW: spatial_pack_nchw,
    QuantizedSpatialPackNHWC: spatial_pack_nhwc,
}
