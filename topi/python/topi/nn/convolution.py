# pylint: disable=invalid-name, unused-variable, too-many-locals
"""Convolution operators"""
from __future__ import absolute_import as _abs
import tvm
from ..util import simplify
from .pad import pad, _spatial2d_pad_option


def conv2d_nchw(Input, Filter, stride, padding):
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
    assert isinstance(stride, int) or len(stride) == 2
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

def spatial_pack_conv2d(data, kernel, wkl, sch):
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    H, W = wkl.height, wkl.width
    CI = wkl.in_filter
    CO = wkl.out_filter
    HK, WK = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    HCAT, WCAT = HK-1, WK-1

    VH = sch.vh
    VW = sch.vw
    VC = sch.vc
    UNROLL = sch.unroll

    TH = H + 2*HPAD
    TW = W + 2*WPAD
    OH = (H + 2*HPAD - HK) / HSTR + 1
    OW = (W + 2*WPAD - WK) / WSTR + 1

    dshape = (1, CI, H, W)
    dpshape = (1, CI, TH, TW)
    dvshape = (1, TH/(VH*HSTR), TW/(VW*WSTR), CI, VH*HSTR+HCAT, VW*WSTR+WCAT)

    kshape = (CO, CI, HK, WK)
    kvshape = (CO/VC, CI, HK, WK, VC)

    ovshape = (1, CO/VC, OH/VH, OW/VW, VH, VW, VC)
    oshape = (1, CO, OH, OW)

    DOPAD = (HPAD != 0 and WPAD != 0)
    if DOPAD:
        data_pad = tvm.compute(dpshape, lambda n, ci, h, w:
            tvm.select(
                tvm.make.Or(tvm.make.Or((h < HPAD), (h >= H + HPAD)),
                            tvm.make.Or((w < WPAD), (w >= W + WPAD))),
                0.0,
                data[n, ci, h - HPAD, w - WPAD]), name='data_pad')
    else:
        data_pad = data

    data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw:
        data_pad[n][ci][h*VH*HSTR+vh][w*VW*WSTR+vw], name='data_vec')

    kernel_vec = tvm.compute(kvshape, lambda co, ci, dh, dw, vc:
        kernel[co*VC+vc][ci][dh][dw], name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    dh = tvm.reduce_axis((0, HK), name='dh')
    dw = tvm.reduce_axis((0, WK), name='dw')

    output_vec = tvm.compute(ovshape, lambda n, co, h, w, vh, vw, vc: \
        tvm.sum(data_vec[n, h, w, ci, vh*HSTR+dh, vw*WSTR+dw] *
                kernel_vec[co, ci, dh, dw, vc],
                axis=[ci, dh, dw]), name='conv')

    output = tvm.compute(oshape, lambda n, co, h, w: \
        output_vec[n][co/VC][h/VH][w/VW][h%VH][w%VW][co%VC],
        name='output', tag='spatial_conv_output')

    return output


def im2col_pack_conv2d(data, kernel, wkl, sch):
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    N = 1
    H, W = wkl.height, wkl.width
    CI = wkl.in_filter
    CO = wkl.out_filter
    HK, WK = wkl.hkernel, wkl.wkernel
    HPAD, WPAD = wkl.hpad, wkl.hpad
    HSTR, WSTR = wkl.hstride, wkl.wstride

    OH = (H + 2*HPAD - HK) / HSTR + 1
    OW = (W + 2*WPAD - WK) / WSTR + 1

    P = sch.vp
    Q = sch.vq
    UNROLL = sch.unroll

    dshape  = (N, CI, H, W)
    dpshape = (N, CI, H+2*HPAD, W+2*WPAD)
    dcshape = (N, OH, OW, CI, HK, WK)
    dvshape = (N, OH * OW / P, CI, HK, WK, P)

    kshape  = (CO, CI, HK, WK)
    kvshape = (CO / Q, CI, HK, WK, Q)

    ovshape = (N, CO/Q, OH * OW / P, P, Q)
    oshape  = (N, CO, OH, OW)

    ############### declaration

    DO_PAD = (wkl.hpad != 0 and wkl.wpad != 0)
    if DO_PAD:
        data_pad = tvm.compute(dpshape, lambda n, ci, h, w:
            tvm.select(
                tvm.make.Or(tvm.make.Or((h < HPAD), (h >= H + HPAD)),
                            tvm.make.Or((w < WPAD), (w >= W + WPAD))),
                0.0,
                data[n, ci, h - HPAD, w - WPAD]), name='data_pad')
    else:
        data_pad = data

    data_col = tvm.compute(dcshape,
        lambda n, oh, ow, ci, hk, wk: data_pad[n][ci][oh*HSTR+hk][ow*WSTR+wk], name='data_col')

    data_vec = tvm.compute(dvshape, lambda n, im, ci, hk, wk, vim: \
        data_col[n][(im*P+vim)/OW][(im*P+vim)%OW][ci][hk][wk], name='data_vec')


    kernel_vec = tvm.compute(kvshape, lambda co, ci, dh, dw, vc:
        kernel[co*Q+vc][ci][dh][dw], name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    hk = tvm.reduce_axis((0, HK), name='hk')
    wk = tvm.reduce_axis((0, WK), name='wk')

    conv = tvm.compute(ovshape, lambda n, co, im, vim, vco: \
        tvm.sum(data_vec[n][im][ci][hk][wk][vim]*kernel_vec[co][ci][hk][wk][vco], axis=[ci, hk, wk]),
        name='conv')

    output = tvm.compute(oshape, lambda n, co, h, w: \
        conv[n][co/Q][(h*OW+w)/P][(h*OW+w)%P][co%Q],
        name='output_vec', tag='im2col_conv_output')

    return output
