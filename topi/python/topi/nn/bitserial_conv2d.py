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
# pylint: disable=invalid-name, too-many-locals, too-many-arguments
"""Bitserial Conv2D operators"""
from __future__ import absolute_import as _abs
import tvm
from tvm import autotvm
from .pad import pad
from .util import get_pad_tuple
from .bitserial_util import bitpack, binary_op_multiplier
from ..util import get_const_tuple

@tvm.target.generic_func
def bitserial_conv2d_nchw(data, kernel, stride, padding, activation_bits, weight_bits,
                          pack_dtype='uint32', out_dtype='int16', unipolar=True):
    """Bitserial Conv2D operator.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two or four ints
        padding size, [pad_height, pad_width], [pad_top, pad_left, pad_down, pad_right]

    activation_bits: int
        number of bits used for activations/input elements

    weight_bits: int
        number of bits used for weight elements

    out_dtype: str
        return type of convolution

    pack_dtype: str
        bit packing type

    unipolar: bool
        if binarization style is in unipolar 1/0 format, instead of bipolar -1/+1 format

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    assert isinstance(stride, int) or len(stride) == 2
    Input_q = bitpack(data, activation_bits, pack_axis=1, bit_axis=2, pack_type=pack_dtype)
    Filter_q = bitpack(filter, weight_bits, pack_axis=1, bit_axis=4, pack_type=pack_dtype)
    batch, in_channel, activation_bits, in_height, in_width = Input_q.shape
    num_filter, _, kernel_h, kernel_w, weight_bits = Filter_q.shape

    if isinstance(padding, int) or (isinstance(padding, (tuple, list)) and len(padding) == 2):
        TPAD, LPAD, DPAD, RPAD = get_pad_tuple(padding, kernel)
    else:
        TPAD, LPAD, DPAD, RPAD = padding
    pad_before = [0, 0, 0, TPAD, LPAD]
    pad_after = [0, 0, 0, DPAD, RPAD]

    PadInput_q = pad(Input_q, pad_before, pad_after, name="pad_temp")
    # compute the output shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    out_channel = num_filter
    out_height = (in_height - kernel_h + TPAD + DPAD) // stride_h + 1
    out_width = (in_width - kernel_w + LPAD + RPAD) // stride_w + 1

    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    b1 = tvm.reduce_axis((0, activation_bits), name='b1')
    b2 = tvm.reduce_axis((0, weight_bits), name='b2')

    if unipolar:
        def _conv(nn, ff, yy, xx):
            b1b2 = (b1+b2).astype(out_dtype)
            return tvm.sum(
                ((tvm.popcount(PadInput_q[nn, rc, b1, yy * stride_h + ry, xx * stride_w + rx] &
                               Filter_q[ff, rc, ry, rx, b2]) -
                  tvm.popcount(PadInput_q[nn, rc, b1, yy * stride_h + ry, xx * stride_w + rx] &
                               ~Filter_q[ff, rc, ry, rx, b2]))
                 << (b1b2)).astype(out_dtype),
                axis=[rc, ry, rx, b2, b1]).astype(out_dtype)
    else:
        def _conv(nn, ff, yy, xx):
            b1b2 = (b1+b2).astype(out_dtype)
            return tvm.sum((tvm.popcount(
                PadInput_q[nn, rc, b1, yy * stride_h + ry, xx * stride_w + rx] &
                Filter_q[ff, rc, ry, rx, b2])<< (b1b2)).astype(out_dtype),
                           axis=[rc, ry, rx, b2, b1]).astype(out_dtype)

    return tvm.compute((batch, out_channel, out_height, out_width), _conv,
                       name="Conv2dOutput", tag="bitserial_conv2d_nchw")

@tvm.target.generic_func
def bitserial_conv2d_nhwc(data, kernel, stride, padding, activation_bits, weight_bits,
                          pack_dtype='uint32', out_dtype='int16', unipolar=True):
    """Bitserial Conv2D operator.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    filter : tvm.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two or four ints
        padding size, [pad_height, pad_width], [pad_top, pad_left, pad_down, pad_right]

    activation_bits: int
        number of bits used for activations/input elements

    weight_bits: int
        number of bits used for weight elements

    out_dtype: str
        return type of convolution

    pack_dtype: str
        bit packing type

    unipolar: bool
        if binarization style is in unipolar 1/0 format, instead of bipolar -1/+1 format

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    assert isinstance(stride, int) or len(stride) == 2
    Input_q = bitpack(data, activation_bits, pack_axis=3, bit_axis=4, pack_type=pack_dtype)
    if len(kernel.shape) == 4:
        Filter_q = bitpack(kernel, weight_bits, pack_axis=2, bit_axis=4, pack_type=pack_dtype)
        kernel_h, kernel_w, _, num_filter, _ = get_const_tuple(Filter_q.shape)
    else:
        Filter_q = kernel
        kernel_h, kernel_w, _, _, num_filter = get_const_tuple(Filter_q.shape)
    batch, in_height, in_width, in_channel_q, _ = get_const_tuple(Input_q.shape)

    if isinstance(padding, int) or (isinstance(padding, (tuple, list)) and len(padding) == 2):
        TPAD, LPAD, DPAD, RPAD = get_pad_tuple(padding, kernel)
    else:
        TPAD, LPAD, DPAD, RPAD = padding
    pad_before = [0, TPAD, LPAD, 0, 0]
    pad_after = [0, DPAD, RPAD, 0, 0]

    # compute the output shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    out_channel = num_filter
    out_height = (in_height - kernel_h + TPAD + DPAD) // stride_h + 1
    out_width = (in_width - kernel_w + LPAD + RPAD) // stride_w + 1
    PadInput_q = pad(Input_q, pad_before, pad_after, name="PaddedInput")

    rc = tvm.reduce_axis((0, in_channel_q), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')
    b1 = tvm.reduce_axis((0, activation_bits), name='b1')
    b2 = tvm.reduce_axis((0, weight_bits), name='b2')

    if unipolar:
        def _conv(nn, yy, xx, ff):
            b1b2 = (b1+b2).astype(out_dtype)
            return tvm.sum(
                ((tvm.popcount(PadInput_q[nn, yy * stride_h + ry, xx * stride_w + rx, rc, b1] &
                               Filter_q[ry, rx, rc, ff, b2]) -
                  tvm.popcount(PadInput_q[nn, yy * stride_h + ry, xx * stride_w + rx, rc, b1] &
                               ~Filter_q[ry, rx, rc, ff, b2]))
                 << b1b2).astype(out_dtype),
                axis=[rc, ry, rx, b2, b1])

    else:
        def _conv(nn, yy, xx, ff):
            b1b2 = (b1+b2).astype(out_dtype)
            return tvm.sum((tvm.popcount(
                PadInput_q[nn, yy * stride_h + ry, xx * stride_w + rx, rc, b1] &
                Filter_q[ry, rx, rc, ff, b2]) << b1b2).astype(out_dtype),
                           axis=[rc, ry, rx, b2, b1])

    conv = tvm.compute((batch, out_height, out_width, out_channel), _conv,
                       name="Conv2dOutput", tag="bitserial_conv2d_nhwc")

    return conv

@autotvm.register_topi_compute(bitserial_conv2d_nchw, ['cpu', 'arm_cpu'], 'direct')
def spatial_pack_nchw(cfg, data, kernel, stride, padding, in_bits, weight_bits,
                      pack_dtype='uint32', out_dtype='int16', unipolar=True):
    """ Compute convolution with pack on spatial axes. """
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    data_q = bitpack(data, in_bits, pack_axis=1, bit_axis=0, pack_type=pack_dtype)
    # Check if kernel is already bitpacked
    if len(kernel.shape) == 4:
        kernel_q = bitpack(kernel, weight_bits, pack_axis=1, bit_axis=0, pack_type=pack_dtype)
        KB, CO, _, KH, KW = get_const_tuple(kernel_q.shape)
    else:
        kernel_vec = kernel
        OCO, _, KH, KW, KB, VC = get_const_tuple(kernel_vec.shape)
        CO = OCO * VC

    IB, N, CI, H, W = get_const_tuple(data_q.shape)
    KB, CO, _, KH, KW = get_const_tuple(kernel_q.shape)

    if isinstance(padding, int) or (isinstance(padding, (tuple, list)) and len(padding) == 2):
        TPAD, LPAD, DPAD, RPAD = get_pad_tuple(padding, kernel)
    else:
        TPAD, LPAD, DPAD, RPAD = padding
    pad_before = [0, 0, 0, TPAD, LPAD]
    pad_after = [0, 0, 0, DPAD, RPAD]

    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    HCAT, WCAT = KH-1, KW-1

    TH = H + TPAD + DPAD
    TW = W + LPAD + RPAD
    OH = (H + TPAD + DPAD - KH) // HSTR + 1
    OW = (W + LPAD + RPAD - KW) // WSTR + 1

     # ==================== define configuration space ====================
    n, co, oh, ow = cfg.axis(N), cfg.axis(CO), cfg.axis(OH), cfg.axis(OW)
    ci, kh, kw = cfg.reduce_axis(CI), cfg.reduce_axis(KH), cfg.reduce_axis(KW)
    ib, kb = cfg.reduce_axis(in_bits), cfg.reduce_axis(weight_bits)

    co, vc = cfg.define_split('tile_co', co, policy='all', num_outputs=2,
                              filter=lambda x: max(x.size[1:]) <= 16)
    oh, vh = cfg.define_split('tile_oh', oh, policy='all', num_outputs=2,
                              filter=lambda x: max(x.size[1:]) <= 16)
    ow, vw = cfg.define_split('tile_ow', ow, policy='all', num_outputs=2,
                              filter=lambda x: max(x.size[1:]) <= 16)
    cfg.define_annotate('ann_reduce', [ib, kb, kh, kw], policy='try_unroll')

    cfg.define_reorder("reorder_0",
                       [n, co, oh, ow, vc, vh, vw, kh, kw, kb, ib, ci],
                       policy='interval_all', interval=(6, 11))
    # binary ops
    cfg.add_flop(2 * N * OH * OW * CO * CI * KH * KW * binary_op_multiplier(pack_dtype))
    # ====================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    dvshape = (1, TH//(VH*HSTR), TW//(VW*WSTR), CI, VH*HSTR+HCAT, VW*WSTR+WCAT, IB)
    kvshape = (CO//VC, CI, KH, KW, KB, VC)
    ovshape = (1, CO//VC, OH//VH, OW//VW, VH, VW, VC)
    oshape = (1, CO, OH, OW)

    if (TPAD != 0 and RPAD != 0):
        data_pad = pad(data_q, pad_before, pad_after, name="data_pad")
    else:
        data_pad = data_q

    data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw, b: \
        data_pad[b][n][ci][h*VH*HSTR+vh][w*VW*WSTR+vw], name='data_vec')

    if len(kernel.shape) == 4:
        kernel_vec = tvm.compute(kvshape, lambda co, ci, dh, dw, b, vc: \
            kernel_q[b][co*VC+vc][ci][dh][dw], name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    dh = tvm.reduce_axis((0, KH), name='dh')
    dw = tvm.reduce_axis((0, KW), name='dw')
    b1 = tvm.reduce_axis((0, IB), name='ib')
    b2 = tvm.reduce_axis((0, KB), name='kb')

    def _conv(n, co, h, w, vh, vw, vc):
        b1b2 = (b1+b2).astype(out_dtype)
        if unipolar:
            return tvm.sum((tvm.popcount(
                data_vec[n, h, w, ci, vh*HSTR+dh, vw*WSTR+dw, b1].astype(out_dtype) &
                kernel_vec[co, ci, dh, dw, b2, vc].astype(out_dtype))  -
                            tvm.popcount(
                                data_vec[n, h, w, ci, vh*HSTR+dh, vw*WSTR+dw, b1].astype(out_dtype)
                                & ~kernel_vec[co, ci, dh, dw, b2, vc]).astype(out_dtype)) << b1b2,
                           axis=[ci, dh, dw, b1, b2])

        return tvm.sum((tvm.popcount(
            data_vec[n, h, w, ci, vh*HSTR+dh, vw*WSTR+dw, b1] &
            kernel_vec[co, ci, dh, dw, b2, vc])).astype(out_dtype) << b1b2,
                       axis=[ci, dh, dw, b1, b2])

    conv = tvm.compute(ovshape, _conv, name='conv_out')

    return tvm.compute(oshape, lambda n, co, h, w:
                       conv[n][co//VC][h//VH][w//VW][h%VH][w%VW][co%VC],
                       name='conv_vec', tag='spatial_bitserial_conv_nchw')

@autotvm.register_topi_compute(bitserial_conv2d_nhwc, 'cpu', 'direct')
def spatial_pack_nhwc(cfg, data, kernel, stride, padding, in_bits, weight_bits,
                      pack_dtype='uint32', out_dtype='int16', unipolar=True):
    """ Compute convolution with pack on spatial axes. """
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    data_q = bitpack(data, in_bits, pack_axis=3, bit_axis=4, pack_type=pack_dtype)
    pack_kernel = len(kernel.shape) == 4

    if pack_kernel:
        kernel_q = bitpack(kernel, weight_bits, pack_axis=2, bit_axis=4, pack_type=pack_dtype)
    else:
        kernel_q = kernel

    KH, KW, _, CO, KB = get_const_tuple(kernel_q.shape)
    N, H, W, CI, IB = get_const_tuple(data_q.shape)

    if isinstance(padding, int) or (isinstance(padding, (tuple, list)) and len(padding) == 2):
        TPAD, LPAD, DPAD, RPAD = get_pad_tuple(padding, kernel)
    else:
        TPAD, LPAD, DPAD, RPAD = padding
    pad_before = [0, TPAD, LPAD, 0, 0]
    pad_after = [0, DPAD, RPAD, 0, 0]

    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    HCAT, WCAT = KH-1, KW-1

    PAD_H = H + (TPAD + DPAD)
    PAD_W = W + (LPAD + RPAD)
    OH = (PAD_H - KH) // HSTR + 1
    OW = (PAD_W - KW) // WSTR + 1
    oshape = (1, OH, OW, CO)

    # ==================== define configuration space ====================
    n, oh, ow, co = cfg.axis(N), cfg.axis(OH), cfg.axis(OW), cfg.axis(CO)
    ci, kh, kw = cfg.reduce_axis(CI), cfg.reduce_axis(KH), cfg.reduce_axis(KW)
    ib, kb = cfg.reduce_axis(in_bits), cfg.reduce_axis(weight_bits)

    co, vc = cfg.define_split('tile_co', co, policy='all', num_outputs=2,
                              filter=lambda x: max(x.size[1:]) <= 16)
    oh, vh = cfg.define_split('tile_oh', oh, policy='all', num_outputs=2,
                              filter=lambda x: max(x.size[1:]) <= 16)
    ow, vw = cfg.define_split('tile_ow', ow, policy='all', num_outputs=2,
                              filter=lambda x: max(x.size[1:]) <= 16)
    cfg.define_annotate('ann_reduce', [ib, kb, kh, kw], policy='try_unroll')
    cfg.define_reorder("reorder_0",
                       [n, oh, ow, co, vh, vw, kh, kw, kb, ib, vc, ci],
                       policy='interval_all', interval=(3, 7))
    # binary ops
    cfg.add_flop(2 * N * OH * OW * CO * CI * KH * KW * binary_op_multiplier(pack_dtype))
    # ====================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    dvshape = (1, PAD_H//(VH*HSTR), PAD_W//(VW*WSTR), VH*HSTR+HCAT, VW*WSTR+WCAT, CI, IB)
    kvshape = (CO, KH, KW, CI, VC, KB)
    ovshape = (1, OH, OW, CO, VH, VW, VC)
    oshape = (1, OH, OW, CO)

    if (DPAD != 0 and RPAD != 0):
        data_pad = pad(data_q, pad_before, pad_after, name="data_pad")
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
        if unipolar:
            return tvm.sum(
                ((tvm.popcount(data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ci, b1] &
                               kernel_vec[co, dh, dw, ci, vc, b2]).astype(out_dtype) -
                  tvm.popcount(data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ci, b1]&
                               ~kernel_vec[co, dh, dw, ci, vc, b2]).astype(out_dtype)) << b1b2),
                axis=[dh, dw, ci, b1, b2])

        return tvm.sum(tvm.popcount(
            data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ci, b1] &
            kernel_vec[co, dh, dw, ci, vc, b2]).astype(out_dtype) << b1b2,
                       axis=[dh, dw, ci, b1, b2])

    conv = tvm.compute(ovshape, _conv, name='conv')

    return tvm.compute(oshape, lambda n, h, w, co:
                       conv[n][h//VH][w//VW][co//VC][h%VH][w%VW][co%VC],
                       name='output_unpack', tag='spatial_bitserial_conv_nhwc')
