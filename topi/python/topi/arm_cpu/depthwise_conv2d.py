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
# pylint: disable=invalid-name,unused-variable
"""Depthwise convolution schedule for ARM CPU"""

import tvm
from tvm import te
from tvm import autotvm

from .. import nn
from ..util import traverse_inline, get_const_tuple, get_const_int
from ..nn.util import get_pad_tuple


@autotvm.register_topi_compute("depthwise_conv2d_nchw.arm_cpu")
def depthwise_conv2d_nchw(_, data, kernel, strides, padding, dilation, out_dtype):
    """Compute depthwise_conv2d with NCHW layout"""
    return nn.depthwise_conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("depthwise_conv2d_nchw.arm_cpu")
def schedule_depthwise_conv2d_nchw(cfg, outs):
    """Schedule depthwise conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The configuration of this template
    outs: Array of Tensor
        The computation graph description of depthwise convolution2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nchw.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule(cfg, s, data, data_pad, kernel, output):
        A, B, C = data, kernel, output
        s[data_pad].compute_inline()

        ##### space definition begin #####
        n, c, h, w = s[output].op.axis
        _, vc = cfg.define_split('tile_c', c, num_outputs=2)
        _, vh = cfg.define_split('tile_h', h, num_outputs=2)
        _, vw = cfg.define_split('tile_w', w, num_outputs=2)
        cfg.define_annotate('ann', [vh, vw, vc], policy='try_unroll_vec')

        # fallback support
        if cfg.is_fallback:
            ref_log = autotvm.tophub.load_reference_log(
                'arm_cpu', 'rk3399', 'depthwise_conv2d_nchw.arm_cpu')
            cfg.fallback_with_reference_log(ref_log)
        ##### space definition end #####

        # park data to vector form  [n, c, h, w] -> [n, C, h, w, VC]
        A0 = s.cache_read(data_pad, "global", C)
        n, c, h, w = s[A0].op.axis
        c, vc = cfg['tile_c'].apply(s, A0, c)
        s[A0].reorder(n, c, h, w, vc)
        A1 = s.cache_write(A0, 'global')
        s[A0].compute_inline()

        # park kernel to vector form  [co, ci, kh, kw] -> [CO, ci, kh, kw, VC]
        B0 = s.cache_read(B, "global", C)
        c, m, h, w = s[B0].op.axis
        c, vc, = cfg['tile_c'].apply(s, B0, c)
        s[B0].reorder(c, m, h, w, vc)
        B1 = s.cache_write(B0, 'global')
        s[B0].compute_inline()

        n, c, h, w = s[C].op.axis
        c, vc, = cfg['tile_c'].apply(s, C, c)
        s[C].reorder(n, c, h, w, vc)

        # depthwise conv
        C0 = s.cache_write(C, 'global')
        _, c, h, w, vc = s[C0].op.axis
        dh, dw = s[C0].op.reduce_axis
        oh, ih = cfg['tile_h'].apply(s, C0, h)
        ow, iw = cfg['tile_w'].apply(s, C0, w)
        s[C0].reorder(c, oh, ow, dh, dw, ih, iw, vc)
        s[A1].compute_at(s[C0], oh)

        # try unroll and vectorization
        cfg['ann'].apply(s, C0, [ih, iw, vc],
                         axis_lens=[cfg['tile_h'].size[-1],
                                    cfg['tile_w'].size[-1],
                                    cfg['tile_c'].size[-1]],
                         max_unroll=16,
                         cfg=cfg)

        # fusion
        if C.op not in s.outputs:
            s[C].compute_inline()

        # mark parallel
        last = outs[0]
        n, c, h, w = s[last].op.axis
        s[last].parallel(c)

        n, c, h, w, vc = s[C0].op.axis
        s[C0].parallel(c)

        c, m, h, w, vc = s[B1].op.axis
        s[B1].parallel(c)

        return s

    def _callback(op):
        if op.tag == 'depthwise_conv2d_nchw':
            output = op.output(0)
            kernel = op.input_tensors[1]
            data = op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]
            _schedule(cfg, s, data, data_pad, kernel, output)

    traverse_inline(s, outs[0].op, _callback)
    return s


# TODO:
# This schedule has incorrect result on some hardware platforms (like NV Jetson TX2)
# Let us comment it out but not remove.
# see discussion:
# https://discuss.tvm.ai/t/autotuner-incorrect-result-after-tuning-mobilenetv2-on-arm-cpu/6088
@autotvm.register_topi_compute("depthwise_conv2d_nchw_spatial_pack.arm_cpu")
def depthwise_conv2d_nchw_spatial_pack(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """TOPI compute callback for depthwise_conv2d nchw

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.te.Tensor
        4-D with shape [num_filter, multiplier, filter_height, filter_width] or
        pre-packed 5-D with shape [num_filter_chunk, multiplier, filter_height,
        filter_width, num_filter_block]

    strides : list of two ints
        [stride_height, stride_width]

    padding : list of two ints
        [pad_height, pad_width]

    dilation : list of two ints
        [dilation_height, dilation_width]

    out_dtype: str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """

    return _decl_spatial_pack(cfg, data, kernel, strides, padding, dilation, out_dtype, num_tile=2)


@autotvm.register_topi_schedule("depthwise_conv2d_nchw_spatial_pack.arm_cpu")
def schedule_depthwise_conv2d_nchw_spatial_pack(cfg, outs):
    """Create the schedule for depthwise_conv2d_nchw_spatial_pack"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'spatial_depthwise_conv2d_nchw_output':
            output = op.output(0)
            conv = op.input_tensors[0]
            data_vec = conv.op.input_tensors[0]
            kernel_vec = conv.op.input_tensors[1]
            if kernel_vec.op.name == 'kernel_vec':
                kernel = kernel_vec.op.input_tensors[0]
            else:
                kernel = kernel_vec
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()
            _schedule_spatial_pack(cfg, s, data_vec, kernel_vec, conv, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


def _decl_spatial_pack(cfg, data, kernel, strides, padding, dilation, out_dtype, num_tile):
    out_dtype = out_dtype or data.dtype

    N, C, IH, IW = get_const_tuple(data.shape)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if len(kernel.shape) == 4:
        pre_packed = False
        C, M, KH, KW = get_const_tuple(kernel.shape)
    else:  # kernel tensor is pre packed
        pre_packed = True
        C, M, KH, KW, VC = get_const_tuple(kernel.shape)
        C = C * VC

    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    OH = (IH + pad_top + pad_down - dilated_kernel_h) // HSTR + 1
    OW = (IW + pad_left + pad_right - dilated_kernel_w) // WSTR + 1
    # pack data
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right
    DOPAD = (HPAD != 0 or WPAD != 0)
    if DOPAD:
        data_pad = nn.pad(data, (0, 0, pad_top, pad_left), (0, 0, pad_down, pad_right),
                          name="data_pad")
    else:
        data_pad = data

    # fallback support
    # Currently, Mali schedule doesn't use it like conv2d.
    if cfg.is_fallback:
        ref_log = autotvm.tophub.load_reference_log(
            'arm_cpu', 'rk3399', 'depthwise_conv2d_nchw_spatial_pack.arm_cpu')
        cfg.fallback_with_reference_log(ref_log)

    # ==================== define configuration space ====================
    n, c, oh, ow = cfg.axis(N), cfg.axis(C), cfg.axis(OH), cfg.axis(OW)
    kh, kw = cfg.reduce_axis(KH), cfg.reduce_axis(KW)

    # Currently, Mali schedule doesn't use it like conv2d.
    # Leave num_tile for possible future use of Mali schedule
    if num_tile == 2:     # for arm cpu
        co, vc = cfg.define_split('tile_co', c, num_outputs=2)
        oh, vh = cfg.define_split('tile_oh', oh, num_outputs=2)
        ow, vw = cfg.define_split('tile_ow', ow, num_outputs=2)
    else:
        raise RuntimeError("Invalid num_tile")

    cfg.define_reorder("reorder_0",
                       [n, co, oh, ow, kh, kw, vh, vw, vc],
                       policy='candidate', candidate=[
                           [n, co, oh, ow, kh, kw, vh, vw, vc],
                           [n, co, oh, ow, kh, kw, vc, vh, vw]])

    cfg.define_reorder("reorder_1",
                       [n, co, oh, ow, vh, vw, vc],
                       policy='candidate', candidate=[
                           [n, co, oh, ow, vh, vw, vc],
                           [n, co, oh, ow, vc, vh, vw],
                           [n, co, oh, ow, vh, vc, vw]])

    cfg.define_annotate("ann_reduce", [kh, kw], policy='try_unroll')
    cfg.define_annotate("ann_spatial", [vh, vw, vc], policy='try_unroll_vec')
    # ====================================================================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    kvshape = (C // VC, M, KH, KW, VC)
    ovshape = (N, C * M // VC, OH // VH, OW // VW, VH, VW, VC)
    oshape = (N, C * M, OH, OW)

    if dilation_h != 1 or dilation_w != 1:
        # undilate input data
        dvshape = (N, OH // VH, OW // VW, C, KH, KW, VH, VW)
        data_vec = te.compute(dvshape, lambda n, h, w, c, kh, kw, vh, vw:
                              data_pad[n][c][(h * VH + vh) * HSTR + kh * dilation_h]
                              [(w*VW+vw)*WSTR+kw*dilation_w],
                              name='data_vec_undilated')
    else:
        dvshape = (N, OH // VH, OW // VW, C, VH*HSTR + KH-1, VW*WSTR + KW-1)
        data_vec = te.compute(dvshape, lambda n, h, w, c, vh, vw:
                              data_pad[n][c][h * VH * HSTR + vh][w * VW * WSTR + vw],
                              name='data_vec')

    if pre_packed:
        kernel_vec = kernel
    else:
        kernel_vec = te.compute(kvshape, lambda co, m, kh, kw, vc:
                                kernel[co*VC+vc][m][kh][kw],
                                name='kernel_vec')

    kh = te.reduce_axis((0, KH), name='kh')
    kw = te.reduce_axis((0, KW), name='kw')

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    if dilation_h != 1 or dilation_w != 1:
        conv = te.compute(
            ovshape, lambda n, co, h, w, vh, vw, vc: \
            te.sum(data_vec[n, h, w, idxdiv(co * VC + vc, M), kh, kw, vh, vw]
                   .astype(out_dtype) *
                   kernel_vec[idxdiv(co, M), idxmod(co, M), kh, kw, vc].astype(out_dtype),
                   axis=[kh, kw]), name='depthwise_conv')
    else:
        conv = te.compute(ovshape, lambda n, co, h, w, vh, vw, vc: \
                          te.sum(data_vec[n, h, w, idxdiv((co * VC + vc), M), vh * HSTR + kh,
                                          vw * WSTR + kw].astype(out_dtype) *
                                 kernel_vec[idxdiv(co, M),
                                            idxmod(co, M),
                                            kh, kw, vc].astype(out_dtype),
                                 axis=[kh, kw]), name='depthwise_conv')

    output = te.compute(oshape, lambda n, co, h, w:
                        conv[n,
                             idxdiv(co, VC), idxdiv(h, VH), idxdiv(w, VW),
                             idxmod(h, VH), idxmod(w, VW), idxmod(co, VC)],
                        name='output_unpack', tag='spatial_depthwise_conv2d_nchw_output')
    return output

def _schedule_spatial_pack(cfg, s, data_vec, kernel_vec,
                           conv, output, last):
    """schedule implementation"""
    n, co, oh, ow, vh, vw, vc = s[conv].op.axis
    kh, kw = s[conv].op.reduce_axis

    if data_vec.op.name == 'data_vec_undilated':
        _, dv_oh, dv_ow, dv_c, _, _, dv_vh, dv_vw = s[data_vec].op.axis
    else:
        _, dv_oh, dv_ow, dv_c, dv_vh, dv_vw = s[data_vec].op.axis

    data_pad = data_vec.op.input_tensors[0]
    if data_pad.op.name == "data_pad":
        assert isinstance(data_pad.op, tvm.te.ComputeOp)
        has_padding = True
    else:
        assert isinstance(data_pad.op, tvm.te.PlaceholderOp)
        has_padding = False

    cfg.define_knob('data_pad_inline', [0, 1, 2, 3, 4])

    if cfg['data_pad_inline'].val == 1 and has_padding:
        s[data_pad].compute_inline()
    if cfg['data_pad_inline'].val == 2 and has_padding:
        s[data_pad].vectorize(list(s[data_pad].op.axis)[-1])
    if cfg['data_pad_inline'].val == 3 and has_padding:
        s[data_pad].vectorize(list(s[data_pad].op.axis)[-1])
        s[data_pad].compute_at(s[data_vec], dv_oh)
    if cfg['data_pad_inline'].val == 4 and has_padding:
        s[data_pad].vectorize(list(s[data_pad].op.axis)[-1])
        s[data_pad].compute_at(s[data_vec], dv_ow)

    cfg.define_knob('data_vec_inline', [0, 1, 2, 3])
    if cfg['data_vec_inline'].val == 1:
        s[data_vec].compute_at(s[conv], oh)
    if cfg['data_vec_inline'].val == 2:
        s[data_vec].compute_at(s[conv], ow)
    if cfg['data_vec_inline'].val == 3:
        s[data_vec].compute_at(s[conv], co)

    # schedule conv
    cfg["reorder_0"].apply(s, conv, [n, co, oh, ow, kh, kw, vh, vw, vc])
    cfg["ann_reduce"].apply(s, conv, [kh, kw],
                            axis_lens=[get_const_int(kh.dom.extent),
                                       get_const_int(kw.dom.extent)],
                            max_unroll=16,
                            cfg=cfg)
    cfg["ann_spatial"].apply(s, conv, [vh, vw, vc],
                             axis_lens=[cfg['tile_oh'].size[-1],
                                        cfg['tile_ow'].size[-1],
                                        cfg['tile_co'].size[-1]],
                             max_unroll=16,
                             cfg=cfg)

    # schedule fusion
    n, co, h, w = s[last].op.axis
    co, vc = cfg['tile_co'].apply(s, last, co)
    oh, vh = cfg['tile_oh'].apply(s, last, h)
    ow, vw = cfg['tile_ow'].apply(s, last, w)
    cfg["reorder_1"].apply(s, last, [n, co, oh, ow, vh, vw, vc])
    if last != output:
        s[output].compute_inline()
        cfg["ann_spatial"].apply(s, last, [vh, vw, vc],
                                 axis_lens=[cfg['tile_oh'].size[-1],
                                            cfg['tile_ow'].size[-1],
                                            cfg['tile_co'].size[-1]],
                                 max_unroll=16,
                                 cfg=cfg)
    else:
        s[last].vectorize(vw)
    cfg.define_knob('conv_inline', [0, 1, 2, 3])
    if cfg['conv_inline'].val == 1:
        s[conv].compute_at(s[last], ow)
    if cfg['conv_inline'].val == 2:
        s[conv].compute_at(s[last], oh)
    if cfg['conv_inline'].val == 3:
        s[conv].compute_at(s[last], co)

    # mark parallel
    s[last].parallel(co)

    if data_vec.op.name == 'data_vec_undilated':
        _, h, _, _, _, _, _, _ = s[data_vec].op.axis
    else:
        _, h, _, _, _, _ = s[data_vec].op.axis
    s[data_vec].parallel(h)

    if kernel_vec.op.name == 'kernel_vec':
        co, _, _, _, _ = s[kernel_vec].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel packing will be pre-computed during compliation, so we skip
            # this part to make tuning records correct
            s[kernel_vec].pragma(co, 'debug_skip_region')
        else:
            s[kernel_vec].parallel(co)

    return s
