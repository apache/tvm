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
# pylint: disable=invalid-name,unused-variable,invalid-name
"""Bitserial conv2d schedule on x86"""
import tvm
from tvm import te
from tvm import autotvm
from .. import tag
from ..utils import get_const_int, get_const_tuple
from ..nn.pad import pad
from ..nn.utils import get_pad_tuple
from ..nn.bitserial_util import bitpack, binary_op_multiplier


@autotvm.register_topi_compute("bitserial_conv2d_nchw.x86")
def bitserial_conv2d_nchw(
    cfg,
    data,
    kernel,
    stride,
    padding,
    in_bits,
    weight_bits,
    pack_dtype="uint32",
    out_dtype="int16",
    unipolar=True,
):
    """Compute convolution with pack on spatial axes."""
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
    HCAT, WCAT = KH - 1, KW - 1

    TH = H + TPAD + DPAD
    TW = W + LPAD + RPAD
    OH = (H + TPAD + DPAD - KH) // HSTR + 1
    OW = (W + LPAD + RPAD - KW) // WSTR + 1

    # ==================== define configuration space ====================
    n, co, oh, ow = cfg.axis(N), cfg.axis(CO), cfg.axis(OH), cfg.axis(OW)
    ci, kh, kw = cfg.reduce_axis(CI), cfg.reduce_axis(KH), cfg.reduce_axis(KW)
    ib, kb = cfg.reduce_axis(in_bits), cfg.reduce_axis(weight_bits)

    co, vc = cfg.define_split("tile_co", co, num_outputs=2, filter=lambda x: max(x.size[1:]) <= 16)
    oh, vh = cfg.define_split("tile_oh", oh, num_outputs=2, filter=lambda x: max(x.size[1:]) <= 16)
    ow, vw = cfg.define_split("tile_ow", ow, num_outputs=2, filter=lambda x: max(x.size[1:]) <= 16)
    cfg.define_annotate("ann_reduce", [ib, kb, kh, kw], policy="try_unroll")

    cfg.define_reorder(
        "reorder_0",
        [n, co, oh, ow, vc, vh, vw, kh, kw, kb, ib, ci],
        policy="interval_all",
        interval=(6, 11),
    )
    # binary ops
    cfg.add_flop(2 * N * OH * OW * CO * CI * KH * KW * binary_op_multiplier(pack_dtype))
    # ====================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    dvshape = (1, TH // (VH * HSTR), TW // (VW * WSTR), CI, VH * HSTR + HCAT, VW * WSTR + WCAT, IB)
    kvshape = (CO // VC, CI, KH, KW, KB, VC)
    ovshape = (1, CO // VC, OH // VH, OW // VW, VH, VW, VC)
    oshape = (1, CO, OH, OW)

    if TPAD != 0 and RPAD != 0:
        data_pad = pad(data_q, pad_before, pad_after, name="data_pad")
    else:
        data_pad = data_q

    data_vec = te.compute(
        dvshape,
        lambda n, h, w, ci, vh, vw, b: data_pad[b][n][ci][h * VH * HSTR + vh][w * VW * WSTR + vw],
        name="data_vec",
    )

    if len(kernel.shape) == 4:
        kernel_vec = te.compute(
            kvshape,
            lambda co, ci, dh, dw, b, vc: kernel_q[b][co * VC + vc][ci][dh][dw],
            name="kernel_vec",
        )

    ci = te.reduce_axis((0, CI), name="ci")
    dh = te.reduce_axis((0, KH), name="dh")
    dw = te.reduce_axis((0, KW), name="dw")
    b1 = te.reduce_axis((0, IB), name="ib")
    b2 = te.reduce_axis((0, KB), name="kb")

    def _conv(n, co, h, w, vh, vw, vc):
        b1b2 = (b1 + b2).astype(out_dtype)
        if unipolar:
            return te.sum(
                (
                    tvm.tir.popcount(
                        data_vec[n, h, w, ci, vh * HSTR + dh, vw * WSTR + dw, b1].astype(out_dtype)
                        & kernel_vec[co, ci, dh, dw, b2, vc].astype(out_dtype)
                    )
                    - tvm.tir.popcount(
                        data_vec[n, h, w, ci, vh * HSTR + dh, vw * WSTR + dw, b1].astype(out_dtype)
                        & ~kernel_vec[co, ci, dh, dw, b2, vc]
                    ).astype(out_dtype)
                )
                << b1b2,
                axis=[ci, dh, dw, b1, b2],
            )

        return te.sum(
            (
                tvm.tir.popcount(
                    data_vec[n, h, w, ci, vh * HSTR + dh, vw * WSTR + dw, b1]
                    & kernel_vec[co, ci, dh, dw, b2, vc]
                )
            ).astype(out_dtype)
            << b1b2,
            axis=[ci, dh, dw, b1, b2],
        )

    conv = te.compute(ovshape, _conv, name="conv_out")
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    return te.compute(
        oshape,
        lambda n, co, h, w: conv[
            n, idxd(co, VC), idxd(h, VH), idxd(w, VW), idxm(h, VH), idxm(w, VW), idxm(co, VC)
        ],
        name="conv_vec",
        tag="spatial_bitserial_conv_nchw",
    )


@autotvm.register_topi_compute("bitserial_conv2d_nhwc.x86")
def bitserial_conv2d_nhwc(
    cfg,
    data,
    kernel,
    stride,
    padding,
    in_bits,
    weight_bits,
    pack_dtype="uint32",
    out_dtype="int16",
    unipolar=True,
):
    """Compute convolution with pack on spatial axes."""
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
    HCAT, WCAT = KH - 1, KW - 1

    PAD_H = H + (TPAD + DPAD)
    PAD_W = W + (LPAD + RPAD)
    OH = (PAD_H - KH) // HSTR + 1
    OW = (PAD_W - KW) // WSTR + 1
    oshape = (1, OH, OW, CO)

    # ==================== define configuration space ====================
    n, oh, ow, co = cfg.axis(N), cfg.axis(OH), cfg.axis(OW), cfg.axis(CO)
    ci, kh, kw = cfg.reduce_axis(CI), cfg.reduce_axis(KH), cfg.reduce_axis(KW)
    ib, kb = cfg.reduce_axis(in_bits), cfg.reduce_axis(weight_bits)

    co, vc = cfg.define_split("tile_co", co, num_outputs=2, filter=lambda x: max(x.size[1:]) <= 16)
    oh, vh = cfg.define_split("tile_oh", oh, num_outputs=2, filter=lambda x: max(x.size[1:]) <= 16)
    ow, vw = cfg.define_split("tile_ow", ow, num_outputs=2, filter=lambda x: max(x.size[1:]) <= 16)
    cfg.define_annotate("ann_reduce", [ib, kb, kh, kw], policy="try_unroll")
    cfg.define_reorder(
        "reorder_0",
        [n, oh, ow, co, vh, vw, kh, kw, kb, ib, vc, ci],
        policy="interval_all",
        interval=(3, 7),
    )
    # binary ops
    cfg.add_flop(2 * N * OH * OW * CO * CI * KH * KW * binary_op_multiplier(pack_dtype))
    # ====================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    dvshape = (
        1,
        PAD_H // (VH * HSTR),
        PAD_W // (VW * WSTR),
        VH * HSTR + HCAT,
        VW * WSTR + WCAT,
        CI,
        IB,
    )
    kvshape = (CO, KH, KW, CI, VC, KB)
    ovshape = (1, OH, OW, CO, VH, VW, VC)
    oshape = (1, OH, OW, CO)

    if DPAD != 0 and RPAD != 0:
        data_pad = pad(data_q, pad_before, pad_after, name="data_pad")
    else:
        data_pad = data_q

    data_vec = te.compute(
        dvshape,
        lambda n, h, w, vh, vw, ci, b: data_pad[n][h * VH * HSTR + vh][w * VW * WSTR + vw][ci][b],
        name="data_vec",
    )

    kernel_vec = te.compute(
        kvshape,
        lambda co, dh, dw, ci, vc, b: kernel_q[dh][dw][ci][co * VC + vc][b],
        name="kernel_vec",
    )

    ci = te.reduce_axis((0, CI), name="ci")
    dh = te.reduce_axis((0, KH), name="dh")
    dw = te.reduce_axis((0, KW), name="dw")
    b1 = te.reduce_axis((0, IB), name="ib")
    b2 = te.reduce_axis((0, KB), name="kb")

    def _conv(n, h, w, co, vh, vw, vc):
        b1b2 = (b1 + b2).astype(out_dtype)
        if unipolar:
            return te.sum(
                (
                    (
                        tvm.tir.popcount(
                            data_vec[n, h, w, vh * HSTR + dh, vw * WSTR + dw, ci, b1]
                            & kernel_vec[co, dh, dw, ci, vc, b2]
                        ).astype(out_dtype)
                        - tvm.tir.popcount(
                            data_vec[n, h, w, vh * HSTR + dh, vw * WSTR + dw, ci, b1]
                            & ~kernel_vec[co, dh, dw, ci, vc, b2]
                        ).astype(out_dtype)
                    )
                    << b1b2
                ),
                axis=[dh, dw, ci, b1, b2],
            )

        return te.sum(
            tvm.tir.popcount(
                data_vec[n, h, w, vh * HSTR + dh, vw * WSTR + dw, ci, b1]
                & kernel_vec[co, dh, dw, ci, vc, b2]
            ).astype(out_dtype)
            << b1b2,
            axis=[dh, dw, ci, b1, b2],
        )

    conv = te.compute(ovshape, _conv, name="conv")

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod
    return te.compute(
        oshape,
        lambda n, h, w, co: conv[
            n, idxd(h, VH), idxd(w, VW), idxd(co, VC), idxm(h, VH), idxm(w, VW), idxm(co, VC)
        ],
        name="output_unpack",
        tag="spatial_bitserial_conv_nhwc",
    )


@autotvm.register_topi_schedule("bitserial_conv2d_nchw.x86")
def schedule_bitserial_conv2d_nchw(cfg, outs):
    return _schedule_bitserial_conv2d(cfg, outs)


@autotvm.register_topi_schedule("bitserial_conv2d_nhwc.x86")
def schedule_bitserial_conv2d_nhwc(cfg, outs):
    return _schedule_bitserial_conv2d(cfg, outs)


def _schedule_bitserial_conv2d(cfg, outs):
    """CPU schedule for bitserial convolutions NCHW and NHWC"""
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        output = op.output(0)
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag) or "elemwise" in op.tag:
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and (tensor.op not in scheduled_ops):
                    if isinstance(tensor.op, tvm.te.ComputeOp):
                        traverse(tensor.op)

        elif "spatial_bitserial_conv_nchw" in op.tag or "spatial_bitserial_conv_nhwc" in op.tag:
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[1]
            kernel_q = kernel_vec.op.input_tensors[0]
            data_vec = conv_out.op.input_tensors[0]
            data_q = data_vec.op.input_tensors[0]
            data = data_q.op.input_tensors[0]
            data_pad = None
            if isinstance(data_q.op, tvm.te.ComputeOp) and "pad" in data_q.op.tag:
                data_pad = data_q
                data_q = data
                data = data_q.op.input_tensors[0]

            if "QuantizeInput" in data.op.name:
                # Need to go up 1 further, from the combine in bitpack
                data = data.op.input_tensors[0]

            if "spatial_bitserial_conv_nchw" in op.tag:
                _schedule_bitserial_conv2d_nchw(
                    cfg,
                    s,
                    data_q,
                    data_pad,
                    data_vec,
                    kernel_q,
                    kernel_vec,
                    conv_out,
                    output,
                    outs[0],
                )
            elif "spatial_bitserial_conv_nhwc" in op.tag:
                _schedule_bitserial_conv2d_nhwc(
                    cfg,
                    s,
                    data_q,
                    data_pad,
                    data_vec,
                    kernel_q,
                    kernel_vec,
                    conv_out,
                    output,
                    outs[0],
                )
        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s


def _schedule_bitserial_conv2d_nchw(
    cfg, s, data_q, data_pad, data_vec, kernel_q, kernel_vec, conv_out, output, last
):
    IB, _, CI, IH, IW = data_q.shape
    KB, CO, _, KH, KW = kernel_q.shape
    _, _, OH, OW = output.shape

    # Infer padding and stride
    if data_pad is None:
        padding = (0, 0)
        TH, TW = IH, IW
    else:
        _, _, _, TH, TW = data_pad.shape
        hpad = get_const_int((TH - IH) // 2)
        wpad = get_const_int((TW - IW) // 2)
        padding = (hpad, wpad)

    hstride = get_const_int((TH - KH) // (OH - 1))
    wstride = get_const_int((TW - KW) // (OW - 1))
    stride = (hstride, wstride)

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    ##### Schedule Data padding, and bitpacking
    if data_pad is not None:
        s[data_pad].compute_inline()

    _, _, h, _, _, _, _ = s[data_vec].op.axis
    cfg.define_split("tile_ah", cfg.axis(h), num_outputs=2, max_factor=32)
    oh, ih = cfg["tile_ah"].apply(s, data_vec, h)
    if cfg["tile_ah"].size[1] == 1:
        oaxis = oh
        paxis = oh
    else:
        oaxis = oh
        paxis = ih

    s[data_vec].parallel(paxis)
    s[data_vec].pragma(oaxis, "parallel_launch_point")
    s[data_vec].pragma(paxis, "parallel_stride_pattern")
    s[data_vec].pragma(oaxis, "parallel_barrier_when_finish")

    ##### Schedule Kenerl bitpacking
    co, _, _, _, _, _ = s[kernel_vec].op.axis
    cfg.define_split("tile_bco", cfg.axis(co), num_outputs=2, max_factor=32)
    oco, ico = cfg["tile_bco"].apply(s, kernel_vec, co)
    if cfg["tile_bco"].size[1] == 1:
        oaxis = oco
        paxis = oco
    else:
        oaxis = oco
        paxis = ico

    s[kernel_vec].parallel(paxis)
    s[kernel_vec].pragma(oaxis, "parallel_launch_point")
    s[kernel_vec].pragma(paxis, "parallel_stride_pattern")
    s[kernel_vec].pragma(oaxis, "parallel_barrier_when_finish")

    ##### Schedule Convolution
    n, co, oh, ow, vh, vw, vc = s[conv_out].op.axis
    ci, dh, dw, ib, kb = s[conv_out].op.reduce_axis

    # s[conv_out].reorder(n, oh, ow, co, vh, vw, dh, dw, ci, vc, b1, b2)
    cfg["reorder_0"].apply(s, conv_out, [n, co, oh, ow, vc, vh, vw, dh, dw, kb, ib, ci])
    cfg["ann_reduce"].apply(
        s,
        conv_out,
        [kb, ib, dh, dw],
        axis_lens=[
            get_const_int(kb.dom.extent),
            get_const_int(ib.dom.extent),
            get_const_int(dh.dom.extent),
            get_const_int(dw.dom.extent),
        ],
        max_unroll=16,
        cfg=cfg,
    )

    s[conv_out].vectorize(vc)

    # # Schedule output
    n, co, h, w = s[last].op.axis
    co, vc = s[last].split(co, VC)
    oh, ow, vh, vw = s[last].tile(h, w, VH, VW)
    s[last].reorder(n, co, oh, ow, vh, vw, vc)
    if last != output:
        s[output].compute_inline()
    s[conv_out].compute_at(s[last], ow)

    oco, ico = cfg["tile_oh"].apply(s, last, co)
    if cfg["tile_oh"].size[1] == 1:
        oaxis = oco
        paxis = oco
    else:
        oco, ico = s[last].split(co, bc)
        oaxis = oco
        paxis = ico

    s[last].parallel(oco)
    return s


def _schedule_bitserial_conv2d_nhwc(
    cfg, s, data_q, data_pad, data_vec, kernel_q, kernel_vec, conv_out, output, last
):
    # no stride and padding info here
    _, IH, IW, CI, IB = data_q.shape
    KH, KW, _, CO, KB = kernel_q.shape
    _, OH, OW, _ = output.shape

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    ##### Schedule data padding and packing
    if data_pad is not None:
        s[data_pad].compute_inline()

    _, h, _, _, _, _, _ = s[data_vec].op.axis
    cfg.define_split("tile_ah", cfg.axis(h), num_outputs=2, max_factor=32)
    oh, ih = cfg["tile_ah"].apply(s, data_vec, h)
    s[data_vec].parallel(oh)

    ##### Schedule kernel packing
    co, _, _, _, _, _ = s[kernel_vec].op.axis
    cfg.define_split("tile_bco", cfg.axis(co), num_outputs=2, max_factor=32)
    oco, ico = cfg["tile_bco"].apply(s, kernel_vec, co)
    s[kernel_vec].parallel(oco)

    ##### Schedule Convolution
    n, oh, ow, co, vh, vw, vc = s[conv_out].op.axis
    dh, dw, ci, b1, b2 = s[conv_out].op.reduce_axis

    # s[conv_out].reorder(n, oh, ow, co, vh, vw, dh, dw, ci, vc, b1, b2)
    cfg["reorder_0"].apply(s, conv_out, [n, oh, ow, co, vh, vw, dh, dw, ci, vc, b1, b2])
    cfg["ann_reduce"].apply(
        s,
        conv_out,
        [b1, b2, dh, dw],
        axis_lens=[
            get_const_int(b1.dom.extent),
            get_const_int(b2.dom.extent),
            get_const_int(dh.dom.extent),
            get_const_int(dw.dom.extent),
        ],
        max_unroll=16,
        cfg=cfg,
    )

    s[conv_out].unroll(b1)
    s[conv_out].unroll(b2)
    s[conv_out].vectorize(vc)

    # # Schedule output
    n, h, w, co = s[last].op.axis
    co, vc = s[last].split(co, VC)
    oh, ow, vh, vw = s[last].tile(h, w, VH, VW)
    s[last].reorder(n, oh, ow, co, vh, vw, vc)
    s[last].vectorize(vc)
    if last != output:
        s[output].compute_inline()
    s[conv_out].compute_at(s[last], ow)

    oho, iho = cfg["tile_oh"].apply(s, last, oh)  # reuse parameter
    s[last].parallel(oho)

    return s
