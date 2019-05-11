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
from tvm import autotvm
from topi.util import get_const_int
from .. import generic, tag

@autotvm.register_topi_schedule(generic.nn.schedule_bitserial_conv2d_nchw, ['cpu'], 'direct')
@autotvm.register_topi_schedule(generic.nn.schedule_bitserial_conv2d_nhwc, ['cpu'], 'direct')
def schedule_bitserial_conv2d(cfg, outs):
    """CPU schedule for bitserial convolutions NCHW and NHWC"""
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        output = op.output(0)
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag) or 'elemwise' in op.tag:
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors and tensor.op not in scheduled_ops:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        elif 'spatial_bitserial_conv_nchw' in op.tag or 'spatial_bitserial_conv_nhwc' in op.tag:
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[1]
            kernel_q = kernel_vec.op.input_tensors[0]
            data_vec = conv_out.op.input_tensors[0]
            data_q = data_vec.op.input_tensors[0]
            data = data_q.op.input_tensors[0]
            data_pad = None
            if isinstance(data_q.op, tvm.tensor.ComputeOp) and "pad" in data_q.op.tag:
                data_pad = data_q
                data_q = data
                data = data_q.op.input_tensors[0]

            if "QuantizeInput" in data.op.name:
                # Need to go up 1 further, from the combine in bitpack
                data = data.op.input_tensors[0]

            if 'spatial_bitserial_conv_nchw' in op.tag:
                _schedule_bitserial_conv2d_nchw(cfg, s, data_q, data_pad, data_vec,
                                                kernel_q, kernel_vec,
                                                conv_out, output, outs[0])
            elif 'spatial_bitserial_conv_nhwc' in op.tag:
                _schedule_bitserial_conv2d_nhwc(cfg, s, data_q, data_pad, data_vec,
                                                kernel_q, kernel_vec,
                                                conv_out, output, outs[0])
        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

def _schedule_bitserial_conv2d_nchw(cfg, s, data_q, data_pad, data_vec,
                                    kernel_q, kernel_vec,
                                    conv_out, output, last):
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
    cfg.define_split("tile_ah", cfg.axis(h), policy="all", num_outputs=2, max_factor=32)
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
    cfg.define_split("tile_bco", cfg.axis(co), policy="all", num_outputs=2, max_factor=32)
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
    cfg["ann_reduce"].apply(s, conv_out, [kb, ib, dh, dw],
                            axis_lens=[get_const_int(kb.dom.extent),
                                       get_const_int(ib.dom.extent),
                                       get_const_int(dh.dom.extent),
                                       get_const_int(dw.dom.extent)],
                            max_unroll=16,
                            cfg=cfg)

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

def _schedule_bitserial_conv2d_nhwc(cfg, s, data_q, data_pad, data_vec,
                                    kernel_q, kernel_vec,
                                    conv_out, output, last):
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
    cfg.define_split("tile_ah", cfg.axis(h), policy="all", num_outputs=2, max_factor=32)
    oh, ih = cfg["tile_ah"].apply(s, data_vec, h)
    s[data_vec].parallel(oh)

    ##### Schedule kernel packing
    co, _, _, _, _, _ = s[kernel_vec].op.axis
    cfg.define_split("tile_bco", cfg.axis(co), policy="all", num_outputs=2, max_factor=32)
    oco, ico = cfg["tile_bco"].apply(s, kernel_vec, co)
    s[kernel_vec].parallel(oco)

    ##### Schedule Convolution
    n, oh, ow, co, vh, vw, vc = s[conv_out].op.axis
    dh, dw, ci, b1, b2 = s[conv_out].op.reduce_axis

    # s[conv_out].reorder(n, oh, ow, co, vh, vw, dh, dw, ci, vc, b1, b2)
    cfg["reorder_0"].apply(s, conv_out, [n, oh, ow, co, vh, vw, dh, dw, ci, vc, b1, b2])
    cfg["ann_reduce"].apply(s, conv_out, [b1, b2, dh, dw],
                            axis_lens=[get_const_int(b1.dom.extent),
                                       get_const_int(b2.dom.extent),
                                       get_const_int(dh.dom.extent),
                                       get_const_int(dw.dom.extent)],
                            max_unroll=16,
                            cfg=cfg)

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
