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
"""Conv2D_transpose operator declaration and schedule registration for VTA."""

import numpy as np

import tvm
from tvm import autotvm
import topi
from topi.util import get_const_tuple
from topi.nn.util import get_pad_tuple

from ..environment import get_env

@autotvm.register_topi_compute(topi.nn.conv2d_transpose_nchw, 'vta', 'direct')
def _declatation_conv2d_transpose(cfg,
                                  data,
                                  kernel,
                                  strides,
                                  padding,
                                  out_dtype):
    env = get_env()

    batch, in_c, in_h, in_w, B_BATCH, B_CI = get_const_tuple(data.shape)
    out_c, _, filter_h, filter_w, B_CO, B_CI = get_const_tuple(kernel.shape)
    stride_h, stride_w = strides

    # padding stage
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(padding, (filter_h, filter_w))
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right

    # padding stage
    FirstPad = topi.nn.pad(data,
                           [0, 0, (bpad_top + stride_h - 1) // stride_h,
                            (bpad_left + stride_w - 1) // stride_w, 0, 0],
                           [0, 0, (bpad_bottom + stride_h - 1) // stride_h,
                            (bpad_right + stride_w - 1) // stride_w, 0, 0],
                           name='pad_data')
    border_h = (stride_h - bpad_top % stride_h) % stride_h  # remove extra padding introduced by dilation
    border_w = (stride_w - bpad_left % stride_w) % stride_w

    # dilation stage
    data = FirstPad
    strides = [1, 1, stride_h, stride_w, 1, 1]
    n = len(data.shape)

    def _dilate(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if not topi.util.equal_const_int(strides[i], 1):
                index_tuple.append(indices[i] // strides[i])
                not_zero.append((indices[i] % strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = tvm.all(*not_zero)
            return tvm.expr.Select(not_zero, data(*index_tuple), tvm.const(0.0, data.dtype))
        return data(*index_tuple)

    # convolution stage
    out_h = (in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (in_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    dc = tvm.reduce_axis((0, in_c), name='dc')
    dh = tvm.reduce_axis((0, filter_h), name='dh')
    dw = tvm.reduce_axis((0, filter_w), name='dw')
    dci = tvm.reduce_axis((0, B_CI), name='dci')

    out = tvm.compute(
        (batch, out_c, out_h, out_w, B_BATCH, B_CO),
        lambda b, c, h, w, b_n, b_co: tvm.sum(
            _dilate(b, dc, h + dh + border_h, w + dw + border_w, b_n, dci).astype(out_dtype) *
            kernel[c, dc, dh, dw, b_co, dci].astype(out_dtype),
            axis=[dc, dh, dw, dci]),
        tag="packed_conv2d_transpose",
        name='res',
        attrs={"workload": (batch * env.BATCH, in_h, in_w, in_c * env.BLOCK_IN, out_c * env.BLOCK_OUT,
                            filter_h, filter_w, padding[0], padding[1], stride_h, stride_w)})

    return out

@autotvm.register_topi_schedule(topi.generic.schedule_conv2d_transpose_nchw, 'vta', 'direct')
def _schedule_conv2d_transpose(cfg, outs):
    assert len(outs) == 1
    output = outs[0]
    ewise_inputs = []
    ewise_ops = []
    conv2d_res = []
    assert output.dtype == "int8"
    assert output.op.input_tensors[0].dtype == "int32"

    def _traverse(op):
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            assert op.tag == "packed_conv2d_transpose"
            conv2d_res.append(op)

    _traverse(output.op)
    assert len(conv2d_res) == 1
    conv2d_stage = conv2d_res[0].output(0)
    s = tvm.create_schedule(output.op)

    ##### space definition begin #####
    b, co, h, w, _, ci = s[conv2d_stage].op.axis
    ci, _, _, _ = s[conv2d_stage].op.reduce_axis
    cfg.define_split('tile_b', b, num_outputs=2)
    cfg.define_split('tile_h', h, num_outputs=2)
    cfg.define_split('tile_w', w, num_outputs=2)
    cfg.define_split('tile_ci', ci, num_outputs=2)
    cfg.define_split('tile_co', co, num_outputs=2)
    cfg.define_knob('oc_nthread', [1, 2])
    cfg.define_knob('h_nthread', [1, 2])
    ###### space definition end ######

    data, kernel = conv2d_stage.op.input_tensors
    if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
        temp = data.op.input_tensors[0]
        pad_data = data
        data = temp
    else:
        pad_data = None

    env = get_env()

    # setup pad
    if pad_data is not None:
        cdata = pad_data
        s[pad_data].set_scope(env.inp_scope)
    else:
        cdata = s.cache_read(data, env.inp_scope, [conv2d_stage])
    ckernel = s.cache_read(kernel, env.wgt_scope, [conv2d_stage])
    s[conv2d_stage].set_scope(env.acc_scope)

    # cache read input
    cache_read_ewise = []
    for consumer, tensor in ewise_inputs:
        cache_read_ewise.append(
            s.cache_read(tensor, env.acc_scope, [consumer]))
    # set ewise scope
    for op in ewise_ops:
        s[op].set_scope(env.acc_scope)
        s[op].pragma(s[op].op.axis[0], env.alu)

    # tile
    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[output].op.axis
    x_co0, x_co1 = cfg['tile_co'].apply(s, output, x_co)
    x_i0, x_i1 = cfg['tile_h'].apply(s, output, x_i)
    x_j0, x_j1 = cfg['tile_w'].apply(s, output, x_j)
    s[output].reorder(x_bo, x_i0, x_co0, x_j0, x_co1, x_i1, x_j1, x_bi, x_ci)
    store_pt = x_j0

    # set all compute scopes
    s[conv2d_stage].compute_at(s[output], store_pt)
    for op in ewise_ops:
        s[op].compute_at(s[output], store_pt)

    for tensor in cache_read_ewise:
        s[tensor].compute_at(s[output], store_pt)
        s[tensor].pragma(s[tensor].op.axis[0], env.dma_copy)

    # virtual threading along output channel axes
    if cfg['oc_nthread'].val > 1:
        _, v_t = s[output].split(x_co0, factor=cfg['oc_nthread'].val)
        s[output].reorder(v_t, x_bo)
        s[output].bind(v_t, tvm.thread_axis("cthread"))

    # virtual threading along spatial rows
    if cfg['h_nthread'].val > 1:
        _, v_t = s[output].split(x_i0, factor=cfg['h_nthread'].val)
        s[output].reorder(v_t, x_bo)
        s[output].bind(v_t, tvm.thread_axis("cthread"))

    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[conv2d_stage].op.axis
    k_o, d_i, d_j, k_i = s[conv2d_stage].op.reduce_axis
    x_i, x_ii = s[conv2d_stage].split(x_i, 4)
    x_j, x_jj = s[conv2d_stage].split(x_j, 2)
    s[conv2d_stage].reorder(x_bo, k_o, x_j, x_co, x_i, x_jj, d_j, d_i, x_ii, x_bi, x_ci, k_i)

    for axis in [d_j, d_i, x_ii, x_jj]:
        s[conv2d_stage].unroll(axis)

    k_o, _ = cfg['tile_ci'].apply(s, conv2d_stage, k_o)
    s[cdata].compute_at(s[conv2d_stage], k_o)
    s[ckernel].compute_at(s[conv2d_stage], k_o)

    # Use VTA instructions
    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy)
    s[ckernel].pragma(s[ckernel].op.axis[0], env.dma_copy)
    s[conv2d_stage].pragma(x_bi, "conv2d_transpose_gemm")
    s[output].pragma(x_co1, env.dma_copy)

    return s
