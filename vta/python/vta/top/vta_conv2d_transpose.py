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
    ishape = get_const_tuple(data.shape)
    kshape = get_const_tuple(kernel.shape)
    b, c_i, i_h, i_w, t_b, t_ci = ishape
    c_o, _, k_h, k_w, t_co, t_ci = kshape
    stride_h, stride_w = strides

    # derive padding parameters
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(padding, (k_h, k_w))
    bpad_top = k_h - 1 - fpad_top
    bpad_bottom = k_h - 1 - fpad_bottom
    bpad_left = k_w - 1 - fpad_left
    bpad_right = k_w - 1 - fpad_right

    # padding stage
    dilated_input = topi.nn.dilate(data, [1, 1, stride_h, stride_w, 1, 1])
    data_pad = topi.nn.pad(dilated_input,
                           [0, 0, bpad_top, bpad_left, 0, 0],
                           [0, 0, bpad_bottom, bpad_right, 0, 0])

    # convolution transpose stage
    out_h = (i_h - 1) * stride_h - fpad_top - fpad_bottom + k_h
    out_w = (i_w - 1) * stride_w - fpad_left - fpad_right + k_w
    oshape = (b, c_o, out_h, out_w, t_b, t_co)
    d_c = tvm.reduce_axis((0, c_i), name='d_c')
    d_h = tvm.reduce_axis((0, k_h), name='d_h')
    d_w = tvm.reduce_axis((0, k_w), name='d_w')
    d_ci = tvm.reduce_axis((0, t_ci), name='d_ci')

    out = tvm.compute(
        oshape,
        lambda i_n, i_c, i_h, i_w, j_n, j_c: tvm.sum(
            data_pad(i_n, d_c, i_h + d_h, i_w + d_w, j_n, d_ci).astype(out_dtype) *
            kernel[i_c, d_c, d_h, d_w, j_c, d_ci].astype(out_dtype),
            axis=[d_c, d_h, d_w, d_ci]),
        tag="packed_conv2d_transpose",
        name='res')

    cfg.add_flop(2 * np.prod(topi.util.get_const_tuple(oshape)) *
                 kshape[2] * kshape[3] * ishape[1] * ishape[-1])

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
    b, c_o, x_i, x_j, _, c_i = s[conv2d_stage].op.axis
    c_i, _, _, _ = s[conv2d_stage].op.reduce_axis
    cfg.define_split('tile_b', b, num_outputs=2)
    cfg.define_split('tile_h', x_i, num_outputs=2)
    cfg.define_split('tile_w', x_j, num_outputs=2)
    cfg.define_split('tile_ci', c_i, num_outputs=2)
    cfg.define_split('tile_co', c_o, num_outputs=2)
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
