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
"""Group conv2D operator declaration and schedule registration for VTA."""

import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm import topi

from ..environment import get_env


@autotvm.register_topi_compute("group_conv2d_packed.vta")
def group_conv2d_packed(cfg, data, kernel, strides, padding, dilation, group, out_dtype):
    """Packed group conv2d nchw function."""
    assert dilation == (1, 1)

    if padding[0]:
        pad_data = topi.nn.pad(data, [0, 0, padding[0], padding[1], 0, 0], name="pad_data")
    else:
        pad_data = data
    assert len(data.shape) == 6
    assert len(kernel.shape) == 6
    assert data.dtype == "int8", data.dtype
    assert kernel.dtype == "int8", kernel.dtype
    assert out_dtype == "int32", out_dtype

    oheight = topi.utils.get_const_int((pad_data.shape[2] - kernel.shape[2]) // strides[0] + 1)
    owidth = topi.utils.get_const_int((pad_data.shape[3] - kernel.shape[3]) // strides[1] + 1)
    oshape = (data.shape[0], kernel.shape[0], oheight, owidth, data.shape[4], kernel.shape[4])

    ishape = topi.utils.get_const_tuple(data.shape)
    kshape = topi.utils.get_const_tuple(kernel.shape)
    assert group * kshape[1] == ishape[1]
    assert kshape[0] % group == 0
    d_i = te.reduce_axis((0, kshape[2]), name="d_i")
    d_j = te.reduce_axis((0, kshape[3]), name="d_j")
    k_o = te.reduce_axis((0, kshape[1]), name="k_o")
    k_i = te.reduce_axis((0, kshape[-1]), name="k_i")
    hstride, wstride = strides
    out = te.compute(
        oshape,
        lambda b_o, c_o, i, j, b_i, c_i: te.sum(
            pad_data[
                b_o,
                c_o // (kshape[0] // group) * kshape[1] + k_o,
                i * hstride + d_i,
                j * wstride + d_j,
                b_i,
                k_i,
            ].astype(out_dtype)
            * kernel[c_o, k_o, d_i, d_j, c_i, k_i].astype(out_dtype),
            axis=[k_o, d_i, d_j, k_i],
        ),
        name="res",
        tag="packed_group_conv2d",
    )

    cfg.add_flop(
        2
        * np.prod(topi.utils.get_const_tuple(oshape))
        * kshape[2]
        * kshape[3]
        * ishape[1]
        * kshape[-1]
    )

    return out


@autotvm.register_topi_schedule("group_conv2d_packed.vta")
def schedule_group_conv2d_packed(cfg, outs):
    """Schedule the packed conv2d."""
    assert len(outs) == 1
    output = outs[0]
    const_ops = []
    ewise_inputs = []
    ewise_ops = []
    conv2d_res = []
    assert output.dtype == "int8"
    assert output.op.input_tensors[0].dtype == "int32"

    def _traverse(op):
        if topi.tag.is_broadcast(op.tag):
            if not op.same_as(output.op):
                if not op.axis:
                    const_ops.append(op)
                else:
                    ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.te.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            assert op.tag == "packed_group_conv2d"
            conv2d_res.append(op)

    _traverse(output.op)
    assert len(conv2d_res) == 1
    conv2d_stage = conv2d_res[0].output(0)
    s = te.create_schedule(output.op)

    ##### space definition begin #####
    b, c_o, x_i, x_j, _, _ = s[conv2d_stage].op.axis
    c_i, _, _, _ = s[conv2d_stage].op.reduce_axis
    cfg.define_split("tile_b", b, num_outputs=2)
    cfg.define_split("tile_h", x_i, num_outputs=2)
    cfg.define_split("tile_w", x_j, num_outputs=2)
    cfg.define_split("tile_ci", c_i, num_outputs=2)
    cfg.define_split("tile_co", c_o, num_outputs=2)
    cfg.define_knob("oc_nthread", [1, 2])
    cfg.define_knob("h_nthread", [1, 2])
    ###### space definition end ######

    data, kernel = conv2d_stage.op.input_tensors
    if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
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
        cache_read_ewise.append(s.cache_read(tensor, env.acc_scope, [consumer]))

    # set ewise scope
    for op in ewise_ops:
        s[op].set_scope(env.acc_scope)
        s[op].pragma(s[op].op.axis[0], env.alu)

    for op in const_ops:
        s[op].compute_inline()

    # tile
    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[output].op.axis
    x_co0, x_co1 = cfg["tile_co"].apply(s, output, x_co)
    x_i0, x_i1 = cfg["tile_h"].apply(s, output, x_i)
    x_j0, x_j1 = cfg["tile_w"].apply(s, output, x_j)
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
    if cfg["oc_nthread"].val > 1:
        _, v_t = s[output].split(x_co0, factor=cfg["oc_nthread"].val)
        s[output].reorder(v_t, x_bo)
        s[output].bind(v_t, te.thread_axis("cthread"))

    # virtual threading along spatial rows
    if cfg["h_nthread"].val > 1:
        _, v_t = s[output].split(x_i0, factor=cfg["h_nthread"].val)
        s[output].reorder(v_t, x_bo)
        s[output].bind(v_t, te.thread_axis("cthread"))

    x_bo, x_co, x_i, x_j, x_bi, x_ci = s[conv2d_stage].op.axis
    k_o, d_i, d_j, k_i = s[conv2d_stage].op.reduce_axis
    s[conv2d_stage].reorder(x_bo, k_o, x_j, d_j, d_i, x_co, x_i, x_bi, x_ci, k_i)

    k_o, _ = cfg["tile_ci"].apply(s, conv2d_stage, k_o)
    s[cdata].compute_at(s[conv2d_stage], k_o)
    s[ckernel].compute_at(s[conv2d_stage], k_o)

    # Use VTA instructions
    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy)
    s[ckernel].pragma(s[ckernel].op.axis[0], env.dma_copy)
    s[conv2d_stage].tensorize(x_bi, env.gemm)
    s[output].pragma(x_co1, env.dma_copy)

    return s
