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
# pylint: disable=unused-argument
"""Dense operator declaration and schedule registration for VTA."""

import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi

from ..environment import get_env


def is_packed_layout(layout):
    """Check if layout is packed layout"""
    if layout == "NCHW":
        return False
    if "n" in layout and "c" in layout:
        return True
    return False


@autotvm.register_topi_compute("dense_packed.vta")
def dense_packed(cfg, data, weight, bias=None, out_dtype=None):
    """Dense function declaration."""

    # Make sure that the dense operator is packed
    if len(data.shape) != 4 or len(weight.shape) != 4:
        raise topi.InvalidShapeError()

    # Derive shapes
    ishape = topi.utils.get_const_tuple(data.shape)
    wshape = topi.utils.get_const_tuple(weight.shape)
    oshape = (data.shape[0], weight.shape[0], data.shape[2], weight.shape[2])

    # Reduction axes (input channel)
    assert ishape[1] == wshape[1]
    assert ishape[3] == wshape[3]
    k_o = te.reduce_axis((0, ishape[1]), name="k_o")
    k_i = te.reduce_axis((0, ishape[3]), name="k_i")
    res = te.compute(
        oshape,
        lambda b_o, c_o, b_i, c_i: te.sum(
            data[b_o, k_o, b_i, k_i].astype(out_dtype)
            * weight[c_o, k_o, c_i, k_i].astype(out_dtype),
            axis=[k_o, k_i],
        ),
        name="res",
        tag="dense_pack",
    )

    cfg.add_flop(2 * np.prod(topi.utils.get_const_tuple(oshape)) * ishape[1] * ishape[3])

    return res


@autotvm.register_topi_schedule("dense_packed.vta")
def schedule_dense_packed(cfg, outs):
    """Packed dense schedule."""

    assert len(outs) == 1
    output = outs[0]
    const_ops = []
    ewise_inputs = []
    ewise_ops = []
    dense_res = []
    assert "int" in output.op.input_tensors[0].dtype

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
            assert op.tag == "dense_pack"
            dense_res.append(op)

    _traverse(output.op)
    assert len(dense_res) == 1
    dense_stage = dense_res[0].output(0)
    s = te.create_schedule(output.op)

    ##### space definition begin #####
    b, c_o, _, _ = s[dense_stage].op.axis
    c_i, _ = s[dense_stage].op.reduce_axis
    cfg.define_split("tile_b", b, num_outputs=2)
    cfg.define_split("tile_ci", c_i, num_outputs=2)
    cfg.define_split("tile_co", c_o, num_outputs=2)
    cfg.define_knob("oc_nthread", [1, 2])
    ###### space definition end ######

    data, weight = dense_stage.op.input_tensors

    env = get_env()

    cdata = s.cache_read(data, env.inp_scope, [dense_stage])
    cweight = s.cache_read(weight, env.wgt_scope, [dense_stage])
    s[dense_stage].set_scope(env.acc_scope)

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

    # apply tiling for SRAM reuse
    x_b, x_c, _, _ = s[output].op.axis
    x_bo, x_bi = cfg["tile_b"].apply(s, output, x_b)
    x_co, x_ci = cfg["tile_co"].apply(s, output, x_c)
    s[output].reorder(x_bo, x_co, x_bi, x_ci)
    store_pt = x_co

    # set all compute scopes
    s[dense_stage].compute_at(s[output], store_pt)
    for op in ewise_ops:
        s[op].compute_at(s[output], store_pt)

    for tensor in cache_read_ewise:
        s[tensor].compute_at(s[output], store_pt)
        s[tensor].pragma(s[tensor].op.axis[0], env.dma_copy)

    # virtual threading along output channel axes
    if cfg["oc_nthread"].val > 1:
        _, v_t = s[output].split(x_co, factor=cfg["oc_nthread"].val)
        s[output].reorder(v_t, x_bo)
        s[output].bind(v_t, te.thread_axis("cthread"))

    x_bo, x_co, x_bi, _ = s[dense_stage].op.axis
    k_o, _ = s[dense_stage].op.reduce_axis
    s[dense_stage].reorder(x_bo, k_o, x_co)

    k_o, _ = cfg["tile_ci"].apply(s, dense_stage, k_o)
    s[cdata].compute_at(s[dense_stage], k_o)
    s[cweight].compute_at(s[dense_stage], k_o)

    # Use VTA instructions
    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy)
    s[cweight].pragma(s[cweight].op.axis[0], env.dma_copy)
    s[dense_stage].tensorize(x_bi, env.gemm)
    s[output].pragma(x_ci, env.dma_copy)

    return s
