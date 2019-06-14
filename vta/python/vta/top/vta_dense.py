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
"""Dense operator declaration and schedule registration for VTA."""

import numpy as np
import tvm
from tvm import autotvm
import topi

from ..environment import get_env

def is_packed_layout(layout):
    """Check if layout is packed layout"""
    if layout == "NCHW":
        return False
    if "n" in layout and "c" in layout:
        return True
    return False

@autotvm.register_topi_compute(topi.nn.dense, 'vta', 'direct')
def _declaration_dense(cfg,
                       data,
                       weight,
                       bias=None,
                       out_dtype=None):
    """Dense function declaration."""

    # Make sure that the dense operator is packed
    assert len(data.shape) == 4
    assert len(weight.shape) == 4
    # Derive output shape
    oshape = (data.shape[0], weight.shape[0], data.shape[2], weight.shape[2])

    # Reduction axes (input channel)
    assert(int(data.shape[1]) == int(weight.shape[1]))
    assert(int(data.shape[3]) == int(weight.shape[3]))
    k_o = tvm.reduce_axis((0, data.shape[1]), name='k_o')
    k_i = tvm.reduce_axis((0, data.shape[3]), name='k_i')
    res = tvm.compute(
        oshape,
        lambda b_o, c_o, b_i, c_i: tvm.sum(
            data[b_o, k_o, b_i, k_i].astype(out_dtype) *
            weight[c_o, k_o, c_i, k_i].astype(out_dtype),
            axis=[k_o, k_i]),
        name="res", tag="packed_dense")

    cfg.add_flop(2 * np.prod(topi.util.get_const_tuple(oshape)) *
                 data.shape[1] * data.shape[3])
    return res

@autotvm.register_topi_schedule(topi.generic.schedule_dense, 'vta', 'direct')
def _schedule_dense(cfg, outs):
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
                if len(op.axis) == 0:
                    const_ops.append(op)
                else:
                    ewise_ops.append(op)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                    ewise_inputs.append((op, tensor))
                else:
                    _traverse(tensor.op)
        else:
            assert op.tag == "packed_dense"
            dense_res.append(op)

    _traverse(output.op)
    assert len(dense_res) == 1
    dense_stage = dense_res[0].output(0)
    s = tvm.create_schedule(output.op)

    ##### space definition begin #####
    b, co, _, _ = s[dense_stage].op.axis
    ci, _ = s[dense_stage].op.reduce_axis
    cfg.define_split('tile_b', b, num_outputs=2)
    cfg.define_split('tile_co', co, num_outputs=2)
    cfg.define_split('tile_ci', ci, num_outputs=2)
    cfg.define_knob('oc_nthread', [1, 2])
    ###### space definition end ######

    data, kernel = dense_stage.op.input_tensors

    env = get_env()

    cdata = s.cache_read(data, env.inp_scope, [dense_stage])
    ckernel = s.cache_read(kernel, env.wgt_scope, [dense_stage])
    s[dense_stage].set_scope(env.acc_scope)

    # cache read input
    cache_read_ewise = []
    for consumer, tensor in ewise_inputs:
        cache_read_ewise.append(
            s.cache_read(tensor, env.acc_scope, [consumer]))

    # set ewise scope
    for op in ewise_ops:
        s[op].set_scope(env.acc_scope)
        s[op].pragma(s[op].op.axis[0], env.alu)

    for op in const_ops:
        s[op].compute_inline()

    # tile
    x_bo, x_co, x_bi, x_ci = s[output].op.axis
    x_bo0, x_bo1 = cfg['tile_b'].apply(s, output, x_bo)
    x_co0, x_co1 = cfg['tile_co'].apply(s, output, x_co)
    s[output].reorder(x_bo0, x_co0, x_bo1, x_co1, x_bi, x_ci)
    store_pt = x_co0

    # set all compute scopes
    s[dense_stage].compute_at(s[output], store_pt)
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

    x_bo, x_co, x_bi, x_ci = s[dense_stage].op.axis
    k_o, k_i = s[dense_stage].op.reduce_axis
    s[dense_stage].reorder(x_bo, k_o, x_co, x_bi, x_ci, k_i)

    k_o, _ = cfg['tile_ci'].apply(s, dense_stage, k_o)
    s[cdata].compute_at(s[dense_stage], k_o)
    s[ckernel].compute_at(s[dense_stage], k_o)

    # Use VTA instructions
    s[cdata].pragma(s[cdata].op.axis[0], env.dma_copy)
    s[ckernel].pragma(s[ckernel].op.axis[0], env.dma_copy)
    s[dense_stage].tensorize(x_bi, env.gemm)
    s[output].pragma(x_co1, env.dma_copy)

    return s