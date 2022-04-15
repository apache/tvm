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

# pylint: disable=invalid-name, unused-argument
"""Schedule for dense operator"""

import logging

from tvm import autotvm, te
from tvm.autotvm.task.space import SplitEntity

from .. import nn
from ..utils import traverse_inline, get_const_tuple

logger = logging.getLogger("topi")


@autotvm.register_topi_compute("dense_small_batch.gpu")
def dense_small_batch(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator on GPU"""
    return nn.dense(data, weight, bias, out_dtype)


@autotvm.register_topi_schedule("dense_small_batch.gpu")
def schedule_dense_small_batch(cfg, outs):
    """Schedule float32/64 dense with small batch size"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "dense":
            _schedule_dense_small_batch(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("matmul_default.gpu")
def matmul_default(
    cfg,
    tensor_a,
    tensor_b,
    bias=None,
    out_dtype=None,
    transpose_a=False,
    transpose_b=False,
):
    """Matmul operator on GPU"""
    return nn.matmul(tensor_a, tensor_b, bias, out_dtype, transpose_a, transpose_b)


@autotvm.register_topi_schedule("matmul_default.gpu")
def schedule_matmul_default(cfg, outs):
    """Schedule matmul on GPU"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "matmul":
            # Temporary use this as a basic schedule for matmul
            # TODO(jcf94): Add a more general schedule for matmul
            _schedule_dense_small_batch(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_dense_small_batch(cfg, s, C):
    A, weights = C.op.input_tensors
    if len(weights.op.input_tensors) == 1 and weights.op.input_tensors[0] == A:
        s[weights].compute_inline()

    _, in_dim_weights = get_const_tuple(weights.shape)
    _, in_dim_A = get_const_tuple(A.shape)

    if isinstance(in_dim_A, int):
        in_dim = in_dim_A
    elif isinstance(in_dim_weights, int):
        in_dim = in_dim_weights
    else:
        in_dim = None

    if in_dim is not None:
        cfg.define_split("tile_k", in_dim, num_outputs=2)
        if cfg.is_fallback:
            cfg["tile_k"] = SplitEntity([-1, 64] if in_dim > 64 else [1, 64])
        _, kf = cfg["tile_k"].apply(s, C, C.op.reduce_axis[0])
    else:
        tile_k = 64
        _, kf = s[C].split(C.op.reduce_axis[0], tile_k)

    CF = s.rfactor(C, kf)

    if C.op in s.outputs:
        Out = C
    else:
        Out = s.outputs[0].output(0)
        s[C].compute_at(s[Out], s[Out].op.axis[1])
    s[Out].bind(s[Out].op.axis[0], te.thread_axis("blockIdx.y"))
    s[Out].bind(s[Out].op.axis[1], te.thread_axis("blockIdx.x"))

    tx = s[C].op.reduce_axis[0]
    thread_x = te.thread_axis("threadIdx.x")
    s[C].bind(tx, thread_x)
    s[CF].compute_at(s[C], tx)
    s[C].set_store_predicate(thread_x.var.equal(0))
    s[Out].set_store_predicate(thread_x.var.equal(0))


@autotvm.register_topi_compute("dense_large_batch.gpu")
def dense_large_batch(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator on GPU"""
    return nn.dense(data, weight, bias, out_dtype)


@autotvm.register_topi_schedule("dense_large_batch.gpu")
def schedule_dense_large_batch(cfg, outs):
    """Schedule float32/64 dense with large batch size"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "dense":
            _schedule_dense_large_batch(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_dense_large_batch(cfg, s, C):
    """Schedule float32/64 dense with large batch size"""
    A, B = C.op.input_tensors
    if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
        s[B].compute_inline()
    batch, in_dim = get_const_tuple(A.shape)
    out_dim, _ = get_const_tuple(B.shape)
    k = C.op.reduce_axis[0]

    # create tuning space
    try:
        block_cand = [64, 128]
        vthread_cand = [2**x for x in range(1, 7)]
        n_thread_cand = [2**x for x in range(3, 7)]
        cfg.define_split(
            "tile_x",
            batch,
            num_outputs=4,
            filter=lambda x: (
                x.size[1] in vthread_cand
                and x.size[2] in n_thread_cand
                and (x.size[1] * x.size[2] * x.size[3]) in block_cand
            ),
        )
        cfg.define_split(
            "tile_y",
            out_dim,
            num_outputs=4,
            filter=lambda x: (
                x.size[1] in vthread_cand
                and x.size[2] in n_thread_cand
                and (x.size[1] * x.size[2] * x.size[3]) in block_cand
            ),
        )
        cfg.define_split("tile_k", in_dim, num_outputs=3, filter=lambda x: x.size[0] > 2)
    except IndexError:
        # Index error happens when no entities left after filtering, which was designed
        # to prune tuning space for better search efficiency.
        logger.debug("Tuning space was created without pruning due to unfit shapes")
        cfg.define_split("tile_x", batch, num_outputs=4)
        cfg.define_split("tile_y", out_dim, num_outputs=4)
        cfg.define_split("tile_k", in_dim, num_outputs=3)

    if cfg.is_fallback:
        if batch > 1:
            cfg["tile_x"] = SplitEntity([-1, 2, 16, 2])
        else:
            cfg["tile_x"] = SplitEntity([1, 1, 1, 1])
        if out_dim > 1:
            cfg["tile_y"] = SplitEntity([-1, 2, 16, 2])
        else:
            cfg["tile_y"] = SplitEntity([1, 1, 1, 1])
        if in_dim > 8:
            cfg["tile_k"] = SplitEntity([-1, 8, 1])
        else:
            cfg["tile_k"] = SplitEntity([-1, 1, 1])

    # Explicit memory access
    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BL = s.cache_read(BB, "local", [C])
    CC = s.cache_write(C, "local")

    # Deal with op fusion
    if C.op not in s.outputs:
        s[C].compute_inline()
        C = s.outputs[0].output(0)

    # Split and reorder computation
    bx, txz, tx, xi = cfg["tile_x"].apply(s, C, C.op.axis[0])
    by, tyz, ty, yi = cfg["tile_y"].apply(s, C, C.op.axis[1])
    s[C].reorder(by, bx, tyz, txz, ty, tx, yi, xi)
    s[CC].compute_at(s[C], tx)

    # Binding
    s[C].bind(by, te.thread_axis("blockIdx.y"))
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tyz, te.thread_axis("vthread"))
    s[C].bind(txz, te.thread_axis("vthread"))
    s[C].bind(ty, te.thread_axis("threadIdx.y"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))

    # Split reduction
    yo, xo = CC.op.axis
    ko, kt, ki = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(ko, kt, ki, yo, xo)
    s[AA].compute_at(s[CC], ko)
    s[BB].compute_at(s[CC], ko)
    s[CC].unroll(kt)
    s[AL].compute_at(s[CC], kt)
    s[BL].compute_at(s[CC], kt)

    # Schedule for A's shared memory load
    num_thread_x = cfg["tile_x"].size[2]
    ty, _ = s[AA].split(s[AA].op.axis[0], nparts=num_thread_x)
    _, xi = s[AA].split(s[AA].op.axis[1], factor=num_thread_x * 4)
    tx, xi = s[AA].split(xi, nparts=num_thread_x)
    s[AA].bind(ty, te.thread_axis("threadIdx.y"))
    s[AA].bind(tx, te.thread_axis("threadIdx.x"))
    s[AA].double_buffer()

    # Schedule for B' shared memory load
    num_thread_y = cfg["tile_y"].size[2]
    ty, _ = s[BB].split(s[BB].op.axis[0], nparts=num_thread_y)
    _, xi = s[BB].split(s[BB].op.axis[1], factor=num_thread_y * 4)
    tx, xi = s[BB].split(xi, nparts=num_thread_y)
    s[BB].bind(ty, te.thread_axis("threadIdx.y"))
    s[BB].bind(tx, te.thread_axis("threadIdx.x"))
    s[BB].double_buffer()
