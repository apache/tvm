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
from tvm import te
import tvm.autotvm as autotvm
from tvm.autotvm.task.space import SplitEntity
from tvm.contrib import cublas
from .tensor_intrin import dp4a
from .. import nn
from .. import tag
from .. import generic
from ..util import traverse_inline, get_const_tuple

logger = logging.getLogger("topi")


@autotvm.register_topi_compute("dense_cublas.cuda")
def dense_cublas(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator on CUDA with CUBLAS"""
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    assert out_dtype == data.dtype, "Mixed precision not supported."
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    matmul = cublas.matmul(data, weight, False, True)
    cfg.add_flop(batch * in_dim * out_dim * 2)
    if bias is not None:
        matmul = te.compute(
            (batch, out_dim), lambda i, j: matmul[i, j] + bias[j], tag=tag.BROADCAST
        )
    return matmul


@autotvm.register_topi_schedule("dense_cublas.cuda")
def schedule_dense_cublas(_, outs):
    """Schedule dense operator using CUBLAS"""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("dense_small_batch.cuda")
def dense_small_batch(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator on CUDA"""
    return nn.dense(data, weight, bias, out_dtype)


@autotvm.register_topi_schedule("dense_small_batch.cuda")
def schedule_dense_small_batch(cfg, outs):
    """Schedule float32/64 dense with small batch size"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "dense":
            _schedule_dense_small_batch(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_dense_small_batch(cfg, s, C):
    A, _ = C.op.input_tensors
    _, in_dim = get_const_tuple(A.shape)
    cfg.define_split("tile_k", in_dim, num_outputs=2)
    if cfg.is_fallback:
        cfg["tile_k"] = SplitEntity([-1, 64] if in_dim > 64 else [1, 64])

    _, kf = cfg["tile_k"].apply(s, C, C.op.reduce_axis[0])
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


@autotvm.register_topi_compute("dense_large_batch.cuda")
def dense_large_batch(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator on CUDA"""
    return nn.dense(data, weight, bias, out_dtype)


@autotvm.register_topi_schedule("dense_large_batch.cuda")
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
    batch, in_dim = get_const_tuple(A.shape)
    out_dim, _ = get_const_tuple(B.shape)
    k = C.op.reduce_axis[0]

    # create tuning space
    try:
        block_cand = [64, 128]
        vthread_cand = [2 ** x for x in range(1, 7)]
        n_thread_cand = [2 ** x for x in range(3, 7)]
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


@autotvm.register_topi_compute("dense_int8.cuda")
def dense_int8(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator for int8 on CUDA"""
    if out_dtype is None:
        out_dtype = data.dtype

    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)
    k = te.reduce_axis((0, in_dim), name="k")

    matmul = te.compute(
        (batch, out_dim),
        lambda i, j: te.sum(
            data[i, k].astype(out_dtype) * weight[j, k].astype(out_dtype), axis=[k]
        ),
        tag="dense_int8",
    )

    cfg.add_flop(batch * in_dim * out_dim * 2)

    if bias is not None:
        matmul = te.compute(
            (batch, out_dim),
            lambda i, j: matmul[i, j] + bias[j].astype(out_dtype),
            tag=tag.BROADCAST,
        )
        cfg.add_flop(batch * out_dim)

    return matmul


@autotvm.register_topi_schedule("dense_int8.cuda")
def schedule_dense_int8(cfg, outs):
    """Dense schedule for int8 on CUDA"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense_int8" in op.tag:
            _schedule_dense_int8(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


_dp4a = dp4a("shared", "shared", "local")


def _schedule_dense_int8(cfg, s, output):
    data, weight = s[output].op.input_tensors

    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)

    in_dim_factor = 4
    assert in_dim % in_dim_factor == 0, "Input dimension must divide {}".format(in_dim_factor)
    if in_dim % 16 == 0:
        in_dim_factor = 16

    # create tuning space
    cfg.define_split("tile_y", batch, num_outputs=4)
    cfg.define_split("tile_x", out_dim, num_outputs=4)
    cfg.define_split("tile_k", in_dim // in_dim_factor, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    # create cache stage
    AA = s.cache_read(data, "shared", [output])
    WW = s.cache_read(weight, "shared", [output])
    CC = s.cache_write(output, "local")

    # handle bias
    if output.op not in s.outputs:
        s[output].compute_inline()
        output = s.outputs[0].output(0)

    n, x = s[output].op.axis

    # this is the scope to attach global config inside this kernel
    kernel_scope, n = s[output].split(n, nparts=1)

    ko = CC.op.reduce_axis[0]
    ko, ki = s[CC].split(ko, factor=4)
    ko, kt = cfg["tile_k"].apply(s, CC, ko)
    s[CC].tensorize(ki, _dp4a)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, n)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    s[output].reorder(by, bx, vy, vx, ty, tx, yi, xi)
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    n_ty = cfg["tile_y"].size[2]
    n_tx = cfg["tile_x"].size[2]

    s[CC].compute_at(s[output], tx)
    yo, xo = CC.op.axis[:2]
    s[CC].reorder(ko, kt, yo, xo, ki)

    for load in [AA, WW]:
        s[load].compute_at(s[CC], ko)

        outer, inner = s[load].split(s[load].op.axis[-1], factor=in_dim_factor)
        s[load].vectorize(inner)
        fused = s[load].op.axis[:-1] + [outer]
        fused = s[load].fuse(*fused)

        fused, tx = s[load].split(fused, factor=n_tx)
        fused, ty = s[load].split(fused, factor=n_ty)
        s[load].bind(tx, te.thread_axis("threadIdx.x"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))

    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[output].pragma(kernel_scope, "unroll_explicit", False)
    return s
