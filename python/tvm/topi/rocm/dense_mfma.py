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
# pylint: disable=invalid-name, too-many-locals, too-many-statements, unused-argument
"""Compute and Schedule definition for dense tensorcore with cuda backend"""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
import tvm.autotvm as autotvm
from .. import tag
from ..util import traverse_inline, get_const_tuple
from .tensor_intrin import (
    intrin_mfma_load_matrix,
    intrin_mfma_store_matrix,
    intrin_mfma_gemm,
)


@autotvm.register_topi_compute("dense_mfma.rocm")
def dense_mfma(cfg, data, weight, bias=None, out_dtype=None):
    """Dense MFMA operator on ROCM"""
    matmul = dense_mfma_rocm(data, weight, bias, out_dtype)
    return matmul


@autotvm.register_topi_schedule("dense_mfma.rocm")
def schedule_dense_mfma(cfg, outs):
    """Schedule dense operator using MFMA"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "dense_mfma":
            _schedule_dense_mfma(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def dense_mfma_rocm(data, weight, bias=None, out_dtype=None):
    """Dense MFMA operator on ROCM"""
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)
    assert (
        batch % 16 == 0 and in_dim % 16 == 0 and out_dim % 16 == 0
    ), "batch, in_dim, and out_dim each must be a multiple of 16"
    k = te.reduce_axis((0, in_dim), name="k")
    data_16 = te.compute((batch, in_dim), lambda b, i: data[b, i].astype("float16"))
    weight_16 = te.compute((out_dim, in_dim), lambda o, i: weight[o, i].astype("float16"))
    matmul = te.compute(
        (batch, out_dim),
        lambda i, j: te.sum(
            data_16[i, k].astype(out_dtype) * weight_16[j, k].astype(out_dtype), axis=k
        ),
        name="T_dense",
        tag="dense_mfma",
    )
    if bias is not None:
        matmul = te.compute(
            (batch, out_dim),
            lambda i, j: matmul[i, j] + bias[j].astype(out_dtype),
            tag=tag.BROADCAST,
        )
    return matmul


def _schedule_dense_mfma(cfg, s, C):
    """Schedule dense operator using MFMA"""
    A, B = s[C].op.input_tensors
    batch, out_dim = get_const_tuple(C.shape)
    out_dtype = C.dtype
    s[A].compute_inline()
    s[B].compute_inline()

    # Explicit memory access
    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AF = s.cache_read(AS, "local", [C])
    BF = s.cache_read(BS, "local", [C])
    CF = s.cache_write(C, "local")
    CS = s.cache_read(CF, "shared", [C])

    # Support op fusion
    if C.op not in s.outputs:
        s[C].compute_inline()
        C = s.outputs[0].output(0)

    cfg.define_knob("block_row_warps", [1, 2, 4])
    cfg.define_knob("block_col_warps", [1, 2, 4])
    cfg.define_knob("warp_row_tiles", [1, 2, 4])
    cfg.define_knob("warp_col_tiles", [1, 2, 4])
    cfg.define_knob("chunk", [1, 2, 4, 8])
    cfg.define_knob("offset", [0, 8])
    cfg.define_knob("offsetCS", [0, 8])
    cfg.define_knob("vec", [1, 2, 4, 8])

    warp_size = 64
    mfma_m = 16
    mfma_n = 16
    mfma_k = 16
    block_row_warps = cfg["block_row_warps"].val
    block_col_warps = cfg["block_col_warps"].val
    warp_row_tiles = cfg["warp_row_tiles"].val
    warp_col_tiles = cfg["warp_col_tiles"].val
    chunk = cfg["chunk"].val
    offset = cfg["offset"].val
    offsetCS = cfg["offsetCS"].val
    vec = cfg["vec"].val

    # Define the stride for tensorization
    AS_align = chunk * mfma_k + offset
    BS_align = chunk * mfma_k + offset
    CS_align = warp_col_tiles * block_col_warps * mfma_n + offsetCS
    AS_stride = [AS_align, 1]
    BS_stride = [BS_align, 1]
    AF_stride = [mfma_k, 1]
    BF_stride = [mfma_k, 1]
    CF_stride = [warp_col_tiles * mfma_n, 1]
    CS_stride = [CS_align, 1]

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    # Schedule for dense computation
    block_factor_b = mfma_m * warp_row_tiles * block_row_warps
    block_factor_o = mfma_n * warp_col_tiles * block_col_warps
    b, o = C.op.axis
    block_i, bc = s[C].split(b, factor=block_factor_b)
    block_j, oc = s[C].split(o, factor=block_factor_o)
    s[C].reorder(block_i, block_j, bc, oc)
    t = s[C].fuse(bc, oc)
    t, vi = s[C].split(t, factor=vec)
    t, tx = s[C].split(t, factor=warp_size)
    t, ty = s[C].split(t, factor=block_row_warps)
    t, tz = s[C].split(t, factor=block_col_warps)
    s[C].bind(block_i, block_x)
    s[C].bind(block_j, block_y)
    s[C].bind(tz, thread_z)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].vectorize(vi)

    # Schedule for fragment store
    s[CS].compute_at(s[C], block_j)
    bb, oo = CS.op.axis
    s[CS].storage_align(bb, CS_align - 1, CS_align)
    bb, bbi = s[CS].split(bb, factor=mfma_m)
    oo, ooi = s[CS].split(oo, factor=mfma_n)
    bb, bbii = s[CS].split(bb, factor=warp_row_tiles)
    oo, ooii = s[CS].split(oo, factor=warp_col_tiles)
    s[CS].reorder(bb, oo, bbii, ooii, bbi, ooi)

    # Schedule for gemm computation
    s[CF].compute_at(s[CS], oo)
    warp_i, warp_j = CF.op.axis
    warp_i, _ii = s[CF].split(warp_i, factor=mfma_m)
    warp_j, _jj = s[CF].split(warp_j, factor=mfma_n)
    (k,) = CF.op.reduce_axis
    k, _k = s[CF].split(k, factor=mfma_k)
    ko, ki = s[CF].split(k, factor=chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)

    # Schedule for tensorized matrix_A load
    s[AF].compute_at(s[CF], ki)
    b, i = AF.op.axis
    b, b_ii = s[AF].split(b, factor=mfma_m)
    i, i_jj = s[AF].split(i, factor=mfma_k)
    s[AF].reorder(b, i, b_ii, i_jj)

    # Schedule for tensorized matrix_B load
    s[BF].compute_at(s[CF], ki)
    o, i = BF.op.axis
    o, o_ii = s[BF].split(o, factor=mfma_n)
    i, i_ii = s[BF].split(i, factor=mfma_k)
    s[BF].reorder(o, i, o_ii, i_ii)

    # Schedule for A's(B's) shared memory load
    def shared_shedule(stage, strides):
        # New thread axes to avoid LLVM bug,
        # llvm/lib/Transforms/Utils/LCSSA.cpp:380:
        # bool llvm::formLCSSA(...): Assertion `L.isLCSSAForm(DT)' failed.`
        thread_x = te.thread_axis("threadIdx.x")
        thread_y = te.thread_axis("threadIdx.y")
        thread_z = te.thread_axis("threadIdx.z")
        s[stage].compute_at(s[CF], ko)
        xo, yo = stage.op.axis
        s[stage].storage_align(xo, strides - 1, strides)
        t = s[stage].fuse(xo, yo)
        t, vi = s[stage].split(t, factor=vec)
        t, tx = s[stage].split(t, factor=warp_size)
        t, ty = s[stage].split(t, factor=block_row_warps)
        _, tz = s[stage].split(t, factor=block_col_warps)
        s[stage].bind(ty, thread_y)
        s[stage].bind(tz, thread_z)
        s[stage].bind(tx, thread_x)
        s[stage].vectorize(vi)

    shared_shedule(AS, AS_align)
    shared_shedule(BS, BS_align)

    def mfma_compute(M, N, K):
        A = te.placeholder((M, K), name="A", dtype="float16")
        B = te.placeholder((K, N), name="B", dtype="float16")
        k = te.reduce_axis((0, K), name="k")
        return (
            A,
            B,
            te.compute(
                (M, N),
                lambda i, j: te.sum(A[i, k].astype("float") * B[j, k].astype("float"), axis=[k]),
                name="C",
            ),
        )

    # Lower the inner loop nest down to MFMA hardware intrinsics
    s[AF].tensorize(
        b_ii,
        intrin_mfma_load_matrix(
            (mfma_m, mfma_n, mfma_k), "A", strides_src=AS_stride, strides_dst=AF_stride
        ),
    )
    s[BF].tensorize(
        o_ii,
        intrin_mfma_load_matrix(
            (mfma_m, mfma_n, mfma_k), "W", strides_src=BS_stride, strides_dst=BF_stride
        ),
    )
    s[CF].tensorize(
        _ii,
        intrin_mfma_gemm(
            (mfma_m, mfma_n, mfma_k),
            mfma_compute,
            input_scope="local",
            strides_A=AF_stride,
            strides_B=BF_stride,
            strides_C=CF_stride,
        ),
    )
    s[CS].tensorize(
        bbi,
        intrin_mfma_store_matrix(
            shape=(mfma_m, mfma_n, mfma_k), strides_src=CF_stride, strides_dst=CS_stride
        ),
    )
