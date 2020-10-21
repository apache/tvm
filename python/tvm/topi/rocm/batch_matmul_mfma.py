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

@autotvm.register_topi_compute("batch_matmul_mfma.rocm")
def batch_matmul_mfma(cfg, x, y, out_shape=None):
    """Computes matrix multiplication of `x` and `y` when
    `x` and `y` are batched matrices via Matrix FMA on ROCM.

    Parameters
    ----------
    cfg : ConfigSpace
        Autotvm tuning config
    x : tvm.te.Tensor
        3-D with shape [batch, M, K]
    y : tvm.te.Tensor
        3-D with shape [batch, N, K]
    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    batch, M, K = get_const_tuple(x.shape)
    _, N, _ = get_const_tuple(y.shape)
    if out_shape is not None:
        assert out_shape[0] == batch, "Input and output batch sizes must match"
        assert out_shape[1] == M and out_shape[2] == N, "Invalid output shape"
    matmul = batch_matmul_mfma_rocm(x, y)
    return matmul


@autotvm.register_topi_schedule("batch_matmul_mfma.rocm")
def schedule_batch_matmul_mfma(cfg, outs):
    """Schedule batch_matmul operator using MFMA"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "batch_matmul_mfma":
            _schedule_batch_matmul_mfma(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def batch_matmul_mfma_rocm(A, B):
    """Batch Matmul MFMA operator on ROCM"""
    assert len(A.shape) == 3 and len(B.shape) == 3, "only support 3-dim batch_matmul"

    batch, M, K = get_const_tuple(A.shape)
    # B is transposed by default in relay
    _, N, _ = get_const_tuple(B.shape)

    assert (M % 16 == 0 and N % 16 == 0 and K % 16 == 0), "M, N, and K each must be a multiple of 16"
    A_16 = te.compute((batch, M, K), lambda b, i, j: A[b, i, j].astype("float16"))
    B_16 = te.compute((batch, N, K), lambda b, i, j: B[b, i, j].astype("float16"))

    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (batch, M, N),
        lambda b, i, j: te.sum(
            A_16[b, i, k].astype("float") * B_16[b, j, k].astype("float"), axis=[k]
        ),
        name="T_batch_matmul",
        tag="batch_matmul_mfma",
    )

    return C


def _schedule_batch_matmul_mfma(cfg, s, C):
    """Schedule batch_matmul operator using MFMA"""
    A, B = s[C].op.input_tensors
    batch, M, N = get_const_tuple(C.shape)

    s[A].compute_inline()
    s[B].compute_inline()

    # Explicit memory access
    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AF = s.cache_read(AS, "local", [C])
    BF = s.cache_read(BS, "local", [C])
    CF = s.cache_write(C, "local")
    CS = s.cache_read(CF, "shared", [C])

    # # Support op fusion
    # if C.op not in s.outputs:
    #     s[C].compute_inline()
    #     C = s.outputs[0].output(0)

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
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    # Schedule for batch_matmul computation
    block_factor_row = mfma_m * warp_row_tiles * block_row_warps
    block_factor_col = mfma_n * warp_col_tiles * block_col_warps
    # b, o = C.op.axis
    b, i, j = s[C].op.axis
    block_i, i = s[C].split(i, factor=block_factor_row)
    block_j, j = s[C].split(j, factor=block_factor_col)
    s[C].reorder(block_i, block_j, i, j)
    t = s[C].fuse(i, j)
    t, vi = s[C].split(t, factor=vec)
    t, tx = s[C].split(t, factor=warp_size)
    t, ty = s[C].split(t, factor=block_row_warps)
    t, tz = s[C].split(t, factor=block_col_warps)
    s[C].bind(b, block_z)
    s[C].bind(block_i, block_x)
    s[C].bind(block_j, block_y)
    s[C].bind(tz, thread_z)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].vectorize(vi)

    # Schedule for fragment store
    s[CS].compute_at(s[C], block_j)
    _, _m, _n = CS.op.axis
    s[CS].storage_align(_m, CS_align - 1, CS_align)
    _m, _mi = s[CS].split(_m, factor=mfma_m)
    _n, _ni = s[CS].split(_n, factor=mfma_n)
    _m, _mo = s[CS].split(_m, factor=warp_row_tiles)
    _n, _no = s[CS].split(_n, factor=warp_col_tiles)
    s[CS].reorder(_m, _n, _mo, _no, _mi, _ni)
    # s[CS].compute_at(s[C], block_j)
    # bb, oo = CS.op.axis
    # s[CS].storage_align(bb, CS_align - 1, CS_align)
    # bb, bbi = s[CS].split(bb, factor=mfma_m)
    # oo, ooi = s[CS].split(oo, factor=mfma_n)
    # bb, bbii = s[CS].split(bb, factor=warp_row_tiles)
    # oo, ooii = s[CS].split(oo, factor=warp_col_tiles)
    # s[CS].reorder(bb, oo, bbii, ooii, bbi, ooi)

    # Schedule for gemm computation
    s[CF].compute_at(s[CS], _n)
    b, warp_i, warp_j = CF.op.axis
    warp_i, _ii = s[CF].split(warp_i, factor=mfma_m)
    warp_j, _jj = s[CF].split(warp_j, factor=mfma_n)
    (k,) = CF.op.reduce_axis
    k, _k = s[CF].split(k, factor=mfma_k)
    ko, ki = s[CF].split(k, factor=chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)

    # Schedule for tensorized matrix_A load
    s[AF].compute_at(s[CF], ki)
    _, _ma, _ka = AF.op.axis
    _ma, _mai = s[AF].split(_ma, factor=mfma_m)
    _ka, _kai = s[AF].split(_ka, factor=mfma_k)
    s[AF].reorder(_ma, _ka, _mai, _kai)

    # Schedule for tensorized matrix_B load
    s[BF].compute_at(s[CF], ki)
    _, _nb, _kb = BF.op.axis
    _nb, _nbi = s[BF].split(_nb, factor=mfma_n)
    _kb, _kbi = s[BF].split(_kb, factor=mfma_k)
    s[BF].reorder(_nb, _kb, _nbi, _kbi)

    # Schedule for A's(B's) shared memory load
    def shared_shedule(stage, strides):
        # New thread axes to avoid LLVM bug,
        # llvm/lib/Transforms/Utils/LCSSA.cpp:380:
        # bool llvm::formLCSSA(...): Assertion `L.isLCSSAForm(DT)' failed.`
        thread_x = te.thread_axis("threadIdx.x")
        thread_y = te.thread_axis("threadIdx.y")
        thread_z = te.thread_axis("threadIdx.z")
        s[stage].compute_at(s[CF], ko)
        _, xo, yo = stage.op.axis
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
        return A, B, te.compute(
            (M, N),
            lambda i, j: te.sum(A[i, k].astype("float") * B[j, k].astype("float") , axis=[k]),
            name="C"
        )

    # Lower the inner loop nest down to MFMA hardware intrinsics
    s[AF].tensorize(_mai, intrin_mfma_load_matrix((mfma_m, mfma_n, mfma_k), "A", strides_src=AS_stride, strides_dst=AF_stride))
    s[BF].tensorize(_nbi, intrin_mfma_load_matrix((mfma_m, mfma_n, mfma_k), "BT", strides_src=BS_stride, strides_dst=BF_stride))
    s[CF].tensorize(_ii, intrin_mfma_gemm((mfma_m, mfma_n, mfma_k), mfma_compute, input_scope="local", strides_A=AF_stride, strides_B=BF_stride, strides_C=CF_stride))
    s[CS].tensorize(_mi, intrin_mfma_store_matrix(shape=(mfma_m, mfma_n, mfma_k), strides_src=CF_stride, strides_dst=CS_stride))
