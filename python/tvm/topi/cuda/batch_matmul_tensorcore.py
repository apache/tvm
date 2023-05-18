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
# pylint: disable=invalid-name,too-many-locals,unused-variable,unused-argument
"""cuda batch_matmul operators"""
import tvm
from tvm import autotvm
from tvm import te
from ..utils import traverse_inline, get_const_tuple
from .tensor_intrin import (
    intrin_wmma_load_matrix_A,
    intrin_wmma_load_matrix_W,
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
)


@autotvm.register_topi_compute("batch_matmul_tensorcore.cuda")
def batch_matmul_tensorcore(
    cfg, x, y, out_shape=None, out_dtype=None, transpose_a=False, transpose_b=True
):
    """batch matmul tensorcore operator on cuda"""
    # TODO(jcf94): Deal with different transpose combinations
    assert not transpose_a and transpose_b
    # TODO(liuxin.ai): Deal with out_shape for broadcast
    del out_shape
    return batch_matmul_tensorcore_cuda(x, y, out_dtype)


@autotvm.register_topi_schedule("batch_matmul_tensorcore.cuda")
def schedule_batch_matmul_tensorcore(cfg, outs):
    """Schedule for batch_matmul operator using Tensorcore

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of batch_matmul
          in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule(cfg, s, C):
        A, B = s[C].op.input_tensors
        if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
            s[B].compute_inline()
        batch, m_dim, k_dim = get_const_tuple(A.shape)
        batch, n_dim, k_dim = get_const_tuple(B.shape)
        data_dtype = A.dtype
        out_dtype = C.dtype

        # Explicit memory access
        AS = s.cache_read(A, "shared", [C])
        BS = s.cache_read(B, "shared", [C])
        AF = s.cache_read(AS, "wmma.matrix_a", [C])
        BF = s.cache_read(BS, "wmma.matrix_b", [C])
        CF = s.cache_write(C, "wmma.accumulator")
        CS = s.cache_read(CF, "shared", [C])

        # fallback support
        target = tvm.target.Target.current()
        if cfg.is_fallback:
            ref_log = autotvm.tophub.load_reference_log(
                target.kind.name, target.model, "batch_matmul_tensorcore.cuda"
            )
            cfg.fallback_with_reference_log(ref_log)

        # Deal with op fusion, such as bias/relu and slice after padding
        if C.op not in s.outputs and "injective" in s.outputs[0].tag:
            s[C].compute_inline()
            C = s.outputs[0].output(0)

        # create tuning space
        cfg.define_knob("block_row_warps", [1, 2, 4])
        cfg.define_knob("block_col_warps", [1, 2, 4])
        cfg.define_knob("warp_row_tiles", [1, 2, 4])
        cfg.define_knob("warp_col_tiles", [1, 2, 4])
        cfg.define_knob("chunk", [1, 2, 4, 8])
        cfg.define_knob("offset", [0, 8])
        cfg.define_knob("offsetCS", [0, 8])
        cfg.define_knob("vec", [1, 2, 4, 8])

        # Ensure that the default parameters are applicable when autotvm is not in use
        if data_dtype in ["float16", "uint8", "int8"]:
            if m_dim % 32 == 0 and n_dim % 8 == 0:
                cfg.define_knob("wmma_m", [32, 16, 8])
            elif m_dim % 16 == 0 and n_dim % 16 == 0:
                cfg.define_knob("wmma_m", [16, 8, 32])
            elif m_dim % 8 == 0 and n_dim % 32 == 0:
                cfg.define_knob("wmma_m", [8, 16, 32])
            wmma_k = 16
            wmma_m = cfg["wmma_m"].val
            if wmma_m == 16:
                wmma_n = 16
            elif wmma_m == 8:
                wmma_n = 32
            elif wmma_m == 32:
                wmma_n = 8
        elif data_dtype in ["int4", "uint4"]:
            wmma_m = wmma_n = 8
            wmma_k = 32
        else:
            raise ValueError(f"data dtype {data_dtype} is not yet supported")

        warp_size = 32
        block_row_warps = cfg["block_row_warps"].val
        block_col_warps = cfg["block_col_warps"].val
        warp_row_tiles = cfg["warp_row_tiles"].val
        warp_col_tiles = cfg["warp_col_tiles"].val
        chunk = cfg["chunk"].val
        offset = cfg["offset"].val
        offsetCS = cfg["offsetCS"].val
        vec = cfg["vec"].val

        # Define the stride of intrin functions
        AS_align = chunk * wmma_k + offset
        BS_align = chunk * wmma_k + offset
        CS_align = warp_col_tiles * block_col_warps * wmma_n + offsetCS
        AS_stride = [AS_align, 1]
        BS_stride = [BS_align, 1]
        AF_stride = [wmma_k, 1]
        BF_stride = [wmma_k, 1]
        CF_stride = [warp_col_tiles * wmma_n, 1]
        CS_stride = [CS_align, 1]

        block_x = te.thread_axis("blockIdx.x")
        block_y = te.thread_axis("blockIdx.y")
        block_z = te.thread_axis("blockIdx.z")
        thread_x = te.thread_axis("threadIdx.x")
        thread_y = te.thread_axis("threadIdx.y")
        thread_z = te.thread_axis("threadIdx.z")

        # Schedule for dense computation
        block_factor_m = wmma_m * warp_row_tiles * block_row_warps
        block_factor_n = wmma_n * warp_col_tiles * block_col_warps
        b, m, n = C.op.axis
        block_i, bc = s[C].split(m, factor=block_factor_m)
        block_j, oc = s[C].split(n, factor=block_factor_n)
        s[C].reorder(b, block_i, block_j, bc, oc)
        t = s[C].fuse(bc, oc)
        t, vi = s[C].split(t, factor=vec)
        t, tx = s[C].split(t, factor=warp_size)
        t, ty = s[C].split(t, factor=block_row_warps)
        t, tz = s[C].split(t, factor=block_col_warps)
        s[C].bind(block_i, block_x)
        s[C].bind(block_j, block_y)
        s[C].bind(b, block_z)
        s[C].bind(tz, thread_z)
        s[C].bind(ty, thread_y)
        s[C].bind(tx, thread_x)
        s[C].vectorize(vi)

        # Schedule for wmma store
        s[CS].compute_at(s[C], block_j)
        bs, bb, oo = CS.op.axis
        s[CS].storage_align(bb, CS_align - 1, CS_align)
        bb, bbi = s[CS].split(bb, factor=wmma_m)
        oo, ooi = s[CS].split(oo, factor=wmma_n)
        bb, bbii = s[CS].split(bb, factor=warp_row_tiles)
        oo, ooii = s[CS].split(oo, factor=warp_col_tiles)
        s[CS].reorder(bs, bb, oo, bbii, ooii, bbi, ooi)
        s[CS].bind(bb, thread_z)
        s[CS].bind(oo, thread_y)

        # Schedule for wmma computation
        s[CF].compute_at(s[CS], oo)
        bs, warp_i, warp_j = CF.op.axis
        warp_i, _ii = s[CF].split(warp_i, factor=wmma_m)
        warp_j, _jj = s[CF].split(warp_j, factor=wmma_n)
        (k,) = CF.op.reduce_axis
        k, _k = s[CF].split(k, factor=wmma_k)
        ko, ki = s[CF].split(k, factor=chunk)
        s[CF].reorder(bs, ko, ki, warp_i, warp_j, _ii, _jj, _k)

        # Schedule for  wmma_matrix_a load
        s[AF].compute_at(s[CF], ki)
        bs, b, i = AF.op.axis
        b, b_ii = s[AF].split(b, factor=wmma_m)
        i, i_jj = s[AF].split(i, factor=wmma_k)
        s[AF].reorder(bs, b, i, b_ii, i_jj)

        # Schedule for  wmma_matrix_b load
        s[BF].compute_at(s[CF], ki)
        bs, o, i = BF.op.axis
        o, o_ii = s[BF].split(o, factor=wmma_n)
        i, i_ii = s[BF].split(i, factor=wmma_k)
        s[BF].reorder(bs, o, i, o_ii, i_ii)

        # Schedule for A's(B's) shared memory load
        def shared_schedule(stage, strides):
            s[stage].compute_at(s[CF], ko)
            bs, xo, yo = stage.op.axis
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

        shared_schedule(AS, AS_align)
        shared_schedule(BS, BS_align)

        shape = (wmma_m, wmma_n, wmma_k)
        AL_gemm = te.placeholder((wmma_m, wmma_k), name="AL_gemm", dtype=data_dtype)
        BL_gemm = te.placeholder((wmma_n, wmma_k), name="BL_gemm", dtype=data_dtype)
        k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
        CL_compute = te.compute(
            (wmma_m, wmma_n),
            lambda ii, jj: te.sum(
                AL_gemm[ii, k_gemm].astype(out_dtype) * BL_gemm[jj, k_gemm].astype(out_dtype),
                axis=k_gemm,
            ),
            name="CL_compute",
        )

        # lower the computation loops down to TensorCore hardware intrinsics
        # by mapping the dense tensorcore to tensor intrinsics
        s[AF].tensorize(
            b_ii,
            intrin_wmma_load_matrix_A(
                AF_stride,
                AS_stride,
                shape,
                "row_major",
                (wmma_m, wmma_k),
                (wmma_m, wmma_k),
                data_dtype,
            ),
        )
        s[BF].tensorize(
            o_ii,
            intrin_wmma_load_matrix_W(
                BF_stride,
                BS_stride,
                shape,
                "col_major",
                (wmma_n, wmma_k),
                (wmma_n, wmma_k),
                data_dtype,
            ),
        )
        s[CF].tensorize(
            _ii,
            intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride, CF_stride, shape),
        )
        s[CS].tensorize(
            bbi,
            intrin_wmma_store_matrix(
                CS_stride, CF_stride, shape, out_dtype, (wmma_m, wmma_n), (wmma_m, wmma_n)
            ),
        )

    def _callback(op):
        if "batch_matmul_tensorcore" in op.tag:
            _schedule(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def batch_matmul_tensorcore_cuda(x, y, out_dtype=None):
    """Computes batch matrix multiplication of `x` and `y` when `x` and `y` are
    data in batch.

    Parameters
    ----------
    x : tvm.te.Tensor
        3-D with shape [batch, M, K]

    y : tvm.te.Tensor
        3-D with shape [batch, N, K]

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    assert len(x.shape) == 3 and len(y.shape) == 3, "only support 3-dim batch_matmul"
    x_shape = get_const_tuple(x.shape)
    y_shape = get_const_tuple(y.shape)
    assert x_shape[0] == y_shape[0], "batch dimension doesn't match"
    assert x_shape[2] == y_shape[2], "shapes of x and y is inconsistent"
    batch, M, K = x.shape
    N = y.shape[1]

    if out_dtype is None:
        out_dtype = x.dtype

    assert x.dtype == y.dtype
    assert x.dtype in ["float16", "uint8", "int8", "uint4", "int4"]
    if x.dtype in ["float16", "uint8", "int8"]:
        assert (
            (M % 8 == 0 and K % 16 == 0 and N % 32 == 0)
            or (M % 16 == 0 and K % 16 == 0 and N % 16 == 0)
            or (M % 32 == 0 and K % 16 == 0 and N % 8 == 0)
        ), "The shape of (M, K, N) must be multiple of (16, 16, 16) or (32, 16, 8) or (8, 16, 32)"
    else:
        assert (
            M % 8 == 0 and K % 32 == 0 and N % 8 == 0
        ), "The shape of (M, K, N) must be multiple of (8, 32, 8)"

    k = te.reduce_axis((0, K), name="k")
    return te.compute(
        (batch, M, N),
        lambda b, i, j: te.sum(x[b, i, k].astype(out_dtype) * y[b, j, k].astype(out_dtype), axis=k),
        tag="batch_matmul_tensorcore",
    )
