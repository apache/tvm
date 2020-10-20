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
    assert (batch % 16 == 0 and in_dim % 16 == 0 and out_dim % 16 == 0), "batch, in_dim, and out_dim each must be a multiple of 16"
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

    # fallback support
    # target = tvm.target.Target.current()
    # if cfg.is_fallback:
    #     ref_log = autotvm.tophub.load_reference_log(
    #         target.kind.name, target.model, "dense_mfma.rocm"
    #     )
    #     cfg.fallback_with_reference_log(ref_log)

    # Deal with op fusion, such as bias and relu
    if C.op not in s.outputs:
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

    warp_size = 64
    wmma_m = 16
    wmma_n = 16
    wmma_k = 16
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
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    # Schedule for dense computation
    block_factor_b = wmma_m * warp_row_tiles * block_row_warps
    block_factor_o = wmma_n * warp_col_tiles * block_col_warps
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

    # Schedule for wmma store
    s[CS].compute_at(s[C], block_j)
    bb, oo = CS.op.axis
    s[CS].storage_align(bb, CS_align - 1, CS_align)
    bb, bbi = s[CS].split(bb, factor=wmma_m)
    oo, ooi = s[CS].split(oo, factor=wmma_n)
    bb, bbii = s[CS].split(bb, factor=warp_row_tiles)
    oo, ooii = s[CS].split(oo, factor=warp_col_tiles)
    s[CS].reorder(bb, oo, bbii, ooii, bbi, ooi)

    # Schedule for wmma computation
    s[CF].compute_at(s[CS], oo)
    warp_i, warp_j = CF.op.axis
    warp_i, _ii = s[CF].split(warp_i, factor=wmma_m)
    warp_j, _jj = s[CF].split(warp_j, factor=wmma_n)
    (k,) = CF.op.reduce_axis
    k, _k = s[CF].split(k, factor=wmma_k)
    ko, ki = s[CF].split(k, factor=chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)

    # Schedule for  wmma_matrix_a load
    s[AF].compute_at(s[CF], ki)
    b, i = AF.op.axis
    b, b_ii = s[AF].split(b, factor=wmma_m)
    i, i_jj = s[AF].split(i, factor=wmma_k)
    s[AF].reorder(b, i, b_ii, i_jj)

    # Schedule for  wmma_matrix_b load
    s[BF].compute_at(s[CF], ki)
    o, i = BF.op.axis
    o, o_ii = s[BF].split(o, factor=wmma_n)
    i, i_ii = s[BF].split(i, factor=wmma_k)
    s[BF].reorder(o, i, o_ii, i_ii)

    # Schedule for A's(B's) shared memory load
    def shared_shedule(stage, strides):
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


    # shape = (wmma_m, wmma_n, wmma_k)
    # in_dtype = "float16"
    # AL_gemm = te.placeholder((wmma_m, wmma_k), name="AL_gemm", dtype=in_dtype)
    # BL_gemm = te.placeholder((wmma_n, wmma_k), name="BL_gemm", dtype=in_dtype)
    # k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
    # CL_compute = te.compute(
    #     (wmma_m, wmma_n),
    #     lambda ii, jj: te.sum(
    #         AL_gemm[ii, k_gemm].astype(out_dtype) * BL_gemm[jj, k_gemm].astype(out_dtype),
    #         axis=k_gemm,
    #     ),
    #     name="CL_compute",
    # )

    # lower the computation loops down to MFMA hardware intrinsics
    # by mapping the dense MFMA to tensor intrinsics
    s[AF].tensorize(b_ii, intrin_mfma_load_matrix((wmma_m, wmma_n, wmma_k), "A", strides_src=AS_stride, strides_dst=AF_stride))
    s[BF].tensorize(o_ii, intrin_mfma_load_matrix((wmma_m, wmma_n, wmma_k), "A", strides_src=BS_stride, strides_dst=BF_stride))
    s[CF].tensorize(_ii, intrin_mfma_gemm(shape=(wmma_m, wmma_n, wmma_k), dtype="float16", input_scope="local", strides_A=AF_stride, strides_B=BF_stride, strides_C=CF_stride))
    s[CS].tensorize(bbi, intrin_mfma_store_matrix(shape=(wmma_m, wmma_n, wmma_k), strides_src=CF_stride, strides_dst=CS_stride))


    # s[AF].tensorize(
    #     b_ii,
    #     intrin_wmma_load_matrix_A(
    #         AF_stride, AS_stride, shape, "row_major", (wmma_m, wmma_k), (wmma_m, wmma_k), "float16"
    #     ),
    # )
    # s[BF].tensorize(
    #     o_ii,
    #     intrin_wmma_load_matrix_W(
    #         BF_stride, BS_stride, shape, "col_major", (wmma_n, wmma_k), (wmma_n, wmma_k), "float16"
    #     ),
    # )
    # s[CF].tensorize(
    #     _ii, intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride, CF_stride, shape)
    # )
    # s[CS].tensorize(
    #     bbi,
    #     intrin_wmma_store_matrix(
    #         CS_stride, CF_stride, shape, out_dtype, (wmma_m, wmma_n), (wmma_m, wmma_n)
    #     ),
    # )

def intrin_mfma_load_matrix(shape, matrix, thread=None, strides_src=None, strides_dst=None):
    M, N, K = shape
    if matrix == "A":
        row, col = M, K
    elif matrix == "B":
        row, col = K, N
    output_shape = (row, col)

    A = te.placeholder(output_shape, name=matrix, dtype="float16")
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="shared", offset_factor=1, strides=strides_src
    )

    C = te.compute(output_shape, lambda i, j: A[i, j], name="C")

    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope="local", offset_factor=1, strides=strides_dst
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]

        tx = thread
        if tx == None:
            tx = te.thread_axis("threadIdx.x")
            ib.scope_attr(tx, "thread_extent", 64)

        blk_td = tx % 16
        offset = tx // 16
        # TODO(csullivan): Using offset works, but using tx directly does not, fix this
        if matrix == "A":
            for blk_id in range(0,4):
                ib.emit(BC.vstore([0, blk_id], BA.vload([blk_td, blk_id*4 + offset], "float16")))
        elif matrix == "B":
            for blk_id in range(0,4):
                ib.emit(BC.vstore([0, blk_id], BA.vload([blk_id*4 + offset, blk_td], "float16")))
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})

def intrin_mfma_store_matrix(shape, thread=None, strides_src=None, strides_dst=None):
    M, N, K = shape
    A = te.placeholder((M, N), name="A", dtype="float32")
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="local", offset_factor=1, strides=strides_src
    )
    C = te.compute((M, N), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope="shared", offset_factor=1, strides=strides_dst
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        tx = thread
        if tx == None:
            tx = te.thread_axis("threadIdx.x")
            ib.scope_attr(tx, "thread_extent", 64)
        vec_width = 4
        blk_id = tx // 16
        blk_td = tx % 16

        # TODO(csullivan): Consider TVM change to BufferVar.__getitem__
        # to convert int to const for quality of life when using vector types.
        ib.emit(BC.vstore([blk_td, blk_id*vec_width], BA.vload([0, 0], "float32x4")))

        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})

def intrin_mfma_gemm(shape, dtype, input_scope, strides_A=None, strides_B=None, strides_C=None):
    M, N, K = shape

    # TODO(csullivan):Replace below with a function to get the correct
    # llvm function based on shape and type
    assert M == 16
    assert N == 16
    assert K == 16
    mfma_instr_name = "llvm.amdgcn.mfma.f32.16x16x16f16"
    llvm_id = tvm.target.codegen.llvm_lookup_intrinsic_id(mfma_instr_name)
    assert llvm_id != 0

    A = te.placeholder((M, K), name="A", dtype="float16")
    B = te.placeholder((K, N), name="B", dtype="float16")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k].astype("float") * B[j, k].astype("float") , axis=[k]),
        name="C"
    )

    Ab = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="Ab", scope=input_scope, offset_factor=1, strides=strides_A
    )
    Bb = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="Bb", scope=input_scope, offset_factor=1, strides=strides_B
    )
    Cb = tvm.tir.decl_buffer(
        C.shape, C.dtype, name="Cb", scope="local", offset_factor=1, strides=strides_C
    )

    def intrin_func(ins, outs):
        Ab, Bb = ins
        (Cb,) = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(Cb.vstore([0, 0], tvm.tir.const(0, "float32x4")))
            return ib.get()

        def update():
            # Each thread is responsible for 4 values along the reduction axis for a single (m, n) pixel
            ib = tvm.tir.ir_builder.create()
            a_vec = Ab.vload([0, 0], "float16x4")
            b_vec = Bb.vload([0, 0], "float16x4")
            c_vec = Cb.vload([0, 0], "float32x4")
            args_6 = tvm.tir.const(6, "uint32")
            # Transpose inputs (equivalent to switching the order in this packed layout)
            # in order to ensure row-major order on the output with coalesced vector writes.
            gemm = tvm.tir.call_llvm_pure_intrin(
                "float32x4",
                mfma_instr_name,
                args_6,
                b_vec,
                a_vec,
                c_vec,
                0,0,0
            )
            ib.emit(Cb.vstore([0, 0], gemm))
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: Ab, B: Bb, C: Cb})
