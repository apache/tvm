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
# pylint: disable=invalid-name, unnecessary-lambda, too-many-arguments
"""MFMA Tensor intrinsics for GFX908."""
import tvm
from tvm import te


def intrin_mfma_load_matrix(shape, matrix, thread=None, strides_src=None, strides_dst=None):
    """Intrin function for loading thread registers for mfma tensorization"""
    M, N, K = shape
    assert M == 16
    assert N == 16
    assert K == 16
    if matrix in ("A", "BT", "W"):
        row, col = M, K
    elif matrix == "B":
        row, col = K, N
    output_shape = (row, col)

    A = te.placeholder(output_shape, name=matrix, dtype="float16")
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope="shared", offset_factor=1, strides=strides_src)

    C = te.compute(output_shape, lambda i, j: A[i, j], name="C")

    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope="local", offset_factor=1, strides=strides_dst)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]

        tx = thread
        if tx is None:
            tx = te.thread_axis("threadIdx.x")
            ib.scope_attr(tx, "thread_extent", 64)

        blk_td = tx % 16
        offset = tx // 16
        # TODO(csullivan): Using offset works, but using tx directly does not, fix this
        if matrix in ("A", "BT", "W"):
            for blk_id in range(0, 4):
                ib.emit(BC.vstore([0, blk_id], BA.vload([blk_td, blk_id * 4 + offset], "float16")))
        elif matrix == "B":
            for blk_id in range(0, 4):
                ib.emit(BC.vstore([0, blk_id], BA.vload([blk_id * 4 + offset, blk_td], "float16")))
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_mfma_store_matrix(shape, thread=None, strides_src=None, strides_dst=None):
    """Intrin function for storing result to accumulator registers"""
    M, N, K = shape
    assert M == 16
    assert N == 16
    assert K == 16
    A = te.placeholder((M, N), name="A", dtype="float32")
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope="local", offset_factor=1, strides=strides_src)
    C = te.compute((M, N), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope="shared", offset_factor=1, strides=strides_dst)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        tx = thread
        if tx is None:
            tx = te.thread_axis("threadIdx.x")
            ib.scope_attr(tx, "thread_extent", 64)
        vec_width = 4
        blk_id = tx // 16
        blk_td = tx % 16

        # TODO(csullivan): Consider TVM change to BufferVar.__getitem__
        # to convert int to const for quality of life when using vector types.
        ib.emit(BC.vstore([blk_td, blk_id * vec_width], BA.vload([0, 0], "float32x4")))

        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_mfma_gemm(
    shape, te_mfma_compute, input_scope, strides_A=None, strides_B=None, strides_C=None
):
    """Intrin definition for mfma_16x16x16f16 compute"""
    M, N, K = shape
    assert M == 16
    assert N == 16
    assert K == 16
    mfma_instr_name = "llvm.amdgcn.mfma.f32.16x16x16f16"
    llvm_id = tvm.target.codegen.llvm_lookup_intrinsic_id(mfma_instr_name)
    assert llvm_id != 0

    A, B, C = te_mfma_compute(*shape)

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
            ib = tvm.tir.ir_builder.create()
            a_vec = Ab.vload([0, 0], "float16x4")
            b_vec = Bb.vload([0, 0], "float16x4")
            c_vec = Cb.vload([0, 0], "float32x4")
            args_6 = tvm.tir.const(6, "uint32")
            # Transpose inputs in order to ensure row-major order on the output for
            # coalesced vector writes.
            gemm = tvm.tir.call_llvm_pure_intrin(
                "float32x4", mfma_instr_name, args_6, b_vec, a_vec, c_vec, 0, 0, 0
            )
            ib.emit(Cb.vstore([0, 0], gemm))
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: Ab, B: Bb, C: Cb})
