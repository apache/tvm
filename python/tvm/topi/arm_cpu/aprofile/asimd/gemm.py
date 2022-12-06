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

import tvm
from tvm import te


def gemm_4x4_int8_int8_int32(M, N, K, unroll, in_type):
    """
    Int8 4x4 matrix multiplication and accumulation using a sequence of
    umull -> uadalp -> umull2 -> uadalp instructions. This function
    takes two arrays of int8 data type  A[4][K] and B[4][K], and produces
    a 4x4 matrix which is equal to A*B'.

    The pseudo code is as follows.

    .. code-block:: c

        void gemm_4x4_int8_int8_int32(int8 A[4][K], int8 B[4][K], int32 C[4][4]){
            for (int i = 0; i < 4; i++){
                for (int j = 0; j < 4; j++){
                    for (int k = 0; k < K; k++){
                        C[i][j] += A[i][k] * B[j][k]
                    }
            }
        }

    Notes:
        * The tiling strategy is picked to maximize register usage.

    Parameters
    ----------
    M : int
        rows of the matrix A
    N : int
        columns of the matrix B
    K : int
        columns of matrix A
    unroll : bool
        Unroll the loop accumulation if True
    in_type : str, {'uint8', 'int8'}

    Returns
    -------
    intrin : TensorIntrin
        The ARM uint8/int8 TensorIntrin that can be used in tensorizing schedule
    """
    assert in_type in ["uint8", "int8"]
    A = te.placeholder((K // 16, te.var("m"), 16), dtype=in_type, name="A")
    B = te.placeholder((K // 16, te.var("n"), 16), dtype=in_type, name="B")
    dtype_vec = in_type + "x16"
    idxm = tvm.tir.indexmod

    k = te.reduce_axis((0, K), "k")
    C = te.compute(
        (te.var("m"), te.var("n")),
        lambda x, y: te.sum(
            A[k // 16, x, idxm(k, 16)].astype("int32") * B[k // 16, y, idxm(k, 16)].astype("int32"),
            axis=k,
        ),
        name="C",
    )

    a_buffer = tvm.tir.decl_buffer(
        A.shape,
        dtype=in_type,
        name="a_buffer",
        offset_factor=1,
        strides=[te.var("sa_1"), te.var("sa_2"), 1],
    )

    b_buffer = tvm.tir.decl_buffer(
        B.shape,
        dtype=in_type,
        name="b_buffer",
        offset_factor=1,
        strides=[te.var("sb_1"), te.var("sb_2"), 1],
    )

    c_buffer = tvm.tir.decl_buffer(
        C.shape, dtype="int32", name="c_buffer", offset_factor=1, strides=[te.var("sc"), 1]
    )

    # Intrinsics used in the following algorithm
    umull_intrin = "llvm.aarch64.neon.umull" if in_type == "uint8" else "llvm.aarch64.neon.smull"
    uaddlp_intrin = "llvm.aarch64.neon.uaddlp" if in_type == "uint8" else "llvm.aarch64.neon.saddlp"
    addp_intrin = "llvm.aarch64.neon.addp"

    def uadalp(a, b):
        """Add pair and accumulate

        Parameters:
        ----------
        a: int16x8 vector
        b: int16x8 vector

        Returns:
        --------
            return a int32x4 vector

        Pseudocode:
        ----------
            a += (b0+b1, b2+b3, b4+b5, b6+b7)
        """

        return a + tvm.tir.call_llvm_pure_intrin(
            "int32x4", uaddlp_intrin, tvm.tir.const(1, "uint32"), b
        )

    def umull(a, b):
        """Multiply long (higher part)

        Parameters:
        ----------
        a: int8x16 vector
        b: int8x16 vector

        Returns:
        --------
            return a int16x8 vector

        Pseudocode:
        ----------
            c = (a0*b0, a1*b1, a2*b2, a3*b3, a4*b4, a5*b5, a6*b6, a7*b7)
        """
        a_high = tvm.tir.call_intrin("int8x8", "tir.vectorhigh", a)
        b_high = tvm.tir.call_intrin("int8x8", "tir.vectorhigh", b)
        c = tvm.tir.call_llvm_pure_intrin(
            "int16x8", umull_intrin, tvm.tir.const(2, "uint32"), a_high, b_high
        )
        return c

    def umull2(a, b):
        """Multiply long (lower part)

        Parameters:
        ----------
        a: int8x16 vector
        b: int8x16 vector

        Returns:
        --------
            return a int16x8 vector

        Pseudocode:
        ----------
            c = (a8*b8, a9*b9, a10*b10, a11*b11, a12*b12, a13*b13, a14*b14, a15*b15)
        """
        a_low = tvm.tir.call_intrin("int8x8", "tir.vectorlow", a)
        b_low = tvm.tir.call_intrin("int8x8", "tir.vectorlow", b)
        c = tvm.tir.call_llvm_pure_intrin(
            "int16x8", umull_intrin, tvm.tir.const(2, "uint32"), a_low, b_low
        )
        return c

    def addp(a, b):
        """Add two vectors in pairs

        Parameters:
        ----------
        a: int32x4 vector
        b: int32x4 vector

        Returns:
        --------
            return a int32x4 vector

        Pseudocode:
        ----------
            c = (a0+a1, a2+a3, b0+b1, b0+b3)
        """
        return tvm.tir.call_llvm_pure_intrin(
            "int32x4", addp_intrin, tvm.tir.const(2, "uint32"), a, b
        )

    def accumulation_loop(M, N, ins, acc, tile_idx):
        """Internal tile accumulation. This function
        takes two arrays of int8 data type  A[tile_idx][4][16] and B[tile_idx][4][16], produces
        a 4x4 matrix which is equal to A*B' and accumulates into C[4][4]

        The pseudo code is as follows.

        .. code-block:: c

            void gemm_4x4_int8_int8_int32(int8 A[tile_idx][4][K],
                                          int8 B[tile_idx][4][K],
                                          int32 C[4][4]){
                for (int i = 0; i < 4; i++){
                    for (int j = 0; j < 4; j++){
                        for (int k = 0; k < 16; k++){
                            C[i][j] += A[tile_idx][i][k] * B[tile_idx][j][k]
                        }
                }
            }

        Notes:
            * The tiling strategy is picked to maximize register usage.

        Parameters:
        ----------
        M : int
            Number of total rows of the output matrix
        N : int
            Number of total columns of the output matrix
        ins : list of tvm.tir.buffer
            Input buffers
        acc : tvm.tir.ir_builder.BufferVar
            Bank of register accumulators
        tiled_idx : int
            Index of a sub-tile of A and B in A[tile_idx][:][:] and B[tile_idx][:][:].
            Please note that  0 <= tile_idx <= K//16

        """
        a0 = ins[0].vload([tile_idx, 0, 0], dtype_vec)
        a1 = tvm.tir.const(0, "int8x16")
        if M > 1:
            a1 = ins[0].vload([tile_idx, 1, 0], dtype_vec)
        a2 = tvm.tir.const(0, "int8x16")
        if M > 2:
            a2 = ins[0].vload([tile_idx, 2, 0], dtype_vec)
        a3 = tvm.tir.const(0, "int8x16")
        if M > 3:
            a3 = ins[0].vload([tile_idx, 3, 0], dtype_vec)

        b0 = ins[1].vload([tile_idx, 0, 0], dtype_vec)
        b1 = tvm.tir.const(0, "int8x16")
        if N > 1:
            b1 = ins[1].vload([tile_idx, 1, 0], dtype_vec)
        b2 = tvm.tir.const(0, "int8x16")
        if N > 2:
            b2 = ins[1].vload([tile_idx, 2, 0], dtype_vec)
        b3 = tvm.tir.const(0, "int8x16")
        if N > 3:
            b3 = ins[1].vload([tile_idx, 3, 0], dtype_vec)

        # First half
        # Lower part of a0 * {b0,b1,b2,b3}
        d00 = umull(a0, b0)
        d01 = umull(a0, b1)
        d02 = umull(a0, b2)
        d03 = umull(a0, b3)

        # Lower part of a1 * {b0,b1,b2,b3}
        d10 = umull(a1, b0)
        d11 = umull(a1, b1)
        d12 = umull(a1, b2)
        d13 = umull(a1, b3)

        # Accumulate
        acc[0] = uadalp(acc[0], d00)
        acc[1] = uadalp(acc[1], d01)
        acc[2] = uadalp(acc[2], d02)
        acc[3] = uadalp(acc[3], d03)
        acc[4] = uadalp(acc[4], d10)
        acc[5] = uadalp(acc[5], d11)
        acc[6] = uadalp(acc[6], d12)
        acc[7] = uadalp(acc[7], d13)

        # Higher part of a0 * {b0,b1,b2,b3}
        d00 = umull2(a0, b0)
        d01 = umull2(a0, b1)
        d02 = umull2(a0, b2)
        d03 = umull2(a0, b3)

        # Higher part of a1 * {b0,b1,b2,b3}
        d10 = umull2(a1, b0)
        d11 = umull2(a1, b1)
        d12 = umull2(a1, b2)
        d13 = umull2(a1, b3)

        # Accumulate again
        acc[0] = uadalp(acc[0], d00)
        acc[1] = uadalp(acc[1], d01)
        acc[2] = uadalp(acc[2], d02)
        acc[3] = uadalp(acc[3], d03)
        acc[4] = uadalp(acc[4], d10)
        acc[5] = uadalp(acc[5], d11)
        acc[6] = uadalp(acc[6], d12)
        acc[7] = uadalp(acc[7], d13)

        # Second half
        # Lower part of a2 * {b0,b1,b2,b3}
        d00 = umull(a2, b0)
        d01 = umull(a2, b1)
        d02 = umull(a2, b2)
        d03 = umull(a2, b3)

        # Lower part of a3 * {b0,b1,b2,b3}
        d10 = umull(a3, b0)
        d11 = umull(a3, b1)
        d12 = umull(a3, b2)
        d13 = umull(a3, b3)

        # Accumulate
        acc[8] = uadalp(acc[8], d00)
        acc[9] = uadalp(acc[9], d01)
        acc[10] = uadalp(acc[10], d02)
        acc[11] = uadalp(acc[11], d03)
        acc[12] = uadalp(acc[12], d10)
        acc[13] = uadalp(acc[13], d11)
        acc[14] = uadalp(acc[14], d12)
        acc[15] = uadalp(acc[15], d13)

        # Higher part of a2 * {b0,b1,b2,b3}
        d00 = umull2(a2, b0)
        d01 = umull2(a2, b1)
        d02 = umull2(a2, b2)
        d03 = umull2(a2, b3)

        # Lower part of a3 * {b0,b1,b2,b3}
        d10 = umull2(a3, b0)
        d11 = umull2(a3, b1)
        d12 = umull2(a3, b2)
        d13 = umull2(a3, b3)

        # Accumulate
        acc[8] = uadalp(acc[8], d00)
        acc[9] = uadalp(acc[9], d01)
        acc[10] = uadalp(acc[10], d02)
        acc[11] = uadalp(acc[11], d03)
        acc[12] = uadalp(acc[12], d10)
        acc[13] = uadalp(acc[13], d11)
        acc[14] = uadalp(acc[14], d12)
        acc[15] = uadalp(acc[15], d13)

    def _intrin_func(ins, outs):
        def _instr():
            ib = tvm.tir.ir_builder.create()
            # Allocate a local buffer (possibly translates to registers)
            acc = ib.allocate("int32x4", 16, name="accs", scope="local")
            m = outs[0].shape[0]
            n = outs[0].shape[1]
            # Initialization
            for i in range(0, 16):
                acc[i] = tvm.tir.const(0, "int32x4")

            if unroll:
                for i in range(0, int(K // 16)):
                    accumulation_loop(M, N, ins, acc, i)
            else:
                with ib.for_range(0, K // 16, name="i") as i:
                    accumulation_loop(M, N, ins, acc, i)

            # Final accumulations
            # acc[4*r + c] contains the partial accumulations of element C[r][c]
            #
            # In particular:
            # acc[4*r] contains the partial sums of a[r,0:K].*b[0,0:K] -> (a,b,c,d)
            # acc[4*r+1] contains the partial sums of a[r, 0:K].*b[1,0:K] -> (e,f,g,h)
            # acc[4*r+2] contains the partial sums of a[r, 0:K].*b[2,0:K] -> (i,j,k,l)
            # acc[4*r+3] contains the partial sums of a[r, 0:K].*b[3,0:K] -> (m,n,o,p)
            #
            # Please note that 0<= r, c < 4

            acc[0] = addp(acc[0], acc[1])  # (a+b, c+d, e+f, g+h)
            acc[1] = addp(acc[2], acc[3])  # (i+j, k+l, m+n, o+p)
            acc[0] = addp(acc[0], acc[1])  # (a+b+c+d, e+f+g+h, i+j+k+l, m+n+o+p)

            acc[4] = addp(acc[4], acc[5])  # (a+b, c+d, e+f, g+h)
            acc[5] = addp(acc[6], acc[7])  # (i+j, k+l, m+n, o+p)
            acc[4] = addp(acc[4], acc[5])  # (a+b+c+d, e+f+g+h, i+j+k+l, m+n+o+p)

            acc[8] = addp(acc[8], acc[9])  # (a+b, c+d, e+f, g+h)
            acc[9] = addp(acc[10], acc[11])  # (i+j, k+l, m+n, o+p)
            acc[8] = addp(acc[8], acc[9])  # (a+b+c+d, e+f+g+h, i+j+k+l, m+n+o+p)

            acc[12] = addp(acc[12], acc[13])  # (a+b, c+d, e+f, g+h)
            acc[13] = addp(acc[14], acc[15])  # (i+j, k+l, m+n, o+p)
            acc[12] = addp(acc[12], acc[13])  # (a+b+c+d, e+f+g+h, i+j+k+l, m+n+o+p)

            # Store the result
            if N > 3:
                out_0 = acc[0]
                out_1 = acc[4]
                out_2 = acc[8]
                out_3 = acc[12]
            elif N > 2:
                out_0 = tvm.tir.call_intrin("int32x3", "tir.reinterpret", acc[0])
                out_1 = tvm.tir.call_intrin("int32x3", "tir.reinterpret", acc[4])
                out_2 = tvm.tir.call_intrin("int32x3", "tir.reinterpret", acc[8])
                out_3 = tvm.tir.call_intrin("int32x3", "tir.reinterpret", acc[12])
            elif N > 1:
                out_0 = tvm.tir.call_intrin("int32x2", "tir.reinterpret", acc[0])
                out_1 = tvm.tir.call_intrin("int32x2", "tir.reinterpret", acc[4])
                out_2 = tvm.tir.call_intrin("int32x2", "tir.reinterpret", acc[8])
                out_3 = tvm.tir.call_intrin("int32x2", "tir.reinterpret", acc[12])
            else:
                out_0 = tvm.tir.call_intrin("int32", "tir.reinterpret", acc[0])
                out_1 = tvm.tir.call_intrin("int32", "tir.reinterpret", acc[4])
                out_2 = tvm.tir.call_intrin("int32", "tir.reinterpret", acc[8])
                out_3 = tvm.tir.call_intrin("int32", "tir.reinterpret", acc[12])

            ib.emit(outs[0].vstore([0, 0], out_0))
            if M > 1:
                ib.emit(outs[0].vstore([1, 0], out_1))
            if M > 2:
                ib.emit(outs[0].vstore([2, 0], out_2))
            if M > 3:
                ib.emit(outs[0].vstore([3, 0], out_3))
            return ib.get()

        # body, reset, update
        return _instr()

    buffer_params = {"offset_factor": 1}
    return te.decl_tensor_intrin(
        C.op,
        _intrin_func,
        binds={A: a_buffer, B: b_buffer, C: c_buffer},
        default_buffer_params=buffer_params,
    )
