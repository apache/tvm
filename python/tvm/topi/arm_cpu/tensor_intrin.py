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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Conv2D int8 schedule on ARM"""

import tvm
from tvm import te
from tvm.ir import register_intrin_lowering


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


def dot_int8_int8_int32_neon_82(int32_lanes, dtype="uint"):
    """
    Int8 dot product by every 4 elements using ARM v8.2 udot.
    This function takes two arrays of int8 datatype -- data[4] and
    kernel[int32_lanes][4] -- and computes a dot product of data[4] with every
    4 elements of kernels, resulting in output[int32_lanes] of uint32 datatype.
    The pseudo code is as follows.

    .. code-block:: c

        void dot_int8_int8_int32(int8 data[4], int8 kernel[16][4], int32 output[16]){
            for (int i = 0; i < int32_lanes; i++){
                out[i] = 0;
                for (int k = 0; k < 4; k++){
                    out[i] += data[k] * kernel[i][k]
                }
            }
        }

    Physically, the kernel array sits in a vector register and
    the data[4] is broadcasted to another vector register. This
    function returns a TensorIntrin that can be used to tensorize
    a schedule.

    Parameters
    ----------
    int32_lanes : int
        How many int32/uint32 to produce
    dtype : str, optional, {"uint", "int"}
        Whether it works on unsigned int or signed int

    Returns
    -------
    intrin : TensorIntrin
        The ARM uint8 TensorIntrin that can be used in tensorizing schedule
    """
    num_int8_elements = 4  # 4 int8 elements in int32

    data = te.placeholder((num_int8_elements,), dtype=f"{dtype}8", name="data")
    kernel = te.placeholder((int32_lanes, num_int8_elements), dtype=f"{dtype}8", name="kernel")

    k = te.reduce_axis((0, num_int8_elements), name="k")
    C = te.compute(
        (int32_lanes,),
        lambda i: te.sum(data[k].astype(f"{dtype}32") * kernel[i, k].astype(f"{dtype}32"), axis=k),
        name="C",
    )

    a_buffer = tvm.tir.decl_buffer(
        data.shape, dtype=f"{dtype}8", name="a_buffer", offset_factor=1, strides=[1]
    )
    b_buffer = tvm.tir.decl_buffer(
        kernel.shape, dtype=f"{dtype}8", name="b_buffer", offset_factor=1, strides=[te.var("s"), 1]
    )

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.tir.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore(0, tvm.tir.const(0, f"{dtype}32x{int32_lanes}")))
                return ib.get()

            dtype_a = f"{dtype}8x{num_int8_elements}"
            dtype_b = f"{dtype}8x{int32_lanes * num_int8_elements}"
            dtype_c = f"{dtype}32x{int32_lanes}"

            a_int8 = ins[0].vload([0], dtype_a)
            re_int32 = tvm.tir.call_intrin(f"{dtype}32", "tir.reinterpret", a_int8)
            # broadcast a
            vec_ai32 = re_int32.astype(dtype_c)

            vec_a = tvm.tir.call_intrin(dtype_b, "tir.reinterpret", vec_ai32)
            vec_b = ins[1].vload([0, 0], dtype_b)
            vec_c = outs[0].vload([0], dtype_c)

            inst = "udot" if dtype == "uint" else "sdot"
            inst = "llvm.aarch64.neon.%s.v%di32.v%di8" % (
                inst,
                int32_lanes,
                int32_lanes * num_int8_elements,
            )
            vdot = tvm.tir.call_llvm_pure_intrin(
                dtype_c, inst, tvm.tir.const(3, "uint32"), vec_c, vec_a, vec_b
            )
            ib.emit(outs[0].vstore(0, vdot))
            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    buffer_params = {"offset_factor": 1}
    return te.decl_tensor_intrin(
        C.op,
        _intrin_func,
        binds={data: a_buffer, kernel: b_buffer},
        default_buffer_params=buffer_params,
    )


def dot_int8_int8_int32_neon():
    """
    Int8 dot product using vmlal instructions

    .. code-block:: c

        void dot_int8_int8_int32(int8 data[4], int8 kernel[4][4], int32 output[4]){
            for (int i = 0; i < 4; i++){
                out[i] = 0;
                for (int k = 0; k < 4; k++){
                    out[i] += data[k] * kernel[i][k]
                }
            }
        }

    We use the smull and saddlp instructions to compute the dot product.
    smull : int8x16 -> int8x16 -> int16x8 elementwise multiplication
    saddlp: int16x8 -> int32x4 pairwise addition of elements

    Data is broadcast across the register
    int8 elements
    |         data      |         data      |
    |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |

                      smull

    int8 elements
    |     kernel[i]     |     kernel[i+1]   |
    |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |

                        =

    int16 elements
    |               data * kernel[i]        |         data * kernel[i+1]            |
    |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |

                                          saddlp =

    int32 elements
    |    partial sum(data * kernel[i])      |  partial sum(data * kernel[i+1])      |
    |         0         |         1         |         2         |         3         |


    We apply the above kernel twice and use addp to compute the second set of pairwise additions

    int32 elements (narrowed for so they fit on a line)
    |    psum d*k[i]    |   psum d*k[i+1]   |           |   psum d*k[i+2]   |   psum d*k[i+3]   |
    |    0    |    1    |    2    |    3    |   addp    |    4    |    5    |    6    |    7    |
                                                 =
    |sum d*ki |sum d*ki1|sum d*ki2|sum d*ki3|
    |    0    |    1    |    2    |    3    |


    """
    int32_lanes = 4  # 4 int32 lanes = 128
    num_int8_elements = 4  # 4 int8 elements in int32
    data = te.placeholder((num_int8_elements,), dtype="int8", name="data")
    kernel = te.placeholder((int32_lanes, num_int8_elements), dtype="int8", name="kernel")
    k = te.reduce_axis((0, num_int8_elements), name="k")
    C = te.compute(
        (int32_lanes,),
        lambda i: te.sum(data[k].astype("int32") * kernel[i, k].astype("int32"), axis=k),
        name="C",
    )

    a_buffer = tvm.tir.decl_buffer(
        data.shape, dtype="int8", name="a_buffer", offset_factor=1, strides=[1]
    )
    b_buffer = tvm.tir.decl_buffer(
        kernel.shape, dtype="int8", name="b_buffer", offset_factor=1, strides=[te.var("ldw"), 1]
    )

    def _intrin_func(ins, outs):
        def _instr(index):
            int_8xl = "int8x8"
            int_32xl = "int32x4"
            ib = tvm.tir.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore(0, tvm.tir.const(0, int_32xl)))
                return ib.get()

            # this broadcasts data to the vector size
            a_int8 = ins[0].vload([0], "int8x4")
            re_int32 = tvm.tir.call_intrin("int32", "tir.reinterpret", a_int8)
            vec_ai32 = re_int32.astype("int32x2")
            vec_a = tvm.tir.call_intrin(int_8xl, "tir.reinterpret", vec_ai32)

            vec_b = ins[1].vload([0, 0], "int8x16")

            def pairwise_add_mul(extract_half):
                vec_b_half = tvm.tir.call_intrin("int8x8", extract_half, vec_b)
                multiply = tvm.tir.call_llvm_pure_intrin(
                    "int16x8",
                    "llvm.aarch64.neon.smull.v8i16",  # saturating pairwise multiplication
                    tvm.tir.const(2, "uint32"),
                    vec_a,
                    vec_b_half,
                )
                pairwise_reduction = tvm.tir.call_llvm_pure_intrin(
                    "int32x4",
                    "llvm.aarch64.neon.saddlp.v4i32.v8i16",
                    tvm.tir.const(1, "uint32"),
                    multiply,
                )
                return pairwise_reduction

            pair_1 = pairwise_add_mul("tir.vectorlow")
            pair_2 = pairwise_add_mul("tir.vectorhigh")
            quad_reduction = tvm.tir.call_llvm_pure_intrin(
                "int32x4",
                "llvm.aarch64.neon.addp.v4i32",
                tvm.tir.const(2, "uint32"),
                pair_1,
                pair_2,
            )
            if index == 0:
                ib.emit(outs[0].vstore(0, quad_reduction))
            else:
                ib.emit(outs[0].vstore(0, quad_reduction + outs[0].vload([0], int_32xl)))
            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    buffer_params = {"offset_factor": 1}
    return te.decl_tensor_intrin(
        C.op,
        _intrin_func,
        binds={data: a_buffer, kernel: b_buffer},
        default_buffer_params=buffer_params,
    )


def select_word(vec, lane, dtype_vec):
    """
    Utility function used to select a int8x4 word within a int8x16 vector
    and replicate 4 times.
    The pseudo-code for this operation is:

    v = [x0, ..., x15]
    vsub(lane) = v[4*lane:4*lane+3]
    replicated_v(lane) = [vsub(lane), vsub(lane), vsub(lane), vsub(lane)]

    Note that 0<=lane<4

     Parameters
    ----------
    vec : tvm.tir.Expr
         int8x16 vector expression
    lane : int
        vector lane we want to replicate
    dtype_vec : str
        vector data type (e.g., int8x16)

    Returns
    ----------
    output : tvm.tir.Expr
        replicated vector
    """
    # Reinterpret vec_a as 4 int32 words
    vec_int32 = tvm.tir.call_intrin("int32x4", "tir.reinterpret", vec)
    # Broadcast the lane-th word
    vec_int32_shuffled = tvm.tir.Shuffle([vec_int32], [lane, lane, lane, lane])
    # Convert back to uint8x16
    vec_int8_broadcast = tvm.tir.call_intrin(dtype_vec, "tir.reinterpret", vec_int32_shuffled)
    return vec_int8_broadcast


def gemm_acc_4x4_int8_int8_int32(dtype):
    """
    Int8 4x4 matrix multiplication and accumulation using sdot/udot
    instructions. This function takes two arrays of int8 datatype
    -- A[4][4] and B[4][4] and produces a 4x4 matrix
    which is equal to A*B'.

    The pseudo code is as follows.

    .. code-block:: c

        void gemm_acc_4x4_int8_int8_int32(int8 A[4][4], int8 B[4][4], int32 C[4][4]){
            for (int i = 0; i < 4; i++){
                for (int j = 0; j < 4; j++){
                    for (int k = 0; k < 4; k++){
                        C[i][j] += A[i][k] * B[j][k]
                    }
            }
        }

    Notes:
        * The tiling strategy is picked to maximize register usage.

    Parameters
    ----------
    dtype : str, {"uint8", "int8"}
        Whether it works on unsigned int or signed int

    Returns
    -------
    intrin : TensorIntrin
        The Arm TensorIntrin that can be used in tensorizing schedule
    """
    assert dtype in ["uint8", "int8"]
    # This needs to be a variable number of "rows" since TVM
    # "thinks" I only need to compute one row because of
    # padding
    A = te.placeholder((te.var("rows"), 4), dtype, name="A")
    B = te.placeholder((4, 4), dtype, name="B")
    dtype_vec = dtype + "x16"

    k = te.reduce_axis((0, 4), name="k")
    C = te.compute(
        (te.var("rows"), 4),
        lambda i, j: te.sum(A[i, k].astype("int32") * B[j, k].astype("int32"), axis=k),
        name="C",
    )

    aa_buffer = tvm.tir.decl_buffer(
        A.shape, dtype, name="aa_buffer", offset_factor=1, strides=[te.var("sa"), 1]
    )
    bb_buffer = tvm.tir.decl_buffer(
        B.shape, dtype, name="bb_buffer", offset_factor=1, strides=[te.var("sb"), 1]
    )
    cc_buffer = tvm.tir.decl_buffer(
        C.shape, dtype="int32", name="cc_buffer", offset_factor=1, strides=[te.var("sc"), 1]
    )

    llvm_intrin = "llvm.aarch64.neon.sdot" if dtype == "int8" else "llvm.aarch64.neon.udot"

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.tir.ir_builder.create()
            if index == 1:
                for i in range(0, 4):
                    ib.emit(outs[0].vstore([i, 0], tvm.tir.const(0, "int32x4")))
                return ib.get()
            # Load all the elements of tile A.
            # vec_a = [a, b, c, d,
            #          e, f, g, h,
            #          l, m, n, o,
            #          p, q, r, s];
            vec_a = ins[0].vload([0, 0], dtype_vec)

            # Replicate 4 times the i-th row of A. For instance,
            # vec_a[0] = [a, b, c, d,
            #             a, b, c, d,
            #             a, b, c, d,
            #             a, b, c, d,];
            vec_aa = [select_word(vec_a, i, dtype_vec) for i in range(0, 4)]

            # Load all the elements of B. Remember that B
            # is transposed:
            # vec_b = [0, 4, 8, 12,
            #          1, 5, 9, 13,
            #          2, 6, 10, 14,
            #          3, 7, 11, 15,];
            vec_b = ins[1].vload([0, 0], dtype_vec)

            # Execute the dot product
            for i in range(0, 4):
                vec_c = outs[0].vload([i, 0], "int32x4")
                # Compute the product between the i-th row of A
                # and all the rows of B. Remember that sdot/udot
                # subdive the input vectors in 16 elements
                # and then take the dot product among each group.
                # The result is stored in a int32x4 register
                #
                # For instance, for i=0, we have:
                # sdot(vec_aa[0], vec_b) = [a*0+b*4+c*8+d*12,
                #                           a*1+b*5+c*9+d*13,
                #                           a*2+b*6+c*10+d*14,
                #                           a*3+b*7+c*11+d*15]
                vdot = tvm.tir.call_llvm_intrin(
                    "int32x4", llvm_intrin, tvm.tir.const(3, "uint32"), vec_c, vec_b, vec_aa[i]
                )

                # Store the result
                ib.emit(outs[0].vstore([i, 0], vdot))

            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    buffer_params = {"offset_factor": 1}
    return te.decl_tensor_intrin(
        C.op,
        _intrin_func,
        binds={A: aa_buffer, B: bb_buffer, C: cc_buffer},
        default_buffer_params=buffer_params,
    )


def gemm_acc_nx16_int8_int8_int32(dtype, rows):
    """
    Int8 nx16 matrix multiplication and accumulation using sdot/udot instructions
    This function takes two arrays of int8 datatype -- A[n][4] and
    B[4][16] and produces a rowsx16 matrix which is equal to A*B'
    The pseudo code is as follows.

    .. code-block:: c

        void mmla_nx16_int8_int8_int32(int8 A[n][16], int8 B[4][16][4], int32 output[n][16]){
            for (int i = 0; i < n; i++){
                for (int j = 0; j < 16; j++){
                    for (int k = 0; k < 16; k++){
                        out[i][j] += A[i][k] * B[k//4][j][k%4]
                    }
                }
            }
        }

    Notes:
        * The tile size of B is 16x4. Since the reduction variable k moves between 0 and 16
          we need 4 tiles of B to compute a single row of the output. The first 4 values of
          k will be fetched from B[0][j][k], the second batch of 4 from B[1][j][k] and so on
        * The tiling strategy is picked to maximize register usage.

    Parameters
    ----------
    dtype : str, {"uint8", "int8"}
        Whether it works on unsigned int or signed int
    rows : int
        Number of the output rows "n"

    Returns
    -------
    intrin : TensorIntrin
        The Arm TensorIntrin that can be used in tensorizing schedule
    """
    assert dtype in ["uint8", "int8"]
    A = te.placeholder((rows, 16), dtype, name="A")
    B = te.placeholder((4, 16, 4), dtype, name="B")
    dtype_vec = dtype + "x16"
    idxm = tvm.tir.indexmod
    k = te.reduce_axis((0, 16), name="k")
    C = te.compute(
        (rows, 16),
        lambda i, j: te.sum(
            A[i, k].astype("int32") * B[k // 4, j, idxm(k, 4)].astype("int32"), axis=k
        ),
        name="C",
    )

    aa_buffer = tvm.tir.decl_buffer(
        A.shape, dtype, name="aa_buffer", offset_factor=1, strides=[te.var("sa"), 1]
    )
    bb_buffer = tvm.tir.decl_buffer(
        B.shape, dtype, name="bb_buffer", offset_factor=1, strides=[te.var("sb0"), te.var("sb1"), 1]
    )
    cc_buffer = tvm.tir.decl_buffer(
        C.shape, dtype="int32", name="cc_buffer", offset_factor=1, strides=[te.var("sc"), 1]
    )

    llvm_intrin = "llvm.aarch64.neon.sdot" if dtype == "int8" else "llvm.aarch64.neon.udot"

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.tir.ir_builder.create()
            if index == 1:
                for i in range(0, rows):
                    ib.emit(outs[0].vstore([i, 0], tvm.tir.const(0, "int32x16")))
                return ib.get()
            # Iterate on the number of rows of the output
            for k in range(0, rows):
                # Load 16 elements of A
                # vec_a = [a, b, c, d, e, f, g, h, l, m, n, o, p, q, r, s];
                vec_a = ins[0].vload([k, 0], dtype_vec)

                # Iterate over each of the 4 rowsx4 tiles of the output
                for j in range(0, 4):
                    # Accumulate over each of the 4 (16x4) tiles contained in B
                    for i in range(0, 4):
                        # Replicate a single 4-element group of A (A[k, i:i+4])
                        vec_aa = select_word(vec_a, i, dtype_vec)

                        # Load 4 rows (each rows with 4 elements) from B (B[i:i+4, j:j+4])
                        # vec_b = [0, 16, 32, 48,
                        #          1, 17, 33, 49,
                        #          2, 18, 34, 50,
                        #          3, 19, 35, 51,];
                        vec_b = ins[1].vload([i, 4 * j, 0], dtype_vec)

                        # Accumulate in the correct part of the output
                        vec_c = outs[0].vload([k, 4 * j], "int32x4")

                        # Compute the dot product between the rowsx4 tile
                        # from A and the 4x4 tile from B
                        #
                        # For instance, for i=0, we have:
                        # sdot(vec_aa[0], vec_b) = [a*0+b*16+c*32+d*48,
                        #                           a*1+b*17+c*33+d*49,
                        #                           a*2+b*18+c*34+d*50,
                        #                           a*3+b*19+c*35+d*51]
                        vdot = tvm.tir.call_llvm_intrin(
                            "int32x4", llvm_intrin, tvm.tir.const(3, "uint32"), vec_c, vec_b, vec_aa
                        )
                        ib.emit(outs[0].vstore([k, 4 * j], vdot))
            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    buffer_params = {"offset_factor": 1}
    return te.decl_tensor_intrin(
        C.op,
        _intrin_func,
        binds={A: aa_buffer, B: bb_buffer, C: cc_buffer},
        default_buffer_params=buffer_params,
    )


def smlal_int16_int32():
    """
    Intrinsic to be used in order to load two int16x8 vectors and multiply
    them together through a pair of smlal/smlal2 instructions. The pseudo-code
    for the algorithm is as follows:

        vec_a = vload(A, "int16x8")
        vec_b = vload(B, "int16x8")

        vec_c[0:4] += vec_a[0:4]*vec_b[0:4] //  -> smlal instruction
        vec_c[4:8] += vec_a[4:8]*vec_b[4:8] // -> smlal2 instruction

    So we load a single int16x8 vector and we accumulate its lower (0:4) and
    higher part separately.
    """
    int16_lanes = 8
    A = te.placeholder((int16_lanes,), dtype="int16", name="A")
    B = te.placeholder((int16_lanes, 1), dtype="int16", name="B")
    C = te.compute(
        (int16_lanes,), lambda i: A[i].astype("int32") * B[i, 0].astype("int32"), name="C"
    )

    a_buffer = tvm.tir.decl_buffer(
        A.shape, dtype="int16", name="a_buffer", offset_factor=1, strides=[1]
    )
    b_buffer = tvm.tir.decl_buffer(
        B.shape, dtype="int16", name="b_buffer", offset_factor=1, strides=[te.var("sb"), 1]
    )
    c_buffer = tvm.tir.decl_buffer(
        C.shape, dtype="int32", name="c_buffer", offset_factor=1, strides=[1]
    )

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.tir.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore(0, tvm.tir.const(0, "int32x8")))
                return ib.get()

            vec_a = ins[0].vload([0], "int16x8")
            vec_b = ins[1].vload([0, 0], "int16x8")
            inst = "llvm.aarch64.neon.smull"

            # Higher part of the vector
            vec_c_h = outs[0].vload([4], "int32x4")
            vec_a_h = tvm.tir.call_intrin("int16x4", "tir.vectorhigh", vec_a)
            vec_b_h = tvm.tir.call_intrin("int16x4", "tir.vectorhigh", vec_b)
            vmull_h = tvm.tir.call_llvm_pure_intrin(
                "int32x4", inst, tvm.tir.const(2, "uint32"), vec_a_h, vec_b_h
            )
            vec_out_h = vec_c_h + vmull_h

            # Lower part of the vector
            vec_c_l = outs[0].vload([0], "int32x4")
            vec_a_l = tvm.tir.call_intrin("int16x4", "tir.vectorlow", vec_a)
            vec_b_l = tvm.tir.call_intrin("int16x4", "tir.vectorlow", vec_b)
            vmull_l = tvm.tir.call_llvm_pure_intrin(
                "int32x4", inst, tvm.tir.const(2, "uint32"), vec_a_l, vec_b_l
            )
            vec_out_l = vec_c_l + vmull_l

            # Combine higher and lower part in a single int32x8 vector to store
            # (this will require two different store instructions, since the
            # length of a NEON vector is fixed at 128
            vec_out = tvm.tir.call_intrin("int32x8", "tir.vectorcombine", vec_out_l, vec_out_h)
            ib.emit(outs[0].vstore(0, vec_out))
            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    buffer_params = {"offset_factor": 1}
    return te.decl_tensor_intrin(
        C.op,
        _intrin_func,
        binds={A: a_buffer, B: b_buffer, C: c_buffer},
        default_buffer_params=buffer_params,
    )


def gemm_acc_2x2_int8_int8_int32(dtype):
    """
    Int8 2x2 matrix multiplication using smmla/ummla instructions
    This function takes two arrays of int8 datatype -- A[2][8] and
    B[2][8] and produces a 2x2 matrix which is equal to A*B'
    The pseudo code is as follows.

    .. code-block:: c

        void mmla_2x2_int8_int8_int32(int8 A[2][8], int8 B[2][8], int32 C[2][2]){
            for (int i = 0; i < 2; i++){
                for (int j = 0; j < 2; j++){
                    for (int k = 0; k < 8; k++){
                        C[i][j] += A[i][k] * B[j][k]
                    }
            }
        }

    Parameters
    ----------
    dtype : str, {"uint8", "int8"}
        Whether it works on unsigned int or signed int

    Returns
    -------
    intrin : TensorIntrin
        The Arm TensorIntrin that can be used in tensorizing schedule
    """
    assert dtype in ["uint8", "int8"]
    A = te.placeholder((2, 8), dtype, name="A")
    B = te.placeholder((2, 8), dtype, name="B")
    dtype_vec = dtype + "x16"

    k = te.reduce_axis((0, 8), name="k")
    C = te.compute(
        (2, 2),
        lambda i, j: te.sum(A[i, k].astype("int32") * B[j, k].astype("int32"), axis=k),
        name="C",
    )

    aa_buffer = tvm.tir.decl_buffer(
        A.shape, dtype, name="aa_buffer", offset_factor=1, strides=[te.var("sa"), 1]
    )
    bb_buffer = tvm.tir.decl_buffer(
        B.shape, dtype, name="bb_buffer", offset_factor=1, strides=[te.var("sb"), 1]
    )
    cc_buffer = tvm.tir.decl_buffer(
        C.shape, dtype="int32", name="cc_buffer", offset_factor=1, strides=[te.var("sc"), 1]
    )

    llvm_intrin = "llvm.aarch64.neon.smmla" if dtype == "int8" else "llvm.aarch64.neon.ummla"

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.tir.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore([0, 0], tvm.tir.const(0, "int32x4")))
                return ib.get()
            # Load in vec_a the two rows of A
            # vec_a = [a, b, c, d, e, f, g, h;
            #          i, j, k, l, m, n, o, p,]
            vec_a = ins[0].vload([0, 0], dtype_vec)
            # Load in vec_b the two rows of B
            # vec_b = [0, 2, 4, 6, 8, 10, 12, 14;
            #          1, 3, 5, 7, 9, 11, 13, 14,]
            vec_b = ins[1].vload([0, 0], dtype_vec)

            # Execute the matrix multiplication via (s/u)mmla:
            # vec_c = [a*0 + b*2 + c*4 + d*6 +e*8 + f*10 + g*12 + h*14;
            #          a*1 + b*3 + c*5 + d*7 +e*9 + f*11 + g*13 + h*15;
            #          i*0 + j*2 + k*4 + l*6 +m*8 + n*10 + o*12 + p*14;
            #          i*1 + j*3 + k*5 + l*7 +m*9 + n*11 + o*13 + p*15]
            vec_c = outs[0].vload([0, 0], "int32x4")
            vmmla = tvm.tir.call_llvm_intrin(
                "int32x4", llvm_intrin, tvm.tir.const(3, "uint32"), vec_c, vec_a, vec_b
            )
            # Store the result
            ib.emit(outs[0].vstore([0, 0], vmmla))
            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    buffer_params = {"offset_factor": 1}
    return te.decl_tensor_intrin(
        C.op,
        _intrin_func,
        binds={A: aa_buffer, B: bb_buffer, C: cc_buffer},
        default_buffer_params=buffer_params,
    )


def _q_multiply_shift_arm(op):
    """
    Implementation of q_multiply_shift_arm through arm intrinsics
    sqrdmulh and srshl when q == 31.

    Please note that this is introducing a small round-up error for
    some corner cases. This is because we are rounding twice instead
    than only once. I.e.:

        * original q_multiply_shift: round(x*y*2^-s)
        * arm q_multiply_shift: round(round(x*y)*2^-s)
    """
    x = op.args[0]
    y = op.args[1]
    q = op.args[2]
    s = op.args[3]

    # Don't use this intrinsic if we don't have a int32x4 vector
    # or if we are not multiplying q31 numbers
    if x.dtype != "int32x4" or q.value != 31:
        return op

    # Case 1, shift is negative
    sqrdmulh = tvm.tir.call_llvm_intrin(
        op.dtype, "llvm.aarch64.neon.sqrdmulh", tvm.tir.const(2, "uint32"), x, y
    )

    fixup = (sqrdmulh & (-s)) >> 31
    fixed_up_x = sqrdmulh + fixup
    out_1 = tvm.tir.call_llvm_intrin(
        op.dtype, "llvm.aarch64.neon.srshl", tvm.tir.const(2, "uint32"), sqrdmulh, s
    )

    # Case 2, shift is positive
    x = x * (1 << (s))
    out_2 = tvm.tir.call_llvm_intrin(
        op.dtype, "llvm.aarch64.neon.sqrdmulh", tvm.tir.const(2, "uint32"), x, y
    )

    # Select depending on the shift
    return tvm.tir.Select(s < 0, out_1, out_2)


register_intrin_lowering(
    "tir.q_multiply_shift", target="llvm.aarch64", f=_q_multiply_shift_arm, level=99
)
