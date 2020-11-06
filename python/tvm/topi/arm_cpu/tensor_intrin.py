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
from tvm.contrib import utils, clang


def gemm_quantized_4_4_batched():
    return """
           // First half
           // Higher part of a0 * {b0,b1,b2,b3}
           "umull v8.8h, v0.8b, v4.8b\\n"
           "umull v9.8h, v0.8b, v5.8b\\n"
           "umull v10.8h, v0.8b, v6.8b\\n"
           "umull v11.8h, v0.8b, v7.8b\\n"

           // Higher part of a1 * {b0,b1,b2,b3}
           "umull v12.8h, v1.8b, v4.8b\\n"
           "umull v13.8h, v1.8b, v5.8b\\n"
           "umull v14.8h, v1.8b, v6.8b\\n"
           "umull v15.8h, v1.8b, v7.8b\\n"

           // Accumulate
           "uadalp v16.4s, v8.8h\\n"
           "uadalp v17.4s, v9.8h\\n"
           "uadalp v18.4s, v10.8h\\n"
           "uadalp v19.4s, v11.8h\\n"
           "uadalp v20.4s, v12.8h\\n"
           "uadalp v21.4s, v13.8h\\n"
           "uadalp v22.4s, v14.8h\\n"
           "uadalp v23.4s, v15.8h\\n"

           // Lower part of a0 * {b0,b1,b2,b3}
           "umull2 v8.8h, v0.16b, v4.16b\\n"
           "umull2 v9.8h, v0.16b, v5.16b\\n"
           "umull2 v10.8h, v0.16b, v6.16b\\n"
           "umull2 v11.8h, v0.16b, v7.16b\\n"

           // Lower part of a1 * {b0,b1,b2,b3}
           "umull2 v12.8h, v1.16b, v4.16b\\n"
           "umull2 v13.8h, v1.16b, v5.16b\\n"
           "umull2 v14.8h, v1.16b, v6.16b\\n"
           "umull2 v15.8h, v1.16b, v7.16b\\n"

            // Accumulate again
           "uadalp v16.4s, v8.8h\\n"
           "uadalp v17.4s, v9.8h\\n"
           "uadalp v18.4s, v10.8h\\n"
           "uadalp v19.4s, v11.8h\\n"
           "uadalp v20.4s, v12.8h\\n"
           "uadalp v21.4s, v13.8h\\n"
           "uadalp v22.4s, v14.8h\\n"
           "uadalp v23.4s, v15.8h\\n"

           // Second half
           // Lower part of a2 * {b0,b1,b2,b3}
           "umull v8.8h, v2.8b, v4.8b\\n"
           "umull v9.8h, v2.8b, v5.8b\\n"
           "umull v10.8h, v2.8b, v6.8b\\n"
           "umull v11.8h, v2.8b, v7.8b\\n"

           // Lower part of a3 * {b0,b1,b2,b3}
           "umull v12.8h, v3.8b, v4.8b\\n"
           "umull v13.8h, v3.8b, v5.8b\\n"
           "umull v14.8h, v3.8b, v6.8b\\n"
           "umull v15.8h, v3.8b, v7.8b\\n"

           // Accumulate
           "uadalp v24.4s, v8.8h\\n"
           "uadalp v25.4s, v9.8h\\n"
           "uadalp v26.4s, v10.8h\\n"
           "uadalp v27.4s, v11.8h\\n"
           "uadalp v28.4s, v12.8h\\n"
           "uadalp v29.4s, v13.8h\\n"
           "uadalp v30.4s, v14.8h\\n"
           "uadalp v31.4s, v15.8h\\n"

           // Higher part of a2 * {b0,b1,b2,b3}
           "umull2 v8.8h, v2.16b, v4.16b\\n"
           "umull2 v9.8h, v2.16b, v5.16b\\n"
           "umull2 v10.8h, v2.16b, v6.16b\\n"
           "umull2 v11.8h, v2.16b, v7.16b\\n"

           // Higher part of a3 * {b0,b1,b2,b3}
           "umull2 v12.8h, v3.16b, v4.16b\\n"
           "umull2 v13.8h, v3.16b, v5.16b\\n"
           "umull2 v14.8h, v3.16b, v6.16b\\n"
           "umull2 v15.8h, v3.16b, v7.16b\\n"

           // Accumulate again
           "uadalp v24.4s, v8.8h\\n"
           "uadalp v25.4s, v9.8h\\n"
           "uadalp v26.4s, v10.8h\\n"
           "uadalp v27.4s, v11.8h\\n"
           "uadalp v28.4s, v12.8h\\n"
           "uadalp v29.4s, v13.8h\\n"
           "uadalp v30.4s, v14.8h\\n"
           "uadalp v31.4s, v15.8h\\n"
    """


def gemm_quantized_4_4_interleaved():
    return """
             // First half
             // Higher part of a0 * {b0,b1,b2,b3} and accumulate
             "umull v8.8h, v0.8b, v4.8b\\n"
             "uadalp v16.4s, v8.8h\\n"
             "umull v9.8h, v0.8b, v5.8b\\n"
             "uadalp v17.4s, v9.8h\\n"
             "umull v10.8h, v0.8b, v6.8b\\n"
             "uadalp v18.4s, v10.8h\\n"
             "umull v11.8h, v0.8b, v7.8b\\n"
             "uadalp v19.4s, v11.8h\\n"

             // Higher part of a1 * {b0,b1,b2,b3} and accumulate
             "umull v12.8h, v1.8b, v4.8b\\n"
             "uadalp v20.4s, v12.8h\\n"
             "umull v13.8h, v1.8b, v5.8b\\n"
             "uadalp v21.4s, v13.8h\\n"
             "umull v14.8h, v1.8b, v6.8b\\n"
             "uadalp v22.4s, v14.8h\\n"
             "umull v15.8h, v1.8b, v7.8b\\n"
             "uadalp v23.4s, v15.8h\\n"

             // Lower part of a0 * {b0,b1,b2,b3} and accumulate
             "umull2 v8.8h, v0.16b, v4.16b\\n"
             "uadalp v16.4s, v8.8h\\n"
             "umull2 v9.8h, v0.16b, v5.16b\\n"
             "uadalp v17.4s, v9.8h\\n"
             "umull2 v10.8h, v0.16b, v6.16b\\n"
             "uadalp v18.4s, v10.8h\\n"
             "umull2 v11.8h, v0.16b, v7.16b\\n"
             "uadalp v19.4s, v11.8h\\n"

             // Lower part of a1 * {b0,b1,b2,b3} and accumulate
             "umull2 v12.8h, v1.16b, v4.16b\\n"
             "uadalp v20.4s, v12.8h\\n"
             "umull2 v13.8h, v1.16b, v5.16b\\n"
             "uadalp v21.4s, v13.8h\\n"
             "umull2 v14.8h, v1.16b, v6.16b\\n"
             "uadalp v22.4s, v14.8h\\n"
             "umull2 v15.8h, v1.16b, v7.16b\\n"
             "uadalp v23.4s, v15.8h\\n"

             // Second half
             // Higher part of a2 * {b0,b1,b2,b3} and accumulate
             "umull v8.8h, v2.8b, v4.8b\\n"
             "uadalp v24.4s, v8.8h\\n"
             "umull v9.8h, v2.8b, v5.8b\\n"
             "uadalp v25.4s, v9.8h\\n"
             "umull v10.8h, v2.8b, v6.8b\\n"
             "uadalp v26.4s, v10.8h\\n"
             "umull v11.8h, v2.8b, v7.8b\\n"
             "uadalp v27.4s, v11.8h\\n"

             // Higher part of a3 * {b0,b1,b2,b3} and accumulate
             "umull v12.8h, v3.8b, v4.8b\\n"
             "uadalp v28.4s, v12.8h\\n"
             "umull v13.8h, v3.8b, v5.8b\\n"
             "uadalp v29.4s, v13.8h\\n"
             "umull v14.8h, v3.8b, v6.8b\\n"
             "uadalp v30.4s, v14.8h\\n"
             "umull v15.8h, v3.8b, v7.8b\\n"
             "uadalp v31.4s, v15.8h\\n"

             // Lower part of a2 * {b0,b1,b2,b3} and accumulate
             "umull2 v8.8h, v2.16b, v4.16b\\n"
             "uadalp v24.4s, v8.8h\\n"
             "umull2 v9.8h, v2.16b, v5.16b\\n"
             "uadalp v25.4s, v9.8h\\n"
             "umull2 v10.8h, v2.16b, v6.16b\\n"
             "uadalp v26.4s, v10.8h\\n"
             "umull2 v11.8h, v2.16b, v7.16b\\n"
             "uadalp v27.4s, v11.8h\\n"

             // Lower part of a3 * {b0,b1,b2,b3} and accumulate
             "umull2 v12.8h, v3.16b, v4.16b\\n"
             "uadalp v28.4s, v12.8h\\n"
             "umull2 v13.8h, v3.16b, v5.16b\\n"
             "uadalp v29.4s, v13.8h\\n"
             "umull2 v14.8h, v3.16b, v6.16b\\n"
             "uadalp v30.4s, v14.8h\\n"
             "umull2 v15.8h, v3.16b, v7.16b\\n"
             "uadalp v31.4s, v15.8h\\n"
    """


def gemm_quantized_impl(M, N, K, unroll, interleave, data_type="uint8"):
    """Assembly implementation of a blocked gemv. Given
    a block a of shape (4, k) and a block b' of shape (4, k)
    produces the output block c = a*b of shape (4,4)"""

    stepA = min(4, M)
    stepB = min(4, N)
    assert data_type in ["uint8", "int8"], "Only uint8/int8 supported for this implementation"

    signature = """extern "C" int gemm_quantized_{0}_{0}_int32_{1}_{2}""".format(
        data_type, stepA, stepB
    )
    if unroll:
        signature += "_" + str(K)

    if interleave:
        signature += "_interleaved"

    signature += """(int *c_buffer,
                      unsigned char *a_buffer,
                      unsigned char *b_buffer,
                      int K, int m, int n)"""

    cc_code = signature
    cc_code += """
    {
            unsigned char * a_ptr = a_buffer;
            unsigned char * b_ptr = b_buffer;
            int * c_ptr = c_buffer;

            int k = K / 16;

            __asm__  __volatile__ (
                "movi v16.4s, #0\\n"
                "movi v17.4s, #0\\n"
                "movi v18.4s, #0\\n"
                "movi v19.4s, #0\\n"
                "movi v20.4s, #0\\n"
                "movi v21.4s, #0\\n"
                "movi v22.4s, #0\\n"
                "movi v23.4s, #0\\n"
                "movi v24.4s, #0\\n"
                "movi v25.4s, #0\\n"
                "movi v26.4s, #0\\n"
                "movi v27.4s, #0\\n"
                "movi v28.4s, #0\\n"
                "movi v29.4s, #0\\n"
                "movi v30.4s, #0\\n"
                "movi v31.4s, #0\\n"
            "1:"
    """

    main_loop = ' "ldr q0, [%[a_ptr]]\\n" '

    if M > 1:
        main_loop += ' "ldr q1, [%[a_ptr], #16]\\n" '
    else:
        main_loop += ' "movi v1.4s, #0\\n" '

    if M > 2:
        main_loop += ' "ldr q2, [%[a_ptr], #32]\\n" '
    else:
        main_loop += ' "movi v2.4s, #0\\n" '

    if M > 3:
        main_loop += ' "ldr q3, [%[a_ptr], #48]\\n" '
    else:
        main_loop += ' "movi v3.4s, #0\\n" '

    main_loop += ' "ldr q4, [%[b_ptr]]\\n" '

    if N > 1:
        main_loop += ' "ldr q5, [%[b_ptr], #16]\\n" '

    if N > 2:
        main_loop += ' "ldr q6, [%[b_ptr], #32]\\n" '

    if N > 3:
        main_loop += ' "ldr q7, [%[b_ptr], #48]\\n" '

    # Main computation can interleave multiply/accumulate instructions
    # or schedule them in batches (first all multiplies then all accumulates)
    if interleave:
        main_loop += gemm_quantized_4_4_interleaved()
    else:
        main_loop += gemm_quantized_4_4_batched()

    blockA = min(64, M * 16)
    blockB = min(64, N * 16)
    main_loop += """// Increment pointers
                    "add %[a_ptr], %[a_ptr], #{0}\\n"
                    "add %[b_ptr], %[b_ptr], #{1}\\n" """.format(
        blockA, blockB
    )

    if unroll:
        k = int(K // 16)
        for l in range(0, k):
            cc_code += main_loop
    else:
        cc_code += main_loop
        cc_code += """
                    "subs %w[k], %w[k], #1\\n"
                    "cbnz %w[k], 1b\\n"
                   """
    cc_code += """
                // Final additions

                // v16 contains the four partial sums of a[0, 0:K].*b[0,0:K], let's call them (a,b,c,d)
                // v17 contains the four partial sums of a[0, 0:K].*b[1,0:K], let's call them (e,f,g,h)
                // v18 contains the four partial sums of a[0, 0:K].*b[2,0:K], let's call them (i,j,k,l)
                // v19 contains the four partial sums of a[0, 0:K].*b[3,0:K], let's call them (m,n,o,p)
                "addp v16.4s, v16.4s, v17.4s\\n" // v16 = (a+b, c+d, e+f, g+h)
                "addp v17.4s, v18.4s, v19.4s\\n" // v17 = (i+j, k+l, m+n, o+p)
                "addp v16.4s, v16.4s, v17.4s\\n" // v16 = (a+b+c+d, e+f+g+h, i+j+k+l, m+n+o+p)

                // v20 contains the four partial sums of a[1, 0:K].*b[0,0:K], let's call them (a,b,c,d)
                // v21 contains the four partial sums of a[1, 0:K].*b[1,0:K], let's call them (e,f,g,h)
                // v22 contains the four partial sums of a[1, 0:K].*b[2,0:K], let's call them (i,j,k,l)
                // v23 contains the four partial sums of a[1, 0:K].*b[3,0:K], let's call them (m,n,o,p)
                "addp v20.4s, v20.4s, v21.4s\\n" // v20 = (a+b, c+d, e+f, g+h)
                "addp v21.4s, v22.4s, v23.4s\\n" // v21 = (i+j, k+l, m+n, o+p)
                "addp v20.4s, v20.4s, v21.4s\\n" // v20 = (a+b+c+d, e+f+g+h, i+j+k+l, m+n+o+p)

                // v24 contains the four partial sums of a[2, 0:K].*b[0,0:K], let's call them (a,b,c,d)
                // v25 contains the four partial sums of a[2, 0:K].*b[1,0:K], let's call them (e,f,g,h)
                // v26 contains the four partial sums of a[2, 0:K].*b[2,0:K], let's call them (i,j,k,l)
                // v27 contains the four partial sums of a[2, 0:K].*b[3,0:K], let's call them (m,n,o,p)
                "addp v24.4s, v24.4s, v25.4s\\n"  // v24 = (a+b, c+d, e+f, g+h)
                "addp v25.4s, v26.4s, v27.4s\\n"  // v25 = (i+j, k+l, m+n, o+p)
                "addp v24.4s, v24.4s, v25.4s\\n"  // v24 = (a+b+c+d, e+f+g+h, i+j+k+l, m+n+o+p)

                // v28 contains the four partial sums of a[3, 0:K].*b[0,0:K], let's call them (a,b,c,d)
                // v29 contains the four partial sums of a[3, 0:K].*b[1,0:K], let's call them (e,f,g,h)
                // v30 contains the four partial sums of a[3, 0:K].*b[2,0:K], let's call them (i,j,k,l)
                // v31 contains the four partial sums of a[3, 0:K].*b[3,0:K], let's call them (m,n,o,p)
                "addp v28.4s, v28.4s, v29.4s\\n" // v28 = (a+b, c+d, e+f, g+h)
                "addp v29.4s, v30.4s, v31.4s\\n" // v29 = (i+j, k+l, m+n, o+p)
                "addp v28.4s, v28.4s, v29.4s\\n" // v28 = (a+b+c+d, e+f+g+h, i+j+k+l, m+n+o+p)

                "str q16, [%[c_ptr]]\\n"
            """

    stepC = min(4, N)
    if M > 1:
        cc_code += ' "str q20, [%[c_ptr], #{0}]\\n" '.format(stepC * 4)

    if M > 2:
        cc_code += ' "str q24, [%[c_ptr], #{0}]\\n" '.format(stepC * 8)

    if M > 3:
        cc_code += ' "str q28, [%[c_ptr], #{0}]\\n" '.format(stepC * 12)

    cc_code += """
             : [c_ptr] "+r" (c_ptr), [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [k] "+r" (k)
             :
             : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                    "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
                    "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
                    "v27", "v28", "v29", "v30", "v31"
             );
        return 0;
        }
    """

    if data_type == "int8":
        cc_code = cc_code.replace("unsigned char", "char")
        cc_code = cc_code.replace("umull", "smull")
        cc_code = cc_code.replace("uadalp", "sadalp")

    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    # Create LLVM ir from c source code
    ll_code = clang.create_llvm(
        cc_code, options=["--target=aarch64-linux-gnu -mattr=+neon"], output=ll_path
    )
    return ll_code


def gemm_quantized(M, N, K, unroll, interleave, in_type, out_type):
    """
    Use integer ARM v8 instructions in order to produce a block c of 4x4 elements
    given two 4xK blocks a and b' (where b' is a Kx4 block transposed). The final
    result is c = a*b (where '*' indicates the matrix product)

    Every row of the matrix c is obtained (for uint8) by a sequence of

          umull -> uadalp -> umull2 -> uadalp

    The block size is constrained by the number of registers available in arvm8. This
    function returns a TensorIntrin that can be used to tensorize
    a schedule.

    Parameters
    ----------
    M: int
        rows of the matrix A
    N: int
        columns of the matrix B
    K: int
        columns of matrix A
    in_type: str, {'uint8', 'int8'}
    out_type: str, {'uint32', 'int32'}

    Returns
    -------
    intrin : TensorIntrin
        The ARM uint8/int8 TensorIntrin that can be used in tensorizing schedule
    """
    A = te.placeholder((K // 16, te.var("m"), 16), dtype=in_type, name="A")
    B = te.placeholder((K // 16, te.var("n"), 16), dtype=in_type, name="B")

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

    def _intrin_func(ins, outs):
        def _instr():
            ib = tvm.tir.ir_builder.create()
            aa, bb = ins
            cc = outs[0]
            stepA = min(4, M)
            stepB = min(4, N)
            intrin_name = "gemm_quantized_{0}_{0}_int32_{1}_{2}".format(in_type, stepA, stepB)
            if unroll:
                intrin_name += "_" + str(K)
            if interleave:
                intrin_name += "_interleaved"
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    intrin_name,
                    outs[0].access_ptr("w"),
                    a_buffer.access_ptr("r"),
                    b_buffer.access_ptr("r"),
                    K,
                )
            )
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


def dot_int8_int8_int32(int32_lanes, dtype="uint"):
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
    int32_lanes: int
        How many int32/uint32 to produce
    dtype: str, optional, {"uint", "int"}
        Whether it works on unsigned int or signed int

    Returns
    -------
    intrin : TensorIntrin
        The ARM uint8 TensorIntrin that can be used in tensorizing schedule
    """
    num_int8_elements = 4  # 4 int8 elements in int32

    data = te.placeholder((num_int8_elements,), dtype="%s8" % dtype, name="data")
    kernel = te.placeholder((int32_lanes, num_int8_elements), dtype="%s8" % dtype, name="kernel")

    k = te.reduce_axis((0, num_int8_elements), name="k")
    C = te.compute(
        (int32_lanes,),
        lambda i: te.sum(
            data[k].astype("%s32" % dtype) * kernel[i, k].astype("%s32" % dtype), axis=k
        ),
        name="C",
    )

    a_buffer = tvm.tir.decl_buffer(
        data.shape, dtype="%s8" % dtype, name="a_buffer", offset_factor=1, strides=[1]
    )
    b_buffer = tvm.tir.decl_buffer(
        kernel.shape,
        dtype="%s8" % dtype,
        name="b_buffer",
        offset_factor=1,
        strides=[te.var("s"), 1],
    )

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.tir.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore(0, tvm.tir.const(0, "%s32x%d" % (dtype, int32_lanes))))
                return ib.get()

            dtype_a = "%s8x%d" % (dtype, num_int8_elements)
            dtype_b = "%s8x%d" % (dtype, int32_lanes * num_int8_elements)
            dtype_c = "%s32x%d" % (dtype, int32_lanes)

            a_int8 = ins[0].vload([0], dtype_a)
            re_int32 = tvm.tir.call_intrin("%s32" % dtype, "tir.reinterpret", a_int8)
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
                dtype_c, inst, tvm.tir.const(2, "uint32"), vec_c, vec_a, vec_b
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
    vec: tvm.tir.Expr
         int8x16 vector expression
    lane: int
        vector lane we want to replicate
    dtype_vec: str
        vector data type (e.g., int8x16)

    Returns
    ----------
    output: tvm.tir.Expr
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
    which is equal to A*B.

    The pseudo code is as follows.

    .. code-block:: c

        void gemm_acc_4x4_int8_int8_int32(int8 A[4][4], int8 B[4][4], int32 C[4][4]){
            for (int i = 0; i < 4; i++){
                for (int j = 0; i < 4; i++){
                    for (int k = 0; k < 4; k++){
                        C[i][j] += A[i][k] * B[j][k]
                    }
            }
        }

    Notes:
        * The rows of matrix B are transposed
        * The tiling strategy is picked to maximize register usage.

    Parameters
    ----------
    dtype: str, {"uint8", "int8"}
        Whether it works on unsigned int or signed int

    Returns
    -------
    intrin : TensorIntrin
        The Arm TensorIntrin that can be used in tensorizing schedule
    """
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
                    "int32x4",
                    llvm_intrin,
                    tvm.tir.const(3, "uint32"),
                    vec_c,
                    vec_b,
                    vec_aa[i],
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
    B[4][16] and produces a rowsx16 matrix which is equal to A*B
    The pseudo code is as follows.

    .. code-block:: c

        void mmla_nx16_int8_int8_int32(int8 A[n][16], int8 B[4][16][4], int32 output[n][16]){
            for (int i = 0; i < n; i++){
                for (int j = 0; i < 16; i++){
                    for (int k = 0; k < 16; k++){
                        out[i][j] += A[i][k] * B[k//4][j][k%4]
                    }
                }
            }
        }

    Notes:
        * The rows of matrix B are transposed
        * The tile size of B is 16x4. Since the reduction variable k moves between 0 and 16
          we need 4 tiles of B to compute a single row of the output. The first 4 values of
          k will be fetched from B[0][j][k], the second batch of 4 from B[1][j][k] and so on
        * The tiling strategy is picked to maximize register usage.

    Parameters
    ----------
    dtype: str, {"uint8", "int8"}
        Whether it works on unsigned int or signed int
    rows: int
        Number of of the output rows "n"

    Returns
    -------
    intrin : TensorIntrin
        The Arm TensorIntrin that can be used in tensorizing schedule
    """
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
        B.shape,
        dtype,
        name="bb_buffer",
        offset_factor=1,
        strides=[te.var("sb0"), te.var("sb1"), 1],
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
                            "int32x4",
                            llvm_intrin,
                            tvm.tir.const(3, "uint32"),
                            vec_c,
                            vec_b,
                            vec_aa,
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
        (int16_lanes,),
        lambda i: A[i].astype("int32") * B[i, 0].astype("int32"),
        name="C",
    )

    a_buffer = tvm.tir.decl_buffer(
        A.shape, dtype="int16", name="a_buffer", offset_factor=1, strides=[1]
    )
    b_buffer = tvm.tir.decl_buffer(
        B.shape,
        dtype="int16",
        name="b_buffer",
        offset_factor=1,
        strides=[te.var("sb"), 1],
    )
    c_buffer = tvm.tir.decl_buffer(
        C.shape,
        dtype="int32",
        name="c_buffer",
        offset_factor=1,
        strides=[1],
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


tvm.target.intrin.register_intrin_rule(
    "llvm.aarch64", "q_multiply_shift", _q_multiply_shift_arm, override=True
)
