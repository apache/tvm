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
from tvm.contrib import util, clang

def gemv_quantized_impl(M, N, data_type='uint8'):
    """ Assembly implementation of a blocked gemv. Given
    a block a of shape (4, k) and a block b' of shape (4, k)
    produces the output block c = a*b of shape (4,4) """

    stepA = min(4, M)
    stepB = min(4, N)
    assert data_type in ['uint8', 'int8'], 'Only uint8/int8 supported for this implementation'

    cc_code = """
          extern "C" int gemv_{0}_{0}_int32_{1}_{2}(int *c_buffer,
                                                    unsigned char *a_buffer,
                                                    unsigned char *b_buffer,
                                                    int K, int m, int n)
              """.format(data_type, stepA, stepB)

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

    cc_code += ' "ldr q0, [%[a_ptr]]\\n" '

    if M > 1:
        cc_code += ' "ldr q1, [%[a_ptr], #16]\\n" '
    else:
        cc_code += ' "movi v1.4s, #0\\n" '

    if M > 2:
        cc_code += ' "ldr q2, [%[a_ptr], #32]\\n" '
    else:
        cc_code += ' "movi v2.4s, #0\\n" '

    if M > 3:
        cc_code += ' "ldr q3, [%[a_ptr], #48]\\n" '
    else:
        cc_code += ' "movi v3.4s, #0\\n" '

    cc_code += ' "ldr q4, [%[b_ptr]]\\n" '

    if N > 1:
        cc_code += ' "ldr q5, [%[b_ptr], #16]\\n" '

    if N > 2:
        cc_code += ' "ldr q6, [%[b_ptr], #32]\\n" '

    if N > 3:
        cc_code += ' "ldr q7, [%[b_ptr], #48]\\n" '

    cc_code += """
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
    blockA = min(64, M * 16)
    blockB = min(64, N * 16)

    cc_code += """
                // Increment pointers and decrement k
                "add %[a_ptr], %[a_ptr], #{0}\\n"
                "add %[b_ptr], %[b_ptr], #{1}\\n"
                "subs %w[k], %w[k], #1\\n"
    """.format(blockA, blockB)

    stepC = min(4, N)

    cc_code += """
                "cbnz %w[k], 1b\\n"

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

    if data_type == 'int8':
        cc_code = cc_code.replace('unsigned char', 'char')
        cc_code = cc_code.replace('umull', 'smull')
        cc_code = cc_code.replace('uadalp', 'sadalp')

    temp = util.tempdir()
    ll_path = temp.relpath("temp.ll")
    # Create LLVM ir from c source code
    ll_code = clang.create_llvm(cc_code,
                                options=["--target=aarch64-linux-gnu -mattr=+neon"],
                                output=ll_path)
    return ll_code


def gemv_quantized(M, N, K, in_type, out_type):
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
    A = te.placeholder((K // 16, te.var("m"), 16), dtype=in_type, name='A')
    B = te.placeholder((K // 16, te.var("n"), 16), dtype=in_type, name='B')

    idxm = tvm.tir.indexmod

    k = te.reduce_axis((0, K), "k")

    C = te.compute((te.var("m"), te.var("n")),
                   lambda x, y: te.sum(A[k // 16, x, idxm(k, 16)].astype(out_type) *
                                       B[k // 16, y, idxm(k, 16)].astype(out_type),
                                       axis=k), name="C")

    a_buffer = tvm.tir.decl_buffer(A.shape, dtype=in_type, name="a_buffer",
                                   offset_factor=1, strides=[te.var('sa_1'), te.var('sa_2'), 1])

    b_buffer = tvm.tir.decl_buffer(B.shape, dtype=in_type, name="b_buffer",
                                   offset_factor=1, strides=[te.var('sb_1'), te.var('sb_2'), 1])

    c_buffer = tvm.tir.decl_buffer(C.shape, dtype=out_type, name="c_buffer",
                                   offset_factor=1, strides=[te.var('sc'), 1])

    def _intrin_func(ins, outs):

        def _instr():
            ib = tvm.tir.ir_builder.create()
            aa, bb = ins
            cc = outs[0]
            stepA = min(4, M)
            stepB = min(4, N)

            if in_type == 'int8':
                ib.emit(tvm.tir.call_extern("int32",
                                            "gemv_int8_int8_int32_{0}_{1}".format(stepA, stepB),
                                            outs[0].access_ptr("w"),
                                            a_buffer.access_ptr("r"),
                                            b_buffer.access_ptr("r"),
                                            K))
            else:
                ib.emit(tvm.tir.call_extern("int32",
                                            "gemv_uint8_uint8_int32_{0}_{1}".format(stepA, stepB),
                                            c_buffer.access_ptr("w"),
                                            a_buffer.access_ptr("r"),
                                            b_buffer.access_ptr("r"),
                                            K,
                                            C.shape[0],  # m, very useful for debug
                                            C.shape[1]))  # n, very useful for debug
            return ib.get()

        # body, reset, update
        return _instr()

    buffer_params = {"offset_factor": 1}
    return te.decl_tensor_intrin(C.op, _intrin_func,
                                 binds={A:a_buffer, B:b_buffer, C:c_buffer},
                                 default_buffer_params=buffer_params)


def dot_int8_int8_int32(int32_lanes, dtype='uint'):
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

    data = te.placeholder((num_int8_elements,), dtype='%s8' % dtype, name='data')
    kernel = te.placeholder((int32_lanes, num_int8_elements), dtype='%s8' % dtype, name='kernel')

    k = te.reduce_axis((0, num_int8_elements), name='k')
    C = te.compute((int32_lanes,),
                   lambda i: te.sum(data[k].astype('%s32' % dtype) *
                                    kernel[i, k].astype('%s32' % dtype),
                                    axis=k), name="C")

    a_buffer = tvm.tir.decl_buffer(data.shape, dtype='%s8' % dtype, name="a_buffer",
                                   offset_factor=1,
                                   strides=[1])
    b_buffer = tvm.tir.decl_buffer(kernel.shape, dtype='%s8' % dtype, name="b_buffer",
                                   offset_factor=1,
                                   strides=[te.var('s'), 1])

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.tir.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore(0, tvm.tir.const(0, '%s32x%d' % (dtype, int32_lanes))))
                return ib.get()

            dtype_a = '%s8x%d' % (dtype, num_int8_elements)
            dtype_b = '%s8x%d' % (dtype, int32_lanes * num_int8_elements)
            dtype_c = '%s32x%d' % (dtype, int32_lanes)

            a_int8 = ins[0].vload([0], dtype_a)
            re_int32 = tvm.tir.call_intrin('%s32' % dtype, 'tir.reinterpret', a_int8)
            # broadcast a
            vec_ai32 = re_int32.astype(dtype_c)

            vec_a = tvm.tir.call_intrin(dtype_b, 'tir.reinterpret', vec_ai32)
            vec_b = ins[1].vload([0, 0], dtype_b)
            vec_c = outs[0].vload([0], dtype_c)

            inst = 'udot' if dtype == 'uint' else 'sdot'
            inst = 'llvm.aarch64.neon.%s.v%di32.v%di8' % (
                inst, int32_lanes, int32_lanes * num_int8_elements)
            vdot = tvm.tir.call_llvm_pure_intrin(
                dtype_c,
                inst,
                tvm.tir.const(2, 'uint32'),
                vec_c, vec_a, vec_b)
            ib.emit(outs[0].vstore(0, vdot))
            return ib.get()

        # body, reset, update
        return _instr(0), _instr(1), _instr(2)

    buffer_params = {"offset_factor": 1}
    return te.decl_tensor_intrin(
        C.op, _intrin_func, binds={data:a_buffer, kernel:b_buffer},
        default_buffer_params=buffer_params)
