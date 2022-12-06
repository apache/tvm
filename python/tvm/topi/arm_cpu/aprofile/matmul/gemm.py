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
                "int32x4",
                llvm_intrin,
                tvm.tir.const(3, "uint32"),
                vec_c,
                vec_a,
                vec_b,
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
