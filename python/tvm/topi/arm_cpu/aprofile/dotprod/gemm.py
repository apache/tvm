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

import tvm
from tvm import te


def _select_word(vec, lane, dtype_vec):
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
            vec_aa = [_select_word(vec_a, i, dtype_vec) for i in range(0, 4)]

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
        Number of of the output rows "n"

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
                        vec_aa = _select_word(vec_a, i, dtype_vec)

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
