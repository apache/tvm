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
from tvm.ir.op import register_intrin_lowering


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


def q_multiply_shift(op):
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
    "tir.q_multiply_shift", target="llvm.aarch64", f=q_multiply_shift, level=99
)
