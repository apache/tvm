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
# pylint: disable=invalid-name
"""Optimized implementation of q_multiply_shift based on LLVM intrinsics"""

import tvm
from tvm.ir import register_intrin_lowering
from tvm import te


def _q_multiply_shift_hexagon(op):
    """
    Implementation of q_multiply_shift through hexagon intrinsics vmpyewuh and vmpyowh when q == 31.

    Please note that this is introducing a small round-up error for some corner cases with negative
    shift argument. This is because we are rounding twice instead than only once. I.e.:

        * original q_multiply_shift: round(x*y*2^-s)
        * hexagon q_multiply_shift: round(round(x*y)*2^-s)
    """
    x = op.args[0]
    y = op.args[1]
    fractional_bits = op.args[2]
    shift = op.args[3]

    # Don't use this intrinsic if we don't have a int32x32 vector
    # or if we are not multiplying q31 numbers
    if x.dtype != "int32x32" or fractional_bits.value != 31:
        return op

    # Case 1, shift is negative
    mul_e_1 = tvm.tir.call_llvm_intrin(
        op.dtype, "llvm.hexagon.V6.vmpyewuh.128B", tvm.tir.const(2, "uint32"), x, y
    )
    mul_o_1 = tvm.tir.call_llvm_intrin(
        op.dtype, "llvm.hexagon.V6.vmpyowh.rnd.sacc.128B", tvm.tir.const(3, "uint32"), mul_e_1, x, y
    )
    fixup = mul_o_1 & (-shift)
    round_mul = mul_o_1 + fixup
    out_negative_shift = tvm.tir.call_llvm_intrin(
        op.dtype, "llvm.hexagon.V6.vaslwv.128B", tvm.tir.const(2, "uint32"), round_mul, shift
    )

    # Case 2, shift is positive
    x = x * (1 << (shift))
    mul_e_2 = tvm.tir.call_llvm_intrin(
        op.dtype, "llvm.hexagon.V6.vmpyewuh.128B", tvm.tir.const(2, "uint32"), x, y
    )
    mul_o_2 = tvm.tir.call_llvm_intrin(
        op.dtype, "llvm.hexagon.V6.vmpyowh.rnd.sacc.128B", tvm.tir.const(3, "uint32"), mul_e_2, x, y
    )

    # Select depending on the shift
    return tvm.tir.Select(shift < 0, out_negative_shift, mul_o_2)


register_intrin_lowering(
    "tir.q_multiply_shift", target="hexagon", f=_q_multiply_shift_hexagon, level=99
)


def dot_vrmpy(x_ty, y_ty):
    """Generates vrmpy instruciton for tensorization."""
    int32_lanes = 32
    num_int8_elements = 4  # 4 int8 elements in int32
    data = te.placeholder((num_int8_elements,), dtype=x_ty, name="data")
    kernel = te.placeholder((int32_lanes, num_int8_elements), dtype=y_ty, name="kernel")
    k = te.reduce_axis((0, num_int8_elements), name="k")
    C = te.compute(
        (int32_lanes,),
        lambda i: te.sum(data[k].astype("int32") * kernel[i, k].astype("int32"), axis=k),
        name="C",
    )

    a_buffer = tvm.tir.decl_buffer(
        data.shape, dtype=x_ty, name="a_buffer", offset_factor=1, strides=[1]
    )
    b_buffer = tvm.tir.decl_buffer(
        kernel.shape, dtype=y_ty, name="b_buffer", offset_factor=1, strides=[te.var("ldw"), 1]
    )

    def _intrin_func(ins, outs):
        def _instr(index):
            ib = tvm.tir.ir_builder.create()
            if index == 1:
                ib.emit(outs[0].vstore(0, tvm.tir.const(0, "int32x32")))
                return ib.get()

            vec_zero = tvm.tir.const(0, "int32x32")

            if x_ty == "uint8" and y_ty == "uint8":
                a_uint8 = ins[0].vload([0], "uint8x4")
                re_int32 = tvm.tir.call_intrin("int32", "tir.reinterpret", a_uint8)
                vec_b = ins[1].vload([0, 0], "uint8x128")

                vrmpy_inst_name = "llvm.hexagon.V6.vrmpyub.acc.128B"

                vec_bi32 = tvm.tir.call_intrin("int32x32", "tir.reinterpret", vec_b)

                quad_reduction = tvm.tir.call_llvm_pure_intrin(
                    "int32x32",
                    vrmpy_inst_name,
                    tvm.tir.const(3, "uint32"),
                    vec_zero,
                    vec_bi32,
                    re_int32,
                )
            elif x_ty == "uint8" and y_ty == "int8":
                a_uint8 = ins[0].vload([0], "uint8x4")
                re_int32 = tvm.tir.call_intrin("int32", "tir.reinterpret", a_uint8)
                vec_b = ins[1].vload([0, 0], "int8x128")

                vrmpy_inst_name = "llvm.hexagon.V6.vrmpybusv.acc.128B"

                vec_bi32 = tvm.tir.call_intrin("int32x32", "tir.reinterpret", vec_b)

                quad_reduction = tvm.tir.call_llvm_pure_intrin(
                    "int32x32",
                    vrmpy_inst_name,
                    tvm.tir.const(3, "uint32"),
                    vec_zero,
                    re_int32.astype("int32x32"),
                    vec_bi32,
                )
            else:
                raise ValueError(f"Only (u8, u8) or (u8, i8) dtype pairs are supported by vrmpy.")

            if index == 0:
                ib.emit(outs[0].vstore(0, quad_reduction))
            else:
                ib.emit(outs[0].vstore(0, quad_reduction + outs[0].vload([0], "int32x32")))
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
