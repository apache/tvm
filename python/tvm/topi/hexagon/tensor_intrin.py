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
"""Optimized implementation of q_multiply_shift based on LLVM intrinsics"""

import tvm
from tvm.ir import register_intrin_lowering


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
