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


def get_lanes(dtype: str):
    if "x" not in dtype:
        return 1

    _, lanes = dtype.split("x")
    return int(lanes)


def is_vector_type(dtype: str):
    return get_lanes(dtype) != 1


def is_power_of_2(n: int):
    return (n & (n - 1) == 0) and n != 0


def _adapt_to_highest_lanes(*args, intrinsic=None, intrinsic_lanes: int = 0):
    """Apply provided lowering intrinsic to arguments with longer vector data type.

    This wrapper will do next actions:
      * Split each argument into chunks with size equal intrinsic_lanes
      * Apply provided intrinsic for each argument chunk
      * Concatenate results

    Parameters
    ----------
    args: List[PrimExpr]
        List of arguments. Each arg expression should have vector type with lanes
        equal `intrinsic_lanes * 2**n`.

    intrinsic: callable
        Intrinsic implementation to apply.

    intrinsic_lanes: int
        Vector length required by intrinsic implementation.

    Returns
    -------
    res : PrimExpr
        Resulting expression.
    """

    def split_args(args_set):
        res_args_set = []
        for args_chunk in args_set:
            res_args_chunk_l = []
            res_args_chunk_h = []
            for arg_chunk in args_chunk:
                element, lanes = arg_chunk.dtype.split("x")
                res_arg_chunk_dtype = f"{element}x{int(lanes) // 2}"

                res_args_chunk_l.append(tvm.tir.op.vectorlow(res_arg_chunk_dtype, arg_chunk))
                res_args_chunk_h.append(tvm.tir.op.vectorhigh(res_arg_chunk_dtype, arg_chunk))
            res_args_set += [res_args_chunk_l, res_args_chunk_h]

        return res_args_set

    def concat_args(res_chunks):
        merged_res_chunks = []
        for i in range(0, len(res_chunks), 2):
            arg_chunk_l = res_chunks[i]
            arg_chunk_h = res_chunks[i + 1]
            element, lanes = arg_chunk_l.dtype.split("x")
            res_arg_chunk_dtype = f"{element}x{int(lanes) * 2}"

            merged_res_chunks.append(
                tvm.tir.op.vectorcombine(res_arg_chunk_dtype, arg_chunk_l, arg_chunk_h)
            )

        return merged_res_chunks

    num_chunks = None
    for arg in args:
        _, lanes = arg.dtype.split("x")
        lanes = int(lanes)
        assert lanes % intrinsic_lanes == 0
        if num_chunks is None:
            assert is_power_of_2(lanes // intrinsic_lanes)
            num_chunks = lanes // intrinsic_lanes

        assert num_chunks == lanes // intrinsic_lanes

    # Split arguments
    lowered_args = [args]
    while len(lowered_args) != num_chunks:
        lowered_args = split_args(lowered_args)

    # Intrinsic application
    lowered_res = []
    for l_arg in lowered_args:
        res = intrinsic(*l_arg)
        lowered_res.append(res)

    # Result concatenation
    while len(lowered_res) != 1:
        lowered_res = concat_args(lowered_res)

    return lowered_res[0]


def _q_multiply_shift_hexagon(op):
    """
    Implementation of q_multiply_shift through hexagon intrinsics vmpyewuh and vmpyowh when q == 31.
    """
    arg_x = op.args[0]
    arg_fractional_bits = op.args[2]

    # Don't use this intrinsic if we are not multiplying q31 numbers
    if arg_fractional_bits.value != 31:
        return op

    x_lanes = get_lanes(arg_x.dtype)
    if x_lanes % 32 != 0 or not is_power_of_2(x_lanes // 32):
        return op

    # pylint: disable=unused-argument
    def intrinsic_lowering_32(x, y, fractional_bits, shift):
        lowered_dtype = "int32x32"

        # Case 1, shift is negative
        mul_e_1 = tvm.tir.call_llvm_intrin(
            lowered_dtype, "llvm.hexagon.V6.vmpyewuh.128B", tvm.tir.const(2, "uint32"), x, y
        )
        mul_o_1 = tvm.tir.call_llvm_intrin(
            lowered_dtype,
            "llvm.hexagon.V6.vmpyowh.sacc.128B",
            tvm.tir.const(3, "uint32"),
            mul_e_1,
            x,
            y,
        )
        fixup = 1 << (-shift - 1)
        round_mul = mul_o_1 + fixup
        out_negative_shift = tvm.tir.call_llvm_intrin(
            lowered_dtype,
            "llvm.hexagon.V6.vaslwv.128B",
            tvm.tir.const(2, "uint32"),
            round_mul,
            shift,
        )

        # Case 2, shift is positive
        x = x * (1 << (shift))
        mul_e_2 = tvm.tir.call_llvm_intrin(
            lowered_dtype, "llvm.hexagon.V6.vmpyewuh.128B", tvm.tir.const(2, "uint32"), x, y
        )
        mul_o_2 = tvm.tir.call_llvm_intrin(
            lowered_dtype,
            "llvm.hexagon.V6.vmpyowh.rnd.sacc.128B",
            tvm.tir.const(3, "uint32"),
            mul_e_2,
            x,
            y,
        )

        # Select depending on the shift
        return tvm.tir.Select(shift < 0, out_negative_shift, mul_o_2)

    return _adapt_to_highest_lanes(*op.args, intrinsic=intrinsic_lowering_32, intrinsic_lanes=32)


register_intrin_lowering(
    "tir.q_multiply_shift", target="hexagon", f=_q_multiply_shift_hexagon, level=99
)


def _q_multiply_shift_per_axis_hexagon(op):
    """
    Implementation of q_multiply_shift_per_axis through hexagon intrinsics vmpyewuh and vmpyowh when
    q == 31.
    """
    arg_x = op.args[0]
    arg_fractional_bits = op.args[4]
    arg_is_lshift_required = op.args[5]
    arg_is_rshift_required = op.args[6]

    # Don't use this intrinsic if we are not multiplying q31 numbers
    if arg_fractional_bits.value != 31:
        return op

    x_lanes = get_lanes(arg_x.dtype)
    if x_lanes % 32 != 0 or not is_power_of_2(x_lanes // 32):
        return op

    # Don't use this intrinsic when we need do both: left and right shifts.
    # For now it is not clear how to implement this case through vector HVX instructions without
    # accuracy drop.
    if arg_is_rshift_required.value and arg_is_lshift_required.value:
        return op

    # pylint: disable=unused-argument
    def intrinsic_impl_32(
        x, y, left_shift, right_shift, fractional_bits, is_lshift_required, is_rshift_required
    ):
        lowered_dtype = "int32x32"

        # Case 1: do the left shift
        shifted_x = x << left_shift
        mul_e_1 = tvm.tir.call_llvm_intrin(
            lowered_dtype, "llvm.hexagon.V6.vmpyewuh.128B", tvm.tir.const(2, "uint32"), shifted_x, y
        )
        left_shift_out = tvm.tir.call_llvm_intrin(
            lowered_dtype,
            "llvm.hexagon.V6.vmpyowh.rnd.sacc.128B",
            tvm.tir.const(3, "uint32"),
            mul_e_1,
            shifted_x,
            y,
        )

        # Case 2: do the right shift
        mul_e_2 = tvm.tir.call_llvm_intrin(
            lowered_dtype, "llvm.hexagon.V6.vmpyewuh.128B", tvm.tir.const(2, "uint32"), x, y
        )
        mul_o_2 = tvm.tir.call_llvm_intrin(
            lowered_dtype,
            "llvm.hexagon.V6.vmpyowh.sacc.128B",
            tvm.tir.const(3, "uint32"),
            mul_e_2,
            x,
            y,
        )
        fixup = 1 << (right_shift - 1)
        round_mul = mul_o_2 + fixup
        right_shift_out = tvm.tir.call_llvm_intrin(
            lowered_dtype,
            "llvm.hexagon.V6.vasrwv.128B",
            tvm.tir.const(2, "uint32"),
            round_mul,
            right_shift,
        )

        # Case 3: do neither right nor left shift
        mul_e_3 = tvm.tir.call_llvm_intrin(
            lowered_dtype, "llvm.hexagon.V6.vmpyewuh.128B", tvm.tir.const(2, "uint32"), x, y
        )
        no_shift_out = tvm.tir.call_llvm_intrin(
            lowered_dtype,
            "llvm.hexagon.V6.vmpyowh.rnd.sacc.128B",
            tvm.tir.const(3, "uint32"),
            mul_e_3,
            x,
            y,
        )

        return tvm.tir.Select(
            tvm.tir.Not(tvm.tir.Or(is_lshift_required, is_rshift_required)),
            no_shift_out,
            tvm.tir.Select(is_lshift_required, left_shift_out, right_shift_out),
        )

    return _adapt_to_highest_lanes(*op.args, intrinsic=intrinsic_impl_32, intrinsic_lanes=32)


register_intrin_lowering(
    "tir.q_multiply_shift_per_axis",
    target="hexagon",
    f=_q_multiply_shift_per_axis_hexagon,
    level=99,
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
                raise ValueError("Only (u8, u8) or (u8, i8) dtype pairs are supported by vrmpy.")

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
