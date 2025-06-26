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
# pylint: disable=invalid-name,missing-function-docstring,unused-import
"""Intrinsics for RVV tensorization, both for C and LLVM targets.
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""
import re
from tvm.script import tir as T
from tvm.target.datatype import lower_call_pure_extern, register, register_op
from .. import TensorIntrin

#####################################################
# LLVM RISC-V Intrinsic usage:
# https://llvm.org/docs//RISCV/RISCVVectorExtension.html
#
# Vector types are represented using scalable vector
# types, of the form <vscale x n x ty>. n and ty
# control LMUL and SEW respectively (see table in docs).
# TVM represents this with dtype = "tyxvscalexn".
#
# n is calculated as (64/SEW)*LMUL.
# VL is passed to each intrinsic.
#
# Some examples (see table in docs):
# int8 vector type with LMUL = 1 => int8xvscalex8
# int16 vector type with LMUL = 4 => int16xvscalex16
# int32 vector type with LMUL = 2 => int32xvscalex4
#
#####################################################

#####################################################
# Helper functions
#####################################################

RISCV_MIN_VL = 4


def get_vlmax(vlen: int, lmul: int, max_sew: int) -> int:
    """Return VLMAX

    Args:
        vlen (int): Actual VLEN
        lmul (int): LMUL
        max_sew (int): SEW

    Returns:
        int: VLMAX
    """
    return (lmul * vlen) // max_sew


def get_vlen_from_mattrs(mattrs: list) -> int:
    """Extract VLEN from LLVM mattrs list

    Args:
        mattrs (list): LLVM list of CPU mattrs

    Returns:
        int: VLEN
    """
    vlen_regex = r"zvl(\d+)b"
    vlen = 0
    for mattr in mattrs:
        match = re.search(vlen_regex, mattr)

        if match:
            vlen = int(match.group(1))
            break
    return vlen


def _dtype_to_bits(dtype: str) -> int:
    """Get bits from data type

    Args:
        dtype (str): Data type

    Returns:
        int: bits
    """
    bits_per_item = int(
        re.match(r"((float)|(int)|(uint))(?P<width_bits>[0-9]+)", dtype).group("width_bits")
    )
    assert bits_per_item is not None, f"don't know how to compute size of type {dtype}"
    return bits_per_item


def _get_dtype_string(dtype: str) -> str:
    """Get only type of data type, without bits

    Args:
        dtype (str): Data type

    Returns:
        str: only string type
    """
    return str(re.match(r"[a-z]+", dtype).group(0))


#####################################################
# Parameterized intrinsics
#####################################################


def rvv_vmacc(J: int, vlmax: int, input_dtype: str, output_dtype: str, lmul: int):
    # pylint: disable=unused-argument
    input_bits = _dtype_to_bits(input_dtype)
    output_bits = _dtype_to_bits(output_dtype)

    output_str_type = _get_dtype_string(output_dtype)

    output_dtype_prefix = output_str_type[0]

    input_lmul = lmul if output_dtype_prefix == "f" else lmul // 2

    load_llvm_intrinsic = "llvm.riscv.vle"
    expand_llvm_intrinsic = "llvm.riscv.vsext"
    init_llvm_intrinsic = "llvm.riscv.vle"
    macc_llvm_intrinsic = "llvm.riscv.vmacc" if output_dtype_prefix != "f" else "llvm.riscv.vfmacc"
    store_llvm_intrinsic = "llvm.riscv.vse"

    # Calculated from https://llvm.org/docs//RISCV/RISCVVectorExtension.html
    n_input_dtype = (64 // input_bits) * input_lmul
    n_output_dtype = (64 // output_bits) * lmul

    llvm_input_dtype = f"{input_dtype}xvscalex{n_input_dtype}"
    llvm_macc_dtype = f"{output_str_type}{output_bits}xvscalex{n_output_dtype}"

    broadcast_input = T.int16(0) if input_dtype == "int16" else T.float32(0)
    broadcast_output = T.int32(0) if output_dtype == "int32" else T.float32(0)

    @T.prim_func
    def rvv_vmacc_desc(
        A: T.Buffer((int(vlmax),), input_dtype, align=4, offset_factor=1),
        B: T.Buffer((int(vlmax),), input_dtype, align=4, offset_factor=1),
        C: T.Buffer((int(vlmax),), output_dtype, align=4, offset_factor=1),
    ) -> None:
        with T.block("root"):
            T.reads(C[0 : int(vlmax)], A[0 : int(vlmax)], B[0 : int(vlmax)])
            T.writes(C[0 : int(vlmax)])
            for j in range(0, int(vlmax)):
                with T.block("update"):
                    vj = T.axis.remap("S", [j])
                    C[vj] = C[vj] + T.cast(A[vj], output_dtype) * T.cast(B[vj], output_dtype)

    @T.prim_func
    def rvv_vmacc_llvm_impl(
        A: T.Buffer((int(vlmax),), input_dtype, align=4, offset_factor=1),
        B: T.Buffer((int(vlmax),), input_dtype, align=4, offset_factor=1),
        C: T.Buffer((int(vlmax),), output_dtype, align=4, offset_factor=1),
    ) -> None:

        with T.block("root"):

            T.reads(A[0 : int(vlmax)], B[0 : int(vlmax)])
            T.writes(C[0 : int(vlmax)])

            vec_A = (
                T.call_llvm_intrin(
                    llvm_macc_dtype,
                    expand_llvm_intrinsic,
                    T.uint32(3),
                    T.broadcast(broadcast_output, n_output_dtype * T.vscale()),
                    T.call_llvm_intrin(
                        llvm_input_dtype,
                        load_llvm_intrinsic,
                        T.uint32(3),
                        T.broadcast(broadcast_input, n_input_dtype * T.vscale()),
                        A.access_ptr(access_mask=A.READ, ptr_type="handle"),
                        T.int64(vlmax),
                    ),
                    T.int64(vlmax),
                )
                if output_dtype_prefix != "f"
                else T.call_llvm_intrin(
                    llvm_input_dtype,
                    load_llvm_intrinsic,
                    T.uint32(3),
                    T.broadcast(broadcast_input, n_input_dtype * T.vscale()),
                    A.access_ptr(access_mask=A.READ, ptr_type="handle"),
                    T.int64(vlmax),
                )
            )

            vec_B = (
                T.call_llvm_intrin(
                    llvm_macc_dtype,
                    expand_llvm_intrinsic,
                    T.uint32(3),
                    T.broadcast(broadcast_output, n_output_dtype * T.vscale()),
                    T.call_llvm_intrin(
                        llvm_input_dtype,
                        load_llvm_intrinsic,
                        T.uint32(3),
                        T.broadcast(broadcast_input, n_input_dtype * T.vscale()),
                        B.access_ptr(access_mask=B.READ, ptr_type="handle"),
                        T.int64(vlmax),
                    ),
                    T.int64(vlmax),
                )
                if output_dtype_prefix != "f"
                else T.call_llvm_intrin(
                    llvm_input_dtype,
                    load_llvm_intrinsic,
                    T.uint32(3),
                    T.broadcast(broadcast_input, n_input_dtype * T.vscale()),
                    B.access_ptr(access_mask=B.READ, ptr_type="handle"),
                    T.int64(vlmax),
                )
            )

            init = T.call_llvm_intrin(
                llvm_macc_dtype,
                init_llvm_intrinsic,
                T.uint32(3),
                T.broadcast(broadcast_output, n_output_dtype * T.vscale()),
                C.access_ptr(access_mask=C.READ, ptr_type="handle"),
                T.uint64(vlmax),
            )

            product = (
                T.call_llvm_intrin(
                    llvm_macc_dtype,
                    macc_llvm_intrinsic,
                    T.uint32(6),
                    init,
                    vec_A,
                    vec_B,
                    T.uint64(7),
                    T.uint64(vlmax),
                    T.uint64(3),
                )
                if output_dtype_prefix == "f"
                else T.call_llvm_intrin(
                    llvm_macc_dtype,
                    macc_llvm_intrinsic,
                    T.uint32(5),
                    init,
                    vec_A,
                    vec_B,
                    T.uint64(vlmax),
                    T.uint64(3),
                )
            )

            T.call_llvm_intrin(
                "",
                store_llvm_intrinsic,
                T.uint32(3),
                product,
                C.access_ptr(access_mask=C.WRITE, ptr_type="handle"),
                T.uint64(vlmax),
            )

    return rvv_vmacc_desc, rvv_vmacc_llvm_impl


def rvv_multivmul(J: int, vlmax: int, input_dtype: str, output_dtype: str, lmul: int):
    # pylint: disable=unused-argument
    assert J > 1

    input_bits = _dtype_to_bits(input_dtype)
    kernel_bits = _dtype_to_bits(input_dtype)
    output_bits = _dtype_to_bits(output_dtype)

    output_str_type = _get_dtype_string(output_dtype)

    output_dtype_prefix = (
        "i" if output_str_type == "int" else ("u" if output_str_type == "uint" else "f")
    )

    intermmediate_bits = output_bits if output_dtype_prefix == "f" else input_bits + kernel_bits
    intermmediate_bits = input_bits

    load_llvm_intrinsic = "llvm.riscv.vle"
    expand_llvm_intrinsic = "llvm.riscv.vsext"
    init_llvm_intrinsic = (
        "llvm.riscv.vmv.v.x" if output_dtype_prefix != "f" else "llvm.riscv.vfmv.v.f"
    )
    mult_llvm_intrinsic = "llvm.riscv.vmul" if output_dtype_prefix != "f" else "llvm.riscv.vfmul"
    redsum_llvm_intrinsic = (
        "llvm.riscv.vwredsum" if output_dtype_prefix != "f" else "llvm.riscv.vfredusum"
    )
    store_llvm_intrinsic = "llvm.riscv.vse"

    # Calculated from https://llvm.org/docs//RISCV/RISCVVectorExtension.html
    # vscale = vlen // 64
    n_input_dtype = (64 // input_bits) * lmul
    n_kernel_dtype = (64 // kernel_bits) * lmul
    n_intermmediate_dtype = (64 // intermmediate_bits) * lmul

    n_redsum_dtype = (64 // output_bits) * 1

    llvm_input_dtype = f"{input_dtype}xvscalex{n_input_dtype}"
    llvm_kernel_dtype = f"{input_dtype}xvscalex{n_kernel_dtype}"
    llvm_redsum_dtype = f"{output_dtype}xvscalex{n_redsum_dtype}"
    llvm_mult_dtype = f"{output_str_type}{intermmediate_bits}xvscalex{n_intermmediate_dtype}"

    broadcast_input = (
        T.int8(0)
        if input_dtype == "int8"
        else (T.int16(0) if input_dtype == "int16" else T.float32(0))
    )
    broadcast_kernel = (
        T.int8(0)
        if input_dtype == "int8"
        else (T.int16(0) if input_dtype == "int16" else T.float32(0))
    )
    broadcast_intermmediate = T.int16(0) if intermmediate_bits == 16 else T.int32(0)
    broadcast_output = T.int32(0) if output_dtype == "int32" else T.float32(0)

    @T.prim_func
    def rvv_multivmul_desc(
        A: T.Buffer((int(vlmax),), input_dtype, align=4, offset_factor=1),
        B: T.Buffer((J, int(vlmax)), kernel_dtype, align=4, offset_factor=1),
        C: T.Buffer((J,), output_dtype, align=4, offset_factor=1),
    ) -> None:
        with T.block("root"):
            T.reads(C[0:J], A[0 : int(vlmax)], B[0:J, 0 : int(vlmax)])
            T.writes(C[0:J])
            for j in range(0, J):
                for k in range(0, int(vlmax)):
                    with T.block("update"):
                        vj, vk = T.axis.remap("SR", [j, k])
                        C[vj] = C[vj] + T.cast(A[vk], output_dtype) * T.cast(
                            B[vj, vk], output_dtype
                        )

    @T.prim_func
    def rvv_multivmul_llvm_impl(
        A: T.Buffer((int(vlmax),), input_dtype, align=4, offset_factor=1),
        B: T.Buffer(
            (J, int(vlmax)), kernel_dtype, align=4, offset_factor=1, strides=[T.int32(), T.int32()]
        ),
        C: T.Buffer((J,), output_dtype, align=4, offset_factor=1),
    ) -> None:

        with T.block("root"):

            T.reads(A[0 : int(vlmax)], B[0:J, 0 : int(vlmax)])
            T.writes(C[0:J])

            vec_A = (
                T.call_llvm_intrin(
                    llvm_mult_dtype,
                    expand_llvm_intrinsic,
                    T.uint32(3),
                    T.broadcast(broadcast_intermmediate, n_intermmediate_dtype * T.vscale()),
                    T.call_llvm_intrin(
                        llvm_input_dtype,
                        load_llvm_intrinsic,
                        T.uint32(3),
                        T.broadcast(broadcast_input, n_input_dtype * T.vscale()),
                        A.access_ptr(access_mask=A.READ, ptr_type="handle"),
                        T.int64(vlmax),
                    ),
                    T.int64(vlmax),
                )
                if output_dtype_prefix != "f"
                else T.call_llvm_intrin(
                    llvm_input_dtype,
                    load_llvm_intrinsic,
                    T.uint32(3),
                    T.broadcast(broadcast_input, n_input_dtype * T.vscale()),
                    A.access_ptr(access_mask=A.READ, ptr_type="handle"),
                    T.int64(vlmax),
                )
            )

            vec_B = (
                T.call_llvm_intrin(
                    llvm_mult_dtype,
                    expand_llvm_intrinsic,
                    T.uint32(3),
                    T.broadcast(broadcast_intermmediate, n_intermmediate_dtype * T.vscale()),
                    T.call_llvm_intrin(
                        llvm_input_dtype,
                        load_llvm_intrinsic,
                        T.uint32(3),
                        T.broadcast(broadcast_input, n_input_dtype * T.vscale()),
                        B.access_ptr(access_mask=B.READ, ptr_type="handle"),
                        T.int64(vlmax),
                    ),
                    T.int64(vlmax),
                )
                if output_dtype_prefix != "f"
                else T.call_llvm_intrin(
                    llvm_kernel_dtype,
                    load_llvm_intrinsic,
                    T.uint32(3),
                    T.broadcast(broadcast_kernel, n_kernel_dtype * T.vscale()),
                    B.access_ptr(access_mask=B.READ, ptr_type="handle"),
                    T.int64(vlmax),
                )
            )

            redsum = T.call_llvm_intrin(
                llvm_redsum_dtype,
                init_llvm_intrinsic,
                T.uint32(3),
                T.broadcast(broadcast_output, n_redsum_dtype * T.vscale()),
                C[0],
                T.uint64(1),
            )

            product = (
                T.call_llvm_intrin(
                    llvm_mult_dtype,
                    mult_llvm_intrinsic,
                    T.uint32(5),
                    T.broadcast(broadcast_output, n_intermmediate_dtype * T.vscale()),
                    vec_A,
                    vec_B,
                    T.uint64(7),
                    T.uint64(vlmax),
                )
                if output_dtype_prefix == "f"
                else T.call_llvm_intrin(
                    llvm_mult_dtype,
                    mult_llvm_intrinsic,
                    T.uint32(4),
                    T.broadcast(broadcast_output, n_intermmediate_dtype * T.vscale()),
                    vec_A,
                    vec_B,
                    T.uint64(vlmax),
                )
            )

            redsum_result = (
                T.call_llvm_intrin(
                    llvm_redsum_dtype,
                    redsum_llvm_intrinsic,
                    T.uint32(5),
                    T.broadcast(broadcast_output, n_redsum_dtype * T.vscale()),
                    product,
                    redsum,
                    T.uint64(7),
                    T.uint64(vlmax),
                )
                if output_dtype_prefix == "f"
                else T.call_llvm_intrin(
                    llvm_redsum_dtype,
                    redsum_llvm_intrinsic,
                    T.uint32(4),
                    T.broadcast(broadcast_output, n_redsum_dtype * T.vscale()),
                    product,
                    redsum,
                    T.uint64(vlmax),
                )
            )

            T.call_llvm_intrin(
                "",
                store_llvm_intrinsic,
                T.uint32(3),
                redsum_result,
                C.access_ptr(access_mask=C.WRITE, ptr_type="handle"),
                T.uint64(1),
            )

    return rvv_multivmul_desc, rvv_multivmul_llvm_impl


def rvv_vmul(J: int, vlmax: int, input_dtype: str, output_dtype: str, lmul: int):
    # pylint: disable=unused-argument
    input_bits = _dtype_to_bits(input_dtype)
    output_bits = _dtype_to_bits(output_dtype)

    output_str_type = _get_dtype_string(output_dtype)

    output_dtype_prefix = (
        "i" if output_str_type == "int" else ("u" if output_str_type == "uint" else "f")
    )

    intermmediate_bits = output_bits if output_dtype_prefix == "f" else input_bits * 2
    intermmediate_bits = input_bits

    load_llvm_intrinsic = "llvm.riscv.vle"
    expand_llvm_intrinsic = "llvm.riscv.vsext"
    init_llvm_intrinsic = (
        "llvm.riscv.vmv.v.x" if output_dtype_prefix != "f" else "llvm.riscv.vfmv.v.f"
    )
    mult_llvm_intrinsic = "llvm.riscv.vmul" if output_dtype_prefix != "f" else "llvm.riscv.vfmul"
    redsum_llvm_intrinsic = (
        "llvm.riscv.vwredsum" if output_dtype_prefix != "f" else "llvm.riscv.vfredusum"
    )
    store_llvm_intrinsic = "llvm.riscv.vse"

    # Calculated from https://llvm.org/docs//RISCV/RISCVVectorExtension.html
    # vscale = vlen // 64
    n_input_dtype = (64 // input_bits) * lmul
    n_kernel_dtype = (64 // input_bits) * lmul
    n_intermmediate_dtype = (64 // intermmediate_bits) * lmul

    n_redsum_dtype = (64 // output_bits) * 1

    llvm_input_dtype = f"{input_dtype}xvscalex{n_input_dtype}"
    llvm_kernel_dtype = f"{input_dtype}xvscalex{n_kernel_dtype}"
    llvm_redsum_dtype = f"{output_dtype}xvscalex{n_redsum_dtype}"
    llvm_mult_dtype = f"{output_str_type}{intermmediate_bits}xvscalex{n_intermmediate_dtype}"

    broadcast_input = (
        T.int8(0)
        if input_dtype == "int8"
        else (T.int16(0) if input_dtype == "int16" else T.float32(0))
    )
    broadcast_kernel = (
        T.int8(0)
        if input_dtype == "int8"
        else (T.int16(0) if input_dtype == "int16" else T.float32(0))
    )
    broadcast_intermmediate = T.int16(0) if intermmediate_bits == 16 else T.int32(0)
    broadcast_output = T.int32(0) if output_dtype == "int32" else T.float32(0)

    @T.prim_func
    def rvv_vmul_desc(
        A: T.Buffer((int(vlmax),), input_dtype, align=4, offset_factor=1),
        B: T.Buffer((int(vlmax),), kernel_dtype, align=4, offset_factor=1),
        C: T.Buffer((1,), output_dtype, align=4, offset_factor=1),
    ) -> None:
        with T.block("root"):
            T.reads(C[0], A[0 : int(vlmax)], B[0 : int(vlmax)])
            T.writes(C[0])
            for k in range(0, int(vlmax)):
                with T.block("update"):
                    vk = T.axis.remap("R", [k])
                    C[0] = C[0] + T.cast(A[vk], output_dtype) * T.cast(B[vk], output_dtype)

    @T.prim_func
    def rvv_vmul_llvm_impl(
        A: T.Buffer((int(vlmax),), input_dtype, align=4, offset_factor=1),
        B: T.Buffer((int(vlmax),), kernel_dtype, align=4, offset_factor=1),
        C: T.Buffer((1,), output_dtype, align=4, offset_factor=1),
    ) -> None:

        with T.block("root"):

            T.reads(A[0 : int(vlmax)], B[0 : int(vlmax)])
            T.writes(C[0])

            vec_A = (
                T.call_llvm_intrin(
                    llvm_mult_dtype,
                    expand_llvm_intrinsic,
                    T.uint32(3),
                    T.broadcast(broadcast_intermmediate, n_intermmediate_dtype * T.vscale()),
                    T.call_llvm_intrin(
                        llvm_input_dtype,
                        load_llvm_intrinsic,
                        T.uint32(3),
                        T.broadcast(broadcast_input, n_input_dtype * T.vscale()),
                        A.access_ptr(access_mask=A.READ, ptr_type="handle"),
                        T.int64(vlmax),
                    ),
                    T.int64(vlmax),
                )
                if output_dtype_prefix != "f"
                else T.call_llvm_intrin(
                    llvm_input_dtype,
                    load_llvm_intrinsic,
                    T.uint32(3),
                    T.broadcast(broadcast_input, n_input_dtype * T.vscale()),
                    A.access_ptr(access_mask=A.READ, ptr_type="handle"),
                    T.int64(vlmax),
                )
            )

            vec_B = (
                T.call_llvm_intrin(
                    llvm_mult_dtype,
                    expand_llvm_intrinsic,
                    T.uint32(3),
                    T.broadcast(broadcast_intermmediate, n_intermmediate_dtype * T.vscale()),
                    T.call_llvm_intrin(
                        llvm_input_dtype,
                        load_llvm_intrinsic,
                        T.uint32(3),
                        T.broadcast(broadcast_input, n_input_dtype * T.vscale()),
                        B.access_ptr(access_mask=B.READ, ptr_type="handle"),
                        T.int64(vlmax),
                    ),
                    T.int64(vlmax),
                )
                if output_dtype_prefix != "f"
                else T.call_llvm_intrin(
                    llvm_kernel_dtype,
                    load_llvm_intrinsic,
                    T.uint32(3),
                    T.broadcast(broadcast_kernel, n_kernel_dtype * T.vscale()),
                    B.access_ptr(access_mask=B.READ, ptr_type="handle"),
                    T.int64(vlmax),
                )
            )

            redsum = T.call_llvm_intrin(
                llvm_redsum_dtype,
                init_llvm_intrinsic,
                T.uint32(3),
                T.broadcast(broadcast_output, n_redsum_dtype * T.vscale()),
                C[0],
                T.uint64(1),
            )

            product = (
                T.call_llvm_intrin(
                    llvm_mult_dtype,
                    mult_llvm_intrinsic,
                    T.uint32(5),
                    T.broadcast(broadcast_output, n_intermmediate_dtype * T.vscale()),
                    vec_A,
                    vec_B,
                    T.uint64(7),
                    T.uint64(vlmax),
                )
                if output_dtype_prefix == "f"
                else T.call_llvm_intrin(
                    llvm_mult_dtype,
                    mult_llvm_intrinsic,
                    T.uint32(4),
                    T.broadcast(broadcast_output, n_intermmediate_dtype * T.vscale()),
                    vec_A,
                    vec_B,
                    T.uint64(vlmax),
                )
            )

            redsum_result = (
                T.call_llvm_intrin(
                    llvm_redsum_dtype,
                    redsum_llvm_intrinsic,
                    T.uint32(5),
                    T.broadcast(broadcast_output, n_redsum_dtype * T.vscale()),
                    product,
                    redsum,
                    T.uint64(7),
                    T.uint64(vlmax),
                )
                if output_dtype_prefix == "f"
                else T.call_llvm_intrin(
                    llvm_redsum_dtype,
                    redsum_llvm_intrinsic,
                    T.uint32(4),
                    T.broadcast(broadcast_output, n_redsum_dtype * T.vscale()),
                    product,
                    redsum,
                    T.uint64(vlmax),
                )
            )

            T.call_llvm_intrin(
                "",
                store_llvm_intrinsic,
                T.uint32(3),
                redsum_result,
                C.access_ptr(access_mask=C.WRITE, ptr_type="handle"),
                T.uint64(1),
            )

    return rvv_vmul_desc, rvv_vmul_llvm_impl


#####################################################
# Registering intrinsics
#####################################################


def register_intrinsic_combinations(
    outer_loops, initial_vlmax, lmul, input_dtype, output_dtype, prefix, generator
):
    for J in outer_loops:
        current_vlmax = initial_vlmax
        while current_vlmax >= RISCV_MIN_VL:

            name = f"{prefix}_{J}_{current_vlmax}_m{lmul}"

            desc, impl = generator(J, current_vlmax, input_dtype, output_dtype, lmul)

            print(f"Registering intrin {name}...")

            TensorIntrin.register(name, desc, impl, override=True)

            current_vlmax = current_vlmax // 2


def register_riscv_tensor_intrinsics(target):
    target_kind = target.kind.name
    assert target_kind in ["llvm"]

    #####################################################
    # Register custom RVV types for C code generation
    #####################################################
    dtype_counter = 0
    for bits in [8, 16, 32, 64]:
        for dtype in ["int", "uint", "float"]:
            for m in [1, 2, 4, 8]:
                custom_rvv_type = f"v{dtype}{bits}m{m}_t"
                register(custom_rvv_type, 150 + dtype_counter)
                register_op(
                    lower_call_pure_extern,
                    "Call",
                    "c",
                    custom_rvv_type,
                    intrinsic_name="tir.call_pure_extern",
                )
                dtype_counter += 1

    vlen = get_vlen_from_mattrs(target.mattr)

    for vmul_type, func, outer_loops in zip(
        ["vmacc", "multivmul", "vmul"],
        [rvv_vmacc, rvv_multivmul, rvv_vmul],
        [[1], [get_vlmax(vlen, lmul=1, max_sew=32)], [1]],
    ):

        for idtype, odtype in zip(["int16", "float32"], ["int32", "float32"]):

            if idtype == "float32" and vmul_type == "multivmul":
                continue

            vlmax = get_vlmax(vlen, lmul=8, max_sew=32)
            register_intrinsic_combinations(
                outer_loops, vlmax, 8, idtype, odtype, f"rvv_{idtype}_{vmul_type}", func
            )

    print("Finished registering all intrinsics.")
