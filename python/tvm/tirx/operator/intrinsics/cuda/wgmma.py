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
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments, too-many-locals, too-many-positional-arguments
"""PTX WGMMA operations (Hopper warpgroup MMA).

One ``device_intrinsic`` registration per PTX form table entry. Bodies are
hand-written ``asm volatile(...)`` strings. Variable-arity register vectors
(``mma_async`` accumulators / A fragments) materialize via the same
device_intrinsic with an attr-driven ``c_signature`` callable.
"""

import tvm

from .._schema import device_intrinsic
from .registry import CODEGEN_REGISTRY, register_codegen
from .types import PTXDataType
from .utils import parse_str

# =============================================================================
# wgmma.fence / commit_group / wait_group — one PTX form each.
# =============================================================================
device_intrinsic(
    "ptx_wgmma_fence",
    helper_name="ptx_wgmma_fence",
    body='    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");',
)
device_intrinsic(
    "ptx_wgmma_commit_group",
    helper_name="ptx_wgmma_commit_group",
    body='    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");',
)
device_intrinsic(
    "ptx_wgmma_wait_group",
    n_attrs=1,
    helper_name=lambda n: f"ptx_wgmma_wait_group_{int(n)}",
    body=lambda n: f'    asm volatile("wgmma.wait_group.sync.aligned {int(n)};" ::: "memory");',
)


# =============================================================================
# wgmma_encode_matrix_descriptor — pure-C bitfield struct fill (no asm).
# =============================================================================
device_intrinsic(
    "ptx_wgmma_encode_matrix_descriptor",
    helper_name="ptx_wgmma_encode_matrix_descriptor",
    c_signature="(uint64_t* desc, void* addr, int ldo, int sdo, int swizzle)",
    body=(
        "  GmmaDescriptor _desc{};  // value-init: reading uncovered pad bits is UB\n"
        "\n"
        "  switch (swizzle) {\n"
        "    case 0: _desc.bitfield.layout_type_ = uint8_t(0); break; // No swizzle\n"
        "    case 1: _desc.bitfield.layout_type_ = uint8_t(3); break; // 32B swizzle\n"
        "    case 2: _desc.bitfield.layout_type_ = uint8_t(2); break; // 64B swizzle\n"
        "    case 3: _desc.bitfield.layout_type_ = uint8_t(1); break; // 128B swizzle\n"
        "  }\n"
        "\n"
        "  uint32_t start_address = __cvta_generic_to_shared(addr);\n"
        "  _desc.bitfield.start_address_ = static_cast<uint16_t>(start_address >> 4);\n"
        "\n"
        "  constexpr uint8_t base_offset = 0;\n"
        "  _desc.bitfield.base_offset_ = base_offset;\n"
        "\n"
        "  _desc.bitfield.stride_byte_offset_  = static_cast<uint32_t>(sdo);\n"
        "  _desc.bitfield.leading_byte_offset_ = static_cast<uint32_t>(ldo);\n"
        "\n"
        "  *desc = (uint64_t)_desc;"
    ),
    extra_deps=("gmma_descriptor",),
)


# =============================================================================
# wgmma_noop_barrier — empty asm with one inout register operand. Two
# device_intrinsic calls, one per supported dtype; dispatcher picks the form
# based on the operand's runtime dtype.
# =============================================================================
device_intrinsic(
    "ptx_wgmma_noop_barrier_uint32",
    helper_name="ptx_wgmma_fence_uint32_t",
    c_signature="(uint32_t reg)",
    body='    asm volatile("" : "+r"(reg) :: "memory");',
)
device_intrinsic(
    "ptx_wgmma_noop_barrier_float32",
    helper_name="ptx_wgmma_fence_float",
    c_signature="(float reg)",
    body='    asm volatile("" : "+f"(reg) :: "memory");',
)


@register_codegen("ptx_wgmma_noop_barrier")
def codegen_ptx_wgmma_noop_barrier(reg):
    dtype = str(reg.dtype)
    dtype_enum = PTXDataType.from_string(dtype)
    if dtype_enum == PTXDataType.UINT32:
        op_name = "tirx.ptx_wgmma_noop_barrier_uint32"
    elif dtype_enum == PTXDataType.FLOAT32:
        op_name = "tirx.ptx_wgmma_noop_barrier_float32"
    else:
        raise ValueError(f"Only support uint32/float32 for wgmma_fence, but got {dtype}.")
    result = CODEGEN_REGISTRY[op_name]([reg])
    return result[0] if isinstance(result, tuple) else result


# =============================================================================
# wgmma.mma_async ss / rs — 2 PTX form table entries. Accumulator count and
# A-register count vary with (M, N, K, in_dtype) but are fully determined by
# attrs at codegen time.
#
# Args layout for ss form (forwarded operand args first, then 9 attr args):
#   *p_acc[0..num_accums-1], p_descA, p_descB, p_scaleD,
#   M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB
#
# Args layout for rs form:
#   *p_acc[0..num_accums-1], *p_A[0..num_A_regs-1], p_descB, p_scaleD,
#   M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB
# =============================================================================


def _coerce_wgmma_attrs(attrs):
    """Decode the trailing 9 attrs (M, N, K, in_dtype, out_dtype, transA,
    transB, scaleA, scaleB) into native Python types."""
    M, N, K = int(attrs[0]), int(attrs[1]), int(attrs[2])
    in_dtype = parse_str(attrs[3])
    out_dtype = parse_str(attrs[4])
    transA = bool(int(attrs[5])) if hasattr(attrs[5], "value") else bool(attrs[5])
    transB = bool(int(attrs[6])) if hasattr(attrs[6], "value") else bool(attrs[6])
    scaleA = bool(int(float(attrs[7])))
    scaleB = bool(int(float(attrs[8])))
    if out_dtype != "float32":
        raise ValueError("WGMMA codegen only supports float32 as output dtype.")
    allow_transpose = in_dtype in {"float16", "bfloat16"}
    if not allow_transpose and (transA or transB):
        raise ValueError("Transpose is only supported for .f16/.bf16 types in WGMMA.")
    return M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB, allow_transpose


def _safe(s):
    return s.replace("::", "_").replace(".", "_")


def _wgmma_helper_name(prefix, M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB):
    return (
        f"{prefix}_{M}x{N}x{K}_{_safe(in_dtype)}_{_safe(out_dtype)}"
        f"_{1 if scaleA else 0}_{1 if scaleB else 0}"
        f"_{1 if transA else 0}_{1 if transB else 0}"
    )


def _wgmma_in_bits(in_dtype):
    return tvm.runtime.DataType(in_dtype).bits


def _wgmma_ss_parts(*args):
    M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB, allow_transpose = (
        _coerce_wgmma_attrs(args[-9:])
    )
    num_accums = M * N // 128

    name = _wgmma_helper_name(
        "ptx_wgmma_mma_async_ss", M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB
    )
    sig = (
        "("
        + ", ".join(
            [f"float& p_acc{i}" for i in range(num_accums)]
            + ["uint64_t p_descA", "uint64_t p_descB", "int p_scaleD"]
        )
        + ")"
    )
    descA_idx = num_accums
    descB_idx = num_accums + 1
    scaleD_idx = num_accums + 2
    scaleA_idx = num_accums + 3
    scaleB_idx = num_accums + 4
    transA_idx = num_accums + 5
    transB_idx = num_accums + 6
    accum_r_list = ", ".join(f"%{i}" for i in range(num_accums))
    accum_constraints = ", ".join(f'"+f"(p_acc{i})' for i in range(num_accums))
    itype = PTXDataType.from_string(in_dtype)
    otype = PTXDataType.from_string(out_dtype)
    if allow_transpose:
        transpose_r_code = f", %{transA_idx}, %{transB_idx}"
        transpose_constraints = f', "n"({1 if transA else 0}), "n"({1 if transB else 0})'
    else:
        transpose_r_code = ""
        transpose_constraints = ""
    instr = (
        f"wgmma.mma_async.sync.aligned.m{M}n{N}k{K}"
        f"{otype.to_string()}{itype.to_string()}{itype.to_string()}"
    )
    asm_inputs = (
        f'"l"(p_descA), "l"(p_descB), "r"(p_scaleD),'
        f' "n"({1 if scaleA else 0}), "n"({1 if scaleB else 0})'
        f"{transpose_constraints}"
    )
    body = (
        "    asm volatile(\n"
        '      "{    \\n"\n'
        '      ".reg .pred p;\\n"\n'
        f'      "setp.ne.b32 p, %{scaleD_idx}, 0;\\n"\n'
        f'      "{instr} "\n'
        f'      "{{{accum_r_list}}},"\n'
        f'      "%{descA_idx}, %{descB_idx},"\n'
        f'      "p, %{scaleA_idx}, %{scaleB_idx}{transpose_r_code};\\n"\n'
        '      "}\\n"\n'
        f"      : {accum_constraints}\n"
        f"      : {asm_inputs}\n"
        "    );"
    )
    return name, sig, body


device_intrinsic(
    "_ptx_wgmma_mma_async_ss_impl",
    n_attrs=9,
    helper_name=lambda *a: _wgmma_ss_parts(*a)[0],
    c_signature=lambda *a: _wgmma_ss_parts(*a)[1],
    body=lambda *a: _wgmma_ss_parts(*a)[2],
)


def _wgmma_rs_parts(*args):
    M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB, allow_transpose = (
        _coerce_wgmma_attrs(args[-9:])
    )
    num_accums = M * N // 128
    in_bits = _wgmma_in_bits(in_dtype)
    num_A_regs = M * K // 128 // (32 // in_bits)

    name = _wgmma_helper_name(
        "ptx_wgmma_mma_async_rs", M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB
    )
    sig = (
        "("
        + ", ".join(
            [f"float& p_acc{i}" for i in range(num_accums)]
            + [f"uint32_t& p_A{i}" for i in range(num_A_regs)]
            + ["uint64_t p_descB", "int p_scaleD"]
        )
        + ")"
    )

    accum_r_list = ", ".join(f"%{i}" for i in range(num_accums))
    A_reg_r_list = ", ".join(f"%{num_accums + i}" for i in range(num_A_regs))
    base_idx = num_accums + num_A_regs
    descB_idx = base_idx
    scaleD_idx = base_idx + 1
    scaleA_idx = base_idx + 2
    scaleB_idx = base_idx + 3
    transB_idx = base_idx + 4
    accum_constraints = ", ".join(f'"+f"(p_acc{i})' for i in range(num_accums))
    A_reg_constraints = ", ".join(f'"r"(p_A{i})' for i in range(num_A_regs))
    itype = PTXDataType.from_string(in_dtype)
    otype = PTXDataType.from_string(out_dtype)
    if allow_transpose:
        transpose_r_code = f", %{transB_idx}"
        transpose_constraints = f', "n"({1 if transB else 0})'
    else:
        transpose_r_code, transpose_constraints = "", ""
    instr = (
        f"wgmma.mma_async.sync.aligned.m{M}n{N}k{K}"
        f"{otype.to_string()}{itype.to_string()}{itype.to_string()}"
    )
    asm_inputs = (
        f'{A_reg_constraints}, "l"(p_descB), "r"(p_scaleD),'
        f' "n"({1 if scaleA else 0}), "n"({1 if scaleB else 0})'
        f"{transpose_constraints}"
    )
    body = (
        "    asm volatile(\n"
        '      "{    \\n"\n'
        '      ".reg .pred p;\\n"\n'
        f'      "setp.ne.b32 p, %{scaleD_idx}, 0;\\n"\n'
        f'      "{instr} "\n'
        f'      "{{{accum_r_list}}},"\n'
        f'      "{{{A_reg_r_list}}}, %{descB_idx},"\n'
        f'      "p, %{scaleA_idx}, %{scaleB_idx}{transpose_r_code};\\n"\n'
        '      "}\\n"\n'
        f"      : {accum_constraints}\n"
        f"      : {asm_inputs}\n"
        "    );"
    )
    return name, sig, body


device_intrinsic(
    "_ptx_wgmma_mma_async_rs_impl",
    n_attrs=9,
    helper_name=lambda *a: _wgmma_rs_parts(*a)[0],
    c_signature=lambda *a: _wgmma_rs_parts(*a)[1],
    body=lambda *a: _wgmma_rs_parts(*a)[2],
)


# User-facing wrappers: just normalise types + reorder positional args to
# put operands first, then attrs, matching the schema convention.


def _wgmma_user_wrapper_ss(*args):
    M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB, scaleD, descA, descB, *accums = (
        args
    )
    M = int(M)
    N = int(N)
    K = int(K)
    in_dtype = parse_str(in_dtype)
    out_dtype = parse_str(out_dtype)
    transA = bool(transA)
    transB = bool(transB)
    scaleA = bool(int(float(scaleA)))
    scaleB = bool(int(float(scaleB)))
    expected = M * N // 128
    if len(accums) != expected:
        raise ValueError(
            "The number of arguments is incorrect. Expected "
            f"{12 + expected} total args (meaning {expected} accumulator args), "
            f"but got {len(accums)}."
        )
    return [
        *accums,
        descA,
        descB,
        scaleD,
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        transA,
        transB,
        scaleA,
        scaleB,
    ]


@register_codegen("ptx_wgmma_mma_async_ss")
def codegen_ptx_wgmma_mma_async_ss(*args):
    forwarded = _wgmma_user_wrapper_ss(*args)
    result = CODEGEN_REGISTRY["tirx._ptx_wgmma_mma_async_ss_impl"](forwarded)
    return result[0] if isinstance(result, tuple) else result


def _wgmma_user_wrapper_rs(*args):
    M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB, scaleD, descB, *reg_list = args
    M = int(M)
    N = int(N)
    K = int(K)
    in_dtype = parse_str(in_dtype)
    out_dtype = parse_str(out_dtype)
    transA = bool(transA)
    transB = bool(transB)
    scaleA = bool(int(float(scaleA)))
    scaleB = bool(int(float(scaleB)))
    if out_dtype != "float32":
        raise ValueError("This generator only supports float32 as the output dtype for WGMMA.")
    in_dtype_bits = tvm.runtime.DataType(in_dtype).bits
    if in_dtype_bits is None:
        raise ValueError(f"Bit width not defined for input dtype: {in_dtype}")
    expected_A_cnt = M * K // 128 // (32 // in_dtype_bits)
    expected_accm_cnt = M * N // 128
    if len(reg_list) != expected_A_cnt + expected_accm_cnt:
        raise ValueError(
            f"Incorrect number of A registers. Expected {expected_A_cnt}, got {len(reg_list)}"
        )
    A_regs = reg_list[:expected_A_cnt]
    accums = reg_list[expected_A_cnt:]
    return [
        *accums,
        *A_regs,
        descB,
        scaleD,
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        transA,
        transB,
        scaleA,
        scaleB,
    ]


@register_codegen("ptx_wgmma_mma_async_rs")
def codegen_ptx_wgmma_mma_async_rs(*args):
    forwarded = _wgmma_user_wrapper_rs(*args)
    result = CODEGEN_REGISTRY["tirx._ptx_wgmma_mma_async_rs_impl"](forwarded)
    return result[0] if isinstance(result, tuple) else result
