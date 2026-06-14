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
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments, too-many-locals, line-too-long, too-many-positional-arguments
"""PTX tcgen05 operations (Blackwell tensor memory, MMA).

One ``device_intrinsic`` registration per PTX form table entry; bodies are
hand-written ``asm volatile(...)`` strings. Variable-arity forms (mma masks,
ld/st register vectors) compute the C signature and body together inside a
shared parts callable.
"""

import tvm

from ._schema import device_intrinsic
from .registry import CODEGEN_REGISTRY, register_codegen
from .types import PTXDataType
from .utils import parse_str, validate_cta_group, validate_power_of_two_range


def _safe(s):
    return s.replace("::", "_").replace(".", "_")


# =============================================================================
# Trivial fence / wait — single PTX line, no operands, no attrs.
# =============================================================================
device_intrinsic(
    "ptx_tcgen05_fence_before_thread_sync",
    body='    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");',
)
device_intrinsic(
    "ptx_tcgen05_fence_after_thread_sync",
    body='    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");',
)
device_intrinsic(
    "ptx_tcgen05_wait_ld", body='    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");'
)
device_intrinsic(
    "ptx_tcgen05_wait_st", body='    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");'
)


# =============================================================================
# tcgen05.shift / relinquish_alloc_permit / alloc / dealloc.
# =============================================================================
device_intrinsic(
    "ptx_tcgen05_shift",
    n_attrs=1,
    c_signature="(uint32_t taddr)",
    helper_name=lambda taddr_, cta_group: f"ptx_tcgen05_shift_cta_group_{int(cta_group)}",
    body=lambda taddr_, cta_group: (
        f'    asm volatile("tcgen05.shift.cta_group::{int(cta_group)}.down [%0];" '
        ': : "r"(taddr) : "memory");'
    ),
)

device_intrinsic(
    "ptx_tcgen05_relinquish_alloc_permit",
    n_attrs=1,
    helper_name=lambda n_cta_group: (
        f"tvm_builtin_ptx_tcgen05_relinquish_alloc_permit_cta_group_{int(n_cta_group)}"
    ),
    body=lambda n_cta_group: (
        f'    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::{int(n_cta_group)}'
        '.sync.aligned;" ::: "memory");'
    ),
)

device_intrinsic(
    "ptx_tcgen05_alloc",
    n_attrs=1,
    c_signature="(void* dst, int nCols)",
    helper_name=lambda dst_, nCols_, n_cta_group: (
        f"tvm_builtin_ptx_tcgen05_alloc_cta_group_{int(n_cta_group)}"
    ),
    body=lambda dst_, nCols_, n_cta_group: (
        "    unsigned int dst_addr = __cvta_generic_to_shared(dst);\n"
        f'    asm volatile("tcgen05.alloc.cta_group::{int(n_cta_group)}'
        '.sync.aligned.shared::cta.b32 [%0], %1;" '
        ': : "r"(dst_addr), "r"(nCols) : "memory");'
    ),
)

device_intrinsic(
    "ptx_tcgen05_dealloc",
    n_attrs=1,
    c_signature="(uint32_t taddr, int nCols)",
    helper_name=lambda taddr_, nCols_, n_cta_group: (
        f"tvm_builtin_ptx_tcgen05_dealloc_cta_group_{int(n_cta_group)}"
    ),
    body=lambda taddr_, nCols_, n_cta_group: (
        f'    asm volatile("tcgen05.dealloc.cta_group::{int(n_cta_group)}'
        '.sync.aligned.b32 %0, %1;" '
        ': : "r"(taddr), "r"(nCols) : "memory");'
    ),
)


# =============================================================================
# tcgen05.ld / tcgen05.st — 2 PTX form table entries each.
#
#   Form 1 (shape ∈ {16x64b, 16x128b, 16x256b, 32x32b}):
#     tcgen05.ld.sync.aligned.<shape>.<num>{.pack}.b32  r, [taddr];
#   Form 2 (shape = 16x32bx2):
#     tcgen05.ld.sync.aligned.16x32bx2.<num>{.pack}.b32 r, [taddr], immHalfSplitoff;
#
# ``r`` is a register vector whose element count is shape * num / 32b (1, 2,
# or 4 elements per ``num``). We materialise the per-element C parameters at
# codegen time from ``shape`` and ``num``.
# =============================================================================


def _tcgen05_ld_st_n_regs(shape, num):
    if shape in ("16x32bx2", "16x64b", "32x32b"):
        return num
    if shape == "16x128b":
        if num > 64:
            raise ValueError(f"shape 16x128b requires num within [1, 64], got {num}")
        return 2 * num
    if shape == "16x256b":
        if num > 32:
            raise ValueError(f"shape 16x256b requires num within [1, 32], got {num}")
        return 4 * num
    raise ValueError(
        f"invalid shape {shape!r}, expected one of [16x32bx2, 16x64b, 32x32b, 16x128b, 16x256b]"
    )


_LD_SHAPE1 = ("16x64b", "16x128b", "16x256b", "32x32b")
_LD_SHAPE2 = ("16x32bx2",)


def _ld_parts(*args):
    # args layout: *reg_addrs, taddr, row_offset, col_offset, shape, num, pack
    shape = parse_str(args[-3])
    num = int(args[-2])
    pack_raw = args[-1]
    pack = bool(int(pack_raw)) if hasattr(pack_raw, "value") else bool(pack_raw)
    n_regs = _tcgen05_ld_st_n_regs(shape, num)
    pack_str = ".pack::16b" if pack else ""
    name = f"tvm_builtin_ptx_tcgen05_ld_{_safe(shape)}_x{num}{'_pack' if pack else ''}"
    sig_parts = [f"void* reg{i}" for i in range(n_regs)]
    sig_parts.extend(["uint32_t taddr", "uint32_t row_offset", "uint32_t col_offset"])
    sig = "(" + ", ".join(sig_parts) + ")"
    regs_slots = ", ".join(f"%{i}" for i in range(n_regs))
    reg_constraints = ", ".join(f'"=r"(*(uint32_t*)reg{i})' for i in range(n_regs))
    imm_arg = f", {2 * num if pack else num}" if shape == "16x32bx2" else ""
    instr = f"tcgen05.ld.sync.aligned.{shape}.x{num}{pack_str}.b32"
    body = (
        "    asm volatile(\n"
        f'        "{instr} "\n'
        f'        "{{{regs_slots}}}, "\n'
        f'        "[%{n_regs}]{imm_arg};\\n"\n'
        f"        :  {reg_constraints}\n"
        '        :  "r"(get_tmem_addr(taddr, row_offset, col_offset))\n'
        "        :\n"
        "    );"
    )
    return name, sig, body


def _register_ld_form(form_op, shapes):
    def _validated_parts(*args):
        shape = parse_str(args[-3])
        if shape not in shapes:
            raise ValueError(f"shape {shape!r} not in {shapes}")
        return _ld_parts(*args)

    device_intrinsic(
        form_op,
        n_attrs=3,
        helper_name=lambda *a: _validated_parts(*a)[0],
        c_signature=lambda *a: _validated_parts(*a)[1],
        body=lambda *a: _validated_parts(*a)[2],
        extra_deps=("get_tmem_addr",),
    )


_register_ld_form("ptx_tcgen05_ld_shape1", _LD_SHAPE1)
_register_ld_form("ptx_tcgen05_ld_shape2", _LD_SHAPE2)


@register_codegen("ptx_tcgen05_ld")
def codegen_ptx_tcgen05_ld(src_addr, row_offset, col_offset, shape, num, pack, *regs):
    shape = parse_str(shape)
    num = validate_power_of_two_range(num, 1, 128, "repeat factor of ptx_tcgen05_ld")
    pack = bool(pack)
    expected_n_regs = _tcgen05_ld_st_n_regs(shape, num)
    if len(regs) != expected_n_regs:
        raise ValueError(
            "The number of arguments for ptx_tcgen05_ld is incorrect, expected "
            f"{6 + expected_n_regs} total args (meaning {expected_n_regs} register args), "
            f"but got {len(regs)} register args."
        )
    op = "ptx_tcgen05_ld_shape2" if shape == "16x32bx2" else "ptx_tcgen05_ld_shape1"
    reg_addrs = [tvm.tirx.address_of(reg) for reg in regs]
    return CODEGEN_REGISTRY[f"tirx.{op}"](
        [*reg_addrs, src_addr, row_offset, col_offset, shape, num, pack]
    )


def _st_parts(*args):
    # args layout: taddr, row_offset, col_offset, *reg_addrs, shape, num, unpack
    shape = parse_str(args[-3])
    num = int(args[-2])
    unpack_raw = args[-1]
    unpack = bool(int(unpack_raw)) if hasattr(unpack_raw, "value") else bool(unpack_raw)
    n_regs = _tcgen05_ld_st_n_regs(shape, num)
    unpack_str = ".unpack::16b" if unpack else ""
    name = f"tvm_builtin_ptx_tcgen05_st_{_safe(shape)}_x{num}{'_unpack' if unpack else ''}"
    sig_parts = ["uint32_t taddr", "uint32_t row_offset", "uint32_t col_offset"]
    sig_parts.extend(f"void* reg{i}" for i in range(n_regs))
    sig = "(" + ", ".join(sig_parts) + ")"
    regs_slots = ", ".join(f"%{i + 1}" for i in range(n_regs))
    reg_constraints = ", ".join(f'"r"(*(uint32_t*)reg{i})' for i in range(n_regs))
    imm_arg = f", {2 * num if unpack else num}" if shape == "16x32bx2" else ""
    instr = f"tcgen05.st.sync.aligned.{shape}.x{num}{unpack_str}.b32"
    body = (
        "    asm volatile(\n"
        f'        "{instr} "\n'
        f'        "[%0]{imm_arg}, "\n'
        f'        "{{{regs_slots}}};\\n"\n'
        "        :\n"
        f'        :  "r"(get_tmem_addr(taddr, row_offset, col_offset)), {reg_constraints}\n'
        "    );"
    )
    return name, sig, body


def _register_st_form(form_op, shapes):
    def _validated_parts(*args):
        shape = parse_str(args[-3])
        if shape not in shapes:
            raise ValueError(f"shape {shape!r} not in {shapes}")
        return _st_parts(*args)

    device_intrinsic(
        form_op,
        n_attrs=3,
        helper_name=lambda *a: _validated_parts(*a)[0],
        c_signature=lambda *a: _validated_parts(*a)[1],
        body=lambda *a: _validated_parts(*a)[2],
        extra_deps=("get_tmem_addr",),
    )


_register_st_form("ptx_tcgen05_st_shape1", _LD_SHAPE1)
_register_st_form("ptx_tcgen05_st_shape2", _LD_SHAPE2)


@register_codegen("ptx_tcgen05_st")
def codegen_ptx_tcgen05_st(dst_addr, row_offset, col_offset, shape, num, unpack, *regs):
    shape = parse_str(shape)
    num = validate_power_of_two_range(num, 1, 128, "repeat factor of ptx_tcgen05_st")
    unpack = bool(unpack)
    expected_n_regs = _tcgen05_ld_st_n_regs(shape, num)
    if len(regs) != expected_n_regs:
        raise ValueError(
            "The number of arguments for ptx_tcgen05_st is incorrect, expected "
            f"{6 + expected_n_regs} total args (meaning {expected_n_regs} register args), "
            f"but got {len(regs)} register args."
        )
    op = "ptx_tcgen05_st_shape2" if shape == "16x32bx2" else "ptx_tcgen05_st_shape1"
    reg_addrs = [tvm.tirx.address_of(reg) for reg in regs]
    return CODEGEN_REGISTRY[f"tirx.{op}"](
        [dst_addr, row_offset, col_offset, *reg_addrs, shape, num, unpack]
    )


# =============================================================================
# tcgen05 SMEM / instr descriptor encoders — pure-C bitfield struct fills.
# =============================================================================
device_intrinsic(
    "ptx_tcgen05_encode_matrix_descriptor",
    helper_name="tvm_builtin_ptx_tcgen05_encode_matrix_descriptor",
    c_signature="(uint64_t* desc, void* addr, int ldo, int sdo, int swizzle)",
    body=(
        "  SmemDescriptor _desc{};  // value-init: reading uncovered pad bits is UB\n"
        "\n"
        "  _desc.version_ = 1;\n"
        "  _desc.lbo_mode_ = 0;\n"
        "\n"
        "  switch (swizzle) {\n"
        "    case 0: _desc.layout_type_ = uint8_t(0); break; // No swizzle\n"
        "    case 1: _desc.layout_type_ = uint8_t(6); break; // 32B swizzle\n"
        "    case 2: _desc.layout_type_ = uint8_t(4); break; // 64B swizzle\n"
        "    case 3: _desc.layout_type_ = uint8_t(2); break; // 128B swizzle\n"
        "    case 4: _desc.layout_type_ = uint8_t(1); break; // 128B_base32B swizzle\n"
        "  }\n"
        "\n"
        "  uint32_t start_address = __cvta_generic_to_shared(addr);\n"
        "  _desc.start_address_ = static_cast<uint16_t>(start_address >> 4);\n"
        "\n"
        "  constexpr uint8_t base_offset = 0;\n"
        "  _desc.base_offset_ = base_offset;\n"
        "\n"
        "  _desc.stride_byte_offset_  = static_cast<uint32_t>(sdo);\n"
        "  _desc.leading_byte_offset_ = static_cast<uint32_t>(ldo);\n"
        "\n"
        "  *desc = (uint64_t)_desc;"
    ),
    extra_deps=("smem_descriptor",),
)


# Dtype sets used to classify tcgen05 MMA variants.
_FP8_FAMILY = frozenset(
    {
        PTXDataType.FLOAT8_E4M3FN,
        PTXDataType.FLOAT8_E4M3FNUZ,
        PTXDataType.FLOAT8_E5M2,
        PTXDataType.FLOAT6_E2M3FN,
        PTXDataType.FLOAT6_E3M2FN,
        PTXDataType.FLOAT4_E2M1FN,
    }
)
_E8M0 = frozenset({PTXDataType.FLOAT8_E8M0FNU})
_E4M3 = frozenset({PTXDataType.FLOAT8_E4M3FN, PTXDataType.FLOAT8_E4M3FNUZ})


_TCGEN05_MMA_RULES = (
    (
        "f16",
        frozenset({PTXDataType.FLOAT16}),
        frozenset({PTXDataType.FLOAT16}),
        frozenset({PTXDataType.FLOAT16}),
        False,
        None,
        None,
    ),
    (
        "f16",
        frozenset({PTXDataType.FLOAT32}),
        frozenset({PTXDataType.FLOAT16, PTXDataType.BFLOAT16}),
        frozenset({PTXDataType.FLOAT16, PTXDataType.BFLOAT16}),
        False,
        None,
        None,
    ),
    (
        "tf32",
        frozenset({PTXDataType.FLOAT32}),
        frozenset({PTXDataType.TENSOR_FLOAT32}),
        frozenset({PTXDataType.TENSOR_FLOAT32}),
        False,
        None,
        None,
    ),
    (
        "i8",
        frozenset({PTXDataType.INT32}),
        frozenset({PTXDataType.INT8, PTXDataType.UINT8}),
        frozenset({PTXDataType.INT8, PTXDataType.UINT8}),
        False,
        None,
        None,
    ),
    (
        "f8f6f4",
        frozenset({PTXDataType.FLOAT32, PTXDataType.FLOAT16}),
        _FP8_FAMILY,
        _FP8_FAMILY,
        False,
        None,
        None,
    ),
    (
        "mxf4",
        frozenset({PTXDataType.FLOAT32}),
        frozenset({PTXDataType.FLOAT4_E2M1FN}),
        frozenset({PTXDataType.FLOAT4_E2M1FN}),
        True,
        _E8M0,
        _E8M0,
    ),
    (
        "mxf4nvf4",
        frozenset({PTXDataType.FLOAT32}),
        frozenset({PTXDataType.FLOAT4_E2M1FN}),
        frozenset({PTXDataType.FLOAT4_E2M1FN}),
        True,
        _E4M3,
        _E4M3,
    ),
    ("mxf8f6f4", frozenset({PTXDataType.FLOAT32}), _FP8_FAMILY, _FP8_FAMILY, True, _E8M0, _E8M0),
)


def _get_tcgen05_mma_kind(d_dtype, a_dtype, b_dtype, sfa_dtype="", sfb_dtype=""):
    d = PTXDataType.from_string(d_dtype)
    a = PTXDataType.from_string(a_dtype)
    b = PTXDataType.from_string(b_dtype)
    has_sf = bool(sfa_dtype) and bool(sfb_dtype)
    sfa = PTXDataType.from_string(sfa_dtype) if sfa_dtype else None
    sfb = PTXDataType.from_string(sfb_dtype) if sfb_dtype else None

    for kind, d_in, a_in, b_in, sf_required, sfa_in, sfb_in in _TCGEN05_MMA_RULES:
        if d not in d_in or a not in a_in or b not in b_in:
            continue
        if sf_required != has_sf:
            continue
        if sf_required and (sfa not in sfa_in or sfb not in sfb_in):
            continue
        return kind

    raise ValueError(
        f"Invalid multiplicand data types for Tcgen05 MMA, check failed for d: {d_dtype}, "
        f"a: {a_dtype}, b: {b_dtype}, scale_a: {sfa_dtype}, scale_b: {sfb_dtype}"
    )


_TCGEN05_MMA_SHAPE_RULES = (
    (frozenset({"f16", "tf32", "f8f6f4"}), 1, {64: 8, 128: 16}, frozenset()),
    (frozenset({"f16", "tf32", "f8f6f4"}), 2, {128: 32, 256: 32}, frozenset()),
    (frozenset({"i8"}), 1, {64: 16, 128: 16}, frozenset({8, 24})),
    (frozenset({"i8"}), 2, {128: 32, 256: 32}, frozenset()),
    (frozenset({"mxf8f6f4", "mxf4", "mxf4nvf4"}), 1, {128: 8}, frozenset()),
    (frozenset({"mxf8f6f4", "mxf4", "mxf4nvf4"}), 2, {128: 16, 256: 16}, frozenset()),
)

_TCGEN05_MMA_K = {
    "f16": (16, 32),
    "tf32": (8, 16),
    "f8f6f4": (32, 64),
    "i8": (32, 64),
    "mxf8f6f4": (32, 64),
    "mxf4": (64, 128),
    "mxf4nvf4": (64, 128),
}


def _check_tcgen05_mma_matrix_shape(kind, cta_group, m, n, k, is_sparse):
    err = (
        f"Invalid matrix shape for Tcgen05 MMA, check failed for kind: {kind}, "
        f"is_sparse: {is_sparse}, cta_group: {cta_group}, M: {m}, N: {n}, K: {k}"
    )

    for kinds, cg, m_to_n_step, extra_ns in _TCGEN05_MMA_SHAPE_RULES:
        if kind not in kinds or cg != cta_group:
            continue
        if kind in {"mxf8f6f4", "mxf4", "mxf4nvf4"} and cta_group == 2 and is_sparse and m != 256:
            raise ValueError(err)
        if m not in m_to_n_step:
            raise ValueError(err)
        n_step = m_to_n_step[m]
        if n not in extra_ns and not (n_step <= n <= 256 and n % n_step == 0):
            raise ValueError(err)
        break
    else:
        raise ValueError(err)

    k_pair = _TCGEN05_MMA_K.get(kind)
    if k_pair is None:
        raise ValueError(err)
    k_dense, k_sparse = k_pair
    expected_k = k_sparse if is_sparse else k_dense
    if k != expected_k:
        raise ValueError(err)

    return True


# tcgen05 instr-descriptor (dense) encoder.
device_intrinsic(
    "_ptx_tcgen05_encode_instr_descriptor_impl",
    helper_name="ptx_tcgen05_encode_instr_descriptor",
    c_signature=(
        "(uint32_t* desc, int M, int N, int d_format, int a_format, int b_format, "
        "bool trans_a, bool trans_b, bool neg_a, bool neg_b, bool sat_d, bool is_sparse)"
    ),
    body=(
        "  InstrDescriptor _desc{};  // value-init: reading uncovered pad bits is UB\n"
        "\n"
        "  _desc.a_format_ = uint8_t(a_format);\n"
        "  _desc.b_format_ = uint8_t(b_format);\n"
        "  _desc.c_format_ = uint8_t(d_format);\n"
        "\n"
        "  _desc.m_dim_ = (M >> 4);\n"
        "  _desc.n_dim_ = (N >> 3);\n"
        "\n"
        "  _desc.a_major_ = static_cast<uint8_t>(trans_a);\n"
        "  _desc.b_major_ = static_cast<uint8_t>(trans_b);\n"
        "\n"
        "  _desc.a_negate_ = static_cast<uint8_t>(neg_a);\n"
        "  _desc.b_negate_ = static_cast<uint8_t>(neg_b);\n"
        "  _desc.saturate_ = static_cast<uint8_t>(sat_d);\n"
        "\n"
        "  _desc.sparse_flag_ = is_sparse;\n"
        "  _desc.sparse_id2_  = 0;                          // should modify in sparse case\n"
        "\n"
        "  _desc.max_shift_ = uint8_t(0);                   // WS not used\n"
        "\n"
        "  *desc = (uint32_t)_desc;"
    ),
    extra_deps=("instr_descriptor",),
)


@register_codegen("ptx_tcgen05_encode_instr_descriptor")
def codegen_ptx_tcgen05_encode_instr_descriptor(
    desc,
    d_dtype,
    a_dtype,
    b_dtype,
    M,
    N,
    K,
    trans_a,
    trans_b,
    n_cta_group,
    neg_a,
    neg_b,
    sat_d,
    is_sparse,
):
    """Validate dtype combinations and shape, translate dtypes to PTX format
    integers, then forward to the schema-driven impl."""
    a_dtype = parse_str(a_dtype)
    b_dtype = parse_str(b_dtype)
    d_dtype = parse_str(d_dtype)
    M = int(M)
    N = int(N)
    K = int(K)
    n_cta_group = validate_cta_group(n_cta_group)
    trans_a = bool(trans_a)
    trans_b = bool(trans_b)
    neg_a = bool(neg_a)
    neg_b = bool(neg_b)
    sat_d = bool(sat_d)
    is_sparse = bool(is_sparse)

    kind = _get_tcgen05_mma_kind(d_dtype, a_dtype, b_dtype)
    if kind not in ["f16", "tf32", "f8f6f4", "i8"]:
        raise ValueError(
            f"Check failed for Data Type Kind. d_dtype: {d_dtype}, a_dtype: {a_dtype}, b_dtype: {b_dtype}"  # noqa: E501
        )
    if not _check_tcgen05_mma_matrix_shape(kind, n_cta_group, M, N, K, is_sparse):
        raise ValueError(f"Invalid matrix shape ({M}, {N}, {K}) for kind '{kind}'")

    format_map = {
        PTXDataType.FLOAT16: 0,
        PTXDataType.BFLOAT16: 1,
        PTXDataType.TENSOR_FLOAT32: 2,
        PTXDataType.FLOAT8_E4M3FN: 0,
        PTXDataType.FLOAT8_E4M3FNUZ: 0,
        PTXDataType.FLOAT8_E5M2: 1,
        PTXDataType.FLOAT6_E2M3FN: 3,
        PTXDataType.FLOAT6_E3M2FN: 4,
        PTXDataType.FLOAT4_E2M1FN: 5,
        PTXDataType.UINT8: 0,
        PTXDataType.INT8: 1,
        PTXDataType.FLOAT32: 1,
        PTXDataType.INT32: 2,
    }
    dtype = PTXDataType.from_string(d_dtype)
    atype = PTXDataType.from_string(a_dtype)
    btype = PTXDataType.from_string(b_dtype)
    d_format = format_map[dtype]
    a_format = format_map[atype]
    b_format = format_map[btype]

    valid_dtypes_for_trans = {
        PTXDataType.FLOAT8_E4M3FN,
        PTXDataType.FLOAT8_E4M3FNUZ,
        PTXDataType.FLOAT8_E5M2,
        PTXDataType.INT8,
        PTXDataType.UINT8,
        PTXDataType.FLOAT16,
        PTXDataType.BFLOAT16,
        PTXDataType.TENSOR_FLOAT32,
    }
    if trans_a and atype not in valid_dtypes_for_trans:
        raise ValueError(f"Invalid a_dtype for transpose: {a_dtype}")
    if trans_b and btype not in valid_dtypes_for_trans:
        raise ValueError(f"Invalid b_dtype for transpose: {b_dtype}")
    if (neg_a or neg_b) and kind not in ["f16", "tf32", "f8f6f4"]:
        raise ValueError(f"Invalid kind for negate: {kind}")
    if sat_d and kind != "i8":
        raise ValueError(f"Invalid kind for saturate: {kind}")

    return CODEGEN_REGISTRY["tirx._ptx_tcgen05_encode_instr_descriptor_impl"](
        [desc, M, N, d_format, a_format, b_format, trans_a, trans_b, neg_a, neg_b, sat_d, is_sparse]
    )


# tcgen05 instr-descriptor (block-scaled) encoder.
device_intrinsic(
    "_ptx_tcgen05_encode_instr_descriptor_block_scaled_impl",
    helper_name="ptx_tcgen05_encode_instr_descriptor_block_scaled",
    c_signature=(
        "(uint32_t* desc, int M, int N, int a_format, int b_format, int s_format, "
        "bool trans_a, bool trans_b, bool neg_a, bool neg_b, bool is_sparse)"
    ),
    body=(
        "  InstrDescriptorBlockScaled _desc{};"
        "  // value-init: reading uncovered pad bits is UB\n"
        "\n"
        "  _desc.a_format_ = uint8_t(a_format);\n"
        "  _desc.b_format_ = uint8_t(b_format);\n"
        "  _desc.scale_format_ = uint8_t(s_format);\n"
        "\n"
        "  _desc.a_sf_id_ = 0;\n"
        "  _desc.b_sf_id_ = 0;\n"
        "\n"
        "  _desc.m_dim_ = (M >> 4);\n"
        "  _desc.n_dim_ = (N >> 3);\n"
        "\n"
        "  _desc.a_major_ = static_cast<uint8_t>(trans_a);\n"
        "  _desc.b_major_ = static_cast<uint8_t>(trans_b);\n"
        "\n"
        "  _desc.a_negate_ = static_cast<uint8_t>(neg_a);\n"
        "  _desc.b_negate_ = static_cast<uint8_t>(neg_b);\n"
        "\n"
        "  _desc.sparse_flag_ = is_sparse;\n"
        "  _desc.sparse_id2_  = 0;                          // should modify in sparse case\n"
        "\n"
        "  *desc = (uint32_t)_desc;"
    ),
    extra_deps=("instr_descriptor_block_scaled",),
)


@register_codegen("ptx_tcgen05_encode_instr_descriptor_block_scaled")
def codegen_ptx_tcgen05_encode_instr_descriptor_block_scaled(
    desc,
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    sfa_tmem_addr,
    sfb_tmem_addr,
    M,
    N,
    K,
    trans_a,
    trans_b,
    n_cta_group,
    neg_a,
    neg_b,
    is_sparse,
):
    a_dtype = parse_str(a_dtype)
    b_dtype = parse_str(b_dtype)
    d_dtype = parse_str(d_dtype)
    sfa_dtype = parse_str(sfa_dtype)
    sfb_dtype = parse_str(sfb_dtype)
    M = int(M)
    N = int(N)
    K = int(K)
    n_cta_group = validate_cta_group(n_cta_group)
    trans_a = bool(trans_a)
    trans_b = bool(trans_b)
    neg_a = bool(neg_a)
    neg_b = bool(neg_b)
    is_sparse = bool(is_sparse)

    kind = _get_tcgen05_mma_kind(d_dtype, a_dtype, b_dtype, sfa_dtype, sfb_dtype)
    valid_kinds = {"mxf8f6f4", "mxf4", "mxf4nvf4"}
    if kind not in valid_kinds:
        raise ValueError(
            f"Check failed for Data Type Kind. Expected one of {valid_kinds}, but got '{kind}' "
            f"for d:{d_dtype}, a:{a_dtype}, b:{b_dtype}, sfa:{sfa_dtype}, sfb:{sfb_dtype}"
        )

    _check_tcgen05_mma_matrix_shape(kind, n_cta_group, M, N, K, is_sparse)

    format_map = {
        PTXDataType.FLOAT8_E4M3FN: 0,
        PTXDataType.FLOAT8_E4M3FNUZ: 0,
        PTXDataType.FLOAT8_E5M2: 1,
        PTXDataType.FLOAT6_E2M3FN: 3,
        PTXDataType.FLOAT6_E3M2FN: 4,
        PTXDataType.FLOAT4_E2M1FN: 5,
    }
    format_map_sf = {
        PTXDataType.FLOAT8_E4M3FN: 0,
        PTXDataType.FLOAT8_E4M3FNUZ: 0,
        PTXDataType.FLOAT8_E8M0FNU: 1,
    }
    atype_enum = PTXDataType.from_string(a_dtype)
    btype_enum = PTXDataType.from_string(b_dtype)
    stype_enum = PTXDataType.from_string(sfa_dtype)

    if kind == "mxf8f6f4":
        a_format = format_map[atype_enum]
        b_format = format_map[btype_enum]
    else:
        a_format = 1
        b_format = 1

    s_format = format_map_sf[stype_enum]

    valid_dtypes_for_trans = {
        PTXDataType.FLOAT8_E4M3FN,
        PTXDataType.FLOAT8_E4M3FNUZ,
        PTXDataType.FLOAT8_E5M2,
    }
    if trans_a and atype_enum not in valid_dtypes_for_trans:
        raise ValueError(f"Invalid a_dtype for transpose: {a_dtype}")
    if trans_b and btype_enum not in valid_dtypes_for_trans:
        raise ValueError(f"Invalid b_dtype for transpose: {b_dtype}")

    return CODEGEN_REGISTRY["tirx._ptx_tcgen05_encode_instr_descriptor_block_scaled_impl"](
        [desc, M, N, a_format, b_format, s_format, trans_a, trans_b, neg_a, neg_b, is_sparse]
    )


# =============================================================================
# tcgen05.mma — 2 PTX form table entries (FP forms 1 / Int form 5) plus block-
# scaled (form 2). Each form is one device_intrinsic; the C signature and
# body both depend on (sparse, use_a_tmem, cta_group, scale_input_d).
# =============================================================================


def _mma_dense_parts(*args):
    """Compute (name, sig, body) for tcgen05.mma forms 1 + 5.

    Args layout: (d_tmem_addr, a_operand, b_desc[, sp_tmem_addr], i_desc,
                  enable_input_d, mask0..maskN-1[, pred],
                  kind, sparse, use_a_tmem, cta_group, scale_input_d, has_pred)
    """
    attrs = args[-6:]
    kind = parse_str(attrs[0])
    sparse_raw = attrs[1]
    sparse = bool(int(sparse_raw)) if hasattr(sparse_raw, "value") else bool(sparse_raw)
    use_a_tmem_raw = attrs[2]
    use_a_tmem = (
        bool(int(use_a_tmem_raw)) if hasattr(use_a_tmem_raw, "value") else bool(use_a_tmem_raw)
    )
    cta_group = int(attrs[3])
    scale_input_d = int(attrs[4])
    has_pred = bool(int(attrs[5]))

    if not 0 <= scale_input_d <= 15:
        raise ValueError(
            f"scale_input_d is incorrect, expected a value within [0, 15], got {scale_input_d}"
        )
    if scale_input_d > 0 and kind not in {"f16", "tf32"}:
        raise ValueError(f"scale_input_d is only valid for kind 'f16' or 'tf32', not '{kind!r}'")
    if scale_input_d > 0 and kind == "i8":
        raise ValueError("Int form: scale_input_d not supported (only valid for f16/tf32)")

    num_masks = 8 if cta_group == 2 else 4
    a_type = "uint32_t" if use_a_tmem else "uint64_t"
    a_constraint = "r" if use_a_tmem else "l"

    # Build C signature.
    sig_parts = ["uint32_t d_tmem_addr", f"{a_type} a_operand", "uint64_t b_desc"]
    if sparse:
        sig_parts.append("uint32_t sp_tmem_addr")
    sig_parts.extend(["uint32_t i_desc", "uint32_t scaleC"])
    sig_parts.extend(f"uint32_t mask{i}" for i in range(num_masks))
    if has_pred:
        sig_parts.append("uint32_t pred")
    sig = "(" + ", ".join(sig_parts) + ")"

    # Helper name.
    name = (
        f"ptx_tcgen05_mma_cta_{cta_group}_kind_{kind}"
        f"{'_sp' if sparse else ''}{'_TS' if use_a_tmem else '_SS'}"
        f"{('_' + str(scale_input_d)) if scale_input_d > 0 else ''}"
        f"{'_pred' if has_pred else ''}"
    )

    # Body — slot layout depends on sparse.
    if sparse:
        p_idx = 5
        sparse_suffix = ".sp"
        sp_str = "[%3], %4,"
        mask_start = 6
    else:
        p_idx = 4
        sparse_suffix = ""
        sp_str = "%3,"
        mask_start = 5
    a_str = "[%1]" if use_a_tmem else "%1"

    mask_phs = ", ".join(f"%{mask_start + i}" for i in range(num_masks))
    scale_ph = f", %{mask_start + num_masks}" if scale_input_d > 0 else ""
    pred_idx = mask_start + num_masks + (1 if scale_input_d > 0 else 0)

    asm_inputs = ['"r"(d_tmem_addr)', f'"{a_constraint}"(a_operand)', '"l"(b_desc)']
    if sparse:
        asm_inputs.append('"r"(sp_tmem_addr)')
    asm_inputs.extend(['"r"(i_desc)', '"r"(scaleC)'])
    asm_inputs.extend(f'"r"(mask{i})' for i in range(num_masks))
    if scale_input_d > 0:
        asm_inputs.append(f'"n"({scale_input_d})')
    if has_pred:
        asm_inputs.append('"r"(pred)')
    inputs_str = ", ".join(asm_inputs)

    instr = (
        f"tcgen05.mma{sparse_suffix}.cta_group::{cta_group}.kind::{kind}"
        f" [%0], {a_str}, %2, {sp_str}"
    )
    pred_prefix = "@p_issue " if has_pred else ""
    pred_reg = ", p_issue" if has_pred else ""
    pred_setp = f'        "setp.ne.b32 p_issue, %{pred_idx}, 0;\\n"\n' if has_pred else ""
    body = (
        "    asm volatile(\n"
        '        "{\\n"\n'
        f'        ".reg .pred p{pred_reg};\\n"\n'
        f'        "setp.ne.b32 p, %{p_idx}, 0;\\n"\n'
        f"{pred_setp}"
        f'        "{pred_prefix}{instr} "\n'
        f'        "{{{mask_phs}}}, p{scale_ph};\\n"\n'
        '        "}\\n"\n'
        "        :\n"
        f"        : {inputs_str}\n"
        "    );"
    )
    return name, sig, body


for _form_op in ("_ptx_tcgen05_mma_fp_form", "_ptx_tcgen05_mma_int_form"):
    device_intrinsic(
        _form_op,
        n_attrs=6,
        helper_name=lambda *a: _mma_dense_parts(*a)[0],
        c_signature=lambda *a: _mma_dense_parts(*a)[1],
        body=lambda *a: _mma_dense_parts(*a)[2],
    )
del _form_op


def _dispatch_tcgen05_mma(
    d_dtype,
    a_dtype,
    b_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d,
    scale_input_d,
    *disable_output_lane,
    pred=None,
    sparse=False,
    sp_tmem_addr=None,
):
    d = parse_str(d_dtype) if not isinstance(d_dtype, str) else d_dtype
    a = parse_str(a_dtype) if not isinstance(a_dtype, str) else a_dtype
    b = parse_str(b_dtype) if not isinstance(b_dtype, str) else b_dtype
    use_a_tmem_b = bool(use_a_tmem)
    cta_group_i = validate_cta_group(cta_group)
    scale_input_d_i = int(scale_input_d)
    has_pred = pred is not None

    expected_vec_size = 8 if cta_group_i == 2 else 4
    if len(disable_output_lane) != expected_vec_size:
        raise ValueError(
            "The number of arguments for ptx_tcgen05_mma is incorrect, expected "
            f"{11 + expected_vec_size} total args (meaning {expected_vec_size} lane mask args), "
            f"but got {len(disable_output_lane)}."
        )

    kind = _get_tcgen05_mma_kind(d, a, b)
    if kind in {"f16", "tf32", "f8f6f4"}:
        op = "_ptx_tcgen05_mma_fp_form"
    elif kind == "i8":
        op = "_ptx_tcgen05_mma_int_form"
    else:
        raise ValueError(
            f"tcgen05.mma: kind {kind!r} not in any supported PTX form (FP form 1 / Int form 5)"
        )

    operand_args = [d_tmem_addr, a_operand, b_desc]
    if sparse:
        operand_args.append(sp_tmem_addr)
    operand_args.extend([i_desc, enable_input_d, *disable_output_lane])
    if has_pred:
        operand_args.append(pred)

    attr_args = [kind, sparse, use_a_tmem_b, cta_group_i, scale_input_d_i, int(has_pred)]
    return CODEGEN_REGISTRY[f"tirx.{op}"](operand_args + attr_args)


@register_codegen("ptx_tcgen05_mma")
def codegen_ptx_tcgen05_mma(
    d_dtype,
    a_dtype,
    b_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d,
    scale_input_d,
    *rest,
):
    # `rest` = disable_output_lane (4 or 8) + optional pred (1 extra).
    cta_group_i = int(cta_group)
    n_lanes = 4 if cta_group_i == 1 else 8
    if len(rest) == n_lanes + 1:
        pred = rest[-1]
        disable_output_lane = rest[:-1]
    else:
        pred = None
        disable_output_lane = rest
    return _dispatch_tcgen05_mma(
        d_dtype,
        a_dtype,
        b_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
        scale_input_d,
        *disable_output_lane,
        pred=pred,
        sparse=False,
        sp_tmem_addr=None,
    )


@register_codegen("ptx_tcgen05_mma_sp")
def codegen_ptx_tcgen05_mma_sp(
    d_dtype,
    a_dtype,
    b_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    sp_tmem_addr,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d,
    scale_input_d,
    *disable_output_lane,
):
    return _dispatch_tcgen05_mma(
        d_dtype,
        a_dtype,
        b_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
        scale_input_d,
        *disable_output_lane,
        sparse=True,
        sp_tmem_addr=sp_tmem_addr,
    )


# tcgen05.mma block-scaled — form 2.


def _get_tcgen05_mma_scale_vec_size(kind, scale_dtype):
    scale_vec_size = 0
    stype = PTXDataType.from_string(scale_dtype)
    if kind == "mxf8f6f4" and stype == PTXDataType.FLOAT8_E8M0FNU:
        scale_vec_size = 1
    elif kind == "mxf4" and stype == PTXDataType.FLOAT8_E8M0FNU:
        scale_vec_size = 2
    elif kind == "mxf4nvf4" and stype == PTXDataType.FLOAT8_E8M0FNU:
        scale_vec_size = 2
    elif kind == "mxf4nvf4" and stype in {PTXDataType.FLOAT8_E4M3FN, PTXDataType.FLOAT8_E4M3FNUZ}:
        scale_vec_size = 4
    if scale_vec_size <= 0:
        raise ValueError(
            f"Invalid scale vector size for Tcgen05 MMA, check failed for kind::{kind}, "
            f"scale_dtype: {scale_dtype}"
        )
    return scale_vec_size


def _mma_block_scaled_parts(*args):
    """Args layout: (d_tmem_addr, a_operand, b_desc[, sp_tmem_addr], i_desc,
    enable_input_d, sfa_tmem_addr, sfb_tmem_addr,
    kind, scale_vec_size, sparse, use_a_tmem, cta_group)."""
    attrs = args[-5:]
    kind = parse_str(attrs[0])
    scale_vec_size = int(attrs[1])
    sparse_raw = attrs[2]
    sparse = bool(int(sparse_raw)) if hasattr(sparse_raw, "value") else bool(sparse_raw)
    use_a_tmem_raw = attrs[3]
    use_a_tmem = (
        bool(int(use_a_tmem_raw)) if hasattr(use_a_tmem_raw, "value") else bool(use_a_tmem_raw)
    )
    cta_group = int(attrs[4])

    a_type = "uint32_t" if use_a_tmem else "uint64_t"
    a_constraint = "r" if use_a_tmem else "l"

    sig_parts = ["uint32_t d_tmem_addr", f"{a_type} a_operand", "uint64_t b_desc"]
    if sparse:
        sig_parts.append("uint32_t sp_tmem_addr")
    sig_parts.extend(
        ["uint32_t i_desc", "uint32_t scaleC", "uint32_t sfa_tmem_addr", "uint32_t sfb_tmem_addr"]
    )
    sig = "(" + ", ".join(sig_parts) + ")"

    name = (
        f"ptx_tcgen05_mma_block_scaled_cta_{cta_group}_kind_{kind}_scale_vec_{scale_vec_size}"
        f"{'_sp' if sparse else ''}{'_TS' if use_a_tmem else '_SS'}"
    )

    sparse_suffix = ".sp" if sparse else ""
    sparse_placeholder = "[%7], " if sparse else ""
    a_str = "[%1]" if use_a_tmem else "%1"
    sp_input = ', "r"(sp_tmem_addr)' if sparse else ""
    instr = (
        f"tcgen05.mma{sparse_suffix}.cta_group::{cta_group}.kind::{kind}"
        f".block_scale.scale_vec::{scale_vec_size}X"
    )
    asm_inputs = (
        f'"r"(d_tmem_addr), "{a_constraint}"(a_operand), "l"(b_desc),'
        f' "r"(i_desc), "r"(scaleC), "r"(sfa_tmem_addr), "r"(sfb_tmem_addr)'
        f"{sp_input}"
    )
    body = (
        "    asm volatile(\n"
        '        "{\\n"\n'
        '        ".reg .pred p;\\n"\n'
        '        "setp.ne.b32 p, %4, 0;\\n"\n'
        f'        "{instr} "\n'
        f'        "[%0], {a_str}, %2, {sparse_placeholder}%3, [%5], [%6], p;\\n"\n'
        '        "}\\n"\n'
        "        :\n"
        f"        : {asm_inputs}\n"
        "    );"
    )
    return name, sig, body


device_intrinsic(
    "_ptx_tcgen05_mma_block_scaled_form",
    n_attrs=5,
    helper_name=lambda *a: _mma_block_scaled_parts(*a)[0],
    c_signature=lambda *a: _mma_block_scaled_parts(*a)[1],
    body=lambda *a: _mma_block_scaled_parts(*a)[2],
)


def _dispatch_tcgen05_mma_block_scaled(
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    sfa_tmem_addr,
    sfb_tmem_addr,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d,
    sparse=False,
    sp_tmem_addr=None,
):
    d_dtype_s = parse_str(d_dtype)
    a_dtype_s = parse_str(a_dtype)
    b_dtype_s = parse_str(b_dtype)
    sfa_dtype_s = parse_str(sfa_dtype)
    sfb_dtype_s = parse_str(sfb_dtype)
    use_a_tmem_b = bool(use_a_tmem)
    cta_group_i = validate_cta_group(cta_group)

    kind = _get_tcgen05_mma_kind(d_dtype_s, a_dtype_s, b_dtype_s, sfa_dtype_s, sfb_dtype_s)
    valid_kinds = {"mxf8f6f4", "mxf4", "mxf4nvf4"}
    if kind not in valid_kinds:
        raise ValueError(
            f"Check failed for Data Type Kind. Expected one of {valid_kinds}, but got '{kind}' "
            f"for d:{d_dtype_s}, a:{a_dtype_s}, b:{b_dtype_s}, sfa:{sfa_dtype_s}, sfb:{sfb_dtype_s}"
        )

    scale_vec_size = _get_tcgen05_mma_scale_vec_size(kind, sfa_dtype_s)

    operand_args = [d_tmem_addr, a_operand, b_desc]
    if sparse:
        operand_args.append(sp_tmem_addr)
    operand_args.extend([i_desc, enable_input_d, sfa_tmem_addr, sfb_tmem_addr])

    attr_args = [kind, scale_vec_size, sparse, use_a_tmem_b, cta_group_i]
    return CODEGEN_REGISTRY["tirx._ptx_tcgen05_mma_block_scaled_form"](operand_args + attr_args)


@register_codegen("ptx_tcgen05_mma_block_scale")
def codegen_ptx_tcgen05_mma_block_scale(
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    sfa_tmem_addr,
    sfb_tmem_addr,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d=1,
):
    return _dispatch_tcgen05_mma_block_scaled(
        d_dtype,
        a_dtype,
        b_dtype,
        sfa_dtype,
        sfb_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        sfa_tmem_addr,
        sfb_tmem_addr,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
    )


@register_codegen("ptx_tcgen05_mma_sp_block_scale")
def codegen_ptx_tcgen05_mma_sp_block_scale(
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    d_tmem_addr,
    a_operand,
    b_desc,
    sfa_tmem_addr,
    sfb_tmem_addr,
    sp_tmem_addr,
    i_desc,
    use_a_tmem,
    cta_group,
    enable_input_d=1,
):
    return _dispatch_tcgen05_mma_block_scaled(
        d_dtype,
        a_dtype,
        b_dtype,
        sfa_dtype,
        sfb_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        sfa_tmem_addr,
        sfb_tmem_addr,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
        sparse=True,
        sp_tmem_addr=sp_tmem_addr,
    )


# =============================================================================
# tcgen05.commit — 2 PTX form table entries (unicast / multicast).
# =============================================================================
device_intrinsic(
    "_ptx_tcgen05_commit_unicast",
    n_attrs=1,
    c_signature="(void* bar)",
    helper_name=lambda bar_, cta_group: f"ptx_tcgen05_commit_cta_group_{int(cta_group)}",
    body=lambda bar_, cta_group: (
        "    unsigned int bar_addr = __cvta_generic_to_shared(bar);\n"
        f'    asm volatile("tcgen05.commit.cta_group::{int(cta_group)}'
        '.mbarrier::arrive::one.shared::cluster.b64 [%0];" '
        ': : "r"(bar_addr) : "memory");'
    ),
)
device_intrinsic(
    "_ptx_tcgen05_commit_multicast",
    n_attrs=1,
    c_signature="(void* bar, uint16_t cta_mask)",
    helper_name=lambda bar_, mask_, cta_group: (
        f"ptx_tcgen05_commit_cta_group_{int(cta_group)}_multicast"
    ),
    body=lambda bar_, mask_, cta_group: (
        "    unsigned int bar_addr = __cvta_generic_to_shared(bar);\n"
        f'    asm volatile("tcgen05.commit.cta_group::{int(cta_group)}'
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        ' [%0], %1;" '
        ': : "r"(bar_addr), "h"(cta_mask) : "memory");'
    ),
)
# Predicated variants — body wraps the commit in `{ setp + @p ... }` so the
# instruction is still issued but its effect is masked by ``pred != 0`` at
# PTX level (preserves single predicated SASS instruction, not a C branch).
device_intrinsic(
    "_ptx_tcgen05_commit_unicast_predicated",
    n_attrs=1,
    c_signature="(void* bar, uint32_t pred)",
    helper_name=lambda bar_, pred_, cta_group: (
        f"ptx_tcgen05_commit_cta_group_{int(cta_group)}_predicated"
    ),
    body=lambda bar_, pred_, cta_group: (
        "    unsigned int bar_addr = __cvta_generic_to_shared(bar);\n"
        "    asm volatile(\n"
        '        "{\\n"\n'
        '        ".reg .pred p;\\n"\n'
        '        "setp.ne.b32 p, %1, 0;\\n"\n'
        f'        "@p tcgen05.commit.cta_group::{int(cta_group)}'
        '.mbarrier::arrive::one.shared::cluster.b64 [%0];\\n"\n'
        '        "}\\n"\n'
        '        : : "r"(bar_addr), "r"(pred) : "memory");'
    ),
)
device_intrinsic(
    "_ptx_tcgen05_commit_multicast_predicated",
    n_attrs=1,
    c_signature="(void* bar, uint16_t cta_mask, uint32_t pred)",
    helper_name=lambda bar_, mask_, pred_, cta_group: (
        f"ptx_tcgen05_commit_cta_group_{int(cta_group)}_multicast_predicated"
    ),
    body=lambda bar_, mask_, pred_, cta_group: (
        "    unsigned int bar_addr = __cvta_generic_to_shared(bar);\n"
        "    asm volatile(\n"
        '        "{\\n"\n'
        '        ".reg .pred p;\\n"\n'
        '        "setp.ne.b32 p, %2, 0;\\n"\n'
        f'        "@p tcgen05.commit.cta_group::{int(cta_group)}'
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        ' [%0], %1;\\n"\n'
        '        "}\\n"\n'
        '        : : "r"(bar_addr), "h"(cta_mask), "r"(pred) : "memory");'
    ),
)


@register_codegen("ptx_tcgen05_commit")
def codegen_ptx_tcgen05_commit(bar, cta_group, cta_mask, *pred_args):
    cta_group = int(cta_group)
    if cta_group not in (1, 2):
        raise ValueError(f"The number of cta_group is incorrect, expected 1 or 2, got {cta_group}")
    is_multicast = not (
        isinstance(cta_mask, tvm.tirx.IntImm) and bin(int(cta_mask)).count("1") <= 1
    )
    has_pred = len(pred_args) == 1
    if has_pred:
        suffix = "_multicast_predicated" if is_multicast else "_unicast_predicated"
        if is_multicast:
            args = [bar, cta_mask, pred_args[0], cta_group]
        else:
            args = [bar, pred_args[0], cta_group]
    else:
        suffix = "_multicast" if is_multicast else "_unicast"
        if is_multicast:
            args = [bar, cta_mask, cta_group]
        else:
            args = [bar, cta_group]
    op_name = f"tirx._ptx_tcgen05_commit{suffix}"
    result = CODEGEN_REGISTRY[op_name](args)
    return result[0] if isinstance(result, tuple) else result


# =============================================================================
# tcgen05.cp — 1 PTX form. Body folds (taddr, row_offset, col_offset) into a
# single asm input slot via ``get_tmem_addr(...)``.
# =============================================================================


def _tcgen05_cp_parts(taddr_, row_, col_, src_desc_, cta_group, shape, multicast, decompress):
    cta_group = int(cta_group)
    shape = parse_str(shape)
    multicast = parse_str(multicast)
    decompress = parse_str(decompress)
    name = (
        f"ptx_tcgen05_cp_cta_group_{cta_group}_shape_{_safe(shape)}"
        f"_multicast_{_safe(multicast)}_decompress_{_safe(decompress)}"
    )
    instr = (
        f"tcgen05.cp.cta_group::{cta_group}.{shape}"
        f"{('.' + multicast) if multicast else ''}"
        f"{('.' + decompress) if decompress else ''}"
    )
    body = (
        "    asm volatile(\n"
        f'        "{instr} [%0], %1;"\n'
        "        :\n"
        '        : "r"(get_tmem_addr(taddr, row_offset, col_offset)), "l"(src_desc)\n'
        "    );"
    )
    return name, body


device_intrinsic(
    "_ptx_tcgen05_cp_impl",
    n_attrs=4,
    c_signature="(uint32_t taddr, int row_offset, int col_offset, uint64_t src_desc)",
    helper_name=lambda *a: _tcgen05_cp_parts(*a)[0],
    body=lambda *a: _tcgen05_cp_parts(*a)[1],
    extra_deps=("get_tmem_addr",),
)


@register_codegen("ptx_tcgen05_cp")
def codegen_ptx_tcgen05_cp(taddr, src_desc, shape, cta_group, multicast, decompress, row, col):
    shape = parse_str(shape)
    multicast = parse_str(multicast)
    decompress = parse_str(decompress)
    cta_group = validate_cta_group(cta_group)
    return CODEGEN_REGISTRY["tirx._ptx_tcgen05_cp_impl"](
        [taddr, row, col, src_desc, cta_group, shape, multicast, decompress]
    )


# =============================================================================
# tcgen05 address / descriptor patch helpers — used by the dispatch wrappers
# in ``tile_primitive/cuda/gemm_async/tcgen05.py``. They live here
# (not in ``memory.py``) because their semantics are tcgen05-specific:
#   - get_tmem_addr packs a TMEM (taddr, row, col) tuple into the uint32 the
#     PTX asm slots expect.
#   - runtime_instr_desc patches the ``b_sf_id_`` (bits [4, 6)) and ``a_sf_id_``
#     (bits [29, 31)) fields of an in-flight ``InstrDescriptorBlockScaled``.
# =============================================================================
device_intrinsic(
    "cuda_get_tmem_addr",
    c_signature="(uint32_t addr, int row_offset, int col_offset)",
    body="    return get_tmem_addr(addr, row_offset, col_offset);",
    return_type="uint32_t",
    tvm_return_type="uint32",
    extra_deps=("get_tmem_addr",),
)

device_intrinsic(
    "cuda_runtime_instr_desc",
    c_signature="(uint32_t* desc, const uint32_t& sf_id)",
    body="    *desc = (*desc & ~0x60000030) | ((sf_id << 29) | (sf_id << 4));",
)
