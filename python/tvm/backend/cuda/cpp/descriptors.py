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
"""Matrix and instruction descriptor encoding for wgmma / tcgen05.

Descriptors are bitfield structs the hardware reads, so each encoder is a
pure-C struct fill over a layout declared in the CUDA header tags -- no
asm, and nothing that maps to a PTX instruction.

The tcgen05 MMA validation tables live here too: they decide the dtype
*kind* and legal (M, N, K) shape that the instruction descriptor encodes,
and ``tile_primitive/gemm_async/tcgen05.py`` reuses them when it folds the
same constraints into its dispatch.
"""

from ..codegen.registry import CODEGEN_REGISTRY, register_codegen
from ..codegen.schema import device_intrinsic
from ..codegen.types import PTXDataType
from ..codegen.utils import parse_str, validate_cta_group

# =============================================================================
# wgmma_encode_matrix_descriptor — pure-C bitfield struct fill (no asm).
# =============================================================================
device_intrinsic(
    "cuda_wgmma_encode_matrix_descriptor",
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
# tcgen05 SMEM / instr descriptor encoders — pure-C bitfield struct fills.
# =============================================================================
device_intrinsic(
    "cuda_tcgen05_encode_matrix_descriptor",
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

# Operand dtypes admitting a transposed (MN-major) SMEM read in dense
# tcgen05.mma; shared with the gemm_async instruction-descriptor fold.
_TCGEN05_MMA_TRANS_DTYPES = frozenset(
    {
        PTXDataType.FLOAT8_E4M3FN,
        PTXDataType.FLOAT8_E4M3FNUZ,
        PTXDataType.FLOAT8_E5M2,
        PTXDataType.INT8,
        PTXDataType.UINT8,
        PTXDataType.FLOAT16,
        PTXDataType.BFLOAT16,
        PTXDataType.TENSOR_FLOAT32,
    }
)


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
    "_cuda_tcgen05_encode_instr_descriptor_impl",
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


@register_codegen("cuda_tcgen05_encode_instr_descriptor")
def codegen_cuda_tcgen05_encode_instr_descriptor(
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

    if trans_a and atype not in _TCGEN05_MMA_TRANS_DTYPES:
        raise ValueError(f"Invalid a_dtype for transpose: {a_dtype}")
    if trans_b and btype not in _TCGEN05_MMA_TRANS_DTYPES:
        raise ValueError(f"Invalid b_dtype for transpose: {b_dtype}")
    if (neg_a or neg_b) and kind not in ["f16", "tf32", "f8f6f4"]:
        raise ValueError(f"Invalid kind for negate: {kind}")
    if sat_d and kind != "i8":
        raise ValueError(f"Invalid kind for saturate: {kind}")

    return CODEGEN_REGISTRY["tirx._cuda_tcgen05_encode_instr_descriptor_impl"](
        [desc, M, N, d_format, a_format, b_format, trans_a, trans_b, neg_a, neg_b, sat_d, is_sparse]
    )


# tcgen05 instr-descriptor (block-scaled) encoder.
device_intrinsic(
    "_cuda_tcgen05_encode_instr_descriptor_block_scaled_impl",
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


@register_codegen("cuda_tcgen05_encode_instr_descriptor_block_scaled")
def codegen_cuda_tcgen05_encode_instr_descriptor_block_scaled(
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

    return CODEGEN_REGISTRY["tirx._cuda_tcgen05_encode_instr_descriptor_block_scaled_impl"](
        [desc, M, N, a_format, b_format, s_format, trans_a, trans_b, neg_a, neg_b, is_sparse]
    )


# =============================================================================
# Scale-vector size for block-scaled MMA — consumed by the gemm_async
# dispatch wrappers, which pass it to the ``T.ptx`` tcgen05.mma forms.
# =============================================================================
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


# =============================================================================
# tcgen05 address / descriptor patch helpers — used by the dispatch wrappers
# in ``tile_primitive/gemm_async/tcgen05.py``. They are tcgen05-specific rather
# than generic address arithmetic:
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


# ---------------------------------------------------------------------------
# Compile-time folds of the same descriptors
#
# The encoders above are pure-C struct fills, so a call site whose descriptor
# fields are all known at build time still pays for an opaque device helper the
# generated CUDA must call and ptxas cannot hoist out of a loop. These fold the
# identical bit layouts in Python instead, so such a call site can pass a
# literal. They mirror the runtime encoders' validation exactly: a fold that
# accepted a combination the runtime encoder rejects would be a silent
# divergence between two spellings of one hardware format.
# ---------------------------------------------------------------------------


def _dtype_name(dtype) -> str:
    """Name of a dtype given as a string, a DataType, or an object holding one."""
    dtype_obj = getattr(dtype, "dtype", None)
    if dtype_obj is not None:
        return str(dtype_obj)
    return str(dtype)


# `a_format_` / `b_format_` / `d_format_`, mirroring `format_map` in the
# runtime encoders above.
_INSTR_DESC_FORMAT_MAP = {
    "float16": 0,
    "bfloat16": 1,
    "tensor_float32": 2,
    "tf32": 2,
    "float8_e4m3fn": 0,
    "float8_e4m3fnuz": 0,
    "float8_e5m2": 1,
    "float6_e2m3fn": 3,
    "float6_e3m2fn": 4,
    "float4_e2m1fn": 5,
    "uint8": 0,
    "int8": 1,
    "float32": 1,
    "int32": 2,
}

# `scale_format_` of the block-scaled descriptor: 0 = E4M3, 1 = E8M0.
_INSTR_DESC_SF_FORMAT_MAP = {
    "float8_e4m3fn": 0,
    "float8_e4m3fnuz": 0,
    "float8_e8m0fnu": 1,
}

# `layout_type_` of SmemDescriptor, keyed by the swizzle enum the runtime
# encoder switches on (0=none, 1=32B, 2=64B, 3=128B, 4=128B_base32B).
_SMEM_DESC_LAYOUT_TYPE = {0: 0, 1: 6, 2: 4, 3: 2, 4: 1}

# PTXDataType spells tensor-float32 as "tf32".
_PTX_DTYPE_ALIAS = {"tensor_float32": "tf32"}


def _check_instr_desc_trans(a_ptx_name, b_ptx_name, trans_a, trans_b):
    """Only a few source formats may be MN-major, per the descriptor's comment."""
    if trans_a and PTXDataType.from_string(a_ptx_name) not in _TCGEN05_MMA_TRANS_DTYPES:
        raise ValueError(f"Invalid a_dtype for transpose: {a_ptx_name}")
    if trans_b and PTXDataType.from_string(b_ptx_name) not in _TCGEN05_MMA_TRANS_DTYPES:
        raise ValueError(f"Invalid b_dtype for transpose: {b_ptx_name}")


def _instr_desc_common_bits(M, N, a_format, b_format, trans_a, trans_b, neg_a, neg_b, is_sparse):
    """The nine fields the dense and block-scaled descriptors share.

    They encode the same hardware bitfield and differ only in what each adds:
    the dense line owns `saturate_`/`d_format_`, the block-scaled one
    `scale_format_`. Keeping the shared bits in one place is what stops a
    correction landing in only one of them.
    """
    desc = 0
    desc |= (int(is_sparse) & 0x1) << 2
    desc |= (a_format & 0x7) << 7
    desc |= (b_format & 0x7) << 10
    desc |= (int(neg_a) & 0x1) << 13
    desc |= (int(neg_b) & 0x1) << 14
    desc |= (int(trans_a) & 0x1) << 15
    desc |= (int(trans_b) & 0x1) << 16
    desc |= ((N >> 3) & 0x3F) << 17
    desc |= ((M >> 4) & 0x1F) << 24
    return desc


def encode_instr_descriptor_dense_uint32(
    M,
    N,
    K,
    d_dtype,
    a_dtype,
    b_dtype,
    trans_a,
    trans_b,
    cta_group=1,
    neg_a=False,
    neg_b=False,
    sat_d=False,
    is_sparse=False,
):
    """Compile-time fold of `codegen_cuda_tcgen05_encode_instr_descriptor`.

    Bit layout: `InstrDescriptor` in `python/tvm/backend/cuda/codegen/header.py`.

    The validation is the runtime encoder's, deliberately: cta_group=1 M=128
    requires N % 16 == 0, which a tile chooser enforcing only N % 8 does not
    guarantee.
    """
    d_name = _dtype_name(d_dtype)
    a_name = _dtype_name(a_dtype)
    b_name = _dtype_name(b_dtype)
    d_ptx = _PTX_DTYPE_ALIAS.get(d_name, d_name)
    a_ptx = _PTX_DTYPE_ALIAS.get(a_name, a_name)
    b_ptx = _PTX_DTYPE_ALIAS.get(b_name, b_name)

    kind = _get_tcgen05_mma_kind(d_ptx, a_ptx, b_ptx)
    if kind not in ("f16", "tf32", "f8f6f4", "i8"):
        raise ValueError(
            f"Check failed for Data Type Kind. d_dtype: {d_name}, "
            f"a_dtype: {a_name}, b_dtype: {b_name}"
        )
    _check_tcgen05_mma_matrix_shape(kind, cta_group, int(M), int(N), int(K), is_sparse)
    _check_instr_desc_trans(a_ptx, b_ptx, trans_a, trans_b)

    desc = _instr_desc_common_bits(
        M,
        N,
        _INSTR_DESC_FORMAT_MAP[a_name],
        _INSTR_DESC_FORMAT_MAP[b_name],
        trans_a,
        trans_b,
        neg_a,
        neg_b,
        is_sparse,
    )
    desc |= (int(sat_d) & 0x1) << 3
    desc |= (_INSTR_DESC_FORMAT_MAP[d_name] & 0x3) << 4
    return desc & 0xFFFFFFFF


def encode_instr_descriptor_block_scaled_uint32(
    M,
    N,
    K,
    d_dtype,
    a_dtype,
    b_dtype,
    sf_dtype,
    trans_a,
    trans_b,
    cta_group=1,
    neg_a=False,
    neg_b=False,
    is_sparse=False,
):
    """Compile-time fold of `codegen_cuda_tcgen05_encode_instr_descriptor_block_scaled`.

    Bit layout: `InstrDescriptorBlockScaled` in
    `python/tvm/backend/cuda/codegen/header.py`.
    """
    d_name = _dtype_name(d_dtype)
    a_name = _dtype_name(a_dtype)
    b_name = _dtype_name(b_dtype)
    sf_name = _dtype_name(sf_dtype)
    d_ptx = _PTX_DTYPE_ALIAS.get(d_name, d_name)
    a_ptx = _PTX_DTYPE_ALIAS.get(a_name, a_name)
    b_ptx = _PTX_DTYPE_ALIAS.get(b_name, b_name)

    kind = _get_tcgen05_mma_kind(d_ptx, a_ptx, b_ptx, sf_name, sf_name)
    valid_kinds = {"mxf8f6f4", "mxf4", "mxf4nvf4"}
    if kind not in valid_kinds:
        raise ValueError(
            f"Check failed for Data Type Kind. Expected one of {valid_kinds}, but got "
            f"'{kind}' for d:{d_name}, a:{a_name}, b:{b_name}, sf:{sf_name}"
        )
    _check_tcgen05_mma_matrix_shape(kind, cta_group, int(M), int(N), int(K), is_sparse)
    _check_instr_desc_trans(a_ptx, b_ptx, trans_a, trans_b)

    # mxf4 / mxf4nvf4 pin both operand formats to 1; only mxf8f6f4 reads them
    # off the dtypes. Same branch as the runtime encoder.
    if kind == "mxf8f6f4":
        a_format = _INSTR_DESC_FORMAT_MAP[a_name]
        b_format = _INSTR_DESC_FORMAT_MAP[b_name]
    else:
        a_format = 1
        b_format = 1

    desc = _instr_desc_common_bits(
        M, N, a_format, b_format, trans_a, trans_b, neg_a, neg_b, is_sparse
    )
    desc |= (_INSTR_DESC_SF_FORMAT_MAP[sf_name] & 0x1) << 23
    # `a_sf_id_` / `b_sf_id_` stay 0, as the runtime encoder leaves them:
    # callers that cycle scale-factor ids patch those bits per MMA.
    return desc & 0xFFFFFFFF


def encode_smem_descriptor_base_uint64(ldo, sdo, swizzle):
    """Compile-time fold of `codegen_cuda_tcgen05_encode_matrix_descriptor`, minus
    the address.

    `start_address_` is the only runtime input, so the descriptor splits into a
    constant the caller bakes in and an `addr >> 4` it ORs into bits [0,14).
    Bit layout: `SmemDescriptor` in `python/tvm/backend/cuda/codegen/header.py`.

    `ldo` / `sdo` are in 16-byte units, matching what the runtime encoder
    assigns straight into the two offset fields.
    """
    desc = 0
    desc |= (int(ldo) & 0x3FFF) << 16
    desc |= (int(sdo) & 0x3FFF) << 32
    desc |= 1 << 46  # version_
    # base_offset_ [49,52) and lbo_mode_ [52,53) are zero, as the encoder sets.
    desc |= (_SMEM_DESC_LAYOUT_TYPE[int(swizzle)] & 0x7) << 61
    return desc & 0xFFFFFFFFFFFFFFFF
