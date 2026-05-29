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
"""PTX MMA / ldmatrix / stmatrix intrinsics.

mma.sync.aligned has 7 form_kinds per the PTX docs (f16 / tf32 / bf16 / fp64
/ int8 / fp8 / subbyte). Each form_kind is one ``device_intrinsic`` registration;
the (shape, layouts, dtypes) modifier slots are attrs. Body computes the per-
fragment register counts at codegen time from M*N*bits/threads/frag_size and
hand-builds the asm constraint list.

ldmatrix / stmatrix each have a single PTX form (the .m8n8 .b16/.b8 variant
that TIRx uses); ``num`` and ``trans`` are modifier attrs.
"""

import re
from dataclasses import dataclass

import tvm
from tvm import DataType

from .._schema import device_intrinsic
from .registry import CODEGEN_REGISTRY, register_codegen
from .types import PTXDataType
from .utils import parse_str


@dataclass
class FragAttrs:
    reg_type: str  # asm constraint letter (r / f / d)
    size: int  # bit width per register slot (32 or 64)
    ptr_type: str  # C type for the cast


_FRAG_ATTRS_MAP = {
    PTXDataType.BIT1: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.INT4: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.UINT4: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.INT8: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.UINT8: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.FLOAT8_E4M3FN: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.FLOAT8_E5M2: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.BIT16: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.FLOAT16: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.BFLOAT16: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.TENSOR_FLOAT32: FragAttrs("r", 32, "uint32_t"),
    PTXDataType.INT32: FragAttrs("r", 32, "int32_t"),
    PTXDataType.FLOAT32: FragAttrs("f", 32, "float"),
    PTXDataType.FLOAT64: FragAttrs("d", 64, "double"),
}


def _parse_mma_shape(shape_str):
    match = re.search(r"m(\d+)n(\d+)k(\d+)", shape_str)
    if not match:
        raise ValueError(f"Cannot parse MMA shape: {shape_str!r}")
    return tuple(map(int, match.groups()))


def _classify_mma_form(d_type, a_type, b_type):
    """Map (d, a, b) dtype triple to one of the 7 PTX form_kind tags."""
    fp16 = {"float16", "fp16"}
    tf32 = {"tensor_float32", "tf32"}
    bf16 = {"bfloat16", "bf16"}
    fp64 = {"float64", "fp64"}
    int_a = {"int8", "uint8", "s8", "u8"}
    fp8 = {"e4m3", "e5m2", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2"}
    subbyte = {"int4", "uint4", "bit1", "s4", "u4", "b1", "int1", "uint1"}
    if a_type in fp16 and b_type in fp16:
        return "f16"
    if a_type in tf32 and b_type in tf32:
        return "tf32"
    if a_type in bf16 and b_type in bf16:
        return "bf16"
    if a_type in fp64 and b_type in fp64:
        return "fp64"
    if a_type in int_a and b_type in int_a:
        return "int8"
    if a_type in fp8 and b_type in fp8:
        return "fp8"
    if a_type in subbyte and b_type in subbyte:
        return "subbyte"
    raise ValueError(
        f"Unknown ptx.mma form for d_type={d_type!r}, a_type={a_type!r}, b_type={b_type!r}"
    )


def _frag(dtype_str):
    return _FRAG_ATTRS_MAP[PTXDataType.from_string(dtype_str)]


def _mma_threads(shape, a_type):
    """Special case: m8n8k4 with f16 a/b uses 8 threads per fragment."""
    m, n, k = _parse_mma_shape(shape)
    if m == 8 and n == 8 and k == 4 and a_type == "float16":
        return 8
    return 32


# PTX dtype abbreviation -> element bit width. Used by _frag_count so that
# callers passing the PTX abbreviation (e.g. "fp32") don't blow up in
# ``DataType("fp32")``.
_PTX_BITS = {
    "fp16": 16,
    "fp32": 32,
    "fp64": 64,
    "bf16": 16,
    "tf32": 32,  # tensor-float32 packs 19 significant bits into a 32-bit slot
    "s8": 8,
    "u8": 8,
    "s32": 32,
    "s4": 4,
    "u4": 4,
    "b1": 1,
    "b16": 16,
    "e4m3": 8,
    "e5m2": 8,
}


def _frag_count(dtype, dim_a, dim_b, threads):
    if dtype in _PTX_BITS:
        bits = _PTX_BITS[dtype]
    else:
        bits = DataType(dtype).bits
    size = _frag(dtype).size
    return dim_a * dim_b * bits // threads // size


# =============================================================================
# Shared helpers for the 7 mma form_kinds.
# Args layout for each form:
#   (d_ptr_in, a_ptr_in, b_ptr_in [, c_ptr_in], shape, a_layout, b_layout,
#    d_type, a_type, b_type, c_type, no_c_ptr [, saturate or bit_op])
# n_attrs = 8 for f16/tf32/bf16/fp64/fp8 (last 8 = shape, layouts, 4 dtypes, no_c_ptr)
# n_attrs = 9 for int8 (+ saturate) and subbyte (+ bit_op)
# =============================================================================


def _mma_form_parts(args, *, has_saturate=False, has_bit_op=False):
    """Compute (helper_name, c_signature, body) for one mma form invocation.

    ``args`` is the full positional arg tuple as received by codegen.
    The trailing ``n_attrs`` (8 or 9) entries are attrs.
    """
    n_extra = (1 if has_saturate else 0) + (1 if has_bit_op else 0)
    n_attrs = 8 + n_extra
    # Split off attr args from the tail (operand args are ahead).
    attrs = args[-n_attrs:]
    shape = parse_str(attrs[0])
    a_layout = parse_str(attrs[1])
    b_layout = parse_str(attrs[2])
    d_type = parse_str(attrs[3])
    a_type = parse_str(attrs[4])
    b_type = parse_str(attrs[5])
    c_type = parse_str(attrs[6])
    no_c_ptr_raw = attrs[7]
    no_c_ptr = bool(int(no_c_ptr_raw)) if hasattr(no_c_ptr_raw, "value") else bool(no_c_ptr_raw)
    saturate = False
    bit_op = ""
    if has_saturate:
        s = attrs[8]
        saturate = bool(int(s)) if hasattr(s, "value") else bool(s)
    if has_bit_op:
        bit_op = parse_str(attrs[8])

    # Build operand-dependent C signature.
    sig_parts = ["void* d_ptr_in", "void* a_ptr_in", "void* b_ptr_in"]
    if not no_c_ptr:
        sig_parts.append("void* c_ptr_in")
    sig = "(" + ", ".join(sig_parts) + ")"

    # Helper name: shape + layouts + dtypes + flags.
    def _safe(s):
        return s.replace("::", "_").replace(".", "_")

    name = (
        f"ptx_mma_{shape}_{a_layout}_{b_layout}"
        f"_{_safe(d_type)}_{_safe(a_type)}_{_safe(b_type)}_{_safe(c_type)}"
        f"{'_no_c_ptr' if no_c_ptr else ''}"
        f"{'_saturate' if saturate else ''}"
    )

    # Body — fragment counts + asm constraint list.
    m, n, k = _parse_mma_shape(shape)
    threads = _mma_threads(shape, a_type)
    d_cnt = _frag_count(d_type, m, n, threads)
    a_cnt = _frag_count(a_type, m, k, threads)
    b_cnt = _frag_count(b_type, k, n, threads)
    c_cnt = _frag_count(c_type, m, n, threads)

    d_frag = _frag(d_type)
    a_frag = _frag(a_type)
    b_frag = _frag(b_type)
    c_frag = _frag(c_type)

    saturate_inst = ".satfinite" if saturate else ""
    # PTX b1 mma requires a `.popc` suffix after the bit op (e.g. `.xor.popc`).
    bit_op_inst = f".{bit_op}.popc" if bit_op else ""

    d_type_inst = PTXDataType.from_string(d_type).to_string()
    c_type_inst = PTXDataType.from_string(c_type).to_string()
    a_type_inst = PTXDataType.from_string(a_type).to_string()
    b_type_inst = PTXDataType.from_string(b_type).to_string()

    def _slot_arr(start, cnt):
        return "{" + ", ".join(f"%{start + i}" for i in range(cnt)) + "}"

    args_template = (
        f"{_slot_arr(0, d_cnt)}, {_slot_arr(d_cnt, a_cnt)}, "
        f"{_slot_arr(d_cnt + a_cnt, b_cnt)}, {_slot_arr(d_cnt + a_cnt + b_cnt, c_cnt)}"
    )

    d_outs = ", ".join(
        f'"=r"((({d_frag.ptr_type}*)d_ptr_in)[{i}])'
        if d_frag.reg_type == "r"
        else f'"={d_frag.reg_type}"((({d_frag.ptr_type}*)d_ptr_in)[{i}])'
        for i in range(d_cnt)
    )
    a_inputs = ", ".join(
        f'"{a_frag.reg_type}"((({a_frag.ptr_type}*)a_ptr_in)[{i}])' for i in range(a_cnt)
    )
    b_inputs = ", ".join(
        f'"{b_frag.reg_type}"((({b_frag.ptr_type}*)b_ptr_in)[{i}])' for i in range(b_cnt)
    )
    if no_c_ptr:
        c_value = "0.f" if c_frag.reg_type == "f" else "0"
        c_inputs = ", ".join(f'"{c_frag.reg_type}"({c_value})' for _ in range(c_cnt))
    else:
        c_inputs = ", ".join(
            f'"{c_frag.reg_type}"((({c_frag.ptr_type}*)c_ptr_in)[{i}])' for i in range(c_cnt)
        )

    body = (
        "    asm volatile(\n"
        f'        "mma.sync.aligned.{shape}.{a_layout}.{b_layout}{saturate_inst}'
        f'{d_type_inst}{a_type_inst}{b_type_inst}{c_type_inst}{bit_op_inst} "\n'
        f'        "{args_template};\\n"\n'
        f"        : {d_outs}\n"
        f"        : {a_inputs}, {b_inputs}, {c_inputs}\n"
        "    );"
    )
    return name, sig, body


def _register_mma_form(form_kind, *, has_saturate=False, has_bit_op=False):
    n_attrs = 8 + (1 if has_saturate else 0) + (1 if has_bit_op else 0)

    def _parts(*args, hs=has_saturate, hb=has_bit_op):
        return _mma_form_parts(args, has_saturate=hs, has_bit_op=hb)

    device_intrinsic(
        f"_ptx_mma_{form_kind}",
        n_attrs=n_attrs,
        helper_name=lambda *a: _parts(*a)[0],
        c_signature=lambda *a: _parts(*a)[1],
        body=lambda *a: _parts(*a)[2],
    )


# Form 1 — f16. Form 2 — tf32. Form 3 — bf16. Form 4 — fp64. Form 6 — fp8.
# All share the same 8-attr layout (no saturate / bit_op).
for _kind in ("f16", "tf32", "bf16", "fp64", "fp8"):
    _register_mma_form(_kind)
del _kind

# Form 5 — int8 (+ saturate).
_register_mma_form("int8", has_saturate=True)

# Form 7 — subbyte (+ bit_op for b1).
_register_mma_form("subbyte", has_bit_op=True)


@register_codegen("ptx_mma")
def codegen_ptx_mma(
    shape,
    a_layout,
    b_layout,
    d_type,
    a_type,
    b_type,
    c_type,
    d_ptr,
    a_ptr,
    b_ptr,
    c_ptr=0,
    saturate=False,
    bit_op=None,
):
    """Classify (d, a, b) dtype triple to one of 7 form_kinds and forward."""
    shape = parse_str(shape)
    a_layout = parse_str(a_layout)
    b_layout = parse_str(b_layout)
    d_type = parse_str(d_type)
    a_type = parse_str(a_type)
    b_type = parse_str(b_type)
    c_type = parse_str(c_type)
    saturate = bool(saturate)
    if isinstance(bit_op, str):
        bit_op_v = parse_str(bit_op)
    elif bit_op is None:
        bit_op_v = ""
    else:
        bit_op_v = bit_op
    if bit_op_v is None:
        bit_op_v = ""

    no_c_ptr = isinstance(c_ptr, tvm.tirx.IntImm) and int(c_ptr) == 0
    kind = _classify_mma_form(d_type, a_type, b_type)

    op_args = [d_ptr, a_ptr, b_ptr]
    if not no_c_ptr:
        op_args.append(c_ptr)

    attr_args = [shape, a_layout, b_layout, d_type, a_type, b_type, c_type, no_c_ptr]
    if kind == "int8":
        attr_args.append(saturate)
    elif kind == "subbyte":
        attr_args.append(bit_op_v)

    result = CODEGEN_REGISTRY[f"tirx._ptx_mma_{kind}"](op_args + attr_args)
    return result[0] if isinstance(result, tuple) else result


# =============================================================================
# ldmatrix / stmatrix — m8n8 fragment load/store. PTX docs lists 3 ldmatrix
# forms (m8n8 + m8n16 + m16n16); TIRx uses only the m8n8 form. 1
# device_intrinsic each. ``num`` (.x1/.x2/.x4) and ``trans`` are modifier
# attrs; the asm body loops over per-register constraints based on
# (num, dtype).
# =============================================================================


def _ldmatrix_parts(*args):
    # args = (smem_ptr, dst0, dst1, ..., dst{N-1}, num, dtype, trans)
    # The last 3 entries are the codegen attrs (n_attrs=3).
    num = int(args[-3])
    dtype = parse_str(args[-2])
    trans_b = bool(int(args[-1])) if hasattr(args[-1], "value") else bool(args[-1])
    if num not in (1, 2, 4):
        raise ValueError(f"ldmatrix .num must be one of {{1, 2, 4}}, got {num}")
    if dtype not in ("b16", "b8"):
        raise ValueError(f"ldmatrix dtype must be 'b16' or 'b8', got {dtype!r}")
    n_regs = num if dtype == "b16" else num // 2
    trans_inst = ".trans" if trans_b else ""
    slot_list = "{" + ", ".join(f"%{i}" for i in range(n_regs)) + "}"
    reg_decls = ", ".join(f"r{i}" for i in range(n_regs))
    out_constraints = ", ".join(f'"=r"(r{i})' for i in range(n_regs))
    dst_assigns = "\n".join(f"    *(uint32_t*)dst{i} = r{i};" for i in range(n_regs))
    name = f"ptx_ldmatrix_{num}_{dtype.replace('::', '_').replace('.', '_')}_{1 if trans_b else 0}"
    sig = "(void* smem_ptr, " + ", ".join(f"void* dst{i}" for i in range(n_regs)) + ")"
    body = (
        f"    uint32_t {reg_decls};\n"
        "    unsigned int addr = __cvta_generic_to_shared(smem_ptr);\n"
        "    asm volatile(\n"
        f'        "ldmatrix.sync.aligned.m8n8.x{num}{trans_inst}.shared.{dtype} '
        f'{slot_list}, [%{n_regs}];"\n'
        f"        : {out_constraints}\n"
        f'        : "r"(addr));\n'
        f"{dst_assigns}"
    )
    return name, sig, body


device_intrinsic(
    "_ptx_ldmatrix_impl",
    n_attrs=3,
    c_signature=lambda *a: _ldmatrix_parts(*a)[1],
    helper_name=lambda *a: _ldmatrix_parts(*a)[0],
    body=lambda *a: _ldmatrix_parts(*a)[2],
)


@register_codegen("ptx_ldmatrix")
def codegen_ptx_ldmatrix(trans, num, dtype, smem_ptr, *dst_handles):
    trans = bool(trans)
    num = int(num)
    dtype = parse_str(dtype)
    if dtype.startswith("."):
        dtype = dtype[1:]
    n_regs = num if dtype == "b16" else num // 2
    if len(dst_handles) != n_regs:
        raise ValueError(
            f"ldmatrix .x{num}.{dtype} codegen expects {n_regs} dst handles, got {len(dst_handles)}"
        )
    result = CODEGEN_REGISTRY["tirx._ptx_ldmatrix_impl"](
        [smem_ptr, *dst_handles, num, dtype, trans]
    )
    return result[0] if isinstance(result, tuple) else result


def _stmatrix_parts(smem_ptr_, local_ptr_, num, trans, shape, ptx_type, space):
    num = int(num)
    trans_b = bool(int(trans)) if hasattr(trans, "value") else bool(trans)
    shape = parse_str(shape)
    ptx_type = parse_str(ptx_type)
    space = parse_str(space)
    if num not in (1, 2, 4):
        raise ValueError(f"stmatrix .num must be one of {{1, 2, 4}}, got {num}")
    if shape not in ("m8n8", "m16n8"):
        raise ValueError(f"stmatrix .shape must be m8n8 or m16n8, got {shape!r}")
    if ptx_type not in ("b16", "b8"):
        raise ValueError(f"stmatrix .type must be b16 or b8, got {ptx_type!r}")
    if space not in ("shared", "shared::cta"):
        raise ValueError(f"stmatrix state space must be shared or shared::cta, got {space!r}")
    if shape == "m16n8" and not trans_b:
        raise ValueError("stmatrix .m16n8 requires .trans")
    trans_inst = ".trans" if trans_b else ""
    slot_list = "{" + ", ".join(f"%{i}" for i in range(num)) + "}"
    constraints = ", ".join(f'"r"(reg[{i}])' for i in range(num))
    name = f"ptx_stmatrix_{shape}_{num}_{1 if trans_b else 0}_{space.replace('::', '_')}_{ptx_type}"
    body = (
        "    uint32_t* reg = (uint32_t*)local_ptr;\n"
        "    unsigned int addr = __cvta_generic_to_shared(smem_ptr);\n"
        "    asm volatile(\n"
        f'        "stmatrix.sync.aligned.{shape}.x{num}{trans_inst}.{space}.{ptx_type} '
        f'[%{num}], {slot_list};"\n'
        "        :\n"
        f'        : {constraints}, "r"(addr));'
    )
    return name, body


device_intrinsic(
    "_ptx_stmatrix_impl",
    n_attrs=5,
    c_signature="(void* smem_ptr, void* local_ptr)",
    helper_name=lambda *a: _stmatrix_parts(*a)[0],
    body=lambda *a: _stmatrix_parts(*a)[1],
)


@register_codegen("ptx_stmatrix")
def codegen_ptx_stmatrix(num, trans, shape, ptx_type, space, smem_ptr, local_ptr):
    num = int(num)
    trans = bool(trans)
    result = CODEGEN_REGISTRY["tirx._ptx_stmatrix_impl"](
        [smem_ptr, local_ptr, num, trans, shape, ptx_type, space]
    )
    return result[0] if isinstance(result, tuple) else result
