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
# ruff: noqa: E501
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments
"""Memory ops (load / store / copy / atomic / address conversion / type punning).

PTX side:
* ``ld.acquire.scope{.ss}.type`` scalar load forms.
* ``ld.volatile{.ss}.type`` scalar load forms.
* Legacy ``ld.global.acquire.gpu`` / ``ld.global.cg`` result-argument helper.
* ``mapa.u64`` — map a SMEM ptr to a peer CTA's SMEM in the cluster.

CUDA side:
* ``__ldg`` (cache-as-read-only load).
* Templated ``atomicAdd`` / ``atomicCAS``.
* half↔float type-punned conversions (single, packed, batch-of-8).
* ``__cvta_generic_to_shared`` and ``cluster_addr → shared u32`` casts.
"""

from tvm import DataType
from tvm.backend.cuda.op import cuda_func_call

from ._schema import device_intrinsic
from .registry import CODEGEN_REGISTRY, register_codegen
from .utils import parse_str


# =============================================================================
# __ldg — templated read-only cached load; ``T`` resolved at call time from
# the ``dtype`` argument. Hand-written because the helper signature uses a
# template parameter for both arg and return.
# =============================================================================
@register_codegen("cuda_ldg")
def codegen_cuda_ldg(addr, dtype):
    dtype = DataType(parse_str(dtype))
    func_name = "tvm_builtin_cuda_ldg"
    source_code = f"""
template <typename T>
__forceinline__ __device__ T {func_name}(T* src) {{
    return __ldg(src);
}}
"""
    return cuda_func_call(func_name, addr, source_code=source_code, return_type=dtype)


# Shared PTX scalar type metadata (ld/st/red/atom).
_PTX_SCALAR_TYPE_INFO = {
    "b8": ("unsigned int", "r", "uint32"),
    "u8": ("unsigned int", "r", "uint32"),
    "s8": ("int", "r", "int32"),
    "b16": ("unsigned short", "h", "uint16"),
    "u16": ("unsigned short", "h", "uint16"),
    "s16": ("short", "h", "int16"),
    "b32": ("unsigned int", "r", "uint32"),
    "u32": ("unsigned int", "r", "uint32"),
    "s32": ("int", "r", "int32"),
    "b64": ("unsigned long long", "l", "uint64"),
    "u64": ("unsigned long long", "l", "uint64"),
    "s64": ("long long", "l", "int64"),
    "f32": ("float", "f", "float32"),
    "f64": ("double", "d", "float64"),
}
_PTX_LD_TYPE_RETURNS = {
    "b32": {"uint32": "unsigned int", "int32": "int"},
    "b64": {"uint64": "unsigned long long", "int64": "long long"},
}
_PTX_VEC_STORE_TYPE = {
    16: "uint4",
    8: "uint2",
    4: "unsigned int",
    2: "unsigned short",
    1: "unsigned char",
}


def _safe_attr(value):
    return parse_str(value).replace("::", "_").replace(".", "_")


def _dot(value):
    value = parse_str(value)
    return f".{value}" if value else ""


def _cache_suffix(cache):
    return ".L2::cache_hint" if cache else ""


def _type_info(ptx_type):
    ptx_type = parse_str(ptx_type)
    if ptx_type not in _PTX_SCALAR_TYPE_INFO:
        raise ValueError(
            f"Unsupported PTX scalar type {ptx_type!r}; expected {sorted(_PTX_SCALAR_TYPE_INFO)}"
        )
    return (ptx_type, *_PTX_SCALAR_TYPE_INFO[ptx_type])


# =============================================================================
# PTX ld forms (ISA table entries registered via ``ptx_ld`` and siblings):
#   ld{.weak}{.ss}{.cop}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};
#   ld{.weak}{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};
#   ld.volatile{.ss}{.level::prefetch_size}{.vec}.type  d, [a];
#   ld.relaxed.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};
#   ld.acquire.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};
#   ld.mmio.sem.sys{.global}.type  d, [a];
# =============================================================================
_PTX_LD_SCOPES = {"cta", "cluster", "gpu", "sys"}
_PTX_LD_SPACES = {"global", "shared", "shared::cta", "shared::cluster", "local"}
_PTX_LD_VOLATILE_SPACES = _PTX_LD_SPACES | {"const"}
_PTX_LD_WEAK_SPACES = _PTX_LD_SPACES | {"const", "param::entry", "param::func"}
_PTX_LD_COPS = {"", "ca", "cg", "cs", "lu", "cv"}
_PTX_VEC = {"", "v2", "v4", "v8"}
_PTX_L1_EVICT = {
    "",
    "L1::evict_normal",
    "L1::evict_unchanged",
    "L1::evict_first",
    "L1::evict_last",
    "L1::no_allocate",
}
_PTX_L2_EVICT = {"", "L2::evict_normal", "L2::evict_first", "L2::evict_last"}
_PTX_PREFETCH = {"", "L2::64B", "L2::128B", "L2::256B"}


def _bool_attr(value):
    return bool(int(value)) if hasattr(value, "value") else bool(value)


def _parse_ld_attrs(return_dtype, ptx_type, scope=None, space="global"):
    return_dtype = parse_str(return_dtype)
    ptx_type = parse_str(ptx_type)
    scope = None if scope is None else parse_str(scope)
    space = parse_str(space)
    ptx_type, _ptx, constraint, default_tvm = _type_info(ptx_type)
    if ptx_type in _PTX_LD_TYPE_RETURNS:
        returns = _PTX_LD_TYPE_RETURNS[ptx_type]
        if return_dtype not in returns:
            raise ValueError(
                f"PTX ld type {ptx_type!r} cannot return TVM dtype {return_dtype!r}; "
                f"expected one of {sorted(returns)}"
            )
        c_type = returns[return_dtype]
    else:
        if return_dtype != default_tvm:
            raise ValueError(
                f"PTX ld type {ptx_type!r} cannot return TVM dtype {return_dtype!r}; "
                f"expected {default_tvm!r}"
            )
        c_type = _ptx
    if scope is not None and scope not in _PTX_LD_SCOPES:
        raise ValueError(
            f"Unsupported PTX ld scope {scope!r}; expected one of {sorted(_PTX_LD_SCOPES)}"
        )
    return return_dtype, ptx_type, scope, space, c_type, constraint


def _validate_ld_space(space: str, allowed: set[str]) -> None:
    if space not in allowed:
        raise ValueError(
            f"Unsupported PTX ld state space {space!r}; expected one of {sorted(allowed)}"
        )


def _ptx_level_suffix(has_cache, l1_evict, l2_evict, prefetch_size):
    suffix = _cache_suffix("cache" if has_cache else "")
    l1_evict = parse_str(l1_evict)
    l2_evict = parse_str(l2_evict)
    prefetch_size = parse_str(prefetch_size)
    if l1_evict:
        suffix += f".{l1_evict}"
    if l2_evict:
        suffix += f".{l2_evict}"
    if prefetch_size:
        suffix += f".{prefetch_size}"
    return suffix


def _ptx_shared_addr(space, ptr_name="address"):
    if parse_str(space).startswith("shared"):
        return (
            f"    unsigned int addr = (unsigned int)__cvta_generic_to_shared({ptr_name});\n",
            '"r"(addr)',
        )
    return "", f'"l"({ptr_name})'


def _ptx_ld_vec_store(num_bytes, vec_len, ptx_type):
    if ptx_type == "u8" and vec_len == 1:
        return "    *reinterpret_cast<unsigned char*>(dst_ptr) = static_cast<unsigned char>(r0);"
    store_type = _PTX_VEC_STORE_TYPE[num_bytes]
    if vec_len > 1:
        return (
            f"    *reinterpret_cast<{store_type}*>(dst_ptr) = "
            + "{"
            + ", ".join(f"r{i}" for i in range(vec_len))
            + "};"
        )
    return f"    *reinterpret_cast<{store_type}*>(dst_ptr) = r0;"


def _ptx_ld_form_parts(form, attr_args):
    if form == "weak":
        (
            return_dtype,
            weak,
            space,
            cop,
            vec,
            ptx_type,
            has_cache_hint,
            to_dst,
            l1_evict,
            l2_evict,
            prefetch_size,
        ) = attr_args
        sem, scope = "", ""
    elif form == "relaxed":
        (
            return_dtype,
            scope,
            space,
            vec,
            ptx_type,
            has_cache_hint,
            to_dst,
            l1_evict,
            l2_evict,
            prefetch_size,
        ) = attr_args
        sem, weak, cop = "", False, ""
    elif form == "acquire":
        (
            return_dtype,
            scope,
            space,
            vec,
            ptx_type,
            has_cache_hint,
            to_dst,
            l1_evict,
            l2_evict,
            prefetch_size,
        ) = attr_args
        sem, weak, cop = "", False, ""
    elif form == "volatile":
        return_dtype, space, vec, ptx_type, to_dst, prefetch_size = attr_args
        sem, scope, weak, cop = "", "", False, ""
        has_cache_hint, l1_evict, l2_evict = False, "", ""
    elif form == "mmio":
        return_dtype, sem, scope, space, ptx_type, to_dst = attr_args
        weak, cop, vec = False, "", ""
        has_cache_hint, l1_evict, l2_evict, prefetch_size = False, "", "", ""
    else:
        raise ValueError(f"unknown ld form {form!r}")

    return_dtype, ptx_type, scope, space, c_type, constraint = _parse_ld_attrs(
        return_dtype, ptx_type, scope if form in ("relaxed", "acquire") else None, space
    )
    sem = parse_str(sem)
    scope = parse_str(scope)
    space = parse_str(space)
    cop = parse_str(cop)
    vec = parse_str(vec)
    l1_evict = parse_str(l1_evict)
    l2_evict = parse_str(l2_evict)
    prefetch_size = parse_str(prefetch_size)
    weak = _bool_attr(weak)
    has_cache = _bool_attr(has_cache_hint)
    to_dst = _bool_attr(to_dst)
    if cop and cop not in _PTX_LD_COPS:
        raise ValueError(f"Unsupported PTX ld cache operation {cop!r}")
    if vec and vec not in _PTX_VEC:
        raise ValueError(f"Unsupported PTX ld vector modifier {vec!r}")
    if l1_evict and l1_evict not in _PTX_L1_EVICT:
        raise ValueError(f"Unsupported PTX ld L1 eviction {l1_evict!r}")
    if l2_evict and l2_evict not in _PTX_L2_EVICT:
        raise ValueError(f"Unsupported PTX ld L2 eviction {l2_evict!r}")
    if prefetch_size and prefetch_size not in _PTX_PREFETCH:
        raise ValueError(f"Unsupported PTX ld prefetch size {prefetch_size!r}")
    if form == "mmio":
        if sem not in ("acquire", "relaxed") or scope != "sys" or space != "global":
            raise ValueError("ld.mmio requires sem in {acquire, relaxed}, scope=sys, space=global")
        prefix = f"ld.mmio.{sem}.{scope}"
    elif form == "relaxed":
        if not scope:
            raise ValueError("ld.relaxed requires scope")
        _validate_ld_space(space, _PTX_LD_SPACES)
        prefix = f"ld.relaxed.{scope}{_dot(space)}"
    elif form == "acquire":
        if not scope:
            raise ValueError("ld.acquire requires scope")
        _validate_ld_space(space, _PTX_LD_SPACES)
        prefix = f"ld.acquire.{scope}{_dot(space)}"
    elif form == "volatile":
        _validate_ld_space(space, _PTX_LD_VOLATILE_SPACES)
        prefix = f"ld.volatile{_dot(space)}"
    else:
        _validate_ld_space(space, _PTX_LD_WEAK_SPACES)
        prefix = f"ld{'.weak' if weak else ''}{_dot(space)}{_dot(cop)}"
    level = _ptx_level_suffix(has_cache, l1_evict, l2_evict, prefetch_size)
    vec_len = int(vec[1:]) if vec else 1
    if vec and not to_dst:
        raise ValueError("vector ld requires to_dst")
    elem_bytes = (
        8
        if ptx_type.endswith("64")
        else 2
        if ptx_type in ("u16", "s16", "b16")
        else 1
        if ptx_type in ("u8", "s8", "b8")
        else 4
    )
    num_bytes = vec_len * elem_bytes if vec else elem_bytes
    name_parts = [
        "tvm_builtin_ptx_ld",
        form if form != "weak" else ("weak" if weak else "plain"),
    ]
    if sem:
        name_parts.append(_safe_attr(sem))
    if scope:
        name_parts.append(_safe_attr(scope))
    name_parts.extend(
        [
            _safe_attr(space),
            _safe_attr(cop) if cop else "",
            _safe_attr(vec) if vec else "",
            ptx_type,
            return_dtype if not to_dst else "to_dst",
        ]
    )
    if has_cache:
        name_parts.append("cache_hint")
    if l1_evict:
        name_parts.append(_safe_attr(l1_evict))
    if l2_evict:
        name_parts.append(_safe_attr(l2_evict))
    if prefetch_size:
        name_parts.append(_safe_attr(prefetch_size))
    name = "_".join(p for p in name_parts if p)
    cache_operand = ', "l"(cache_policy)' if has_cache else ""
    addr_decl, addr_operand = _ptx_shared_addr(space, "src_ptr" if to_dst else "address")
    if to_dst:
        reg_decls = "".join(f"    {c_type} r{i};\n" for i in range(vec_len))
        if vec_len > 1:
            out_slot = "{" + ", ".join(f"%{i}" for i in range(vec_len)) + "}"
            out_constraints = ", ".join(f'"={constraint}"(r{i})' for i in range(vec_len))
            addr_idx = vec_len
        else:
            out_slot = "%0"
            out_constraints = f'"={constraint}"(r0)'
            addr_idx = 1
        cache_slot = f", %{addr_idx + 1}" if has_cache else ""
        instr = f"{prefix}{level}{_dot(vec)}.{ptx_type}"
        body = (
            f"{addr_decl}{reg_decls}"
            f'    asm volatile("{instr} {out_slot}, [%{addr_idx}]{cache_slot};"\n'
            f"                 : {out_constraints}\n"
            f"                 : {addr_operand}{cache_operand});\n"
            f"{_ptx_ld_vec_store(num_bytes, vec_len, ptx_type)}"
        )
        return (
            name,
            "(void* dst_ptr, void* src_ptr, unsigned long long cache_policy)",
            "void",
            "",
            body,
        )
    cache_slot = ", %2" if has_cache else ""
    instr = f"{prefix}{level}{_dot(vec)}.{ptx_type}"
    body = (
        f"    {c_type} ret;\n"
        f"{addr_decl}"
        f'    asm volatile("{instr} %0, [%1]{cache_slot};"\n'
        f'                 : "={constraint}"(ret)\n'
        f"                 : {addr_operand}{cache_operand});\n"
        "    return ret;"
    )
    sig = (
        "(void* address, unsigned long long cache_policy)" if form == "weak" else "(void* address)"
    )
    return name, sig, c_type, return_dtype, body


def _register_ptx_ld(op_name, form, n_attrs):
    def _parts(*args):
        return _ptx_ld_form_parts(form, args[-n_attrs:])

    device_intrinsic(
        op_name,
        n_attrs=n_attrs,
        helper_name=lambda *a, _p=_parts: _p(*a)[0],
        c_signature=lambda *a, _p=_parts: _p(*a)[1],
        return_type=lambda *a, _p=_parts: _p(*a)[2],
        tvm_return_type=lambda *a, _p=_parts: (
            None if _p(*a)[2] == "void" else parse_str(a[-n_attrs])
        ),
        body=lambda *a, _p=_parts: _p(*a)[4],
    )


_register_ptx_ld("ptx_ld", "weak", 11)
_register_ptx_ld("ptx_ld_relaxed", "relaxed", 10)
_register_ptx_ld("ptx_ld_acquire", "acquire", 10)
_register_ptx_ld("ptx_ld_volatile", "volatile", 6)
_register_ptx_ld("ptx_ld_mmio", "mmio", 6)


# =============================================================================
# Legacy acquire-load lvalue API — compatibility wrapper over
# ``ld.acquire.gpu.global`` / ``ld.global.cg`` forms, dispatched on dtype.
# Wrapper picks .b32/.b64 + matching constraint by dtype.
#
# The body uses ``#if __CUDA_ARCH__ >= 700`` to select acquire on SM70+ and
# fall back to .cg on older arches. This is two PTX form table entries
# combined in one device helper for arch portability.
# =============================================================================
_LD_GLOBAL_ACQUIRE_DTYPES = {
    "uint32": ("uint32_t", "b32", "r"),
    "int32": ("int32_t", "b32", "r"),
    "uint64": ("uint64_t", "b64", "l"),
    "int64": ("int64_t", "b64", "l"),
}


def _ld_global_acquire_body(ptx_type: str, spec: str) -> str:
    return (
        "  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700\n"
        f'  asm volatile ("ld.acquire.gpu.global.{ptx_type} %0, [%1];\\n"\n'
        f'                : "={spec}"(res) : "l"(addr));\n'
        "  #else\n"
        f'  asm volatile ("ld.global.cg.{ptx_type} %0, [%1];\\n"\n'
        f'                : "={spec}"(res) : "l"(addr));\n'
        "  #endif"
    )


for _dtype, (_c_type, _ptx_type, _spec) in _LD_GLOBAL_ACQUIRE_DTYPES.items():
    device_intrinsic(
        f"ptx_ld_global_acquire_{_dtype}",
        c_signature=f"({_c_type}& res, {_c_type}* addr)",
        body=_ld_global_acquire_body(_ptx_type, _spec),
    )
del _dtype, _c_type, _ptx_type, _spec


@register_codegen("ptx_ld_global_acquire")
def codegen_ptx_ld_global_acquire(res, addr):
    """Dispatch to the dtype-specific helper."""
    dtype = str(res.ty)
    if dtype not in _LD_GLOBAL_ACQUIRE_DTYPES:
        raise ValueError(f"Unsupported data type for ld.global.acquire: {dtype}")
    result = CODEGEN_REGISTRY[f"tirx.ptx_ld_global_acquire_{dtype}"]([res, addr])
    return result[0] if isinstance(result, tuple) else result


# =============================================================================
# Atomics — templated wrappers around CUDA's ``atomicAdd`` / ``atomicCAS``.
# =============================================================================
device_intrinsic(
    "cuda_atomic_add",
    helper_name="tvm_builtin_cuda_atomic_add",
    c_signature="(T* addr, T value)",
    body="    return atomicAdd(addr, value);",
    return_type="T",
    templated=True,
    tvm_return_type=lambda _addr, value: value.ty,
)
device_intrinsic(
    "cuda_atomic_cas",
    helper_name="tvm_builtin_cuda_atomic_cas",
    c_signature="(T* address, T compare, T val)",
    body="    return atomicCAS(address, compare, val);",
    return_type="T",
    templated=True,
    tvm_return_type=lambda _p, old, _n: old.ty,
)


# =============================================================================
# half / bfloat16 ↔ float type-punned conversions.
# =============================================================================
device_intrinsic(
    "cuda_half2float",
    c_signature="(half src)",
    body="    return __half2float(src);",
    return_type="float",
    tvm_return_type="float32",
)
device_intrinsic(
    "cuda_bfloat162float",
    c_signature="(nv_bfloat16 src)",
    body="    return __bfloat162float(src);",
    return_type="float",
    tvm_return_type="float32",
)
device_intrinsic(
    "cuda_float22half2",
    c_signature="(void* dst, void* src)",
    body=(
        "    half2* dst_p = (half2*) dst;\n"
        "    float2* src_p = (float2*) src;\n"
        "    *dst_p = __float22half2_rn(*src_p);"
    ),
)
device_intrinsic(
    "cuda_half8tofloat8",
    c_signature="(void* src_addr, void* dst_addr)",
    body=(
        "    half2* source = (half2*) src_addr;\n"
        "    float2* dest = (float2*) dst_addr;\n"
        "    for (int i = 0; i < 4; i++) {\n"
        "        dest[i] = __half22float2(source[i]);\n"
        "    }"
    ),
)
device_intrinsic(
    "cuda_float8tohalf8",
    c_signature="(void* src_addr, void* dst_addr)",
    body=(
        "    float2* source = (float2*) src_addr;\n"
        "    half2* dest = (half2*) dst_addr;\n"
        "    for (int i = 0; i < 4; i++) {\n"
        "        dest[i] = __float22half2_rn(source[i]);\n"
        "    }"
    ),
)


# =============================================================================
# Address-conversion helpers used by op-wrapper-side dispatch in tvm.tirx.op.
# Each precomputes a value that the schema's specialized op then takes as a
# typed scalar input (instead of doing the conversion inside the asm helper).
# =============================================================================
device_intrinsic(
    "cuda_cvta_generic_to_shared",
    c_signature="(void* p)",
    body="    return __cvta_generic_to_shared(p);",
    return_type="unsigned int",
    tvm_return_type="uint32",
)

device_intrinsic(
    "cuda_smem_addr_from_uint64",
    c_signature="(uint64_t cluster_addr)",
    body="    return static_cast<unsigned int>(cluster_addr);",
    return_type="unsigned int",
    tvm_return_type="uint32",
)

# =============================================================================
# PTX mapa form:
#   mapa{.space}.type d, a, b;
#   .space = {.shared::cluster}; .type = {.u32, .u64}
# =============================================================================


def _ptx_mapa_parts(_addr, _rank, space, ptx_type, return_dtype):
    space = parse_str(space)
    ptx_type = parse_str(ptx_type)
    return_dtype = parse_str(return_dtype)
    if space not in ("", "shared::cluster"):
        raise ValueError(f"Unsupported mapa space {space!r}")
    if ptx_type not in ("u32", "u64"):
        raise ValueError(f"Unsupported mapa type {ptx_type!r}")
    c_type = "uint32_t" if ptx_type == "u32" else "uint64_t"
    constraint = "r" if ptx_type == "u32" else "l"
    name = f"tvm_builtin_ptx_mapa{('_' + _safe_attr(space)) if space else ''}_{ptx_type}"
    body = (
        f"    {c_type} result;\n"
        f'    asm volatile("mapa{_dot(space)}.{ptx_type} %0, %1, %2;"\n'
        f'                 : "={constraint}"(result) : "l"(addr), "r"(rank));\n'
        "    return result;"
    )
    return name, c_type, return_dtype, body


device_intrinsic(
    "ptx_mapa",
    n_attrs=3,
    helper_name=lambda *a: _ptx_mapa_parts(*a)[0],
    c_signature="(void* addr, uint32_t rank)",
    return_type=lambda *a: _ptx_mapa_parts(*a)[1],
    tvm_return_type=lambda *a: _ptx_mapa_parts(*a)[2],
    body=lambda *a: _ptx_mapa_parts(*a)[3],
)


# =============================================================================
# Generic PTX memory forms. Compatibility wrappers in ``tvm.tirx.op`` bind
# concrete sem/scope/space/op/type parameters for existing call sites.
# =============================================================================


# PTX red scalar form:
#   red{.sem}{.scope}{.space}.op{.level::cache_hint}.type [a], b{, cache-policy};
def _ptx_red_scalar_parts(*args):
    sem, scope, space, op, ptx_type, has_cache_hint = args[-6:]
    sem = parse_str(sem)
    scope = parse_str(scope)
    space = parse_str(space)
    op = parse_str(op)
    ptx_type, c_type, constraint, _tvm_dtype = _type_info(ptx_type)
    has_cache = (
        bool(int(has_cache_hint)) if hasattr(has_cache_hint, "value") else bool(has_cache_hint)
    )
    modifiers = f"{_dot(sem)}{_dot(scope)}{_dot(space)}"
    instr = f"red{modifiers}.{op}{_cache_suffix('cache' if has_cache else '')}.{ptx_type}"
    name = (
        "tvm_builtin_ptx_red_scalar"
        f"{_dot(sem).replace('.', '_')}{_dot(scope).replace('.', '_')}"
        f"_{_safe_attr(space)}_{op}_{ptx_type}{'_cache_hint' if has_cache else ''}"
    )
    cache_operand = ', "l"(cache_policy)' if has_cache else ""
    addr_decl = ""
    addr_operand = '"l"(address)'
    if space.startswith("shared"):
        addr_decl = "    unsigned int addr = (unsigned int)__cvta_generic_to_shared(address);\n"
        addr_operand = '"r"(addr)'
    body = (
        f"{addr_decl}"
        f'    asm volatile("{instr} [%0], %1{", %2" if has_cache else ""};"\n'
        "                 :\n"
        f'                 : {addr_operand}, "{constraint}"(value)'
        f"{cache_operand}\n"
        '                 : "memory");'
    )
    return name, f"(void* address, {c_type} value, unsigned long long cache_policy)", body


device_intrinsic(
    "ptx_red_scalar",
    n_attrs=6,
    helper_name=lambda *a: _ptx_red_scalar_parts(*a)[0],
    c_signature=lambda *a: _ptx_red_scalar_parts(*a)[1],
    body=lambda *a: _ptx_red_scalar_parts(*a)[2],
)


# PTX atom scalar one-source-operand form:
#   atom{.sem}{.scope}{.space}.op{.level::cache_hint}.type d, [a], b{, cache-policy};
def _ptx_atom_scalar_parts(*args):
    sem, scope, space, op, ptx_type, has_cache_hint = args[-6:]
    sem = parse_str(sem)
    scope = parse_str(scope)
    space = parse_str(space)
    op = parse_str(op)
    ptx_type, c_type, constraint, tvm_dtype = _type_info(ptx_type)
    has_cache = (
        bool(int(has_cache_hint)) if hasattr(has_cache_hint, "value") else bool(has_cache_hint)
    )
    modifiers = f"{_dot(sem)}{_dot(scope)}{_dot(space)}"
    instr = f"atom{modifiers}.{op}{_cache_suffix('cache' if has_cache else '')}.{ptx_type}"
    name = (
        "tvm_builtin_ptx_atom_scalar"
        f"{_dot(sem).replace('.', '_')}{_dot(scope).replace('.', '_')}"
        f"_{_safe_attr(space)}_{op}_{ptx_type}{'_cache_hint' if has_cache else ''}"
    )
    cache_operand = ', "l"(cache_policy)' if has_cache else ""
    addr_decl = ""
    addr_operand = '"l"(address)'
    if space.startswith("shared"):
        addr_decl = "    unsigned int addr = (unsigned int)__cvta_generic_to_shared(address);\n"
        addr_operand = '"r"(addr)'
    body = (
        f"{addr_decl}"
        f"    {c_type} ret;\n"
        f'    asm volatile("{instr} %0, [%1], %2{", %3" if has_cache else ""};"\n'
        f'                 : "={constraint}"(ret)\n'
        f'                 : {addr_operand}, "{constraint}"(value)'
        f"{cache_operand}\n"
        '                 : "memory");\n'
        "    return ret;"
    )
    return (
        name,
        f"(void* address, {c_type} value, unsigned long long cache_policy)",
        c_type,
        tvm_dtype,
        body,
    )


device_intrinsic(
    "ptx_atom_scalar",
    n_attrs=6,
    helper_name=lambda *a: _ptx_atom_scalar_parts(*a)[0],
    c_signature=lambda *a: _ptx_atom_scalar_parts(*a)[1],
    return_type=lambda *a: _ptx_atom_scalar_parts(*a)[2],
    tvm_return_type=lambda *a: _ptx_atom_scalar_parts(*a)[3],
    body=lambda *a: _ptx_atom_scalar_parts(*a)[4],
)


# PTX prefetch tensormap form:
#   prefetch{.tensormap_space}.tensormap [a];
def _prefetch_tensormap_parts(_tensor_map, tensormap_space):
    space = parse_str(tensormap_space)
    instr = f"prefetch{_dot(space)}.tensormap"
    name = f"tvm_builtin_ptx_prefetch{('_' + _safe_attr(space)) if space else ''}_tensormap"
    body = (
        f'    asm volatile("{instr} [%0];"\n'
        "                 :\n"
        '                 : "l"(tensor_map_addr)\n'
        '                 : "memory");'
    )
    return name, body


device_intrinsic(
    "ptx_prefetch_tensormap",
    n_attrs=1,
    helper_name=lambda *a: _prefetch_tensormap_parts(*a)[0],
    c_signature="(unsigned long long tensor_map_addr)",
    body=lambda *a: _prefetch_tensormap_parts(*a)[1],
)


# PTX st forms (ISA table entries registered via ``ptx_st`` and siblings):
#   st{.weak}{.ss}{.cop}{.level::cache_hint}{.vec}.type [a], b{, cache-policy};
#   st{.weak}{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.vec}.type [a], b{, cache-policy};
#   st.volatile{.ss}{.vec}.type [a], b;
#   st.relaxed.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.vec}.type [a], b{, cache-policy};
#   st.release.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.vec}.type [a], b{, cache-policy};
#   st.mmio.sem.sys{.global}.type [a], b;
_PTX_ST_COPS = {"", "wb", "cg", "cs", "wt"}
_PTX_ST_SPACES = {"global", "shared", "shared::cta", "shared::cluster", "local", "param::func"}


def _ptx_st_load_src(num_bytes, vec_len, ptx_type, c_type):
    if ptx_type == "u8" and vec_len == 1:
        return "    unsigned int r0 = *reinterpret_cast<unsigned char*>(src_ptr);\n"
    store_type = _PTX_VEC_STORE_TYPE[num_bytes]
    if vec_len > 1:
        return f"    {store_type} src_ = *reinterpret_cast<{store_type}*>(src_ptr);\n" + "".join(
            f"    {c_type} r{i} = src_.{c};\n" for i, c in enumerate("xyzw"[:vec_len])
        )
    return f"    {c_type} r0 = *reinterpret_cast<{c_type}*>(src_ptr);\n"


def _ptx_st_form_parts(form, attr_args, from_src):
    if form == "weak":
        weak, space, cop, vec, ptx_type, has_cache_hint, l1_evict, l2_evict = attr_args
        sem, scope = "", ""
    elif form == "relaxed":
        scope, space, vec, ptx_type, has_cache_hint, l1_evict, l2_evict = attr_args
        sem, weak, cop = "relaxed", False, ""
    elif form == "release":
        scope, space, vec, ptx_type, has_cache_hint, l1_evict, l2_evict = attr_args
        sem, weak, cop = "release", False, ""
    elif form == "volatile":
        space, vec, ptx_type = attr_args
        sem, scope, weak, cop = "", "", False, ""
        has_cache_hint, l1_evict, l2_evict = False, "", ""
    elif form == "mmio":
        sem, scope, space, ptx_type = attr_args
        weak, cop, vec = False, "", ""
        has_cache_hint, l1_evict, l2_evict = False, "", ""
    else:
        raise ValueError(f"unknown st form {form!r}")

    sem = parse_str(sem)
    scope = parse_str(scope)
    space = parse_str(space)
    cop = parse_str(cop)
    vec = parse_str(vec)
    l1_evict = parse_str(l1_evict)
    l2_evict = parse_str(l2_evict)
    weak = _bool_attr(weak)
    has_cache = _bool_attr(has_cache_hint)
    ptx_type, c_type, constraint, _tvm_dtype = _type_info(ptx_type)
    if cop and cop not in _PTX_ST_COPS:
        raise ValueError(f"Unsupported PTX st cache operation {cop!r}")
    if vec and vec not in _PTX_VEC:
        raise ValueError(f"Unsupported PTX st vector modifier {vec!r}")
    if space not in _PTX_ST_SPACES and not (form == "mmio" and space == "global"):
        raise ValueError(f"Unsupported PTX st state space {space!r}")
    vec_len = int(vec[1:]) if vec else 1
    elem_bytes = (
        8
        if ptx_type.endswith("64")
        else 2
        if ptx_type in ("u16", "s16", "b16")
        else 1
        if ptx_type in ("u8", "s8", "b8")
        else 4
    )
    num_bytes = vec_len * elem_bytes if vec else elem_bytes
    use_cache_policy = form in ("weak", "relaxed", "release")
    if form == "mmio":
        if sem not in ("acquire", "relaxed", "release") or scope != "sys" or space != "global":
            raise ValueError("st.mmio requires sem, scope=sys, space=global")
        prefix = f"st.mmio.{sem}.{scope}"
    elif form == "relaxed":
        if not scope:
            raise ValueError("st.relaxed requires scope")
        prefix = f"st.relaxed.{scope}{_dot(space)}"
    elif form == "release":
        if not scope:
            raise ValueError("st.release requires scope")
        prefix = f"st.release.{scope}{_dot(space)}"
    elif form == "volatile":
        prefix = f"st.volatile{_dot(space)}"
    else:
        prefix = f"st{'.weak' if weak else ''}{_dot(space)}{_dot(cop)}"
    level = _ptx_level_suffix(has_cache, l1_evict, l2_evict, "")
    instr = f"{prefix}{level}{_dot(vec)}.{ptx_type}"
    name_parts = ["tvm_builtin_ptx_st", form if form != "weak" else ("weak" if weak else "plain")]
    if sem:
        name_parts.append(_safe_attr(sem))
    if scope:
        name_parts.append(_safe_attr(scope))
    name_parts.extend(
        [
            _safe_attr(space),
            _safe_attr(cop) if cop else "",
            _safe_attr(vec) if vec else "",
            ptx_type,
            "from_src" if from_src else "values",
        ]
    )
    if has_cache:
        name_parts.append("cache_hint")
    name = "_".join(p for p in name_parts if p)
    values = f"{{{', '.join(f'%{i + 1}' for i in range(vec_len))}}}" if vec_len > 1 else "%1"
    value_constraints = "".join(f', "{constraint}"(value{i})' for i in range(vec_len))
    cache_slot = f", %{vec_len + 1}" if has_cache else ""
    cache_operand = ', "l"(cache_policy)' if has_cache else ""
    addr_decl, addr_operand = _ptx_shared_addr(space, "address")
    if from_src:
        load_regs = _ptx_st_load_src(num_bytes, vec_len, ptx_type, c_type)
        if vec_len > 1:
            in_constraints = ", ".join(f'"{constraint}"(r{i})' for i in range(vec_len))
            value_args = f", {in_constraints}"
        else:
            value_args = f', "{constraint}"(r0)'
        body = (
            f"{addr_decl}{load_regs}"
            f'    asm volatile("{instr} [%0], {values}{cache_slot};"\n'
            "                 :\n"
            f"                 : {addr_operand}{value_args}{cache_operand}\n"
            '                 : "memory");'
        )
        if use_cache_policy:
            sig = "(void* address, void* src_ptr, unsigned long long cache_policy)"
        else:
            sig = "(void* address, void* src_ptr)"
    else:
        body = (
            f"{addr_decl}"
            f'    asm volatile("{instr} [%0], {values}{cache_slot};"\n'
            "                 :\n"
            f"                 : {addr_operand}{value_constraints}{cache_operand}\n"
            '                 : "memory");'
        )
        if form == "mmio":
            sig = f"(void* address, {c_type} value0)"
        elif use_cache_policy:
            value_params = ", ".join(f"{c_type} value{i}" for i in range(vec_len))
            sig = f"(void* address, {value_params}, unsigned long long cache_policy)"
        else:
            value_params = ", ".join(f"{c_type} value{i}" for i in range(vec_len))
            sig = f"(void* address, {value_params})"
    return name, sig, body


def _register_ptx_st(op_name, form, n_attrs, *, with_cache_policy=True):
    def codegen(*args):
        from_src = _bool_attr(args[-1])
        st_attrs = args[-n_attrs:-1]
        parts = _ptx_st_form_parts(form, st_attrs, from_src)
        forward = args[:-(n_attrs)]
        name, sig, body_str = parts
        source_code = f"\n__forceinline__ __device__ void {name}{sig} {{\n{body_str}\n}}\n"
        return cuda_func_call(name, *forward, source_code=source_code)

    codegen.__name__ = f"codegen_{op_name}"
    register_codegen(op_name)(codegen)


def _register_ptx_st_mmio(op_name, form, n_attrs):
    def codegen(*args):
        parts = _ptx_st_form_parts(form, args[-n_attrs:], False)
        forward = args[:-n_attrs]
        name, sig, body_str = parts
        source_code = f"\n__forceinline__ __device__ void {name}{sig} {{\n{body_str}\n}}\n"
        return cuda_func_call(name, *forward, source_code=source_code)

    codegen.__name__ = f"codegen_{op_name}"
    register_codegen(op_name)(codegen)


_register_ptx_st("ptx_st", "weak", 9)
_register_ptx_st("ptx_st_relaxed", "relaxed", 8)
_register_ptx_st("ptx_st_release", "release", 8)
_register_ptx_st("ptx_st_volatile", "volatile", 4)
_register_ptx_st_mmio("ptx_st_mmio", "mmio", 4)


# PTX st.bulk form:
#   st.bulk{.weak}{.shared::cta} [a], size, initval;
# ``initval`` is an immediate operand whose only legal value is 0.
def _ptx_st_bulk_parts(_ptr, _num_bytes, weak, space):
    weak = bool(int(weak)) if hasattr(weak, "value") else bool(weak)
    space = parse_str(space)
    instr = f"st.bulk{'.weak' if weak else ''}{_dot(space)}"
    name = f"tvm_builtin_ptx_st_bulk{'_weak' if weak else ''}{('_' + _safe_attr(space)) if space else ''}"
    addr_arg = (
        '"r"((unsigned int)__cvta_generic_to_shared(ptr))' if space == "shared::cta" else '"l"(ptr)'
    )
    body = (
        f'    asm volatile("{instr} [%0], %1, 0;"\n'
        "                 :\n"
        f"                 : {addr_arg}, "
        '"l"(static_cast<uint64_t>(num_bytes))\n'
        '                 : "memory");'
    )
    return name, body


device_intrinsic(
    "ptx_st_bulk",
    n_attrs=2,
    helper_name=lambda *a: _ptx_st_bulk_parts(*a)[0],
    c_signature="(void* ptr, unsigned int num_bytes)",
    body=lambda *a: _ptx_st_bulk_parts(*a)[1],
)

device_intrinsic(
    "cuda_uint_as_float",
    helper_name="tvm_builtin_uint_as_float",
    c_signature="(unsigned int bits)",
    return_type="float",
    body="    return __uint_as_float(bits);",
)
device_intrinsic(
    "cuda_float_as_uint",
    helper_name="tvm_builtin_float_as_uint",
    c_signature="(float x)",
    return_type="unsigned int",
    body="    return __float_as_uint(x);",
)
