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
* Typed N-byte copy helpers (1/2/4/8/16 bytes via uint{2,4} / unsigned).
* ``__ldg`` (cache-as-read-only load).
* Templated ``atomicAdd`` / ``atomicCAS``.
* half↔float type-punned conversions (single, packed, batch-of-8).
* ``__cvta_generic_to_shared`` and ``cluster_addr → shared u32`` casts.
"""

from tvm import DataType
from tvm.tirx.op import cuda_func_call

from .._schema import device_intrinsic
from .registry import CODEGEN_REGISTRY, register_codegen
from .utils import parse_str

# =============================================================================
# Typed N-byte copies — one helper per (1, 2, 4, 8, 16)-byte width.
# Dispatcher picks by ``num_bytes``.
# =============================================================================
_TYPE_MAP = {16: "uint4", 8: "uint2", 4: "unsigned int", 2: "unsigned short", 1: "unsigned char"}


for _num_bytes, _cpp_type in _TYPE_MAP.items():
    device_intrinsic(
        f"_cuda_copy_bytes_{_num_bytes}_impl",
        helper_name=f"tvm_builtin_copy_{_num_bytes * 8}b",
        c_signature="(void* dst_ptr, void* src_ptr)",
        body=(
            f"    {_cpp_type}* src_ = reinterpret_cast<{_cpp_type}*>(src_ptr);\n"
            f"    {_cpp_type}* dst_ = reinterpret_cast<{_cpp_type}*>(dst_ptr);\n"
            "    *dst_ = *src_;"
        ),
    )
del _num_bytes, _cpp_type


@register_codegen("cuda_copy_bytes")
def codegen_cuda_copy_bytes(dst, src, num_bytes):
    """Dispatch to the size-specific helper based on ``num_bytes``."""
    num_bytes_int = int(num_bytes)
    if num_bytes_int not in _TYPE_MAP:
        raise ValueError(
            f"Unsupported cuda_copy_bytes num_bytes {num_bytes_int}, "
            f"expected one of {sorted(_TYPE_MAP)}"
        )
    result = CODEGEN_REGISTRY[f"tirx._cuda_copy_bytes_{num_bytes_int}_impl"]([dst, src])
    return result[0] if isinstance(result, tuple) else result


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


# =============================================================================
# PTX ld forms:
#   ld{.weak}{.ss}{.cop}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};
#   ld.acquire.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};
#   ld.volatile{.ss}{.level::prefetch_size}{.vec}.type  d, [a];
#
# These are registered from the PTX ISA ld grammar. The current helpers cover
# the scalar no-cache-policy/no-vector instances currently registered. Scope,
# state space, PTX type, and TVM return dtype are explicit instead of being
# inferred from a generic "load" helper.
# =============================================================================
_PTX_LD_SCOPES = {"cta", "cluster", "gpu", "sys"}
_PTX_LD_SPACES = {"global", "shared", "shared::cta", "shared::cluster", "local"}
_PTX_LD_VOLATILE_SPACES = _PTX_LD_SPACES | {"const"}
_PTX_LD_COPS = {"", "ca", "cg", "cs", "lu", "cv"}
_PTX_LD_TYPES = {
    "b32": {"constraint": "r", "returns": {"uint32": "unsigned int", "int32": "int"}},
    "u32": {"constraint": "r", "returns": {"uint32": "unsigned int"}},
    "u64": {"constraint": "l", "returns": {"uint64": "unsigned long long"}},
    "s32": {"constraint": "r", "returns": {"int32": "int"}},
    "f32": {"constraint": "f", "returns": {"float32": "float"}},
}


def _parse_ld_attrs(return_dtype, ptx_type, scope=None, space="global"):
    return_dtype = parse_str(return_dtype)
    ptx_type = parse_str(ptx_type)
    scope = None if scope is None else parse_str(scope)
    space = parse_str(space)
    if ptx_type not in _PTX_LD_TYPES:
        raise ValueError(
            f"Unsupported PTX ld type {ptx_type!r}; expected one of {sorted(_PTX_LD_TYPES)}"
        )
    returns = _PTX_LD_TYPES[ptx_type]["returns"]
    if return_dtype not in returns:
        raise ValueError(
            f"PTX ld type {ptx_type!r} cannot return TVM dtype {return_dtype!r}; "
            f"expected one of {sorted(returns)}"
        )
    if scope is not None and scope not in _PTX_LD_SCOPES:
        raise ValueError(
            f"Unsupported PTX ld scope {scope!r}; expected one of {sorted(_PTX_LD_SCOPES)}"
        )
    return return_dtype, ptx_type, scope, space, returns[return_dtype]


def _validate_ld_space(space: str, allowed: set[str]) -> None:
    if space not in allowed:
        raise ValueError(
            f"Unsupported PTX ld state space {space!r}; expected one of {sorted(allowed)}"
        )


def _ptx_ld_helper_name(kind: str, return_dtype: str, ptx_type: str, scope: str | None, space: str):
    parts = ["tvm_builtin_ptx_ld", kind]
    if scope is not None:
        parts.append(scope.replace("::", "_"))
    parts.extend([space.replace("::", "_"), ptx_type, return_dtype])
    return "_".join(parts)


def _ptx_ld_parts(return_dtype, ptx_type, weak, space, cop, has_cache_hint):
    return_dtype, ptx_type, _scope, space, c_type = _parse_ld_attrs(
        return_dtype, ptx_type, None, space
    )
    cop = parse_str(cop)
    if cop not in _PTX_LD_COPS:
        raise ValueError(f"Unsupported PTX ld cache operation {cop!r}")
    weak = bool(int(weak)) if hasattr(weak, "value") else bool(weak)
    has_cache = (
        bool(int(has_cache_hint)) if hasattr(has_cache_hint, "value") else bool(has_cache_hint)
    )
    _validate_ld_space(space, _PTX_LD_VOLATILE_SPACES | {"param::entry", "param::func"})
    spec = _PTX_LD_TYPES[ptx_type]["constraint"]
    addr_decl = ""
    addr_operand = '"l"(address)'
    if space.startswith("shared"):
        addr_decl = "    unsigned int addr = (unsigned int)__cvta_generic_to_shared(address);\n"
        addr_operand = '"r"(addr)'
    modifiers = f"{'.weak' if weak else ''}.{space}{('.' + cop) if cop else ''}"
    cache_inst = ".L2::cache_hint" if has_cache else ""
    cache_slot = ", %2" if has_cache else ""
    cache_operand = ', "l"(cache_policy)' if has_cache else ""
    name = (
        "tvm_builtin_ptx_ld"
        f"{'_weak' if weak else ''}_{space.replace('::', '_').replace('.', '_')}"
        f"{('_' + cop) if cop else ''}_{ptx_type}_{return_dtype}"
        f"{'_cache_hint' if has_cache else ''}"
    )
    body = (
        f"    {c_type} ret;\n"
        f"{addr_decl}"
        f'    asm volatile("ld{modifiers}{cache_inst}.{ptx_type} %0, [%1]{cache_slot};" '
        f': "={spec}"(ret) : {addr_operand}{cache_operand});\n'
        "    return ret;"
    )
    return name, c_type, return_dtype, body


device_intrinsic(
    "ptx_ld",
    n_attrs=6,
    helper_name=lambda _addr, _cache_policy, return_dtype, weak, space, cop, ptx_type, has_cache: (
        _ptx_ld_parts(return_dtype, ptx_type, weak, space, cop, has_cache)[0]
    ),
    c_signature="(void* address, unsigned long long cache_policy)",
    return_type=lambda _addr, _cache_policy, return_dtype, weak, space, cop, ptx_type, has_cache: (
        _ptx_ld_parts(return_dtype, ptx_type, weak, space, cop, has_cache)[1]
    ),
    tvm_return_type=lambda _addr,
    _cache_policy,
    return_dtype,
    _weak,
    _space,
    _cop,
    _ptx_type,
    _has_cache: (parse_str(return_dtype)),
    body=lambda _addr, _cache_policy, return_dtype, weak, space, cop, ptx_type, has_cache: (
        _ptx_ld_parts(return_dtype, ptx_type, weak, space, cop, has_cache)[3]
    ),
)


def _ptx_ld_acquire_parts(return_dtype, ptx_type, scope, space):
    return_dtype, ptx_type, scope, space, c_type = _parse_ld_attrs(
        return_dtype, ptx_type, scope, space
    )
    _validate_ld_space(space, _PTX_LD_SPACES)
    spec = _PTX_LD_TYPES[ptx_type]["constraint"]
    addr_decl = ""
    addr_operand = '"l"(address)'
    if space.startswith("shared"):
        addr_decl = "    unsigned int addr = (unsigned int)__cvta_generic_to_shared(address);\n"
        addr_operand = '"r"(addr)'
    return (
        _ptx_ld_helper_name("acquire", return_dtype, ptx_type, scope, space),
        c_type,
        (
            f"    {c_type} ret;\n"
            f"{addr_decl}"
            f'    asm volatile("ld.acquire.{scope}.{space}.{ptx_type} %0, [%1];" '
            f': "={spec}"(ret) : {addr_operand});\n'
            "    return ret;"
        ),
        return_dtype,
    )


device_intrinsic(
    "ptx_ld_acquire",
    n_attrs=4,
    helper_name=lambda _addr, return_dtype, ptx_type, scope, space: _ptx_ld_acquire_parts(
        return_dtype, ptx_type, scope, space
    )[0],
    c_signature="(void* address)",
    return_type=lambda _addr, return_dtype, ptx_type, scope, space: _ptx_ld_acquire_parts(
        return_dtype, ptx_type, scope, space
    )[1],
    tvm_return_type=lambda _addr, return_dtype, _ptx_type, _scope, _space: parse_str(return_dtype),
    body=lambda _addr, return_dtype, ptx_type, scope, space: _ptx_ld_acquire_parts(
        return_dtype, ptx_type, scope, space
    )[2],
)


def _ptx_ld_volatile_parts(return_dtype, ptx_type, space):
    return_dtype, ptx_type, _scope, space, c_type = _parse_ld_attrs(
        return_dtype, ptx_type, None, space
    )
    _validate_ld_space(space, _PTX_LD_VOLATILE_SPACES)
    spec = _PTX_LD_TYPES[ptx_type]["constraint"]
    addr_decl = ""
    addr_operand = '"l"(address)'
    if space.startswith("shared"):
        addr_decl = "    unsigned int addr = (unsigned int)__cvta_generic_to_shared(address);\n"
        addr_operand = '"r"(addr)'
    return (
        _ptx_ld_helper_name("volatile", return_dtype, ptx_type, None, space),
        c_type,
        (
            f"    {c_type} ret;\n"
            f"{addr_decl}"
            f'    asm volatile("ld.volatile.{space}.{ptx_type} %0, [%1];" '
            f': "={spec}"(ret) : {addr_operand});\n'
            "    return ret;"
        ),
        return_dtype,
    )


device_intrinsic(
    "ptx_ld_volatile",
    n_attrs=3,
    helper_name=lambda _addr, return_dtype, ptx_type, space: _ptx_ld_volatile_parts(
        return_dtype, ptx_type, space
    )[0],
    c_signature="(void* address)",
    return_type=lambda _addr, return_dtype, ptx_type, space: _ptx_ld_volatile_parts(
        return_dtype, ptx_type, space
    )[1],
    tvm_return_type=lambda _addr, return_dtype, _ptx_type, _space: parse_str(return_dtype),
    body=lambda _addr, return_dtype, ptx_type, space: _ptx_ld_volatile_parts(
        return_dtype, ptx_type, space
    )[2],
)


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
    dtype = str(res.dtype)
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
    tvm_return_type=lambda _addr, value: value.dtype,
)
device_intrinsic(
    "cuda_atomic_cas",
    helper_name="tvm_builtin_cuda_atomic_cas",
    c_signature="(T* address, T compare, T val)",
    body="    return atomicCAS(address, compare, val);",
    return_type="T",
    templated=True,
    tvm_return_type=lambda _p, old, _n: old.dtype,
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

_PTX_SCALAR_TYPE_INFO = {
    "b32": ("unsigned int", "r", "uint32"),
    "u32": ("unsigned int", "r", "uint32"),
    "s32": ("int", "r", "int32"),
    "b64": ("unsigned long long", "l", "uint64"),
    "u64": ("unsigned long long", "l", "uint64"),
    "s64": ("long long", "l", "int64"),
    "f32": ("float", "f", "float32"),
    "f64": ("double", "d", "float64"),
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


# PTX st weak scalar/vector form:
#   st{.weak}{.ss}{.cop}{.level::cache_hint}{.vec}.type [a], b{, cache-policy};
def _ptx_st_parts(*args):
    weak, space, cop, vec, ptx_type, has_cache_hint = args[-6:]
    weak = bool(int(weak)) if hasattr(weak, "value") else bool(weak)
    space = parse_str(space)
    cop = parse_str(cop)
    vec = parse_str(vec)
    ptx_type, c_type, constraint, _tvm_dtype = _type_info(ptx_type)
    has_cache = (
        bool(int(has_cache_hint)) if hasattr(has_cache_hint, "value") else bool(has_cache_hint)
    )
    vec_len = int(vec[1:]) if vec else 1
    modifiers = f"{'.weak' if weak else ''}{_dot(space)}{_dot(cop)}"
    instr = f"st{modifiers}{_cache_suffix('cache' if has_cache else '')}{_dot(vec)}.{ptx_type}"
    name = (
        "tvm_builtin_ptx_st"
        f"{'_weak' if weak else ''}_{_safe_attr(space)}"
        f"{('_' + _safe_attr(cop)) if cop else ''}"
        f"{('_' + _safe_attr(vec)) if vec else ''}_{ptx_type}"
        f"{'_cache_hint' if has_cache else ''}"
    )
    value_params = ", ".join(f"{c_type} value{i}" for i in range(vec_len))
    c_signature = f"(void* address, {value_params}, unsigned long long cache_policy)"
    values = f"{{{', '.join(f'%{i + 1}' for i in range(vec_len))}}}" if vec else "%1"
    value_constraints = "".join(f', "{constraint}"(value{i})' for i in range(vec_len))
    cache_slot = f", %{vec_len + 1}" if has_cache else ""
    cache_operand = ', "l"(cache_policy)' if has_cache else ""
    addr_decl = ""
    addr_operand = '"l"(address)'
    if space.startswith("shared"):
        addr_decl = "    unsigned int addr = (unsigned int)__cvta_generic_to_shared(address);\n"
        addr_operand = '"r"(addr)'
    body = (
        f"{addr_decl}"
        f'    asm volatile("{instr} [%0], {values}{cache_slot};"\n'
        "                 :\n"
        f"                 : {addr_operand}{value_constraints}"
        f"{cache_operand}\n"
        '                 : "memory");'
    )
    return name, c_signature, body


device_intrinsic(
    "ptx_st",
    n_attrs=6,
    helper_name=lambda *a: _ptx_st_parts(*a)[0],
    c_signature=lambda *a: _ptx_st_parts(*a)[1],
    body=lambda *a: _ptx_st_parts(*a)[2],
)


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
