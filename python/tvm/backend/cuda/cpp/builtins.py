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
# pylint: disable=redefined-builtin, invalid-name
"""Thin wrappers around CUDA builtins and library calls.

Everything here compiles to a plain CUDA C++ expression -- an intrinsic
(``__ldg``, ``__shfl_xor_sync``, ``atomicAdd``), a type-punning
reinterpret, or a small arithmetic helper. Nothing in this module emits
inline asm; anything that needs a hand-written PTX body lives in
:mod:`.asm`, and single PTX instructions live in the ``T.ptx`` table.

* warp / CTA reductions (templated butterfly shuffle-XOR)
* packed float2 / bf16 / fp8 conversions and arithmetic
* ``__ldg`` scalar and vector loads, atomics, address casts
* thread queries, scheduling hints, and the CUDA-side sync helpers
"""

from tvm import DataType
from tvm.backend.cuda.op import cuda_func_call

from ..codegen.registry import register_codegen
from ..codegen.schema import device_intrinsic
from ..codegen.utils import parse_str, validate_power_of_two_range

# =============================================================================
# Scalar float math.
# =============================================================================
device_intrinsic(
    "cuda_fdividef",
    helper_name="tvm_builtin_cuda_fdividef",
    c_signature="(float x, float y)",
    return_type="float",
    tvm_return_type="float32",
    body="    return __fdividef(x, y);",
)


# =============================================================================
# CUDA-side warp / CTA reductions (templated butterfly shuffle-XOR).
# Emitted directly via ``cuda_func_call`` — the helper signature uses a
# single template parameter ``T`` for both arg and return, which doesn't
# match the operand-driven C signature pattern.
# =============================================================================

# (accumulation expression, identity value for cross-warp padding)
_OP_TABLE = {
    "sum": ("val += shuffled;", "T(0)"),
    "max": ("val = max(val, shuffled);", "-INFINITY"),
    "min": ("val = min(val, shuffled);", "INFINITY"),
}


def _validate_op(op_str, context):
    if op_str not in _OP_TABLE:
        raise ValueError(f"Unsupported {context} op '{op_str}', expected one of {list(_OP_TABLE)}")
    return _OP_TABLE[op_str]


def _warp_reduce_source(func_name, width_int, step_expr):
    return (
        f"\ntemplate <typename T>\n"
        f"__forceinline__ __device__ T {func_name}(T val) {{\n"
        f"    #pragma unroll\n"
        f"    for (int mask = {width_int} >> 1; mask > 0; mask >>= 1) {{\n"
        "        T shuffled = __shfl_xor_sync(0xFFFFFFFF, val, mask);\n"
        f"        {step_expr}\n"
        "    }\n"
        "    return val;\n"
        "}\n"
    )


@register_codegen("cuda_warp_reduce")
def codegen_cuda_warp_reduce(value, op, width):
    op_str = parse_str(op)
    width_int = validate_power_of_two_range(width, 2, 32, "warp_reduce width")
    step_expr, _ = _validate_op(op_str, "warp_reduce")

    func_name = f"tvm_builtin_cuda_warp_reduce_{op_str}_{width_int}"
    source_code = _warp_reduce_source(func_name, width_int, step_expr)
    return cuda_func_call(func_name, value, source_code=source_code, return_type=value.ty)


@register_codegen("cuda_cta_reduce")
def codegen_cuda_cta_reduce(value, op, num_warps, scratch):
    op_str = parse_str(op)
    nw = validate_power_of_two_range(num_warps, 1, 32, "cta_reduce num_warps")
    step_expr, identity = _validate_op(op_str, "cta_reduce")

    warp_reduce_name = f"tvm_builtin_cuda_warp_reduce_{op_str}_32"
    func_name = f"tvm_builtin_cuda_cta_reduce_{op_str}_{nw}"

    cta_body = (
        f"{_warp_reduce_source(warp_reduce_name, 32, step_expr)}"
        "template <typename T>\n"
        f"__forceinline__ __device__ T {func_name}(T val, void* scratch_raw) {{\n"
        "    T* scratch = reinterpret_cast<T*>(scratch_raw);\n"
        f"    val = {warp_reduce_name}(val);\n"
        "    int tid = threadIdx.x + threadIdx.y * blockDim.x"
        " + threadIdx.z * blockDim.x * blockDim.y;\n"
        "    int warp_id = tid / 32;\n"
        "    int lane_id = tid % 32;\n"
        "    if (lane_id == 0) scratch[warp_id] = val;\n"
        "    __syncthreads();\n"
        "    if (warp_id == 0) {\n"
        f"        T partial = (lane_id < {nw}) ? scratch[lane_id] : {identity};\n"
        f"        partial = {warp_reduce_name}(partial);\n"
        "        if (lane_id == 0) scratch[0] = partial;\n"
        "    }\n"
        "    __syncthreads();\n"
        "    return scratch[0];\n"
        "}\n"
    )
    return cuda_func_call(func_name, value, scratch, source_code=cta_body, return_type=value.ty)


# =============================================================================
# FP8 / BF16 packing, integer, and activation helpers.
# =============================================================================
device_intrinsic(
    "cuda_ffs_u32",
    helper_name="tvm_builtin_ffs_u32",
    c_signature="(unsigned int value)",
    return_type="int",
    body="    return __ffs(value);",
)

device_intrinsic(
    "cuda_make_float2",
    helper_name="tvm_builtin_make_float2",
    c_signature="(float x, float y)",
    return_type="unsigned long long",
    body=(
        "    float2 value = make_float2(x, y);\n"
        "    return *reinterpret_cast<unsigned long long*>(&value);"
    ),
)

device_intrinsic(
    "cuda_float2_x",
    helper_name="tvm_builtin_float2_x",
    c_signature="(unsigned long long packed)",
    return_type="float",
    body=("    float2 value = *reinterpret_cast<float2*>(&packed);\n    return value.x;"),
)

device_intrinsic(
    "cuda_float2_y",
    helper_name="tvm_builtin_float2_y",
    c_signature="(unsigned long long packed)",
    return_type="float",
    body=("    float2 value = *reinterpret_cast<float2*>(&packed);\n    return value.y;"),
)

device_intrinsic(
    "cuda_fmul2_rn",
    helper_name="tvm_builtin_fmul2_rn",
    c_signature="(unsigned long long a, unsigned long long b)",
    return_type="unsigned long long",
    body=(
        "    float2 lhs = *reinterpret_cast<float2*>(&a);\n"
        "    float2 rhs = *reinterpret_cast<float2*>(&b);\n"
        "    float2 result = __fmul2_rn(lhs, rhs);\n"
        "    return *reinterpret_cast<unsigned long long*>(&result);"
    ),
)

device_intrinsic(
    "cuda_fadd2_rn",
    helper_name="tvm_builtin_fadd2_rn",
    c_signature="(unsigned long long a, unsigned long long b)",
    return_type="unsigned long long",
    body=(
        "    float2 lhs = *reinterpret_cast<float2*>(&a);\n"
        "    float2 rhs = *reinterpret_cast<float2*>(&b);\n"
        "    float2 result = __fadd2_rn(lhs, rhs);\n"
        "    return *reinterpret_cast<unsigned long long*>(&result);"
    ),
)

device_intrinsic(
    "cuda_float22bfloat162_rn",
    helper_name="tvm_builtin_float22bfloat162_rn",
    c_signature="(float x, float y)",
    return_type="unsigned int",
    body=(
        "    __nv_bfloat162 value = __float22bfloat162_rn(make_float2(x, y));\n"
        "    return *reinterpret_cast<unsigned int*>(&value);"
    ),
    extra_deps=("bf16",),
)

device_intrinsic(
    "cuda_float22bfloat162_rn_from_float2",
    helper_name="tvm_builtin_float22bfloat162_rn_from_float2",
    c_signature="(unsigned long long packed)",
    return_type="unsigned int",
    body=(
        "    float2 value = *reinterpret_cast<float2*>(&packed);\n"
        "    __nv_bfloat162 result = __float22bfloat162_rn(value);\n"
        "    return *reinterpret_cast<unsigned int*>(&result);"
    ),
    extra_deps=("bf16",),
)

device_intrinsic(
    "cuda_bfloat1622float2",
    helper_name="tvm_builtin_bfloat1622float2",
    c_signature="(unsigned int packed)",
    return_type="unsigned long long",
    body=(
        "    __nv_bfloat162 value;\n"
        "    *reinterpret_cast<unsigned int*>(&value) = packed;\n"
        "    float2 result = __bfloat1622float2(value);\n"
        "    return *reinterpret_cast<unsigned long long*>(&result);"
    ),
    extra_deps=("bf16",),
)

device_intrinsic(
    "cuda_hmin2",
    helper_name="tvm_builtin_hmin2",
    c_signature="(unsigned int a, unsigned int b)",
    return_type="unsigned int",
    body=(
        "    __nv_bfloat162 lhs;\n"
        "    __nv_bfloat162 rhs;\n"
        "    *reinterpret_cast<unsigned int*>(&lhs) = a;\n"
        "    *reinterpret_cast<unsigned int*>(&rhs) = b;\n"
        "    __nv_bfloat162 result = __hmin2(lhs, rhs);\n"
        "    return *reinterpret_cast<unsigned int*>(&result);"
    ),
    extra_deps=("bf16",),
)

device_intrinsic(
    "cuda_hmax2",
    helper_name="tvm_builtin_hmax2",
    c_signature="(unsigned int a, unsigned int b)",
    return_type="unsigned int",
    body=(
        "    __nv_bfloat162 lhs;\n"
        "    __nv_bfloat162 rhs;\n"
        "    *reinterpret_cast<unsigned int*>(&lhs) = a;\n"
        "    *reinterpret_cast<unsigned int*>(&rhs) = b;\n"
        "    __nv_bfloat162 result = __hmax2(lhs, rhs);\n"
        "    return *reinterpret_cast<unsigned int*>(&result);"
    ),
    extra_deps=("bf16",),
)

device_intrinsic(
    "cuda_fp8x4_e4m3_from_float4",
    helper_name="tvm_builtin_fp8x4_e4m3_from_float4",
    c_signature="(float x, float y, float z, float w)",
    return_type="unsigned int",
    body=(
        "    __nv_fp8x4_e4m3 result = __nv_fp8x4_e4m3(make_float4(x, y, z, w));\n"
        "    return *reinterpret_cast<unsigned int*>(&result);"
    ),
    extra_deps=("fp8",),
)


# =============================================================================
# __ldg — typed read-only cached loads, scalar and vector. Source is ``void*``
# so callers may pass a typed pointer or handle_add_byte_offset result; the
# helper casts per ``dtype``.
# =============================================================================


def _int_attr(value):
    return int(value.value) if hasattr(value, "value") else int(value)


_CUDA_LDG_CTYPES = {
    "int8": "signed char",
    "uint8": "unsigned char",
    "int16": "short",
    "uint16": "unsigned short",
    "int32": "int",
    "uint32": "unsigned int",
    "int64": "long long",
    "uint64": "unsigned long long",
    "float16": "half",
    "bfloat16": "nv_bfloat16",
    "float32": "float",
    "float64": "double",
}

_CUDA_LDG_VECTOR_CTYPES = {
    "int32": "int",
    "uint32": "unsigned int",
    "float32": "float",
}
_CUDA_LDG_VECTOR_BASES = {"int32": "int", "uint32": "uint", "float32": "float"}


def _cuda_ldg_suffix(dtype: str) -> str:
    return dtype.replace("float", "f").replace("uint", "u").replace("int", "i")


@register_codegen("cuda_ldg")
def codegen_cuda_ldg(*args):
    if len(args) == 2:
        addr, dtype = args
        dtype = str(DataType(parse_str(dtype)))
        if dtype not in _CUDA_LDG_CTYPES:
            raise ValueError(f"Unsupported CUDA __ldg dtype {dtype!r}")
        c_type = _CUDA_LDG_CTYPES[dtype]
        func_name = f"tvm_builtin_cuda_ldg_{_cuda_ldg_suffix(dtype)}"
        source_code = f"""
__forceinline__ __device__ {c_type} {func_name}(void* src) {{
    return __ldg(reinterpret_cast<const {c_type}*>(src));
}}
"""
        return cuda_func_call(func_name, addr, source_code=source_code, return_type=dtype)

    if len(args) < 5:
        raise ValueError(f"cuda_ldg expects 2 args or vector form, got {len(args)}")
    *dsts, addr, dtype, vec, dst_count = args
    dtype = str(DataType(parse_str(dtype)))
    vec = parse_str(vec)
    dst_count = _int_attr(dst_count)
    vec_len = int(vec[1:]) if vec else 1
    if dtype not in _CUDA_LDG_VECTOR_CTYPES:
        raise ValueError(f"Unsupported vector CUDA __ldg dtype {dtype!r}")
    if vec not in ("v2", "v4") or dst_count != vec_len or len(dsts) != vec_len:
        raise ValueError(
            f"vector CUDA __ldg expects dst_count=len(dsts)=vec_len for v2/v4, "
            f"got vec={vec!r}, dst_count={dst_count}, len(dsts)={len(dsts)}"
        )
    c_type = _CUDA_LDG_VECTOR_CTYPES[dtype]
    vec_type = f"{_CUDA_LDG_VECTOR_BASES[dtype]}{vec_len}"
    members = ("x", "y", "z", "w")[:vec_len]
    func_name = f"tvm_builtin_cuda_ldg_{_cuda_ldg_suffix(dtype)}_{vec}_to_dst{dst_count}"
    params = ", ".join(f"void* dst{i}" for i in range(vec_len))
    stores = "\n".join(
        f"    *reinterpret_cast<{c_type}*>(dst{i}) = v.{member};"
        for i, member in enumerate(members)
    )
    source_code = f"""
__forceinline__ __device__ void {func_name}({params}, void* src) {{
    {vec_type} v = __ldg(reinterpret_cast<const {vec_type}*>(src));
{stores}
}}
"""
    return cuda_func_call(func_name, *dsts, addr, source_code=source_code, return_type="void")


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


# =============================================================================
# Per-thread queries / scheduling hints.
# =============================================================================
device_intrinsic(
    "cuda_thread_rank",
    body=(
        "    namespace cg = cooperative_groups;\n    return cg::this_thread_block().thread_rank();"
    ),
    return_type="int",
    tvm_return_type="int32",
    extra_deps=("cooperative_groups",),
)
device_intrinsic("cuda_nano_sleep", c_signature="(uint64_t time)", body="    __nanosleep(time);")


# =============================================================================
# elect.sync — TIRx uses the CUDA builtin ``tvm_builtin_elect_one_sync()``
# helper (declared in the CUDA header tags), not direct PTX.
# =============================================================================
device_intrinsic(
    "cuda_elect_sync",
    helper_name="tvm_builtin_elect_one_sync_op",
    return_type="uint32_t",
    body="    return tvm_builtin_elect_one_sync();",
    extra_deps=("elect_one_sync",),
)


# =============================================================================
# __any_sync — warp-vote (pure CUDA helper).
# =============================================================================
device_intrinsic(
    "cuda_any_sync",
    c_signature="(unsigned mask, int pred)",
    body="    return __any_sync(mask, pred);",
    return_type="int",
    tvm_return_type="int32",
)


# =============================================================================
# CUDA-side sync helpers (zero-arg void unless noted).
# =============================================================================
device_intrinsic("cuda_thread_fence", body="    __threadfence();")
device_intrinsic("cuda_warp_sync", body="    __syncwarp();")
device_intrinsic("cuda_cta_sync", body="    __syncthreads();")
device_intrinsic(
    "cuda_grid_sync",
    body="    namespace cg = cooperative_groups;\n    cg::this_grid().sync();",
    extra_deps=("cooperative_groups",),
)
device_intrinsic(
    "cuda_syncthreads_and",
    c_signature="(int predicate)",
    body="    return __syncthreads_and(predicate);",
    return_type="int",
    tvm_return_type="int32",
)
device_intrinsic(
    "cuda_syncthreads_or",
    c_signature="(int predicate)",
    body="    return __syncthreads_or(predicate);",
    return_type="int",
    tvm_return_type="int32",
)


# =============================================================================
# Warp collectives — ballot and the hardware ``__reduce_*_sync`` reductions.
# =============================================================================
device_intrinsic(
    "cuda_ballot_sync",
    helper_name="tvm_builtin_ballot_sync",
    c_signature="(unsigned int mask, int pred)",
    return_type="unsigned int",
    body="    return __ballot_sync(mask, pred);",
)
device_intrinsic(
    "cuda_reduce_add_sync_u32",
    helper_name="tvm_builtin_reduce_add_sync_u32",
    c_signature="(unsigned int mask, unsigned int value)",
    return_type="unsigned int",
    body="    return __reduce_add_sync(mask, value);",
)
device_intrinsic(
    "cuda_reduce_min_sync_u32",
    helper_name="tvm_builtin_reduce_min_sync_u32",
    c_signature="(unsigned int mask, unsigned int value)",
    return_type="unsigned int",
    body="    return __reduce_min_sync(mask, value);",
)
