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
"""Math intrinsics.

PTX side:
* ``add{.rnd}{.ftz}.f32x2`` / ``sub`` / ``mul`` / ``fma`` — packed f32x2.
* ``ex2.approx.ftz.f32`` / ``rcp.approx.ftz.f32`` — special functions.
* ``max.f32`` / ``min.f32`` — 3-operand reduction form.

CUDA side:
* warp / CTA reductions (templated butterfly shuffle-XOR).
"""

from tvm.backend.cuda.op import cuda_func_call

from ._schema import device_intrinsic
from .registry import register_codegen
from .utils import parse_str, validate_power_of_two_range

# =============================================================================
# Packed f32x2 arithmetic — `add{.rnd}{.ftz}.f32x2 d, a, b ;` and friends.
# Inputs are packed into a `.b64` register (low half = elem 0, high half =
# elem 1); the body packs/unpacks via ``make_float2`` + ``reinterpret_cast``.
# =============================================================================

# PTX add/sub/mul/fma over (f32 | f32x2 | f64), DPS form.
#   add{.rnd}{.ftz}{.sat}.f32     [d], a, b
#   add{.rnd}{.ftz}.f32x2          [d], a, b      (a,b are packed-as-u64)
#   add{.rnd}.f64                  [d], a, b
#   (sub / mul same shape; fma adds a `c` operand)
# Inputs a/b/c are register operands (scalar fp32 / packed u64 / scalar fp64).
# Result is written through `d` (a pointer).
_PACKED_ROUNDING = ("rz", "rn", "rm", "rp")


# Per-dtype operand types and asm constraints.
#  - c_in: C type of input register operand (matches PTX register type)
#  - out_cast: pointer cast applied at d_addr (callers may pass float*/double*/...)
#  - in_cstr / out_cstr: GCC asm constraint letter
_DTYPE_INFO = {
    "f32": {"c_in": "float", "out_cast": "float*", "in_cstr": "f", "out_cstr": "f"},
    "f32x2": {
        "c_in": "unsigned long long",
        "out_cast": "uint64_t*",
        "in_cstr": "l",
        "out_cstr": "l",
    },
    "f64": {"c_in": "double", "out_cast": "double*", "in_cstr": "d", "out_cstr": "d"},
}


def _ptx_arith_modifier_string(dtype, rounding, ftz, sat):
    """Build the `.rnd.ftz.sat` modifier substring + name suffix."""
    rnd = parse_str(rounding)
    assert rnd in _PACKED_ROUNDING, f"invalid rounding {rnd!r}, expected one of {_PACKED_ROUNDING}"
    ftz_b = bool(int(ftz)) if hasattr(ftz, "value") else bool(ftz)
    sat_b = bool(int(sat)) if hasattr(sat, "value") else bool(sat)
    if dtype == "f64" and (ftz_b or sat_b):
        raise ValueError("PTX <op>.f64 does not accept .ftz or .sat")
    if dtype == "f32x2" and sat_b:
        raise ValueError("PTX <op>.f32x2 does not accept .sat")
    mod = f".{rnd}"
    if ftz_b:
        mod += ".ftz"
    if sat_b:
        mod += ".sat"
    name_suffix = f"_{rnd}"
    if ftz_b:
        name_suffix += "_ftz"
    if sat_b:
        name_suffix += "_sat"
    return mod, name_suffix


def _ptx_binary_arith_parts(op, dtype):
    """Return (name_fn, sig, body_fn) for ptx_{op}_{dtype} binary form."""
    info = _DTYPE_INFO[dtype]
    # Destination is ``void*`` so callers can pass any element-type pointer
    # (float* / double* / uint64_t*); body reinterpret-casts to the right type.
    sig = f"(void* d, {info['c_in']} a, {info['c_in']} b)"

    def _name(d, a, b, rounding, ftz, sat):
        _, suf = _ptx_arith_modifier_string(dtype, rounding, ftz, sat)
        return f"tvm_builtin_ptx_{op}_{dtype}{suf}"

    out_c = info["out_cstr"]
    in_c = info["in_cstr"]
    out_cast = info["out_cast"]

    def _body(d, a, b, rounding, ftz, sat):
        mod, _ = _ptx_arith_modifier_string(dtype, rounding, ftz, sat)
        return (
            f'    asm volatile("{op}{mod}.{dtype} %0, %1, %2;"\n'
            f'        : "={out_c}"(*reinterpret_cast<{out_cast}>(d))\n'
            f'        : "{in_c}"(a), "{in_c}"(b));'
        )

    return _name, sig, _body


def _ptx_fma_parts(dtype):
    """Return (name_fn, sig, body_fn) for ptx_fma_{dtype}."""
    info = _DTYPE_INFO[dtype]
    sig = f"(void* d, {info['c_in']} a, {info['c_in']} b, {info['c_in']} c)"

    def _name(d, a, b, c, rounding, ftz, sat):
        _, suf = _ptx_arith_modifier_string(dtype, rounding, ftz, sat)
        return f"tvm_builtin_ptx_fma_{dtype}{suf}"

    out_c = info["out_cstr"]
    in_c = info["in_cstr"]
    out_cast = info["out_cast"]

    def _body(d, a, b, c, rounding, ftz, sat):
        mod, _ = _ptx_arith_modifier_string(dtype, rounding, ftz, sat)
        return (
            f'    asm volatile("fma{mod}.{dtype} %0, %1, %2, %3;"\n'
            f'        : "={out_c}"(*reinterpret_cast<{out_cast}>(d))\n'
            f'        : "{in_c}"(a), "{in_c}"(b), "{in_c}"(c));'
        )

    return _name, sig, _body


# Register 12 ops: {add, sub, mul, fma} x {f32, f32x2, f64}.
for _dtype in ("f32", "f32x2", "f64"):
    for _op in ("add", "sub", "mul"):
        _name_fn, _sig, _body_fn = _ptx_binary_arith_parts(_op, _dtype)
        device_intrinsic(
            f"ptx_{_op}_{_dtype}",
            n_attrs=3,  # rounding, ftz, sat
            helper_name=_name_fn,
            c_signature=_sig,
            body=_body_fn,
        )
    _name_fn, _sig, _body_fn = _ptx_fma_parts(_dtype)
    device_intrinsic(
        f"ptx_fma_{_dtype}",
        n_attrs=3,
        helper_name=_name_fn,
        c_signature=_sig,
        body=_body_fn,
    )
del _dtype, _op, _name_fn, _sig, _body_fn


# =============================================================================
# ex2.approx.ftz.f32 / rcp.approx.ftz.f32 — 1 form each.
# =============================================================================
device_intrinsic(
    "ptx_exp2",
    c_signature="(float x)",
    return_type="float",
    body=(
        "    float result;\n"
        '    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));\n'
        "    return result;"
    ),
)
device_intrinsic(
    "ptx_rcp",
    c_signature="(float x)",
    return_type="float",
    body=(
        "    float result;\n"
        '    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));\n'
        "    return result;"
    ),
)


# =============================================================================
# 3-operand max.f32 / min.f32 — the f32, 3-operand form-table entry of the
# redux/reduction-style fp32 max/min ops.
# =============================================================================
_ABC_SIG = "(float a, float b, float c)"
device_intrinsic(
    "ptx_reduce3_max_f32",
    c_signature=_ABC_SIG,
    return_type="float",
    body=(
        "    float result;\n"
        '    asm volatile("max.f32 %0, %1, %2, %3;"\n'
        '                 : "=f"(result) : "f"(a), "f"(b), "f"(c));\n'
        "    return result;"
    ),
)
device_intrinsic(
    "ptx_reduce3_min_f32",
    c_signature=_ABC_SIG,
    return_type="float",
    body=(
        "    float result;\n"
        '    asm volatile("min.f32 %0, %1, %2, %3;"\n'
        '                 : "=f"(result) : "f"(a), "f"(b), "f"(c));\n'
        "    return result;"
    ),
)


_BINARY_F32_SIG = "(float a, float b)"


def _ptx_max_f32_body(a, b, ftz, nan):
    ftz_b = bool(int(ftz)) if hasattr(ftz, "value") else bool(ftz)
    nan_b = bool(int(nan)) if hasattr(nan, "value") else bool(nan)
    ftz_suffix = ".ftz" if ftz_b else ""
    nan_suffix = ".NaN" if nan_b else ""
    return (
        "    float result;\n"
        f'    asm volatile("max{ftz_suffix}{nan_suffix}.f32 %0, %1, %2;"\n'
        '                 : "=f"(result) : "f"(a), "f"(b));\n'
        "    return result;"
    )


def _ptx_max_f32_name(a, b, ftz, nan):
    ftz_b = bool(int(ftz)) if hasattr(ftz, "value") else bool(ftz)
    nan_b = bool(int(nan)) if hasattr(nan, "value") else bool(nan)
    suffix = ""
    if ftz_b:
        suffix += "_ftz"
    if nan_b:
        suffix += "_nan"
    return f"tvm_builtin_ptx_max_f32{suffix}"


device_intrinsic(
    "ptx_max_f32",
    n_attrs=2,
    helper_name=_ptx_max_f32_name,
    c_signature=_BINARY_F32_SIG,
    return_type="float",
    body=_ptx_max_f32_body,
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
# Additional FP8/BF16 packing, integer, and activation helpers.
# =============================================================================

# PTX integer bit-search form:
#   fns.b32 d, mask, base, offset;
device_intrinsic(
    "ptx_fns_b32",
    helper_name="tvm_builtin_ptx_fns_b32",
    c_signature="(unsigned int mask, unsigned int base, int offset)",
    return_type="unsigned int",
    body=(
        "    unsigned int ret;\n"
        '    asm("fns.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(mask), "r"(base), "r"(offset));\n'
        "    return ret;"
    ),
)

device_intrinsic(
    "cuda_ffs_u32",
    helper_name="tvm_builtin_ffs_u32",
    c_signature="(unsigned int value)",
    return_type="int",
    body="    return __ffs(value);",
)

device_intrinsic(
    "ptx_add_rn_f32_bf16",
    helper_name="tvm_builtin_ptx_add_rn_f32_bf16",
    c_signature="(float acc, unsigned short x)",
    return_type="float",
    body=('    asm("add.rn.f32.bf16 %0, %1, %0;" : "+f"(acc) : "h"(x));\n    return acc;'),
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
