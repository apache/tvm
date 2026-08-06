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
# Inputs are packed into a `.b64` register (low half = elem 0, high half =
# elem 1); the body packs/unpacks via ``make_float2`` + ``reinterpret_cast``.
# =============================================================================


def _as_bool_attr(value):
    return bool(int(value))


device_intrinsic(
    "cuda_fdividef",
    helper_name="tvm_builtin_cuda_fdividef",
    c_signature="(float x, float y)",
    return_type="float",
    tvm_return_type="float32",
    body="    return __fdividef(x, y);",
)


# =============================================================================
# 3-operand max.f32 / min.f32 — the f32, 3-operand form-table entry of the
# redux/reduction-style fp32 max/min ops.
# =============================================================================
_ABC_SIG = "(float a, float b, float c)"


_BINARY_F32_SIG = "(float a, float b)"


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
