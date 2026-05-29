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
# pylint: disable=redefined-builtin, invalid-name
"""Miscellaneous device helpers.

Catch-all for ops that don't fit the (sync / mma / cp_async / memory / math /
nvshmem) feature buckets:

* PTX register-allocation control: ``setmaxnreg`` / ``mov`` from special reg.
* Per-thread queries / scheduling hints: ``thread_rank`` / ``nano_sleep``.
* Profiler timer hooks (``timer_init/start/end/finalize``).
* Debug helpers: ``printf`` / ``trap`` on assert failure.
"""

import hashlib
import json

import tvm
from tvm.tirx.op import cuda_func_call

from .._schema import device_intrinsic
from .registry import CODEGEN_REGISTRY, register_codegen
from .utils import parse_str

# =============================================================================
# setmaxnreg.{inc,dec}.sync.aligned.u32 — 1 PTX form (.action picks inc/dec).
# =============================================================================


def _ptx_setmaxnreg(inc, nreg):
    inc = bool(int(inc)) if hasattr(inc, "value") else bool(inc)
    nreg = int(nreg)
    action = "inc" if inc else "dec"
    return (
        f"tvm_builtin_ptx_setmaxnreg_{action}_{nreg}",
        f'    asm volatile("setmaxnreg.{action}.sync.aligned.u32 {nreg};");',
    )


device_intrinsic(
    "ptx_setmaxnreg",
    n_attrs=2,
    helper_name=lambda inc, nreg: _ptx_setmaxnreg(inc, nreg)[0],
    body=lambda inc, nreg: _ptx_setmaxnreg(inc, nreg)[1],
)


# =============================================================================
# mov.u32/u64 from special register — 1 PTX form (Form 2 of mov.type d, sreg).
# Each (bits, reg) emits a distinct helper because the special reg name is
# baked into the PTX text.
# =============================================================================


def _ptx_fetch_register_body(bits):
    spec = "l" if bits == 64 else "r"

    def _body(reg):
        reg = parse_str(reg)
        return (
            f"    uint{bits}_t x;\n"
            f'    asm volatile("mov.u{bits} %0, %{reg};" : "={spec}"(x));\n'
            f"    return (int{bits}_t)x;"
        )

    return _body


for _bits in (32, 64):
    device_intrinsic(
        f"ptx_fetch_register_{_bits}",
        n_attrs=1,
        helper_name=(
            lambda *a, bits=_bits: (
                f"tvm_builtin_ptx_fetch_register_"
                f"{parse_str(a[-1]).replace('::', '_').replace('.', '_')}"
            )
        ),
        return_type=f"int{_bits}_t",
        body=_ptx_fetch_register_body(_bits),
    )
del _bits


@register_codegen("ptx_fetch_register")
def codegen_ptx_fetch_register(bits, reg):
    bits = int(bits)
    reg = parse_str(reg)
    if bits not in (32, 64):
        raise ValueError(f"Only support 32/64 bits for ptx_fetch_register, but got {bits}.")
    result = CODEGEN_REGISTRY[f"tirx.ptx_fetch_register_{bits}"]([reg])
    return result[0] if isinstance(result, tuple) else result


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
# Profiler timer hooks.
# =============================================================================
_COMMON_PARAMS = (
    "uint64_t* profiler_buffer, uint64_t* profiler_tag, "
    "uint32_t* profiler_write_offset, int profiler_write_stride, bool leader_cond"
)
_EVENT_PARAMS = f"int event_type, {_COMMON_PARAMS}"


def _write_event(event_bits: str) -> str:
    return (
        "profiler_buffer[profiler_write_offset[0]] = "
        "((uint64_t)tvm_builtin_get_timestamp() << 32) | "
        f"(profiler_tag[0] | {event_bits});\n"
        "        profiler_write_offset[0] += profiler_write_stride;"
    )


device_intrinsic(
    "timer_init_cuda",
    c_signature=(
        "(uint64_t* profiler_buffer, uint64_t* profiler_tag, "
        "uint32_t* profiler_write_offset, int num_groups, int group_id)"
    ),
    body=(
        "    const uint32_t NBLOCKS = (uint32_t)(gridDim.x * gridDim.y * gridDim.z);\n"
        "    const uint32_t BLOCK_IDX = (uint32_t)("
        "(blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);\n"
        "    const uint32_t NGROUPS = num_groups;\n"
        "    const uint32_t GROUP_ID = group_id;\n"
        "    const uint32_t BLOCK_GROUP_IDX = BLOCK_IDX * NGROUPS + GROUP_ID;\n"
        "    if ((blockIdx.x == 0) && (blockIdx.y == 0) && "
        "(blockIdx.z == 0) && (threadIdx.x == 0)) {\n"
        "        profiler_buffer[0] = ((uint64_t)NGROUPS << 32) | NBLOCKS;\n"
        "    }\n"
        "    profiler_write_offset[0] = 1 + BLOCK_GROUP_IDX;\n"
        "    profiler_tag[0] = (uint64_t)BLOCK_GROUP_IDX << 12;"
    ),
)

device_intrinsic(
    "timer_start_cuda",
    c_signature=f"({_EVENT_PARAMS})",
    body=(
        f"    if (leader_cond) {{\n        {_write_event('(uint32_t)event_type << 2 | 0x0')}\n    }}\n"
        "    __threadfence_block();"
    ),
    extra_deps=("get_time_stamp",),
)

device_intrinsic(
    "timer_end_cuda",
    c_signature=f"({_EVENT_PARAMS})",
    body=(
        "    __threadfence_block();\n"
        f"    if (leader_cond) {{\n        {_write_event('(uint32_t)event_type << 2 | 0x1')}\n    }}"
    ),
    extra_deps=("get_time_stamp",),
)

device_intrinsic(
    "timer_finalize_cuda",
    c_signature=f"({_COMMON_PARAMS})",
    body=(
        f"    __threadfence_block();\n    if (leader_cond) {{\n        {_write_event('0x3')}\n    }}"
    ),
    extra_deps=("get_time_stamp",),
)


# =============================================================================
# Debug helpers — ``printf`` (variadic templated) and ``trap`` on assert.
# =============================================================================
device_intrinsic(
    "cuda_trap_when_assert_failed",
    c_signature="(bool cond)",
    body='    do {\n        if (not (cond))\n            asm("trap;");\n    } while (0);',
)


@register_codegen("cuda_printf")
def codegen_cuda_printf(fmt, *args):
    if isinstance(fmt, tvm.tirx.StringImm):
        fmt = fmt.value
    if not isinstance(fmt, str):
        raise ValueError("Tx.cuda.printf format must be a string literal")
    fmt_literal = json.dumps(fmt)
    arg_dtypes = [str(arg.dtype) for arg in args]
    signature = "|".join([fmt, *arg_dtypes])
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()
    func_name = f"tvm_builtin_cuda_printf_{len(args)}_{digest}"

    def c_type(dtype: str) -> str:
        if dtype == "float32":
            return "float"
        if dtype == "float64":
            return "double"
        if dtype in {"int8", "int16", "int32"}:
            return "int"
        if dtype == "int64":
            return "long long"
        if dtype in {"uint8", "uint16", "uint32"}:
            return "unsigned int"
        if dtype == "uint64":
            return "unsigned long long"
        if dtype == "bool":
            return "int"
        if dtype == "handle":
            return "void*"
        raise ValueError(f"Unsupported Tx.cuda.printf argument dtype: {dtype}")

    params = ", ".join(f"{c_type(dtype)} arg{i}" for i, dtype in enumerate(arg_dtypes))
    call_args = ", ".join(f"arg{i}" for i in range(len(args)))
    comma_call_args = f", {call_args}" if call_args else ""
    source_code = f"""
__noinline__ __device__ void {func_name}({params}) {{
    printf({fmt_literal}{comma_call_args});
}}
"""
    return cuda_func_call(func_name, *args, source_code=source_code)


device_intrinsic(
    "cuda_clock64",
    helper_name="tvm_builtin_clock64",
    return_type="unsigned long long",
    body="    return clock64();",
)
