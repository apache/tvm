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
"""Profiling, tracing, and debug hooks.

Helpers that observe a running kernel rather than compute anything in it:
the TIRx profiler timer ring, the official IKET NativeDump event, the
cycle counter, and the ``printf`` / ``trap`` debug escapes.
"""

import hashlib
import json

import tvm
from tvm.backend.cuda.op import cuda_func_call

from ..codegen.registry import register_codegen
from ..codegen.schema import device_intrinsic
from ..codegen.utils import parse_str

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
# Official IKET NativeDump placeholder.
# =============================================================================
@register_codegen("cuda_iket_official_event")
def codegen_cuda_iket_official_event(event_id, source_code, payload=None):
    if isinstance(source_code, tvm.tirx.StringImm):
        source_code = source_code.value
    else:
        source_code = parse_str(source_code)
    args = (event_id,) if payload is None else (event_id, payload)
    return cuda_func_call(
        "tvm_builtin_iket_official_event",
        *args,
        source_code=source_code,
        return_type="uint32",
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
        raise ValueError("T.cuda.printf format must be a string literal")
    fmt_literal = json.dumps(fmt)
    arg_dtypes = [str(arg.ty) for arg in args]
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
        raise ValueError(f"Unsupported T.cuda.printf argument dtype: {dtype}")

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
