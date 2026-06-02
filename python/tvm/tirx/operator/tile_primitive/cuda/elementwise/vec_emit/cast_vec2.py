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

"""Packed vec_len=2 cast via CUDA pair intrinsics.

Each supported (src_dtype, dst_dtype) pair has a CUDA builtin that converts a
packed-2 source to a packed-2 destination in one instruction
(e.g. ``__float22half2_rn``). The intrinsic takes pointers to the first
element of each packed pair on either side.
"""

from __future__ import annotations

from tvm.ir.expr import PrimExpr
from tvm.script import tirx as Tx

from ..ops import VecImpl

_VEC2_CAST_INTRINSICS = {
    ("float32", "float16"): "__float22half2_rn",
    ("float16", "float32"): "__half22float2",
    ("bfloat16", "float32"): "__bfloat1622float2",
    ("float32", "bfloat16"): "__float22bfloat162_rn",
}
_DTYPE_X2_NAME = {"float32": "float2", "float16": "half2", "bfloat16": "nv_bfloat162"}


def _intrinsic_name(src_dtype, dst_dtype):
    return f"tvm_builtin_cast_{src_dtype}x2_{dst_dtype}x2"


def _intrinsic_source(src_dtype, dst_dtype):
    intrinsic = _VEC2_CAST_INTRINSICS[(src_dtype, dst_dtype)]
    return (
        f"\n__forceinline__ __device__ void {_intrinsic_name(src_dtype, dst_dtype)}"
        f"(void* dst, void* src) {{\n"
        f"    (({_DTYPE_X2_NAME[dst_dtype]}*)dst)[0] = "
        f"{intrinsic}((({_DTYPE_X2_NAME[src_dtype]}*)src)[0]);\n"
        "}\n"
    )


def _cast_vec2_applies(op_call, sctx, plan):
    if len(plan.srcs) != 1 or plan.srcs[0].is_scalar:
        return False, "cast requires 1 buffer src"
    src = plan.srcs[0]
    if src.index_fn is not None:
        return False, "broadcasting src not supported by cast vec2"
    src_dtype = src.buf_region.buffer.dtype
    dst_dtype = plan.dst.buffer.dtype
    if (src_dtype, dst_dtype) not in _VEC2_CAST_INTRINSICS:
        return False, f"no vec2 intrinsic for {src_dtype}->{dst_dtype}"
    return True, None


def _emit_cast_vec2(dst_buf, dst_lane_indices, src_args, extras) -> PrimExpr:
    src_arg = src_args[0]
    # cast_vec2 requires buffer src (guarded by applies()).
    assert isinstance(src_arg, tuple), "cast vec2 src must be a buffer"
    src_buf, src_lane_indices = src_arg
    func_name = _intrinsic_name(src_buf.dtype, dst_buf.dtype)
    source_code = _intrinsic_source(src_buf.dtype, dst_buf.dtype)
    return Tx.cuda.func_call(
        func_name,
        Tx.address_of(dst_buf[tuple(dst_lane_indices[0])]),
        Tx.address_of(src_buf[tuple(src_lane_indices[0])]),
        source_code=source_code,
    )


CAST_VEC2_IMPL = VecImpl(
    vec_len=2,
    applies=_cast_vec2_applies,
    emit=_emit_cast_vec2,
)
