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

"""Packed f32x2 VecImpl for FMA on sm_100+.

PTX: ``fma.<rm>.ftz.f32x2 d, a, b, c`` — 2 f32 FMAs per call. Same Python-side
shape collapse as binary_f32x2.
"""

from __future__ import annotations

from tvm.ir.expr import PrimExpr
from tvm.script import tirx as T

from ..ops import VecImpl
from .binary_f32x2 import _lane


def _fma_f32x2_applies(op_call, sctx, plan):
    from ...common import sm_version_ok

    if plan.dst.buffer.dtype != "float32":
        return False, "dst dtype not f32"
    if not sm_version_ok(op_call, sctx, min_version=100)[0]:
        return False, "sm version < 100"
    if len(plan.srcs) != 3:
        return False, "fma requires 3 srcs"
    a, b, c = plan.srcs
    if a.is_scalar:
        return False, "fma 'a' must be a buffer (no scalar-a packed FMA)"
    if a.buf_region.buffer.dtype != "float32":
        return False, "src a dtype not f32"
    if a.index_fn is not None:
        return False, "broadcasting src a not supported"
    for s in (b, c):
        if s.is_scalar:
            if s.scalar.dtype != "float32":
                return False, "scalar b/c dtype not f32"
        else:
            if s.buf_region.buffer.dtype != "float32":
                return False, "buffer b/c dtype not f32"
            if s.index_fn is not None:
                return False, "broadcasting src b/c not supported"
    return True, None


def _emit_fma_f32x2(dst_buf, dst_lane_indices, src_args, extras) -> PrimExpr:
    a_arg, b_arg, c_arg = src_args
    rm = extras.get("rounding_mode", "rz")
    return T.ptx.fma_f32x2(
        T.address_of(dst_buf[tuple(dst_lane_indices[0])]),
        T.cuda.make_float2(_lane(a_arg, 0), _lane(a_arg, 1)),
        T.cuda.make_float2(_lane(b_arg, 0), _lane(b_arg, 1)),
        T.cuda.make_float2(_lane(c_arg, 0), _lane(c_arg, 1)),
        rounding=rm,
        ftz=True,
    )


FMA_F32X2_IMPL = VecImpl(
    vec_len=2,
    applies=_fma_f32x2_applies,
    emit=_emit_fma_f32x2,
)
