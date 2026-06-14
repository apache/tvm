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

"""Packed f32x2 VecImpls for binary add/sub/mul on sm_100+.

PTX op family: ``{add,sub,mul}.<rm>.ftz.f32x2``. Each call processes 2 f32s
per operand. The old ``_make_binary_packed_f32x2_factory`` (240+ lines, 8
``@T.prim_func`` shape combos per op) collapses to one ``emit`` per op
because operand-shape branching is now Python-level (outside any
``@T.prim_func``).
"""

from __future__ import annotations

from tvm.ir.expr import PrimExpr
from tvm.script import tirx as T

from ..ops import VecImpl


def _lane(arg, k):
    """Read lane ``k`` of one operand argument.

    arg is either a scalar Expr (broadcast) or ``(Buffer, lane_indices)``.
    """
    if isinstance(arg, tuple):
        buf, lane_indices = arg
        return buf[tuple(lane_indices[k])]
    return arg


def _f32x2_applies(op_name):
    """Predicate: f32 dst+srcs, sm_100+, no broadcasting srcs, two srcs."""

    def applies(op_call, sctx, plan):
        from ...common import sm_version_ok

        if plan.dst.buffer.dtype != "float32":
            return False, "dst dtype not f32"
        if not sm_version_ok(op_call, sctx, min_version=100)[0]:
            return False, "sm version < 100"
        if len(plan.srcs) != 2:
            return False, "binary requires 2 srcs"
        for s in plan.srcs:
            if s.is_scalar:
                if s.scalar.dtype != "float32":
                    return False, "scalar src dtype not f32"
            else:
                if s.buf_region.buffer.dtype != "float32":
                    return False, "buffer src dtype not f32"
                if s.index_fn is not None:
                    return False, "broadcasting src not supported by f32x2 packed"
        return True, None

    return applies


def _emit_binary_f32x2_for(op_name):
    op_func = getattr(T.ptx, f"{op_name}_f32x2")

    def emit(dst_buf, dst_lane_indices, src_args, extras) -> PrimExpr:
        a_arg, b_arg = src_args
        rm = extras.get("rounding_mode", "rz")
        return op_func(
            T.address_of(dst_buf[tuple(dst_lane_indices[0])]),
            T.cuda.make_float2(_lane(a_arg, 0), _lane(a_arg, 1)),
            T.cuda.make_float2(_lane(b_arg, 0), _lane(b_arg, 1)),
            rounding=rm,
            ftz=True,
        )

    return emit


BINARY_F32X2_IMPLS: dict[str, VecImpl] = {
    name: VecImpl(
        vec_len=2,
        applies=_f32x2_applies(name),
        emit=_emit_binary_f32x2_for(name),
    )
    for name in ("add", "sub", "mul")
}
