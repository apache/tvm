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

"""Binary elementwise ops: add / sub / mul / fdiv / maximum.

Includes constant-lhs commute logic. Broadcasting (extent=1 dims) is
handled at the layout level in dispatch's ``_broadcast_lift``, not here —
parser just records each src as-is.

``add``/``sub``/``mul`` attach a ``VecImpl`` for sm_100+ packed f32x2;
``fdiv``/``maximum`` have no packed PTX (scalar fallback only — ``max`` lowers
to a single ``FMNMX``/``max.f32``, which is exact, so there is no rounding/ftz
variant to pack).
"""

from __future__ import annotations

import functools
import operator
from typing import Any

from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion, TilePrimitiveCall

from ..vec_emit.binary_f32x2 import BINARY_F32X2_IMPLS
from . import OpSpec, Plan, SrcSpec

_COMMUTATIVE = frozenset({"add", "mul", "maximum"})


def _parse_binary_for(op_name: str):
    """Build a ``parse(op_call) -> (Plan, msg)`` for a specific binary op."""

    def parse(op: TilePrimitiveCall) -> tuple[Plan | None, str | None]:
        _dst: BufferRegion = op.args[0]
        _src1 = op.args[1]
        _src2 = op.args[2]

        s1_scalar = not isinstance(_src1, BufferRegion)
        s2_scalar = not isinstance(_src2, BufferRegion)
        if s1_scalar and s2_scalar:
            return None, "both inputs are constants"

        # Move constant to rhs (commute if allowed; else reject).
        if s1_scalar:
            if op_name not in _COMMUTATIVE:
                return None, f"non-commutative op {op_name} cannot have constant lhs"
            _src1, _src2 = _src2, _src1
            s2_scalar = True

        # If rhs is a smaller buffer (broadcast), swap if commutative so the
        # bigger one is in src1 — keeps src1 == dst convention.
        if not s2_scalar:
            s1_n = functools.reduce(operator.mul, [r.extent for r in _src1.region], 1)
            s2_n = functools.reduce(operator.mul, [r.extent for r in _src2.region], 1)
            if s1_n < s2_n:
                if op_name not in _COMMUTATIVE:
                    return None, f"non-commutative op {op_name} cannot swap to broadcast"
                _src1, _src2 = _src2, _src1

        srcs: list[SrcSpec] = [SrcSpec(buf_region=_src1)]
        if s2_scalar:
            srcs.append(SrcSpec(scalar=_src2))
        else:
            srcs.append(SrcSpec(buf_region=_src2))

        extras: dict[str, Any] = {}
        rm = op.config.get("rounding_mode", None)
        if rm is not None:
            extras["rounding_mode"] = rm
        return Plan(dst=_dst, srcs=srcs, extras=extras), None

    return parse


def _compute_add(src_vals, extras, dt):
    return src_vals[0] + src_vals[1]


def _compute_sub(src_vals, extras, dt):
    return src_vals[0] - src_vals[1]


def _compute_mul(src_vals, extras, dt):
    return src_vals[0] * src_vals[1]


def _compute_fdiv(src_vals, extras, dt):
    return src_vals[0] / src_vals[1]


def _compute_maximum(src_vals, extras, dt):
    return Tx.max(src_vals[0], src_vals[1])


BINARY_OPS: dict[str, OpSpec] = {
    "add": OpSpec(
        "add",
        _parse_binary_for("add"),
        _compute_add,
        vec_impls=[BINARY_F32X2_IMPLS["add"]],
    ),
    "sub": OpSpec(
        "sub",
        _parse_binary_for("sub"),
        _compute_sub,
        vec_impls=[BINARY_F32X2_IMPLS["sub"]],
    ),
    "mul": OpSpec(
        "mul",
        _parse_binary_for("mul"),
        _compute_mul,
        vec_impls=[BINARY_F32X2_IMPLS["mul"]],
    ),
    "fdiv": OpSpec(
        "fdiv",
        _parse_binary_for("fdiv"),
        _compute_fdiv,
    ),
    "maximum": OpSpec(
        "maximum",
        _parse_binary_for("maximum"),
        _compute_maximum,
    ),
}
