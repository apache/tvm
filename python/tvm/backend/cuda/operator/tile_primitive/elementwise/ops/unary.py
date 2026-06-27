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

"""Unary elementwise ops: zero / fill / reciprocal / sqrt / exp / exp2 / silu.

All carry the same ``T.<unary>(dst, src[, bias, scale])`` shape (bias / scale
optional; ``silu`` ignores bias/scale to preserve legacy behavior).
"""

from __future__ import annotations

from typing import Any

from tvm.ir.expr import PrimExpr
from tvm.script import tirx as T
from tvm.tirx import BufferRegion, TilePrimitiveCall
from tvm.tirx.expr import FloatImm

from .._common import scalar_dtype
from . import OpSpec, Plan, SrcSpec


def _parse_unary(op: TilePrimitiveCall) -> tuple[Plan | None, str | None]:
    """T.<unary>(dst, src[, bias, scale]) → Plan."""
    _dst: BufferRegion = op.args[0]
    _src = op.args[1]
    _bias = op.args[2] if len(op.args) > 2 else None
    _scale = op.args[3] if len(op.args) > 2 else None

    srcs: list[SrcSpec] = []
    if isinstance(_src, BufferRegion):
        srcs.append(SrcSpec(buf_region=_src))
    elif isinstance(_src, PrimExpr):
        srcs.append(SrcSpec(scalar=_src))
    else:
        return None, f"unsupported src type {type(_src).__name__}"

    extras: dict[str, Any] = {
        "scale": _scale,
        "bias_const": _bias if isinstance(_bias, FloatImm) else None,
    }
    if isinstance(_bias, BufferRegion):
        srcs.append(SrcSpec(buf_region=_bias))
        extras["has_bias_buf"] = True
    else:
        extras["has_bias_buf"] = False
    return Plan(dst=_dst, srcs=srcs, extras=extras), None


def _check_unary_extras(extras: dict, compute_dtype: str) -> tuple[bool, str | None]:
    scale = extras.get("scale")
    if scale is not None and scalar_dtype(scale) != compute_dtype:
        return False, f"scale dtype {scalar_dtype(scale)} != compute dtype {compute_dtype}"
    bias_const = extras.get("bias_const")
    if bias_const is not None and scalar_dtype(bias_const) != compute_dtype:
        return (
            False,
            f"bias_const dtype {scalar_dtype(bias_const)} != compute dtype {compute_dtype}",
        )
    return True, None


def _with_bias_scale(raw_op):
    """Wrap ``raw_op`` (e.g. ``T.exp``) into a compute that applies bias/scale first."""

    def compute(src_vals, extras, dt):
        x = src_vals[0]
        scale = extras.get("scale")
        if scale is not None:
            x = x * scale
        if extras.get("has_bias_buf"):
            x = x + src_vals[1]
        elif extras.get("bias_const") is not None:
            x = x + extras["bias_const"]
        return raw_op(x)

    return compute


def _compute_zero(src_vals, extras, dt):
    return 0.0


def _compute_fill(src_vals, extras, dt):
    return src_vals[0]


def _compute_reciprocal(src_vals, extras, dt):
    x = src_vals[0]
    return T.FloatImm(x.ty, 1.0) / x


def _compute_silu(src_vals, extras, dt):
    # Legacy: silu doesn't apply bias/scale.
    x = src_vals[0]
    return x / (T.FloatImm(x.ty, 1.0) + T.exp(T.FloatImm(x.ty, 0.0) - x))


UNARY_OPS: dict[str, OpSpec] = {
    "zero": OpSpec("zero", _parse_unary, _compute_zero, _check_unary_extras),
    "fill": OpSpec("fill", _parse_unary, _compute_fill, _check_unary_extras),
    "reciprocal": OpSpec("reciprocal", _parse_unary, _compute_reciprocal, _check_unary_extras),
    "sqrt": OpSpec("sqrt", _parse_unary, _with_bias_scale(T.sqrt), _check_unary_extras),
    "exp": OpSpec("exp", _parse_unary, _with_bias_scale(T.exp), _check_unary_extras),
    "exp2": OpSpec("exp2", _parse_unary, _with_bias_scale(T.exp2), _check_unary_extras),
    "silu": OpSpec("silu", _parse_unary, _compute_silu, _check_unary_extras),
}
