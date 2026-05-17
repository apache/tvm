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

"""Op-agnostic helpers shared by the three elementwise schedules."""

from __future__ import annotations

import functools
import operator
from typing import Literal

from tvm.arith.analyzer import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion, TilePrimitiveCall
from tvm.tirx.layout import TileLayout
from tvm.tirx.operator.tile_primitive import DispatchContext

from ..common import get_indices, get_st_extent, get_vec_len, match_scope
from ..layout_utils import get_local_region, get_sublayout_from_region, layout_signature, sig_equal
from .schema import Plan, SrcSpec


# -----------------------------------------------------------------------------
# Plan helpers
# -----------------------------------------------------------------------------
def buffer_regions(plan: Plan) -> list[BufferRegion]:
    """All BufferRegion args (dst + buffer-region srcs), in order."""
    out: list[BufferRegion] = [plan.dst]
    for s in plan.srcs:
        if s.buf_region is not None:
            out.append(s.buf_region)
    return out


def compute_dtype_of(plan: Plan) -> str:
    """Pick the dtype used for ops.compute (max bit-width of dst and bufferred srcs)."""
    candidates = [plan.dst.buffer.dtype]
    for s in plan.srcs:
        if s.buf_region is not None:
            candidates.append(s.buf_region.buffer.dtype)
        elif s.scalar is not None:
            candidates.append(s.scalar.dtype)
    # Pick widest in bits; tiebreak: dst dtype first
    widest = candidates[0]
    widest_bits = DataType(widest).bits
    for d in candidates[1:]:
        b = DataType(d).bits
        if b > widest_bits:
            widest, widest_bits = d, b
    return widest


def n_elements(buf_region: BufferRegion) -> int:
    _, ext = get_st_extent(buf_region)
    return functools.reduce(operator.mul, ext, 1)


def is_full_region(buf_region: BufferRegion | None) -> bool:
    """Region covers the whole buffer (start=0, extent=shape)."""
    if buf_region is None:
        return True
    st, ext = get_st_extent(buf_region)
    a = Analyzer()
    return all(a.can_prove_equal(e, s) for e, s in zip(ext, buf_region.buffer.shape)) and all(
        a.can_prove_equal(s, 0) for s in st
    )


# -----------------------------------------------------------------------------
# Storage scope predicate (works for any arity)
# -----------------------------------------------------------------------------
def match_all_scope(
    op_call: TilePrimitiveCall,
    sctx: DispatchContext,
    expected_scope: list[Literal["global", "shared*", "local"]],
) -> tuple[bool, str | None]:
    """Predicate: dst + every BufferRegion src is in one of expected_scope."""
    from .schema import ALL_OPS  # avoid cycle

    spec = ALL_OPS.get(op_call.op.name.removeprefix("tirx."))
    if spec is None:
        return False, f"unknown op {op_call.op.name}"
    plan, msg = spec.parse(op_call)
    if msg is not None or plan is None:
        return False, msg

    scopes = [plan.dst.buffer.scope()]
    for s in plan.srcs:
        if s.buf_region is not None:
            scopes.append(s.buf_region.buffer.scope())
    ok = any(all(match_scope(sc, want) for sc in scopes) for want in expected_scope)
    if ok:
        return True, None
    return False, f"storage scope mismatch: {scopes}; expected {expected_scope}"


# -----------------------------------------------------------------------------
# Layout/sig checks (used by tile_local and shared validators)
# -----------------------------------------------------------------------------
def slice_and_sig(buf_region: BufferRegion):
    st, ext = get_st_extent(buf_region)
    sliced = get_sublayout_from_region(buf_region.buffer.layout, buf_region.buffer.shape, st, ext)
    canonical = sliced.canonicalize() if hasattr(sliced, "canonicalize") else sliced
    return st, ext, sliced, layout_signature(canonical)


def basic_layout_checks(
    cur: BufferRegion,
    ref: BufferRegion,
    analyzer: Analyzer,
    *,
    disallow_swizzle: bool,
) -> bool:
    cur_buf, ref_buf = cur.buffer, ref.buffer
    cur_region = [r.extent for r in cur.region]
    ref_region = [r.extent for r in ref.region]
    return (
        len(cur_region) == len(ref_region)
        and all(analyzer.can_prove_equal(r, rr) for r, rr in zip(cur_region, ref_region))
        and (cur_buf.layout is not None and ref_buf.layout is not None)
        and isinstance(cur_buf.layout, TileLayout)
        and isinstance(ref_buf.layout, TileLayout)
        and getattr(cur_buf.layout, "shard", None)
        and getattr(ref_buf.layout, "shard", None)
        and not (disallow_swizzle and (cur_buf.layout.is_swizzle() or ref_buf.layout.is_swizzle()))
    )


def sigs_equal(analyzer: Analyzer, *sigs) -> bool:
    """All non-None sigs equal."""
    ref = None
    for s in sigs:
        if s is None:
            continue
        if ref is None:
            ref = s
            continue
        if not sig_equal(analyzer, s, ref):
            return False
    return True


# -----------------------------------------------------------------------------
# vec_len inference (arity-agnostic)
# -----------------------------------------------------------------------------
def infer_vec_len(
    op: TilePrimitiveCall, plan: Plan, thread_cnt: int, *, fallback_to_scalar: bool
) -> int | None:
    """Infer vectorization length common to dst + all buffer-region srcs."""
    explicit = op.config.get("vec_len", None)
    if explicit is not None:
        return explicit

    ele_size = DataType(plan.dst.buffer.dtype).bits
    for s in plan.srcs:
        if s.buf_region is not None:
            ele_size = max(ele_size, DataType(s.buf_region.buffer.dtype).bits)
    candidates = [128 // ele_size, 64 // ele_size, 32 // ele_size, 1]

    vec = None
    for src in plan.srcs:
        if src.buf_region is None:
            continue
        v = get_vec_len(src.buf_region, plan.dst, candidates, thread_cnt)
        if v is None:
            return 1 if fallback_to_scalar else None
        candidates = [vl for vl in candidates if vl <= v]
        vec = v
    if vec is None:
        # No buffer srcs (scalar-only): use dst against itself
        vec = get_vec_len(plan.dst, plan.dst, candidates, thread_cnt)
    if vec is None and fallback_to_scalar:
        return 1
    return vec


# -----------------------------------------------------------------------------
# Scope sync / tid expressions
# -----------------------------------------------------------------------------
def emit_scope_sync(scope_kind: str):
    @Tx.inline
    def sync():
        if scope_kind == "cta":
            Tx.cuda.cta_sync()
        elif scope_kind == "warpgroup":
            Tx.cuda.warpgroup_sync(8)  # TODO: derive from launch config
        elif scope_kind == "warp":
            Tx.cuda.warp_sync()
        # thread: no sync needed

    return sync


def tid_in_scope_expr(sctx: DispatchContext, thread_cnt: int):
    """Per-scope tid expression for fused-tid distribution."""
    tx_var = sctx.launch_params["threadIdx.x"].var
    if sctx.scope_kind == "cta":
        return tx_var
    if sctx.scope_kind in ("warp", "warpgroup"):
        return tx_var % thread_cnt
    if sctx.scope_kind == "thread":
        return 0
    return None


# -----------------------------------------------------------------------------
# Per-element source fetch — uniform for buffer/scalar/broadcast srcs.
# -----------------------------------------------------------------------------
def fetch_src_value(src: SrcSpec, fused, dst_indices, dst_start, dst_extent):
    """Build the per-element value expression for one src."""
    if src.is_scalar:
        return src.scalar
    region = src.buf_region
    src_st, src_ext = get_st_extent(region)
    if src.index_fn is not None:
        idx = src.index_fn(dst_indices, dst_start, dst_extent, src_st, src_ext)
    else:
        idx = get_indices(fused, src_st, src_ext)
    return region.buffer[tuple(idx)]


__all__ = [
    "Plan",
    "SrcSpec",
    "basic_layout_checks",
    "buffer_regions",
    "compute_dtype_of",
    "emit_scope_sync",
    "fetch_src_value",
    "get_local_region",
    "infer_vec_len",
    "is_full_region",
    "match_all_scope",
    "n_elements",
    "sigs_equal",
    "slice_and_sig",
    "tid_in_scope_expr",
]
