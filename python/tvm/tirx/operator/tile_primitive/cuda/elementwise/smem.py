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

"""Elementwise dispatch when all operands live in ``shared*``.

Mirrors ``cuda/copy/gmem_smem.py``: no operand carries a per-thread partition,
so partition is *synthesized* from ``sctx.intra`` (``thread_cnt = ∏ intra``).
Each thread takes ``ceildiv(total, vec_chunk * thread_cnt)`` strided chunks.

The shared buffers are indexed multi-dim via ``get_indices(fused, dst_st,
dst_ext)`` and the buffer's own layout resolves to physical addresses at
codegen time. Packed-vec emit requires the innermost dim to have stride 1
(non-swizzle slice) so lanes are physically contiguous; checked in
``_max_layout_vec``.
"""

from __future__ import annotations

from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tirx import PrimFunc, TilePrimitiveCall
from tvm.tirx.operator.tile_primitive import DispatchContext
from tvm.tirx.operator.tile_primitive.dispatcher import fail

from ..common import get_indices, get_st_extent, get_thread_cnt
from ._common import (
    _TID_AXIS_FOR_SCOPE,
    _all_threads_active,
    _axis_decl,
    _broadcast_indices,
    _tensor_shape_of,
    buffer_regions,
    compute_dtype_of,
    emit_scope_sync,
    fetch_src_value,
    n_elements,
    pick_vec_chunk,
    shape_broadcast_compat,
)


# -----------------------------------------------------------------------------
# Predicate
# -----------------------------------------------------------------------------
def is_smem_ewise(spec):
    """Predicate factory: dispatch accepted iff all operands in ``shared*``."""

    def check(op_call: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
        if not sctx.is_cuda:
            return False, "non-cuda target"
        if sctx.scope_kind not in ("thread", "warp", "warpgroup", "cta"):
            return False, f"unsupported scope {sctx.scope_kind}"
        ok, reason = _all_threads_active(sctx)
        if not ok:
            return False, reason
        plan, msg = spec.parse(op_call)
        if msg is not None or plan is None:
            return False, msg
        for br in buffer_regions(plan):
            if not br.buffer.scope().startswith("shared"):
                return False, f"operand scope {br.buffer.scope()} != shared*"
            if br.buffer.layout is None:
                return False, "shared operand has no layout"
        if spec.check_extras is not None:
            ok2, reason2 = spec.check_extras(plan.extras, compute_dtype_of(plan))
            if not ok2:
                return False, reason2
        # NumPy-style right-aligned broadcast: anchor = plan.dst; every src
        # must be shape-compatible with anchor (extent matches or is 1).
        anchor_tshape = _tensor_shape_of(plan.dst.region)
        for s in plan.srcs:
            if s.buf_region is None:
                continue
            src_tshape = _tensor_shape_of(s.buf_region.region)
            ok_b, reason_b = shape_broadcast_compat(src_tshape, anchor_tshape)
            if not ok_b:
                return False, f"shape incompat: {reason_b}"
        return True, None

    return check


# -----------------------------------------------------------------------------
# vec_chunk selection
# -----------------------------------------------------------------------------
def _max_layout_vec(plan, total: int, thread_cnt: int) -> int:
    """Widest vec_chunk dividing all operands' innermost extents AND
    ``total / thread_cnt``, within dtype-bit candidates ``{128,64,32,16,8}``."""
    max_bits = DataType(plan.dst.buffer.dtype).bits
    for s in plan.srcs:
        if s.buf_region is not None:
            max_bits = max(max_bits, DataType(s.buf_region.buffer.dtype).bits)
    per_thread = total // thread_cnt if thread_cnt > 0 else total
    if total % thread_cnt != 0:
        return 1

    inners = [int(plan.dst.region[-1].extent)]
    for s in plan.srcs:
        if s.buf_region is None or s.index_fn is not None:
            continue
        inners.append(int(s.buf_region.region[-1].extent))

    for cand_bits in (128, 64, 32, 16, 8):
        n = cand_bits // max_bits
        if n <= 0:
            continue
        if per_thread % n != 0:
            continue
        if all(i % n == 0 for i in inners):
            return n
    return 1


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def emit_smem(op_call: TilePrimitiveCall, spec, sctx: DispatchContext) -> PrimFunc:
    plan, msg = spec.parse(op_call)
    if msg is not None or plan is None:
        fail(msg or "parse failed")

    # Use cuda/common.py:get_thread_cnt rather than copy/_common.py:_thread_cnt
    # — the latter computes ``∏ sctx.intra`` which silently returns 0 for
    # sub-warp counts at cta scope (warpid extent rounds down to 0). The
    # former reads launch_params["threadIdx.x"].dom.extent and is correct
    # for all scopes.
    thread_cnt = get_thread_cnt(sctx)
    if thread_cnt is None:
        fail(f"unsupported scope {sctx.scope_kind} for smem emit")
    thread_cnt = int(thread_cnt)
    if thread_cnt <= 0:
        fail(f"non-positive thread_cnt {thread_cnt}")
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params, (
        "smem emit currently assumes 1D threadIdx"
    )

    total = n_elements(plan.dst)
    vec_max = _max_layout_vec(plan, total, thread_cnt)
    vec_chunk, vec_impl = pick_vec_chunk(spec, op_call, sctx, plan, vec_max)

    if vec_impl is not None:
        return _emit_packed(plan, vec_impl, vec_chunk, total, thread_cnt, sctx)
    return _emit_scalar(plan, spec, vec_chunk, total, thread_cnt, sctx)


def _tid_expr(sctx: DispatchContext):
    """Per-scope tid expr. ``thread`` scope returns 0; collective scopes use
    ``_axis_decl`` (Tx.lane_id / Tx.thread_id_in_wg / threadIdx.x)."""
    if sctx.scope_kind == "thread":
        return 0
    axis_name = _TID_AXIS_FOR_SCOPE[sctx.scope_kind]
    return _axis_decl(axis_name, sctx)


# -----------------------------------------------------------------------------
# Per-lane src index helper (handles broadcast via right-aligned compat)
# -----------------------------------------------------------------------------
def _src_lane_indices(src_br, dst_lane_indices, dst_st, dst_ext, vec_chunk, fused0):
    """Return the per-lane multi-dim index list for ``src_br``.

    If src's region shape matches dst's, fall through to ``get_indices``
    (same as the legacy non-broadcast path). Otherwise derive each lane's
    src index from the corresponding dst lane index via right-aligned
    broadcast compat.
    """
    src_st, src_ext = get_st_extent(src_br)
    if tuple(int(e) for e in src_ext) == tuple(int(e) for e in dst_ext):
        return [get_indices(fused0 + k, src_st, src_ext) for k in range(vec_chunk)]
    return [
        _broadcast_indices(dst_lane_indices[k], dst_st, dst_ext, src_st, src_ext)
        for k in range(vec_chunk)
    ]


# -----------------------------------------------------------------------------
# Emit — packed
# -----------------------------------------------------------------------------
def _emit_packed(plan, vec_impl, vec_chunk, total, thread_cnt, sctx) -> PrimFunc:
    extras = plan.extras
    srcs = plan.srcs
    dst_buf = plan.dst.buffer
    dst_st, dst_ext = get_st_extent(plan.dst)
    sync = emit_scope_sync(sctx.scope_kind)
    n_outer = (total + vec_chunk * thread_cnt - 1) // (vec_chunk * thread_cnt)

    @Tx.prim_func(check_well_formed=False)
    def impl():
        tid = _tid_expr(sctx)
        for s in Tx.serial(0, n_outer):
            # First lane's fused index for this thread, this chunk.
            fused0 = Tx.meta_var(s * vec_chunk * thread_cnt + tid * vec_chunk)
            # Predicate the call (skip the trailing partial chunk).
            if fused0 + vec_chunk <= total:
                dst_lane_indices = Tx.meta_var(
                    [get_indices(fused0 + k, dst_st, dst_ext) for k in range(vec_chunk)]
                )
                src_args = Tx.meta_var(
                    [
                        srcs[i].scalar
                        if srcs[i].is_scalar
                        else (
                            srcs[i].buf_region.buffer,
                            _src_lane_indices(
                                srcs[i].buf_region,
                                dst_lane_indices,
                                dst_st,
                                dst_ext,
                                vec_chunk,
                                fused0,
                            ),
                        )
                        for i in range(len(srcs))
                    ]
                )
                Tx.evaluate(vec_impl.emit(dst_buf, dst_lane_indices, src_args, extras))
        sync()

    return impl


# -----------------------------------------------------------------------------
# Emit — scalar fallback
# -----------------------------------------------------------------------------
def _emit_scalar(plan, spec, vec_chunk, total, thread_cnt, sctx) -> PrimFunc:
    extras = plan.extras
    srcs = plan.srcs
    dst_buf = plan.dst.buffer
    dst_st, dst_ext = get_st_extent(plan.dst)
    dst_dtype = dst_buf.dtype
    compute = spec.compute_scalar
    sync = emit_scope_sync(sctx.scope_kind)
    n_outer = (total + vec_chunk * thread_cnt - 1) // (vec_chunk * thread_cnt)

    @Tx.prim_func(check_well_formed=False)
    def impl():
        tid = _tid_expr(sctx)
        for s in Tx.serial(0, n_outer):
            for vec in Tx.vectorized(vec_chunk):
                fused = Tx.meta_var(s * vec_chunk * thread_cnt + tid * vec_chunk + vec)
                if fused < total:
                    dst_idx = Tx.meta_var(get_indices(fused, dst_st, dst_ext))
                    src_vals = Tx.meta_var(
                        [fetch_src_value(src, fused, dst_idx, dst_st, dst_ext) for src in srcs]
                    )
                    dst_buf[tuple(dst_idx)] = Tx.cast(
                        compute(src_vals, extras, dst_dtype), dst_dtype
                    )
        sync()

    return impl
