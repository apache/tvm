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

"""Elementwise dispatch when all operands live in ``local`` (registers).

Mirrors ``cuda/copy/reg.py``: the partition is *induced* by the layout that
carries thread-axis info (the "anchor" operand). The region slice is absorbed
into the sliced layout up front via ``align_operands_to_anchor`` — emit
operates on a flat 1D per-thread view and indexes it with a scalar offset, so
codegen never sees multi-dim ``get_indices`` inside ``Tx.vectorized``.

Two paths inside emit:
  * induced (anchor exists)   — atom-based, exactly mirrors copy reg.py
  * trivial (no anchor)       — flat full region, every thread runs the full
                                loop on its private storage
"""

from __future__ import annotations

import functools
import operator

from tvm.arith import Analyzer
from tvm.script import tirx as Tx
from tvm.tirx import PrimFunc, TilePrimitiveCall
from tvm.tirx.layout import TileLayout
from tvm.tirx.operator.tile_primitive import DispatchContext
from tvm.tirx.operator.tile_primitive.dispatcher import fail

from ..common import get_st_extent
from ..copy._common import _carve_tail, _verify_s_tail_contig
from ..layout_utils import get_sublayout_from_region, layout_signature
from ._common import (
    _all_threads_active,
    _tensor_shape_of,
    align_operands_to_anchor,
    buffer_regions,
    compute_dtype_of,
    pick_anchor,
    shape_broadcast_compat,
)


# -----------------------------------------------------------------------------
# Predicate
# -----------------------------------------------------------------------------
def _validate_anchor_layout(anchor_br) -> tuple[bool, str | None]:
    layout = anchor_br.buffer.layout
    if layout.is_swizzle():
        return False, "anchor layout is swizzle"
    if not isinstance(layout, TileLayout):
        return False, f"anchor layout is {type(layout).__name__}, not TileLayout"
    return True, None


def _check_layout_operands_agree(plan) -> tuple[bool, str | None]:
    """Replica sigs must match across non-trivial-layout operands.

    ``align_operands_to_anchor`` normalizes thread + local parts via
    permute/group, but the replica part isn't touched by alignment — if
    operands disagree there, alignment can't fix it and emit will be wrong.
    Thread / local mismatches that alignment can't resolve will raise
    cleanly at align time, so we don't pre-check them.
    """
    # All operands have a layout (predicate already enforced this); just
    # iterate them all.
    layout_brs = list(buffer_regions(plan))
    if len(layout_brs) < 2:
        return True, None
    analyzer = Analyzer()
    replica_sigs = []
    for br in layout_brs:
        st, ext = get_st_extent(br)
        sliced = get_sublayout_from_region(br.buffer.layout, br.buffer.shape, st, ext)
        canon = sliced.canonicalize() if hasattr(sliced, "canonicalize") else sliced
        sig = layout_signature(canon)
        if sig is None:
            return False, "layout has no signature (not a TileLayout?)"
        # layout_signature returns (thread_sig, local_sig, replica_sig)
        replica_sigs.append(sig[2])
    for s in replica_sigs[1:]:
        # Compare replica entries (axis_key, extent, stride) element-wise.
        if len(s) != len(replica_sigs[0]):
            return False, "replica sig mismatch (different number of replica iters)"
        for (k_a, e_a, st_a), (k_b, e_b, st_b) in zip(replica_sigs[0], s):
            if k_a != k_b:
                return False, "replica sig mismatch (axis key)"
            if not analyzer.can_prove_equal(e_a, e_b):
                return False, "replica sig mismatch (extent)"
            if not analyzer.can_prove_equal(st_a, st_b):
                return False, "replica sig mismatch (stride)"
    return True, None


def is_reg_ewise(spec):
    """Predicate factory: dispatch accepted iff all operands in ``local`` scope."""

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
            if br.buffer.scope() != "local":
                return False, f"operand scope {br.buffer.scope()} != local"
            if br.buffer.layout is None:
                return False, f"operand {br} has no layout"
        if spec.check_extras is not None:
            ok2, reason2 = spec.check_extras(plan.extras, compute_dtype_of(plan))
            if not ok2:
                return False, reason2
        anchor = pick_anchor(plan)
        ok3, reason3 = _validate_anchor_layout(anchor)
        if not ok3:
            return False, reason3
        # Shape compat (NumPy-style broadcast): anchor's tensor shape is the
        # result shape; every operand must broadcast TO anchor.
        anchor_tshape = _tensor_shape_of(anchor.region)
        for br in buffer_regions(plan):
            if br is anchor:
                continue
            op_tshape = _tensor_shape_of(br.region)
            ok_b, reason_b = shape_broadcast_compat(op_tshape, anchor_tshape)
            if not ok_b:
                return False, f"shape incompat: {reason_b}"
        ok4, reason4 = _check_layout_operands_agree(plan)
        if not ok4:
            return False, reason4
        return True, None

    return check


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def _prod(it) -> int:
    return functools.reduce(operator.mul, it, 1)


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def emit_reg(op_call: TilePrimitiveCall, spec, sctx: DispatchContext) -> PrimFunc:
    plan, msg = spec.parse(op_call)
    if msg is not None or plan is None:
        fail(msg or "parse failed")
    return _emit_induced(plan, spec, sctx, op_call, pick_anchor(plan))


# -----------------------------------------------------------------------------
# Induced path — anchor with non-trivial layout drives partition
# -----------------------------------------------------------------------------
def _strip_thread(layout):
    """Return a new TileLayout with thread iters removed from shard."""
    mem_iters = [it for it in layout.shard if not it.axis.is_thread()]
    return TileLayout.from_iters(mem_iters, list(layout.replica), dict(layout.offset))


def _pick_vec_and_carve(spec, op_call, sctx, plan, per_op_mem_layouts):
    """Pick ``(vec_len, vec_impl, carved_layouts)``.

    Enumerate ``spec.vec_impls`` widest-first. For each candidate ``vec_len``:
      1. Try ``_carve_tail`` on each operand's mem-only layout (may split a
         boundary iter so the tail product equals ``vec_len``).
      2. Verify the carved tail is physically contiguous (stride-1 chain whose
         product equals ``vec_len``).
      3. Call ``impl.applies(op_call, sctx, plan)``.
    First candidate that passes all three on EVERY operand wins. Otherwise
    fall back to scalar: ``vec_len=1``, ``vec_impl=None``, original (uncarved)
    layouts.
    """
    impls = sorted(getattr(spec, "vec_impls", []), key=lambda i: -i.vec_len)
    for impl in impls:
        cand = impl.vec_len
        carved_try = {}
        all_ok = True
        for op_br, layout in per_op_mem_layouts.items():
            new_iters = _carve_tail(list(layout.shard), cand)
            if new_iters is None:
                all_ok = False
                break
            new_layout = TileLayout.from_iters(new_iters, list(layout.replica), dict(layout.offset))
            if not _verify_s_tail_contig(new_layout, cand):
                all_ok = False
                break
            carved_try[op_br] = new_layout
        if not all_ok:
            continue
        ok, _ = impl.applies(op_call, sctx, plan)
        if not ok:
            continue
        return cand, impl, carved_try
    # Scalar fallback — use uncarved mem layouts as-is.
    return 1, None, dict(per_op_mem_layouts)


def _emit_induced(plan, spec, sctx, op_call, anchor_br) -> PrimFunc:
    # Every buffer-region operand has a layout (enforced by predicate);
    # trivial / identity layouts are fine — the algorithm is robust to
    # layouts with no thread axes (strip is no-op, placeholders empty).
    layout_others = [br for br in buffer_regions(plan) if br is not anchor_br]

    # Step 1: slice + permute (region offset absorbed into op_p.offset; per-iter
    # strides reflect post-slice physical addressing). No (st, ext) leaks out.
    with sctx.target:
        anchor_p, per_op_aligned = align_operands_to_anchor(anchor_br, layout_others)

    # Step 2: post-align thread-equality check. ``align`` is supposed to
    # normalize the thread part; we verify it did. (Replica was pre-checked
    # in the predicate.)
    def _thread_iters(layout):
        c = layout.canonicalize()
        return [(it.axis, int(it.extent), int(it.stride)) for it in c.shard if it.axis.is_thread()]

    anchor_thread = _thread_iters(anchor_p)
    for op_br, (op_p, _) in per_op_aligned.items():
        if _thread_iters(op_p) != anchor_thread:
            fail("thread part mismatch between anchor and operand after alignment")

    # Step 3: drop thread iters; from here on operands have mem-only layouts.
    per_op_mem = {anchor_br: _strip_thread(anchor_p)}
    for op_br, (op_p, _) in per_op_aligned.items():
        per_op_mem[op_br] = _strip_thread(op_p)

    # Step 4: enumerate spec.vec_impls widest-first; try carve tail for each
    # operand. First candidate that all operands can carve + impl.applies wins.
    # Otherwise scalar fallback (vec=1, no inner loop).
    vec_len, vec_impl, per_op_carved = _pick_vec_and_carve(spec, op_call, sctx, plan, per_op_mem)

    # Step 5: totals + emit. per_thread_total = ∏ extents of (carved) mem
    # layout. All operands have the same per_thread_total (alignment invariant).
    per_thread_total = _prod(int(it.extent) for it in per_op_carved[anchor_br].shard)
    outer_total = per_thread_total // vec_len if vec_len > 0 else per_thread_total

    if vec_impl is not None:
        result = _emit_induced_packed(
            plan,
            vec_impl,
            vec_len,
            outer_total,
            per_thread_total,
            per_op_carved,
            anchor_br,
        )
    else:
        result = _emit_induced_scalar(
            plan,
            spec,
            outer_total,
            per_thread_total,
            per_op_carved,
            anchor_br,
        )

    return result


def _make_views_meta(per_op_carved, per_thread_total):
    """Build the per-operand 1D buffer view dict.

    Each view aliases the operand's physical storage as a 1D shape of
    ``per_thread_total`` elements, with layout = the operand's carved mem-only
    TileLayout. Scalar indexing into the view goes through this layout's
    iter strides at codegen time.
    """
    return {
        op_br: Tx.decl_buffer(
            (per_thread_total,),
            op_br.buffer.dtype,
            op_br.buffer.data,
            scope="local",
            layout=per_op_carved[op_br],
        )
        for op_br in per_op_carved
    }


# -----------------------------------------------------------------------------
# Emit — packed (one PTX/CUDA call per outer chunk; no Tx.vectorized inside)
# -----------------------------------------------------------------------------
def _emit_induced_packed(
    plan, vec_impl, vec_len, outer_total, per_thread_total, per_op_carved, anchor_br
) -> PrimFunc:
    extras = plan.extras
    srcs = plan.srcs
    dst_br = plan.dst

    @Tx.prim_func(check_well_formed=False)
    def impl():
        views = Tx.meta_var(_make_views_meta(per_op_carved, per_thread_total))
        # Serial loop (not Tx.unroll): Tx.unroll materializes each per-iter
        # ``dst_lane_indices`` / ``src_args`` buffer as a fresh int[1]
        # declaration, multiplying by outer_total. ptxas unrolls the
        # static-bound loop without that scratch explosion.
        for f in range(outer_total):
            # Pass logical 1D coord; each buffer's own layout maps it to
            # physical at access time (handles wgmma, broadcast, etc.).
            dst_lane_indices = [[f * vec_len + k] for k in range(vec_len)]
            src_args = Tx.meta_var(
                [
                    src.scalar
                    if src.is_scalar
                    else (
                        views[src.buf_region],
                        [[f * vec_len + k] for k in range(vec_len)],
                    )
                    for src in srcs
                ]
            )
            Tx.evaluate(vec_impl.emit(views[dst_br], dst_lane_indices, src_args, extras))

    return impl


# -----------------------------------------------------------------------------
# Emit — scalar fallback (vec_len = 1; one element per outer iter; no
# Tx.vectorized inside, so no codegen vec-packing of multi-dim indices).
# -----------------------------------------------------------------------------
def _emit_induced_scalar(
    plan, spec, outer_total, per_thread_total, per_op_carved, anchor_br
) -> PrimFunc:
    extras = plan.extras
    srcs = plan.srcs
    dst_br = plan.dst
    dst_dtype = dst_br.buffer.dtype
    compute = spec.compute_scalar

    @Tx.prim_func(check_well_formed=False)
    def impl():
        views = Tx.meta_var(_make_views_meta(per_op_carved, per_thread_total))
        # Serial loop (not Tx.unroll) — see _emit_induced_packed for why.
        for f in range(outer_total):
            # Logical 1D coord = f (vec_len = 1 in scalar path); each
            # buffer's layout maps to physical at access time.
            src_vals = Tx.meta_var(
                [src.scalar if src.is_scalar else views[src.buf_region][f] for src in srcs]
            )
            views[dst_br][f] = Tx.cast(compute(src_vals, extras, dst_dtype), dst_dtype)

    return impl
