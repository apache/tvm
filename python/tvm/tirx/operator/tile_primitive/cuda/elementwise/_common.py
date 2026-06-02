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

"""Shared layout / vec-selection / emit helpers for ``reg.py`` and ``smem.py``.

Borrows directly from ``cuda/copy/reg.py`` (induced partition) and
``cuda/copy/_common.py`` (synthesized partition), extended to N operands.

The dispatch split mirrors copy:
    reg.py   — all operands in ``local``  → partition induced by anchor's layout
    smem.py  — all operands in ``shared*`` → partition synthesized from ``sctx.intra``
"""

from __future__ import annotations

import functools
import operator

from tvm.arith.analyzer import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion
from tvm.tirx.layout import Axis, Iter, TileLayout

from ..common import get_indices, get_st_extent

# Re-use copy's primitives (PR-640) — same algorithm, same scope_id machinery.
from ..copy._common import _TID_AXIS_FOR_SCOPE, _extract_tile, _thread_cnt
from ..copy.reg import _all_threads_active, _axis_decl, _compute_perm_r


# -----------------------------------------------------------------------------
# Plan helpers
# -----------------------------------------------------------------------------
def buffer_regions(plan) -> list[BufferRegion]:
    """All BufferRegion args (dst + buffer-region srcs), in plan order."""
    out: list[BufferRegion] = [plan.dst]
    for s in plan.srcs:
        if s.buf_region is not None:
            out.append(s.buf_region)
    return out


def compute_dtype_of(plan) -> str:
    """Widest dtype in bits across dst + buffer/scalar srcs (dst breaks ties)."""
    candidates = [plan.dst.buffer.dtype]
    for s in plan.srcs:
        if s.buf_region is not None:
            candidates.append(s.buf_region.buffer.dtype)
        elif s.scalar is not None:
            candidates.append(s.scalar.dtype)
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


# -----------------------------------------------------------------------------
# Anchor selection (reg.py)
# -----------------------------------------------------------------------------
def pick_anchor(plan) -> BufferRegion:
    """Anchor is always ``plan.dst`` — every operand must have a layout
    (enforced by predicate); dst's layout drives iteration. No choice to make.
    """
    return plan.dst


# -----------------------------------------------------------------------------
# Broadcast support (NumPy-style right-aligned, anchor = result shape)
# -----------------------------------------------------------------------------
def _tensor_shape_of(region) -> tuple[int, ...]:
    """Per-dim region extent (post-slice tensor shape, NOT layout shape).

    Accepts either ``[(start, end), ...]`` pairs (as built locally from a
    ``BufferRegion``) or the ``BufferRegion.region`` sequence of ``Range``
    objects directly. ``Range.extent`` is already simplified by the
    front-end, so we avoid computing ``end - start`` on raw PrimExpr (which
    yields an un-simplified ``Sub`` and breaks ``int(...)``).
    """
    out = []
    a = Analyzer()
    for r in region:
        if hasattr(r, "extent"):
            ext = r.extent
        else:
            start, end = r
            ext = a.simplify(end - start)
        out.append(int(ext))
    return tuple(out)


def shape_broadcast_compat(op_shape, anchor_shape) -> tuple[bool, str | None]:
    """NumPy-style: right-align op against anchor; per-dim extent must equal
    anchor's or be 1. anchor is the result shape; op broadcasts TO anchor.
    """
    pad = len(anchor_shape) - len(op_shape)
    if pad < 0:
        return False, f"op rank {len(op_shape)} > anchor rank {len(anchor_shape)}"
    for d in range(len(op_shape)):
        e_op = int(op_shape[d])
        e_a = int(anchor_shape[pad + d])
        if e_op != e_a and e_op != 1:
            return False, f"dim {d}: op extent {e_op} vs anchor {e_a} (need equal or 1)"
    return True, None


def _broadcast_lift(op_layout, op_tensor_shape, anchor_tensor_shape):
    """Lift ``op_layout`` to ``anchor_tensor_shape`` by inserting stride-0
    iters for padded leading dims and replacing extent-1 buckets with a
    single stride-0 iter of anchor's extent. Offset and replica list are
    preserved untouched.

    The lift preserves the physical-address function: new iters have
    stride 0 so they contribute ``coord * 0 = 0`` to the address
    regardless of which virtual index is supplied, and dropped extent-1
    iters contributed ``0 * stride = 0`` already.
    """
    pad = len(anchor_tensor_shape) - len(op_tensor_shape)
    assert pad >= 0, "shape_broadcast_compat should have rejected this"

    try:
        grouped, seps = op_layout.group(list(op_tensor_shape))
    except Exception as e:  # pylint: disable=broad-except
        raise ValueError(
            f"op layout {op_layout} not groupable by tensor shape {op_tensor_shape}: {e}"
        ) from e

    new_shard: list = []
    # (1) Padded leading dims — one stride-0 iter each.
    for d in range(pad):
        new_shard.append(Iter(int(anchor_tensor_shape[d]), 0, Axis.get("m")))
    # (2) Aligned dims.
    for d_op in range(len(op_tensor_shape)):
        e_op = int(op_tensor_shape[d_op])
        e_a = int(anchor_tensor_shape[pad + d_op])
        bucket = list(grouped.shard[seps[d_op] : seps[d_op + 1]])
        if e_op == e_a:
            new_shard.extend(bucket)
        elif e_op == 1:
            new_shard.append(Iter(e_a, 0, Axis.get("m")))
        else:
            raise ValueError(
                f"dim {d_op}: op extent {e_op} vs anchor {e_a}"
                " (shape_broadcast_compat should have rejected)"
            )

    return TileLayout.from_iters(new_shard, grouped.replica, grouped.offset)


# -----------------------------------------------------------------------------
# Shared preprocess: slice each operand by its region, broadcast-lift to
# anchor's tensor shape. Output: every operand has a layout whose logical
# shape equals ``anchor_tensor_shape``. Reg / smem diverge from here.
# -----------------------------------------------------------------------------
def preprocess_operand(op_br, anchor_tshape):
    """Slice ``op_br``'s buffer layout by region (region offset absorbed into
    layout.offset), then broadcast-lift to ``anchor_tshape`` if shapes differ.

    Raises ``ValueError`` if the lift is not broadcast-compatible (caller
    should have verified via ``shape_broadcast_compat`` in the predicate).
    """
    op_layout = op_br.buffer.layout
    op_shape = op_br.buffer.shape
    op_region = [(r.min, r.min + r.extent) for r in op_br.region]
    sliced = op_layout.slice(list(op_shape), op_region).canonicalize()
    sliced = _extract_tile(sliced, op_region)
    op_tshape = _tensor_shape_of(op_br.region)
    if op_tshape != tuple(anchor_tshape):
        sliced = _broadcast_lift(sliced, op_tshape, anchor_tshape)
    return sliced


def preprocess_operands(plan):
    """Shared entry for reg.py and smem.py: returns
    ``(anchor_tensor_shape, {op_br: sliced_lifted_layout})``.

    Every output layout has logical shape == ``anchor_tensor_shape``.
    Broadcast iters carry stride 0 with the default mem axis. Reg's induced
    partition (`align_operands_to_anchor`) and smem's synthesized partition
    both build on this output.
    """
    anchor_tshape = _tensor_shape_of(plan.dst.region)
    out: dict = {}
    for br in buffer_regions(plan):
        out[br] = preprocess_operand(br, anchor_tshape)
    return anchor_tshape, out


# -----------------------------------------------------------------------------
# Multi-operand layout alignment for reg.py (induced)
# -----------------------------------------------------------------------------
def _align_layouts_no_post_canon(r_layout, r_shape, r_region, s_layout, s_shape, s_region):
    """Variant of copy ``reg.py:align_layouts_raw`` that omits the final
    ``canonicalize()`` on ``r_p``.

    Copy's version returns ``r_p = r.permute_dims(perm).canonicalize()`` —
    that post-permute canonicalize can fuse adjacent iters (e.g. wgmma
    layout's 5 iters collapse to 2), but ``s_seps`` is built from
    ``perm`` of length ``len(r.shard) pre-canon``. The two lengths then
    disagree and ``s_p.shard[s_seps[k]:s_seps[k+1]]`` indexes into the
    wrong sub-range.

    Dropping the final canonicalize keeps ``r_p.shard`` and ``s_seps`` in
    1-to-1 correspondence. Copy's tests don't hit this because R is
    typically 1D and doesn't fuse further after permute.
    """
    r = r_layout.slice(list(r_shape), r_region).canonicalize()
    s = s_layout.slice(list(s_shape), s_region).canonicalize()
    s = _extract_tile(s, s_region)
    # Broadcast lift: when op's post-slice tensor shape != anchor's, expand
    # s via stride-0 iters so group() below can partition along anchor's
    # iter structure. Legality must be enforced upstream by the predicate.
    r_tshape = _tensor_shape_of(r_region)
    s_tshape = _tensor_shape_of(s_region)
    if s_tshape != r_tshape:
        s = _broadcast_lift(s, s_tshape, r_tshape)
    perm = _compute_perm_r(r)
    r_shape_for_group = [int(it.extent) for it in r.shard]
    s_grp, seps = s.group(r_shape_for_group)
    s_p = s_grp.permute_by_groups(list(seps), perm)
    r_p = r.permute_dims(perm)  # NO post-canonicalize
    sizes = [seps[i + 1] - seps[i] for i in range(len(seps) - 1)]
    s_seps = [0]
    for p in perm:
        s_seps.append(s_seps[-1] + sizes[p])
    return r_p, s_p, s_seps


def align_operands_to_anchor(anchor_br, layout_others_br):
    """Align every layout-bearing non-anchor operand to ``anchor_br``.

    Returns ``(anchor_p, per_op_aligned)`` where ``per_op_aligned[op_br] =
    (op_p, op_seps)``. Trivial-layout operands are NOT included here —
    caller indexes them directly via their region. Scalar srcs likewise
    live outside this map.

    Caller must enter ``with sctx.target:`` so ``canonicalize()`` runs the
    scope-aware fusers (e.g. laneid+wid_in_wg → tid_in_wg).

    Uses ``_align_layouts_no_post_canon`` (not copy's ``align_layouts_raw``
    directly) so ``anchor_p.shard`` length matches ``op_seps`` groupings.
    """
    anchor_layout = anchor_br.buffer.layout
    anchor_shape = anchor_br.buffer.shape
    anchor_region = [(r.min, r.min + r.extent) for r in anchor_br.region]

    per_op_aligned: dict = {}
    anchor_p = None
    if not layout_others_br:
        # Just slice + permute anchor alone (no post-canon — keep iters
        # 1-to-1 with how they'd appear with srcs).
        r = anchor_layout.slice(list(anchor_shape), anchor_region).canonicalize()
        perm = _compute_perm_r(r)
        anchor_p = r.permute_dims(perm)
        return anchor_p, per_op_aligned

    for op_br in layout_others_br:
        op_layout = op_br.buffer.layout
        op_shape = op_br.buffer.shape
        op_region = [(r.min, r.min + r.extent) for r in op_br.region]
        r_p, op_p, op_seps = _align_layouts_no_post_canon(
            anchor_layout,
            anchor_shape,
            anchor_region,
            op_layout,
            op_shape,
            op_region,
        )
        if anchor_p is None:
            anchor_p = r_p
        per_op_aligned[op_br] = (op_p, op_seps)
    return anchor_p, per_op_aligned


# -----------------------------------------------------------------------------
# vec_chunk selection
# -----------------------------------------------------------------------------
def pick_vec_chunk(spec, op_call, sctx, plan, max_layout_vec_len: int):
    """Pick widest ``(vec_chunk, vec_impl)`` such that:
      - ``vec_impl.vec_len`` divides ``max_layout_vec_len`` AND ``vec_impl.applies(...)``
      - Or no vec_impl matches → scalar fallback ``(max_layout_vec_len, None)``

    ``spec.vec_impls`` is assumed pre-sorted widest-first.
    """
    if max_layout_vec_len <= 0:
        return 1, None
    for impl in getattr(spec, "vec_impls", []):
        if impl.vec_len > max_layout_vec_len:
            continue
        if max_layout_vec_len % impl.vec_len != 0:
            continue
        ok, _ = impl.applies(op_call, sctx, plan)
        if ok:
            return impl.vec_len, impl
    return max_layout_vec_len, None


# -----------------------------------------------------------------------------
# Emit-time helpers
# -----------------------------------------------------------------------------
def _broadcast_indices(dst_indices, dst_start, dst_extent, op_start, op_ext):
    """NumPy-style right-aligned broadcast: derive op's per-dim indices from
    dst's. For matching extents copies dst's coord (rebased); for op_ext[d]
    == 1 returns the constant start (the only valid index for that dim).
    """
    pad = len(dst_extent) - len(op_ext)
    return [
        (dst_indices[i + pad] - dst_start[i + pad]) + op_start[i]
        if int(op_ext[i]) != 1
        else op_start[i]
        for i in range(len(op_ext))
    ]


def fetch_src_value(src, fused, dst_indices, dst_start, dst_extent):
    """Per-element load Expr for one src. Handles buffer / scalar / broadcast srcs."""
    if src.is_scalar:
        return src.scalar
    region = src.buf_region
    src_st, src_ext = get_st_extent(region)
    if src.index_fn is not None:
        idx = src.index_fn(dst_indices, dst_start, dst_extent, src_st, src_ext)
    elif tuple(int(e) for e in src_ext) != tuple(int(e) for e in dst_extent):
        # Broadcast — derive src indices from dst's via right-aligned compat.
        idx = _broadcast_indices(dst_indices, dst_start, dst_extent, src_st, src_ext)
    else:
        idx = get_indices(fused, src_st, src_ext)
    return region.buffer[tuple(idx)]


def emit_scope_sync(scope_kind: str):
    """Returns an ``@Tx.inline`` sync helper matched to the exec scope."""

    @Tx.inline
    def sync():
        if scope_kind == "cta":
            Tx.cuda.cta_sync()
        elif scope_kind == "warpgroup":
            Tx.cuda.warpgroup_sync(8)
        elif scope_kind == "warp":
            Tx.cuda.warp_sync()

    return sync


__all__ = [
    "_TID_AXIS_FOR_SCOPE",
    "_all_threads_active",
    "_axis_decl",
    "_broadcast_indices",
    "_tensor_shape_of",
    "_thread_cnt",
    "align_operands_to_anchor",
    "buffer_regions",
    "compute_dtype_of",
    "emit_scope_sync",
    "fetch_src_value",
    "n_elements",
    "pick_anchor",
    "pick_vec_chunk",
    "preprocess_operand",
    "preprocess_operands",
    "shape_broadcast_compat",
]
