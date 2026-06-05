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

"""copy_async dispatch variant: tma (unified algorithm).

One algorithm handles all global↔shared TMA copies, respecting the user's
logical OOB spec through alignment conditions on the reshape. No more
aggressive vs exact family split; ``oob`` only selects the hardware fill
kind (0 = zero, 1 = NaN) in the cuTensorMap.

Pipeline:

L1  Canonicalize smem+gmem layouts; group gmem by buffer shape; split any
    multi-iter gmem group into t separate iters (requires g_st, copy_ext
    divisible by the inner-product u); slice smem by copy region; regroup
    smem by the "copy shape with ext=1 dropped".
L2  For each ext>1 gmem iter (paired with one smem shard sequence), choose
    a contiguous chain prefix of selected smem shards (j from max to 0).
    Cut the gmem axis into segments at each selected position; each segment
    reduces to Case 1 (has selected → box>1 desc dim) or Case 2 (no
    selected → box=1 desc dim). Segment 0 absorbs the G-vs-copy_ext slack
    via a non-full copy_range; alignment requires g_st, G divisible by
    u_{p_0}. Every unselected shard becomes an issue axis.
L3  Stack desc dims across all gmem iters; nest issue axes as an unrolled
    loop; validate hardware constraints (rank≤5, swizzle atom, unit inner
    stride). Shrink j and retry on failure; bail out when j=0 fails.
Emit Single unrolled loop over the flat mixed-radix decomposition; each
    iter computes (smem offset, per-desc-dim tma coord) and emits one
    cp_async_bulk_tensor. Host init emits one cuTensorMapEncodeTiled
    (deduped by cache key).
"""

from dataclasses import dataclass

import tvm
from tvm.arith import Analyzer
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.layout import ComposeLayout, Layout, S, SwizzleLayout, TileLayout
from tvm.tirx.operator.tile_primitive import (
    DispatchContext,
    fail,
    predicate,
    register_dispatch,
)
from tvm.tirx.stmt import TilePrimitiveCall

from ..common import validate_copy_op
from ..exec_scope_utils import single_thread
from ..tma_utils import SwizzleMode, get_swizzle_mode_from_layout, tma_atom_shape

# ==============================================================================
# Data types
# ==============================================================================


@dataclass(frozen=True)
class GmemIter:
    """One gmem logical dim after multi-iter group splitting.

    ``shape`` and ``stride`` come from the canonicalized gmem layout for
    this dim. ``copy_start`` / ``copy_ext`` carve out the user-requested
    sub-range. ``copy_ext == 1`` collapses the iter into a trivial
    coord-only descriptor dim (no smem shards, no issue axes).
    """

    shape: object
    stride: object
    copy_start: object
    copy_ext: object

    @property
    def is_ext1(self) -> bool:
        return Analyzer().can_prove_equal(self.copy_ext, 1)


@dataclass(frozen=True)
class SmemShard:
    """One canonicalized smem shard inside a group (after slice+regroup)."""

    extent: object
    smem_stride: object


@dataclass
class SmemGroup:
    """Smem shards paired with a single ext>1 gmem iter, outer→inner.

    After L1, each ext>1 gmem iter has a matching smem group whose shards'
    extents multiply to the iter's ``copy_ext``.
    """

    shards: list  # list[SmemShard], outer→inner
    bound_gmem_iter_idx: int


@dataclass
class Segment:
    """One reshape segment produced by the chain-prefix cut.

    ``local_shape * local_stride`` is the axis's gmem span; the
    ``local_copy_range`` is where the user-requested slice lives on this
    axis. A segment is "selected" when it ends with a chosen smem shard
    (→ Case 1: box = selected extent); "trailing" otherwise (→ Case 2:
    box = 1).
    """

    local_shape: object
    local_stride: object
    local_copy_start: object  # lo endpoint of local_copy_range
    local_copy_extent: object  # width of local_copy_range
    # ``selected_shard_extent`` is the extent of the selected smem shard at
    # this segment's inner end (only meaningful when ``is_selected``).
    is_selected: bool
    selected_shard_extent: object
    # Unselected shards within this segment become issue axes contributing
    # to this segment's descriptor dim. Each entry is (extent, u_k) where
    # u_k is the shard's gmem-units-per-step value divided by the
    # segment's selected u (so coord_advance = u_k directly); see
    # ``_segment_issue_contribs``.
    unselected_contribs: list  # list[(extent, coord_advance, smem_stride)]


@dataclass(frozen=True)
class DescDim:
    """One cuTensorMap descriptor dim."""

    shape: object
    stride: object  # gmem stride (elements, not bytes)
    box: object
    coord_base: object


@dataclass(frozen=True)
class IssueAxis:
    """One issue axis = one unselected smem shard becoming a loop iter.

    Each iteration advances one desc dim's coord by ``coord_advance`` and
    one smem region by ``smem_stride``. ``dim_idx`` is the index of the
    owning desc dim in the final ``TmaPlan.dims`` list.
    """

    extent: object
    dim_idx: int
    coord_advance: object
    smem_stride: object


@dataclass(frozen=True)
class TmaPlan:
    """Final descriptor + loop plan."""

    swizzle_mode: SwizzleMode
    dims: list  # list[DescDim], in cuTensorMap outer→inner order
    issue_axes: list  # list[IssueAxis], outer→inner nesting order
    tensor_ptr: object
    # Element size used by the cuTensorMap descriptor.  Defaults to the
    # underlying buffer's dtype size; merge can promote this (e.g. uint8 →
    # uint16) when adjacent contiguous dims would exceed boxDim≤256 in the
    # native dtype.  Strides/extents/boxes in ``dims`` are in this unit.
    elem_bytes: int = 1
    elem_dtype: str = "uint8"

    @property
    def rank(self) -> int:
        return len(self.dims)

    @property
    def shape(self) -> list:
        return [d.shape for d in self.dims]

    @property
    def box_dim(self) -> list:
        return [d.box for d in self.dims]

    @property
    def g_strides(self) -> list:
        return [d.stride for d in self.dims]

    def flatten_total_extent(self) -> object:
        total: object = 1
        for axis in self.issue_axes:
            total = total * axis.extent
        return total

    def offsets_and_coords(self, loop_var):
        """Decompose ``loop_var`` into (smem offset, per-dim coord vector).

        Axes are stored outer→inner. The innermost axis has cum=1; each
        outer axis's cum is the product of inner axes' extents.
        """
        total = 1
        cum_per_axis: list = [None] * len(self.issue_axes)
        for idx in range(len(self.issue_axes) - 1, -1, -1):
            cum_per_axis[idx] = total
            total = total * self.issue_axes[idx].extent

        s_offset: object = 0
        coords: list = [d.coord_base for d in self.dims]
        for axis, cum in zip(self.issue_axes, cum_per_axis):
            iter_val = tvm.tirx.floormod(tvm.tirx.floordiv(loop_var, cum), axis.extent)
            s_offset = s_offset + iter_val * axis.smem_stride
            coords[axis.dim_idx] = coords[axis.dim_idx] + iter_val * axis.coord_advance
        return s_offset, coords


# ==============================================================================
# Common helpers
# ==============================================================================


def _to_tile_layout(layout: Layout, shape: list) -> TileLayout:
    """Normalize the shared layout so pointer arithmetic always sees a TileLayout."""

    if isinstance(layout, ComposeLayout):
        return layout.tile_layout
    if isinstance(layout, SwizzleLayout):
        return TileLayout(S[tuple(shape)])
    return layout


def _assert_memory_only(layout: TileLayout, label: str) -> None:
    for shard in layout.shard:
        if not shard.axis.is_memory():
            raise ValueError(
                f"TMA {label} layout must be pure memory; saw non-memory axis "
                f"{shard.axis} in {layout}"
            )


def _normalize_oob_mode(dtype: str, oob_mode):
    """Validate the user-visible ``oob`` contract flag.

    ``None`` / ``"zero"`` → hardware fill kind 0.
    ``"nan"`` → hardware fill kind 1 (floating-point only).
    """
    if oob_mode is None:
        return None
    if oob_mode not in ("zero", "nan"):
        fail(f"Unsupported TMA oob mode: {oob_mode!r}. Expected None, 'zero', or 'nan'.")
    if oob_mode == "nan" and dtype not in ("float16", "float32", "float64", "bfloat16"):
        fail("TMA oob='nan' requires a floating-point dtype")
    return oob_mode


def _oob_fill_kind(oob_mode) -> int:
    if oob_mode is None or oob_mode == "zero":
        return 0
    if oob_mode == "nan":
        return 1
    raise ValueError(f"Unexpected oob mode: {oob_mode}")


def _swizzle_inner_box_fits(dtype: str, swizzle_mode: SwizzleMode, inner_box) -> bool:
    """Hardware check: innermost ``boxDim[0] * elementSize`` fits swizzle atom."""
    if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
        return True
    atom = tma_atom_shape(dtype, swizzle_mode)
    return bool(Analyzer().can_prove(inner_box <= atom[-1]))


def _divides(a, b, analyzer: Analyzer) -> bool:
    """Return True when ``a`` divides ``b`` (``b % a == 0``)."""
    return analyzer.can_prove_equal(tvm.tirx.floormod(b, a), 0)


def _simplify_with_var_ranges(exprs, var_ranges, sctx: DispatchContext):
    """Simplify expressions under dispatch-context and loop-variable ranges."""
    local_analyzer = Analyzer()
    for var, value_range in sctx.var_range_map.items():
        local_analyzer.bind(var, value_range)
    for var, extent in var_ranges:
        if isinstance(var, tvm.tirx.Var):
            local_analyzer.bind(var, tvm.ir.Range.from_min_extent(0, extent))
    return [local_analyzer.simplify(expr) for expr in exprs]


# ==============================================================================
# L1: layout prerequisite analysis
# ==============================================================================


@dataclass
class L1Result:
    """Output of L1: all gmem iters (ext=1 and ext>1), paired smem groups."""

    swizzle_mode: SwizzleMode
    # All gmem iters in positional order (outer→inner across the splitted
    # logical dims). Mix of ext=1 and ext>1.
    gmem_iters: list  # list[GmemIter]
    # One entry per ext>1 gmem iter, in the same order they appear in
    # ``gmem_iters`` (but excluding ext=1 iters).
    smem_groups: list  # list[SmemGroup]


def _canonicalize_gmem(g_buf: Buffer) -> TileLayout:
    layout = g_buf.layout
    if not isinstance(layout, TileLayout):
        # cuTensorMap requires a plain memory layout on gmem side.
        raise ValueError(f"TMA gmem layout must be a TileLayout; got {type(layout).__name__}")
    return layout.canonicalize()


def _canonicalize_smem(s_buf: Buffer) -> TileLayout:
    return _to_tile_layout(s_buf.layout, s_buf.shape).canonicalize()


def _group_gmem_by_buffer_shape(gmem_canon: TileLayout, buffer_shape: list):
    """Group gmem canonicalized layout by the buffer shape. Returns
    ``(grouped, separators)`` or raises on failure."""
    try:
        grouped, seps = gmem_canon.group(list(buffer_shape))
    except Exception as err:
        raise ValueError(f"Cannot group gmem layout by buffer shape: {err}") from err
    return grouped, seps


def _split_multi_iter_group(
    grouped: TileLayout, separators: list, group_idx: int, copy_start, copy_ext, analyzer: Analyzer
):
    """Handle a gmem group containing t ≥ 1 iters.

    Returns a list of ``GmemIter`` for this group (outer→inner within the
    group). For t=1 → one iter (direct passthrough). For t≥2 → requires
    ``copy_start % u == 0`` and ``copy_ext % u == 0`` where
    ``u = prod(x_1, ..., x_{t-1})`` (everything except the outermost iter
    of this group); splits into t iters where the outermost carries the
    partial copy range and the inner t-1 carry full ranges.
    """
    start = separators[group_idx]
    end = separators[group_idx + 1]
    # Drop ext=1 padding iters (canonicalize may have inserted trivial ones).
    raw_shards = [
        sh for sh in grouped.shard[start:end] if not analyzer.can_prove_equal(sh.extent, 1)
    ]
    if not raw_shards:
        # Degenerate extent-1 group (e.g. batch dim with size 1); emit a
        # placeholder iter that's flagged ext=1 by copy_ext==1.
        return [GmemIter(shape=1, stride=0, copy_start=copy_start, copy_ext=copy_ext)]

    # Canonicalize ordering: outer→inner is the same order as in ``grouped``
    # (TileLayout.group gives outer-first shards per group by construction).
    # t = len(raw_shards).
    if len(raw_shards) == 1:
        sh = raw_shards[0]
        return [
            GmemIter(shape=sh.extent, stride=sh.stride, copy_start=copy_start, copy_ext=copy_ext)
        ]

    # Multi-iter group: require alignment.
    u: object = 1
    for sh in raw_shards[1:]:
        u = u * sh.extent

    if not _divides(u, copy_start, analyzer):
        fail(
            f"TMA multi-iter gmem group requires copy_start % {u} == 0; got copy_start={copy_start}"
        )
    if not _divides(u, copy_ext, analyzer):
        fail(f"TMA multi-iter gmem group requires copy_ext % {u} == 0; got copy_ext={copy_ext}")

    outer = raw_shards[0]
    outer_start = analyzer.simplify(tvm.tirx.floordiv(copy_start, u))
    outer_ext = analyzer.simplify(tvm.tirx.floordiv(copy_ext, u))
    iters = [
        GmemIter(
            shape=outer.extent, stride=outer.stride, copy_start=outer_start, copy_ext=outer_ext
        )
    ]
    for sh in raw_shards[1:]:
        iters.append(GmemIter(shape=sh.extent, stride=sh.stride, copy_start=0, copy_ext=sh.extent))
    return iters


def _slice_and_canonicalize_smem(
    smem_canon: TileLayout, buffer_shape: list, s_st: list, s_ext: list
) -> TileLayout:
    region = [(st, st + ext) for st, ext in zip(s_st, s_ext)]
    sliced = smem_canon.slice(list(buffer_shape), region)
    if sliced is None:
        raise ValueError("Cannot slice smem layout for TMA copy")
    return sliced.canonicalize()


def _regroup_smem_by_extgt1_shape(sliced_smem: TileLayout, extgt1_shape: list) -> tuple:
    """Group the sliced smem layout by the ext>1 copy shape. Returns
    ``(grouped, separators)`` or ``None`` on failure."""
    try:
        return sliced_smem.group(list(extgt1_shape))
    except Exception:
        return None


def _build_l1_result(
    s_buf: Buffer, g_buf: Buffer, g_st: list, g_ext: list, s_st: list, s_ext: list
) -> L1Result:
    """Run the L1 pipeline. Raises ``ValueError`` or ``DispatchFail`` on
    prerequisite violations; the caller treats these as bail-outs."""

    analyzer = Analyzer()

    swizzle_mode = get_swizzle_mode_from_layout(s_buf.layout)
    if swizzle_mode is None:
        raise ValueError(f"Cannot determine swizzle mode from layout: {s_buf.layout}")

    smem_canon = _canonicalize_smem(s_buf)
    _assert_memory_only(smem_canon, "shared")
    gmem_canon = _canonicalize_gmem(g_buf)
    _assert_memory_only(gmem_canon, "global")

    # --- gmem: group by buffer shape, then split each group ---
    grouped_g, sep_g = _group_gmem_by_buffer_shape(gmem_canon, g_buf.shape)

    gmem_iters: list = []
    # Track which gmem_iters correspond to each original buffer dim to
    # later align with the copy region's extent!=1 dims.
    per_group_iter_slices: list = []  # list of (start_idx, end_idx) in gmem_iters
    for d in range(len(g_buf.shape)):
        before = len(gmem_iters)
        gmem_iters.extend(_split_multi_iter_group(grouped_g, sep_g, d, g_st[d], g_ext[d], analyzer))
        per_group_iter_slices.append((before, len(gmem_iters)))

    # --- smem: slice then regroup by "copy shape with ext=1 dropped" ---
    sliced_smem = _slice_and_canonicalize_smem(smem_canon, s_buf.shape, s_st, s_ext)

    # The post-split "copy shape" (per iter): for ext=1 iters, skip; for
    # ext>1 iters, use copy_ext.
    extgt1_iter_indices = [i for i, it in enumerate(gmem_iters) if not it.is_ext1]
    extgt1_shape = [gmem_iters[i].copy_ext for i in extgt1_iter_indices]

    if not extgt1_shape:
        # Entire copy is ext=1 everywhere: single element. Emit one
        # trivial DescDim per ext=1 iter at assembly time; no smem groups.
        return L1Result(swizzle_mode=swizzle_mode, gmem_iters=gmem_iters, smem_groups=[])

    regrouped = _regroup_smem_by_extgt1_shape(sliced_smem, extgt1_shape)
    if regrouped is None:
        raise ValueError(f"Cannot regroup smem layout by ext>1 copy shape {extgt1_shape}")
    grouped_s, sep_s = regrouped

    smem_groups: list = []
    for logical_idx, iter_idx in enumerate(extgt1_iter_indices):
        start = sep_s[logical_idx]
        end = sep_s[logical_idx + 1]
        shards = [
            SmemShard(extent=sh.extent, smem_stride=sh.stride)
            for sh in grouped_s.shard[start:end]
            if not analyzer.can_prove_equal(sh.extent, 1)
        ]
        smem_groups.append(SmemGroup(shards=shards, bound_gmem_iter_idx=iter_idx))

    return L1Result(swizzle_mode=swizzle_mode, gmem_iters=gmem_iters, smem_groups=smem_groups)


# ==============================================================================
# L2: segment algorithm
# ==============================================================================


def _find_contiguous_chain_prefix(smem_groups: list) -> list:
    """Return the indices (flat, across groups) of the maximal stride-1
    contiguous chain within the innermost smem group(s).

    Returns a list of (group_idx, shard_idx_within_group) tuples, ordered
    from inner to outer. Length of this list = max candidate j.
    """
    analyzer = Analyzer()
    # Concatenate all shards across groups, innermost→outermost. The chain
    # must start with stride 1 and each successive stride equals the product
    # of prior extents.
    flat = []
    for gi, group in enumerate(smem_groups):
        for si, sh in enumerate(group.shards):
            flat.append((gi, si, sh))

    if not flat:
        return []

    chain: list = []
    consumed: set = set()
    expected_stride: object = 1

    while True:
        for key, (gi, si, sh) in enumerate(flat):
            if key in consumed:
                continue
            if analyzer.can_prove_equal(sh.smem_stride, expected_stride):
                consumed.add(key)
                chain.append((gi, si))
                expected_stride = analyzer.simplify(expected_stride * sh.extent)
                break
        else:
            break

    return chain


def _distribute_selection(chain: list, smem_groups: list) -> dict:
    """From a chain prefix (inner→outer), return a per-group mapping
    ``group_idx -> sorted list of selected shard indices (outer→inner)``.

    Only the first ``prefix_len`` chain entries are used; caller slices
    ``chain[:prefix_len]`` before passing in.
    """
    per_group: dict = {}
    for gi, si in chain:
        per_group.setdefault(gi, []).append(si)
    for gi in per_group:
        per_group[gi].sort()
    # Each selected position in the chain must be a contiguous prefix of
    # the selected positions within that group (no gaps by construction of
    # the chain walk). Caller relies on this for u_{p_0} arithmetic.
    return per_group


def _check_alignment(
    gmem_iter: GmemIter, selected_positions: list, shards: list, analyzer: Analyzer
) -> bool:
    """Alignment: when j ≥ 1, ``u_{p_0} | G`` and ``u_{p_0} | copy_start``.

    ``p_0`` is the outermost selected position; ``u_{p_0}`` is the product
    of shard extents strictly inside ``p_0`` in the group's outer→inner
    order.
    """
    if not selected_positions:
        return True  # j=0: trivially ok

    p0 = selected_positions[0]
    u_p0: object = 1
    for si in range(p0 + 1, len(shards)):
        u_p0 = u_p0 * shards[si].extent
    u_p0 = analyzer.simplify(u_p0)

    if not _divides(u_p0, gmem_iter.shape, analyzer):
        return False
    if not _divides(u_p0, gmem_iter.copy_start, analyzer):
        return False
    return True


def _build_segments(
    gmem_iter: GmemIter, selected_positions: list, shards: list, analyzer: Analyzer
) -> list:
    """Cut the gmem axis into segments per the chain-prefix-selection rule.

    Segments (outer→inner):
      * Segment 0 (if j≥1): positions [0, p_0], extent G/u_{p_0},
        stride s·u_{p_0}, copy_range [g_st/u_{p_0}, g_st/u_{p_0}+E_0).
      * Segment i (i=1..j-1): positions [p_{i-1}+1, p_i], extent E_i,
        stride s·u_{p_i}, copy_range [0, E_i).
      * Trailing (if p_{j-1} < q-1): positions [p_{j-1}+1, q-1],
        extent E_j, stride s·1, copy_range [0, E_j).
      * j=0: single "trailing"-style segment covering the whole axis:
        extent G, stride s, copy_range [copy_start, copy_start+copy_ext).
    """
    G = gmem_iter.shape
    s = gmem_iter.stride
    copy_start = gmem_iter.copy_start
    copy_ext = gmem_iter.copy_ext
    q = len(shards)

    def _u_at(k: int) -> object:
        """u_k = prod(shards[m].extent for m > k)."""
        out: object = 1
        for m in range(k + 1, q):
            out = out * shards[m].extent
        return analyzer.simplify(out)

    # Helper: for a segment spanning positions [lo, hi] (inclusive), the
    # unselected shards inside contribute issue axes on the segment's desc
    # dim. Each contribution is (extent, coord_advance, smem_stride) where
    # coord_advance (in the segment's desc coord units) = u_k / u_{hi}.
    def _unselected_contribs(lo: int, hi: int) -> list:
        u_hi = _u_at(hi)
        out: list = []
        for m in range(lo, hi + 1):
            if m in selected_positions:
                continue
            u_m = _u_at(m)
            coord_advance = (
                analyzer.simplify(tvm.tirx.floordiv(u_m, u_hi))
                if not analyzer.can_prove_equal(u_hi, 1)
                else u_m
            )
            out.append((shards[m].extent, coord_advance, shards[m].smem_stride))
        return out

    segments: list = []

    if not selected_positions:
        # Case 2 applied to entire axis. The "selected position" at the
        # inner end is effectively q-1 with u=1, so unselected contribs
        # keep their full u_m as coord_advance.
        trailing_contribs = []
        for m in range(q):
            trailing_contribs.append((shards[m].extent, _u_at(m), shards[m].smem_stride))
        segments.append(
            Segment(
                local_shape=G,
                local_stride=s,
                local_copy_start=copy_start,
                local_copy_extent=copy_ext,
                is_selected=False,
                selected_shard_extent=1,
                unselected_contribs=trailing_contribs,
            )
        )
        return segments

    j = len(selected_positions)
    p_first = selected_positions[0]
    p_last = selected_positions[-1]

    # Segment 0 (outermost selected segment: positions [0, p_0])
    u_p0 = _u_at(p_first)
    E0: object = 1
    for m in range(0, p_first + 1):
        E0 = E0 * shards[m].extent
    E0 = analyzer.simplify(E0)

    seg0_shape = analyzer.simplify(tvm.tirx.floordiv(G, u_p0))
    seg0_stride = analyzer.simplify(s * u_p0)
    seg0_copy_start = analyzer.simplify(tvm.tirx.floordiv(copy_start, u_p0))
    segments.append(
        Segment(
            local_shape=seg0_shape,
            local_stride=seg0_stride,
            local_copy_start=seg0_copy_start,
            local_copy_extent=E0,
            is_selected=True,
            selected_shard_extent=shards[p_first].extent,
            unselected_contribs=_unselected_contribs(0, p_first),
        )
    )

    # Inner selected segments (i=1..j-1): positions [p_{i-1}+1, p_i]
    for i in range(1, j):
        lo = selected_positions[i - 1] + 1
        hi = selected_positions[i]
        Ei: object = 1
        for m in range(lo, hi + 1):
            Ei = Ei * shards[m].extent
        Ei = analyzer.simplify(Ei)
        u_pi = _u_at(hi)
        segments.append(
            Segment(
                local_shape=Ei,
                local_stride=analyzer.simplify(s * u_pi),
                local_copy_start=0,
                local_copy_extent=Ei,
                is_selected=True,
                selected_shard_extent=shards[hi].extent,
                unselected_contribs=_unselected_contribs(lo, hi),
            )
        )

    # Trailing (if p_{j-1} < q-1): positions [p_{j-1}+1, q-1]
    if p_last < q - 1:
        Ej: object = 1
        for m in range(p_last + 1, q):
            Ej = Ej * shards[m].extent
        Ej = analyzer.simplify(Ej)
        # For trailing, every position is unselected; "selected u" at the
        # inner end is u_{q-1} = 1, so coord_advance = u_m.
        trailing_contribs = []
        for m in range(p_last + 1, q):
            trailing_contribs.append((shards[m].extent, _u_at(m), shards[m].smem_stride))
        segments.append(
            Segment(
                local_shape=Ej,
                local_stride=s,
                local_copy_start=0,
                local_copy_extent=Ej,
                is_selected=False,
                selected_shard_extent=1,
                unselected_contribs=trailing_contribs,
            )
        )

    return segments


# ==============================================================================
# L3: assembly + hardware constraint validation + shrink
# ==============================================================================


def _assemble_plan(
    l1: L1Result, per_iter_selected: dict, chain: list, g_buf: Buffer, analyzer: Analyzer
) -> TmaPlan:
    """Build the final ``TmaPlan`` by stacking desc dims from all gmem iters.

    Emission (natural) order:
      * ext=1 gmem iters (in positional order) → one desc dim each (box=1).
      * ext>1 gmem iters (in positional order): for each, segments in
        outer→inner order produce desc dims; selected segments contribute
        box>1 dims, trailing contributes a box=1 dim.

    Then we **reorder** the desc dims so:
      * All box=1 dims (ext=1 iters and trailing segments) come first, in
        natural order.
      * All box>1 dims (selected segments) come last, in the reverse of
        the chain order — i.e. the outermost selected shard in the chain
        walk becomes the outermost box>1 desc dim, and the innermost
        selected shard (chain[0]) becomes the innermost desc dim. This
        matches how the TMA hardware writes the tile into swizzled smem:
        the innermost box dim (stride = 1 in gmem, ideally stride = 1 in
        smem too) must align with the innermost smem atom axis.

    Issue axes' ``dim_idx`` are remapped to the new positions.
    """

    dims_natural: list = []
    origins: list = []  # parallel to dims_natural: 'ext1' | 'trailing' | ('selected', chain_idx)
    issue_axes_natural: list = []

    # --- First pass: ext=1 iters ---
    for _, it in enumerate(l1.gmem_iters):
        if not it.is_ext1:
            continue
        dims_natural.append(
            DescDim(shape=it.shape, stride=it.stride, box=1, coord_base=it.copy_start)
        )
        origins.append("ext1")

    # --- Second pass: ext>1 iters ---
    for gi, group in enumerate(l1.smem_groups):
        iter_idx = group.bound_gmem_iter_idx
        gmem_iter = l1.gmem_iters[iter_idx]
        shards = group.shards
        selected_positions = per_iter_selected.get(gi, [])
        segments = _build_segments(gmem_iter, selected_positions, shards, analyzer)

        # For each selected position in this group, pre-compute its chain index.
        selected_chain_idx: dict = {}
        for p in selected_positions:
            for ci, (cgi, csi) in enumerate(chain):
                if cgi == gi and csi == p:
                    selected_chain_idx[p] = ci
                    break

        for i_seg, seg in enumerate(segments):
            dim_idx = len(dims_natural)
            box = seg.selected_shard_extent if seg.is_selected else 1
            dims_natural.append(
                DescDim(
                    shape=seg.local_shape,
                    stride=seg.local_stride,
                    box=box,
                    coord_base=seg.local_copy_start,
                )
            )
            if seg.is_selected:
                # Selected segments are emitted in the same order as
                # selected_positions (Segment 0 anchors p_0, etc.), so
                # i_seg directly indexes selected_positions for selected
                # segments. Trailing segments don't anchor any selection.
                p_anchor = selected_positions[i_seg]
                origins.append(("selected", selected_chain_idx[p_anchor]))
            else:
                origins.append("trailing")
            # Segment's unselected shards become issue axes on this dim.
            for extent, coord_advance, smem_stride in seg.unselected_contribs:
                issue_axes_natural.append(
                    IssueAxis(
                        extent=extent,
                        dim_idx=dim_idx,
                        coord_advance=coord_advance,
                        smem_stride=smem_stride,
                    )
                )

    # --- Permute: box=1 first (natural order), box>1 last (chain DESC) ---
    non_sel_indices = [
        idx for idx, o in enumerate(origins) if not (isinstance(o, tuple) and o[0] == "selected")
    ]
    sel_entries = [
        (idx, o[1]) for idx, o in enumerate(origins) if isinstance(o, tuple) and o[0] == "selected"
    ]
    sel_entries.sort(key=lambda x: -x[1])  # chain index descending = outer selected first
    new_order = non_sel_indices + [idx for idx, _ in sel_entries]
    old_to_new = {old: new for new, old in enumerate(new_order)}

    dims = [dims_natural[old] for old in new_order]
    issue_axes = [
        IssueAxis(
            extent=ax.extent,
            dim_idx=old_to_new[ax.dim_idx],
            coord_advance=ax.coord_advance,
            smem_stride=ax.smem_stride,
        )
        for ax in issue_axes_natural
    ]

    elem_bytes = tvm.DataType(g_buf.dtype).bits // 8
    plan = TmaPlan(
        swizzle_mode=l1.swizzle_mode,
        dims=dims,
        issue_axes=issue_axes,
        tensor_ptr=g_buf.data,
        elem_bytes=elem_bytes,
        elem_dtype=g_buf.dtype,
    )
    return _merge_contig_full_box_dims(plan, analyzer)


def _plan_needs_alignment_fix(dims, elem_bytes, analyzer: Analyzer) -> bool:
    """``True`` iff some non-innermost dim has a byte-stride that isn't a
    multiple of 16. cuTensorMap rejects such descriptors; merge+promote is
    the way out. If the plan already satisfies the constraint, leave it
    alone — the natural shape is what kernels expect and what existing
    codegen tests pin.
    """
    if len(dims) <= 1:
        return False
    for d in dims[:-1]:
        byte_stride = analyzer.simplify(d.stride * elem_bytes)
        if not analyzer.can_prove_equal(tvm.tirx.floormod(byte_stride, 16), 0):
            return True
    return False


def _merge_contig_full_box_dims(plan: TmaPlan, analyzer: Analyzer) -> TmaPlan:
    """Collapse adjacent fully-boxed dims that are physically contiguous.

    Two adjacent dims ``outer`` (at i) and ``inner`` (at i+1) merge when ALL of:

      1. Physically contiguous: ``outer.stride == inner.shape * inner.stride``.
         Walking inner.shape elements at inner.stride lands exactly on the
         next outer element, so the two dims jointly cover one stride-1 run.
      2. Both fully boxed (``box == shape``).  A partial box is a strided
         slice; flattening it would change which elements the descriptor
         touches.
      3. Runtime coord on each dim is provably 0.  The descriptor coord for
         dim d at iteration t equals
             d.coord_base + Σ(iter_val · ax.coord_advance for ax in issue_axes
                              if ax.dim_idx == d)
         For the merged dim's coord to be a constant 0 (matching the implicit
         coord of the collapsed pair), both halves must satisfy:
           * static term: ``coord_base == 0``,
           * dynamic term: no ``IssueAxis`` binds this dim_idx.
      4. Merged ``box <= 256`` (TMA hardware limit on boxDim).

    Scan inner→outer (greedy from rank-2 down to 0) so the innermost stride
    boundary is fixed first.

    When a candidate pair is blocked solely by ``merged_box > 256`` and the
    layout admits an element-type promotion (current ``elem_bytes < 8``,
    innermost extent even, all non-innermost element-strides even, no
    issue_axis on innermost), promote ``elem_bytes`` one step (x2), halve
    the innermost extent/box and the non-innermost strides, and retry the
    merge.  Promotion preserves byte-level semantics: byte-stride is
    ``stride * elem_bytes`` and stays unchanged across promotion.

    Repeats until no merges and no promotions are possible.  ``issue_axes``
    dim indices are shifted to track removed dims; the innermost
    ``coord_advance`` is also halved on each promotion (it's in element
    units).
    """
    dims = list(plan.dims)
    issue_axes = list(plan.issue_axes)
    elem_bytes = plan.elem_bytes
    elem_dtype = plan.elem_dtype

    # Only attempt the merge+promote rewrite when the original plan
    # already violates cuTensorMap's 16-byte non-innermost-stride rule.
    # An aligned plan is left intact: descriptor shape matches the
    # natural buffer layout, which is what users (and goldens) expect.
    if not _plan_needs_alignment_fix(dims, elem_bytes, analyzer):
        return plan

    def has_issue_axis(idx):
        return any(ax.dim_idx == idx for ax in issue_axes)

    def shift_issue_axes_after_remove(axes, removed_i):
        return [
            IssueAxis(
                extent=ax.extent,
                dim_idx=ax.dim_idx if ax.dim_idx <= removed_i else ax.dim_idx - 1,
                coord_advance=ax.coord_advance,
                smem_stride=ax.smem_stride,
            )
            for ax in axes
        ]

    def try_merge_at(i, dims_, axes_):
        outer, inner = dims_[i], dims_[i + 1]
        if any(ax.dim_idx in (i, i + 1) for ax in axes_):
            return None, None
        if not analyzer.can_prove_equal(outer.coord_base, 0):
            return None, None
        if not analyzer.can_prove_equal(inner.coord_base, 0):
            return None, None
        if not analyzer.can_prove_equal(outer.box, outer.shape):
            return None, None
        if not analyzer.can_prove_equal(inner.box, inner.shape):
            return None, None
        if not analyzer.can_prove_equal(outer.stride, inner.shape * inner.stride):
            return None, None
        merged_box = analyzer.simplify(outer.box * inner.box)
        if not analyzer.can_prove(merged_box <= 256):
            # signal "blocked only by box>256" so caller can try promotion
            return "blocked_box", merged_box
        merged = DescDim(
            shape=analyzer.simplify(outer.shape * inner.shape),
            stride=inner.stride,
            box=merged_box,
            coord_base=0,
        )
        new_dims = [*dims_[:i], merged, *dims_[i + 2 :]]
        new_axes = shift_issue_axes_after_remove(axes_, i)
        return new_dims, new_axes

    _PROMOTE_CHAIN = {1: ("uint16", 2), 2: ("uint32", 4), 4: ("uint64", 8)}

    def try_promote(dims_, axes_, eb, edt):
        if eb not in _PROMOTE_CHAIN:
            return None
        if not dims_:
            return None
        innermost_idx = len(dims_) - 1
        if any(ax.dim_idx == innermost_idx for ax in axes_):
            return None
        inner = dims_[innermost_idx]
        if not analyzer.can_prove_equal(inner.stride, 1):
            return None
        if not analyzer.can_prove_equal(tvm.tirx.floormod(inner.shape, 2), 0):
            return None
        for d in dims_[:-1]:
            if not analyzer.can_prove_equal(tvm.tirx.floormod(d.stride, 2), 0):
                return None
        new_dtype, new_eb = _PROMOTE_CHAIN[eb]
        new_dims = []
        for j, d in enumerate(dims_):
            if j == innermost_idx:
                new_dims.append(
                    DescDim(
                        shape=analyzer.simplify(tvm.tirx.floordiv(d.shape, 2)),
                        stride=d.stride,
                        box=analyzer.simplify(tvm.tirx.floordiv(d.box, 2)),
                        coord_base=analyzer.simplify(tvm.tirx.floordiv(d.coord_base, 2)),
                    )
                )
            else:
                new_dims.append(
                    DescDim(
                        shape=d.shape,
                        stride=analyzer.simplify(tvm.tirx.floordiv(d.stride, 2)),
                        box=d.box,
                        coord_base=d.coord_base,
                    )
                )
        new_axes = [
            IssueAxis(
                extent=ax.extent,
                dim_idx=ax.dim_idx,
                coord_advance=(
                    analyzer.simplify(tvm.tirx.floordiv(ax.coord_advance, 2))
                    if ax.dim_idx == innermost_idx
                    else ax.coord_advance
                ),
                smem_stride=ax.smem_stride,
            )
            for ax in axes_
        ]
        return new_dims, new_axes, new_eb, new_dtype

    while True:
        # Greedy inner→outer merge sweep.
        merged_any = False
        blocked_by_box = False
        for i in range(len(dims) - 2, -1, -1):
            res, _info = try_merge_at(i, dims, issue_axes)
            if res == "blocked_box":
                blocked_by_box = True
                continue
            if res is not None:
                dims, issue_axes = res, _info
                merged_any = True
                break
        if merged_any:
            continue
        # Nothing merged this pass; try promotion if any pair was box-blocked.
        if not blocked_by_box:
            break
        promoted = try_promote(dims, issue_axes, elem_bytes, elem_dtype)
        if promoted is None:
            break
        dims, issue_axes, elem_bytes, elem_dtype = promoted

    return TmaPlan(
        swizzle_mode=plan.swizzle_mode,
        dims=dims,
        issue_axes=issue_axes,
        tensor_ptr=plan.tensor_ptr,
        elem_bytes=elem_bytes,
        elem_dtype=elem_dtype,
    )


def _validate_hw_constraints(plan: TmaPlan, dtype: str) -> tuple:
    """Return ``(ok, reason)``. ``reason`` is the error string when ``ok`` is False."""
    analyzer = Analyzer()

    if plan.rank == 0:
        return False, "TMA descriptor rank must be ≥ 1"
    if plan.rank > 5:
        return False, f"TMA descriptor rank {plan.rank} exceeds hardware limit of 5"

    # Innermost dim stride must be 1 (unit stride).
    inner = plan.dims[-1]
    if not analyzer.can_prove_equal(inner.stride, 1):
        return False, f"TMA innermost dim must have unit stride; got {inner.stride}"

    # Innermost box times element size must fit the swizzle atom.
    if not _swizzle_inner_box_fits(dtype, plan.swizzle_mode, inner.box):
        return False, "TMA innermost box exceeds the swizzle atom size"

    return True, ""


def _build_plan_with_shrink(l1: L1Result, g_buf: Buffer, s_buf: Buffer) -> TmaPlan:
    """Enumerate chain prefix length j from max down to 0, validate
    alignment per gmem iter, build and validate the plan. Return the first
    plan that passes everything. Raise when j=0 still fails.
    """
    analyzer = Analyzer()
    chain = _find_contiguous_chain_prefix(l1.smem_groups)
    max_j = len(chain)

    # Empty-smem_groups case (all ext=1): the assembly still yields a
    # valid plan (trivial desc dims).
    if not l1.smem_groups:
        plan = _assemble_plan(l1, {}, [], g_buf, analyzer)
        ok, reason = _validate_hw_constraints(plan, s_buf.dtype)
        if ok:
            return plan
        fail(f"TMA plan (no smem groups) failed hardware check: {reason}")

    last_reason = "no valid plan"
    for j in range(max_j, -1, -1):
        per_iter_selected: dict = _distribute_selection(chain[:j], l1.smem_groups)

        # Check alignment for each ext>1 iter.
        aligned = True
        for gi, group in enumerate(l1.smem_groups):
            iter_idx = group.bound_gmem_iter_idx
            sel = per_iter_selected.get(gi, [])
            if not _check_alignment(l1.gmem_iters[iter_idx], sel, group.shards, analyzer):
                aligned = False
                last_reason = f"alignment fails for gmem iter {iter_idx} at j={j}"
                break
        if not aligned:
            continue

        plan = _assemble_plan(l1, per_iter_selected, chain[:j], g_buf, analyzer)
        ok, reason = _validate_hw_constraints(plan, s_buf.dtype)
        if ok:
            return plan
        last_reason = reason

    fail(f"TMA plan: all chain prefix lengths rejected; last reason: {last_reason}")


# ==============================================================================
# Emit layer + entry point
# ==============================================================================


def copy_tma_impl(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    """Lower global<->shared copy_async to TMA using the unified algorithm.

    Emits a device-side unrolled loop over the flat issue-axis extent and
    a host-side ``cuTensorMapEncodeTiled`` (deduped via cache key).
    """
    op_call = TilePrimitiveCall.downcast(op_call)
    dst_buffer_region, src_buffer_region = op_call.dst, op_call.src
    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer

    src_scope, dst_scope = src.scope(), dst.scope()
    if src_scope == "global" and dst_scope.startswith("shared"):
        direction = "g2s"
        s_buf, g_buf = dst, src
        shared_region, global_region = dst_buffer_region, src_buffer_region
    elif src_scope.startswith("shared") and dst_scope == "global":
        direction = "s2g"
        s_buf, g_buf = src, dst
        shared_region, global_region = src_buffer_region, dst_buffer_region
    else:
        raise ValueError(
            f"Unsupported combination of src and dst scopes: src={src_scope} dst={dst_scope}"
        )

    g_st = [region.min for region in global_region.region]
    g_ext = [region.extent for region in global_region.region]
    s_st = [region.min for region in shared_region.region]
    s_ext = [region.extent for region in shared_region.region]

    oob_mode = _normalize_oob_mode(s_buf.dtype, op_call.config.get("oob", None))
    oob_fill_kind = _oob_fill_kind(oob_mode)

    # L1 → L2 → L3
    l1 = _build_l1_result(s_buf, g_buf, g_st, g_ext, s_st, s_ext)
    plan = _build_plan_with_shrink(l1, g_buf, s_buf)

    # Direction / runtime-config bits that don't affect the plan itself.
    cta_group = op_call.config.get("cta_group", None)
    if cta_group is None:
        cta_group = 1 if sctx.target.arch == "sm_100a" else -1

    cta_mask = op_call.config.get("cta_mask", None)
    if cta_mask is not None:
        assert direction == "g2s", "cta_mask is only supported for global to shared copy"
    else:
        cta_mask = 0

    if direction == "g2s":
        mbar = op_call.config.get("mbar", None)
        if mbar is None:
            raise ValueError("mbar is not set in config")
    use_tma_reduce = op_call.config.get("use_tma_reduce", None)

    dtype_bytes = plan.elem_bytes
    tma_global_strides = [stride * dtype_bytes for stride in plan.g_strides]
    # cuTensorMap omits the last dim's stride (implicit element size).
    tma_g_strides_for_map = tma_global_strides[:-1] if plan.rank > 1 else []
    element_strides = [1] * plan.rank

    flat_total_extent = plan.flatten_total_extent()

    def compute_offsets_and_tma_coords(loop_var):
        s_offset, coords = plan.offsets_and_coords(loop_var)
        simplified = _simplify_with_var_ranges(
            [s_offset, *coords], [(loop_var, flat_total_extent)], sctx
        )
        return simplified[0], reversed(simplified[1:])

    def val_key(value) -> str:
        return str(value)

    tensormap_cache_key = (
        f"tensormap:{hash(plan.tensor_ptr)}:{g_buf.dtype}:{val_key(plan.rank)}"
        f":{tuple(val_key(v) for v in plan.shape)}"
        f":{tuple(val_key(v) for v in tma_g_strides_for_map)}"
        f":{tuple(val_key(v) for v in plan.box_dim)}"
        f":{val_key(plan.swizzle_mode.value)}:{oob_fill_kind}"
    )

    cached_tensormap = sctx.cache_get(tensormap_cache_key)
    if cached_tensormap is not None:
        tensor_map = cached_tensormap
        tensormap_is_cached = True
    else:
        tensor_map = T.Var(
            g_buf.data.name + "_tensormap", dtype=T.handle("tensormap").type_annotation
        )
        tensormap_is_cached = False

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def impl():
        for loop_vars in T.unroll(flat_total_extent):
            s_offset, tma_coords = T.meta_var(compute_offsets_and_tma_coords(loop_vars))
            s_buf_w_offset = T.decl_buffer(
                s_buf.shape,
                s_buf.dtype,
                s_buf.data,
                elem_offset=s_buf.elem_offset + s_offset,
                scope=s_buf.scope(),
                layout=_to_tile_layout(s_buf.layout, s_buf.shape),
            )

            if direction == "g2s":
                T.ptx.cp_async.bulk.tensor.g2c(
                    plan.rank,
                    s_buf_w_offset.ptr_to(s_st),
                    mbar,
                    T.address_of(tensor_map),
                    cta_mask,
                    cta_group,
                    op_call.config.get("cache_hint", ""),
                    *tma_coords,
                )
            else:
                if use_tma_reduce is None:
                    T.ptx.cp_async.bulk.tensor.s2g(
                        plan.rank,
                        s_buf_w_offset.ptr_to(s_st),
                        T.address_of(tensor_map),
                        op_call.config.get("cache_hint", ""),
                        *tma_coords,
                    )
                else:
                    T.ptx.cp_async.bulk.tensor.s2g_reduce(
                        plan.rank,
                        s_buf_w_offset.ptr_to(s_st),
                        T.address_of(tensor_map),
                        op_call.config.get("cache_hint", ""),
                        use_tma_reduce,
                        *tma_coords,
                    )
    # fmt: on

    if not tensormap_is_cached:
        # fmt: off
        @T.prim_func(check_well_formed=False)
        def create_tensor_map():
            T.Bind(T.tvm_stack_alloca("tensormap", 1), var=tensor_map)
            T.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                tensor_map,
                plan.elem_dtype,
                plan.rank,
                plan.tensor_ptr,
                *reversed(plan.shape),
                *reversed(tma_g_strides_for_map) if plan.rank > 1 else [],
                *reversed(plan.box_dim),
                *element_strides,
                0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
                plan.swizzle_mode.value,
                2,  # CU_TENSOR_MAP_L2_PROMOTION_L2_128B
                oob_fill_kind,
            )
            Tx.tvm_kernel_replace_point()
        # fmt: on

        sctx.add_init_stmt(create_tensor_map.body, host=True)
        sctx.cache_set(tensormap_cache_key, tensor_map)

    if bool(op_call.config.get("prefetch_tensormap", False)):
        if "warp_id_in_cta" not in sctx.launch_params:
            fail("tma prefetch_tensormap requires warp_id_in_cta launch param")
        prefetch_cache_key = f"prefetch_tensormap:{tensormap_cache_key}"
        if sctx.cache_get(prefetch_cache_key) is None:
            warp_id_in_cta = sctx.launch_params["warp_id_in_cta"].var

            # fmt: off
            @T.prim_func(check_well_formed=False)
            def prefetch_tensor_map():
                if warp_id_in_cta == 0:
                    T.ptx.prefetch_tensormap(T.address_of(tensor_map))
                Tx.tvm_kernel_replace_point()
            # fmt: on

            sctx.add_init_stmt(prefetch_tensor_map.body)
            sctx.cache_set(prefetch_cache_key, tensor_map)

    return impl


# Variant: copy_async/tma (priority=10). Applies at single-thread exec scope
# on Hopper+ (SM90+) for global↔shared copies; DispatchFail otherwise.
@register_dispatch(
    "copy_async",
    "cuda",
    variant="tma",
    priority=10,
    when=[
        predicate(
            "validate_copy_op", lambda op, sctx: (validate_copy_op(op, sctx), "not a valid copy op")
        ),
        predicate(
            "single_thread",
            lambda op, sctx: (
                single_thread(op, sctx),
                f"unsupported exec_scope {sctx.exec_scope}, expected single thread",
            ),
        ),
    ],
)
def copy_async_dispatch_tma(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return copy_tma_impl(op, sctx)
