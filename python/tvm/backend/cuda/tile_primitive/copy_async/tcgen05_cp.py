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

"""smem->tmem dispatch via ``tcgen05.cp`` — generic planner for all PTX shapes.

``tcgen05.cp`` is inherently async; this dispatch emits the cp loop only and
leaves completion signaling (``tcgen05.commit`` against a barrier) to the
caller. Callers who want sync semantics should issue ``tcgen05.commit``
themselves after the copy.

Routing
-------
- ``shape=`` config: forces that shape (see ``_CP_SHAPE_MULTICASTS`` below;
  atom rows/bits are parsed from the shape name itself).
- No ``shape`` config: the shape is inferred from the buffer layouts — each
  candidate (widest atom first) is planned until one validates. Bare warpx4
  copies resolve to ``32x128b.warpx4`` exactly as the historical dispatch.

Descriptor fields are always derived from the layouts.

Shape table (PTX ISA 8.8 §9.7.16.9.2)
-------------------------------------
Each ``tcgen05.cp`` shape is ``rows x bits-per-row``; some shapes REQUIRE a
multicast qualifier. Multicast replicates the copied data across warp lane
slabs of the destination TMEM:

===========  ==========================  ======================  ==============
shape        multicast                   t lane pattern          t replica
===========  ==========================  ======================  ==============
128x256b     (none)                      (128, 1@TLane)          —
4x256b       (none)                      (4, 32@TLane)           —
128x128b     (none)                      (128, 1@TLane)          —
64x128b      warpx2::02_13 (required)    (64, 1@TLane)           (2, 64@TLane)
64x128b      warpx2::01_23 (required)    (2,32):(64,1)@TLane     (2, 32@TLane)
32x128b      warpx4 (required)           (32, 1@TLane)           (4, 32@TLane)
===========  ==========================  ======================  ==============

The ``warpx2`` row→lane mappings follow PTX ISA 8.8 p675 ("each warp in the
warp pair receives half of the data", pairs 02_13 = {0,2},{1,3} and 01_23 =
{0,1},{2,3}): the two warps of a pair mirror each other and the pairs split
the 64 rows in listed order. For 02_13 this is exactly the Layout E / B
datapath organization (Figures 213/207, §9.7.16.10.5: M 0-31 in warp-ranks 0
and 2, M 32-63 in warp-ranks 1 and 3). ``4x256b`` writes one row per
warp-quadrant datapath (lanes 0/32/64/96). All row→lane mappings and replica
structures are verified bit-exactly on B200 hardware by the round-trip tests
in ``test_tcgen05_cp.py``.

Algorithm
---------
Given ``Tx.copy_async(t_region, s_region)`` where t is in tmem (declared with
the replica pattern of the requested (shape, multicast)), and s is in shared
memory:

A. Slice + canonicalize both layouts at the given regions.
B. Verify ``t.replica`` matches the shape table entry (empty for
   non-multicast shapes).
C. Compute permutation that puts TLane first, then TCol stride-descending;
   apply to t.permute_dims and to s via group + permute_by_groups.
D. Canonicalize again.
E. Isolate broadcast: split-by-stride-zero on both t and s; their split
   sequences must match (same distinct prefix prods + broadcast extents).
   Drop stride-0 iters → ``t_iso`` and ``s_iso``.
F. Split each side's iters into three segments by grouping the flattened
   element space as ``atom_rows x n_mid x elem_per_atom``
   (``elem_per_atom = atom_bits / dtype_bits``). ``*_lane`` covers exactly one
   instruction's rows (product == atom_rows); ``*_col`` = one instruction's
   own columns; ``*_middle`` = everything between — the outer columns (which
   instruction) and, when the region has more rows than one atom (e.g. 4x256b
   tiling each warp's first 16 rows: the M=64 Layout-F scatter), the extra row
   iters, stepped through the taddr lane half-word. ``t_*`` = tmem side,
   ``s_*`` = smem side.
   Validate:
   - t_lane (row → TMEM lane) matches the shape table lane pattern (above)
   - t_col: one row's elements land in contiguous TMEM columns
   - s_col: one row = contiguous 16B units in smem; for 256b atoms the two
     units may sit at a non-contiguous stride when unswizzled (→ LDO field)
   - s_lane: rows sit in smem as (atom_rows/8, 8) groups with strides
     (SDO_stride, atom_K_stride); 4x256b is a single 4-row group (SDO=0)
   - atom_K_byte ∈ {16, 32, 64, 128} → swizzle_mode 0..3, which must match
     s_buf.layout's swizzle (if any)
G. Alignment checks:
   - t_iso TCol offset ≡ 0 (mod 128-bit)
   - t_iso TLane offset folds into the taddr lane half-word
   - s_iso m offset ≡ 0 (mod 16B for sw=0; mod atom_size for sw>0)
   - middle iter strides 16B-aligned
H. middle 1-1 correspondence (simple-mode): t_middle and s_middle have same
   iter count and matching extents per position.
I. Emit:
   - SmemDescriptor encoded once at SMEM base (hoisted via post_buffer_def_stmt).
   - Loop over middle iters; each cp uses ``desc.add_16B_offset(init + loop)``
     and writes to ``tmem_addr + t_addr_off + Σ i_j * t_step_j``.
"""

import functools
import operator

import tvm
from tvm.arith import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.layout import ComposeLayout, TCol, TileLayout, TLane
from tvm.tirx.layout import m as m_axis
from tvm.tirx.operator.tile_primitive import DispatchContext, predicate, register_dispatch
from tvm.tirx.stmt import AllocBuffer, Evaluate, SeqStmt
from tvm.tirx.tile_primitive import TilePrimitiveCall

from ..copy import _single_thread_exec

# Allowed .multicast qualifiers per shape (PTX ISA 8.8 §9.7.16.9.2): "" = none,
# 64x128b needs an explicit warpx2 pick.
_CP_SHAPE_MULTICASTS = {
    "128x256b": ("",),
    "4x256b": ("",),
    "128x128b": ("",),
    "64x128b": ("warpx2::02_13", "warpx2::01_23"),
    "32x128b": ("warpx4",),
}


def _shape_dims(shape):
    """(atom_rows, atom_bits) parsed from the PTX shape name "<rows>x<bits>b"."""
    rows, bits = shape.split("x")
    return int(rows), int(bits[:-1])


# Inference order for bare copies: widest atom first (a 256b source also fits
# 128b at 2x instructions); other candidates are mutually exclusive.
_CP_SHAPE_CANDIDATES = (
    ("128x256b", ""),
    ("4x256b", ""),
    ("128x128b", ""),
    ("64x128b", "warpx2::02_13"),
    ("64x128b", "warpx2::01_23"),
    ("32x128b", "warpx4"),
)


def _cp_lane_replica_pattern(shape: str, multicast: str):
    """Expected (t_lane pattern, t.replica pattern) for a (shape, multicast).

    Both are lists of ``(extent, stride)`` on the TLane axis. The replica
    strides are the warp-slab offsets that receive multicast copies; the lane
    pattern is the row → lane mapping of one copy.

    Multicast semantics (PTX ISA 8.8 §9.7.16.9.2, p675): for warpx2 the data
    is "multicasted into warp pairs and each warp in the warp pair receives
    half of the data" — the two warps OF a pair MIRROR each other (hold the
    same half), and the pairs split the 64 rows in listed order:

    - ``warpx2::02_13`` (pairs {0,2}, {1,3}): rows 0-31 to warps 0/2, rows
      32-63 to warps 1/3. One copy occupies lanes 0-63 in row order (warp0 +
      warp1), replicated at lane offset +64 (warp2 + warp3) → lane
      (64, 1@TLane), replica (2, 64@TLane). This is exactly the Layout E /
      Layout B datapath organization (PTX ISA 8.8 Figures 213 and 207:
      M 0-31 in warp-ranks 0 and 2, M 32-63 in warp-ranks 1 and 3).
    - ``warpx2::01_23`` (pairs {0,1}, {2,3}): rows 0-31 to warps 0/1, rows
      32-63 to warps 2/3 → one copy at lanes 0-31 (rows 0-31) + lanes 64-95
      (rows 32-63), replicated at +32 → lane (2, 32):(64, 1)@TLane, replica
      (2, 32@TLane).

    ``4x256b`` writes one row per warp-quadrant datapath: rows 0..3 land at
    lanes 0, 32, 64, 96 → lane (4, 32@TLane), no replica.

    All mappings are verified bit-exactly on B200 hardware by the round-trip
    tests in ``test_tcgen05_cp.py``.
    """
    rows, _ = _shape_dims(shape)
    if multicast == "":
        if rows == 4:
            return [(4, 32)], []
        return [(rows, 1)], []
    if multicast == "warpx4":
        return [(32, 1)], [(4, 32)]
    if multicast == "warpx2::02_13":
        return [(64, 1)], [(2, 64)]
    if multicast == "warpx2::01_23":
        return [(2, 64), (32, 1)], [(2, 32)]
    raise ValueError(f"unknown tcgen05.cp multicast {multicast!r}")


def _resolve_cp_shape(op_call: TilePrimitiveCall):
    """Resolve (shape, multicast) from an explicit ``shape=`` config."""
    shape = str(op_call.config["shape"])
    multicast = op_call.config.get("multicast")
    allowed = _CP_SHAPE_MULTICASTS.get(shape)
    if allowed is None:
        raise ValueError(
            f"unknown tcgen05.cp shape {shape!r}; expected one of {sorted(_CP_SHAPE_MULTICASTS)}"
        )
    if multicast is None:
        if len(allowed) == 1:
            multicast = allowed[0]
        else:
            raise ValueError(
                f"tcgen05.cp shape {shape!r} requires an explicit multicast config; "
                f"choose one of {list(allowed)}"
            )
    else:
        multicast = str(multicast)
        if multicast not in allowed:
            raise ValueError(
                f"illegal multicast {multicast!r} for tcgen05.cp shape {shape!r}; "
                f"allowed: {list(allowed)}"
            )
    return shape, multicast


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _compute_perm(t):
    def key(p):
        it = p[1]
        return (0 if it.axis == TLane else 1, -int(it.stride))

    return [i for i, _ in sorted(enumerate(t.shard), key=key)]


def _split_by_zero(lay):
    """Split lay.shard into segments at stride==0 positions.
    Returns (split_seq, kept_iters_with_nonzero_stride)."""
    new_seq = []
    keep = []
    cur = 1
    for it in lay.shard:
        e, st = int(it.extent), int(it.stride)
        if st == 0:
            if cur > 1:
                new_seq.append(cur)
            new_seq.append(e)
            cur = 1
        else:
            cur *= e
            keep.append(it)
    if cur > 1:
        new_seq.append(cur)
    return new_seq, keep


def _align_middles(t_middle, s_middle):
    """Sub-group both middles by union-of-boundaries so they become 1-1.

    Both inputs must be post-canonicalize iter lists with equal extent products.
    The shape is the consecutive ratios of sorted(B_t U B_s) where B_x is the
    set of cumulative extent boundaries of x_middle. Each segment then contains
    at most one iter per side (whole or sub-divided), so trivially single iter.

    Returns (new_t_middle, new_s_middle) with len() == len() == k segments,
    each segment a single Iter on each side.
    """

    def cum_bounds(iters):
        b, p = [], 1
        for it in iters:
            p *= int(it.extent)
            b.append(p)
        return b

    t_bounds = cum_bounds(t_middle)
    s_bounds = cum_bounds(s_middle)
    if not t_bounds and not s_bounds:
        return t_middle, s_middle
    N = t_bounds[-1] if t_bounds else s_bounds[-1]
    if (s_bounds and s_bounds[-1] != N) or (t_bounds and t_bounds[-1] != N):
        raise ValueError(f"middle extent mismatch: t={N} s={s_bounds[-1] if s_bounds else 0}")

    cuts = sorted(set(t_bounds) | set(s_bounds))
    shape, prev = [], 1
    for c in cuts:
        if c % prev != 0:
            raise ValueError(
                f"middle align failed: cut {c} not divisible by prev cut {prev} "
                f"(t_bounds={t_bounds}, s_bounds={s_bounds})"
            )
        shape.append(c // prev)
        prev = c

    def subgroup(iters):
        if len(iters) == 1 and shape == [int(iters[0].extent)]:
            return iters
        lay, _seps = TileLayout.from_iters(iters, [], {}).group(shape)
        seps = list(_seps)
        out = []
        for i in range(len(shape)):
            seg = list(lay.shard[seps[i] : seps[i + 1]])
            seg_canon = list(TileLayout.from_iters(seg, [], {}).canonicalize().shard)
            if len(seg_canon) != 1:
                raise ValueError(
                    f"middle sub-group seg[{i}] not single iter after canon: {seg_canon}"
                )
            out.append(seg_canon[0])
        return out

    return subgroup(t_middle), subgroup(s_middle)


# -----------------------------------------------------------------------------
# Plan (state object)
# -----------------------------------------------------------------------------
def _build_plan(op_call: TilePrimitiveCall):
    """Run A..H and return a dispatch plan.

    Plan fields:
      - s_buf, t_buf
      - dtype, dtype_bits
      - shape, multicast (PTX qualifier strings)
      - elem_per_atom, elem_per_32b
      - SmemSwizzleMode (int)
      - LDO_field, SDO_field, atom_K_byte
      - middle_iters: list of (extent, s_step_16B, t_step_32bcol)
      - init_off_16B (Expr)
      - t_addr_off (Expr, taddr offset of the first cp: 32-bit col
        offset plus the region row offset in the lane half-word)
    """
    op_call = TilePrimitiveCall.downcast(op_call)
    if op_call.config.get("shape") is not None:
        shape, multicast = _resolve_cp_shape(op_call)
        return _plan_for_shape(op_call, shape, multicast)
    # No shape config: infer from the buffer layouts.
    multicast_cfg = op_call.config.get("multicast")
    errors = []
    for shape, multicast in _CP_SHAPE_CANDIDATES:
        if multicast_cfg is not None and str(multicast_cfg) != multicast:
            continue
        try:
            return _plan_for_shape(op_call, shape, multicast)
        except ValueError as err:
            errors.append(f"{shape}{('.' + multicast) if multicast else ''}: {err}")
    raise ValueError(
        "no tcgen05.cp shape fits the given layouts; candidates rejected:\n  " + "\n  ".join(errors)
    )


def _plan_for_shape(op_call: TilePrimitiveCall, shape: str, multicast: str):
    """Run A..I for one (shape, multicast); raises ValueError on any mismatch."""
    dst_region, src_region = op_call.args[:2]
    s_buf: Buffer = src_region.buffer
    t_buf: Buffer = dst_region.buffer
    dtype = s_buf.dtype
    dtype_bits = DataType(dtype).bits
    elem_per_128b = 128 // dtype_bits  # elements per 16B descriptor unit
    elem_per_32b = 32 // dtype_bits
    atom_rows, atom_bits = _shape_dims(shape)
    elem_per_atom = atom_bits // dtype_bits  # per-lane elements per cp

    # C: slice + canonicalize.
    s_region = [(r.min, r.min + r.extent) for r in src_region.region]
    t_region = [(r.min, r.min + r.extent) for r in dst_region.region]
    s_sliced = s_buf.layout.slice(list(s_buf.shape), s_region)
    s = s_sliced.canonicalize()
    t = t_buf.layout.slice(list(t_buf.shape), t_region).canonicalize()

    # If s is ComposeLayout (swizzle over a tile), peel off the swizzle for
    # stride analysis; record the swizzle object for cross-checks.
    s_swizzle_mode_from_layout = 0
    s_swizzle_obj = None
    if isinstance(s, ComposeLayout):
        s_swizzle_obj = s
        s_swizzle_mode_from_layout = int(s.swizzle_len)
        s = s.tile_layout

    # B: replica router check per (shape, multicast).
    lane_pattern, replica_pattern = _cp_lane_replica_pattern(shape, multicast)
    rep = t.replica
    rep_ok = len(rep) == len(replica_pattern) and all(
        int(r.extent) == e and int(r.stride) == st and r.axis == TLane
        for r, (e, st) in zip(rep, replica_pattern)
    )
    if not rep_ok:
        raise ValueError(
            f"replica mismatch for tcgen05.cp {shape}"
            f"{('.' + multicast) if multicast else ''}: expected "
            f"{[f'({e}, {st}@TLane)' for e, st in replica_pattern]}, got t.replica = "
            f"{[(int(r.extent), int(r.stride), str(r.axis)) for r in rep]}"
        )

    # C: permute (TLane first, TCol stride desc).
    perm = _compute_perm(t)
    t_shape_for_group = [int(it.extent) for it in t.shard]
    s_grp, seps = s.group(t_shape_for_group)
    s_p = s_grp.permute_by_groups(list(seps), perm).canonicalize()
    t_p = t.permute_dims(perm).canonicalize()

    # E: isolate broadcast.
    seq_t, keep_t = _split_by_zero(t_p)
    seq_s, keep_s = _split_by_zero(s_p)
    if seq_t != seq_s:
        raise ValueError(f"isolate split mismatch: t={seq_t} s={seq_s}")
    s_iso = TileLayout.from_iters(keep_s, list(s_p.replica), dict(s_p.offset))
    t_iso = TileLayout.from_iters(keep_t, list(t_p.replica), dict(t_p.offset))

    # F: group into (atom_rows, middle, elem_per_atom).
    def shard_prod(lay):
        return functools.reduce(operator.mul, [int(it.extent) for it in lay.shard], 1)

    n_lane, n_col = atom_rows, elem_per_atom
    atom_elems = n_lane * n_col
    for side, total in (("t", shard_prod(t_iso)), ("s", shard_prod(s_iso))):
        if total < atom_elems or total % atom_elems:
            raise ValueError(
                f"{side} region ({total} elems) is not a multiple of one "
                f"{shape} atom ({n_lane}x{n_col} elems)"
            )
    n_mid_t = shard_prod(t_iso) // atom_elems
    n_mid_s = shard_prod(s_iso) // atom_elems
    t_grp, t_seps = t_iso.group([n_lane, n_mid_t, n_col])
    s_grp2, s_seps = s_iso.group([n_lane, n_mid_s, n_col])
    t_seps = list(t_seps)
    s_seps = list(s_seps)

    def _canon_segment(iters):
        return TileLayout.from_iters(iters, [], {}).canonicalize().shard

    t_lane = list(_canon_segment(list(t_grp.shard[t_seps[0] : t_seps[1]])))
    t_middle = list(_canon_segment(list(t_grp.shard[t_seps[1] : t_seps[2]])))
    t_col = list(_canon_segment(list(t_grp.shard[t_seps[2] : t_seps[3]])))
    s_lane = list(s_grp2.shard[s_seps[0] : s_seps[1]])
    s_middle = list(_canon_segment(list(s_grp2.shard[s_seps[1] : s_seps[2]])))
    s_col = list(_canon_segment(list(s_grp2.shard[s_seps[2] : s_seps[3]])))

    # F.5: align middles via union-cut sub-grouping. Both t_middle and s_middle
    # are post-canonicalize. To make their structure 1-1 we sub-group both by
    # the union of their internal cumulative-extent boundaries.
    t_middle, s_middle = _align_middles(t_middle, s_middle)

    # F.1: lane / col validation.
    if len(t_lane) != len(lane_pattern) or not all(
        int(li.extent) == e and int(li.stride) == st and li.axis == TLane
        for li, (e, st) in zip(t_lane, lane_pattern)
    ):
        raise ValueError(
            f"t_lane for tcgen05.cp {shape}{('.' + multicast) if multicast else ''} "
            f"must canonicalize to {[f'({e}, {st}@TLane)' for e, st in lane_pattern]}, "
            f"got {t_lane}"
        )
    if len(t_col) != 1:
        raise ValueError(f"t_col must canonicalize to single iter, got {t_col}")
    ci = t_col[0]
    if not (int(ci.extent) == elem_per_atom and int(ci.stride) == 1 and ci.axis == TCol):
        raise ValueError(f"t_col must be ({elem_per_atom}, 1@TCol), got {ci}")

    # F.2: s_lane → (atom_rows/8, 8) groups: atom_K = row stride within an 8-row
    # core matrix, SDO = stride between 8-row groups (single group: SDO 0).
    s_lane_layout = TileLayout.from_iters(s_lane, [], {})
    rows_per_group = min(atom_rows, 8)
    n_row_groups = atom_rows // rows_per_group
    if n_row_groups == 1:
        blk_rows = list(_canon_segment(s_lane))
        if len(blk_rows) != 1:
            raise ValueError(f"s_lane must canonicalize to a single row iter, got {blk_rows}")
        SDO_byte = 0
        atom_K_byte = int(blk_rows[0].stride) * dtype_bits // 8
    else:
        s_lane_grp, s_lane_seps = s_lane_layout.group([n_row_groups, rows_per_group])
        s_lane_seps = list(s_lane_seps)
        blk_grp = list(s_lane_grp.shard[s_lane_seps[0] : s_lane_seps[1]])
        blk_8 = list(s_lane_grp.shard[s_lane_seps[1] : s_lane_seps[2]])
        if len(blk_grp) != 1 or len(blk_8) != 1:
            raise ValueError(
                f"s_lane must group into single iter per ({n_row_groups}, {rows_per_group}) "
                f"block: blk_{n_row_groups}={blk_grp}, blk_8={blk_8}"
            )
        SDO_byte = int(blk_grp[0].stride) * dtype_bits // 8
        atom_K_byte = int(blk_8[0].stride) * dtype_bits // 8
        if SDO_byte % 16 != 0:
            raise ValueError(
                f"s_lane row-group stride {SDO_byte}B not 16B-aligned "
                "(descriptor SBO is encoded in 16B units)"
            )
    sw_candidates = {16: 0, 32: 1, 64: 2, 128: 3}
    if atom_K_byte not in sw_candidates:
        raise ValueError(f"atom_K_byte {atom_K_byte} not in {{16,32,64,128}}")
    derived_sw = sw_candidates[atom_K_byte]
    if s_swizzle_mode_from_layout != derived_sw:
        raise ValueError(
            f"swizzle mode mismatch: s_layout swizzle_len="
            f"{s_swizzle_mode_from_layout} but atom_K_byte={atom_K_byte} "
            f"implies sw={derived_sw}"
        )
    if s_swizzle_obj is not None:
        # The descriptor's address permutation assumes the canonical mma atom
        # family (128-bit atoms of 8 rows); other per_element/atom_len mis-plans.
        expected_per_element = (128 // dtype_bits).bit_length() - 1
        if (
            int(s_swizzle_obj.per_element) != expected_per_element
            or int(s_swizzle_obj.atom_len) != 3
        ):
            raise ValueError(
                "swizzle family mismatch: expected canonical mma atom "
                f"ComposeLayout(per_element={expected_per_element}, "
                f"swizzle_len={derived_sw}, atom_len=3); got "
                f"per_element={int(s_swizzle_obj.per_element)}, "
                f"atom_len={int(s_swizzle_obj.atom_len)}"
            )
        # The hardware walk is the swizzle_inner=True permutation; inner=False
        # would be mis-copied. sw==0: both directions identity, don't-care.
        if int(s_swizzle_obj.swizzle_len) > 0 and not bool(s_swizzle_obj.swizzle_inner):
            raise ValueError(
                "swizzle direction mismatch: the tcgen05.cp smem descriptor "
                "implements the swizzle_inner=True address permutation "
                "(x ^ ((x & outer_mask) >> atom_len)); got a ComposeLayout "
                f"with swizzle_inner=False (per_element="
                f"{int(s_swizzle_obj.per_element)}, swizzle_len="
                f"{int(s_swizzle_obj.swizzle_len)}, atom_len="
                f"{int(s_swizzle_obj.atom_len)}), whose mirrored permutation "
                "would be silently mis-copied"
            )

    # F.3: s_col = one cp's per-lane columns. 128b: one 16B unit (LDO=0). 256b:
    # two 16B units — adjacent when swizzled (LBO=1), else stride in LDO field.
    if atom_bits == 128:
        if len(s_col) != 1:
            raise ValueError(f"s_col must canonicalize to single iter, got {s_col}")
        sci = s_col[0]
        if not (int(sci.extent) == elem_per_128b and int(sci.stride) == 1):
            raise ValueError(f"s_col must be ({elem_per_128b}, 1, m), got {sci}")
        LDO_field = 0
    else:
        s_col_layout = TileLayout.from_iters(s_col, [], {})
        s_col_grp, s_col_seps = s_col_layout.group([2, elem_per_128b])
        s_col_seps = list(s_col_seps)
        blk_u = list(s_col_grp.shard[s_col_seps[0] : s_col_seps[1]])
        blk_e = list(s_col_grp.shard[s_col_seps[1] : s_col_seps[2]])
        if len(blk_u) != 1 or len(blk_e) != 1:
            raise ValueError(
                f"s_col must group into (2, {elem_per_128b}) 16B units: "
                f"units={blk_u}, elems={blk_e}"
            )
        if not (int(blk_e[0].extent) == elem_per_128b and int(blk_e[0].stride) == 1):
            raise ValueError(f"s_col 16B unit must be ({elem_per_128b}, 1, m), got {blk_e[0]}")
        ldo_byte = int(blk_u[0].stride) * dtype_bits // 8
        if ldo_byte % 16 != 0:
            raise ValueError(f"s_col unit stride {ldo_byte}B not 16B-aligned")
        if derived_sw == 0:
            LDO_field = ldo_byte // 16
        else:
            if ldo_byte != 16:
                raise ValueError(
                    f"swizzled {atom_bits}b atom requires the two 16B units "
                    f"adjacent in the linear layout, got stride {ldo_byte}B"
                )
            LDO_field = 1  # ignored by hardware for swizzled layouts (assumed 1)

    analyzer = Analyzer()

    # G: alignments.
    # G.1: tcgen05.cp taddr is 128-bit aligned. The layout offset is in
    # element units, so require four complete 32-bit TMEM cells.
    t_col_offset_expr = 0
    for ax, val in t_iso.offset.items():
        if ax == TCol:
            t_col_offset_expr = val
            break
    elem_per_128b_tmem = 4 * elem_per_32b
    if not analyzer.can_prove_equal(t_col_offset_expr % elem_per_128b_tmem, 0):
        raise ValueError(f"t TCol offset {t_col_offset_expr} not provably 128b-aligned")

    # G.1b: a region row offset folds into the taddr lane half-word ([31:16]).
    # The full lane footprint must still fit the 128-lane space.
    t_lane_offset_expr = 0
    for ax, val in t_iso.offset.items():
        if ax == TLane:
            t_lane_offset_expr = val
            break
    lane_span = 1
    for it in list(t_p.shard) + list(t_p.replica):
        if it.axis == TLane:
            lane_span += (int(it.extent) - 1) * int(it.stride)
    if not analyzer.can_prove(t_lane_offset_expr + lane_span <= 128):
        raise ValueError(
            f"t TLane offset {t_lane_offset_expr} overflows the 128-lane space "
            f"(footprint spans {lane_span} lanes)"
        )

    # G.2: s_iso m offset alignment.
    s_m_offset_expr = 0
    for ax, val in s_iso.offset.items():
        if ax == m_axis:
            s_m_offset_expr = val
            break
    elem_per_16B = 16 * 8 // dtype_bits
    if derived_sw == 0:
        align_elem = elem_per_16B
        align_label = "16B"
    else:
        atom_size_byte = 8 * atom_K_byte
        align_elem = atom_size_byte * 8 // dtype_bits
        align_label = f"atom={atom_size_byte}B"
    if not analyzer.can_prove_equal(s_m_offset_expr % align_elem, 0):
        raise ValueError(
            f"s offset {s_m_offset_expr} not provably aligned to {align_label} "
            f"({align_elem} {dtype} elements)"
        )

    # H: middle 1-1 correspondence.
    if len(t_middle) != len(s_middle):
        raise ValueError(
            f"t_middle iter count {len(t_middle)} != s_middle {len(s_middle)} "
            "(simple-mode requires 1-1)"
        )
    middle_iters = []
    for i, (ti, si) in enumerate(zip(t_middle, s_middle)):
        if int(ti.extent) != int(si.extent):
            raise ValueError(f"middle[{i}] extent: t={int(ti.extent)} s={int(si.extent)}")
        n = int(ti.extent)
        if n == 1:
            continue
        if ti.axis == TCol:
            if int(ti.stride) % elem_per_32b != 0:
                raise ValueError(
                    f"t_middle[{i}] stride {int(ti.stride)} elems not 32b-aligned "
                    f"(need multiple of {elem_per_32b})"
                )
            t_step = int(ti.stride) // elem_per_32b
        elif ti.axis == TLane:
            # Lane-tiled atoms (e.g. 4x256b filling each warp's first 16 rows
            # — the M=64 Layout-F scatter): step the taddr lane half-word.
            t_step = int(ti.stride) << 16
        else:
            raise ValueError(f"middle[{i}] t axis must be TCol or TLane, got {ti.axis}")
        s_stride_byte = int(si.stride) * dtype_bits // 8
        if s_stride_byte % 16 != 0:
            raise ValueError(f"s_middle[{i}] stride {s_stride_byte}B not 16B-aligned")
        middle_iters.append((n, s_stride_byte // 16, t_step))

    SDO_field = SDO_byte // 16
    init_off_16B = s_m_offset_expr * dtype_bits // 8 // 16
    t_addr_off = t_col_offset_expr // elem_per_32b
    if not (isinstance(t_lane_offset_expr, int) and t_lane_offset_expr == 0):
        t_addr_off = t_addr_off + t_lane_offset_expr * 65536

    return {
        "s_buf": s_buf,
        "t_buf": t_buf,
        "dtype": dtype,
        "dtype_bits": dtype_bits,
        "shape": shape,
        "multicast": multicast,
        "elem_per_atom": elem_per_atom,
        "elem_per_32b": elem_per_32b,
        "swizzle_mode": derived_sw,
        "LDO_field": LDO_field,
        "SDO_field": SDO_field,
        "atom_K_byte": atom_K_byte,
        "middle_iters": middle_iters,
        "init_off_16B": init_off_16B,
        "t_addr_off": t_addr_off,
    }


# -----------------------------------------------------------------------------
# Descriptor caching: one (smem_buf, ldo, sdo, swizzle) → one desc_buf,
# encoded once at SMEM base, hoisted to right after SMEM alloc via
# add_post_buffer_def_stmt.
# -----------------------------------------------------------------------------
def _get_or_create_desc(sctx, s_buf, ldo, sdo, swizzle):
    # Cache descriptor template at SMEM 0; patch addr per cp.
    cache_key = f"smem_tmem_desc:{int(ldo)}:{int(sdo)}:{int(swizzle)}"
    cached = sctx.cache_get(cache_key)
    if cached is not None:
        return cached

    desc_buf = tvm.tirx.decl_buffer((1,), "uint64", name="cp_desc", scope="local")
    encode_call = T.cuda.tcgen05.encode_matrix_descriptor(
        desc_buf.data, T.reinterpret("handle", T.uint64(0)), ldo, sdo, swizzle
    )
    wrap = SeqStmt([AllocBuffer(desc_buf), Evaluate(encode_call)])
    sctx.add_post_buffer_def_stmt(s_buf, wrap)
    sctx.cache_set(cache_key, desc_buf)
    return desc_buf


def _desc_set_addr(desc_val, addr_ptr):
    """Patch a SMEM matrix descriptor's 14-bit address field with cvta(addr)>>4 —
    matches the hand-rolled ``replace_smem_desc_addr`` (descriptor encoded at 0)."""
    start_addr = T.cast(
        T.bitwise_and(
            T.shift_right(T.cuda.cvta_generic_to_shared(addr_ptr), T.uint32(4)),
            T.uint32(0x3FFF),
        ),
        "uint64",
    )
    return T.bitwise_or(T.bitwise_and(desc_val, T.bitwise_not(T.uint64(0x3FFF))), start_addr)


def _validate_smem_tmem_copy(op_call: TilePrimitiveCall, sctx: DispatchContext):
    """Memory-scope envelope only; shape resolution/inference and the detailed
    layout validation raise readable ValueErrors in ``_build_plan``."""
    dst_region, src_region = op_call.args[:2]
    src: Buffer = src_region.buffer
    dst: Buffer = dst_region.buffer
    return (
        src.scope().startswith("shared")
        and dst.scope() == "tmem"
        and src.layout is not None
        and dst.layout is not None
        # Byte reinterpret is legal (nvfp4 stages sf as uint8, reads fp8); a
        # true width change needs decompress, rejected in copy_smem_tmem_impl.
        and DataType(src.dtype).bits == DataType(dst.dtype).bits
        and dst.allocated_addr is not None
    )


# -----------------------------------------------------------------------------
# Core impl: emits the cp loop given a plan + cp config. Async only — caller
# is responsible for issuing ``tcgen05.commit`` against a barrier if they
# need synchronization.
# -----------------------------------------------------------------------------
def copy_smem_tmem_impl(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc | None:
    if op_call.config.get("decompress"):
        # fp4/fp6->fp8 in-flight decompression needs dtype-pair plan derivation
        # the planner can't lower; reject loudly rather than copy undecompressed.
        raise ValueError("tcgen05.cp planner does not support decompress")
    # NOTE: descriptor templates use base_offset=0, so the smem buffer base must
    # align to the swizzle period (8 * atom_K bytes); align=1024 discharges this.
    plan = _build_plan(op_call)
    s_buf = plan["s_buf"]
    t_buf = plan["t_buf"]
    shape = plan["shape"]
    multicast = plan["multicast"]
    SDO_field = plan["SDO_field"]
    sw = plan["swizzle_mode"]
    middle_iters = plan["middle_iters"]
    init_off_16B = plan["init_off_16B"]
    t_addr_off = plan["t_addr_off"]

    # 128b atoms never read the descriptor LDO (single 16B unit per lane):
    # keep the legacy LDO=0 encoding. 256b atoms carry the derived field.
    LDO_field = plan["LDO_field"]

    cta_group = op_call.config.get("cta_group", 1)

    desc_buf = _get_or_create_desc(sctx, s_buf, LDO_field, SDO_field, sw)
    t_addr = t_buf.allocated_addr
    s_rank = len(s_buf.shape)

    def _cp_desc(off_16B):
        addr = T.ptr_byte_offset(s_buf.ptr_to([0] * s_rank), off_16B * 16, s_buf.dtype)
        return _desc_set_addr(desc_buf[0], addr)

    # Flatten the N-D middle loop into one T.unroll: per-dim index is
    # (flat // div) % extent, summed into the t/s offsets. total == 1 is
    # special-cased to avoid a degenerate T.unroll(1).
    total = functools.reduce(operator.mul, [n for n, _, _ in middle_iters], 1)

    # One instruction spelling for both impls: the ISA writes multicast
    # after the shape, and this planner never emits the decompression pair.
    cp_chain = f"tcgen05.cp.cta_group::{cta_group}.{shape}"
    if multicast:
        cp_chain += f".{multicast}"

    # fmt: off
    if total == 1:
        @T.prim_func(check_well_formed=False)
        def impl():
            T.ptx[cp_chain](T.cast(t_addr[0] + t_addr_off, "uint32"), _cp_desc(init_off_16B))
    else:
        def compute_offsets(flat):
            t_off = 0
            s_off = 0
            div = 1
            for n, s_step, t_step in middle_iters:
                idx = (flat // div) % n
                div = div * n
                t_off = t_off + idx * t_step
                s_off = s_off + idx * s_step
            return t_off, s_off

        @T.prim_func(check_well_formed=False)
        def impl():
            for flat in T.unroll(total):
                t_off, s_off = T.meta_var(compute_offsets(flat))
                T.ptx[cp_chain](
                    T.cast(t_addr[0] + t_addr_off + t_off, "uint32"),
                    _cp_desc(init_off_16B + s_off),
                )
    # fmt: on

    return impl


# === Variant: copy_async/smem->tmem (priority=10) ===
@register_dispatch(
    "copy_async",
    "cuda",
    variant="smem->tmem",
    priority=10,
    when=[
        predicate("validate_smem_tmem_copy", _validate_smem_tmem_copy),
        predicate("exec_scope", _single_thread_exec),
    ],
)
def copy_async_schedule_smem_tmem(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return copy_smem_tmem_impl(op_call, sctx)
