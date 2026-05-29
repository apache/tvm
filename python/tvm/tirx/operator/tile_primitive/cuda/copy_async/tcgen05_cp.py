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

"""smem->tmem dispatch via tcgen05.cp.32x128b.warpx4.

``tcgen05.cp`` is inherently async; this dispatch emits the cp loop only and
leaves completion signaling (``tcgen05.commit`` against a barrier) to the
caller. Callers who want sync semantics should issue ``tcgen05.commit``
themselves after the copy.

Algorithm
---------
Given ``Tx.copy_async(t_region, s_region)`` where t is in tmem (with
R[4:32@TLane] indicating warpx4 broadcast), and s is in shared memory:

A. Slice + canonicalize both layouts at the given regions.
B. Verify ``t.replica == [4:32@TLane]`` (warpx4 router).
C. Compute permutation that puts TLane first, then TCol stride-descending;
   apply to t.permute_dims and to s via group + permute_by_groups.
D. Canonicalize again.
E. Isolate broadcast: split-by-stride-zero on both t and s; their split
   sequences must match (same distinct prefix prods + broadcast extents).
   Drop stride-0 iters → ``t_iso`` and ``s_iso``.
F. Group both into ``(32, middle, elem_per_128b)``. Validate:
   - t_lane = (32, 1@TLane)
   - t_col = (elem_per_128b, 1@TCol)
   - s_col = (elem_per_128b, 1)
   - s_lane refines into (4, 8) on m axis with strides (SDO_stride, atom_K_stride)
   - atom_K_byte ∈ {16, 32, 64, 128} → swizzle_mode 0..3
   - swizzle_mode matches s_buf.layout's SwizzleLayout (if any)
G. Alignment checks:
   - t_iso TCol offset ≡ 0 (mod 32-bit)
   - s_iso m offset ≡ 0 (mod 16B for sw=0; mod atom_size for sw>0)
   - middle iter strides 16B-aligned
H. middle 1-1 correspondence (simple-mode): t_middle and s_middle have same
   iter count and matching extents per position.
I. Emit:
   - SmemDescriptor encoded once at SMEM base (hoisted via post_buffer_def_stmt).
   - Loop over middle iters; each cp uses ``desc.add_16B_offset(init + loop)``
     and writes to ``tmem_addr + t_col0 + Σ i_j * t_step_j``.
"""

import functools
import operator

import tvm
from tvm.arith import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.layout import ComposeLayout, SwizzleLayout, TCol, TileLayout, TLane
from tvm.tirx.layout import m as m_axis
from tvm.tirx.operator.tile_primitive import DispatchContext, predicate, register_dispatch
from tvm.tirx.stmt import AllocBuffer, Evaluate, SeqStmt, TilePrimitiveCall

from ..copy import _is_valid_smem_tmem_copy, _single_thread_exec


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
def _build_plan(op_call: TilePrimitiveCall, sctx: DispatchContext):
    """Run A..H and return a dispatch plan.

    Plan fields:
      - s_buf, t_buf
      - dtype, dtype_bits
      - elem_per_128b, elem_per_32b
      - SmemSwizzleMode (int)
      - SDO_field, atom_K_byte
      - middle_iters: list of (extent, s_step_16B, t_step_32bcol)
      - init_off_16B (PrimExpr)
      - t_col0 (PrimExpr, TMEM 32-bit col offset for cp's first call)
    """
    op_call = TilePrimitiveCall.downcast(op_call)
    dst_region, src_region = op_call.args[:2]
    s_buf: Buffer = src_region.buffer
    t_buf: Buffer = dst_region.buffer
    dtype = s_buf.dtype
    dtype_bits = DataType(dtype).bits
    elem_per_128b = 128 // dtype_bits
    elem_per_32b = 32 // dtype_bits

    # C: slice + canonicalize.
    s_region = [(r.min, r.min + r.extent) for r in src_region.region]
    t_region = [(r.min, r.min + r.extent) for r in dst_region.region]
    s = s_buf.layout.slice(list(s_buf.shape), s_region).canonicalize()
    t = t_buf.layout.slice(list(t_buf.shape), t_region).canonicalize()

    # If s is ComposeLayout (SwizzleLayout∘TileLayout), peel off the swizzle
    # for stride analysis; record swizzle_len for cross-check.
    s_swizzle_mode_from_layout = 0
    if isinstance(s, ComposeLayout):
        s_swizzle_mode_from_layout = int(s.swizzle.swizzle_len)
        s = s.tile_layout
    elif isinstance(s, SwizzleLayout):
        raise ValueError("s slice produced bare SwizzleLayout (unexpected)")

    # B: warpx4 router check.
    rep = t.replica
    if not (
        len(rep) == 1
        and int(rep[0].extent) == 4
        and int(rep[0].stride) == 32
        and rep[0].axis == TLane
    ):
        raise ValueError(
            f"warpx4 router fail: t.replica = "
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

    # F: group into (32, middle, elem_per_128b).
    def shard_prod(lay):
        return functools.reduce(operator.mul, [int(it.extent) for it in lay.shard], 1)

    n_lane, n_col = 32, elem_per_128b
    n_mid_t = shard_prod(t_iso) // (n_lane * n_col)
    n_mid_s = shard_prod(s_iso) // (n_lane * n_col)
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
    if len(t_lane) != 1:
        raise ValueError(f"t_lane must canonicalize to single iter, got {t_lane}")
    if len(t_col) != 1:
        raise ValueError(f"t_col must canonicalize to single iter, got {t_col}")
    if len(s_col) != 1:
        raise ValueError(f"s_col must canonicalize to single iter, got {s_col}")
    li = t_lane[0]
    if not (int(li.extent) == 32 and int(li.stride) == 1 and li.axis == TLane):
        raise ValueError(f"t_lane must be (32, 1@TLane), got {li}")
    ci = t_col[0]
    if not (int(ci.extent) == elem_per_128b and int(ci.stride) == 1 and ci.axis == TCol):
        raise ValueError(f"t_col must be ({elem_per_128b}, 1@TCol), got {ci}")
    sci = s_col[0]
    if not (int(sci.extent) == elem_per_128b and int(sci.stride) == 1):
        raise ValueError(f"s_col must be ({elem_per_128b}, 1, m), got {sci}")

    # F.2: s_lane → group (4, 8) → (SDO_stride, atom_K_stride)
    s_lane_layout = TileLayout.from_iters(s_lane, [], {})
    s_lane_grp, s_lane_seps = s_lane_layout.group([4, 8])
    s_lane_seps = list(s_lane_seps)
    blk_4 = list(s_lane_grp.shard[s_lane_seps[0] : s_lane_seps[1]])
    blk_8 = list(s_lane_grp.shard[s_lane_seps[1] : s_lane_seps[2]])
    if len(blk_4) != 1 or len(blk_8) != 1:
        raise ValueError(
            f"s_lane must group into single iter per block: blk_4={blk_4}, blk_8={blk_8}"
        )
    SDO_byte = int(blk_4[0].stride) * dtype_bits // 8
    atom_K_byte = int(blk_8[0].stride) * dtype_bits // 8
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

    analyzer = Analyzer()

    # G: alignments.
    # G.1: t_iso TCol offset ≡ 0 (mod 32-bit element count).
    t_col_offset_expr = 0
    for ax, val in t_iso.offset.items():
        if ax == TCol:
            t_col_offset_expr = val
            break
    if not analyzer.can_prove_equal(t_col_offset_expr % elem_per_32b, 0):
        raise ValueError(f"t TCol offset {t_col_offset_expr} not provably 32b-aligned")

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
        if ti.axis != TCol:
            raise ValueError(f"middle[{i}] t axis must be TCol, got {ti.axis}")
        s_stride_byte = int(si.stride) * dtype_bits // 8
        if s_stride_byte % 16 != 0:
            raise ValueError(f"s_middle[{i}] stride {s_stride_byte}B not 16B-aligned")
        middle_iters.append((n, s_stride_byte // 16, int(ti.stride) // elem_per_32b))

    SDO_field = SDO_byte // 16
    init_off_16B = s_m_offset_expr * dtype_bits // 8 // 16
    t_col0 = t_col_offset_expr // elem_per_32b

    return {
        "s_buf": s_buf,
        "t_buf": t_buf,
        "dtype": dtype,
        "dtype_bits": dtype_bits,
        "elem_per_128b": elem_per_128b,
        "elem_per_32b": elem_per_32b,
        "swizzle_mode": derived_sw,
        "SDO_field": SDO_field,
        "atom_K_byte": atom_K_byte,
        "middle_iters": middle_iters,
        "init_off_16B": init_off_16B,
        "t_col0": t_col0,
    }


# -----------------------------------------------------------------------------
# Descriptor caching: one (smem_buf, ldo, sdo, swizzle) → one desc_buf,
# encoded once at SMEM base, hoisted to right after SMEM alloc via
# add_post_buffer_def_stmt.
# -----------------------------------------------------------------------------
def _get_or_create_desc(sctx, s_buf, ldo, sdo, swizzle):
    cache_key = f"smem_tmem_desc:{hash(s_buf)}:{int(ldo)}:{int(sdo)}:{int(swizzle)}"
    cached = sctx.cache_get(cache_key)
    if cached is not None:
        return cached

    desc_buf = tvm.tirx.decl_buffer((1,), "uint64", name="cp_desc", scope="local")
    encode_call = Tx.ptx.tcgen05.encode_matrix_descriptor(
        desc_buf.data, s_buf.ptr_to([0] * len(s_buf.shape)), ldo, sdo, swizzle
    )
    wrap = SeqStmt([AllocBuffer(desc_buf), Evaluate(encode_call)])
    sctx.add_post_buffer_def_stmt(s_buf, wrap)
    sctx.cache_set(cache_key, desc_buf)
    return desc_buf


# -----------------------------------------------------------------------------
# Core impl: emits the cp loop given a plan + cp config. Async only — caller
# is responsible for issuing ``tcgen05.commit`` against a barrier if they
# need synchronization.
# -----------------------------------------------------------------------------
def copy_smem_tmem_impl(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc | None:
    plan = _build_plan(op_call, sctx)
    s_buf = plan["s_buf"]
    t_buf = plan["t_buf"]
    SDO_field = plan["SDO_field"]
    sw = plan["swizzle_mode"]
    middle_iters = plan["middle_iters"]
    init_off_16B = plan["init_off_16B"]
    t_col0 = plan["t_col0"]

    LDO_field = 16  # cp 32x128b ignores LDO; placeholder

    cta_group = op_call.config.get("cta_group", 1)

    desc_buf = _get_or_create_desc(sctx, s_buf, LDO_field, SDO_field, sw)
    t_addr = t_buf.allocated_addr
    from tvm.tirx.operator.tile_primitive.cuda.common import smem_desc_add_16B_offset

    # Flatten the N-D middle iteration into a single Tx.unroll. Each iteration's
    # per-dim index is (flat // stride) % extent, summed into the t/s offsets.
    # Works uniformly for n_mid ∈ {0, 1, 2, ...}; total == 1 (no middle dims) is
    # special-cased to avoid a degenerate Tx.unroll(1).
    total = functools.reduce(operator.mul, [n for n, _, _ in middle_iters], 1)

    # fmt: off
    if total == 1:
        @Tx.prim_func(check_well_formed=False)
        def impl():
            Tx.ptx.tcgen05.cp(
                t_addr[0] + t_col0,
                smem_desc_add_16B_offset(desc_buf[0], init_off_16B),
                shape="32x128b", cta_group=cta_group, multicast="warpx4",
            )
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

        @Tx.prim_func(check_well_formed=False)
        def impl():
            for flat in Tx.unroll(total):
                t_off, s_off = Tx.meta_var(compute_offsets(flat))
                Tx.ptx.tcgen05.cp(
                    t_addr[0] + t_col0 + t_off,
                    smem_desc_add_16B_offset(desc_buf[0], init_off_16B + s_off),
                    shape="32x128b", cta_group=cta_group, multicast="warpx4",
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
        predicate("validate_smem_tmem_copy", _is_valid_smem_tmem_copy),
        predicate("exec_scope", _single_thread_exec),
    ],
)
def copy_async_schedule_smem_tmem(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return copy_smem_tmem_impl(op_call, sctx)
