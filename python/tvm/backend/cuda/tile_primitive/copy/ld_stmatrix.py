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

"""copy dispatch variant: ldmatrix / stmatrix (TBD algorithm).

Handles register ↔ shared copies on CUDA via PTX ``ldmatrix`` / ``stmatrix``.
Direction (ld vs st) and exec scope (warp / warpgroup) are decided inside
``_emit`` from the src/dst scopes and ``sctx.scope_kind``.
"""

from math import prod

from tvm.script import tirx as T
from tvm.tirx import PrimFunc
from tvm.tirx.layout import ComposeLayout, S, TileLayout
from tvm.tirx.operator.tile_primitive.dispatcher import fail, predicate, register_dispatch
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.tile_primitive import TilePrimitiveCall

from ._common import (  # noqa: F401  (_carve_tail reserved for future variants)
    _carve_tail,
    _extract_tile,
)
from .utils import _is_valid_copy, _scope_allowed
from .vec_auto_reg import _all_threads_active, _ptr_off

_REG_SMEM_PAIRS = [
    ("local", "shared*"),
    ("shared*", "local"),
]

_VALID_R_LANE_AXES = {"laneid", "tid_in_wg", "tx"}


def _compute_r_perm(r):
    """Permutation: thread iters first (stride-desc), then memory iters (stride-desc)."""

    def key(p):
        it = p[1]
        return (0 if it.axis.is_thread() else 1, -int(it.stride))

    return [i for i, _ in sorted(enumerate(r.shard), key=key)]


def _is_ldstmatrix(op_call: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
    if not sctx.is_target("cuda"):
        return False, "non-cuda target"
    if sctx.scope_kind not in ("warp", "warpgroup", "cta"):
        return False, f"unsupported exec_scope {sctx.scope_kind} (need warp, warpgroup, or cta)"
    for check in (
        lambda: _all_threads_active(sctx),
        lambda: _is_valid_copy(op_call, sctx),
        lambda: _scope_allowed(op_call, sctx, allowed_pairs=_REG_SMEM_PAIRS),
    ):
        ok, msg = check()
        if not ok:
            return False, msg
    return True, None


def _emit(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    op_call = TilePrimitiveCall.downcast(op_call)

    # Step 1: identify reg / smem sides and pull their tensor shape + layout.
    src_br = op_call.src
    dst_br = op_call.dst
    if src_br.buffer.scope() == "local":
        r_br, s_br = src_br, dst_br
        direction = "st"  # reg -> smem (stmatrix)
    else:
        r_br, s_br = dst_br, src_br
        direction = "ld"  # smem -> reg (ldmatrix)
    r_buf = r_br.buffer
    s_buf = s_br.buffer
    r_shape = list(r_buf.shape)
    r_layout = r_buf.layout
    s_shape = list(s_buf.shape)
    s_layout = s_buf.layout

    r_region = [(r.min, r.min + r.extent) for r in r_br.region]
    s_region = [(r.min, r.min + r.extent) for r in s_br.region]
    with sctx.target:
        r_sliced = r_layout.slice(r_shape, r_region)
        s_sliced = s_layout.slice(s_shape, s_region)
        r = r_sliced.canonicalize()
        s = s_sliced.canonicalize()

    # Peel the S-side wrapper before structural checks, retaining its
    # parameters so the selected TileLayout can be wrapped again at emit.
    s_swizzle = s_buf.layout if isinstance(s_buf.layout, ComposeLayout) else None
    if s_swizzle is not None and s_swizzle.per_element < 3:
        # ldmatrix/stmatrix .b16 reads/writes 8 fp16 = 128b per lane in one
        # contiguous chunk. The swizzle preserves the lowest ``per_element``
        # bits of the address (in-chunk offset). For the per-lane 128b unit
        # to stay contiguous post-swizzle, ``2^per_element >= 8`` ⇒ p >= 3.
        fail(
            f"swizzle per_element={s_swizzle.per_element} < 3 incompatible "
            f"with .b16 ldmatrix/stmatrix (need 8-fp16 chunk integrity)"
        )
    s = _extract_tile(s, s_region)

    # Step 3: ldstmatrix doesn't broadcast — require zero replica on both sides.
    if len(r.replica) != 0:
        fail(f"R layout has replica {list(r.replica)}; ldstmatrix requires no replica")
    if len(s.replica) != 0:
        fail(f"S layout has replica {list(s.replica)}; ldstmatrix requires no replica")

    # Step 4: R must have exactly one kind of lane axis from the valid set.
    r_thread_axes = {it.axis.name for it in r.shard if it.axis.is_thread()}
    if len(r_thread_axes) != 1:
        fail(f"R must have exactly one thread axis name; got {sorted(r_thread_axes)}")
    r_lane_axis = next(iter(r_thread_axes))
    if r_lane_axis not in _VALID_R_LANE_AXES:
        fail(f"R thread axis {r_lane_axis!r} not in {sorted(_VALID_R_LANE_AXES)}")

    # Step 5: group S by R's iter extents (one S group per R iter, outer→inner).
    r_group_shape = [int(it.extent) for it in r.shard]
    s_grp, s_seps = s.group(r_group_shape)

    # Step 6: permute R so thread iters come first (stride-desc), then memory
    # iters (stride-desc).
    r_perm = _compute_r_perm(r)
    r = r.permute_dims(r_perm)

    # Step 7: apply R's perm to S in group units (1-to-1 with R's iters), and
    # rebuild s_seps to track group boundaries in the new order.
    s = s_grp.permute_by_groups(list(s_seps), r_perm)
    old_sizes = [s_seps[i + 1] - s_seps[i] for i in range(len(s_seps) - 1)]
    s_seps = [0]
    for pi in r_perm:
        s_seps.append(s_seps[-1] + old_sizes[pi])

    # Step 7.5: canonicalize both R and S after permute. Fuses adjacent
    # contig iters — keeps step 8's group input clean. Push target so
    # scope-aware fusers run (laneid+wid_in_wg → tid_in_wg, etc.).
    with sctx.target:
        r = r.canonicalize()
        s = s.canonicalize()

    t_total = prod(int(it.extent) for it in r.shard if it.axis.is_thread())
    m_total = prod(int(it.extent) for it in r.shard if not it.axis.is_thread())
    if t_total % 32 != 0:
        fail(f"R thread section total {t_total} not divisible by 32")

    def _strs(lay, seps):
        # Atoms 8 / 4 / 2 (segs 1, 2, 5) must be single iters — their strides
        # feed downstream stride checks (lane partition + fragment 2-fp16
        # contig). The num atom (seg 4) may be MULTI-ITER: we return its iter
        # list and let layout.apply handle the decomposition at emit time.
        fixed_segs = [list(lay.shard[seps[i] : seps[i + 1]]) for i in (1, 2, 5)]
        if not all(len(g) == 1 for g in fixed_segs):
            return None
        num_iters = list(lay.shard[seps[4] : seps[5]])
        return (
            int(fixed_segs[0][0].stride),  # 8 atom stride
            int(fixed_segs[1][0].stride),  # 4 atom stride
            num_iters,  # num atom iter list (multi-iter OK)
            int(fixed_segs[2][0].stride),  # 2 atom stride
        )

    def _try_num(r_in, s_in, num):
        """Try grouping (r_in, s_in) with [T/32, 8, 4, M/(2num), num, 2].

        Returns (rg, rsep, sg, ssep, trans, p, num) if structural checks pass,
        else None. ``trans`` is the ldmatrix .trans flag; ``p`` is the
        per-tile-row S stride used at emit.
        """
        gs = [t_total // 32, 8, 4, m_total // (num * 2), num, 2]
        try:
            rg, rsep = r_in.group(gs)
            sg, ssep = s_in.group(gs)
        except Exception:
            return None
        # R seg 0 (T/32 outer): require single iter with stride 32. When
        # T/32 == 1 the segment is trivial — skip.
        if t_total > 32:
            seg0 = list(rg.shard[rsep[0] : rsep[1]])
            if len(seg0) != 1 or int(seg0[0].stride) != 32:
                return None
        rs, ss = _strs(rg, rsep), _strs(sg, ssep)
        if rs is None or ss is None:
            return None
        r8, r4, _r_num_iters, r2 = rs
        s8, s4, s_num_iters, s2 = ss
        if (r8, r4, r2) != (4, 1, 1):
            return None
        # S num atom: every iter must have stride > 0 and multiple of 8 (the
        # per-tile spacing geometry of ldmatrix m8n8; 8 fp16 = 16 bytes = one
        # tile column dimension).
        if num > 1 and not all(
            int(it.stride) > 0 and int(it.stride) % 8 == 0 for it in s_num_iters
        ):
            return None
        # m_outer (seg 3) iters: each per-mm advance must keep the per-lane
        # SMEM address 16-byte aligned (ldmatrix .b16 reads 8 fp16 = 16 bytes
        # per lane), so the m_outer S-stride must also be a multiple of 8.
        # Without this, mm > 0 iterations land at unaligned addresses and
        # silently read garbage even though the layout group succeeds.
        # Skip extent-1 trivial iters — they contribute no per-mm advance,
        # so their (placeholder) stride is irrelevant.
        m_outer_iters = list(sg.shard[ssep[3] : ssep[4]])
        if not all(int(it.extent) == 1 or int(it.stride) % 8 == 0 for it in m_outer_iters):
            return None
        if (s4, s2) == (2, 1) and s8 > 0 and s8 % 8 == 0:
            return (rg, rsep, sg, ssep, False, s8, num)
        if s8 == 1 and s2 > 0 and s2 % 8 == 0 and s4 == 2 * s2:
            return (rg, rsep, sg, ssep, True, s2, num)
        return None

    # Try the **sorted** variant: 5D-group, sub-group R's M/2 by S's M/2
    # extents, sort the sub-groups by descending S-stride, rebuild. If
    # anything in the rebuild raises (e.g. M/2 can't be sub-grouped by S's
    # extents), silently fall back to the no-sort path below.
    r_sort = s_sort = None
    try:
        gs5 = [t_total // 32, 8, 4, m_total // 2, 2]
        rg5, rsep5 = r.group(gs5)
        sg5, ssep5 = s.group(gs5)
        r_m_iters = list(rg5.shard[rsep5[3] : rsep5[4]])
        s_m_iters = list(sg5.shard[ssep5[3] : ssep5[4]])
        s_m_extents = [int(it.extent) for it in s_m_iters]
        # Sub-group R's M/2 iters by S's M/2 iter extents. This 1-to-1's
        # the R sub-groups with the S iters so we can permute them together.
        r_m_sub = TileLayout.from_iters(r_m_iters)
        r_m_grouped, r_m_seps = r_m_sub.group(s_m_extents)
        # Sort S iters by S-stride descending; permute R sub-groups in lockstep.
        perm = sorted(range(len(s_m_iters)), key=lambda i: -int(s_m_iters[i].stride))
        if perm != list(range(len(perm))):
            r_m_permuted = r_m_grouped.permute_by_groups(list(r_m_seps), perm)
            s_m_permuted = [s_m_iters[i] for i in perm]
            r_sort = TileLayout.from_iters(
                list(rg5.shard[: rsep5[3]])
                + list(r_m_permuted.shard)
                + list(rg5.shard[rsep5[4] :]),
                offset=dict(rg5.offset),
            )
            s_sort = TileLayout.from_iters(
                list(sg5.shard[: ssep5[3]]) + list(s_m_permuted) + list(sg5.shard[ssep5[4] :]),
                offset=dict(sg5.offset),
            )
        # If perm is identity, sorted == unsorted; no need to build duplicate layouts.
    except Exception:
        r_sort = s_sort = None

    # Enumerate num largest-first; for each num try sorted then unsorted.
    chosen = None
    for num in (4, 2, 1):
        if m_total % (num * 2):
            continue
        if r_sort is not None:
            res = _try_num(r_sort, s_sort, num)
            if res is not None:
                chosen = res
                break
        res = _try_num(r, s, num)
        if res is not None:
            chosen = res
            break

    if chosen is None:
        fail("ldstmatrix layout doesn't fit any num ∈ {4,2,1}")
    r, r_seps, s, s_seps, trans, _row_stride, num = chosen

    # Order the num atom (seg 4) iters by R m-stride descending. The dst of
    # each ldmatrix/stmatrix must be 4 consecutive registers; when the seg-4
    # decomposition order matches R's fragment word order (largest m-stride
    # slowest), the frag words land in consecutive dst registers and ptxas
    # needs no register-shuffle MOVs. R and S are permuted in lockstep, so
    # the R↔S element pairing (and thus numerics) is unchanged.
    r_num_iters = list(r.shard[r_seps[4] : r_seps[5]])
    if len(r_num_iters) > 1:
        num_order = sorted(range(len(r_num_iters)), key=lambda i: -int(r_num_iters[i].stride))
        if num_order != list(range(len(r_num_iters))):
            r_iters = list(r.shard)
            s_iters = list(s.shard)
            r_seg = [r_iters[r_seps[4] : r_seps[5]][i] for i in num_order]
            s_seg = [s_iters[s_seps[4] : s_seps[5]][i] for i in num_order]
            r = TileLayout.from_iters(
                r_iters[: r_seps[4]] + r_seg + r_iters[r_seps[5] :],
                offset=dict(r.offset),
            )
            s = TileLayout.from_iters(
                s_iters[: s_seps[4]] + s_seg + s_iters[s_seps[5] :],
                offset=dict(s.offset),
            )

    # Step 10: emit one ldmatrix/stmatrix per mm, per warp.

    def _get_warp_idx_in_T():
        # T.warp_id_in_wg() / T.warp_id() must be called from inside a
        # @T.prim_func body — wrap so the prim_func parser calls us at parse
        # time (Python `if` here is plain control flow, not TIR-intercepted).
        if r_lane_axis == "laneid":
            return 0
        if r_lane_axis == "tid_in_wg":
            return T.warp_id_in_wg()
        return T.warp_id()  # "tx"

    def _seg4_coord(laneid_expr):
        # num=1: seg 4 trivially extent-1, pass 0. num>1: use lane//8 (tile
        # index in ldmatrix lane convention); layout.apply decomposes through
        # the seg's iter structure (single or multi-iter).
        if num > 1:
            return laneid_expr // 8
        return 0

    # Built out here: binding a Python string inside the traced body is not
    # something the parser can carry.
    _trans_seg = ".trans" if trans else ""
    ld_chain = f"ldmatrix.sync.aligned.m8n8.x{num}{_trans_seg}.shared.b16"
    st_chain = f"stmatrix.sync.aligned.m8n8.x{num}{_trans_seg}.shared.b16"

    apply_shape = [t_total // 32, 8, 4, m_total // (num * 2), num, 2]
    r_mem_axis = r.shard[r_seps[5]].axis.name
    s_mem_axis = s.shard[s_seps[5]].axis.name
    s_mem_axis_obj = s.shard[s_seps[5]].axis
    m_outer = m_total // (num * 2)
    s_zero = [0] * len(s_buf.shape)

    def _apply_s_layout(warp_idx, lane_idx, mm_idx):
        # The PTX lane-row contribution is not always one of the grouped S
        # coordinates (the transposed form takes its stride from another
        # group). Add it to the TileLayout offset so ComposeLayout sees the
        # complete structured address and can prove its bounded low part.
        offset = dict(s.offset)
        row_offset = (lane_idx % 8) * _row_stride
        offset[s_mem_axis_obj] = offset.get(s_mem_axis_obj, 0) + row_offset
        row_tile = TileLayout.from_iters(list(s.shard), list(s.replica), offset)
        row_layout = row_tile
        if s_swizzle is not None:
            row_layout = ComposeLayout(
                s_swizzle.per_element,
                s_swizzle.swizzle_len,
                s_swizzle.atom_len,
                row_tile,
                s_swizzle.swizzle_inner,
            )
        return row_layout.apply(
            warp_idx,
            0,
            0,
            mm_idx,
            _seg4_coord(lane_idx),
            0,
            shape=apply_shape,
        )[s_mem_axis]

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def impl():
        r_local = r_buf.local(m_total, layout=TileLayout(S[(m_total,)]))
        laneid = T.lane_id()
        warp_idx_in_T = _get_warp_idx_in_T()
        for mm in T.unroll(m_outer):
            smem_off = _apply_s_layout(warp_idx_in_T, laneid, mm)
            smem_ptr = _ptr_off(s_buf.ptr_to(s_zero), smem_off)
            # Both instructions move num b32 registers; the fragment buffer is
            # b16, so they land through a uint32 view, two elements per word
            # (the tcgen05_ldst 16-bit pattern).
            r_words = r_local.view("uint32")
            words = [
                r_words[r.apply(0, 0, 0, mm, i, 0, shape=apply_shape)[r_mem_axis] // 2]
                for i in range(num)
            ]
            if direction == "ld":
                T.ptxd[ld_chain](*words, smem_ptr)
            else:
                # stmatrix reverses ldmatrix's operand order: address first.
                T.ptxd[st_chain](smem_ptr, *words)
    # fmt: on
    return impl


@register_dispatch(
    "copy",
    "cuda",
    variant="ldstmatrix",
    priority=10,
    when=[predicate("ldstmatrix_applicable", _is_ldstmatrix)],
)
def copy_schedule_ldstmatrix(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return _emit(op_call, sctx)


__all__ = ["copy_schedule_ldstmatrix"]
