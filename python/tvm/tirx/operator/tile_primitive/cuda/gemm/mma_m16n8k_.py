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

"""Warp-level ``mma.sync`` GEMM lowering for the synchronous ``gemm`` op on CUDA."""

from dataclasses import dataclass

from tvm.arith.analyzer import Analyzer
from tvm.script import tirx as T
from tvm.tirx import PrimFunc
from tvm.tirx.layout import TileLayout
from tvm.tirx.operator.tile_primitive import (
    DispatchContext,
    fail,
    predicate,
    register_dispatch,
)
from tvm.tirx.stmt import TilePrimitiveCall


@dataclass(frozen=True)
class MmaInst:
    """One concrete mma.sync instruction we can emit. A given shape appears once
    per dtype signature. Adding an instruction is just adding an entry to
    MMA_INSTRUCTIONS; the feasibility checks below stay generic. The lane->coord
    mapping is the fixed m16n8 family structure (hardcoded in the match); the
    only per-instruction fragment detail is ``k_pack`` (it is not always
    32/dtype_bits, so it is stored explicitly rather than derived)."""

    name: str  # distinguishing tag, e.g. "m16n8k16.bf16"
    m: int
    n: int
    k: int
    dtype: tuple[str, str, str, str]  # the single (A, B, C, D) dtype signature
    k_pack: int  # contiguous-along-K elements packed into one register for A and B


# One entry per concrete instruction. The m16n8 family fixes M/N at 16/8; K is
# 16 or 8. f16/bf16 inputs with f32 accumulation, packing 2 bf16/f16 per b32
# along K (f16 accumulation, int8, fp8, etc. are added as more entries).
_BF16_F32 = ("bfloat16", "bfloat16", "float32", "float32")
_F16_F32 = ("float16", "float16", "float32", "float32")
MMA_INSTRUCTIONS = (
    MmaInst("m16n8k16.bf16", 16, 8, 16, _BF16_F32, k_pack=2),
    MmaInst("m16n8k16.f16", 16, 8, 16, _F16_F32, k_pack=2),
    MmaInst("m16n8k8.bf16", 16, 8, 8, _BF16_F32, k_pack=2),
    MmaInst("m16n8k8.f16", 16, 8, 8, _F16_F32, k_pack=2),
)


def _split(grouped, seps):
    """Split a grouped layout into one shard-only sub-layout per group, plus the
    layout's offset (returned separately rather than distributed into the subs)."""
    subs = [
        TileLayout.from_iters(list(grouped.shard[seps[g] : seps[g + 1]]), [], {})
        for g in range(len(seps) - 1)
    ]
    return subs, dict(grouped.offset)


def _combine(layouts, offset):
    """Concatenate sub-layouts' shards and apply the separately tracked offset."""
    shard = [it for lay in layouts for it in lay.shard]
    return TileLayout.from_iters(shard, [], offset)


def _canon_perm(iters):
    """Permutation putting thread iters left, memory iters right; stride-desc within each."""
    thr = sorted(
        (i for i, it in enumerate(iters) if it.axis.is_thread()),
        key=lambda i: -int(iters[i].stride),
    )
    mem = sorted(
        (i for i, it in enumerate(iters) if not it.axis.is_thread()),
        key=lambda i: -int(iters[i].stride),
    )
    return thr + mem


def _canon(layout):
    """Reorder an anchor sub-layout into canonical (thread-left/memory-right) order."""
    return layout.permute_dims(_canon_perm(list(layout.shard)))


def _align(anchor, follower, name):
    """Regroup follower by anchor's iter extents, then permute its groups to follow
    the anchor's canonical order (follower keeps its own strides)."""
    extents = [int(it.extent) for it in anchor.shard]
    perm = _canon_perm(list(anchor.shard))
    try:
        grp, seps = follower.group(extents)
    except Exception as e:  # follower dim layout incompatible with anchor decomposition
        fail(f"gemm mma: {name} not alignable to anchor extents {extents}: {e}")
    return grp.permute_by_groups(seps, perm)


def _region_totals(layout):
    """(product of thread-axis iter extents, product of memory-axis iter extents).

    Computed once on each anchor to fix that dim's thread/memory region lengths;
    followers reuse the anchor's split rather than re-deriving it from their own
    iters (which can misclassify -- e.g. B's N, whose register slot is actually a
    lane iter, would otherwise report memory length 1)."""
    thr, mem = 1, 1
    for it in layout.shard:
        if it.axis.is_thread():
            thr *= int(it.extent)
        else:
            mem *= int(it.extent)
    return thr, mem


def _frag_group(layout, lane, mem, thread_total, mem_total):
    """Group one logical-dim sub-layout into the fragment shape, then optionally
    verify it.

    ``lane`` and ``mem`` are lists of ``(extent, stride, want_thread)``;
    ``thread_total`` and ``mem_total`` are this dim's thread/memory region lengths
    (from its anchor via _region_totals). The group shape is
    ``[thread_total // prod(lane), *lane, mem_total // prod(mem), *mem]``: the lane
    extents are carved off the thread region and the mem extents off the memory
    region (innermost last, e.g. ``[(reg, ...)]`` for an accumulator dim or
    ``[(kHi, ...), (k_pack, ...)]`` for A/B's K). The input must already be in
    canonical (thread-left/memory-right) order.

    Every carved group is verified: it must be a single iter, its axis must be a
    thread axis when ``want_thread`` else a memory axis (only is_thread is checked,
    since scope varies the exact thread axis; e.g. B's N register slot is actually
    a lane, so want_thread=True there), and a non-None ``stride`` pins that iter's
    stride. Raises on a tiling or verification failure, so a non-matching caller
    layout is declined via the caller's try/except.
    """
    lane_ext = [e for e, _, _ in lane]
    mem_ext = [e for e, _, _ in mem]
    lane_prod, mem_prod = 1, 1
    for e in lane_ext:
        lane_prod *= e
    for e in mem_ext:
        mem_prod *= e
    grouped, seps = layout.group(
        [thread_total // lane_prod, *lane_ext, mem_total // mem_prod, *mem_ext]
    )
    # group order: [thread_rest, *lane (from idx 1), mem_rest, *mem (after)].
    specs = [(1 + j, s, t) for j, (_, s, t) in enumerate(lane)]
    specs += [(2 + len(lane) + j, s, t) for j, (_, s, t) in enumerate(mem)]
    for idx, stride, want_thread in specs:
        grp = grouped.shard[seps[idx] : seps[idx + 1]]
        if len(grp) != 1:
            raise ValueError(f"frag group {idx} is not a single iter")
        if int(grp[0].extent) == 1:
            # An extent-1 group iterates nothing, so its axis/stride is
            # meaningless and gets dropped downstream (cf. _same_iters /
            # _reg_layout). This is the kHi == 1 case of m16n8k8: there is a
            # single high-K register group, which .group() may materialize as a
            # degenerate split of the (thread) lane axis.
            continue
        if grp[0].axis.is_thread() != want_thread:
            raise ValueError(f"frag group {idx} thread/memory axis mismatch")
        if stride is not None and int(grp[0].stride) != stride:
            raise ValueError(f"frag group {idx} stride {int(grp[0].stride)} != {stride}")
    return grouped, seps


def _grp(grouped, seps, i):
    """The iters of group ``i`` of a grouped layout: ``shard[seps[i]:seps[i+1]]``."""
    return grouped.shard[seps[i] : seps[i + 1]]


def _ext(iters):
    """Product of the extents of an iter list."""
    p = 1
    for it in iters:
        p *= int(it.extent)
    return p


def _reg_layout(groups, offset):
    """Per-thread register layout from per-logical-dim iter groups (dropping
    thread-axis offset terms), plus the matching local-view shape (each dim is
    the product of that group's extents).

    Extent-1 iters are dropped from the layout: they iterate nothing (offset
    always 0, so harmless to the mapping) but would otherwise pin a degenerate
    axis -- e.g. B's N has no real register, so its "register" slot is a single
    extent-1 lane iter that must not make the register buffer thread-axis."""
    iters = [it for g in groups for it in g if int(it.extent) != 1]
    layout = TileLayout.from_iters(
        iters, [], {ax: v for ax, v in offset.items() if not ax.is_thread()}
    )
    return layout, [_ext(g) for g in groups]


def _same_iters(a, b):
    """True iff iter lists ``a`` and ``b`` match elementwise on (extent, stride,
    axis), ignoring extent-1 iters (they iterate nothing, so their stride/axis is
    meaningless -- e.g. the degenerate thread-rest left by a group shape's '1')."""
    a = [it for it in a if int(it.extent) != 1]
    b = [it for it in b if int(it.extent) != 1]
    if len(a) != len(b):
        return False
    return all(
        int(x.extent) == int(y.extent)
        and int(x.stride) == int(y.stride)
        and x.axis.name == y.axis.name
        for x, y in zip(a, b, strict=True)
    )


def _full_active_lanes(op: TilePrimitiveCall, sctx: DispatchContext):
    """The active thread set (sctx.intra) must be complete and un-narrowed.

    mma.sync.aligned is collective over every active thread; an enclosing if
    that narrows any intra axis makes the .aligned instruction undefined. So
    each intra axis must be at offset 0 with its full extent: laneid=32,
    wid_in_wg=4 (warpgroup), and warpid=warps-per-CTA (cta) from the launch
    config. Any other axis (e.g. cta_id at cluster scope) is not supported.
    """
    full = {"laneid": 32, "wid_in_wg": 4}
    if "warpid" in sctx.intra:
        tx = sctx.launch_params.get("threadIdx.x")
        if tx is None:
            return False, "cta scope needs threadIdx.x in launch_params"
        try:
            full["warpid"] = int(tx.dom.extent) // 32
        except (TypeError, ValueError):
            return False, f"non-static threadIdx.x extent {tx.dom.extent}"
    for axis, rng in sctx.intra.items():
        if axis not in full:
            return False, f"unsupported active-set axis {axis!r}"
        extent, offset = int(rng[0]), int(rng[1])
        if extent != full[axis] or offset != 0:
            return False, (
                f"active {axis} is [{offset}, {offset + extent}), need full [0, {full[axis]})"
            )
    return True


def _no_replica(op: TilePrimitiveCall, sctx: DispatchContext):
    """All operand layouts must have no replica (no broadcast/duplicated axes)."""
    for region, name in zip(op.args[:4], ("D", "A", "B", "C")):
        if region.buffer.layout.replica:
            return False, f"{name} layout has replica {region.buffer.layout.replica}"
    return True


@register_dispatch(
    "gemm",
    "cuda",
    variant="mma.m16n8k*",
    priority=10,
    when=[
        predicate("full_active_lanes", _full_active_lanes),
        predicate("no_replica", _no_replica),
    ],
)
def gemm_cuda_mma_dispatch(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    """``gemm`` -> warp-level ``mma.sync`` of the m16n8k* family.

    This is the ``"mma.m16n8k*"`` variant. It targets the m16n8k* tensor-core
    instructions -- currently m16n8k16 / m16n8k8 with bf16/f16 inputs and f32
    accumulation (see MMA_INSTRUCTIONS); other K (and other shapes/dtypes) are
    added as more entries. Pure-register path: A/B fragments and C/D
    accumulators all live in registers.
    """
    # gemm op args: D = alpha * A @ B + beta * C
    # D (args[0]) is the output; C (args[3]) is the beta-accumulator input.
    D_region, A_region, B_region, C_region, transpose_A, transpose_B, alpha, beta = op.args
    D, A, B, C = D_region.buffer, A_region.buffer, B_region.buffer, C_region.buffer

    # Pure-register mma path: A/B fragments and C/D accumulators all live in
    # registers ("local"). The caller is responsible for staging A/B into
    # registers (e.g. via ldmatrix) beforehand.
    for buf, name in ((D, "D"), (A, "A"), (B, "B"), (C, "C")):
        if buf.scope() != "local":
            fail(f"gemm mma requires {name} in register (local) scope, got {buf.scope()}")

    # transpose_A/transpose_B only describe the input's logical orientation; we
    # normalize to the standard form A=[M,K], B=[K,N] (D/C are always [M,N]).
    #   transpose_A: False -> buffer is [M,K], True -> [K,M]
    #   transpose_B: False -> buffer is [K,N], True -> [N,K]
    # The .row.col K-major requirement is not enforced here -- it is checked
    # later by the per-instruction fragment match against the real layout.
    analyzer = Analyzer()

    # mma.sync computes D = A·B + C natively (no scalar scaling), so we support
    # only alpha=1 and beta in {0, 1}; beta selects whether C is the accumulator
    # (1 -> c_ptr=C, 0 -> c_ptr=0). General alpha/beta is declined.
    def _const_scalar(expr):
        s = analyzer.simplify(expr)
        try:
            return float(s.value)
        except (AttributeError, TypeError, ValueError):
            return None

    if _const_scalar(alpha) != 1.0:
        fail(f"gemm mma supports only alpha=1, got alpha={alpha}")
    if _const_scalar(beta) not in (0.0, 1.0):
        fail(f"gemm mma supports only beta in {{0, 1}}, got beta={beta}")

    def _mat_extents(region, name):
        ext = [r.extent for r in region.region if not analyzer.can_prove_equal(r.extent, 1)]
        if len(ext) != 2:
            fail(f"gemm mma expects 2D {name}, got non-unit extents {ext}")
        return ext

    A_ext = _mat_extents(A_region, "A")
    B_ext = _mat_extents(B_region, "B")
    D_M, D_N = _mat_extents(D_region, "D")
    C_M, C_N = _mat_extents(C_region, "C")
    M, K = (A_ext[1], A_ext[0]) if transpose_A else (A_ext[0], A_ext[1])
    B_K, N = (B_ext[1], B_ext[0]) if transpose_B else (B_ext[0], B_ext[1])
    assert analyzer.can_prove_equal(B_K, K), f"gemm mma: A K={K} != B K={B_K}"
    assert analyzer.can_prove_equal(D_M, M) and analyzer.can_prove_equal(D_N, N), (
        f"gemm mma: D dims ({D_M}, {D_N}) != (M={M}, N={N})"
    )
    assert analyzer.can_prove_equal(C_M, M) and analyzer.can_prove_equal(C_N, N), (
        f"gemm mma: C dims ({C_M}, {C_N}) != (M={M}, N={N})"
    )

    # Tiling into instructions needs static extents.
    def _const(expr, name):
        try:
            return int(analyzer.simplify(expr))
        except (TypeError, ValueError):
            fail(f"gemm mma needs static {name} extent, got {expr}")

    M, N, K = _const(M, "M"), _const(N, "N"), _const(K, "K")

    # Slice each operand's layout to its region, then group it into its 2D
    # buffer-order shape; the split below maps those groups to standard
    # (M,K)/(K,N). group() raises if the layout can't be tiled that way, so a
    # caller layout that doesn't match the operand shape is declined cleanly.
    def _slice_group(buf, region, shape2d, name):
        # slice() itself groups internally, so both slice and group can raise the
        # ICHECK when the layout can't be tiled as shape2d -- guard both.
        canon = None
        try:
            sliced = buf.layout.slice(buf.shape, region.region)
            if sliced is not None:
                canon = sliced.canonicalize()
        except Exception as e:  # ICHECK failure -> layout not tileable as shape2d
            fail(f"gemm mma: {name} layout not tileable as {tuple(shape2d)}: {e}")
        if canon is None:
            fail(f"gemm mma: cannot slice {name} layout to its region")
        # All thread iters must share one thread axis (e.g. all laneid). _frag_group
        # only checks is_thread (not the exact axis, since scope varies), so a layout
        # mixing two thread axes (e.g. laneid + wid_in_wg) would carve ambiguously --
        # decline it here, like ldmatrix validating its thread structure.
        thread_axes = {it.axis.name for it in canon.shard if it.axis.is_thread()}
        if len(thread_axes) > 1:
            fail(f"gemm mma: {name} has >1 thread axis {sorted(thread_axes)}, only one supported")
        try:
            return canon.group(list(shape2d))
        except Exception as e:  # ICHECK failure -> layout not tileable as shape2d
            fail(f"gemm mma: {name} layout not tileable as {tuple(shape2d)}: {e}")

    A_grouped, A_seps = _slice_group(A, A_region, (K, M) if transpose_A else (M, K), "A")
    B_grouped, B_seps = _slice_group(B, B_region, (N, K) if transpose_B else (K, N), "B")
    C_grouped, C_seps = _slice_group(C, C_region, (M, N), "C")
    D_grouped, D_seps = _slice_group(D, D_region, (M, N), "D")

    # Split each operand into per-logical-dim sub-layouts (+ its offset), mapping
    # the buffer-order subs to standard (M,K)/(K,N) per the transpose flags.
    (DM, DN), D_off = _split(D_grouped, D_seps)
    (CM, CN), C_off = _split(C_grouped, C_seps)
    A_subs, A_off = _split(A_grouped, A_seps)
    B_subs, B_off = _split(B_grouped, B_seps)
    AM, AK = (A_subs[1], A_subs[0]) if transpose_A else (A_subs[0], A_subs[1])
    BK, BN = (B_subs[1], B_subs[0]) if transpose_B else (B_subs[0], B_subs[1])

    # Anchor-align so every operand decomposes each shared logical dim the same
    # way: M anchor = DM -> AM, CM ; N anchor = DN -> BN, CN ; K anchor = AK -> BK.
    # _align uses each anchor's raw (pre-canon) per-iter extents to group the
    # follower and reorders the follower's groups into the anchor's canonical
    # order, so the followers come out canonical. Canon the anchors themselves
    # afterwards. Every sub-layout is then in canonical (thread-left/memory-right)
    # order before the loop, so _frag_group groups directly without re-canon.
    AM = _align(DM, AM, "A.M")
    CM = _align(DM, CM, "C.M")
    BN = _align(DN, BN, "B.N")
    CN = _align(DN, CN, "C.N")
    BK = _align(AK, BK, "B.K")
    DM, DN, AK = _canon(DM), _canon(DN), _canon(AK)

    # Each dim's thread/memory region lengths, fixed once from its anchor (3
    # anchors x 2 parts = 6 lengths). Every operand of that dim reuses them in
    # _frag_group, so a follower whose register slot is actually a lane (B's N)
    # still gets the anchor's memory length instead of its own (mis)classified one.
    m_thr, m_mem = _region_totals(DM)
    n_thr, n_mem = _region_totals(DN)
    k_thr, k_mem = _region_totals(AK)

    # Per-instruction selection: try each candidate in order and use the first
    # whose shape / dtype / (later) fragment layout all fit. A failing check just
    # moves on to the next instruction; if none fit, decline.
    sig = (str(A.dtype), str(B.dtype), str(C.dtype), str(D.dtype))
    for inst in MMA_INSTRUCTIONS:
        assert inst.m % 8 == 0 and inst.n % 8 == 0 and inst.k % 8 == 0, (
            f"mma instruction {inst.name} m/n/k must be multiples of 8"
        )
        if M % inst.m or N % inst.n or K % inst.k:
            continue
        if sig != inst.dtype:
            continue
        # Group every operand into this instruction's fragment shape. The m16n8
        # lane split is g (8 lanes) along M and t (4 lanes) along N/K:
        #   C/D accumulator: M = g + 8*rM (inst.m//8 regs),  N = 2*t + rN (inst.n//4 regs)
        #   A multiplicand:  M as C/D's M,  K = 2*t + p + 8*kHi
        #   B multiplicand:  K as A's K,    N = g (lane 8, no reg: B has no M so N
        #                                        reuses the 8-lane g group)
        # so K's memory tail is [kHi, k_pack] with k_pack the innermost (stride-1)
        # contiguous pack and kHi = inst.k // (4 * k_pack) high-K register groups.
        # _frag_group raises if the caller layout can't be tiled or (when any
        # stride is given) fails the fragment checks -> move on to the next
        # instruction. lane/mem are [(extent, stride), ...]: the lane stride pins
        # the laneid stride (g=4, t=1), a mem stride pins a register iter (rN /
        # k_pack = 1; rM / kHi free = None). C shares D's accumulator fragment and
        # B's K shares A's K. B's N is the pure 8-lane g group, but aligned to the
        # accumulator's N (lane 4 + reg 2) it splits into lane g_hi (4, laneid
        # stride 8) and a "register" g_lo (2, laneid stride 4) -- a lane iter in the
        # register slot (B's N has no real register). Region lengths come from the
        # per-dim anchor (m/n/k _thr,_mem), so this split still tiles correctly.
        # _frag_group(layout, lane, mem, thread_total, mem_total); each carve is
        # (extent, stride, want_thread): lanes are thread, registers memory, except
        # B.N's register slot (g_lo) which is itself a lane (want_thread=True).
        kHi = inst.k // (4 * inst.k_pack)
        try:
            DM_g, DM_seps = _frag_group(
                DM, [(8, 4, True)], [(inst.m // 8, None, False)], m_thr, m_mem
            )
            DN_g, DN_seps = _frag_group(DN, [(4, 1, True)], [(inst.n // 4, 1, False)], n_thr, n_mem)
            CM_g, CM_seps = _frag_group(
                CM, [(8, 4, True)], [(inst.m // 8, None, False)], m_thr, m_mem
            )
            CN_g, CN_seps = _frag_group(CN, [(4, 1, True)], [(inst.n // 4, 1, False)], n_thr, n_mem)
            AM_g, AM_seps = _frag_group(
                AM, [(8, 4, True)], [(inst.m // 8, None, False)], m_thr, m_mem
            )
            AK_g, AK_seps = _frag_group(
                AK, [(4, 1, True)], [(kHi, None, False), (inst.k_pack, 1, False)], k_thr, k_mem
            )
            BK_g, BK_seps = _frag_group(
                BK, [(4, 1, True)], [(kHi, None, False), (inst.k_pack, 1, False)], k_thr, k_mem
            )
            BN_g, BN_seps = _frag_group(BN, [(4, 8, True)], [(2, 4, True)], n_thr, n_mem)
        except Exception:
            continue
        # M.to (M's warp tiling, group 0) must match across D, A, C so the same
        # logical M-block lands on the same warp in all three operands.
        m_to = _grp(DM_g, DM_seps, 0)
        if not (
            _same_iters(m_to, _grp(AM_g, AM_seps, 0)) and _same_iters(m_to, _grp(CM_g, CM_seps, 0))
        ):
            continue
        # N.to (N's warp tiling, group 0) must match across D, B, C so the same
        # logical N-block lands on the same warp in all three operands.
        n_to = _grp(DN_g, DN_seps, 0)
        if not (
            _same_iters(n_to, _grp(BN_g, BN_seps, 0)) and _same_iters(n_to, _grp(CN_g, CN_seps, 0))
        ):
            continue
        # K.to (K's warp tiling, group 0) must match across A, B so the same
        # logical K-block lands on the same warp in both operands.
        if not _same_iters(_grp(AK_g, AK_seps, 0), _grp(BK_g, BK_seps, 0)):
            continue
        break
    else:
        fail(f"no mma instruction fits M={M}, N={N}, K={K}, dtypes={sig}")

    # Per-operand register layout + matching local-view shape, grouped per logical
    # dim (offset drops thread-axis terms). Iter order = shape dim order:
    #   D/C -> [M.mo, N.mo, rM, rN]   A -> [M.mo, K.mo, rM, kHi, k_pack]
    #   B   -> [K.mo, N.mo, kHi, k_pack]   (k_pack innermost / contiguous)
    D_reg, d_shape = _reg_layout(
        [
            _grp(DM_g, DM_seps, 2),
            _grp(DN_g, DN_seps, 2),
            _grp(DM_g, DM_seps, 3),
            _grp(DN_g, DN_seps, 3),
        ],
        D_off,
    )
    C_reg, c_shape = _reg_layout(
        [
            _grp(CM_g, CM_seps, 2),
            _grp(CN_g, CN_seps, 2),
            _grp(CM_g, CM_seps, 3),
            _grp(CN_g, CN_seps, 3),
        ],
        C_off,
    )
    A_reg, a_shape = _reg_layout(
        [
            _grp(AM_g, AM_seps, 2),
            _grp(AK_g, AK_seps, 2),
            _grp(AM_g, AM_seps, 3),
            _grp(AK_g, AK_seps, 3),
            _grp(AK_g, AK_seps, 4),
        ],
        A_off,
    )
    B_reg, b_shape = _reg_layout(
        [
            _grp(BK_g, BK_seps, 2),
            _grp(BN_g, BN_seps, 2),
            _grp(BK_g, BK_seps, 3),
            _grp(BK_g, BK_seps, 4),
        ],
        B_off,
    )

    # Emit one mma per (m, n) output tile, accumulating over K. The tile / init /
    # K loops use T.unroll: the UnrollLoop pass fully expands them in TIR (their
    # bounds are compile-time constants), so the local-buffer indices resolve to
    # static register slots -- mma register operands must be constant.
    #
    # mma is d = a·b + c. D's accumulator is initialized once per output tile --
    # copying C when beta==1, clearing to 0 when beta==0 -- then every K step
    # accumulates in place with c = d, giving a single uniform mma form.
    M_tiles, N_tiles, K_tiles = d_shape[0], d_shape[1], a_shape[1]
    shape_str = f"m{inst.m}n{inst.n}k{inst.k}"
    a_type, b_type, c_type, d_type = inst.dtype
    use_c = _const_scalar(beta) == 1.0

    # Per-register counts in the fixed PTX enumeration order (derived from the
    # instruction, NOT hardcoded, so m16n8k8 with kHi==1 also works):
    #   D/C accumulator: rM = inst.m // 8 regs along M, rN = inst.n // 4 along N
    #                    c_id = 2 * rM + rN  (4 f32 for m16n8k16, also 4 for k8)
    #   A multiplicand:  rM = inst.m // 8, kHi = inst.k // (4 * inst.k_pack)
    #                    b32 = rM + 2 * kHi  (4 b32 for k16, 2 b32 for k8)
    #   B multiplicand:  kHi = inst.k // (4 * inst.k_pack)
    #                    b32 = kHi          (2 b32 for k16, 1 b32 for k8)
    n_rM = inst.m // 8
    n_rN = inst.n // 4
    n_kHi = inst.k // (4 * inst.k_pack)

    @T.prim_func(check_well_formed=False)
    def impl():
        d_local = D.local(*d_shape, layout=D_reg)
        c_local = C.local(*c_shape, layout=C_reg)
        a_local = A.local(*a_shape, layout=A_reg)
        b_local = B.local(*b_shape, layout=B_reg)
        for m in T.unroll(M_tiles):
            for n in T.unroll(N_tiles):
                # Initialize D[m, n]: copy C (beta==1) or clear to 0 (beta==0).
                for rM in T.unroll(n_rM):
                    for rN in T.unroll(n_rN):
                        if use_c:
                            d_local[m, n, rM, rN] = c_local[m, n, rM, rN]
                        else:
                            d_local[m, n, rM, rN] = T.float32(0)
                # Accumulate over K in place: d = a·b + d.
                for k in T.unroll(K_tiles):
                    # D: 4 f32 in PTX order c_id = 2*rM + rN.
                    d_ptrs = [
                        d_local.ptr_to([m, n, rM, rN]) for rM in range(n_rM) for rN in range(n_rN)
                    ]
                    # A: b32 regs in PTX order b32 = rM + 2*kHi (kHi outer, rM inner).
                    a_ptrs = [
                        a_local.ptr_to([m, k, rM, kHi, 0])
                        for kHi in range(n_kHi)
                        for rM in range(n_rM)
                    ]
                    # B: b32 regs in PTX order b32 = kHi.
                    b_ptrs = [b_local.ptr_to([k, n, kHi, 0]) for kHi in range(n_kHi)]
                    # Accumulate in place into D's own regs: c = d.
                    T.ptx.mma(
                        shape_str,
                        "row",
                        "col",
                        d_type,
                        a_type,
                        b_type,
                        c_type,
                        d_ptrs,
                        a_ptrs,
                        b_ptrs,
                        d_ptrs,
                    )

    return impl
