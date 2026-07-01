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

"""Generic swizzle-aware iter pattern for CUDA copy dispatches.

When the per-thread outer-iter loop satisfies (C1)+(C2) below for a
``SwizzleLayout(per_element=p, swizzle_len=sw, atom_len=at,
swizzle_inner=True)`` on the SMEM side, the swizzled physical address
at unrolled iter ``k`` reduces to

    addr(k) = base_off + sum_{j : bit_j(k)=1} signed_strides[j]

where ``base_off`` and the ``signed_strides[j]`` are per-thread runtime
constants set once at thread setup. Per-iter cost is then ``popcount(k)``
register adds instead of a full ``swizzle.apply(...)`` per iter.

Notation. Each binary outer iter has element-stride ``2^(bj + p)`` for
some chunk bit position ``bj >= 0`` (so ``stride / C = 2^bj`` where
``C = 1 << p``). The chunk index ``q(M0) = M0 // C`` partitions into
four bit ranges by where ``bj`` lands:

  * ``[0, sw)``        — "inner" (Case 1.A in the proof)
  * ``[sw, at)``       — "mid"   (Case 1.B)
  * ``[at, at + sw)``  — "outer" (Case 1.C; the bit overlaps the swizzle
                         outer mask, so its addition produces a
                         secondary contribution at ``bj - at``)
  * ``[at + sw, ∞)``   — "above" (Case 1.D)

Conditions for the linear-combination fast path:

  (C1) bit-clear no-carry: ``bit_bj(q(M0)) = 0`` for every binary iter.
  (C2) support disjointness: no inner-outer pair ``(bj_A, bj_C)`` with
       ``bj_C in [at, at+sw)`` and ``bj_A = bj_C - at`` both present.

  (distinctness) The ``bj`` values across all binary iters must be
       distinct — two iters at the same ``bj`` collapse into bit
       ``bj + 1`` whose case behavior may differ.

Under (C1)+(C2)+(distinctness), for each binary iter at position ``bj``:

    T(bj)        = 2^(bj + p)            # element stride
    sigma_b(M0)  = 1 - 2 * bit_b(q(M0))  # ∈ {+1, -1}

    signed_strides[j] = sigma_(at + bj)(M0) * T(bj)        bj in [0, sw)
                     = T(bj)                               bj in [sw, at)
                     = T(bj) + sigma_(bj - at)(M0) * T(bj - at)
                                                           bj in [at, at + sw)
                     = T(bj)                               bj >= at + sw

The ``swizzle_inner=False`` mode swaps the inner/outer roles and is not
yet covered; ``try_recognize`` gates on this.
"""

from dataclasses import dataclass

import tvm
from tvm import arith
from tvm.script import tirx as T
from tvm.tirx.expr import IntImm as _IntImm
from tvm.tirx.layout import ComposeLayout, SwizzleLayout


@dataclass
class _BitIter:
    """Pow2-extent outer iter, binary-split into ``n_bits`` chunk-bit flips.

    ``slot_start..slot_start + n_bits`` is this iter's range in the global
    ``bit_positions`` / ``iter_strides_elems`` / ``signed_strides`` arrays.
    Slot ``slot_start + b`` corresponds to bit position ``n_bits - 1 - b``
    of this iter's per-iter coord (outermost binary bit first).
    """

    ext: int
    n_bits: int
    slot_start: int


@dataclass
class _LinearIter:
    """Outer iter contributing ``c * stride`` to the offset (no bit decomp).

    Used when ``stride`` is a multiple of ``2^(p + at + sw)`` (pure Case 1.D
    regime: swizzle XOR has no effect on bits the iter flips). ``ext`` does
    not need to be a power of two.
    """

    ext: int
    stride: int


@dataclass
class SwizzlePattern:
    """A recognized swizzle iter pattern.

    ``bit_positions[j]`` and ``iter_strides_elems[j]`` collect the binary
    sub-iters from every BitIter in outer-iter order (outermost first).
    ``outer_iters`` lists every outer iter (BitIter or LinearIter) in
    outermost-first order; ``emit_iter_offset`` walks this list to
    decompose ``mm`` per-iter. Empty lists = trivially recognized
    degenerate case (no outer iter, just base_off).
    """

    swizzle: SwizzleLayout
    bit_positions: list[int]
    iter_strides_elems: list[int]
    outer_iters: "list[_BitIter | _LinearIter]"

    @property
    def n_binary_iters(self) -> int:
        return len(self.bit_positions)


def get_swizzle(layout) -> SwizzleLayout | None:
    """Return the SwizzleLayout from ``layout`` if present, else ``None``.

    Accepts ``ComposeLayout(SwizzleLayout, TileLayout)`` (the common case
    when a TileLayout is wrapped by a swizzle), or a bare ``SwizzleLayout``.
    """
    if isinstance(layout, ComposeLayout):
        return layout.swizzle
    if isinstance(layout, SwizzleLayout):
        return layout
    return None


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def try_recognize(
    swizzle: SwizzleLayout,
    iter_extents: list[int],
    iter_strides: list[int],
    s_off_template,
    var_bounds: dict | None = None,
) -> SwizzlePattern | None:
    """Return a ``SwizzlePattern`` if (C1)+(C2)+(distinctness) hold, else ``None``.

    ``iter_extents`` / ``iter_strides``: the outer-iter list on the S side
    (excluding T iter and vec iter), in outermost-first order matching
    ``s_p.shard[:-2]`` (or the atom-derived analog in ``reg.py``).
    Strides are in element units.

    Each outer iter with ``extent=2^k`` and ``stride=s`` is conceptually
    split into ``k`` binary iters of strides ``2^(k-1)*s, ..., 2*s, s``
    (outermost first within the split — this matches ``_flat_outer_coords``
    semantics, since the highest-stride iter must change slowest in the
    flat-index decomposition).

    ``s_off_template`` is the per-thread linear base offset expression
    (with a placeholder var for the thread-id contribution). It is used
    only to check condition (C1) symbolically via ``arith.Analyzer``;
    ``emit_init`` takes the resolved form separately.

    ``var_bounds`` is an optional ``{Var: tvm.ir.Range}`` map of placeholder
    bounds to ``analyzer.bind`` before the (C1) check. Without bounds,
    structurally-OK forms like ``(lane // 8) * 8 + (lane % 8) * Q`` where
    ``lane < 32`` make ``(... // (C·2^bj)) % 2 == 0`` unprovable — the
    bit is in fact always 0 but the analyzer can't conclude it universally.
    Pass ``{lane_ph: Range(0, 32), warp_ph: Range(0, n_warps)}`` (or the
    scope's equivalents) to let the (C1) check fire on these templates.
    """
    # swizzle_inner=False swaps the inner/outer xor direction — Cases 1.A
    # and 1.C roles flip. Not derived/tested yet; reject for safety.
    if not swizzle.swizzle_inner:
        return None

    p = swizzle.per_element
    sw = swizzle.swizzle_len
    at = swizzle.atom_len
    C = 1 << p
    # Pure Case 1.D threshold: stride a multiple of this means every chunk-bit
    # the iter flips is at position >= at + sw (above the swizzle XOR region),
    # so swizzle has no effect and the contribution is purely linear in the
    # iter coord — no power-of-2 ext requirement.
    pure_1d = 1 << (p + at + sw)

    bit_positions: list[int] = []
    iter_strides_elems: list[int] = []
    outer_iters: list = []

    for ext, stride in zip(iter_extents, iter_strides):
        # Zero-stride iters degrade dq=0 → log2 undefined. Explicit guard.
        if stride == 0 or stride % C != 0:
            return None
        if ext <= 0:
            return None
        if ext == 1:
            # Trivial iter contributes nothing; skip without forcing pow2.
            continue
        if not _is_pow2(ext):
            # Non-pow2 ext can only be handled by the linear path. That in turn
            # requires the iter to be in pure Case 1.D (stride a multiple of
            # the swizzle period) so the swizzle does not interact with the
            # per-coord contribution.
            if stride % pure_1d != 0:
                return None
            outer_iters.append(_LinearIter(ext=ext, stride=stride))
            continue
        # pow2 ext: binary split (existing path).
        k = ext.bit_length() - 1  # log2(ext)
        slot_start = len(bit_positions)
        # Split into k binary iters; the outermost (within this split) carries
        # the largest stride so that flat-index bit decomp matches our
        # outer-iter list ordering.
        for j in range(k - 1, -1, -1):
            substride = stride * (1 << j)
            dq = substride // C
            # dq must be a single bit set (so this binary iter flips exactly
            # one bit of the chunk index). _is_pow2 also rejects dq=0.
            if not _is_pow2(dq):
                return None
            bj = dq.bit_length() - 1
            # All bj >= 0 accepted; case branching happens in emit_init.
            bit_positions.append(bj)
            iter_strides_elems.append(substride)
        outer_iters.append(_BitIter(ext=ext, n_bits=k, slot_start=slot_start))

    # Distinctness: two binary iters at the same bj collapse to bj+1, whose
    # case behavior may differ from bj. See module docstring NB.
    if len(set(bit_positions)) != len(bit_positions):
        return None

    bj_set = set(bit_positions)

    # (C2) support disjointness: the only possible collision is between a
    # Case-1.A iter at bj_A and a Case-1.C iter at bj_A + at. Checking the
    # 1.C direction alone is symmetric and complete.
    for bj in bj_set:
        if at <= bj < at + sw and (bj - at) in bj_set:
            return None  # inner-outer pair collision

    # (C1) per-iter no-carry on q(M0). Must hold *symbolically over all*
    # free lane / warp placeholders in s_off_template — ``can_prove_equal``
    # returns False if the analyzer can't discharge the equality
    # universally, conservatively forcing a fallback.
    analyzer = arith.Analyzer()
    if var_bounds:
        for var, rng in var_bounds.items():
            analyzer.bind(var, rng)
    for bj in bj_set:
        divisor = C * (1 << bj)
        check = tvm.tirx.floormod(
            tvm.tirx.floordiv(s_off_template, _IntImm("int32", divisor)),
            _IntImm("int32", 2),
        )
        if not analyzer.can_prove_equal(check, _IntImm("int32", 0)):
            return None

    return SwizzlePattern(
        swizzle=swizzle,
        bit_positions=bit_positions,
        iter_strides_elems=iter_strides_elems,
        outer_iters=outer_iters,
    )


def emit_init(pattern: SwizzlePattern, s_off_resolved):
    """Emit at thread setup (call from inside the @T.prim_func body):

      1. ``base_off = swizzle.apply(s_off_resolved)`` — runtime, per-thread,
         computed once.
      2. ``signed_strides[j]`` for each binary iter j, written into a local
         buffer using the sigma formula above.

    Returns ``(signed_strides_buffer_or_None, base_off_primexpr)``. The
    buffer is ``None`` when ``pattern.n_binary_iters == 0`` (no outer
    iter, no signed_strides needed).

    ``s_off_resolved`` is the per-thread offset with the real tid Var
    substituted in (not the placeholder).
    """
    swizzle = pattern.swizzle
    p = swizzle.per_element
    sw = swizzle.swizzle_len
    at = swizzle.atom_len
    C = 1 << p

    base_off = swizzle.apply(s_off_resolved)["m"]

    n = pattern.n_binary_iters
    if n == 0:
        return None, base_off

    signed_strides = T.alloc_buffer([n], "int32", scope="local")
    q = tvm.tirx.floordiv(s_off_resolved, C)

    def _sigma_bit(bit_pos: int):
        # 1 - 2 * bit_(bit_pos)(q); ∈ {+1, -1}.
        row_bit = tvm.tirx.bitwise_and(
            tvm.tirx.shift_right(q, _IntImm("int32", bit_pos)),
            _IntImm("int32", 1),
        )
        return _IntImm("int32", 1) - row_bit * _IntImm("int32", 2)

    for j, (bj, stride) in enumerate(zip(pattern.bit_positions, pattern.iter_strides_elems)):
        stride_pow = stride  # = 2^(bj + p) elements
        if 0 <= bj < sw:
            # Case 1.A (inner): signed_stride = sigma_(at + bj) · T.
            value = _sigma_bit(at + bj) * _IntImm("int32", stride_pow)
        elif sw <= bj < at:
            # Case 1.B (mid): signed_stride = +T.
            value = _IntImm("int32", stride_pow)
        elif at <= bj < at + sw:
            # Case 1.C (outer): signed_stride = T + sigma_(bj - at) · T_sec.
            # Invariant: bj >= at, so T_sec = T >> at = 2^(bj - at + p)
            # = T(bj - at) is well-defined (no underflow).
            stride_sec = stride_pow >> at
            value = _IntImm("int32", stride_pow) + _sigma_bit(bj - at) * _IntImm(
                "int32", stride_sec
            )
        else:  # bj >= at + sw, Case 1.D (above)
            # No swizzle effect at this bit; signed_stride = +T.
            value = _IntImm("int32", stride_pow)
        # NB: Buffer.__setitem__ syntax (``signed_strides[j] = value``) is
        # intercepted by the TIRx script parser but not by raw Python when
        # this function is called from outside an @T.inline body. Use the
        # low-level buffer_store builder instead.
        T.buffer_store(signed_strides, value, [_IntImm("int32", j)])

    return signed_strides, base_off


def emit_iter_offset(pattern: SwizzlePattern, signed_strides, base_off, k):
    """Compute the per-mm physical S offset = ``base_off`` + sum of per-iter
    contributions.

    ``k`` is the flat outer iter index ∈ ``[0, prod(it.ext for it in outer_iters))``.
    Decomposed innermost-first across ``pattern.outer_iters`` into per-iter
    coords ``c_i``. Each iter contributes:

      * ``_BitIter``: ``sum_b bit_(n_bits-1-b)(c_i) * signed_strides[slot_start + b]``,
        i.e. each binary bit of ``c_i`` selects its precomputed sigma-stride.
        The slot order (outermost-first within the iter) means the highest
        bit of ``c_i`` indexes the slot at ``slot_start``.
      * ``_LinearIter``: ``c_i * stride`` (no bit decomposition; used when
        ``stride`` is a multiple of ``2^(p + at + sw)`` so swizzle has no
        XOR effect and ``ext`` need not be pow2).

    Two paths per iter:
      * Python int ``k`` — coords and bits known at parse time; emits only
        the necessary adds, no runtime shift/mask.
      * TIRx Var ``k`` — emits floormod/floordiv + bit-and/shift; relies on
        downstream unroll + constant-fold.
    """
    if not pattern.outer_iters:
        return base_off

    off = base_off
    remaining = k
    is_const = isinstance(k, int)
    for it in reversed(pattern.outer_iters):  # innermost first
        ext = it.ext
        if is_const:
            c = remaining % ext
            remaining = remaining // ext
        else:
            c = tvm.tirx.floormod(remaining, _IntImm("int32", ext))
            remaining = tvm.tirx.floordiv(remaining, _IntImm("int32", ext))
        if isinstance(it, _LinearIter):
            if is_const:
                if c != 0:
                    off = off + c * it.stride
            else:
                off = off + c * _IntImm("int32", it.stride)
            continue
        # _BitIter
        for b in range(it.n_bits):
            bit_pos = it.n_bits - 1 - b
            slot = it.slot_start + b
            if is_const:
                if (c >> bit_pos) & 1:
                    off = off + signed_strides[slot]
            else:
                bit = tvm.tirx.bitwise_and(
                    tvm.tirx.shift_right(c, _IntImm("int32", bit_pos)),
                    _IntImm("int32", 1),
                )
                off = off + bit * signed_strides[slot]
    return off


def emit_fallback_offset(swizzle: SwizzleLayout, s_off_resolved, ds_k):
    """Slow but always-correct path: full ``swizzle.apply(s_off + ds_k)``
    per iter. Use when ``try_recognize`` returns ``None``.

    ``ds_k`` is the outer-iter delta for unrolled iter k — typically a
    Expr (a function of the unroll var that simplifies to a constant
    after unrolling) or a Python int. ``s_off_resolved`` is the per-thread
    base linear offset with the real tid Var substituted.
    """
    return swizzle.apply(s_off_resolved + ds_k)["m"]
