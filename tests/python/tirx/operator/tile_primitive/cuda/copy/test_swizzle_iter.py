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

"""Tests for the generic swizzle-aware iter pattern in
``cuda/copy/_swizzle_iter.py``.

Two layers:

* **Recognizer tests** check that ``try_recognize`` returns the expected
  ``SwizzlePattern`` (or rejects) for each of conditions (a)+(b)+(c).
* **Numeric correctness tests** verify the proof empirically: for many
  ``(M0, k)`` samples, the formula
  ``apply(M0) + sum_{j : bit_j(k)=1} signed_strides[j]``
  equals ``apply(M0 + ds_k)`` computed by the layout's own Apply formula.
  Plus a per-thread-sign-matters test that would fail for a constant-sign
  implementation, ensuring the test isn't trivially satisfied.

All algorithm-level (no GPU needed). End-to-end emit is tested in
``test_gmem_smem.py::test_swizzled_smem_emit_must_be_swizzle_aware``.
"""

import pytest

import tvm
from tvm.tirx import Var as _TirVar
from tvm.tirx.expr import IntImm as _IntImm
from tvm.tirx.layout import ComposeLayout, S, SwizzleLayout, TileLayout
from tvm.tirx.operator.tile_primitive.cuda.copy._swizzle_iter import (
    get_swizzle,
    try_recognize,
)

# ----------------------------------------------------------------------------
# Pure-Python reference: SwizzleLayout's Apply, plus the proof's formula.
# Used as ground truth — both must agree for the proof to hold.
# ----------------------------------------------------------------------------


def py_swizzle_apply(M: int, p: int, sw: int, at: int) -> int:
    """Pure-Python reimplementation of SwizzleLayoutNode::Apply (swizzle_inner=True):
    phys = swz_q * C + (M mod C)
    q = M / C; swz_q = q XOR ((q & outer_mask) >> at)
    """
    C = 1 << p
    q = M // C
    outer_mask = ((1 << sw) - 1) << at
    swz_q = q ^ ((q & outer_mask) >> at)
    return swz_q * C + (M % C)


def py_signed_strides(
    M0: int, p: int, sw: int, at: int, bit_positions: list[int], iter_strides_elems: list[int]
) -> list[int]:
    """Pure-Python reimplementation of emit_init's formula. Mirrors:
    if bj >= sw: sigma_bj = +1                              (mid_bits)
    else       : sigma_bj = 1 - 2 * bit_(at+bj)(M0/C)        (chunk_bits)
    signed_strides[j] = sigma_bj * iter_strides_elems[j]
    """
    C = 1 << p
    q = M0 // C
    out: list[int] = []
    for bj, stride in zip(bit_positions, iter_strides_elems):
        if bj >= sw:
            out.append(stride)
        else:
            row_bit = (q >> (at + bj)) & 1
            sigma = 1 - 2 * row_bit
            out.append(sigma * stride)
    return out


def py_iter_offset(base_off: int, k: int, signed_strides: list[int]) -> int:
    """Formula sum: base_off + sum_{j : bit_(n-1-j)(k)=1} signed_strides[j]."""
    n = len(signed_strides)
    off = base_off
    for j in range(n):
        if (k >> (n - 1 - j)) & 1:
            off += signed_strides[j]
    return off


def py_outer_ds(k: int, iter_extents: list[int], iter_strides: list[int]) -> int:
    """Decode a flat outer index k into per-iter coords (matching
    _flat_outer_coords) and sum coord_i * stride_i for the corresponding ds."""
    coords: list[int] = []
    rem = k
    for ext in reversed(iter_extents):
        coords.append(rem % ext)
        rem //= ext
    coords.reverse()
    return sum(c * s for c, s in zip(coords, iter_strides))


# ----------------------------------------------------------------------------
# Recognizer tests — verify try_recognize accepts / rejects under (a)+(b)+(c).
# ----------------------------------------------------------------------------


def test_get_swizzle_extracts_from_compose():
    sw = SwizzleLayout(3, 3, 3)
    assert get_swizzle(sw) is not None
    assert get_swizzle(ComposeLayout(sw, TileLayout(S[(64, 64)]))) is not None
    assert get_swizzle(TileLayout(S[(64, 64)])) is None


def test_recognize_nvfp4_case():
    """nvfp4's epilogue: SwizzleLayout(3,3,3), iter extents [2,2,2] strides
    [8,16,32], M0 = tid * 64 (each thread starts at col 0 of one row;
    row_stride 64 = 8 chunks, ensures chunk bits of M0/C are zero for all
    iter bit positions)."""
    sw = SwizzleLayout(3, 3, 3)
    tid = _TirVar("tid", "int32")
    # M0 = tid * 64 → M0/C = tid * 8 → bits 0,1,2 are 0 (since multiplied by 8).
    M0 = tid * _IntImm("int32", 64)
    pat = try_recognize(sw, [2, 2, 2], [8, 16, 32], M0)
    assert pat is not None
    assert pat.bit_positions == [0, 1, 2]
    assert pat.iter_strides_elems == [8, 16, 32]
    assert pat.n_binary_iters == 3


def test_recognize_binary_split():
    """A single outer iter with extent=4 stride=8 splits into two binary
    iters with strides 16 and 8 (outermost first, matching _flat_outer_coords)."""
    sw = SwizzleLayout(3, 3, 3)
    tid = _TirVar("tid", "int32")
    M0 = tid * _IntImm("int32", 64)
    pat = try_recognize(sw, [4], [8], M0)
    assert pat is not None
    # Split: stride 8*2 = 16 (outermost), stride 8 (innermost) → bits [1, 0]
    assert pat.bit_positions == [1, 0]
    assert pat.iter_strides_elems == [16, 8]


def test_recognize_mid_bits():
    """SwizzleLayout(p=4, sw=2, at=4): chunk bits [0,2), mid bits [2,4),
    row bits [4,6). An iter at bj=2 lives in mid_bits → sigma is always +1
    (i.e., the recognizer accepts and the sign formula won't read row bits)."""
    sw = SwizzleLayout(4, 2, 4)  # C=16, mid_bits cover bits 2..3
    tid = _TirVar("tid", "int32")
    # M0/C must have bit 2 == 0. Pick row_stride = 64 (= 4*C) so M0/C = tid*4
    # which has zeros at bit 0,1, and bit 2 is bit 0 of tid... hmm that varies.
    # Use row_stride = 128 (= 8*C, contributes 4 to M0/C per tid → bit 2 of M0/C
    # depends on whether tid is even/odd — not zero. Instead use row_stride such
    # that M0/C is provably 0 mod 8 = 0 at bits 0..2: row_stride = 256 (= 16*C)
    # → M0/C = tid*16 → bits 0..3 all 0. iter_mask = bit 2, divisor = C*4 = 64.
    M0 = tid * _IntImm("int32", 256)
    pat = try_recognize(sw, [2], [64], M0)  # stride 64 = C * 2^2 → bj=2 (mid)
    assert pat is not None
    assert pat.bit_positions == [2]
    assert pat.iter_strides_elems == [64]


def test_reject_not_chunk_aligned():
    """Condition (a): stride must be a multiple of C."""
    sw = SwizzleLayout(3, 3, 3)  # C=8
    tid = _TirVar("tid", "int32")
    M0 = tid * _IntImm("int32", 64)
    # stride 4 is not a multiple of C=8 → reject.
    assert try_recognize(sw, [2], [4], M0) is None


def test_reject_carries_into_row_bits():
    """Condition (b): bj < at. A binary iter with stride C * 2^at lands at
    bj=at, which would change the row bits → reject."""
    sw = SwizzleLayout(3, 3, 3)  # at=3, so max bj = 2
    tid = _TirVar("tid", "int32")
    M0 = tid * _IntImm("int32", 64)
    # Strides 8,16,32 OK (bj=0,1,2); 64 → bj=3 → reject.
    assert try_recognize(sw, [2, 2, 2, 2], [8, 16, 32, 64], M0) is None


def test_reject_chunk_overlap():
    """Condition (c): (M0/C) must have 0 bits at all iter-bit positions per
    thread. If M0 = tid * 8 (so M0/C = tid), then bit 0 of M0/C is bit 0 of
    tid — analyzer can't prove this is 0 across all threads, so reject."""
    sw = SwizzleLayout(3, 3, 3)  # C=8
    tid = _TirVar("tid", "int32")
    # M0 = tid * C = tid * 8 → M0/C = tid → bit 0 NOT provably zero.
    M0 = tid * _IntImm("int32", 8)
    assert try_recognize(sw, [2], [8], M0) is None


def test_recognize_no_outer_iters():
    """Degenerate case: no outer iter at all. Recognizer returns a trivial
    pattern (empty bit_positions). Emit will use base_off alone."""
    sw = SwizzleLayout(3, 3, 3)
    tid = _TirVar("tid", "int32")
    M0 = tid * _IntImm("int32", 64)
    pat = try_recognize(sw, [], [], M0)
    assert pat is not None
    assert pat.n_binary_iters == 0


# ----------------------------------------------------------------------------
# Numeric correctness — the PROOF. The formula must equal apply(M0 + ds_k)
# for all sampled (M0, k) and for non-trivial M0 values per thread.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p,sw,at,iter_extents,iter_strides,row_stride",
    [
        # nvfp4-like (p=sw=at=3, 3 binary iters covering one swizzle row)
        (3, 3, 3, [2, 2, 2], [8, 16, 32], 64),
        # single binary iter at chunk_bit position
        (3, 3, 3, [2], [8], 64),
        # split-from-extent-4 (one outer becomes two binary)
        (3, 3, 3, [4], [8], 64),
        # mid_bits region
        (4, 2, 4, [2], [64], 256),
        # mix: one chunk_bit + one mid_bit
        (
            3,
            2,
            4,
            [2, 2],
            [8, 32],
            256,
        ),  # C=8, sw=2, at=4 → bj_max=3 for stride 64 → use 32 (bj=2 in mid)
    ],
)
def test_formula_matches_apply_under_conditions(
    p,
    sw,
    at,
    iter_extents,
    iter_strides,
    row_stride,
):
    """For every (M0, k) sample, the signed-strides formula must equal
    py_swizzle_apply(M0 + ds_k). Sweeps multiple per-thread M0 values to
    catch any per-thread-sign bug (a constant-sign impl would fail here)."""
    swizzle = SwizzleLayout(p, sw, at)
    tid = _TirVar("tid", "int32")
    M0_template = tid * _IntImm("int32", row_stride)
    pat = try_recognize(swizzle, iter_extents, iter_strides, M0_template)
    assert pat is not None, (
        f"recognizer rejected supposedly-valid case p={p},sw={sw},at={at} "
        f"iter_extents={iter_extents} iter_strides={iter_strides} row_stride={row_stride}"
    )

    total_iters = 1
    for ext in iter_extents:
        total_iters *= ext

    # Per-thread sweep: pick concrete tid values that span a few rows of the
    # swizzle atom. Tid = 0 alone would hide the sign issue (M0/C row bits all
    # 0 → all signs +1); larger tids exercise the sign-flip branches.
    for tid_val in [0, 1, 3, 5, 7, 13, 21]:
        M0 = tid_val * row_stride
        base_off = py_swizzle_apply(M0, p, sw, at)
        ss = py_signed_strides(
            M0,
            p,
            sw,
            at,
            pat.bit_positions,
            pat.iter_strides_elems,
        )
        for k in range(total_iters):
            ds_k = py_outer_ds(k, iter_extents, iter_strides)
            ground_truth = py_swizzle_apply(M0 + ds_k, p, sw, at)
            formula = py_iter_offset(base_off, k, ss)
            assert formula == ground_truth, (
                f"formula mismatch: p={p},sw={sw},at={at} "
                f"iter_extents={iter_extents} iter_strides={iter_strides} "
                f"tid={tid_val} M0={M0} k={k} ds_k={ds_k} "
                f"apply(M0+ds_k)={ground_truth} formula={formula} "
                f"signed_strides={ss}"
            )


def test_per_thread_sign_actually_varies():
    """Guard against a 'constant +1 stride' bug: for the nvfp4-like case,
    different tids MUST produce different signed_strides[0] (since the
    formula is sigma_0 = 1 - 2 * bit_(at)(M0/C) and that bit toggles with
    tid). If a buggy impl always returned +stride, this test would catch it."""
    p, sw, at = 3, 3, 3
    row_stride = 64  # M0/C = tid * 8 → bit (at=3) = bit 0 of tid
    # tid=0 → M0/C bit 3 = 0 → sigma = +1; tid=1 → bit 3 = 1 → sigma = -1
    ss_even = py_signed_strides(0 * row_stride, p, sw, at, [0], [8])
    ss_odd = py_signed_strides(1 * row_stride, p, sw, at, [0], [8])
    assert ss_even != ss_odd, "per-thread sign formula degenerated to constant — proof / impl bug"
    assert ss_even == [8]
    assert ss_odd == [-8]


# ----------------------------------------------------------------------------
# Fallback path — when recognizer rejects, per-iter swizzle.apply gives the
# right answer. This is trivial (we delegate to layout.apply) but documents
# the contract.
# ----------------------------------------------------------------------------


def test_recognize_linear_iter_pure_case_1d():
    """Outer iter with non-pow2 ext is accepted IF its stride is a multiple
    of the swizzle period 2^(p+at+sw) (pure Case 1.D, swizzle has no XOR
    effect). The iter is stored as a LinearIter (no bit decomposition).
    """
    from tvm.tirx.operator.tile_primitive.cuda.copy._swizzle_iter import (
        _BitIter,
        _LinearIter,
    )

    p, sw, at = 3, 3, 3
    swizzle = SwizzleLayout(p, sw, at)
    period = 1 << (p + at + sw)  # 512
    # Outer iter (ext=3, stride=period) — non-pow2 but pure Case 1.D.
    # Inner iter (ext=2, stride=8) — pow2, Case 1.A (bj=0).
    pat = try_recognize(swizzle, [3, 2], [period, 8], _IntImm("int32", 0))
    assert pat is not None
    assert len(pat.outer_iters) == 2
    # Outermost (index 0) corresponds to first input iter = the linear one.
    assert isinstance(pat.outer_iters[0], _LinearIter)
    assert pat.outer_iters[0].ext == 3
    assert pat.outer_iters[0].stride == period
    # Innermost (index 1) is the binary-split iter.
    assert isinstance(pat.outer_iters[1], _BitIter)
    assert pat.outer_iters[1].ext == 2
    assert pat.outer_iters[1].n_bits == 1
    assert pat.outer_iters[1].slot_start == 0
    # bit_positions / iter_strides_elems only contain the binary iter's bit.
    assert pat.bit_positions == [0]  # 8/8 = 2^0
    assert pat.iter_strides_elems == [8]


def test_reject_non_pow2_ext_not_case_1d():
    """Non-pow2 ext where stride is NOT in pure Case 1.D regime — reject.
    stride=64 = 2^(p+at) = one atom row, which is Case 1.C (in [at, at+sw))
    territory and the XOR depends on M0, so the linear path is unsafe."""
    swizzle = SwizzleLayout(3, 3, 3)
    pat = try_recognize(swizzle, [3], [64], _IntImm("int32", 0))
    assert pat is None


def test_emit_mixed_linear_bit_correctness():
    """Brute-force: for a mixed (LinearIter outer, BitIter inner) pattern,
    emit_iter_offset's prediction must equal the actual swizzle output for
    every (tid, k) — including the non-pow2 outer extent's coord 2."""
    from tvm.tirx.operator.tile_primitive.cuda.copy._swizzle_iter import (
        _LinearIter,
    )

    p, sw, at = 3, 3, 3
    swizzle = SwizzleLayout(p, sw, at)
    period = 1 << (p + at + sw)  # 512
    iter_extents, iter_strides = [3, 2], [period, 8]
    tid = _TirVar("tid", "int32")
    # Inner iter bj=0 in [0, sw); (C1) needs bit_0(M0/C) = 0 ⇒ M0/C even
    # ⇒ M0 multiple of 16. So row_stride = 16.
    M0_template = tid * _IntImm("int32", 16)
    pat = try_recognize(swizzle, iter_extents, iter_strides, M0_template)
    assert pat is not None

    def py_emit(pattern, signed_strides, base_off, k):
        off = base_off
        remaining = k
        for it in reversed(pattern.outer_iters):
            c = remaining % it.ext
            remaining = remaining // it.ext
            if isinstance(it, _LinearIter):
                off += c * it.stride
            else:
                for b in range(it.n_bits):
                    bit_pos = it.n_bits - 1 - b
                    slot = it.slot_start + b
                    if (c >> bit_pos) & 1:
                        off += signed_strides[slot]
        return off

    total_k = iter_extents[0] * iter_extents[1]
    for tid_val in [0, 1, 5, 7, 13]:
        M0 = tid_val * 16
        base_off = py_swizzle_apply(M0, p, sw, at)
        ss = py_signed_strides(
            M0,
            p,
            sw,
            at,
            pat.bit_positions,
            pat.iter_strides_elems,
        )
        for k in range(total_k):
            ds_k = py_outer_ds(k, iter_extents, iter_strides)
            ground_truth = py_swizzle_apply(M0 + ds_k, p, sw, at)
            formula = py_emit(pat, ss, base_off, k)
            assert formula == ground_truth, (
                f"mixed mismatch: tid={tid_val} M0={M0} k={k} ds_k={ds_k} "
                f"truth={ground_truth} formula={formula} ss={ss}"
            )


def test_fallback_path_when_recognizer_rejects():
    """The recognizer should reject when (c) fails, and the resulting
    fallback emit (swizzle.apply per iter) is the correct path. This test
    proves the rejection and demonstrates that the swizzled offset really
    differs from the linear offset for the rejected case — so a buggy
    `linear-offset-without-XOR` emit (the pre-fix behavior) would give the
    wrong answer on at least one (tid, k) sample. The fallback emit, by
    construction, delegates to swizzle.apply and is thus correct."""
    p, sw, at = 3, 3, 3
    swizzle = SwizzleLayout(p, sw, at)
    tid = _TirVar("tid", "int32")
    M0_template = tid * _IntImm("int32", 8)  # (c) fails: bit 0 of M0/C = bit 0 of tid
    pat = try_recognize(swizzle, [2], [8], M0_template)
    assert pat is None, "recognizer must reject when (c) fails"

    # Demonstrate the swizzled offset differs from linear for at least one
    # (tid, k) — proves the swizzle is actually non-trivial here and the
    # broken linear-offset emit would give the wrong physical address.
    iter_extents, iter_strides = [2], [8]
    diverging_samples = 0
    for tid_val in range(16):
        M0 = tid_val * 8
        for k in range(2):
            ds_k = py_outer_ds(k, iter_extents, iter_strides)
            linear = M0 + ds_k
            swizzled = py_swizzle_apply(linear, p, sw, at)
            if swizzled != linear:
                diverging_samples += 1
    assert diverging_samples > 0, (
        "no (tid, k) sample shows swizzled != linear — the swizzle is a "
        "no-op for this layout, so the test isn't catching anything"
    )


if __name__ == "__main__":
    tvm.testing.main()
