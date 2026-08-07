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
# pylint: disable=invalid-name, missing-function-docstring
"""Tests for the tcgen05 TMEM load/store (ld/st) dispatch.

Covers ``.32x32b`` tmem<->reg / smem<->tmem copy_async and bit-exact
``.16x{64,128,256}b`` atom verification. For each ``(shape, rep, dtype,
direction)`` in the ``.16x*`` sweeps we:

1. Fill a (128, FULL_W) host buffer ``A`` with random values.
2. Stage ``A`` into TMEM via the existing ``.32x32b`` ld/st round-trip.
3. Issue the new ``.16x*b`` atom via ``T.copy_async`` to read a (64, K_cols)
   fragment from TMEM into a register tile shaped by ``tcgen05_atom_layout``.
4. Dump the register tile to a ``(128, regs_per_thread)`` global buffer indexed
   ``B[tid_in_wg, r]``.
5. Reconstruct the expected ``B[t, r]`` on the host from the per-(lane, reg) →
   (frag_row, frag_col) formula. The M=64 fragment occupies TMEM lanes
   ``warp_id * 32 + (0..15)``, so ``frag_row R`` maps to TMEM lane
   ``(R // 16) * 32 + (R % 16)``.

For the store direction we run the inverse: prefill the register tile via host →
``B`` → ``.32x32b.ld``-staged read, write to TMEM via the new ``.16x*b.st``,
then read TMEM back via ``.32x32b.ld`` into a (128, FULL_W) buffer and check
that the M=64 fragment's row positions hold the expected register data.
"""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.backend.cuda.lang.alloc_pool import _default_tmem_layout
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx.layout import (
    S,
    TCol,
    TileLayout,
    TLane,
    tcgen05_atom_layout,
    tmem_datapath_layout,
)
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg

# --------------------------------------------------------------------------
# Shape metadata + host-side layout reconstruction
# --------------------------------------------------------------------------

# (.shape, .num) ranges supported by PTX Table 49.
_SHAPE_REPS = {
    "32x32b": (1, 2, 4, 8, 16, 32, 64, 128),
    "16x64b": (1, 2, 4, 8, 16, 32, 64, 128),
    "16x128b": (1, 2, 4, 8, 16, 32, 64),
    "16x256b": (1, 2, 4, 8, 16, 32),
}

# Per-warp fp32 column span = factor * rep.
_COL_FACTOR_FP32 = {"32x32b": 1, "16x64b": 2, "16x128b": 4, "16x256b": 8}

# Per-thread 32-bit register count = factor * rep.
_REGS_FACTOR = {"32x32b": 1, "16x64b": 1, "16x128b": 2, "16x256b": 4}

# Per-warpgroup fragment row count.
_FRAG_ROWS = {"32x32b": 128, "16x64b": 64, "16x128b": 64, "16x256b": 64}


def _decompose_fp32(shape: str, t: int, r: int) -> tuple[int, int]:
    """Return ``(frag_row, frag_col)`` in fp32 element units for the fp32 atom."""
    laneid = t & 31
    wid_in_wg = t >> 5
    if shape == "32x32b":
        # M=128 fragment: each thread t owns full row t with N consecutive cols.
        row = t
        col = r
    elif shape == "16x64b":
        t0 = laneid & 1
        t1 = (laneid >> 1) & 1
        t2 = laneid >> 2
        row = t2 + 8 * t0 + 16 * wid_in_wg
        col = t1 + 2 * r
    elif shape == "16x128b":
        t0 = laneid & 3
        t1 = laneid >> 2
        ra = r & 1
        rb = r >> 1
        row = t1 + 8 * ra + 16 * wid_in_wg
        col = t0 + 4 * rb
    elif shape == "16x256b":
        t0 = laneid & 3
        t1 = laneid >> 2
        v0p = r & 1
        va = (r >> 1) & 1
        vb = r >> 2
        row = t1 + 8 * va + 16 * wid_in_wg
        col = v0p + 2 * t0 + 8 * vb
    else:
        raise ValueError(shape)
    return row, col


def _frag_row_to_tmem_lane(shape: str, R: int) -> int:
    """Map fragment row R to its physical TMEM lane.

    For ``.32x32b`` (M=128) the mapping is identity: row R lives at TMEM lane R.
    For ``.16x*b`` (M=64) the fragment occupies the first 16 lanes of each
    warp's 32-lane slab, so ``R`` ∈ [0, 64) lives at lane ``(R // 16) * 32 + (R % 16)``.
    """
    if shape == "32x32b":
        return R
    return (R // 16) * 32 + (R % 16)


def _expected_reg_value_fp32(
    A: np.ndarray, shape: str, rep: int, tmem_col_off: int, t: int, r: int
) -> np.uint32:
    """fp32 path: return the bit-pattern (as uint32) that thread ``t`` register
    ``r`` should hold after ``.<shape>.x<rep>`` reads ``A`` (staged into TMEM) at
    column offset ``tmem_col_off``."""
    row, col = _decompose_fp32(shape, t, r)
    tmem_lane = _frag_row_to_tmem_lane(shape, row)
    val = np.float32(A[tmem_lane, tmem_col_off + col])
    return val.view(np.uint32)


def _expected_reg_value_16b(
    A: np.ndarray, shape: str, rep: int, tmem_col_off: int, t: int, r: int, dtype_np
) -> np.uint32:
    """16-bit path (fp16 / bf16 with .pack::16b): each fp32 register packs two
    16-bit elements at adjacent columns ``(2*col_fp32, 2*col_fp32 + 1)``."""
    row, col_fp32 = _decompose_fp32(shape, t, r)
    tmem_lane = _frag_row_to_tmem_lane(shape, row)
    lo = dtype_np(A[tmem_lane, tmem_col_off + 2 * col_fp32])
    hi = dtype_np(A[tmem_lane, tmem_col_off + 2 * col_fp32 + 1])
    lo_u16 = lo.view(np.uint16)
    hi_u16 = hi.view(np.uint16)
    return np.uint32(int(lo_u16) | (int(hi_u16) << 16))


# --------------------------------------------------------------------------
# Test 1: load direction
# --------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("shape", list(_SHAPE_REPS))
@pytest.mark.parametrize("rep", [1, 2, 4, 8, 16, 32])  # subset; full reps below
@pytest.mark.parametrize("dtype", ["float32"])
def test_tcgen05_ld_16xnb_load_fp32(shape, rep, dtype):
    """Bit-exact verification of ``tcgen05.<shape>.x<rep>.b32`` load."""
    if rep not in _SHAPE_REPS[shape]:
        pytest.skip(f"rep {rep} not valid for {shape}")
    _run_load_test(shape, rep, dtype)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize(
    "shape, rep",
    [
        ("16x64b", 64),
        ("16x64b", 128),
        ("16x128b", 64),
    ],
)
def test_tcgen05_ld_16xnb_load_fp32_large_rep(shape, rep):
    """High-rep entries that aren't in the parametrize-cross above."""
    _run_load_test(shape, rep, "float32")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("shape", list(_SHAPE_REPS))
@pytest.mark.parametrize("rep", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_tcgen05_16xnb_roundtrip_16b(shape, rep, dtype):
    """Self-consistent round-trip for 16-bit pack::16b path.

    The fp32 ``test_tcgen05_ld_16xnb_load_fp32`` already validates the
    ``(lane, reg) → (frag_row, frag_col)`` mapping bit-exactly against the
    standard ``.32x32b`` staging. For the 16-bit case the staging convention
    differs (``.32x32b.st`` packs two fp16 per 32-bit TMEM cell, whereas
    ``.16x*b.ld.pack::16b`` reads two fp16 from the LOW halves of adjacent
    32-bit cells), so we instead verify the new dispatch round-trips
    per-thread data via ``.16x*b.st.unpack::16b`` → ``.16x*b.ld.pack::16b``.
    A bit-exact round-trip is sufficient evidence that the per-thread
    register-layout matches between the load and store atom families.
    """
    if rep not in _SHAPE_REPS[shape]:
        pytest.skip(f"rep {rep} not valid for {shape}")
    _run_roundtrip_16b(shape, rep, dtype)


# ``.16x*b`` atom can also span M=128 by emitting two issues per copy_async
# (row=0 + row=16), covering the full 32-lane TMEM partition of each warp.
# We only need to spot-check that the dispatch fires correctly and the per-
# thread reg ↔ TMEM mapping round-trips bit-exactly — the M=64 sweep above
# already covers the (lane, reg) decomposition, so a sparse rep set suffices.
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("shape", ["16x64b", "16x128b", "16x256b"])
@pytest.mark.parametrize("rep", [1, 2, 4])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_tcgen05_16xnb_roundtrip_16b_M128(shape, rep, dtype):
    if rep not in _SHAPE_REPS[shape]:
        pytest.skip(f"rep {rep} not valid for {shape}")
    _run_roundtrip_16b(shape, rep, dtype, frag_rows_override=128)


# Layout F (M=64 non-``.ws``, scattered) round-trip: the buffer is declared
# with the scatter-encoded TileLayout that ``tmem_datapath_layout("F", ...)``
# produces. ``.16x*b`` M=64 PTX has the matching scatter built in, so the
# round-trip is bit-exact in the same way as Layout D + M=64.
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("shape", ["16x64b", "16x128b", "16x256b"])
@pytest.mark.parametrize("rep", [1, 2, 4])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_tcgen05_16xnb_roundtrip_16b_layout_F(shape, rep, dtype):
    if rep not in _SHAPE_REPS[shape]:
        pytest.skip(f"rep {rep} not valid for {shape}")
    _run_roundtrip_16b(shape, rep, dtype, tmem_datapath="F")


def _run_roundtrip_16b(
    shape: str,
    rep: int,
    dtype: str,
    *,
    frag_rows_override=None,
    tmem_datapath: str = "D",
):
    bits = tvm.runtime.DataType(dtype).bits
    assert bits == 16
    elem_per_32b = 2
    K_cols_fp32 = _COL_FACTOR_FP32[shape] * rep
    K_cols_elem = K_cols_fp32 * elem_per_32b
    regs_per_thread = _REGS_FACTOR[shape] * rep
    if frag_rows_override is not None:
        # M=128 doubles per-thread registers (second 16-row slab per warp).
        assert frag_rows_override == 128 and _FRAG_ROWS[shape] == 64
        regs_per_thread *= 2
    per_thread_elems = regs_per_thread * elem_per_32b
    frag_rows = frag_rows_override if frag_rows_override is not None else _FRAG_ROWS[shape]
    if tmem_datapath == "F":
        # Layout F is only valid with M=64 (per the datapath table); M=128
        # would need to read the high slab, which Layout F doesn't expose.
        assert frag_rows == 64, "Layout F + M=128 is an invalid pairing"
    tmem_rows = 64 if tmem_datapath == "F" else 128

    # The 16-bit round-trip writes and reads exclusively through .16x*b atoms,
    # so the TMEM column footprint is whatever ``K_cols_fp32`` says — no
    # .32x32b staging constraint applies here.
    tmem_col_width_32b = max(32, _next_pow2(K_cols_fp32))
    stage_width_elem = tmem_col_width_32b * elem_per_32b
    atom_view = tcgen05_atom_layout(shape, (frag_rows, K_cols_elem), dtype)
    tmem_layout = tmem_datapath_layout(tmem_datapath, tmem_rows, stage_width_elem)

    @T.prim_func
    def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
        # Per-thread input/output: A[tid_in_wg, i] feeds register slot i of the
        # warpgroup-collective fragment; B[tid_in_wg, i] is what comes back
        # after a .16x*b.st → .16x*b.ld round-trip.
        A = T.match_buffer(A_ptr, (128, per_thread_elems), dtype)
        B = T.match_buffer(B_ptr, (128, per_thread_elems), dtype)

        T.device_entry()
        warp_id = T.warp_id([128 // 32])
        T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        tid_in_wg = T.thread_id([128])

        tmem_addr = T.alloc_shared([1], "uint32")

        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    T.address_of(tmem_addr), T.uint32(tmem_col_width_32b)
                )

            T.tvm_storage_sync("shared")

            tmem = T.decl_buffer(
                (tmem_rows, stage_width_elem),
                dtype,
                scope="tmem",
                allocated_addr=tmem_addr[0],
                layout=tmem_layout,
            )

            # Load per-thread A → reg_in
            reg_in = T.alloc_local((per_thread_elems,), dtype)
            for i in range(per_thread_elems):
                reg_in[i] = A[tid_in_wg, i]
            T.cuda.cta_sync()

            # reg_in -> TMEM via .<shape>.x<rep>.st.unpack::16b
            frag_in = reg_in.view(frag_rows, K_cols_elem, layout=atom_view)
            Tx.wg.copy_async(tmem[0:frag_rows, 0:K_cols_elem], frag_in[:, :])
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.cuda.cta_sync()

            # TMEM -> reg_out via .<shape>.x<rep>.ld.pack::16b
            reg_out = T.alloc_local((per_thread_elems,), dtype)
            frag_out = reg_out.view(frag_rows, K_cols_elem, layout=atom_view)
            Tx.wg.copy_async(frag_out[:, :], tmem[0:frag_rows, 0:K_cols_elem])
            T.ptx.tcgen05.wait__ld.sync.aligned()
            T.cuda.cta_sync()
            for i in range(per_thread_elems):
                B[tid_in_wg, i] = reg_out[i]

            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    tmem_addr[0], T.uint32(tmem_col_width_32b)
                )

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, per_thread_elems))
        B_np = np.zeros((128, per_thread_elems), dtype=dtype)

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            # Round-trip should preserve every per-thread bit pattern.
            A_view = A.numpy().view(np.uint16)
            B_view = B.numpy().view(np.uint16)
            np.testing.assert_array_equal(B_view, A_view)

        tvm.testing.run_with_gpu_lock(run_and_check)


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


# Unit test: pin down the (row, col) → (TLane, TCol) mapping that the
# ``tmem_datapath_layout`` factory encodes. A self-consistent round-trip
# (write + read with the same factory output) can't catch a layout that
# encodes a *wrong* scatter — the labels would still match structurally
# even if the row→lane formula doesn't match PTX's actual behavior. This
# test bypasses compilation and checks the layout's ``apply`` method
# directly against ``_frag_row_to_tmem_lane`` for every M=64 logical row.
@pytest.mark.parametrize("sub_slab", [0, 1])
def test_tmem_datapath_layout_F_row_to_lane_mapping(sub_slab):
    """Layout F: every logical row r ∈ [0, 64) must land at physical TMEM
    lane ``(r // 16) * 32 + sub_slab * 16 + (r % 16)``. The default lower
    sub-slab is the canonical M=64 scatter; sub-slab 1 is its upper-half view.
    """
    cols = 32
    layout = tmem_datapath_layout("F", 64, cols, sub_slab=sub_slab)
    for r in range(64):
        for c in [0, 1, 7, 16, 31]:
            # Use ``apply(coord, shape=[64, cols])`` so (r, c) gets flattened
            # row-major before SplitCoord into the shard iters.
            axis_values = layout.apply(r, c, shape=[64, cols])
            expected_lane = (r // 16) * 32 + sub_slab * 16 + (r % 16)
            assert int(axis_values["TLane"]) == expected_lane, (
                f"(r={r}, c={c}) mapped to TLane {int(axis_values['TLane'])}, "
                f"expected {expected_lane} "
                f"(= (r//16)*32 + {sub_slab}*16 + (r%16))"
            )
            assert int(axis_values["TCol"]) == c, (
                f"(r={r}, c={c}) mapped to TCol {int(axis_values['TCol'])}, expected {c}"
            )


def test_tmem_datapath_layout_rejects_invalid_sub_slab():
    with pytest.raises(ValueError, match="sub_slab must be 0 or 1"):
        tmem_datapath_layout("F", 64, 8, sub_slab=2)
    with pytest.raises(ValueError, match="already spans both sub-slabs"):
        tmem_datapath_layout("D", 128, 8, sub_slab=1)


def test_default_tmem_layout_M128_matches_datapath_D():
    """The default accumulator layout and named Layout D must not drift."""
    tvm.ir.assert_structural_equal(
        _default_tmem_layout(128, 32), tmem_datapath_layout("D", 128, 32)
    )


@pytest.mark.parametrize("shape", ["16x64b", "16x128b", "16x256b"])
@pytest.mark.parametrize("rep", [1, 2, 4])
def test_tcgen05_atom_layout_apply_matches_decompose_fp32(shape, rep):
    """``tcgen05_atom_layout`` is supposed to be the inverse of
    ``_decompose_fp32`` — i.e. for every (row, col) in the M=64 fragment,
    ``layout.apply(row, col)`` must return the (laneid, wid_in_wg, m)
    tuple that PTX puts at frag element ``(row, col)``.

    The factory's per-shape iter lists are written low-to-high (natural
    decomposition); the reversal added below is what aligns the resulting
    TileLayout with ``SplitCoord`` (high-to-low). Without the reversal the
    factory used to silently produce a layout that disagreed with PTX —
    the round-trip tests didn't catch it because the dispatch ignores the
    layout label and emits raw PTX. This sweep is the structural fence.
    """
    if rep not in _SHAPE_REPS[shape]:
        pytest.skip(f"rep {rep} not valid for {shape}")
    cols = _COL_FACTOR_FP32[shape] * rep  # K_cols_fp32
    layout = tcgen05_atom_layout(shape, (64, cols), "float32")
    for thread in range(128):
        laneid = thread & 31
        wid_in_wg = thread >> 5
        regs_per_thread = _REGS_FACTOR[shape] * rep
        for reg in range(regs_per_thread):
            row, col = _decompose_fp32(shape, thread, reg)
            axis_values = layout.apply(row, col, shape=[64, cols])
            assert int(axis_values.get("laneid", 0)) == laneid, (
                f"shape={shape} rep={rep}: (row={row}, col={col}) "
                f"mapped to laneid {int(axis_values.get('laneid', 0))}, expected {laneid}"
            )
            assert int(axis_values.get("wid_in_wg", 0)) == wid_in_wg, (
                f"shape={shape} rep={rep}: (row={row}, col={col}) "
                f"mapped to wid_in_wg {int(axis_values.get('wid_in_wg', 0))}, expected {wid_in_wg}"
            )
            assert int(axis_values.get("m", 0)) == reg, (
                f"shape={shape} rep={rep}: (row={row}, col={col}) "
                f"mapped to m {int(axis_values.get('m', 0))}, expected {reg}"
            )


def test_tmem_datapath_layout_D_row_to_lane_mapping():
    """Layout D: identity row→lane (no scatter)."""
    cols = 32
    layout = tmem_datapath_layout("D", 128, cols)
    for r in [0, 1, 15, 16, 31, 32, 63, 64, 127]:
        axis_values = layout.apply(r, 0, shape=[128, cols])
        assert int(axis_values["TLane"]) == r, (
            f"r={r} mapped to TLane {int(axis_values['TLane'])}, expected {r}"
        )


def test_tmem_datapath_layout_B_col_split_mapping():
    """Layout B splits N/2 columns across each 64-lane half."""
    n_cols = 128
    n_half = n_cols // 2
    layout = tmem_datapath_layout("B", 64, n_cols)

    for row in [0, 1, 31, 63]:
        for col in [0, 1, n_half - 1, n_half, n_half + 1, n_cols - 1]:
            axis_values = layout.apply(row, col, shape=[64, n_cols])
            assert int(axis_values["TLane"]) == row + 64 * (col // n_half)
            assert int(axis_values["TCol"]) == col % n_half


def test_tcgen05_atom_layout_32x32b_datapath_B_mapping():
    """The Layout B register image mirrors TLane/TCol as tid_in_wg/m."""
    n_cols = 128
    n_half = n_cols // 2
    layout = tcgen05_atom_layout("32x32b", (64, n_cols), "float32")

    for row in [0, 1, 31, 63]:
        for col in [0, 1, n_half - 1, n_half, n_half + 1, n_cols - 1]:
            axis_values = layout.apply(row, col, shape=[64, n_cols])
            assert int(axis_values["tid_in_wg"]) == row + 64 * (col // n_half)
            assert int(axis_values["m"]) == col % n_half


def test_datapath_B_layout_factories_reject_invalid_inputs():
    with pytest.raises(ValueError, match="expects rows=64"):
        tmem_datapath_layout("B", 128, 32)
    with pytest.raises(ValueError, match="expects even cols"):
        tmem_datapath_layout("B", 64, 31)
    with pytest.raises(ValueError, match="sub_slab must be 0"):
        tmem_datapath_layout("B", 64, 32, sub_slab=1)
    with pytest.raises(ValueError, match="fp32-only"):
        tcgen05_atom_layout("32x32b", (64, 32), "float16")
    with pytest.raises(ValueError, match="expects even N"):
        tcgen05_atom_layout("32x32b", (64, 31), "float32")
    with pytest.raises(ValueError, match="PTX Table 49"):
        tcgen05_atom_layout("32x32b", (64, 6), "float32")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("shape,rep", [("16x256b", 4), ("16x128b", 4), ("16x64b", 8)])
def test_tcgen05_16xnb_sub_slab_view_read(shape, rep):
    """F sub-slab views split a known-correct M=128 read into low/high halves."""
    dtype = "float32"
    cols = _COL_FACTOR_FP32[shape] * rep
    regs128 = _REGS_FACTOR[shape] * rep * 2
    regs64 = _REGS_FACTOR[shape] * rep
    tmem_cols = max(32, _next_pow2(cols))
    atom128 = tcgen05_atom_layout(shape, (128, cols), dtype)
    atom64 = tcgen05_atom_layout(shape, (64, cols), dtype)
    layout_d = tmem_datapath_layout("D", 128, tmem_cols)
    layout_f0 = tmem_datapath_layout("F", 64, tmem_cols, sub_slab=0)
    layout_f1 = tmem_datapath_layout("F", 64, tmem_cols, sub_slab=1)

    @T.prim_func
    def kernel(A_ptr: T.handle, B128_ptr: T.handle, B0_ptr: T.handle, B1_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, regs128), dtype)
        B128 = T.match_buffer(B128_ptr, (128, regs128), dtype)
        B0 = T.match_buffer(B0_ptr, (128, regs64), dtype)
        B1 = T.match_buffer(B1_ptr, (128, regs64), dtype)
        T.device_entry()
        warp_id = T.warp_id([4])
        T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        tid = T.thread_id([128])
        tmem_addr = T.alloc_shared([1], "uint32")
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    T.address_of(tmem_addr), T.uint32(tmem_cols)
                )
            T.tvm_storage_sync("shared")
            tmem_d = T.decl_buffer(
                (128, tmem_cols),
                dtype,
                scope="tmem",
                allocated_addr=tmem_addr[0],
                layout=layout_d,
            )
            tmem_f0 = T.decl_buffer(
                (64, tmem_cols),
                dtype,
                scope="tmem",
                allocated_addr=tmem_addr[0],
                layout=layout_f0,
            )
            tmem_f1 = T.decl_buffer(
                (64, tmem_cols),
                dtype,
                scope="tmem",
                allocated_addr=tmem_addr[0],
                layout=layout_f1,
            )
            source = T.alloc_local((regs128,), dtype)
            for i in range(regs128):
                source[i] = A[tid, i]
            T.cuda.cta_sync()
            Tx.wg.copy_async(tmem_d[0:128, 0:cols], source.view(128, cols, layout=atom128))
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.cuda.cta_sync()

            full = T.alloc_local((regs128,), dtype)
            Tx.wg.copy_async(full.view(128, cols, layout=atom128), tmem_d[0:128, 0:cols])
            T.ptx.tcgen05.wait__ld.sync.aligned()
            T.cuda.cta_sync()
            for i in range(regs128):
                B128[tid, i] = full[i]

            lower = T.alloc_local((regs64,), dtype)
            Tx.wg.copy_async(lower.view(64, cols, layout=atom64), tmem_f0[0:64, 0:cols])
            T.ptx.tcgen05.wait__ld.sync.aligned()
            T.cuda.cta_sync()
            for i in range(regs64):
                B0[tid, i] = lower[i]

            upper = T.alloc_local((regs64,), dtype)
            Tx.wg.copy_async(upper.view(64, cols, layout=atom64), tmem_f1[0:64, 0:cols])
            T.ptx.tcgen05.wait__ld.sync.aligned()
            T.cuda.cta_sync()
            for i in range(regs64):
                B1[tid, i] = upper[i]

            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    tmem_addr[0], T.uint32(tmem_cols)
                )

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
        source_np = tvm.testing.generate_random_array(dtype, (128, regs128))

        def run_and_check():
            dev = tvm.cuda(0)
            source_dev = tvm.runtime.tensor(source_np, dev)
            full_dev = tvm.runtime.tensor(np.zeros((128, regs128), dtype), dev)
            lower_dev = tvm.runtime.tensor(np.zeros((128, regs64), dtype), dev)
            upper_dev = tvm.runtime.tensor(np.zeros((128, regs64), dtype), dev)
            mod(source_dev, full_dev, lower_dev, upper_dev)
            full_np = full_dev.numpy()
            np.testing.assert_array_equal(lower_dev.numpy(), full_np[:, :regs64])
            np.testing.assert_array_equal(upper_dev.numpy(), full_np[:, regs64:])

        tvm.testing.run_with_gpu_lock(run_and_check)


# Negative tests: the datapath/atom pairing matrix in ``tcgen05_ldst.py``
# must reject mismatched combinations. We construct a Layout F TMEM buffer
# (64 rows, scattered) and try to read it with a ``.16x*b`` M=128 atom,
# which would interpret the second slab (lanes 16..31 of each warp) as
# meaningful data — but Layout F leaves that slab undefined. Compilation
# must raise a clear error, not silently emit a broken kernel.
@pytest.mark.parametrize("atom_kind,frag_rows", [("16x*b", 128), ("32x32b", 128)])
def test_layout_F_rejects_incompatible_atoms(atom_kind, frag_rows):
    """Layout F + (.16x*b M=128 or .32x32b) must raise at compile time."""
    if atom_kind == "16x*b":
        shape = "16x256b"
        rep = 1
        # Local fragment shape for M=128 .16x256b rep=1 = (128, 8) fp32.
        atom_view = tcgen05_atom_layout(shape, (128, 8), "float32")
        local_extent_rows = 128
        local_cols = 8
    else:  # .32x32b path: local (128, 32) fp32
        atom_view = TileLayout(S[(128, 32) : (1 @ axis_tid_in_wg, 1)])
        local_extent_rows = 128
        local_cols = 32

    tmem_layout = tmem_datapath_layout("F", 64, max(32, local_cols))
    tmem_rows = 64
    stage_width_elem = max(32, local_cols)

    @T.prim_func
    def kernel() -> None:
        T.device_entry()
        T.warp_id([128 // 32])
        T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        T.thread_id([128])
        tmem_addr = T.alloc_shared([1], "uint32")
        if wg_id == 0:
            T.tvm_storage_sync("shared")
            tmem = T.decl_buffer(
                (tmem_rows, stage_width_elem),
                "float32",
                scope="tmem",
                allocated_addr=tmem_addr[0],
                layout=tmem_layout,
            )
            frag = T.alloc_local((local_extent_rows * local_cols // 128,), "float32")
            frag_view = frag.view(local_extent_rows, local_cols, layout=atom_view)
            Tx.wg.copy_async(frag_view[:, :], tmem[0:local_extent_rows, 0:local_cols])

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        with pytest.raises((ValueError, RuntimeError), match="datapath"):
            tvm.compile(mod, target=target, tir_pipeline="tirx")


def test_layout_B_rejects_16xnb_fragment():
    """Layout B must not silently take the ordinary M=64 .16x*b path."""
    n_cols = 128
    tmem_layout = tmem_datapath_layout("B", 64, n_cols)
    wrong_layout = tcgen05_atom_layout("16x256b", (64, n_cols), "float32")

    @T.prim_func
    def kernel() -> None:
        T.device_entry()
        T.cta_id([1])
        T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        tmem_addr = T.alloc_shared([1], "uint32")
        tmem = T.decl_buffer(
            (64, n_cols),
            "float32",
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=tmem_layout,
        )
        frag = T.alloc_local((n_cols // 2,), "float32")
        frag_view = frag.view(64, n_cols, layout=wrong_layout)
        Tx.wg.copy_async(frag_view[:, :], tmem[:, :])

    target = tvm.target.Target("cuda")
    with target:
        with pytest.raises((ValueError, RuntimeError), match="datapath B"):
            tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")


def test_layout_B_rejects_partial_column_copy():
    """A logical B column slice is not one contiguous physical tcol interval."""
    n_cols = 64

    @T.prim_func
    def kernel() -> None:
        T.device_entry()
        T.cta_id([1])
        T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        tmem_addr = T.alloc_shared([1], "uint32")
        tmem = T.decl_buffer(
            (64, n_cols),
            "float32",
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=tmem_datapath_layout("B", 64, n_cols),
        )
        frag = T.alloc_tcgen05_ldst_frag("32x32b", (64, n_cols), "float32")
        Tx.wg.copy_async(frag[:, : n_cols // 2], tmem[:, : n_cols // 2])

    target = tvm.target.Target("cuda")
    with target:
        with pytest.raises((ValueError, RuntimeError), match=r"full \(64, N\)"):
            tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")


@pytest.mark.parametrize("direction", ["ld", "st"])
def test_datapath_B_codegen(direction):
    """Both directions emit one physical .32x32b.x32 instruction."""
    n_cols = 64

    @T.prim_func
    def kernel() -> None:
        T.device_entry()
        T.cta_id([1])
        T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        tmem_addr = T.alloc_shared([1], "uint32")
        tmem = T.decl_buffer(
            (64, n_cols),
            "float32",
            scope="tmem",
            allocated_addr=tmem_addr[0] + 32,
            layout=tmem_datapath_layout("B", 64, n_cols),
        )
        frag = T.alloc_tcgen05_ldst_frag("32x32b", (64, n_cols), "float32")
        if direction == "ld":
            Tx.wg.copy_async(frag[:, :], tmem[:, :])
        else:
            Tx.wg.copy_async(tmem[:, :], frag[:, :])

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
    source = mod.mod.imports[0].inspect_source()
    assert f"tcgen05.{direction}" in source
    assert "32x32b.x32" in source


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("n_cols", [32, 64, 128, 256])
@pytest.mark.parametrize("col_offset", [0, 32])
def test_datapath_B_ld_st_roundtrip(n_cols, col_offset):
    """Layout B store/load preserves every register, including a nonzero base."""
    n_half = n_cols // 2
    tmem_cols = _next_pow2(max(32, col_offset + n_half))

    @T.prim_func
    def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, n_half), "float32")
        B = T.match_buffer(B_ptr, (128, n_half), "float32")
        T.device_entry()
        warp_id = T.warp_id([4])
        T.cta_id([1])
        wg_id = T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        tid = T.thread_id_in_wg([128])
        tmem_addr = T.alloc_shared([1], "uint32")

        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    T.address_of(tmem_addr), T.uint32(tmem_cols)
                )
            T.tvm_storage_sync("shared")
            tmem = T.decl_buffer(
                (64, n_cols),
                "float32",
                scope="tmem",
                allocated_addr=tmem_addr[0] + col_offset,
                layout=tmem_datapath_layout("B", 64, n_cols),
            )

            frag_in = T.alloc_tcgen05_ldst_frag("32x32b", (64, n_cols), "float32")
            frag_in_local = frag_in.local()
            for i in range(n_half):
                frag_in_local[i] = A[tid, i]
            T.cuda.cta_sync()
            Tx.wg.copy_async(tmem[:, :], frag_in[:, :])
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.cuda.cta_sync()

            frag_out = T.alloc_tcgen05_ldst_frag("32x32b", (64, n_cols), "float32")
            Tx.wg.copy_async(frag_out[:, :], tmem[:, :])
            T.ptx.tcgen05.wait__ld.sync.aligned()
            T.cuda.cta_sync()
            frag_out_local = frag_out.local()
            for i in range(n_half):
                B[tid, i] = frag_out_local[i]

            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    tmem_addr[0], T.uint32(tmem_cols)
                )

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
        source_np = tvm.testing.generate_random_array("float32", (128, n_half))

        def run_and_check():
            dev = tvm.cuda(0)
            source = tvm.runtime.tensor(source_np, dev)
            result = tvm.runtime.tensor(np.zeros((128, n_half), dtype="float32"), dev)
            mod(source, result)
            np.testing.assert_array_equal(result.numpy(), source_np)

        tvm.testing.run_with_gpu_lock(run_and_check)


def _run_load_test(shape: str, rep: int, dtype: str):
    """Stage A into TMEM via .32x32b, then read it back as the fragment via
    .<shape>.x<rep> (through ``T.alloc_tcgen05_ldst_frag``), and compare each
    thread's registers against the expected layout-derived value."""
    bits = tvm.runtime.DataType(dtype).bits
    elem_per_32b = 32 // bits
    # Per-warp fp32 col span x number of warps in one warpgroup covers the
    # fragment column footprint. The TMEM allocation is sized for the same
    # element-column count.
    K_cols_fp32 = _COL_FACTOR_FP32[shape] * rep
    K_cols_elem = K_cols_fp32 * elem_per_32b
    regs_per_thread = _REGS_FACTOR[shape] * rep  # 32-bit register count
    per_thread_elems = regs_per_thread * elem_per_32b
    frag_rows = _FRAG_ROWS[shape]

    tmem_col_width_32b = max(32, _next_pow2(K_cols_fp32))

    # Staging via .32x32b caps at num=128 (= 128 fp32 cols) per atom call. For
    # configs whose K_cols_fp32 exceeds 128 we split the stage into multiple
    # chunks of CHUNK_FP32 fp32 cols each.
    CHUNK_FP32 = 128
    chunk_elem = CHUNK_FP32 * elem_per_32b
    num_chunks = tmem_col_width_32b // CHUNK_FP32 if tmem_col_width_32b > CHUNK_FP32 else 1
    chunk_width_32b = tmem_col_width_32b if num_chunks == 1 else CHUNK_FP32
    chunk_width_elem = chunk_width_32b * elem_per_32b
    stage_width_elem = tmem_col_width_32b * elem_per_32b

    # Vector length for global<->local copies (in elements).
    VEC_LEN = 128 // bits
    if stage_width_elem % VEC_LEN != 0:
        pytest.skip(f"stage_width_elem {stage_width_elem} % VEC_LEN {VEC_LEN} != 0")

    g_layout = TileLayout(
        S[(128, stage_width_elem // VEC_LEN, VEC_LEN) : (stage_width_elem, VEC_LEN, 1)]
    )
    chunk_view = TileLayout(S[(128, chunk_width_elem) : (1 @ axis_tid_in_wg, 1)])
    # The factory + wrapper both go through ``tcgen05_atom_layout``; we use it
    # explicitly here so that ``frag_local`` has the canonical layout that
    # ``T.copy_async`` matches when dispatching to the right atom path.
    atom_view = tcgen05_atom_layout(shape, (frag_rows, K_cols_elem), dtype)

    @T.prim_func
    def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
        # A is the host data we stage into TMEM via the standard .32x32b path.
        A = T.match_buffer(A_ptr, (128, stage_width_elem), dtype)
        # B is a per-thread register dump: B[tid_in_wg, reg_idx_in_elements].
        B = T.match_buffer(B_ptr, (128, per_thread_elems), dtype)

        A_flat = A.view(-1)

        T.device_entry()
        warp_id = T.warp_id([128 // 32])
        T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        tid_in_wg = T.thread_id([128])

        tmem_addr = T.alloc_shared([1], "uint32")

        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    T.address_of(tmem_addr), T.uint32(tmem_col_width_32b)
                )

            T.tvm_storage_sync("shared")

            tmem = T.decl_buffer(
                (128, stage_width_elem),
                dtype,
                scope="tmem",
                allocated_addr=tmem_addr[0],
                layout=TileLayout(S[(128, stage_width_elem) : (1 @ TLane, 1 @ TCol)]),
            )

            # Per-thread chunk staging buffer (CHUNK_FP32 fp32 worth).
            stage_reg = T.alloc_local((chunk_width_elem,), dtype)
            stage_local = stage_reg.view(128, chunk_width_elem, layout=chunk_view)

            # Walk chunks: A[:, ck:ck+chunk] -> stage_reg -> TMEM[:, ck:ck+chunk]
            for chunk_idx in range(num_chunks):
                col_off_elem = chunk_idx * chunk_width_elem
                for i in range(chunk_width_elem // VEC_LEN):
                    # Each thread's row offset in A_flat: stage_width_elem; within
                    # the row, this chunk starts at col_off_elem and each vector
                    # picks up VEC_LEN elements at slot i.
                    g_offset = T.meta_var(tid_in_wg * stage_width_elem + col_off_elem + i * VEC_LEN)
                    Tx.copy(
                        stage_reg[i * VEC_LEN : i * VEC_LEN + VEC_LEN],
                        A_flat[g_offset : g_offset + VEC_LEN],
                    )
                T.cuda.cta_sync()
                Tx.wg.copy_async(
                    tmem[:, col_off_elem : col_off_elem + chunk_width_elem],
                    stage_local[:, :],
                )
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.cuda.cta_sync()

            # TMEM[0:frag_rows, 0:K_cols] -> frag_local via .<shape>.x<rep>.ld.
            # Use ``tcgen05_atom_layout`` so dispatch matches the new path
            # (or stays on .32x32b for instr_shape="32x32b"). Keep the flat
            # ``frag_reg`` for the per-thread dump below.
            frag_reg = T.alloc_local((per_thread_elems,), dtype)
            frag_local = frag_reg.view(frag_rows, K_cols_elem, layout=atom_view)
            Tx.wg.copy_async(frag_local[:, :], tmem[0:frag_rows, 0:K_cols_elem])
            T.ptx.tcgen05.wait__ld.sync.aligned()
            T.cuda.cta_sync()
            for i in range(per_thread_elems):
                B[tid_in_wg, i] = frag_reg[i]

            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    tmem_addr[0], T.uint32(tmem_col_width_32b)
                )

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, stage_width_elem))
        B_np = np.zeros((128, per_thread_elems), dtype=dtype)

    # Build the expected register dump from the layout before acquiring the GPU.
    if bits == 32:
        # Each register slot in B[t, r] holds a single fp32; compare bit-exactly.
        B_expected = np.zeros((128, per_thread_elems), dtype=np.uint32)
        for t in range(128):
            for r in range(regs_per_thread):
                B_expected[t, r] = _expected_reg_value_fp32(A_np, shape, rep, 0, t, r)
    else:
        # B[t, :] holds per_thread_elems 16-bit values; each fp32 register packs
        # two of them in (low, high) order. Compare bit-exactly via uint32 view.
        dtype_np = np.float16 if dtype == "float16" else np.dtype("bfloat16")
        if dtype == "bfloat16":
            # numpy doesn't have a stable bfloat16 dtype across versions; use ml_dtypes.
            try:
                from ml_dtypes import bfloat16 as _bf16

                dtype_np = _bf16
            except ImportError:
                pytest.skip("bfloat16 verification needs ml_dtypes")
        B_expected = np.zeros((128, regs_per_thread), dtype=np.uint32)
        for t in range(128):
            for r in range(regs_per_thread):
                B_expected[t, r] = _expected_reg_value_16b(A_np, shape, rep, 0, t, r, dtype_np)

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)
        B_view = B.numpy().view(np.uint32)
        if bits != 32:
            B_view = B_view.reshape(128, regs_per_thread)
        np.testing.assert_array_equal(B_view, B_expected)

    tvm.testing.run_with_gpu_lock(run_and_check)


# --------------------------------------------------------------------------
# Test 2: store direction (mirror of test 1, with .st instead of .ld)
# --------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("shape", list(_SHAPE_REPS))
@pytest.mark.parametrize("rep", [1, 4, 16])
@pytest.mark.parametrize("dtype", ["float32"])
def test_tcgen05_st_16xnb_store(shape, rep, dtype):
    """Round-trip test: write the M=64 fragment via .<shape>.x<rep>.st then read
    via the standard .32x32b path; verify the host-known fragment data ends up
    at the expected TMEM lane positions.

    Only fp32 here — the 16-bit case has a different staging convention
    (pack::16b reads/writes the LOW halves of adjacent cells, not low/high of
    one cell) and is covered by ``test_tcgen05_16xnb_roundtrip_16b`` via a
    self-consistent .16x*b.st → .16x*b.ld loop.
    """
    if rep not in _SHAPE_REPS[shape]:
        pytest.skip(f"rep {rep} not valid for {shape}")
    bits = tvm.runtime.DataType(dtype).bits
    elem_per_32b = 32 // bits
    K_cols_fp32 = _COL_FACTOR_FP32[shape] * rep
    K_cols_elem = K_cols_fp32 * elem_per_32b
    regs_per_thread = _REGS_FACTOR[shape] * rep
    per_thread_elems = regs_per_thread * elem_per_32b
    frag_rows = _FRAG_ROWS[shape]

    tmem_col_width_32b = max(32, _next_pow2(K_cols_fp32))
    if tmem_col_width_32b > 128:
        pytest.skip(
            f"tmem_col_width_32b {tmem_col_width_32b} > 128 not supported by .32x32b staging"
        )
    stage_width_elem = tmem_col_width_32b * elem_per_32b
    VEC_LEN = 128 // bits
    if stage_width_elem % VEC_LEN != 0:
        pytest.skip(f"stage_width_elem {stage_width_elem} % VEC_LEN {VEC_LEN} != 0")

    g_layout = TileLayout(
        S[(128, stage_width_elem // VEC_LEN, VEC_LEN) : (stage_width_elem, VEC_LEN, 1)]
    )
    stage_view = TileLayout(S[(128, stage_width_elem) : (1 @ axis_tid_in_wg, 1)])
    atom_view = tcgen05_atom_layout(shape, (frag_rows, K_cols_elem), dtype)

    @T.prim_func
    def kernel(A_ptr: T.handle, B_ptr: T.handle) -> None:
        # A[tid_in_wg, i] is the i-th per-thread element to feed into the atom store.
        A = T.match_buffer(A_ptr, (128, per_thread_elems), dtype)
        # B[lane, col] is the TMEM-staged readout after the round-trip.
        B = T.match_buffer(B_ptr, (128, stage_width_elem), dtype)
        B_flat = B.view(-1)

        T.device_entry()
        warp_id = T.warp_id([128 // 32])
        T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        tid_in_wg = T.thread_id([128])

        tmem_addr = T.alloc_shared([1], "uint32")

        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    T.address_of(tmem_addr), T.uint32(tmem_col_width_32b)
                )

            T.tvm_storage_sync("shared")

            tmem = T.decl_buffer(
                (128, stage_width_elem),
                dtype,
                scope="tmem",
                allocated_addr=tmem_addr[0],
                layout=TileLayout(S[(128, stage_width_elem) : (1 @ TLane, 1 @ TCol)]),
            )

            # Load per-thread A → frag_reg
            frag_reg = T.alloc_local((per_thread_elems,), dtype)
            for i in range(per_thread_elems):
                frag_reg[i] = A[tid_in_wg, i]
            T.cuda.cta_sync()

            # frag_local -> TMEM via .<shape>.x<rep>.st
            frag_local = frag_reg.view(frag_rows, K_cols_elem, layout=atom_view)
            Tx.wg.copy_async(tmem[0:frag_rows, 0:K_cols_elem], frag_local[:, :])
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.cuda.cta_sync()

            # TMEM -> readout via .32x32b.ld
            stage_reg = T.alloc_local((stage_width_elem,), dtype)
            stage_local = stage_reg.view(128, stage_width_elem, layout=stage_view)
            Tx.wg.copy_async(stage_local[:, :], tmem[:, :])
            T.ptx.tcgen05.wait__ld.sync.aligned()
            T.cuda.cta_sync()
            for i in range(stage_width_elem // VEC_LEN):
                g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                Tx.copy(
                    B_flat[g_offset : g_offset + VEC_LEN],
                    stage_reg[i * VEC_LEN : i * VEC_LEN + VEC_LEN],
                )

            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    tmem_addr[0], T.uint32(tmem_col_width_32b)
                )

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, per_thread_elems))
        B_np = np.zeros((128, stage_width_elem), dtype=dtype)

    # Build expected TMEM staging: only rows that the M=64 fragment writes to
    # should match A's per-thread data; other rows are untouched (we set B_np to
    # zero and the .32x32b.ld reads whatever the TMEM allocator left, which may
    # be arbitrary, so only check the fragment positions).
    expected_values = []
    if bits == 32:
        for t in range(128):
            for r in range(regs_per_thread):
                row, col = _decompose_fp32(shape, t, r)
                tmem_lane = _frag_row_to_tmem_lane(shape, row)
                expected = np.float32(A_np[t, r]).view(np.uint32)
                expected_values.append((tmem_lane, col, expected, t, r, row))
    else:
        # 16-bit: each fp32 reg packs two 16-bit elements at adjacent TMEM cols.
        if dtype != "float16":
            pytest.skip("16b store check restricted to float16")
        for t in range(128):
            for r in range(regs_per_thread):
                row, col_fp32 = _decompose_fp32(shape, t, r)
                tmem_lane = _frag_row_to_tmem_lane(shape, row)
                lo = np.float16(A_np[t, 2 * r]).view(np.uint16)
                hi = np.float16(A_np[t, 2 * r + 1]).view(np.uint16)
                expected_values.append((tmem_lane, 2 * col_fp32, lo, t, r, row))
                expected_values.append((tmem_lane, 2 * col_fp32 + 1, hi, t, r, row))

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)
        view = B.numpy().view(np.uint32 if bits == 32 else np.uint16)
        for tmem_lane, col, expected, t, r, row in expected_values:
            assert view[tmem_lane, col] == expected, (
                f"{shape}.x{rep} {dtype}: thread {t} reg {r} → "
                f"(row={row}, col={col}) tmem_lane={tmem_lane} got "
                f"{view[tmem_lane, col]:#x} want {expected:#x}"
            )

    tvm.testing.run_with_gpu_lock(run_and_check)


# --------------------------------------------------------------------------
# Wrapper test: exercise T.alloc_tcgen05_ldst_frag directly (compile-only smoke).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, frag_rows, K_cols",
    [
        ("32x32b", 128, 32),  # .32x32b.x32 fp32: simple thread-rows layout
        ("32x32b", 128, 64),  # .32x32b.x64 fp32
        ("16x64b", 64, 64),  # .16x64b.x32 fp32
        ("16x128b", 64, 64),  # .16x128b.x16 fp32
        ("16x256b", 64, 64),  # .16x256b.x8 fp32
    ],
)
@pytest.mark.gpu
def test_alloc_tcgen05_frag_wrapper_compiles(shape, frag_rows, K_cols):
    """Ensure T.alloc_tcgen05_ldst_frag yields a buffer that ``T.copy_async`` accepts
    and lowers to the correct tcgen05 atom for each supported instr_shape."""

    @T.prim_func
    def kernel(A_ptr: T.handle) -> None:
        T.match_buffer(A_ptr, (128, K_cols), "float32")
        T.device_entry()
        warp_id = T.warp_id([4])
        T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        T.thread_id([128])

        tmem_addr = T.alloc_shared([1], "uint32")
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    T.address_of(tmem_addr), T.uint32(max(32, K_cols))
                )
            T.tvm_storage_sync("shared")
            tmem = T.decl_buffer(
                (128, K_cols),
                "float32",
                scope="tmem",
                allocated_addr=tmem_addr[0],
                layout=TileLayout(S[(128, K_cols) : (1 @ TLane, 1 @ TCol)]),
            )
            # One-liner: wrapper handles per-thread storage + layout.
            frag = T.alloc_tcgen05_ldst_frag(shape, (frag_rows, K_cols), "float32")
            Tx.wg.copy_async(frag[:, :], tmem[0:frag_rows, 0:K_cols])
            T.ptx.tcgen05.wait__ld.sync.aligned()
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    tmem_addr[0], T.uint32(max(32, K_cols))
                )

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    # Compiles cleanly + the generated CUDA contains the expected PTX shape.
    src = mod.mod.imports[0].inspect_source()
    assert shape in src, (
        f"expected .{shape}.x? in generated PTX, but `{shape}` not found in CUDA source"
    )


def test_tcgen05_32x32b_float32_keeps_typed_register_operands():
    """The .32x32b float32 lowering should pass the local fragment's typed
    registers to the PTX helper. The helper reinterprets operands as b32, so a
    call-site uint32 view is unnecessary and makes generated CUDA diverge from
    handwritten ``T.ptx.tcgen05`` calls."""

    K_cols = 32

    @T.prim_func
    def kernel(A_ptr: T.handle) -> None:
        T.match_buffer(A_ptr, (128, K_cols), "float32")
        T.device_entry()
        warp_id = T.warp_id([4])
        T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        T.thread_id([128])

        tmem_addr = T.alloc_shared([1], "uint32")
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    T.address_of(tmem_addr), T.uint32(K_cols)
                )
            T.tvm_storage_sync("shared")
            tmem = T.decl_buffer(
                (128, K_cols),
                "float32",
                scope="tmem",
                allocated_addr=tmem_addr[0],
                layout=TileLayout(S[(128, K_cols) : (1 @ TLane, 1 @ TCol)]),
            )
            frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, K_cols), "float32")
            Tx.wg.copy_async(frag[:, :], tmem[:, :])
            T.ptx.tcgen05.wait__ld.sync.aligned()
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(K_cols))

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    call_lines = [
        line
        for line in src.splitlines()
        if "ptx_tcgen05_ld_ld_sync_aligned_32x32b_x32_b32_f32(" in line
        and "__forceinline__" not in line
    ]
    assert call_lines
    assert "((uint*)" not in call_lines[0]


def test_tcgen05_ldst_constant_tmem_address_is_uint32():
    """A constant TMEM base address must still be passed as uint32.

    Sparse FlashMLA uses pool-relative constant TMEM columns. Handwritten
    ``T.ptx.tcgen05`` calls pass ``T.uint32(0)`` for the base address; the tile
    primitive should emit the same ABI shape instead of a bare integer literal.
    """

    K_cols = 32

    @T.prim_func
    def kernel() -> None:
        T.device_entry()
        T.warp_id([4])
        T.cta_id([1])
        wg_id = T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        T.thread_id([128])

        if wg_id == 0:
            tmem = T.decl_buffer(
                (128, K_cols),
                "float32",
                scope="tmem",
                allocated_addr=0,
                layout=TileLayout(S[(128, K_cols) : (1 @ TLane, 1 @ TCol)]),
            )
            frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, K_cols), "float32")
            Tx.wg.copy_async(frag[:, :], tmem[:, :])
            T.ptx.tcgen05.wait__ld.sync.aligned()
            Tx.wg.copy_async(tmem[:, :], frag[:, :])
            T.ptx.tcgen05.wait__st.sync.aligned()

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    ld_lines = [
        line
        for line in src.splitlines()
        if "ptx_tcgen05_ld_ld_sync_aligned_32x32b_x32_b32_f32(" in line
        and "__forceinline__" not in line
    ]
    st_lines = [
        line
        for line in src.splitlines()
        if "ptx_tcgen05_st_st_sync_aligned_32x32b_x32_b32_f32(" in line
        and "__forceinline__" not in line
    ]
    assert ld_lines
    assert st_lines
    assert "(uint)0" in ld_lines[0]
    assert "(uint)0" in st_lines[0]


# --------------------------------------------------------------------------
# Test 3: column-slice loads of a wider frag
#
# An epilogue may allocate one wide ``(128, K)`` frag and load it from TMEM in
# EPI_TILE-wide column chunks (``frag[:, c:c+w]``) so all loads are in flight
# before a single ``wait.ld``. The ``.16x*b`` dispatch must emit each slice as
# its own atom (``num_eff`` derived from the slice width) at the correct
# per-slab register offset. We verify this is *bit-exact identical* to one
# full-width load of the same frag — which the sweeps above already validate
# against the layout-derived expectation. M=128 here exercises the 2-slab path
# (the slice's two slabs live ``regs_per_thread_per_slab`` apart, not adjacent).
# --------------------------------------------------------------------------


def _run_sliced_vs_full_load(shape, full_rep, n_chunks):
    dtype = "float32"
    K_cols_fp32 = _COL_FACTOR_FP32[shape] * full_rep
    assert K_cols_fp32 % n_chunks == 0
    chunk_elem = K_cols_fp32 // n_chunks  # fp32: elem == fp32 col
    frag_rows = 128  # M=128 => 2 slabs
    per_thread_elems = _REGS_FACTOR[shape] * full_rep * 2  # *2 for the second slab

    tmem_col_width_32b = max(32, _next_pow2(K_cols_fp32))
    stage_width_elem = tmem_col_width_32b
    CHUNK_FP32 = 128
    n_stage = tmem_col_width_32b // CHUNK_FP32 if tmem_col_width_32b > CHUNK_FP32 else 1
    stage_w = tmem_col_width_32b if n_stage == 1 else CHUNK_FP32
    VEC_LEN = 4  # 128-bit / fp32

    atom_view = tcgen05_atom_layout(shape, (frag_rows, K_cols_fp32), dtype)
    stage_view = TileLayout(S[(128, stage_w) : (1 @ axis_tid_in_wg, 1)])

    @T.prim_func
    def kernel(A_ptr: T.handle, Bf_ptr: T.handle, Bs_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, stage_width_elem), dtype)
        Bf = T.match_buffer(Bf_ptr, (128, per_thread_elems), dtype)  # full-load dump
        Bs = T.match_buffer(Bs_ptr, (128, per_thread_elems), dtype)  # sliced-load dump
        A_flat = A.view(-1)

        T.device_entry()
        warp_id = T.warp_id([4])
        T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        tid_in_wg = T.thread_id([128])

        tmem_addr = T.alloc_shared([1], "uint32")
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    T.address_of(tmem_addr), T.uint32(tmem_col_width_32b)
                )
            T.tvm_storage_sync("shared")
            tmem = T.decl_buffer(
                (128, stage_width_elem),
                dtype,
                scope="tmem",
                allocated_addr=tmem_addr[0],
                layout=TileLayout(S[(128, stage_width_elem) : (1 @ TLane, 1 @ TCol)]),
            )
            # Stage A -> TMEM via the standard .32x32b path.
            stage_reg = T.alloc_local((stage_w,), dtype)
            stage_local = stage_reg.view(128, stage_w, layout=stage_view)
            for ci in range(n_stage):
                coff = ci * stage_w
                for i in range(stage_w // VEC_LEN):
                    g = T.meta_var(tid_in_wg * stage_width_elem + coff + i * VEC_LEN)
                    Tx.copy(stage_reg[i * VEC_LEN : i * VEC_LEN + VEC_LEN], A_flat[g : g + VEC_LEN])
                T.cuda.cta_sync()
                Tx.wg.copy_async(tmem[:, coff : coff + stage_w], stage_local[:, :])
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.cuda.cta_sync()

            # (a) one full-width load
            ff = T.alloc_local((per_thread_elems,), dtype)
            ffl = ff.view(frag_rows, K_cols_fp32, layout=atom_view)
            Tx.wg.copy_async(ffl[:, :], tmem[0:frag_rows, 0:K_cols_fp32])
            T.ptx.tcgen05.wait__ld.sync.aligned()
            T.cuda.cta_sync()
            for i in range(per_thread_elems):
                Bf[tid_in_wg, i] = ff[i]

            # (b) the same frag loaded in n_chunks column slices
            sf = T.alloc_local((per_thread_elems,), dtype)
            sfl = sf.view(frag_rows, K_cols_fp32, layout=atom_view)
            for ck in range(n_chunks):
                lo = T.meta_var(ck * chunk_elem)
                Tx.wg.copy_async(
                    sfl[:, lo : lo + chunk_elem], tmem[0:frag_rows, lo : lo + chunk_elem]
                )
            T.ptx.tcgen05.wait__ld.sync.aligned()
            T.cuda.cta_sync()
            for i in range(per_thread_elems):
                Bs[tid_in_wg, i] = sf[i]

            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    tmem_addr[0], T.uint32(tmem_col_width_32b)
                )

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, stage_width_elem))
        Bf_np = np.zeros((128, per_thread_elems), dtype=dtype)
        Bs_np = np.zeros((128, per_thread_elems), dtype=dtype)

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            Bf = tvm.runtime.tensor(Bf_np, dev)
            Bs = tvm.runtime.tensor(Bs_np, dev)
            mod(A, Bf, Bs)
            # Sliced load must reproduce the full-width load bit-for-bit.
            np.testing.assert_array_equal(Bs.numpy().view(np.uint32), Bf.numpy().view(np.uint32))

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize(
    "full_rep, n_chunks",
    [
        (32, 8),  # 16x256b.x32 (256 fp32 cols) loaded in 8 chunks of 32 cols (nvfp4 EPI_TILE=32)
        (32, 16),  # ...in 16 chunks of 16 cols (nvfp4 EPI_TILE=16)
        (32, 4),  # ...in 4 chunks of 64 cols
        (16, 8),  # 16x256b.x16 (128 fp32 cols) in 8 chunks of 16 cols
        (16, 2),  # ...in 2 chunks of 64 cols
    ],
)
def test_tcgen05_ld_16x256b_sliced_matches_full_M128(full_rep, n_chunks):
    """Per-chunk column-slice load of a wide M=128 frag == full-width load."""
    _run_sliced_vs_full_load("16x256b", full_rep, n_chunks)


# .32x32b tmem<->reg and smem<->tmem copy_async (migrated from test_tmem.py).


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("width_32b", [4, 8, 16, 32])
def test_copy_tmem2reg_async(dtype, width_32b):
    """Test async tmem<->local copy using copy_async instead of copy.

    This tests the new copy_async dispatch for tmem<->local that doesn't
    immediately wait after the operation, allowing for pipelining.
    """

    def next_power_of_2(x):
        """Return the smallest power of 2 greater than or equal to x."""
        if x <= 1:
            return 1
        return 1 << (x - 1).bit_length()

    bits = tvm.runtime.DataType(dtype).bits
    if 128 % bits != 0 or 32 % bits != 0:
        pytest.skip(f"dtype {dtype} is not supported")

    WIDTH = width_32b * (32 // bits)
    VEC_LEN = 128 // bits
    if WIDTH % VEC_LEN != 0:
        pytest.skip(f"dtype {dtype} + width {width_32b} is not supported")

    g_layout = TileLayout(S[(128, WIDTH // VEC_LEN, VEC_LEN) : (WIDTH, VEC_LEN, 1)])
    local_view = TileLayout(S[(128, WIDTH) : (1 @ axis_tid_in_wg, 1)])

    # fmt: off
    @T.prim_func
    def copy_async_test(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, WIDTH), dtype)
        B = T.match_buffer(B_ptr, (128, WIDTH), dtype)

        A_flat = A.view(-1)
        B_flat = B.view(-1)

        T.device_entry()
        warp_id = T.warp_id([(128) // 32])
        cta_id = T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        warp_id_in_wg = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
        tid_in_wg = T.thread_id([128])

        tmem_addr = T.alloc_shared([1], "uint32")

        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(tmem_addr), T.uint32(max(32, next_power_of_2(width_32b))))  # noqa: E501

            T.tvm_storage_sync("shared")

            tmem = T.decl_buffer((128, WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],
                                 layout=TileLayout(S[(128, WIDTH) : (1 @ TLane, 1 @ TCol)]))

            A_reg = T.alloc_local((WIDTH), dtype)
            B_reg = T.alloc_local((WIDTH), dtype)
            A_local = A_reg.view(128, WIDTH, layout=local_view)
            B_local = B_reg.view(128, WIDTH, layout=local_view)
            for i in range(WIDTH // VEC_LEN):
                g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                Tx.copy(A_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN], A_flat[g_offset: g_offset + VEC_LEN])  # noqa: E501
            for i in range(WIDTH):
                B_reg[i] = T.cast(0, dtype)
            T.cuda.cta_sync()

                    # A_local -> tmem (async)
            Tx.wg.copy_async(tmem[:, :], A_local[:, :])
            T.ptx.tcgen05.wait__st.sync.aligned()  # explicit wait
            T.cuda.cta_sync()

                    # tmem -> B_local (async)
            Tx.wg.copy_async(B_local[:, :], tmem[:, :])
            T.ptx.tcgen05.wait__ld.sync.aligned()  # explicit wait
            T.cuda.cta_sync()
            for i in range(WIDTH // VEC_LEN):
                g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                Tx.copy(B_flat[g_offset: g_offset + VEC_LEN], B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])  # noqa: E501

            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(max(32, next_power_of_2(width_32b))))  # noqa: E501
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_async_test})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, WIDTH))
        B_np = np.zeros((128, WIDTH), dtype=dtype)
        DEV = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, DEV)
        B = tvm.runtime.tensor(B_np, DEV)
        mod(A, B)
        np.testing.assert_allclose(B.numpy(), A_np)


# tmem<->reg round-trip via T.copy_async (migrated from test_copy_sync.py):
# the kernels are the real async tmem dispatch tests; G<->L copies just stage.


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("dtype", ["uint8", "float16", "float32"])
@pytest.mark.parametrize("width_32b", [2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("offset_32b", [0, 3, 10])
def test_copy_tmem2reg(dtype, width_32b, offset_32b):
    def next_power_of_2(x):
        if x <= 1:
            return 1
        return 1 << (x - 1).bit_length()

    bits = tvm.runtime.DataType(dtype).bits
    if 128 % bits != 0 or 32 % bits != 0:
        pytest.skip(f"dtype {dtype} is not supported")

    WIDTH = width_32b * (32 // bits)
    OFFSET = offset_32b * (32 // bits)
    VEC_LEN = 128 // bits
    if WIDTH % VEC_LEN != 0:
        pytest.skip(f"dtype {dtype} + width {width_32b} is not supported")

    g_layout = TileLayout(S[(128, WIDTH // VEC_LEN, VEC_LEN) : (WIDTH, VEC_LEN, 1)])
    local_view = TileLayout(S[(128, WIDTH) : (1 @ axis_tid_in_wg, 1)])

    # fmt: off
    @T.prim_func
    def copy_sync(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, WIDTH), dtype)
        B = T.match_buffer(B_ptr, (128, WIDTH), dtype)

        A_flat = A.view(-1)
        B_flat = B.view(-1)

        T.device_entry()
        warp_id = T.warp_id([(128) // 32])
        T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        tid_in_wg = T.thread_id([128])

        tmem_addr = T.alloc_shared([1], "uint32")

        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(tmem_addr), T.uint32(max(32, next_power_of_2(offset_32b + width_32b))))  # noqa: E501

            T.tvm_storage_sync("shared")

            tmem = T.decl_buffer((128, OFFSET + WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],  # noqa: E501
                                 layout=TileLayout(S[(128, OFFSET + WIDTH) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501

            A_reg = T.alloc_local((WIDTH), dtype)
            B_reg = T.alloc_local((WIDTH), dtype)
            A_local = A_reg.view(128, WIDTH, layout=local_view)
            B_local = B_reg.view(128, WIDTH, layout=local_view)
            for i in range(WIDTH // VEC_LEN):
                g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                Tx.copy(A_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN], A_flat[g_offset: g_offset + VEC_LEN])  # noqa: E501
            for i in range(WIDTH):
                B_reg[i] = T.cast(0, dtype)
            T.cuda.cta_sync()

                    # A_local -> tmem
            Tx.wg.copy_async(tmem[:, OFFSET: OFFSET + WIDTH], A_local[:, :])
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.cuda.cta_sync()

                    # tmem -> B_local
            Tx.wg.copy_async(B_local[:, :], tmem[:, OFFSET: OFFSET + WIDTH])
            T.ptx.tcgen05.wait__ld.sync.aligned()
            T.cuda.cta_sync()
            for i in range(WIDTH // VEC_LEN):
                g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                Tx.copy(B_flat[g_offset: g_offset + VEC_LEN], B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])  # noqa: E501

            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(max(32, next_power_of_2(offset_32b + width_32b))))  # noqa: E501
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_sync})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, WIDTH))
        B_np = np.zeros((128, WIDTH), dtype=dtype)
        DEV = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, DEV)
        B = tvm.runtime.tensor(B_np, DEV)
        mod(A, B)
        np.testing.assert_allclose(B.numpy(), A_np)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("width_32b", [4, 8, 16, 32])
@pytest.mark.parametrize("local_offset_32b", [0, 2, 4])
def test_copy_tmem2reg_sliced_local(dtype, width_32b, local_offset_32b):
    """tmem<->local copy with a sliced local buffer region."""

    def next_power_of_2(x):
        if x <= 1:
            return 1
        return 1 << (x - 1).bit_length()

    bits = tvm.runtime.DataType(dtype).bits
    if 128 % bits != 0 or 32 % bits != 0:
        pytest.skip(f"dtype {dtype} is not supported")

    WIDTH = width_32b * (32 // bits)
    LOCAL_OFFSET = local_offset_32b * (32 // bits)
    TOTAL_LOCAL_WIDTH = WIDTH + LOCAL_OFFSET
    VEC_LEN = 128 // bits
    if WIDTH % VEC_LEN != 0 or TOTAL_LOCAL_WIDTH % VEC_LEN != 0:
        pytest.skip(
            f"dtype {dtype} + width {width_32b} + offset {local_offset_32b} is not supported"
        )

    g_layout = TileLayout(S[(128, WIDTH // VEC_LEN, VEC_LEN) : (WIDTH, VEC_LEN, 1)])
    local_view = TileLayout(S[(128, TOTAL_LOCAL_WIDTH) : (1 @ axis_tid_in_wg, 1)])

    # fmt: off
    @T.prim_func
    def copy_sync(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, WIDTH), dtype)
        B = T.match_buffer(B_ptr, (128, WIDTH), dtype)

        A_flat = A.view(-1)
        B_flat = B.view(-1)

        T.device_entry()
        warp_id = T.warp_id([(128) // 32])
        T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        tid_in_wg = T.thread_id([128])

        tmem_addr = T.alloc_shared([1], "uint32")

        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(T.address_of(tmem_addr), T.uint32(max(32, next_power_of_2(width_32b))))  # noqa: E501

            T.tvm_storage_sync("shared")

            tmem = T.decl_buffer((128, WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],
                                 layout=TileLayout(S[(128, WIDTH) : (1 @ TLane, 1 @ TCol)]))

            A_reg = T.alloc_local((TOTAL_LOCAL_WIDTH), dtype)
            B_reg = T.alloc_local((TOTAL_LOCAL_WIDTH), dtype)
            A_local = A_reg.view(128, TOTAL_LOCAL_WIDTH, layout=local_view)
            B_local = B_reg.view(128, TOTAL_LOCAL_WIDTH, layout=local_view)
            for i in range(WIDTH // VEC_LEN):
                g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                Tx.copy(A_reg[LOCAL_OFFSET + i * VEC_LEN: LOCAL_OFFSET + i * VEC_LEN + VEC_LEN], A_flat[g_offset: g_offset + VEC_LEN])  # noqa: E501
            for i in range(TOTAL_LOCAL_WIDTH):
                B_reg[i] = T.cast(0, dtype)
            T.cuda.cta_sync()

                    # A_local[sliced] -> tmem (use sliced region)
            Tx.wg.copy_async(tmem[:, 0:WIDTH], A_local[:, LOCAL_OFFSET:LOCAL_OFFSET + WIDTH])
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.cuda.cta_sync()

                    # tmem -> B_local[sliced] (use sliced region)
            Tx.wg.copy_async(B_local[:, LOCAL_OFFSET:LOCAL_OFFSET + WIDTH], tmem[:, 0:WIDTH])
            T.ptx.tcgen05.wait__ld.sync.aligned()
            T.cuda.cta_sync()
            for i in range(WIDTH // VEC_LEN):
                g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                Tx.copy(B_flat[g_offset: g_offset + VEC_LEN], B_reg[LOCAL_OFFSET + i * VEC_LEN: LOCAL_OFFSET + i * VEC_LEN + VEC_LEN])  # noqa: E501

            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(max(32, next_power_of_2(width_32b))))  # noqa: E501
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_sync})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, WIDTH))
        B_np = np.zeros((128, WIDTH), dtype=dtype)
        DEV = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, DEV)
        B = tvm.runtime.tensor(B_np, DEV)
        mod(A, B)
        np.testing.assert_allclose(B.numpy(), A_np)


if __name__ == "__main__":
    tvm.testing.main()
