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
"""Bit-exact tests for the ``.16x{64,128,256}b`` ``tcgen05.{ld,st}`` dispatch.

For each ``(shape, rep, dtype, direction)`` we:

1. Fill a (128, FULL_W) host buffer ``A`` with random values.
2. Stage ``A`` into TMEM via the existing ``.32x32b`` ld/st round-trip.
3. Issue the new ``.16x*b`` atom via ``Tx.copy_async`` to read a (64, K_cols)
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
from tvm.script import tirx as Tx
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


@pytest.mark.parametrize("shape", list(_SHAPE_REPS))
@pytest.mark.parametrize("rep", [1, 2, 4, 8, 16, 32])  # subset; full reps below
@pytest.mark.parametrize("dtype", ["float32"])
def test_tcgen05_ld_16xnb_load_fp32(shape, rep, dtype):
    """Bit-exact verification of ``tcgen05.<shape>.x<rep>.b32`` load."""
    if rep not in _SHAPE_REPS[shape]:
        pytest.skip(f"rep {rep} not valid for {shape}")
    _run_load_test(shape, rep, dtype)


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

    @Tx.prim_func
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        # Per-thread input/output: A[tid_in_wg, i] feeds register slot i of the
        # warpgroup-collective fragment; B[tid_in_wg, i] is what comes back
        # after a .16x*b.st → .16x*b.ld round-trip.
        A = Tx.match_buffer(A_ptr, (128, per_thread_elems), dtype)
        B = Tx.match_buffer(B_ptr, (128, per_thread_elems), dtype)

        Tx.device_entry()
        warp_id = Tx.warp_id([128 // 32])
        Tx.cta_id([2])
        wg_id = Tx.warpgroup_id([1])
        Tx.warp_id_in_wg([4])
        Tx.lane_id([32])
        tid_in_wg = Tx.thread_id([128])

        tmem_addr = Tx.alloc_shared([1], "uint32")

        if wg_id == 0:
            with Tx.warpgroup():
                if warp_id == 0:
                    with Tx.warp():
                        Tx.ptx.tcgen05.alloc(
                            Tx.address_of(tmem_addr),
                            n_cols=tmem_col_width_32b,
                            cta_group=1,
                        )

                Tx.tvm_storage_sync("shared")

                tmem = Tx.decl_buffer(
                    (tmem_rows, stage_width_elem),
                    dtype,
                    scope="tmem",
                    allocated_addr=tmem_addr[0],
                    layout=tmem_layout,
                )

                # Load per-thread A → reg_in
                reg_in = Tx.alloc_local((per_thread_elems,), dtype)
                with Tx.thread():
                    for i in range(per_thread_elems):
                        reg_in[i] = A[tid_in_wg, i]
                Tx.cuda.cta_sync()

                # reg_in -> TMEM via .<shape>.x<rep>.st.unpack::16b
                frag_in = reg_in.view(frag_rows, K_cols_elem, layout=atom_view)
                Tx.copy_async(tmem[0:frag_rows, 0:K_cols_elem], frag_in[:, :])
                Tx.ptx.tcgen05.wait.st()
                Tx.cuda.cta_sync()

                # TMEM -> reg_out via .<shape>.x<rep>.ld.pack::16b
                reg_out = Tx.alloc_local((per_thread_elems,), dtype)
                frag_out = reg_out.view(frag_rows, K_cols_elem, layout=atom_view)
                Tx.copy_async(frag_out[:, :], tmem[0:frag_rows, 0:K_cols_elem])
                Tx.ptx.tcgen05.wait.ld()
                Tx.cuda.cta_sync()

                # reg_out -> B
                with Tx.thread():
                    for i in range(per_thread_elems):
                        B[tid_in_wg, i] = reg_out[i]

                if warp_id == 0:
                    with Tx.warp():
                        Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                        Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=tmem_col_width_32b, cta_group=1)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, per_thread_elems))
        B_np = np.zeros((128, per_thread_elems), dtype=dtype)
        DEV = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, DEV)
        B = tvm.runtime.tensor(B_np, DEV)
        mod(A, B)
        # Round-trip should preserve every per-thread bit pattern.
        A_view = A.numpy().view(np.uint16)
        B_view = B.numpy().view(np.uint16)
        np.testing.assert_array_equal(B_view, A_view)


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
def test_tmem_datapath_layout_F_row_to_lane_mapping():
    """Layout F: every logical row r ∈ [0, 64) must land at physical TMEM
    lane ``(r // 16) * 32 + (r % 16)`` — the canonical scatter that the
    ``.16x*b`` M=64 PTX accesses (warp i on lanes ``i * 32 .. i * 32 + 15``).
    """
    cols = 32
    layout = tmem_datapath_layout("F", 64, cols)
    for r in range(64):
        for c in [0, 1, 7, 16, 31]:
            # Use ``apply(coord, shape=[64, cols])`` so (r, c) gets flattened
            # row-major before SplitCoord into the shard iters.
            axis_values = layout.apply(r, c, shape=[64, cols])
            expected_lane = (r // 16) * 32 + (r % 16)
            assert int(axis_values["TLane"]) == expected_lane, (
                f"(r={r}, c={c}) mapped to TLane {int(axis_values['TLane'])}, "
                f"expected {expected_lane} (= (r//16)*32 + (r%16))"
            )
            assert int(axis_values["TCol"]) == c, (
                f"(r={r}, c={c}) mapped to TCol {int(axis_values['TCol'])}, expected {c}"
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

    @Tx.prim_func
    def kernel() -> None:
        Tx.device_entry()
        Tx.warp_id([128 // 32])
        Tx.cta_id([2])
        wg_id = Tx.warpgroup_id([1])
        Tx.warp_id_in_wg([4])
        Tx.lane_id([32])
        Tx.thread_id([128])
        tmem_addr = Tx.alloc_shared([1], "uint32")
        if wg_id == 0:
            with Tx.warpgroup():
                Tx.tvm_storage_sync("shared")
                tmem = Tx.decl_buffer(
                    (tmem_rows, stage_width_elem),
                    "float32",
                    scope="tmem",
                    allocated_addr=tmem_addr[0],
                    layout=tmem_layout,
                )
                frag = Tx.alloc_local((local_extent_rows * local_cols // 128,), "float32")
                frag_view = frag.view(local_extent_rows, local_cols, layout=atom_view)
                Tx.copy_async(frag_view[:, :], tmem[0:local_extent_rows, 0:local_cols])

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        with pytest.raises((ValueError, RuntimeError), match="datapath"):
            tvm.compile(mod, target=target, tir_pipeline="tirx")


def _run_load_test(shape: str, rep: int, dtype: str):
    """Stage A into TMEM via .32x32b, then read it back as the fragment via
    .<shape>.x<rep> (through ``Tx.alloc_tcgen05_ldst_frag``), and compare each
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
    # ``Tx.copy_async`` matches when dispatching to the right atom path.
    atom_view = tcgen05_atom_layout(shape, (frag_rows, K_cols_elem), dtype)

    @Tx.prim_func
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        # A is the host data we stage into TMEM via the standard .32x32b path.
        A = Tx.match_buffer(A_ptr, (128, stage_width_elem), dtype)
        # B is a per-thread register dump: B[tid_in_wg, reg_idx_in_elements].
        B = Tx.match_buffer(B_ptr, (128, per_thread_elems), dtype)

        A_flat = A.view(-1)

        Tx.device_entry()
        warp_id = Tx.warp_id([128 // 32])
        Tx.cta_id([2])
        wg_id = Tx.warpgroup_id([1])
        Tx.warp_id_in_wg([4])
        Tx.lane_id([32])
        tid_in_wg = Tx.thread_id([128])

        tmem_addr = Tx.alloc_shared([1], "uint32")

        if wg_id == 0:
            with Tx.warpgroup():
                if warp_id == 0:
                    with Tx.warp():
                        Tx.ptx.tcgen05.alloc(
                            Tx.address_of(tmem_addr),
                            n_cols=tmem_col_width_32b,
                            cta_group=1,
                        )

                Tx.tvm_storage_sync("shared")

                tmem = Tx.decl_buffer(
                    (128, stage_width_elem),
                    dtype,
                    scope="tmem",
                    allocated_addr=tmem_addr[0],
                    layout=TileLayout(S[(128, stage_width_elem) : (1 @ TLane, 1 @ TCol)]),
                )

                # Per-thread chunk staging buffer (CHUNK_FP32 fp32 worth).
                stage_reg = Tx.alloc_local((chunk_width_elem,), dtype)
                stage_local = stage_reg.view(128, chunk_width_elem, layout=chunk_view)

                # Walk chunks: A[:, ck:ck+chunk] -> stage_reg -> TMEM[:, ck:ck+chunk]
                for chunk_idx in range(num_chunks):
                    col_off_elem = chunk_idx * chunk_width_elem
                    with Tx.thread():
                        for i in range(chunk_width_elem // VEC_LEN):
                            # Each thread's row offset in A_flat: stage_width_elem; within
                            # the row, this chunk starts at col_off_elem and each vector
                            # picks up VEC_LEN elements at slot i.
                            g_offset = Tx.meta_var(
                                tid_in_wg * stage_width_elem + col_off_elem + i * VEC_LEN
                            )
                            Tx.copy(
                                stage_reg[i * VEC_LEN : i * VEC_LEN + VEC_LEN],
                                A_flat[g_offset : g_offset + VEC_LEN],
                            )
                    Tx.cuda.cta_sync()
                    Tx.copy_async(
                        tmem[:, col_off_elem : col_off_elem + chunk_width_elem],
                        stage_local[:, :],
                    )
                Tx.ptx.tcgen05.wait.st()
                Tx.cuda.cta_sync()

                # TMEM[0:frag_rows, 0:K_cols] -> frag_local via .<shape>.x<rep>.ld.
                # Use ``tcgen05_atom_layout`` so dispatch matches the new path
                # (or stays on .32x32b for instr_shape="32x32b"). Keep the flat
                # ``frag_reg`` for the per-thread dump below.
                frag_reg = Tx.alloc_local((per_thread_elems,), dtype)
                frag_local = frag_reg.view(frag_rows, K_cols_elem, layout=atom_view)
                Tx.copy_async(frag_local[:, :], tmem[0:frag_rows, 0:K_cols_elem])
                Tx.ptx.tcgen05.wait.ld()
                Tx.cuda.cta_sync()

                # Dump per-thread regs to B[tid_in_wg, :]
                with Tx.thread():
                    for i in range(per_thread_elems):
                        B[tid_in_wg, i] = frag_reg[i]

                if warp_id == 0:
                    with Tx.warp():
                        Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                        Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=tmem_col_width_32b, cta_group=1)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, stage_width_elem))
        B_np = np.zeros((128, per_thread_elems), dtype=dtype)
        DEV = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, DEV)
        B = tvm.runtime.tensor(B_np, DEV)
        mod(A, B)
        B_out = B.numpy()

    # Build expected B_out from the layout.
    if bits == 32:
        # Each register slot in B[t, r] holds a single fp32; compare bit-exactly.
        B_expected = np.zeros((128, per_thread_elems), dtype=np.uint32)
        for t in range(128):
            for r in range(regs_per_thread):
                B_expected[t, r] = _expected_reg_value_fp32(A_np, shape, rep, 0, t, r)
        B_view = B_out.view(np.uint32)
        np.testing.assert_array_equal(B_view, B_expected)
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
        B_view = B_out.view(np.uint32).reshape(128, regs_per_thread)
        B_expected = np.zeros((128, regs_per_thread), dtype=np.uint32)
        for t in range(128):
            for r in range(regs_per_thread):
                B_expected[t, r] = _expected_reg_value_16b(A_np, shape, rep, 0, t, r, dtype_np)
        np.testing.assert_array_equal(B_view, B_expected)


# --------------------------------------------------------------------------
# Test 2: store direction (mirror of test 1, with .st instead of .ld)
# --------------------------------------------------------------------------


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

    @Tx.prim_func
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        # A[tid_in_wg, i] is the i-th per-thread element to feed into the atom store.
        A = Tx.match_buffer(A_ptr, (128, per_thread_elems), dtype)
        # B[lane, col] is the TMEM-staged readout after the round-trip.
        B = Tx.match_buffer(B_ptr, (128, stage_width_elem), dtype)
        B_flat = B.view(-1)

        Tx.device_entry()
        warp_id = Tx.warp_id([128 // 32])
        Tx.cta_id([2])
        wg_id = Tx.warpgroup_id([1])
        Tx.warp_id_in_wg([4])
        Tx.lane_id([32])
        tid_in_wg = Tx.thread_id([128])

        tmem_addr = Tx.alloc_shared([1], "uint32")

        if wg_id == 0:
            with Tx.warpgroup():
                if warp_id == 0:
                    with Tx.warp():
                        Tx.ptx.tcgen05.alloc(
                            Tx.address_of(tmem_addr),
                            n_cols=tmem_col_width_32b,
                            cta_group=1,
                        )

                Tx.tvm_storage_sync("shared")

                tmem = Tx.decl_buffer(
                    (128, stage_width_elem),
                    dtype,
                    scope="tmem",
                    allocated_addr=tmem_addr[0],
                    layout=TileLayout(S[(128, stage_width_elem) : (1 @ TLane, 1 @ TCol)]),
                )

                # Load per-thread A → frag_reg
                frag_reg = Tx.alloc_local((per_thread_elems,), dtype)
                with Tx.thread():
                    for i in range(per_thread_elems):
                        frag_reg[i] = A[tid_in_wg, i]
                Tx.cuda.cta_sync()

                # frag_local -> TMEM via .<shape>.x<rep>.st
                frag_local = frag_reg.view(frag_rows, K_cols_elem, layout=atom_view)
                Tx.copy_async(tmem[0:frag_rows, 0:K_cols_elem], frag_local[:, :])
                Tx.ptx.tcgen05.wait.st()
                Tx.cuda.cta_sync()

                # TMEM -> readout via .32x32b.ld
                stage_reg = Tx.alloc_local((stage_width_elem,), dtype)
                stage_local = stage_reg.view(128, stage_width_elem, layout=stage_view)
                Tx.copy_async(stage_local[:, :], tmem[:, :])
                Tx.ptx.tcgen05.wait.ld()
                Tx.cuda.cta_sync()

                # readout -> B (full 128xstage_width_elem dump)
                with Tx.thread():
                    for i in range(stage_width_elem // VEC_LEN):
                        g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                        Tx.copy(
                            B_flat[g_offset : g_offset + VEC_LEN],
                            stage_reg[i * VEC_LEN : i * VEC_LEN + VEC_LEN],
                        )

                if warp_id == 0:
                    with Tx.warp():
                        Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                        Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=tmem_col_width_32b, cta_group=1)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, per_thread_elems))
        B_np = np.zeros((128, stage_width_elem), dtype=dtype)
        DEV = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, DEV)
        B = tvm.runtime.tensor(B_np, DEV)
        mod(A, B)
        B_out = B.numpy()

    # Build expected TMEM staging: only rows that the M=64 fragment writes to
    # should match A's per-thread data; other rows are untouched (we set B_np to
    # zero and the .32x32b.ld reads whatever the TMEM allocator left, which may
    # be arbitrary, so only check the fragment positions).
    if bits == 32:
        view = B_out.view(np.uint32)
        for t in range(128):
            for r in range(regs_per_thread):
                row, col = _decompose_fp32(shape, t, r)
                tmem_lane = _frag_row_to_tmem_lane(shape, row)
                expected = np.float32(A_np[t, r]).view(np.uint32)
                assert view[tmem_lane, col] == expected, (
                    f"{shape}.x{rep} {dtype}: thread {t} reg {r} → "
                    f"(row={row}, col={col}) tmem_lane={tmem_lane} got "
                    f"{view[tmem_lane, col]:#x} want {expected:#x}"
                )
    else:
        # 16-bit: each fp32 reg packs two 16-bit elements at adjacent TMEM cols.
        view = B_out.view(np.uint16)
        for t in range(128):
            for r in range(regs_per_thread):
                row, col_fp32 = _decompose_fp32(shape, t, r)
                tmem_lane = _frag_row_to_tmem_lane(shape, row)
                lo = np.float16(A_np[t, 2 * r]).view(np.uint16) if dtype == "float16" else None
                # bfloat16 (numpy) lacks a clean .view(uint16); skip in store mode
                # for now to keep this test path bit-exact only for float16.
                if dtype != "float16":
                    pytest.skip("16b store check restricted to float16")
                hi = np.float16(A_np[t, 2 * r + 1]).view(np.uint16)
                assert view[tmem_lane, 2 * col_fp32] == lo, (
                    f"{shape}.x{rep} {dtype}: t={t} r={r} lo "
                    f"({tmem_lane=}, {col_fp32=}) got {view[tmem_lane, 2 * col_fp32]:#x} "
                    f"want {lo:#x}"
                )
                assert view[tmem_lane, 2 * col_fp32 + 1] == hi


# --------------------------------------------------------------------------
# Wrapper test: exercise Tx.alloc_tcgen05_ldst_frag directly (compile-only smoke).
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
def test_alloc_tcgen05_frag_wrapper_compiles(shape, frag_rows, K_cols):
    """Ensure Tx.alloc_tcgen05_ldst_frag yields a buffer that ``Tx.copy_async`` accepts
    and lowers to the correct tcgen05 atom for each supported instr_shape."""

    @Tx.prim_func
    def kernel(A_ptr: Tx.handle) -> None:
        Tx.match_buffer(A_ptr, (128, K_cols), "float32")
        Tx.device_entry()
        warp_id = Tx.warp_id([4])
        Tx.cta_id([2])
        wg_id = Tx.warpgroup_id([1])
        Tx.warp_id_in_wg([4])
        Tx.lane_id([32])
        Tx.thread_id([128])

        tmem_addr = Tx.alloc_shared([1], "uint32")
        if wg_id == 0:
            with Tx.warpgroup():
                if warp_id == 0:
                    with Tx.warp():
                        Tx.ptx.tcgen05.alloc(
                            Tx.address_of(tmem_addr), n_cols=max(32, K_cols), cta_group=1
                        )
                Tx.tvm_storage_sync("shared")
                tmem = Tx.decl_buffer(
                    (128, K_cols),
                    "float32",
                    scope="tmem",
                    allocated_addr=tmem_addr[0],
                    layout=TileLayout(S[(128, K_cols) : (1 @ TLane, 1 @ TCol)]),
                )
                # One-liner: wrapper handles per-thread storage + layout.
                frag = Tx.alloc_tcgen05_ldst_frag(shape, (frag_rows, K_cols), "float32")
                Tx.copy_async(frag[:, :], tmem[0:frag_rows, 0:K_cols])
                Tx.ptx.tcgen05.wait.ld()
                if warp_id == 0:
                    with Tx.warp():
                        Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                        Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=max(32, K_cols), cta_group=1)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    # Compiles cleanly + the generated CUDA contains the expected PTX shape.
    src = mod.mod.imports[0].inspect_source()
    assert shape in src, (
        f"expected .{shape}.x? in generated PTX, but `{shape}` not found in CUDA source"
    )


if __name__ == "__main__":
    tvm.testing.main()
