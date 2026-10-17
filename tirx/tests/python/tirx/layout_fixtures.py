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
"""Operand layouts shared by CUDA kernel tests; not part of the public API."""

import tvm_ffi

from tvm_tirx.backend.cuda.axis import Axis as CUDAAxis
from tvm_tirx.tirx.layout import Iter, R, S, TileLayout

# Per-warp fp32-column factor for each instr_shape. For .16x*b atoms the
# warpgroup fragment is 64 rows x (factor * rep) fp32 cols; for .32x32b the
# fragment is 128 rows x (factor * rep) fp32 cols with factor=1.
_TCGEN05_COL_FACTOR_FP32 = {"32x32b": 1, "16x64b": 2, "16x128b": 4, "16x256b": 8}


def tcgen05_register_layout(instr_shape, tensor_shape, dtype):
    """Build the logical register image exercised by TCGEN05 readback tests."""
    bits = tvm_ffi.dtype(dtype).bits
    rows, cols = tensor_shape

    if instr_shape == "32x32b" and rows == 64:
        # Layout B's logical (64, N) tile physically occupies all 128 lanes and
        # N/2 tcols. A .32x32b thread therefore owns its physical lane and N/2
        # fp32 registers. Re-label that (128, N/2) register file as (64, N).
        n_half = cols // 2
        return TileLayout.from_iters(
            [
                Iter(64, 1, CUDAAxis.tid_in_wg),
                Iter(2, 64, CUDAAxis.tid_in_wg),
                Iter(n_half, 1, "m"),
            ],
            [],
            {},
        )

    elem_per_32b = 32 // bits
    col_factor_elem = _TCGEN05_COL_FACTOR_FP32[instr_shape] * elem_per_32b
    rep = cols // col_factor_elem

    laneid = CUDAAxis.laneid
    wid = CUDAAxis.wid_in_wg
    N = rep
    shape = instr_shape
    # All m-strides below are written in fp32-reg units; we multiply by
    # elem_per_32b at the end and append a C_pack iter for the 16-bit case
    # (each fp32 reg packs ``elem_per_32b`` elements at adjacent col positions).

    if shape == "32x32b":
        # M=128 fragment, simple thread-rows layout:
        #   (rows=128, cols=K) : (1@tid_in_wg, 1)
        # Each of 128 warpgroup threads owns one row; cols are contiguous in
        # the per-thread storage. For 16-bit dtypes the K cols are packed two
        # per 32-bit register (handled by the per-thread storage element count
        # naturally — m-stride 1 in element units).
        iters = [
            Iter(rows, 1, CUDAAxis.tid_in_wg),
            Iter(cols, 1, "m"),
        ]
        return TileLayout.from_iters(iters, [], {})

    # Iter lists are written high-to-low: ``TileLayout`` decomposes a flat
    # coordinate via ``SplitCoord`` (src/tirx/ir/layout/utils.cc) using
    # row-major ordering, where the FIRST iter receives the *high* bits and
    # the LAST iter receives the *low* bits. So R_w (highest-stride row
    # contribution) comes first in row_iters_fp32 and R_t1/t2 (lowest)
    # comes last; same for col.
    if shape == "16x64b":
        # Per-warp tile (fp32 view): (16 rows, 2N cols). Per-lane regs = N.
        # Lane (t0, t1, t2): t0 = laneid & 1, t1 = (laneid >> 1) & 1, t2 = laneid >> 2.
        #   Row = t2 + 8*t0 + 16*wid_in_wg
        #   Col (fp32) = t1 + 2*r,   r ∈ [0, N)
        row_iters_fp32 = [
            (4, 1, wid),  # R_w:  wid_in_wg       → R bits 4..5
            (2, 1, laneid),  # R_t0: laneid bit 0    → R bit 3
            (8, 4, laneid),  # R_t2: laneid bits 2..4 → R bits 0..2
        ]
        col_iters_fp32 = [
            (N, 1, "m"),  # C_r:  register slot   → C bits 1..
            (2, 2, laneid),  # C_t1: laneid bit 1    → C bit 0
        ]
        m_used_M64 = N
    elif shape == "16x128b":
        # Per-warp tile (fp32 view): (16 rows, 4N cols). Per-lane regs = 2N.
        # Lane (t0, t1): t0 = laneid & 3, t1 = laneid >> 2.
        # Reg r = ra + 2*rb, ra ∈ {0,1}, rb ∈ [0, N).
        #   Row = t1 + 8*ra + 16*wid_in_wg
        #   Col (fp32) = t0 + 4*rb
        row_iters_fp32 = [
            (4, 1, wid),  # R_w
            (2, 1, "m"),  # R_ra: reg bit 0        → R bit 3
            (8, 4, laneid),  # R_t1: laneid bits 2..4 → R bits 0..2
        ]
        col_iters_fp32 = [
            (N, 2, "m"),  # C_rb: reg bits 1..     → C bits 2..
            (4, 1, laneid),  # C_t0: laneid bits 0..1 → C bits 0..1
        ]
        m_used_M64 = 2 * N
    else:  # 16x256b
        # Per-warp tile (fp32 view): (16 rows, 8N cols). Per-lane regs = 4N.
        # Lane (t0, t1) as for 16x128b. Reg r = v0p + 2*va + 4*vb.
        #   Row = t1 + 8*va + 16*wid_in_wg
        #   Col (fp32) = v0p + 2*t0 + 8*vb
        row_iters_fp32 = [
            (4, 1, wid),  # R_w
            (2, 2, "m"),  # R_va: reg bit 1 → R bit 3
            (8, 4, laneid),  # R_t1
        ]
        col_iters_fp32 = [
            (N, 4, "m"),  # C_vb:  reg bits 2.. → C bits 3..
            (4, 1, laneid),  # C_t0
            (2, 1, "m"),  # C_v0p: reg bit 0  → C bit 0
        ]
        m_used_M64 = 4 * N

    if rows == 128:
        # M=128 covers both 16-row half-slabs of each warp's 32-lane TMEM
        # partition (the M=64 atom covers only lanes 0..15; the high half
        # 16..31 needs a second PTX issue with row offset 16). We surface
        # the combined fragment as a single (128, K) tile by inserting a
        # v_slab iter right *after* R_w (i.e. as the next-highest row bit).
        # v_slab claims one m-bit at the next free offset
        # (stride = m_used_M64) so reg indices [0, m_used_M64) hold the
        # low slab and [m_used_M64, 2*m_used_M64) hold the high slab — the
        # split the test uses for its explicit instruction views. The
        # inserted iter also doubles wid_in_wg's row stride from 16 to 32,
        # so the four warps now tile rows 0..31 / 32..63 / 64..95 / 96..127.
        new_row_iters = []
        for ext, stride, axis in row_iters_fp32:
            new_row_iters.append((ext, stride, axis))
            if axis is wid:
                new_row_iters.append((2, m_used_M64, "m"))
        row_iters_fp32 = new_row_iters

    def _scale(iters):
        out = []
        for ext, stride, axis in iters:
            if axis == "m":
                out.append((ext, stride * elem_per_32b, axis))
            else:
                out.append((ext, stride, axis))
        return out

    row_iters = _scale(row_iters_fp32)
    col_iters = _scale(col_iters_fp32)

    # For the 16-bit packed variant each fp32 register holds two adjacent
    # column elements (low / high halves). Add a C_pack iter of extent
    # ``elem_per_32b`` and m-stride 1 at the *low* end of the col axis —
    # i.e. as the LAST col iter under SplitCoord's high-to-low ordering.
    if elem_per_32b > 1:
        col_iters.append((elem_per_32b, 1, "m"))

    iters = [Iter(ext, stride, axis) for ext, stride, axis in row_iters + col_iters]
    return TileLayout.from_iters(iters, [], {})


def scale_factor_tmem_layout(rows, SF_K, sf_per_mma, sf_reuse=1, pipe_depth=None):
    """Build the scale-factor storage used by the block-scaled MMA cases."""
    groups = rows // 32
    k_tiles = SF_K // sf_per_mma
    pack = 4 // sf_per_mma
    while pack > 1 and k_tiles % pack:
        pack //= 2
    # One group spans 32 lanes. Four 8-bit factors share a TMEM column.
    shape = [groups, 32, k_tiles // pack, pack, sf_per_mma]
    strides = [
        4 @ CUDAAxis.TCol,
        1 @ CUDAAxis.TLane,
        groups * 4 @ CUDAAxis.TCol,
        sf_per_mma @ CUDAAxis.TCol,
        1 @ CUDAAxis.TCol,
    ]
    base = TileLayout(S[tuple(shape) : tuple(strides)] + R[4 : 32 @ CUDAAxis.TLane]).canonicalize()
    shard = list(base.shard)
    if sf_reuse > 1:
        shard.append(Iter(sf_reuse, 0, shard[0].axis))
    if pipe_depth is not None:
        shard.insert(0, Iter(pipe_depth, groups * 4, shard[0].axis))
    return TileLayout.from_iters(shard, list(base.replica), dict(base.offset))
