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
"""CUDA-specific memory and instruction layout factories."""

import copy
from enum import Enum

import tvm
from tvm.backend.cuda.axis import Axis as CUDAAxis
from tvm.tirx.layout import Axis, ComposeLayout, Iter, Layout, R, S, TileLayout

# ============================================================================
# TMEM datapath layouts (PTX ISA §9.7.16.10.5)
# ============================================================================
#
# ``tcgen05.mma`` writes its output matrix C into TMEM using one of several
# **datapath layouts** depending on the MMA's M dimension and ``.ws`` mode.
# Each layout determines *which* physical TMEM lanes (rows) the matrix
# occupies. The identity ``(rows, cols) : (1@TLane, 1@TCol)`` mapping
# is correct only for Layout D (M=128 full datapath). For Layout F
# (M=64 non-``.ws``) the MMA writes scattered lanes
# ``{0..15, 32..47, 64..79, 96..111}`` — half of each warp's 32-lane
# partition — and the readback path (``.16x*b`` M=64 atom) has the matching
# scatter built into the PTX. To keep the buffer's logical row indexing in
# sync with the physical scatter, the buffer's TileLayout must encode the
# scatter directly.
#
# We surface this via the factory below. Callers pass the datapath letter
# (``"D"`` / ``"F"``) and the logical ``(rows, cols)``; the factory returns
# the appropriate TileLayout. ``tmem_pool.alloc(..., layout=...)`` plumbs
# this into the buffer's layout so the dispatch can structurally verify
# atom ↔ datapath compatibility instead of silently accepting mismatches.
#
# Supported today:
#   - ``"D"``: M=128, ``.cta_group::1``, full datapath. Identity row→lane.
#   - ``"F"``: M=64, non-``.ws``, half datapath (4x1 lane utilization).
#     Logical row r → physical lane
#     (r // 16) * 32 + sub_slab * 16 + (r % 16).
#   - ``"B"``: per-CTA M=64, ``.cta_group::2``, Dense A ("2x2" datapath).
#     PTX names the CTA-pair shape M=128; each CTA owns a logical ``(64, N)``
#     accumulator. Its N columns split into two N/2 halves across physical
#     lanes 0..63 and 64..127:
#       (r, c) → (TLane=r + 64 * (c // (N/2)), TCol=c % (N/2)).
#
# Layouts A / C / E / G are reserved for future expansion.


_TMEM_DATAPATH_ROWS = {"A": 128, "B": 64, "C": 64, "D": 128, "E": 64, "F": 64, "G": 32}


def tmem_datapath_layout(datapath: str, rows: int, cols: int, sub_slab: int = 0) -> "TileLayout":
    """Return the ``TileLayout`` for a tcgen05 MMA datapath.

    See PTX ISA §9.7.16.10.5 for the datapath enumeration. The returned
    layout is shape-compatible with a buffer of ``(rows, cols)`` and
    encodes the logical-row → physical-TMEM-lane mapping that the
    corresponding MMA writes to (and that the matching ``.16x*b`` /
    ``.32x32b`` atom expects to read).

    Parameters
    ----------
    datapath : str
        PTX data path layout letter (all cta_group::2 layouts are the
        per-CTA view of the M-rows this CTA of the pair owns):

        - ``"A"``: M=256, ``.cta_group::2`` — identity over 128 rows.
        - ``"B"``: M=128, ``.cta_group::2``, dense A — 64 rows, column
          halves folded across the two 64-lane halves.
        - ``"C"``: M=128, ``.cta_group::2``, sparse A — 64 rows scattered
          16-per-warp, half datapath (same organization as F).
        - ``"D"``: M=128, ``.cta_group::1`` — identity, full datapath.
        - ``"E"``: M=64, ``.ws`` — 64 rows, column halves folded (same
          organization as B).
        - ``"F"``: M=64, non-``.ws`` — 64 rows scattered 16-per-warp,
          half datapath.
        - ``"G"``: M=32, ``.ws`` — 32 rows, column quarters folded across
          the four 32-lane warp slabs.
    rows : int
        Logical row count of the TMEM buffer. Must match the datapath's M
        dimension: 128 for A/D, 64 for B/C/E/F, 32 for G.
    cols : int
        Logical column count. Datapath B requires an even count because its
        columns split into two equal lane halves.
    sub_slab : int
        For Layout F, select the lower (``0``) or upper (``1``) 16-lane
        half of each warp's 32-lane TMEM partition. The upper half is useful
        as a 64-row read/write view of the high half-slab of a Layout D
        accumulator. Layouts D and B already span both halves and therefore
        only accept ``0``.

    Returns
    -------
    TileLayout
        Var-shape-compatible layout for ``(rows, cols)``.
    """
    if datapath not in _TMEM_DATAPATH_ROWS:
        raise ValueError(
            f"tmem_datapath_layout: unknown datapath {datapath!r}; "
            f"supported: {sorted(_TMEM_DATAPATH_ROWS)}"
        )
    expected = _TMEM_DATAPATH_ROWS[datapath]
    if rows != expected:
        raise ValueError(
            f"tmem_datapath_layout: datapath={datapath!r} expects rows={expected}, got {rows}"
        )
    if sub_slab not in (0, 1):
        raise ValueError(f"tmem_datapath_layout: sub_slab must be 0 or 1, got {sub_slab}")
    tlane = Axis.get("TLane")
    tcol = Axis.get("TCol")
    if datapath in ("A", "D"):
        # Identity row→lane (r → lane r). D is the full cta_group::1 datapath;
        # A is the per-CTA view of M=256/cta_group::2 (this CTA's 128 M-rows).
        # Both already span both 16-lane sub-slabs of every warp partition.
        if sub_slab != 0:
            raise ValueError(
                f"tmem_datapath_layout: datapath={datapath!r} (M=128) already spans both "
                "sub-slabs; sub_slab must be 0"
            )
        return TileLayout(S[(rows, cols) : (1 @ tlane, 1 @ tcol)])
    if datapath == "G":
        # Layout G (M=32, .ws): quarter datapath — column space splits into 4
        # quarters, lane = row + 32*q, physical col = col % (cols/4).
        if sub_slab != 0:
            raise ValueError(
                "tmem_datapath_layout: datapath='G' folds column quarters across "
                "full 32-lane slabs; sub_slab must be 0"
            )
        if cols % 4 != 0:
            raise ValueError(
                f"tmem_datapath_layout: datapath='G' requires cols % 4 == 0, got {cols}"
            )
        return TileLayout(S[(rows, 4, cols // 4) : (1 @ tlane, 32 @ tlane, 1 @ tcol)])
    if datapath in ("B", "E"):
        # Layouts B (M=128 cta_group::2 dense) and E (M=64 .ws): half datapath,
        # lane = row + 64 * (col >= cols/2), physical col = col % (cols/2).
        if sub_slab != 0:
            raise ValueError(
                f"tmem_datapath_layout: datapath={datapath!r} folds column halves across "
                "full 64-lane halves; sub_slab must be 0"
            )
        if cols % 2 != 0:
            raise ValueError(
                f"tmem_datapath_layout: datapath={datapath!r} expects even cols, got {cols}"
            )
        return TileLayout(S[(rows, 2, cols // 2) : (1 @ tlane, 64 @ tlane, 1 @ tcol)])
    # Layouts C (M=128 cta_group::2 sparse) and F (M=64 non-.ws): rows-per-CTA
    # scattered, half datapath — logical row wid*16+intra → physical lane
    # wid*32 + sub_slab*16 + intra.
    # ``TileLayout`` decomposes a scalar row index via ``SplitCoord``
    # (src/tirx/ir/layout/utils.cc), which uses row-major ordering: with
    # shape ``(s0, s1)`` the FIRST iter receives ``coord // s1`` (the high
    # bits) and the SECOND receives ``coord % s1`` (the low bits). So we
    # pin the warp selector to iter 0 (extent 4, TLane stride 32) and the
    # within-slab lane to iter 1 (extent 16, TLane stride 1), then shift the
    # TMEM lane offset by 16 for the upper sub-slab.
    spec = S[(4, 16, cols) : (32 @ tlane, 1 @ tlane, 1 @ tcol)]
    if sub_slab:
        spec = spec + (16 @ tlane)
    return TileLayout(spec)


def _mma_datapath_letter(M, cta_group, ws=False, sparse=False):
    """Map (instruction ``M``, ``cta_group``, ``ws``, ``sparse``) to the PTX
    tcgen05 datapath letter. Raises for families rejected in the v1 semantic
    allocator API (sparse Layout C, M=32 Layout G). See
    the documented PTX TMEM datapaths."""
    if sparse:
        raise ValueError(
            f"alloc_tcgen05_mma_AB: sparse A (Layout C) not supported in v1 "
            f"(M={M}, cta_group={cta_group})"
        )
    if ws and not (cta_group == 1 and M == 64):
        raise ValueError(
            f"alloc_tcgen05_mma_AB: ws=True only valid for cta_group=1, M=64 (Layout E); "
            f"got M={M}, cta_group={cta_group}"
        )
    if cta_group == 1:
        if M == 128:
            return "D"
        if M == 64:
            return "E" if ws else "F"
    elif cta_group == 2:
        if M == 256:
            return "A"
        if M == 128:
            return "B"
    raise ValueError(
        f"alloc_tcgen05_mma_AB: no supported datapath for M={M}, cta_group={cta_group}, "
        f"ws={ws} (supported: cta1 M in {{64,128}}, cta2 M in {{128,256}})"
    )


def tmem_mma_operand_layout(
    operand, shape, dtype, *, M, cta_group, ws=False, sparse=False, group=None
):
    """Resolve the TMEM ``TileLayout`` for a tcgen05 MMA operand from operand
    role + instruction params.

    ``operand``: ``"A"`` (A-in-TMEM) or ``"D"`` (accumulator / C).
    ``M`` is the PTX ``tcgen05.mma`` **instruction** M (256/128/64), NOT the
    per-CTA buffer rows; ``per_cta_rows = M // cta_group``. Physical 32-bit
    column count is left to the pool's ``_resolve_cols`` (layout-aware).

    Selects the documented PTX datapath from explicit instruction parameters.
    """
    tlane = Axis.get("TLane")
    tcol = Axis.get("TCol")
    datapath = _mma_datapath_letter(M, cta_group, ws, sparse)
    per_cta_rows = M // cta_group
    ext = [int(s) for s in shape]

    def _chk_rows(rows):
        if rows != per_cta_rows:
            raise ValueError(
                f"alloc_mma_{operand}: M={M}, cta_group={cta_group} expects "
                f"per-CTA rows {per_cta_rows}, got {rows} (shape {ext})"
            )

    if operand == "D":
        if group is not None:
            # flat buffer (m, N) + explicit grouping -> (m, s, 2, n) layout
            # Keep the kernel output tensor 2D and its physical layout 4-axis.
            if len(ext) != 2:
                raise ValueError(f"alloc_tcgen05_mma_D group= requires 2D (m,N), got {ext}")
            m, N = ext
            gs = [int(g) for g in group]
            if len(gs) != 3 or gs[1] != 2:
                raise ValueError(f"alloc_tcgen05_mma_D group must be (s,2,n), got {group}")
            s2, _, n2 = gs
            if s2 * 2 * n2 != N:
                raise ValueError(f"alloc_tcgen05_mma_D group {group} product != N={N}")
            if datapath not in ("B", "E"):
                raise ValueError(f"alloc_tcgen05_mma_D group= only for Layout B/E, got {datapath}")
            _chk_rows(m)
            return TileLayout(
                S[(m, s2, 2, n2) : (1 @ tlane, n2 @ tcol, 64 @ tlane, 1 @ tcol)]
            ).canonicalize()
        # Grouped grammar (C3): (m,N) | (m,2,N/2) | (m,s,2,n); plus batched
        # C[2,m,N/2] normalizing to packed. bank axis is fixed by position.
        if len(ext) == 2:
            m, N = ext
            _chk_rows(m)
            layout = tmem_datapath_layout(datapath, m, N)
        elif len(ext) == 3 and ext[0] == 2 and datapath in ("B", "E"):
            _, m, Nh = ext  # batched C[2, m, N/2]
            _chk_rows(m)
            layout = TileLayout(S[(2, m, Nh) : (64 @ tlane, 1 @ tlane, 1 @ tcol)])
        elif len(ext) == 3 and ext[1] == 2 and datapath in ("B", "E"):
            m, _, Nh = ext  # explicit (m, 2, N/2)
            _chk_rows(m)
            layout = TileLayout(S[(m, 2, Nh) : (1 @ tlane, 64 @ tlane, 1 @ tcol)])
        elif len(ext) == 4 and ext[2] == 2 and datapath in ("B", "E"):
            m, s, _, n = ext  # column-segmented (m, s, 2, n): flatten N=s*2*n
            _chk_rows(m)
            layout = TileLayout(S[(m, s, 2, n) : (1 @ tlane, n @ tcol, 64 @ tlane, 1 @ tcol)])
        else:
            raise ValueError(
                f"alloc_tcgen05_mma_D: unsupported D shape {ext} for datapath {datapath}"
            )
    elif operand == "A":
        if datapath in ("A", "D"):
            if len(ext) != 2:
                raise ValueError(
                    f"alloc_tcgen05_mma_A: datapath {datapath} identity needs 2D (m,K), got {ext}"
                )
            m, K = ext
            _chk_rows(m)
            layout = tmem_datapath_layout(datapath, m, K)  # identity
        elif datapath == "B":
            if len(ext) == 2:  # replica A
                m, K = ext
                _chk_rows(m)
                layout = TileLayout(S[(m, K) : (1 @ tlane, 1 @ tcol)] + R[2 : 64 @ tlane])
            elif len(ext) == 3 and ext[0] == 2:  # bank-batched A
                _, m, K = ext
                _chk_rows(m)
                layout = TileLayout(S[(2, m, K) : (64 @ tlane, 1 @ tlane, 1 @ tcol)])
            else:
                raise ValueError(
                    f"alloc_tcgen05_mma_A: datapath B needs (64,K) replica or (2,64,K) "
                    f"bank-batched, got {ext}"
                )
        elif datapath == "E":  # M=64 .ws: bank-batched only, flat rejected
            if len(ext) == 3 and ext[0] == 2:
                _, m, K = ext
                _chk_rows(m)
                layout = TileLayout(S[(2, m, K) : (64 @ tlane, 1 @ tlane, 1 @ tcol)])
            else:
                raise ValueError(
                    f"alloc_tcgen05_mma_A: datapath E (M=64 .ws) requires bank-batched "
                    f"(2,64,K); flat {ext} rejected (hidden second-half read)"
                )
        else:  # F (M=64 non-.ws): A occupies lanes 0..63 identically
            if len(ext) != 2:
                raise ValueError(f"alloc_tcgen05_mma_A: datapath F needs 2D (m,K), got {ext}")
            m, K = ext
            _chk_rows(m)
            layout = TileLayout(S[(m, K) : (1 @ tlane, 1 @ tcol)])
    else:
        raise ValueError(f"tmem_mma_operand_layout: operand must be 'A' or 'D', got {operand!r}")
    # PTX matrix D keeps sub-word accumulators in the low bits of a b32 cell.
    # A remains packed. TCol strides are expressed in tensor element units.
    from tvm.runtime import DataType

    if operand == "D" and DataType(dtype).bits == 16:
        layout = TileLayout.from_iters(
            shard=[
                Iter(i.extent, i.stride * (2 if i.axis.name == "TCol" else 1), i.axis)
                for i in layout.shard
            ],
            replica=layout.replica,
        )
    return layout.canonicalize()


def wg_local_layout(cols, rows=128):
    """Return a warpgroup-local register layout.

    The logical ``(rows, cols)`` tile is distributed on ``tid_in_wg`` along rows,
    so each thread owns one row and contiguous ``cols`` local elements.
    """
    return TileLayout(S[(rows, cols) : (1 @ CUDAAxis.tid_in_wg, 1)])


# Allowed (.shape, .num) combinations for tcgen05.ld/st atoms.
# Source: PTX ISA Table 49 (tcgen05-num-shapes-ld).
_TCGEN05_ATOM_REPS = {
    "32x32b": (1, 2, 4, 8, 16, 32, 64, 128),
    "16x64b": (1, 2, 4, 8, 16, 32, 64, 128),
    "16x128b": (1, 2, 4, 8, 16, 32, 64),
    "16x256b": (1, 2, 4, 8, 16, 32),
}


# Per-warp fp32-column factor for each instr_shape. For .16x*b atoms the
# warpgroup fragment is 64 rows x (factor * rep) fp32 cols; for .32x32b the
# fragment is 128 rows x (factor * rep) fp32 cols with factor=1.
_TCGEN05_COL_FACTOR_FP32 = {"32x32b": 1, "16x64b": 2, "16x128b": 4, "16x256b": 8}

# Allowed fragment row counts per warpgroup for each instr_shape. ``.32x32b``
# normally covers M=128, but a 64-row shape denotes the Layout B readback image:
# the logical (64, N) tile is physically a (128, N/2) ``.32x32b`` register
# file. ``.16x*b`` natively covers M=64 (one 16-row slab per warp, using lanes
# 0..15 of each warp's 32-lane TMEM partition) and can be extended to M=128 by
# issuing the atom twice with row offsets 0 and 16. The M=128 variant doubles
# per-thread registers and treats the extra slab as the highest m-bit.
_TCGEN05_FRAG_ROWS = {
    "32x32b": (64, 128),
    "16x64b": (64, 128),
    "16x128b": (64, 128),
    "16x256b": (64, 128),
}


def tcgen05_atom_layout(instr_shape: str, tensor_shape: tuple[int, int], dtype) -> "TileLayout":
    """Register-side ``TileLayout`` for ``tcgen05.ld``/``tcgen05.st`` atoms.

    Describes the per-warpgroup register tile that ``Tx.copy_async`` produces
    when reading a TMEM fragment via ``tcgen05.{ld,st}.<instr_shape>.xN``.
    ``rep`` (the ``.xN`` qualifier) is inferred from ``tensor_shape``.

    Fragment row count is determined by ``instr_shape``: ``.32x32b`` normally
    covers an M=128 fragment (128 rows per warpgroup), and
    ``.16x{64,128,256}b`` covers an M=64 fragment (64 rows per warpgroup).
    ``("32x32b", (64, N))`` is the fp32 Layout B readback image for a
    ``.cta_group::2`` M=64 accumulator.

    TMEM is kept **dense** for 16-bit dtypes: two 16-bit elements per 32-bit
    TMEM cell (matching the existing ``.32x32b`` convention). The PTX op is
    issued with the plain ``.b32`` form (no ``.pack::16b`` qualifier), and
    the returned layout describes the per-thread register file with two
    packed 16-bit elements per 32-bit register.

    Parameters
    ----------
    instr_shape : str
        The PTX atom's ``.shape`` qualifier. One of ``"32x32b"``, ``"16x64b"``,
        ``"16x128b"``, ``"16x256b"``.
    tensor_shape : tuple[int, int]
        The logical fragment shape in **element units**. Must be
        ``(frag_rows, K)`` where ``frag_rows`` is ``128`` for ``.32x32b`` and
        ``64`` for the other shapes. The fp32 Layout B image is the exception:
        it uses ``("32x32b", (64, N))``. The column extent is divisible by the
        per-warp column factor for the chosen instr_shape and dtype::

            K must be a power-of-two multiple of (factor_fp32 * elem_per_32b)

        where ``factor_fp32`` is ``1`` / ``2`` / ``4`` / ``8`` for ``.32x32b`` /
        ``.16x64b`` / ``.16x128b`` / ``.16x256b``, and ``elem_per_32b`` is
        ``1`` for fp32 and ``2`` for fp16/bf16. The inferred rep must be in PTX
        Table 49's supported set for the chosen instr_shape.
    dtype : str | tvm.DataType
        Element dtype. ``"float32"``, ``"float16"``, or ``"bfloat16"``.

    Returns
    -------
    TileLayout
        A ``tensor_shape``-shaped tile layout. The factory builds it as a
        sequence of fine-grained iters describing the per-(lane, register)
        destination position.

    Examples
    --------
    ``tcgen05_atom_layout("16x64b", (64, 64), "float32")`` → ``.16x64b.x32`` (rep=32, fp32).

    ``tcgen05_atom_layout("16x128b", (64, 256), "float16")`` → ``.16x128b.x32`` (rep=32,
    fp16; two fp16 elements packed per 32-bit register and per 32-bit TMEM cell).
    """
    if instr_shape not in _TCGEN05_ATOM_REPS:
        raise ValueError(
            f"tcgen05_atom_layout instr_shape must be one of "
            f"{list(_TCGEN05_ATOM_REPS)}, got {instr_shape!r}"
        )
    bits = tvm.runtime.DataType(dtype).bits
    if bits not in (16, 32):
        raise ValueError(
            f"tcgen05_atom_layout dtype must be a 32-bit or 16-bit type, got {dtype} ({bits} bits)"
        )
    if len(tensor_shape) != 2:
        raise ValueError(
            f"tcgen05_atom_layout tensor_shape must be 2-D (rows, cols), got {tensor_shape!r}"
        )
    rows, cols = tensor_shape
    allowed_rows = _TCGEN05_FRAG_ROWS[instr_shape]
    if rows not in allowed_rows:
        raise ValueError(
            f"tcgen05_atom_layout {instr_shape!r} expects rows ∈ {allowed_rows}, got {rows}"
        )

    if instr_shape == "32x32b" and rows == 64:
        # Layout B's logical (64, N) tile physically occupies all 128 lanes and
        # N/2 tcols. A .32x32b thread therefore owns its physical lane and N/2
        # fp32 registers. Re-label that (128, N/2) register file as (64, N).
        if bits != 32:
            raise ValueError(
                "tcgen05_atom_layout: 32x32b with 64 rows is the datapath B "
                f"readback image and is fp32-only, got {dtype} ({bits} bits)"
            )
        if cols % 2 != 0:
            raise ValueError(
                f"tcgen05_atom_layout: 32x32b (64, N) datapath B image expects even N, got {cols}"
            )
        n_half = cols // 2
        if n_half not in _TCGEN05_ATOM_REPS["32x32b"]:
            raise ValueError(
                f"tcgen05_atom_layout: 32x32b (64, N) datapath B image needs N/2={n_half} "
                f"(from N={cols}) in the PTX Table 49 set {_TCGEN05_ATOM_REPS['32x32b']}"
            )
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
    if cols % col_factor_elem != 0:
        raise ValueError(
            f"tcgen05_atom_layout cols={cols} not divisible by the per-rep column "
            f"factor {col_factor_elem} for instr_shape={instr_shape!r} dtype={dtype}; "
            f"valid cols are k * {col_factor_elem} for k in "
            f"{_TCGEN05_ATOM_REPS[instr_shape]}"
        )
    rep = cols // col_factor_elem
    if rep not in _TCGEN05_ATOM_REPS[instr_shape]:
        raise ValueError(
            f"tcgen05_atom_layout inferred rep={rep} (from cols={cols}) is not in "
            f"the PTX Table 49 supported set for {instr_shape}: "
            f"{_TCGEN05_ATOM_REPS[instr_shape]}"
        )

    laneid = CUDAAxis.laneid
    wid = CUDAAxis.wid_in_wg
    N = rep
    shape = instr_shape
    # All m-strides below are written in fp32-reg units; we multiply by
    # elem_per_32b at the end and prepend a C_pack iter for the 16-bit case
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
        # split the dispatch uses when emitting the two PTX calls. The
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


class SwizzleMode(Enum):
    """The swizzle mode of the TMA."""

    SWIZZLE_NONE = 0
    SWIZZLE_32B_ATOM = 1
    SWIZZLE_64B_ATOM = 2
    SWIZZLE_128B_ATOM = 3


def mma_atom_layout(dtype: str, swizzle_mode: SwizzleMode | int) -> ComposeLayout:
    """Generate the MMA-compatible shared-memory atom layout."""
    bits = tvm.DataType(dtype).bits
    if isinstance(swizzle_mode, int):
        swizzle_mode = SwizzleMode(swizzle_mode)
    per_element = (128 // bits).bit_length() - 1
    swizzle_len = swizzle_mode.value
    atom_len = 3
    period = 1 << (per_element + swizzle_len + atom_len)
    return ComposeLayout(per_element, swizzle_len, atom_len, TileLayout(S[(period,)]))


def mma_atom_shape(dtype: str, swizzle_mode: SwizzleMode | int, shape: list[int] | None = None):
    """Generate the MMA-compatible shared-memory atom shape."""
    bits = tvm.DataType(dtype).bits
    if isinstance(swizzle_mode, int):
        swizzle_mode = SwizzleMode(swizzle_mode)
    atom_shape = {
        SwizzleMode.SWIZZLE_32B_ATOM: [8, 256],
        SwizzleMode.SWIZZLE_64B_ATOM: [8, 512],
        SwizzleMode.SWIZZLE_128B_ATOM: [8, 1024],
    }[swizzle_mode]
    atom_shape[-1] //= bits
    if shape is None:
        return atom_shape
    atom_shape = [1] * (len(shape) - len(atom_shape)) + atom_shape
    return atom_shape


def mma_shared_layout(dtype: str, swizzle_mode: SwizzleMode | int, shape) -> Layout:
    """Generate the MMA-compatible shared-memory layout for shape and dtype.

    It uses a default tiling strategy to tile the TMA atom layout into the shared memory.
    """
    if isinstance(swizzle_mode, int):
        swizzle_mode = SwizzleMode(swizzle_mode)
    if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
        # No-swizzle MMA smem is the packed-16B atom: offset e16*m +
        # M*e16*(k//e16) + k%e16 (e16=128/bits); plain tile if K % e16 != 0.
        bits = tvm.DataType(dtype).bits
        e16 = 128 // bits
        m, k = int(shape[-2]), int(shape[-1])
        if k % e16 == 0:
            lead = [int(s) for s in shape[:-2]]
            extents = [*lead, m, k // e16, e16]
            strides = []
            stride = m * k
            for e in reversed(lead):
                strides.insert(0, stride)
                stride *= e
            strides += [e16, m * e16, 1]
            return TileLayout(S[tuple(extents) : tuple(strides)]).canonicalize()
        return TileLayout(S[tuple(shape)]).canonicalize()
    atom_shape = mma_atom_shape(dtype, swizzle_mode, shape)
    layout = mma_atom_layout(dtype, swizzle_mode)
    tile_to_shape = copy.copy(atom_shape)
    tile_to_shape[-2] = shape[-2]
    return layout.tile_to(tile_to_shape, atom_shape).tile_to(shape, tile_to_shape).canonicalize()


def sf_smem_layout(rows, SF_K, sf_per_mma, sf_reuse=1, pipe_depth=None):
    """SMEM-side layout for SF in tcgen05.cp scale-factor copy.

    The hardware reads SFs in 128-row super-blocks: 32 lanes x 16 bytes/lane.
    The 16 bytes per lane row decompose as
    ``M_SF_INNER (=4) x sf_per_mma x in_lane_K`` where
    ``in_lane_K = epc / sf_per_mma`` and ``epc = 4`` (32-bit TMEM cell / 8-bit
    SF). The remaining ``K_outer = SF_K / epc`` super-blocks march along K
    with stride 512. ``sf_reuse > 1`` appends a stride-0 broadcast dim.

    Buffer shape: ``(rows, SF_K * sf_reuse)`` (or ``(pipe_depth, rows,
    SF_K * sf_reuse)``). Mirrors :func:`sf_tmem_layout` parameterization.

    Args:
        rows:        Multiple of 128; M-direction rows.
        SF_K:        Number of unique SFs along K per row (multiple of 4).
        sf_per_mma:  Atom inner SFs per MMA-K step (must divide 4).
                     nvfp4=4, mxfp4=2, fp8=1.
        sf_reuse:    Broadcast factor (stride-0 dim). 1 = no broadcast.
        pipe_depth:  Optional pipeline depth as outermost dim.
    """
    epc = 4
    M_SUPER_ROWS = 128
    LANE = 32
    M_SF_INNER = M_SUPER_ROWS // LANE
    if rows % M_SUPER_ROWS != 0:
        raise ValueError(f"rows={rows} must be a multiple of {M_SUPER_ROWS}")
    if epc % sf_per_mma != 0:
        raise ValueError(f"sf_per_mma={sf_per_mma} must divide epc={epc}")
    if SF_K % epc != 0:
        raise ValueError(f"SF_K={SF_K} must be a multiple of epc={epc}")

    in_lane_K = epc // sf_per_mma
    K_outer = SF_K // epc
    M_super = rows // M_SUPER_ROWS
    LANE_BYTES = epc * M_SF_INNER  # 16
    SUPER_BYTES = LANE_BYTES * LANE  # 512
    K_TOTAL_BYTES = SUPER_BYTES * K_outer
    STAGE_BYTES = K_TOTAL_BYTES * M_super

    raw_shape = [M_super, M_SF_INNER, LANE, K_outer, sf_per_mma, in_lane_K]
    raw_strides = [K_TOTAL_BYTES, epc, LANE_BYTES, SUPER_BYTES, in_lane_K, 1]
    if sf_reuse > 1:
        raw_shape.append(sf_reuse)
        raw_strides.append(0)
    # Drop unit (extent-1) dims for cleaner canonical form.
    shape = [s for s in raw_shape if s != 1]
    strides = [st for s, st in zip(raw_shape, raw_strides) if s != 1]
    if pipe_depth is not None:
        shape = [pipe_depth, *shape]
        strides = [STAGE_BYTES, *strides]
    return TileLayout(S[tuple(shape) : tuple(strides)])


def sf_tmem_layout(rows, SF_K, sf_per_mma, sf_reuse=1, pipe_depth=None):
    """Create a TileLayout for SFA/SFB TMEM via atom direct_sum outer (+ optional reuse dim).

    Args:
        rows:        CTA M-direction row count (multiple of 32).
        SF_K:        Number of *unique* SFs along K per row (loaded from gmem).
        sf_per_mma:  Atom inner SFs — number of SFs one MMA reads in K.
                     Equals ``mma_k // sf_vec``: nvfp4=4 (mma_k=64,sf_vec=16),
                     mxfp4=2 (64,32), fp8=1 (32,32).
        sf_reuse:    Number of MMAs that reuse one physical SF group via a
                     stride-0 broadcast dim. Equals ``quant_size // mma_k``;
                     default 1 (no reuse). fp8 blockwise with quant=128 and
                     mma_k=32 → ``sf_reuse=4``.
        pipe_depth:  Optional outer pipe-depth dim for double-buffered TMEM SF
                     allocations. Stride is ``M*epc @ CUDAAxis.TCol`` (one stage spans
                     ``M*epc`` cols). When ``None`` no pipe dim is added.

    Buffer shape: ``(rows, SF_K * sf_reuse)`` (or ``(pipe_depth, rows,
    SF_K * sf_reuse)``). Gemm dispatch iterates the last dim
    ``SF_K * sf_reuse`` MMA times; only ``SF_K`` distinct SFs are physically
    stored due to broadcast. Scale factor dtype is assumed 8-bit (epc=4);
    all current SF formats (e8m0fnu, e4m3fn) fit.
    """
    if SF_K % sf_per_mma != 0:
        raise ValueError(f"SF_K={SF_K} must be a multiple of sf_per_mma={sf_per_mma}")
    K = SF_K // sf_per_mma  # outer K iterations of unique SFs

    M = rows // 32
    epc = 4  # 32-bit TMEM column / 8-bit SF

    # Atom: one 32-row chunk, one MMA's worth of SF.
    atom = TileLayout(
        S[(32, sf_per_mma) : (1 @ CUDAAxis.TLane, 1 @ CUDAAxis.TCol)] + R[4 : 32 @ CUDAAxis.TLane]
    )

    if K == 1:
        outer = TileLayout(S[M : epc @ CUDAAxis.TCol])
    else:
        # Pack consecutive ki's within one uint32 TMEM column when possible.
        pack_factor = epc // sf_per_mma
        while pack_factor > 1 and K % pack_factor != 0:
            pack_factor //= 2
        if pack_factor > 1:
            K_outer = K // pack_factor
            if K_outer == 1:
                outer = TileLayout(
                    S[(M, pack_factor) : (epc @ CUDAAxis.TCol, sf_per_mma @ CUDAAxis.TCol)]
                )
            else:
                outer = TileLayout(
                    S[
                        (M, K_outer, pack_factor) : (
                            epc @ CUDAAxis.TCol,
                            M * epc @ CUDAAxis.TCol,
                            sf_per_mma @ CUDAAxis.TCol,
                        )
                    ]
                )
        else:
            outer = TileLayout(S[(M, K) : (epc @ CUDAAxis.TCol, M * epc @ CUDAAxis.TCol)])

    base = atom.direct_sum(outer, left_shape=[M, K], right_shape=[32, sf_per_mma])
    if sf_reuse == 1 and pipe_depth is None:
        return base
    shard = list(base.shard)
    if sf_reuse > 1:
        # Append a stride-0 reuse dim on CUDAAxis.TCol for fp8 blockwise (vec_NX) mode.
        shard.append(Iter(sf_reuse, 0, shard[0].axis))
    if pipe_depth is not None:
        # Prepend a pipe-depth dim that strides one stage (M*epc CUDAAxis.TCols).
        shard.insert(0, Iter(pipe_depth, M * epc, shard[0].axis))
    return TileLayout.from_iters(shard, list(base.replica), dict(base.offset))
