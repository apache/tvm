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

import tvm_ffi

from tvm_tirx.tirx.layout import Axis, ComposeLayout, Iter, Layout, R, S, TileLayout

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
    if ws and not (cta_group == 1 and M in (64, 128)):
        raise ValueError(
            f"alloc_tcgen05_mma_AB: ws=True requires cta_group=1, M=64 or M=128; "
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
    if operand == "D" and tvm_ffi.dtype(dtype).bits == 16:
        layout = TileLayout.from_iters(
            shard=[
                Iter(i.extent, i.stride * (2 if i.axis.name == "TCol" else 1), i.axis)
                for i in layout.shard
            ],
            replica=layout.replica,
        )
    return layout.canonicalize()


# Allowed (.shape, .num) combinations for tcgen05.ld/st atoms.
# Source: PTX ISA Table 49 (tcgen05-num-shapes-ld).
class SwizzleMode(Enum):
    """The swizzle mode of the TMA."""

    SWIZZLE_NONE = 0
    SWIZZLE_32B_ATOM = 1
    SWIZZLE_64B_ATOM = 2
    SWIZZLE_128B_ATOM = 3


def mma_atom_layout(dtype: str, swizzle_mode: SwizzleMode | int) -> ComposeLayout:
    """Generate the MMA-compatible shared-memory atom layout."""
    bits = tvm_ffi.dtype(dtype).bits
    if isinstance(swizzle_mode, int):
        swizzle_mode = SwizzleMode(swizzle_mode)
    per_element = (128 // bits).bit_length() - 1
    swizzle_len = swizzle_mode.value
    atom_len = 3
    period = 1 << (per_element + swizzle_len + atom_len)
    return ComposeLayout(per_element, swizzle_len, atom_len, TileLayout(S[(period,)]))


def mma_atom_shape(dtype: str, swizzle_mode: SwizzleMode | int, shape: list[int] | None = None):
    """Generate the MMA-compatible shared-memory atom shape."""
    bits = tvm_ffi.dtype(dtype).bits
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
        bits = tvm_ffi.dtype(dtype).bits
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
