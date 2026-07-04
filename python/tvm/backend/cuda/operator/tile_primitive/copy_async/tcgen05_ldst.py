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

"""copy_async dispatch: ``tcgen05.ld`` / ``tcgen05.st`` (tmem <-> local registers).

Both are inherently async; this dispatch emits the PTX instruction only and
leaves completion (``tcgen05.wait.ld`` / ``tcgen05.wait.st``) to the caller.
Callers that want sync semantics should issue the matching wait after the copy.
"""

import tvm
from tvm.arith import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.layout import (
    S,
    TCol,
    TileLayout,
    TLane,
    tcgen05_atom_layout,
    tid_in_wg,
    tmem_datapath_layout,
)
from tvm.tirx.operator.tile_primitive import DispatchContext, predicate, register_dispatch
from tvm.tirx.tile_primitive import TilePrimitiveCall

from ..common import get_st_extent
from ..copy import _is_valid_copy, _scope_allowed
from ..exec_scope_utils import exec_scope_ok

# Per-warp fp32-column factor for each instr_shape (mirrors
# ``_TCGEN05_COL_FACTOR_FP32`` in ``tvm.tirx.layout``; .16x64b → 2,
# .16x128b → 4, .16x256b → 8). Source: PTX ISA Table 49.
_TCGEN05_COL_FACTOR_FP32 = {"16x64b": 2, "16x128b": 4, "16x256b": 8}


def _match_tcgen05_atom_layout(buf):
    """Return ``(instr_shape, rep, frag_rows)`` if ``buf.layout`` matches a
    tcgen05 ``.16x*b`` atom layout for some supported ``instr_shape``.

    The local buffer shape ``(frag_rows, K)`` (``frag_rows`` ∈ {64, 128})
    together with the dtype determines the candidate ``rep`` for each
    ``instr_shape``; we just probe the three shapes x two frag_rows and
    structurally compare. ``None`` if no atom layout matches.
    """
    if len(buf.shape) != 2:
        return None
    rows, cols = int(buf.shape[0]), int(buf.shape[1])
    if rows not in (64, 128):
        return None
    dtype = buf.dtype
    layout_c = buf.layout.canonicalize()
    for shape in _TCGEN05_COL_FACTOR_FP32:
        try:
            cand = tcgen05_atom_layout(shape, (rows, cols), dtype).canonicalize()
        except ValueError:
            continue
        try:
            tvm.ir.assert_structural_equal(layout_c, cand)
        except (AssertionError, ValueError):
            continue
        # Recover rep from cols (same arithmetic the factory uses).
        elem_per_32b = 32 // DataType(dtype).bits
        rep = cols // (_TCGEN05_COL_FACTOR_FP32[shape] * elem_per_32b)
        return shape, rep, rows
    return None


def _classify_tmem_datapath(tmem_buf):
    """Return ``"D"`` / ``"F"`` if ``tmem_buf.layout`` matches a known tcgen05
    datapath (PTX ISA §9.7.16.10.5), else ``None``.

    Layout D (M=128, identity row→lane) is the default returned by
    ``_default_tmem_layout``. Layout F (M=64 non-``.ws``, scattered) is the
    explicit opt-in produced by ``tmem_pool.alloc(..., datapath="F")``.
    The dispatch uses this to pair each ``.16x*b`` / ``.32x32b`` atom with a
    compatible layout — see ``_check_tmem_layout_for_atom``.
    """
    if tmem_buf.layout is None:
        return None
    buf_layout = tmem_buf.layout.canonicalize()
    rows = int(tmem_buf.shape[0])
    if rows == 128:
        cand = tmem_datapath_layout("D", 128, tmem_buf.shape[1]).canonicalize()
        try:
            tvm.ir.assert_structural_equal(buf_layout, cand)
            return "D"
        except (AssertionError, ValueError):
            return None
    if rows == 64:
        cand = tmem_datapath_layout("F", 64, tmem_buf.shape[1]).canonicalize()
        try:
            tvm.ir.assert_structural_equal(buf_layout, cand)
            return "F"
        except (AssertionError, ValueError):
            return None
    return None


# Compatibility matrix between the TMEM buffer's datapath layout and the
# tcgen05 ld/st atom requested by ``T.copy_async``:
#
#   datapath x atom              | accepted? | rationale
#   ---------------------------- | --------- | --------------------------------
#   D (M=128 full)  x .32x32b    | yes       | full 128 lanes, all 32 per warp
#   D (M=128 full)  x .16x*b M=64| yes       | reads first half-slab (lanes
#                                |           |   0..15 of each warp partition)
#                                |           |   — the rest of acc is wasted
#                                |           |   for this atom but valid data
#   D (M=128 full)  x .16x*b M=128| yes      | reads all 128 lanes via row=0
#                                |           |   and row=16 PTX issues
#   F (M=64 scatter)x .16x*b M=64| yes       | canonical pairing - F's row
#                                |           |   indexing matches the atom's
#                                |           |   scatter access
#   F (M=64 scatter)x .16x*b M=128| no       | F only writes the low slab; the
#                                |           |   high slab (row=16) is garbage
#   F (M=64 scatter)x .32x32b    | no       | F only utilizes 16 of each
#                                |           |   warp's 32 lanes
_TMEM_ATOM_COMPAT = {
    ("D", "32x32b", 128): True,
    ("D", "16x*b", 64): True,
    ("D", "16x*b", 128): True,
    ("F", "32x32b", 128): False,
    ("F", "16x*b", 64): True,
    ("F", "16x*b", 128): False,
}


def _check_tmem_layout_for_atom(tmem_buf, atom_kind, frag_rows):
    """Raise ``ValueError`` if the TMEM buffer's datapath layout is
    incompatible with the requested ``tcgen05`` atom.

    ``atom_kind`` is ``"32x32b"`` or ``"16x*b"``; ``frag_rows`` is the
    register-side fragment row count (128 for ``.32x32b`` and ``.16x*b``
    M=128 variants, 64 for ``.16x*b`` M=64). If the buffer's layout is
    unrecognized (i.e. it isn't Layout D or Layout F), the dispatch falls
    back to the structural assertions below.
    """
    datapath = _classify_tmem_datapath(tmem_buf)
    if datapath is None:
        return None
    allowed = _TMEM_ATOM_COMPAT.get((datapath, atom_kind, frag_rows), False)
    if not allowed:
        raise ValueError(
            f"tcgen05 dispatch: TMEM buffer with datapath={datapath!r} is "
            f"incompatible with atom={atom_kind!r} (frag_rows={frag_rows}). "
            f"See PTX ISA §9.7.16.10.5 for datapath/atom pairings; the "
            f"buffer was allocated via tmem_pool.alloc(..., "
            f"datapath={datapath!r})."
        )
    return datapath


def copy_tmem_local_impl(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc | None:
    op_call = TilePrimitiveCall.downcast(op_call)
    dst_buffer_region, src_buffer_region = op_call.dst, op_call.src
    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    if src.scope() == "tmem" and dst.scope() == "local":
        direction = "tmem2local"
        tmem_region, local_region = src_buffer_region, dst_buffer_region
    elif src.scope() == "local" and dst.scope() == "tmem":
        direction = "local2tmem"
        local_region, tmem_region = src_buffer_region, dst_buffer_region
    else:
        raise ValueError(f"Unsupported src scope {src.scope()} and dst scope {dst.scope()}")

    tmem_buf, local_buf = tmem_region.buffer, local_region.buffer

    assert tmem_buf.layout is not None
    assert local_buf.layout is not None
    assert tmem_buf.dtype == local_buf.dtype
    assert tmem_buf.allocated_addr is not None

    analyzer = Analyzer()
    elem_size = DataType(local_buf.dtype).bits
    elem_per_32b = 32 // elem_size
    assert len(local_buf.shape) == len(tmem_buf.shape) == 2

    # Try the .16x* (M=64) path first by structural-matching the register-side
    # layout against ``tcgen05_atom_layout(instr_shape, (64, K), dtype)``. The
    # TMEM-side layout is the standard (128, W):(1@TLane, 1@TCol); the M=64
    # fragment lives at lanes 0..15 of each warp's accessible slab (per PTX
    # 9.7.16.8.1), so each warp issues with row_offset=0 and collectively the
    # 4 warps cover all 64 rows.
    atom_match = _match_tcgen05_atom_layout(local_buf)

    if atom_match is not None:
        shape, num, frag_rows = atom_match
        return _emit_16xnb_path(
            shape=shape,
            num=num,
            frag_rows=frag_rows,
            direction=direction,
            tmem_buf=tmem_buf,
            local_buf=local_buf,
            tmem_region=tmem_region,
            local_region=local_region,
            elem_per_32b=elem_per_32b,
            analyzer=analyzer,
        )

    # Fall through to the existing .32x32b (M=128) path.
    return _emit_32x32b_path(
        direction=direction,
        tmem_buf=tmem_buf,
        local_buf=local_buf,
        tmem_region=tmem_region,
        local_region=local_region,
        elem_per_32b=elem_per_32b,
        analyzer=analyzer,
    )


def _emit_32x32b_path(
    *, direction, tmem_buf, local_buf, tmem_region, local_region, elem_per_32b, analyzer
) -> PrimFunc:
    """Original M=128 fragment path using ``tcgen05.{ld,st}.32x32b.xN``."""
    # local: 128xWIDTH <-> tmem: 128xSHAPE[1]
    # ``.32x32b`` accesses 32 lanes per warp — the full warp partition — so
    # the TMEM buffer must be Layout D (M=128 full datapath). Reject Layout F.
    _check_tmem_layout_for_atom(tmem_buf, "32x32b", 128)
    assert analyzer.can_prove_equal(local_buf.shape[0], 128)
    assert analyzer.can_prove_equal(tmem_buf.shape[0], 128)

    # Check width is valid for 32x32b, and determine num
    width = local_region.region[1].extent
    candidates = [1, 2, 4, 8, 16, 32, 64, 128]

    if not analyzer.can_prove_equal(tvm.tirx.floormod(width, elem_per_32b), 0):
        raise ValueError(f"Width {width} is not valid for tcgen05.ld/st with shape 32x32b")

    num = None
    for n in candidates:
        if analyzer.can_prove_equal(tvm.tirx.floordiv(width, elem_per_32b), n):
            num = n
            break
    else:
        raise ValueError(f"Width {width} is not valid for tcgen05.ld/st with shape 32x32b")

    tmem_st, tmem_extent = get_st_extent(tmem_region)
    local_st, local_extent = get_st_extent(local_region)
    # tmem layout (128, WIDTH):(1@TLane, 1@TCol)
    tmem_layout = TileLayout(S[(128, tmem_buf.shape[1]) : (1 @ TLane, 1 @ TCol)]).canonicalize()
    # local layout
    TileLayout(S[(128, width) : (1 @ tid_in_wg, 1)]).canonicalize()

    tvm.ir.assert_structural_equal(tmem_buf.layout.canonicalize(), tmem_layout)
    # local: [0:128, 0:WIDTH] <-> tmem: [0:128, st:st+WIDTH]
    assert analyzer.can_prove_equal(tmem_st[0], 0)
    assert analyzer.can_prove_equal(tmem_extent[0], 128)

    assert analyzer.can_prove_equal(local_st[0], 0)
    assert analyzer.can_prove_equal(local_extent[0], 128)

    offset = tmem_st[1]
    assert analyzer.can_prove_equal(tvm.tirx.floormod(offset, elem_per_32b), 0)
    offset_32b = tvm.tirx.floordiv(offset, elem_per_32b)
    assert analyzer.can_prove_equal(tmem_extent[1], width), (
        f"tmem_extent[1]: {tmem_extent[1]}, width: {width}"
    )

    # assert analyzer.can_prove_equal(local_st[1], 0)
    assert analyzer.can_prove_equal(local_extent[1], width)

    op = T.ptx.tcgen05.ld if direction == "tmem2local" else T.ptx.tcgen05.st

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def impl():
        local_storage = local_buf.view(local_buf.shape[1] * elem_per_32b, layout=TileLayout(S[num * elem_per_32b]))  # noqa: E501
        local_32b = local_storage.view("uint32")
        op(tmem_buf.allocated_addr[0], *[local_32b[local_st[1] // elem_per_32b+i] for i in range(num)], shape="32x32b", num=num, row=0, col=offset_32b)  # noqa: E501
    # fmt: on
    return impl


def _emit_16xnb_path(
    *,
    shape,
    num,
    frag_rows,
    direction,
    tmem_buf,
    local_buf,
    tmem_region,
    local_region,
    elem_per_32b,
    analyzer,
) -> PrimFunc:
    """``.16x*b`` fragment path using ``tcgen05.{ld,st}.<shape>.x<num>`` (one
    of ``.16x64b``, ``.16x128b``, ``.16x256b``).

    Each of the warpgroup's 4 warps issues the atom with ``row_offset=0`` to
    cover lanes 0..15 of its 32-lane TMEM partition (one 16-row slab); the
    four warps collectively span M=64 rows. When ``frag_rows == 128`` the
    dispatch emits a second issue with ``row_offset=16`` to also cover lanes
    16..31 of each warp's partition, doubling the fragment's row coverage to
    M=128. The two atoms share the same column footprint; the layout factory
    surfaces the combined per-thread register vector with the second slab's
    regs in the high half of the m-axis (so the dispatch can split regs
    contiguously between the two PTX calls).
    """
    # Per-atom column footprint in fp32 columns:
    #   .16x64b  → 2N    .16x128b → 4N    .16x256b → 8N
    col_factor_fp32 = {"16x64b": 2, "16x128b": 4, "16x256b": 8}[shape]
    # Per-thread register count per 16-row slab (in 32-bit units):
    #   .16x64b.xN  → N        .16x128b.xN → 2N      .16x256b.xN → 4N
    regs_per_thread_per_slab = {"16x64b": num, "16x128b": 2 * num, "16x256b": 4 * num}[shape]
    n_slabs = frag_rows // 64  # 1 for M=64, 2 for M=128
    assert n_slabs in (1, 2)
    regs_per_thread = regs_per_thread_per_slab * n_slabs
    # Logical column width that the local buffer view exposes (in element units).
    width_elems = col_factor_fp32 * num * elem_per_32b
    # Per-thread storage in element units (same total bits as the register vector).
    per_thread_elems = regs_per_thread * elem_per_32b

    # Local-side: shape (frag_rows, K_cols)
    assert analyzer.can_prove_equal(local_buf.shape[0], frag_rows), (
        f".16x*b path expects local_buf rows={frag_rows}, got {local_buf.shape[0]}"
    )
    assert analyzer.can_prove_equal(local_buf.shape[1], width_elems), (
        f".16x*b path expects local_buf cols={width_elems}, got {local_buf.shape[1]}"
    )

    # TMEM-side: structurally classify the buffer's datapath (D or F) and
    # reject incompatible pairings. The PTX is identical in either case (the
    # warp partition rule and the atom's lane access pattern are baked into
    # the hardware); the layout classification just keeps the buffer's
    # logical row indexing in sync with the physical TMEM occupation.
    datapath = _check_tmem_layout_for_atom(tmem_buf, "16x*b", frag_rows)

    if datapath == "F":
        # Layout F: buffer shape (64, W), scattered row→lane.
        assert analyzer.can_prove_equal(tmem_buf.shape[0], 64), (
            f".16x*b Layout F expects tmem_buf rows=64, got {tmem_buf.shape[0]}"
        )
        tmem_rows = 64
    else:
        # Layout D (or untagged legacy buffers): shape (128, W), identity.
        # The legacy structural check below still fires for untagged buffers
        # so we don't silently accept arbitrary layouts.
        assert analyzer.can_prove_equal(tmem_buf.shape[0], 128), (
            f".16x*b path expects tmem_buf rows=128, got {tmem_buf.shape[0]}"
        )
        if datapath is None:
            tmem_layout = TileLayout(
                S[(128, tmem_buf.shape[1]) : (1 @ TLane, 1 @ TCol)]
            ).canonicalize()
            tvm.ir.assert_structural_equal(tmem_buf.layout.canonicalize(), tmem_layout)
        tmem_rows = 128

    tmem_st, tmem_extent = get_st_extent(tmem_region)
    local_st, local_extent = get_st_extent(local_region)

    # Rows must span the full frag. The COLUMN extent may be a sub-multiple of
    # the atom's full width ``width_elems`` — i.e. a per-chunk column slice of a
    # wider frag (e.g. an epilogue that loads one big (128, MMA_N) frag in
    # EPI_TILE-wide chunks). The atom layout maps consecutive columns to
    # consecutive registers within each slab, so a column slice occupies a
    # contiguous register window; we emit ``num_eff`` (the slice's atom rep) at
    # the slab base + the column's register offset. When the slice IS the full
    # atom (the common case), num_eff == num and reg offset == 0 (no change).
    assert analyzer.can_prove_equal(local_st[0], 0)
    assert analyzer.can_prove_equal(local_extent[0], frag_rows)
    assert analyzer.can_prove_equal(tmem_st[0], 0)
    assert analyzer.can_prove_equal(tmem_extent[0], frag_rows)
    # local and tmem column slices must match and divide the atom's full width.
    assert analyzer.can_prove_equal(local_extent[1], tmem_extent[1])
    slice_w = int(local_extent[1])
    assert width_elems % slice_w == 0, f"slice width {slice_w} must divide atom width {width_elems}"
    num_eff = num * slice_w // width_elems
    regs_eff = regs_per_thread_per_slab * slice_w // width_elems
    del tmem_rows  # only used for the structural check above

    col_off = tmem_st[1]
    assert analyzer.can_prove_equal(tvm.tirx.floormod(col_off, elem_per_32b), 0)
    col_off_32b = tvm.tirx.floordiv(col_off, elem_per_32b)
    local_col_off = local_st[1]
    assert analyzer.can_prove_equal(tvm.tirx.floormod(local_col_off, elem_per_32b), 0)
    local_col_off_elems = local_col_off

    is_load = direction == "tmem2local"
    op = T.ptx.tcgen05.ld if is_load else T.ptx.tcgen05.st
    # We intentionally do *not* emit ``.pack::16b`` / ``.unpack::16b`` for
    # 16-bit dtypes. That qualifier would store one 16-bit element per 32-bit
    # TMEM cell (LOW half only, HIGH half wasted) — fine for some CUTLASS
    # epilogues but a 2x TMEM waste vs. the existing ``.32x32b`` convention,
    # which packs two 16-bit elements per cell. By using the plain ``.b32``
    # form we keep TMEM dense (2 elements per 32-bit cell); the per-thread
    # register file holds two packed 16-bit values per 32-bit register, and
    # the layout factory's iters describe that packing.

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def impl():
        # Per-thread 1-D flat view of the local storage, then a uint32 view
        # for the register-pointer arguments of the PTX builtin.
        local_storage = local_buf.view(per_thread_elems, layout=TileLayout(S[per_thread_elems]))
        local_32b = local_storage.view("uint32")
        # Register offset of the column slice within each slab. The old
        # ``local_col_off // elem_per_32b`` is only correct when the slice IS the
        # full atom; in general consecutive columns advance registers at the rate
        # (regs_per_thread_per_slab / width_elems). For a full-atom load the
        # offset is 0 either way, so existing callers are unaffected.
        local_reg_base = local_col_off_elems * regs_per_thread_per_slab // width_elems
        for slab in range(n_slabs):
            reg_base = slab * regs_per_thread_per_slab
            op(
                tmem_buf.allocated_addr[0],
                *[local_32b[local_reg_base + reg_base + i] for i in range(regs_eff)],
                shape=shape, num=num_eff, row=slab * 16, col=col_off_32b,
            )
    # fmt: on
    return impl


# === Variant: copy_async/tmem<->local (priority=10) ===
#
# When: one buffer is in tmem (tensor memory, Blackwell SM100+) and the other
# is in local scope, at warpgroup exec scope.
#
# Emits: T.ptx.tcgen05.ld / T.ptx.tcgen05.st (async). The caller is
# responsible for issuing the matching ``T.ptx.tcgen05.wait.ld`` /
# ``T.ptx.tcgen05.wait.st`` when synchronization is required.
@register_dispatch(
    "copy_async",
    "cuda",
    variant="tmem<->local",
    priority=10,
    when=[
        predicate("validate_copy_op", _is_valid_copy),
        predicate("exec_scope", exec_scope_ok, expected_scopes=["warpgroup"]),
        predicate(
            "storage_scope", _scope_allowed, allowed_pairs=[("tmem", "local"), ("local", "tmem")]
        ),
    ],
)
def copy_async_schedule_tmem_local_async(
    op_call: TilePrimitiveCall, sctx: DispatchContext
) -> PrimFunc:
    return copy_tmem_local_impl(op_call, sctx)
