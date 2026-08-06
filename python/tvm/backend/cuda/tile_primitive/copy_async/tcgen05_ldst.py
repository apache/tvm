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
#   B (M=64 2x2)    x bare atom  | no       | B splits N into two N/2
#                                |           |   lane-halves; use the dedicated
#                                |           |   logical (64, N) Layout B image
#                                |           |   handled by
#                                |           |   _emit_datapath_b_path
_TMEM_ATOM_COMPAT = {
    ("D", "32x32b", 128): True,
    ("D", "16x*b", 64): True,
    ("D", "16x*b", 128): True,
    ("F", "32x32b", 128): False,
    ("F", "16x*b", 64): True,
    ("F", "16x*b", 128): False,
    ("B", "32x32b", 128): False,
    ("B", "16x*b", 64): False,
    ("B", "16x*b", 128): False,
}


def _classify_tmem_datapath(tmem_buf):
    """Return ``(datapath, sub_slab)`` if ``tmem_buf.layout`` matches a known
    tcgen05 datapath (PTX ISA §9.7.16.10.5), else ``None``.

    Layout D (M=128, identity row→lane) is the default returned by
    ``_default_tmem_layout``. Layout F (M=64 non-``.ws``, scattered) is the
    explicit opt-in produced by ``tmem_pool.alloc(..., datapath="F")``.
    Layout B is the per-CTA M=64, ``.cta_group::2`` "2x2" placement produced
    by ``tmem_pool.alloc(..., datapath="B")``.
    The dispatch uses this to pair each ``.16x*b`` / ``.32x32b`` atom with a
    compatible layout — see ``_check_tmem_layout_for_atom``.

    ``sub_slab`` is always 0 for Layout D and B. For Layout F it selects the
    lower (0) or upper (1) 16-lane half of each warp's 32-lane partition.
    """
    if tmem_buf.layout is None:
        return None
    buf_layout = tmem_buf.layout.canonicalize()
    rows = int(tmem_buf.shape[0])
    if rows == 128:
        cand = tmem_datapath_layout("D", 128, tmem_buf.shape[1]).canonicalize()
        try:
            tvm.ir.assert_structural_equal(buf_layout, cand)
            return ("D", 0)
        except (AssertionError, ValueError):
            return None
    if rows == 64:
        # Layout B splits N into two N/2 column halves, together spanning all
        # 128 lanes. Its structure is disjoint from Layout F; try it first.
        if int(tmem_buf.shape[1]) % 2 == 0:
            cand = tmem_datapath_layout("B", 64, tmem_buf.shape[1]).canonicalize()
            try:
                tvm.ir.assert_structural_equal(buf_layout, cand)
                return ("B", 0)
            except (AssertionError, ValueError):
                pass
        # Layout F may occupy either 16-lane half of each warp's 32-lane
        # partition. The layout carries that choice as a +16 TLane offset;
        # thread it through to the PTX row immediate instead of adding an
        # out-of-band copy_async option.
        for sub_slab in (0, 1):
            cand = tmem_datapath_layout(
                "F", 64, tmem_buf.shape[1], sub_slab=sub_slab
            ).canonicalize()
            try:
                tvm.ir.assert_structural_equal(buf_layout, cand)
                return ("F", sub_slab)
            except (AssertionError, ValueError):
                continue
        return None
    return None


def _tmem_window(tmem_buf, tmem_region, atom_kind, frag_rows, analyzer):
    """Resolve a tcgen05 ld/st TMEM operand region to its
    ``(width, col_off, sub_slab)`` column window (element units).

    The region is ``(frag_rows, width)`` optionally prefixed by point-indexed
    dims (a staged view's stage axis), which fold into the TCol offset when the
    layout is sliced. ``frag_rows`` (128 or 64) selects datapath D or F; the
    datapath x atom pairing is gated by ``_TMEM_ATOM_COMPAT`` (PTX ISA
    §9.7.16.10.5) and the sliced window is checked against the datapath layout.

    ``sub_slab`` is 0 for Layout D. A Layout F view may occupy either 16-lane
    half of each warp's 32-lane TMEM partition; the layout carries that choice
    as a ``+16`` TLane offset, and it is reported here so the caller can bias
    the PTX row immediate rather than silently dropping the half-slab.
    """
    _, extent = get_st_extent(tmem_region)
    for d in range(len(extent) - 2):
        assert analyzer.can_prove_equal(extent[d], 1), (
            f"tcgen05 ld/st: leading tmem region dims must be points, got extent {extent[d]}"
        )
    assert analyzer.can_prove_equal(extent[-2], frag_rows), (
        f"tcgen05 ld/st: tmem row extent must be {frag_rows}, got {extent[-2]}"
    )
    width = extent[-1]
    datapath = "D" if int(tmem_buf.shape[-2]) == 128 else "F"
    if not _TMEM_ATOM_COMPAT.get((datapath, atom_kind, frag_rows), False):
        raise ValueError(
            f"tcgen05 dispatch: TMEM buffer with datapath={datapath!r} is "
            f"incompatible with atom={atom_kind!r} (frag_rows={frag_rows}). "
            f"See PTX ISA §9.7.16.10.5 for datapath/atom pairings."
        )
    window = tmem_buf.layout.slice(tmem_buf.shape, tmem_region.region).canonicalize()
    if datapath == "D":
        base = TileLayout(S[(frag_rows, width) : (1 @ TLane, 1 @ TCol)])
    else:
        base = tmem_datapath_layout("F", 64, width)
    expected = TileLayout.from_iters(base.shard, base.replica, window.offset).canonicalize()
    tvm.ir.assert_structural_equal(window, expected)
    lane_off = int(window.offset.get(TLane, 0))
    if lane_off not in (0, 16):
        raise ValueError(
            f"tcgen05 dispatch: TMEM window has TLane offset {lane_off}; only 0 or 16 "
            "(the Layout F sub-slab selector) are representable in the PTX row immediate."
        )
    if lane_off and datapath != "F":
        raise ValueError(
            f"tcgen05 dispatch: datapath={datapath!r} spans both sub-slabs and cannot "
            f"carry a TLane offset (got {lane_off})."
        )
    return width, window.offset.get(TCol, 0), lane_off // 16


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
    assert len(local_buf.shape) == 2

    # Datapath B is identified from the TMEM side before probing ordinary
    # register atoms. A logical (64, N) Layout B tile is physically a
    # (128, N/2) .32x32b transfer and therefore needs its dedicated reshape.
    if _classify_tmem_datapath(tmem_buf) == ("B", 0):
        return _emit_datapath_b_path(
            direction=direction,
            tmem_buf=tmem_buf,
            local_buf=local_buf,
            tmem_region=tmem_region,
            local_region=local_region,
            elem_per_32b=elem_per_32b,
            analyzer=analyzer,
        )

    # Try the .16x* (M=64) path first by structural-matching the register-side
    # layout against ``tcgen05_atom_layout(instr_shape, (64, K), dtype)``. The
    # An M=64 TMEM-side Layout F fragment lives in either lanes 0..15 or
    # 16..31 of each warp's accessible slab (per PTX 9.7.16.8.1). The layout
    # selects the half-slab, and the four warps collectively cover all 64
    # logical rows.
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
    assert analyzer.can_prove_equal(local_buf.shape[0], 128)

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

    local_st, local_extent = get_st_extent(local_region)
    # local: [0:128, 0:WIDTH] <-> tmem window: [0:128, off:off+WIDTH]
    tmem_width, offset, _ = _tmem_window(tmem_buf, tmem_region, "32x32b", 128, analyzer)

    assert analyzer.can_prove_equal(local_st[0], 0)
    assert analyzer.can_prove_equal(local_extent[0], 128)

    assert analyzer.can_prove_equal(tvm.tirx.floormod(offset, elem_per_32b), 0)
    offset_32b = tvm.tirx.floordiv(offset, elem_per_32b)
    assert analyzer.can_prove_equal(tmem_width, width), f"tmem width: {tmem_width}, width: {width}"

    # assert analyzer.can_prove_equal(local_st[1], 0)
    assert analyzer.can_prove_equal(local_extent[1], width)

    emit = _tcgen05_ldst_emitter(direction == "tmem2local", "32x32b", num)

    if elem_per_32b == 1:
        # Keep 32-bit fragments in source dtype; b32 helper makes a uint32 view change codegen.
        # fmt: off
        @T.prim_func(check_well_formed=False)
        def impl():
            local_storage = local_buf.view(local_buf.shape[1], layout=TileLayout(S[num]))
            emit(tmem_buf.allocated_addr[0], 0, offset_32b, [local_storage[local_st[1]+i] for i in range(num)])  # noqa: E501
        # fmt: on
    else:
        # 16-bit fragments are packed two elements per b32 register operand.
        # fmt: off
        @T.prim_func(check_well_formed=False)
        def impl():
            local_storage = local_buf.view(local_buf.shape[1] * elem_per_32b, layout=TileLayout(S[num * elem_per_32b]))  # noqa: E501
            local_32b = local_storage.view("uint32")
            emit(tmem_buf.allocated_addr[0], 0, offset_32b, [local_32b[local_st[1] // elem_per_32b+i] for i in range(num)])  # noqa: E501
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

    For an M=64 Layout F fragment, each warp issues the atom at the lower or
    upper 16-lane half selected by the TMEM layout, and the four warps
    collectively span 64 rows. For an M=128 Layout D fragment, the dispatch
    emits both ``row_offset=0`` and ``row_offset=16`` issues. The two atoms
    share the same column footprint; the layout factory surfaces the combined
    per-thread register vector with the second slab's regs in the high half of
    the m-axis (so the dispatch can split regs contiguously between the two
    PTX calls).
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

    # TMEM-side: resolve the region to its column window; datapath (D or F)
    # classification and atom gating happen inside _tmem_window.
    tmem_width, col_off, sub_slab = _tmem_window(
        tmem_buf, tmem_region, "16x*b", frag_rows, analyzer
    )
    # A .16x*b issue covers one 16-lane half-slab. A 64-row fragment issues once;
    # a 128-row fragment issues for both halves. ``sub_slab`` comes from the TMEM
    # layout and shifts the first issue to the upper half.
    assert sub_slab + n_slabs <= 2, (
        f".16x*b sub_slab={sub_slab} with frag_rows={frag_rows} exceeds the 2 "
        "sub-slabs of each warp's 32-lane TMEM partition"
    )
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
    # local and tmem column slices must match and divide the atom's full width.
    assert analyzer.can_prove_equal(local_extent[1], tmem_width)
    slice_w = int(local_extent[1])
    assert width_elems % slice_w == 0, f"slice width {slice_w} must divide atom width {width_elems}"
    num_eff = num * slice_w // width_elems
    regs_eff = regs_per_thread_per_slab * slice_w // width_elems

    assert analyzer.can_prove_equal(tvm.tirx.floormod(col_off, elem_per_32b), 0)
    col_off_32b = tvm.tirx.floordiv(col_off, elem_per_32b)
    local_col_off = local_st[1]
    assert analyzer.can_prove_equal(tvm.tirx.floormod(local_col_off, elem_per_32b), 0)
    local_col_off_elems = local_col_off

    is_load = direction == "tmem2local"
    emit = _tcgen05_ldst_emitter(is_load, shape, num_eff)
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
            emit(
                tmem_buf.allocated_addr[0],
                (sub_slab + slab) * 16,
                col_off_32b,
                [local_32b[local_reg_base + reg_base + i] for i in range(regs_eff)],
            )
    # fmt: on
    return impl


def _tcgen05_ldst_emitter(is_load, shape, num):
    """A callable that emits one tcgen05.ld / .st.

    Returns the Call so a traced call site evaluates it; the two instructions
    mirror each other's operand order, which is why this is not just a chain
    string. The tmem address is composed at the call site now -- ptxd is one
    instruction per call, so the row/col packing the legacy helper did
    internally moves out to `T.cuda.get_tmem_addr`.
    """
    action = "ld" if is_load else "st"
    chain = f"tcgen05.{action}.sync.aligned.{shape}.x{num}.b32"

    def emit(taddr, row, col, regs):
        addr = T.uint32(taddr)
        if (row, col) != (0, 0):
            # Only compose when there is something to add: the packing is
            # (row << 16 | col), so a zero row and column leave the base alone.
            addr = T.cuda.get_tmem_addr(addr, row, col)
        return T.ptxd[chain](*regs, addr) if is_load else T.ptxd[chain](addr, *regs)

    return emit


def _emit_datapath_b_path(
    *, direction, tmem_buf, local_buf, tmem_region, local_region, elem_per_32b, analyzer
) -> PrimFunc:
    """Read or write a Layout B (per-CTA M=64, ``.cta_group::2``) accumulator.

    Layout B splits a logical ``(64, N)`` tile into two ``N/2`` column halves
    over physical lanes 0..63 and 64..127. It is therefore transferred as one
    physical ``.32x32b`` ``(128, N/2)`` register file, re-labeled by
    ``tcgen05_atom_layout("32x32b", (64, N), "float32")``.
    """
    if elem_per_32b != 1:
        raise ValueError(
            "datapath B readback expects an fp32 fragment (32-bit cells), got "
            f"dtype={local_buf.dtype!r}"
        )
    if int(local_buf.shape[0]) != 64:
        raise ValueError(
            "datapath B (.cta_group::2 M=64) fragment must be (64, N); a "
            f"128-row .32x32b fragment reads the wrong region. Got rows={local_buf.shape[0]}. "
            "Allocate it with T.alloc_tcgen05_ldst_frag('32x32b', (64, N), 'float32')."
        )

    n_cols = int(local_buf.shape[1])
    n_half = n_cols // 2
    expected_local = tcgen05_atom_layout("32x32b", (64, n_cols), local_buf.dtype).canonicalize()
    try:
        tvm.ir.assert_structural_equal(local_buf.layout.canonicalize(), expected_local)
    except (AssertionError, ValueError) as err:
        raise ValueError(
            "datapath B (.cta_group::2 M=64) requires a matching Layout B "
            "register fragment. Allocate it with "
            "T.alloc_tcgen05_ldst_frag('32x32b', (64, N), 'float32'); a "
            ".16x*b fragment reads the wrong physical lanes and columns. "
            f"(fragment layout mismatch: {err})"
        ) from err

    # A partial logical-column slice is not contiguous after the two-way lane
    # split. Keep this first implementation deliberately strict and transfer
    # the complete logical tile on both sides.
    tmem_st, tmem_extent = get_st_extent(tmem_region)
    local_st, local_extent = get_st_extent(local_region)
    if not (
        analyzer.can_prove_equal(tmem_st[0], 0)
        and analyzer.can_prove_equal(tmem_st[1], 0)
        and analyzer.can_prove_equal(tmem_extent[0], 64)
        and analyzer.can_prove_equal(tmem_extent[1], n_cols)
    ):
        raise ValueError("datapath B copy must cover the full (64, N) TMEM buffer")
    if not (
        analyzer.can_prove_equal(local_st[0], 0)
        and analyzer.can_prove_equal(local_st[1], 0)
        and analyzer.can_prove_equal(local_extent[0], 64)
        and analyzer.can_prove_equal(local_extent[1], n_cols)
    ):
        raise ValueError("datapath B copy must cover the full (64, N) register fragment")

    emit = _tcgen05_ldst_emitter(direction == "tmem2local", "32x32b", n_half)

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def impl():
        local_storage = local_buf.view(n_half, layout=TileLayout(S[n_half]))
        local_32b = local_storage.view("uint32")
        emit(tmem_buf.allocated_addr[0], 0, 0, [local_32b[i] for i in range(n_half)])
    # fmt: on
    return impl


# === Variant: copy_async/tmem<->local (priority=10) ===
#
# When: one buffer is in tmem (tensor memory, Blackwell SM100+) and the other
# is in local scope, at warpgroup exec scope.
#
# Emits: T.ptxd.tcgen05.ld / T.ptxd.tcgen05.st (async). The caller is
# responsible for issuing the matching ``T.ptxd.tcgen05.wait__ld`` /
# ``T.ptxd.tcgen05.wait__st`` when synchronization is required.
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
