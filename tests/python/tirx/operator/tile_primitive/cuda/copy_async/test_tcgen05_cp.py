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
"""End-to-end tests for the ``tcgen05.cp`` smem->tmem dispatch.

The bulk is bit-exact GPU round-trips through the generic planner; the
"legacy 32x128b.warpx4" section at the bottom keeps the original
single-shape tests (uint8 sub-region copies, NVFP4 SFB middle alignment,
shared-descriptor emission) that predate the generalization.

For each ``(shape, multicast, swizzle, dtype, tile size)`` combination we:

1. Fill a host buffer ``A`` with random values and stage it into shared
   memory with an MMA-style (optionally swizzled) layout.
2. Issue a generic ``Tx.copy_async(tmem_region, smem_region, shape=...,
   multicast=...)`` — no ``desc_*`` fields, so the generic planner derives
   the matrix descriptor and cp loop.
3. Read every TMEM cell back via per-warp ``tcgen05.ld.32x32b`` (each of the
   4 warps reads its own 32-lane slab at taddr lane 0) into a
   ``(128, W32)`` uint32 dump.
4. Rebuild the expected dump on the host purely from the *destination* TMEM
   layout: logical element ``i`` of the copied region carries ``A``'s value
   at the matching source coordinate, lands at the (TLane, TCol) position
   the tmem layout assigns, and is replicated at the multicast lane offsets.
   Only cells covered by the copy are compared.

The multicast row→lane mappings this asserts (verified on B200 hardware,
PTX ISA 8.8 §9.7.16.9.2 p675 + Layout E/B figures 213/207):

=============  ==============================  ====================
multicast      one copy (row → lane)           replica lane offsets
=============  ==============================  ====================
warpx4         rows 0-31 → lanes 0-31          +0 / +32 / +64 / +96
warpx2::02_13  rows 0-63 → lanes 0-63          +0 / +64
warpx2::01_23  rows 0-31 → lanes 0-31,         +0 / +32
               rows 32-63 → lanes 64-95
(4x256b)       rows 0-3 → lanes 0/32/64/96     (none)
=============  ==============================  ====================
"""

import itertools

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tvm.tirx.layout import ComposeLayout, R, S, TCol, TileLayout, TLane

# Multicast replica lane offsets: (extent, stride) pairs on TLane. These are
# the warp-slab offsets receiving copies of the data (see module docstring).
_REPLICA_PATTERN = {
    None: [],
    "warpx4": [(4, 32)],
    "warpx2::02_13": [(2, 64)],
    "warpx2::01_23": [(2, 32)],
}

# The full (shape, multicast) matrix of the generic planner.
_SHAPE_MULTICAST = [
    ("128x256b", None),
    ("128x128b", None),
    ("64x128b", "warpx2::02_13"),
    ("64x128b", "warpx2::01_23"),
    ("32x128b", "warpx4"),
    ("4x256b", None),
]


def _shape_dims(shape):
    rows, bits = shape.split("x")
    return int(rows), int(bits[:-1])


def _tmem_layout_for(shape, multicast, C):
    """Destination TMEM layout + logical buffer shape for one cp footprint.

    The layout encodes the hardware row→lane mapping of ONE copy; the
    multicast replicas are declared via ``R[...]`` so the planner can verify
    them (and the host expectation applies them explicitly).
    """
    if shape in ("128x256b", "128x128b"):
        return TileLayout(S[(128, C) : (1 @ TLane, 1 @ TCol)]), [128, C]
    if shape == "4x256b":
        # One row per warp-quadrant datapath: rows 0..3 at lanes 0/32/64/96.
        return TileLayout(S[(4, C) : (32 @ TLane, 1 @ TCol)]), [4, C]
    if shape == "32x128b":
        return TileLayout(S[(32, C) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]), [32, C]
    if shape == "64x128b":
        if multicast == "warpx2::02_13":
            # Rows 0-63 contiguous on lanes 0-63; mirrored at +64.
            return (
                TileLayout(S[(64, C) : (1 @ TLane, 1 @ TCol)] + R[2 : 64 @ TLane]),
                [64, C],
            )
        # 01_23: rows 0-31 at lanes 0-31, rows 32-63 at lanes 64-95; +32 mirror.
        # Lane split lives in the iters; buffer stays the natural (64, C) tile.
        return (
            TileLayout(S[(2, 32, C) : (64 @ TLane, 1 @ TLane, 1 @ TCol)] + R[2 : 32 @ TLane]),
            [64, C],
        )
    raise ValueError(shape)


def _next_pow2(x):
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _make_cp_kernel(
    s_full,
    s_full_shape,
    s_region,
    t_full,
    t_full_shape,
    t_region,
    dtype,
    shape,
    multicast,
    W32,
    n_tmem_cols,
    extra_cfg=None,
    infer=False,
    pre_zero=False,
):
    s_sl = tuple(slice(a, b) for a, b in s_region)
    t_sl = tuple(slice(a, b) for a, b in t_region)
    s_full_sl = tuple(slice(0, e) for e in s_full_shape)
    cfg = {"cta_group": 1} if infer else {"shape": shape, "cta_group": 1}
    if not infer and multicast is not None:
        cfg["multicast"] = multicast
    if extra_cfg:
        cfg.update(extra_cfg)

    @T.prim_func(check_well_formed=False)
    def kernel(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, s_full_shape, dtype)
        B = T.match_buffer(B_ptr, (128, W32), "uint32")
        T.device_entry()
        warp_id = T.warp_id([4])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])
        T.lane_id([32])
        A_smem = T.alloc_buffer(s_full_shape, dtype, scope="shared", layout=s_full, align=1024)
        tmem_addr = T.alloc_shared([1], "uint32")
        cp_mbar = T.alloc_shared([1], "uint64")
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=n_tmem_cols, cta_group=1)
            if tid_in_wg == 0:
                T.ptx.mbarrier.init(cp_mbar.ptr_to([0]), 1)
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.cta_sync()
            Tx.cta.copy(A_smem[s_full_sl], A[s_full_sl])
            T.cuda.cta_sync()
            tmem = T.decl_buffer(
                t_full_shape, dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=t_full
            )
            if pre_zero:
                zero_reg = T.alloc_buffer((W32,), "uint32", scope="local")
                for i in range(W32):
                    zero_reg[i] = T.uint32(0)
                for i in range(W32):
                    T.ptx.tcgen05.st(tmem_addr[0], zero_reg[i], shape="32x32b", num=1, row=0, col=i)
                T.ptx.tcgen05.wait.st()
                T.cuda.cta_sync()
                T.ptx.tcgen05.fence.after_thread_sync()
            if tid_in_wg == 0:
                Tx.copy_async(tmem[t_sl], A_smem[s_sl], **cfg)
                T.ptx.tcgen05.commit(cp_mbar.ptr_to([0]), cta_group=1)
            T.ptx.mbarrier.try_wait(cp_mbar.ptr_to([0]), 0)
            T.cuda.cta_sync()
            T.ptx.tcgen05.fence.after_thread_sync()
            # Each of the 4 warps reads its own 32-lane slab (taddr lane 0 is
            # warp-slab-relative for .32x32b), covering all 128 TMEM lanes.
            reg = T.alloc_buffer((W32,), "uint32", scope="local")
            for i in range(W32):
                T.ptx.tcgen05.ld(tmem_addr[0], reg[i], shape="32x32b", num=1, row=0, col=i)
            T.ptx.tcgen05.wait.ld()
            for i in range(W32):
                B[tid_in_wg, i] = reg[i]
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=n_tmem_cols, cta_group=1)

    return kernel


def _expected_readback(A_bits, s_region, t_full, t_full_shape, t_region, multicast, bits, W32):
    """Host reconstruction of the (128, W32) uint32 TMEM dump (+ mask)."""
    expected = np.zeros((128, W32), np.uint32)
    mask = np.zeros((128, W32), bool)
    rep_offs = [0]
    for e, st in _REPLICA_PATTERN[multicast]:
        rep_offs = [o + k * st for o in rep_offs for k in range(e)]
    t_coords = itertools.product(*[range(a, b) for a, b in t_region])
    s_coords = itertools.product(*[range(a, b) for a, b in s_region])
    per32 = 32 // bits
    for tc, sc in zip(t_coords, s_coords):
        val = int(A_bits[sc])
        av = t_full.apply(*tc, shape=t_full_shape)
        lane = int(av.get("TLane", 0))
        ce = int(av.get("TCol", 0))
        c32, sub = divmod(ce, per32)
        for off in rep_offs:
            expected[lane + off, c32] |= np.uint32(val << (sub * bits))
            mask[lane + off, c32] = True
    return expected, mask


def _build_case(
    shape, multicast, sw, dtype, n_mid, s_row_off=0, t_col_off_e=0, extra_cfg=None, infer=False
):
    """Assemble (kernel, host-check closure) for one cp configuration."""
    bits = tvm.runtime.DataType(dtype).bits
    rows, atom_bits = _shape_dims(shape)
    epa = atom_bits // bits  # elems per lane per cp instruction
    C = epa * n_mid  # copied cols (elems)
    t_C = C + t_col_off_e  # tmem buffer cols (elems)
    W32 = t_C * bits // 32
    n_tmem_cols = max(32, _next_pow2(W32))

    s_rows = max(rows + s_row_off, 8)
    if sw == 0:
        if atom_bits == 128:
            # Contiguous rows of exactly one 16B unit: atom_K = 16B → sw0.
            s_shape = [s_rows, epa]
            s_full = TileLayout(S[(s_rows, epa) : (epa, 1)])
        else:
            # 256b rows with sw0 need a real descriptor LDO: 16B unit (R, c)
            # sits at R%8 * 16B + c * LDO + R//8 * SDO.
            u = 128 // bits  # elems per 16B unit
            n_grp = s_rows // 8
            LDO_e = 8 * u
            SDO_e = 16 * u
            s_shape = [s_rows, 2 * u]
            s_full = TileLayout(S[(n_grp, 8, 2, u) : (SDO_e, u, LDO_e, 1)])
    else:
        atom_row_elems = (16 << sw) // (bits // 8)
        s_cols = max(C, atom_row_elems)
        s_shape = [s_rows, s_cols]
        s_full = mma_shared_layout(dtype, SwizzleMode(sw), s_shape)
    s_region = [(s_row_off, s_row_off + rows), (0, C)]

    t_full, t_full_shape = _tmem_layout_for(shape, multicast, t_C)
    t_region = [(0, t_full_shape[0]), (t_col_off_e, t_C)]

    kernel = _make_cp_kernel(
        s_full,
        s_shape,
        s_region,
        t_full,
        t_full_shape,
        t_region,
        dtype,
        shape,
        multicast,
        W32,
        n_tmem_cols,
        extra_cfg=extra_cfg,
        infer=infer,
    )

    def check(mod):
        A_np = tvm.testing.generate_random_array(dtype, tuple(s_shape))
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(np.zeros((128, W32), np.uint32), dev)
        mod(A, B)
        B_out = B.numpy()
        A_bits = np.asarray(A_np).view(np.uint16 if bits == 16 else np.uint32)
        exp, mask = _expected_readback(
            A_bits, s_region, t_full, t_full_shape, t_region, multicast, bits, W32
        )
        np.testing.assert_array_equal(np.where(mask, B_out, 0), np.where(mask, exp, 0))

    return kernel, check


def _compile(kernel):
    target = tvm.target.Target("cuda")
    with target:
        return tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")


def _run_case(shape, multicast, sw, dtype, n_mid, s_row_off=0, t_col_off_e=0):
    kernel, check = _build_case(
        shape, multicast, sw, dtype, n_mid, s_row_off=s_row_off, t_col_off_e=t_col_off_e
    )
    check(_compile(kernel))


# GPU round-trip: full shape x multicast x swizzle x dtype x size matrix


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("shape,multicast", _SHAPE_MULTICAST)
@pytest.mark.parametrize("sw", [1, 2, 3], ids=["SW32", "SW64", "SW128"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("n_mid", [1, 4], ids=["single_cp", "multi_cp"])
def test_cp_shape_roundtrip_swizzled(shape, multicast, sw, dtype, n_mid):
    """Bit-exact smem→tmem round-trip through the generic planner."""
    _run_case(shape, multicast, sw, dtype, n_mid)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize(
    "shape,multicast,dtype",
    [
        ("128x128b", None, "bfloat16"),
        ("128x256b", None, "bfloat16"),  # exercises the derived non-trivial LDO
        ("64x128b", "warpx2::02_13", "bfloat16"),
        ("64x128b", "warpx2::01_23", "float32"),
        ("32x128b", "warpx4", "float32"),
        ("4x256b", None, "float32"),
    ],
)
def test_cp_shape_roundtrip_nonswizzled(shape, multicast, dtype):
    """sw=0 sources: 128b rows are single 16B units; 256b rows carry LDO."""
    _run_case(shape, multicast, 0, dtype, 1)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize(
    "shape,multicast,sw,dtype,n_mid,s_row_off,t_col_off_e",
    [
        ("32x128b", "warpx4", 3, "bfloat16", 1, 8, 0),  # smem row offset
        ("64x128b", "warpx2::02_13", 3, "bfloat16", 2, 0, 8),  # tmem col offset
        ("128x256b", None, 2, "float32", 2, 0, 8),
    ],
)
def test_cp_shape_roundtrip_offsets(shape, multicast, sw, dtype, n_mid, s_row_off, t_col_off_e):
    """Sub-region copies: non-zero smem row offset / TMEM column offset."""
    _run_case(shape, multicast, sw, dtype, n_mid, s_row_off=s_row_off, t_col_off_e=t_col_off_e)


# Compile-level checks


def test_cp_4x256b_compile_emits_shape_and_count():
    """4x256b, 2 middle iterations → exactly 2 cp instructions of that shape."""
    kernel, _ = _build_case("4x256b", None, 1, "bfloat16", 2)
    mod = _compile(kernel)
    src = mod.mod.imports[0].inspect_source()
    assert "tcgen05.cp.cta_group::1.4x256b" in src, f"cp asm missing; src=\n{src}"
    helper_refs = src.count("ptx_tcgen05_cp_cta_group_1_shape_4x256b")
    assert helper_refs - 1 == 2, f"expected 2 cp calls, got {helper_refs - 1}; src=\n{src}"


def test_cp_shape_config_routes_to_generic_planner():
    """``shape=`` without desc_* must use the generic planner, not the
    explicit path: one hoisted descriptor encoded at SMEM base 0 with the
    14-bit address patch per cp."""
    kernel, _ = _build_case("128x256b", None, 3, "bfloat16", 4)
    mod = _compile(kernel)
    src = mod.mod.imports[0].inspect_source()
    assert "tcgen05.cp.cta_group::1.128x256b" in src
    # Generic-planner signature: descriptor template encoded at address 0...
    assert "reinterpret_cast<void*>((uint64_t)0)" in src
    # ...and patched per cp via the 0x3FFF address-field mask.
    assert src.count("cp_desc_ptr[0] &") == 4, f"expected 4 patched cps; src=\n{src}"


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("shape,multicast", [sm for sm in _SHAPE_MULTICAST if sm[0] != "128x128b"])
def test_cp_shape_inferred_from_layouts_matches_explicit(shape, multicast):
    """A bare copy_async infers (shape, multicast) from the buffer layouts and
    must lower byte-identically to the explicitly configured call. (128x128b
    is excluded: its layouts also fit the wider 128x256b atom, which inference
    prefers — covered by the dedicated test below.)"""
    explicit, _ = _build_case(shape, multicast, 2, "bfloat16", 2)
    inferred, _ = _build_case(shape, multicast, 2, "bfloat16", 2, infer=True)
    src_explicit = _compile(explicit).mod.imports[0].inspect_source()
    src_inferred = _compile(inferred).mod.imports[0].inspect_source()
    assert src_explicit == src_inferred


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_cp_128x128b_layout_infers_wider_256b_atom():
    """A 128x128b-compatible source also fits the 128x256b atom at half the
    instruction count; inference must pick the wider atom and stay bit-exact."""
    inferred, check = _build_case("128x128b", None, 2, "bfloat16", 2, infer=True)
    mod = _compile(inferred)
    src = mod.mod.imports[0].inspect_source()
    assert "tcgen05.cp.cta_group::1.128x256b" in src, f"expected wider atom; src=\n{src}"
    check(mod)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_cp_4x256b_lane_tiled_partial_window():
    """A region row offset folds into the taddr lane half-word: after
    pre-zeroing tmem, copying into rows [4:12) of the (16, 4, C) Layout-F
    scatter view lands on EXACTLY lanes {32q + 4..11} — those match the smem
    rows bit-for-bit and every other lane reads back all-zero."""
    dtype, bits = "bfloat16", 16
    C = 16
    W32 = C * bits // 32
    lo, hi = 4, 12
    s_full = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_32B_ATOM, [64, C])
    t_full = TileLayout(S[(16, 4, C) : (1 @ TLane, 32 @ TLane, 1 @ TCol)])
    kernel = _make_cp_kernel(
        s_full,
        [64, C],
        [(0, (hi - lo) * 4), (0, C)],
        t_full,
        [64, C],
        [(lo * 4, hi * 4), (0, C)],
        dtype,
        None,
        None,
        W32,
        32,
        infer=True,
        pre_zero=True,
    )
    mod = _compile(kernel)
    A_np = tvm.testing.generate_random_array(dtype, (64, C))
    dev = tvm.cuda(0)
    A = tvm.runtime.tensor(A_np, dev)
    B = tvm.runtime.tensor(np.zeros((128, W32), np.uint32), dev)
    mod(A, B)
    B_u16 = B.numpy().view(np.uint16).reshape(128, W32 * 2)
    A_u16 = np.asarray(A_np).view(np.uint16)
    written = {32 * q + lo + il for q in range(4) for il in range(hi - lo)}
    for lane in range(128):
        if lane in written:
            q, il = lane // 32, lane % 32 - lo
            np.testing.assert_array_equal(
                B_u16[lane, :C], A_u16[il * 4 + q], err_msg=f"lane {lane}"
            )
        else:
            assert not B_u16[lane].any(), f"lane {lane} must stay zero"


def _make_cp_kernel_cta2(s_full, s_shape, t_full, t_shape, dtype, cfg, W32, n_cols):
    """2-CTA cluster harness: each CTA stages its own A[cbx] slice into smem;
    only the even CTA issues the cp (cta_group=2, commit cta_mask=3); both CTAs
    dump their own tmem into B[cbx*128 + lane]."""
    s_sl = tuple(slice(0, e) for e in s_shape)
    t_sl = tuple(slice(0, e) for e in t_shape)

    @T.prim_func(check_well_formed=False)
    def kernel(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (2, *s_shape), dtype)
        B = T.match_buffer(B_ptr, (256, W32), "uint32")
        T.device_entry()
        warp_id = T.warp_id([4])
        cbx, cby = T.cta_id_in_cluster([2, 1])
        cta_id = T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])
        A_smem = T.alloc_buffer(s_shape, dtype, scope="shared", layout=s_full, align=1024)
        tmem_addr = T.alloc_shared([1], "uint32")
        cp_mbar = T.alloc_shared([1], "uint64")
        if tid_in_wg == 0:
            T.ptx.mbarrier.init(cp_mbar.ptr_to([0]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=n_cols, cta_group=2)
        tmem = T.decl_buffer(
            t_shape, dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=t_full
        )
        T.ptx.fence.mbarrier_init()
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        T.cuda.cluster_sync()
        # Pre-zero both CTAs' tmem: alloc does not clear it, and the
        # 128x256b test asserts the odd CTA stays untouched.
        zero_reg = T.alloc_buffer((W32,), "uint32", scope="local")
        for i in range(W32):
            zero_reg[i] = T.uint32(0)
        for i in range(W32):
            T.ptx.tcgen05.st(tmem_addr[0], zero_reg[i], shape="32x32b", num=1, row=0, col=i)
        T.ptx.tcgen05.wait.st()
        Tx.cta.copy(A_smem[s_sl], A[(cbx, *s_sl)])
        T.cuda.cta_sync()
        T.cuda.cluster_sync()
        if cbx == 0:
            if tid_in_wg == 0:
                Tx.copy_async(tmem[t_sl], A_smem[s_sl], **cfg)
                T.ptx.tcgen05.commit(cp_mbar.ptr_to([0]), cta_group=2, cta_mask=3)
        T.ptx.mbarrier.try_wait(cp_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence.after_thread_sync()
        reg = T.alloc_buffer((W32,), "uint32", scope="local")
        for i in range(W32):
            T.ptx.tcgen05.ld(tmem_addr[0], reg[i], shape="32x32b", num=1, row=0, col=i)
        T.ptx.tcgen05.wait.ld()
        for i in range(W32):
            B[cbx * 128 + tid_in_wg, i] = reg[i]
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=n_cols, cta_group=2)

    return kernel


def _run_cta2(s_full, s_shape, t_full, t_shape, cfg, W32):
    kernel = _make_cp_kernel_cta2(s_full, s_shape, t_full, t_shape, "bfloat16", cfg, W32, 32)
    mod = _compile(kernel)
    A_np = tvm.testing.generate_random_array("bfloat16", (2, *s_shape))
    dev = tvm.cuda(0)
    A = tvm.runtime.tensor(A_np, dev)
    B = tvm.runtime.tensor(np.zeros((256, W32), np.uint32), dev)
    mod(A, B)
    B_u16 = B.numpy().view(np.uint16).reshape(2, 128, W32 * 2)
    A_u16 = np.asarray(A_np).view(np.uint16)
    return A_u16, B_u16


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_cp_cta_group_2_128x256b_pair_collective():
    """Production small_topk q-cp form: 128x256b + cta_group=2. One even-CTA
    issue makes EACH CTA copy from its own smem into its own tmem (descriptor
    smem address resolves per CTA rank). B200-pinned."""
    C = 16
    s_full = mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_32B_ATOM, [128, C])
    t_full = TileLayout(S[(128, C) : (1 @ TLane, 1 @ TCol)])
    A, B = _run_cta2(
        s_full, [128, C], t_full, [128, C], {"shape": "128x256b", "cta_group": 2}, C * 16 // 32
    )
    for cta in range(2):
        for lane in range(128):
            np.testing.assert_array_equal(
                B[cta, lane, :C], A[cta, lane], err_msg=f"cta{cta} lane {lane}"
            )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_cp_cta_group_2_64x128b_warpx2_pair_collective():
    """Production head128 q-cp form: 64x128b.warpx2::02_13 + cta_group=2 is a
    pair-collective — one even-CTA issue makes EACH CTA copy from its own smem
    into its own tmem (descriptor smem address resolves per CTA rank, like the
    cta2 MMA B descriptor), with the warpx2 mirror at lane +64. B200-pinned."""
    C = 8
    s_full = TileLayout(S[(64, C) : (C, 1)])
    t_full = TileLayout(S[(64, C) : (1 @ TLane, 1 @ TCol)] + R[2 : 64 @ TLane])
    A, B = _run_cta2(
        s_full,
        [64, C],
        t_full,
        [64, C],
        {"shape": "64x128b", "multicast": "warpx2::02_13", "cta_group": 2},
        C * 16 // 32,
    )
    for cta in range(2):
        for lane in range(128):
            np.testing.assert_array_equal(
                B[cta, lane, :C], A[cta, lane % 64], err_msg=f"cta{cta} lane {lane}"
            )


def test_cp_4x256b_lane_tiled_layout_f_scatter():
    """4x256b atoms tiled through the taddr lane half-word: 16 cps land 16
    distinct rows on each warp's first 16 lanes (the M=64 Layout-F scatter).
    Smem holds the rows quadrant-interleaved (row i*4+q = logical (q, i))."""
    dtype, bits = "bfloat16", 16
    C = 16  # one 256b atom row = 16 bf16 elements
    W32 = C * bits // 32
    s_full = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_32B_ATOM, [64, C])
    t_full = TileLayout(S[(16, 4, C) : (1 @ TLane, 32 @ TLane, 1 @ TCol)])
    kernel = _make_cp_kernel(
        s_full,
        [64, C],
        [(0, 64), (0, C)],
        t_full,
        [64, C],
        [(0, 64), (0, C)],
        dtype,
        None,
        None,
        W32,
        32,
        infer=True,
    )
    mod = _compile(kernel)
    src = mod.mod.imports[0].inspect_source()
    assert "tcgen05.cp.cta_group::1.4x256b" in src, f"expected 4x256b atoms; src=\n{src}"

    A_np = tvm.testing.generate_random_array(dtype, (64, C))
    dev = tvm.cuda(0)
    A = tvm.runtime.tensor(A_np, dev)
    B = tvm.runtime.tensor(np.zeros((128, W32), np.uint32), dev)
    mod(A, B)
    B_out = B.numpy()
    A_bits = np.asarray(A_np).view(np.uint16)
    for i in range(16):
        for q in range(4):
            lane = 32 * q + i
            exp = A_bits[i * 4 + q].copy().view(np.uint32)
            np.testing.assert_array_equal(B_out[lane, :W32], exp, err_msg=f"lane {lane}")


def test_cp_default_32x128b_instruction_sequence_unchanged():
    """Back-compat pin: a config-less smem->tmem copy_async must emit the
    exact legacy 32x128b.warpx4 sequence (hardcoded from the pre-generalization
    planner output): one descriptor encode with (ldo=0, sdo=8, swizzle=0) and
    four cps at tmem cols 0/4/8/12 reading smem bytes 0/512/1024/1536."""
    s_full = TileLayout(S[(4, 32, 16) : (512, 16, 1)])
    t_full = TileLayout(S[(4, 32, 16) : (16 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])

    @T.prim_func(check_well_formed=False)
    def kernel(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (4, 32, 16), "uint8")
        T.device_entry()
        warp_id = T.warp_id([4])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])
        T.lane_id([32])
        A_smem = T.alloc_buffer((4, 32, 16), "uint8", scope="shared", layout=s_full, align=1024)
        tmem_addr = T.alloc_shared([1], "uint32")
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=32, cta_group=1)
            T.cuda.cta_sync()
            Tx.cta.copy(A_smem[:, :, :], A[:, :, :])
            T.cuda.cta_sync()
            tmem = T.decl_buffer(
                (4, 32, 16), "uint8", scope="tmem", allocated_addr=tmem_addr[0], layout=t_full
            )
            if tid_in_wg == 0:
                # NOTE: no shape/multicast config — the legacy default route.
                Tx.copy_async(tmem[:, :, :], A_smem[:, :, :], cta_group=1)
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=32, cta_group=1)

    mod = _compile(kernel)
    src = mod.mod.imports[0].inspect_source()

    encode_lines = [
        line
        for line in src.splitlines()
        if "tvm_builtin_ptx_tcgen05_encode_matrix_descriptor(" in line
        and "__forceinline__" not in line
    ]
    assert len(encode_lines) == 1, encode_lines
    # (ldo, sdo, swizzle) = (0, 8, 0), template encoded at smem address 0.
    assert "reinterpret_cast<void*>((uint64_t)0), 0, 8, 0);" in encode_lines[0], encode_lines[0]

    cp_lines = [
        line
        for line in src.splitlines()
        if "ptx_tcgen05_cp_cta_group_1_shape_32x128b_multicast_warpx4_decompress_(" in line
        and "__forceinline__" not in line
    ]
    assert len(cp_lines) == 4, f"expected 4 cp calls, got {len(cp_lines)}"
    for i, (t_off, s_off) in enumerate([(0, 0), (4, 512), (8, 1024), (12, 1536)]):
        t_tok = "[0], 0, 0," if t_off == 0 else f"[0] + (uint){t_off}), 0, 0,"
        assert t_tok in cp_lines[i], f"cp[{i}] tmem col: want {t_tok!r} in {cp_lines[i]!r}"
        s_tok = f"+ {s_off}))"
        assert s_tok in cp_lines[i], f"cp[{i}] smem byte off: want {s_tok!r} in {cp_lines[i]!r}"
        assert "cp_desc_ptr[0] &" in cp_lines[i] and "16383" in cp_lines[i], cp_lines[i]


# Negative tests: readable ValueErrors from the planner


def _assert_compile_raises(kernel, fragment):
    target = tvm.target.Target("cuda")
    with target:
        with pytest.raises(Exception, match=fragment):
            tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")


def test_cp_rejects_wrong_replica_for_multicast():
    """A 02_13-declared TMEM layout used with 01_23 (and vice versa) must
    fail the replica router with a readable message."""
    bits = 16
    C = 128 // bits
    t_full, t_full_shape = _tmem_layout_for("64x128b", "warpx2::02_13", C)
    s_full = mma_shared_layout("bfloat16", SwizzleMode(3), [64, 64])
    kernel = _make_cp_kernel(
        s_full,
        [64, 64],
        [(0, 64), (0, C)],
        t_full,
        t_full_shape,
        [(0, 64), (0, C)],
        "bfloat16",
        "64x128b",
        "warpx2::01_23",  # wrong variant for the declared replica structure
        C * bits // 32,
        32,
    )
    _assert_compile_raises(kernel, "replica mismatch")


def test_cp_rejects_illegal_shape_multicast_combo():
    kernel, _ = _build_case("128x256b", None, 3, "bfloat16", 1, extra_cfg={"multicast": "warpx4"})
    _assert_compile_raises(kernel, "illegal multicast")


def test_cp_rejects_missing_multicast_for_64x128b():
    bits = 16
    C = 128 // bits
    t_full, t_full_shape = _tmem_layout_for("64x128b", "warpx2::02_13", C)
    s_full = mma_shared_layout("bfloat16", SwizzleMode(3), [64, 64])
    kernel = _make_cp_kernel(
        s_full,
        [64, 64],
        [(0, 64), (0, C)],
        t_full,
        t_full_shape,
        [(0, 64), (0, C)],
        "bfloat16",
        "64x128b",
        None,  # 64x128b has two legal multicasts: must be explicit
        C * bits // 32,
        32,
    )
    _assert_compile_raises(kernel, "requires an explicit multicast")


def test_cp_rejects_non_128b_tcol_offset():
    """A whole 32-bit TMEM cell is still below tcgen05.cp's 128-bit alignment."""
    bits = 16
    epa = 128 // bits
    C = 2 * epa
    t_C = C + 8  # headroom so the misaligned region stays in range
    t_full, t_full_shape = _tmem_layout_for("32x128b", "warpx4", t_C)
    s_full = mma_shared_layout("bfloat16", SwizzleMode(3), [32, 64])
    kernel = _make_cp_kernel(
        s_full,
        [32, 64],
        [(0, 32), (0, C)],
        t_full,
        t_full_shape,
        [(0, 32), (2, C + 2)],  # TCol offset 2 bf16 elems = one 32-bit cell
        "bfloat16",
        "32x128b",
        "warpx4",
        t_C * bits // 32,
        32,
    )
    _assert_compile_raises(kernel, "not provably 128b-aligned")


def test_cp_rejects_unknown_shape():
    # Valid 32x128b setup, but the shape string is overridden with a bogus one.
    kernel, _ = _build_case("32x128b", None, 3, "bfloat16", 1, extra_cfg={"shape": "96x128b"})
    _assert_compile_raises(kernel, "unknown tcgen05.cp shape")


def test_cp_rejects_nonzero_tmem_lane_offset():
    """A lane offset whose footprint overflows lane space must be rejected:
    warpx2's +64 mirror spans all 128 lanes, so any nonzero offset would
    write past lane 127."""
    bits = 16
    C = 128 // bits
    t_full = TileLayout(S[(128, C) : (1 @ TLane, 1 @ TCol)] + R[2 : 64 @ TLane])
    s_full = mma_shared_layout("bfloat16", SwizzleMode(3), [64, 64])
    kernel = _make_cp_kernel(
        s_full,
        [64, 64],
        [(0, 64), (0, C)],
        t_full,
        [128, C],
        [(64, 128), (0, C)],  # lane offset 64
        "bfloat16",
        "64x128b",
        "warpx2::02_13",
        C * bits // 32,
        32,
    )
    _assert_compile_raises(kernel, "overflows the 128-lane space")


def test_cp_rejects_decompress_on_generic_path():
    """decompress is a real PTX feature the planner cannot lower; it must be
    rejected loudly instead of silently copying without decompression."""
    kernel, _ = _build_case(
        "128x256b", None, 3, "bfloat16", 1, extra_cfg={"decompress": "b8x16.b6x16_p32"}
    )
    _assert_compile_raises(kernel, "does not support decompress")


def test_cp_rejects_non_16b_aligned_row_group_stride():
    """The descriptor SBO field is encoded in 16B units: an smem layout whose
    8-row-group stride is not 16B-divisible must be rejected."""
    bits = 16
    C = 128 // bits  # 8 bf16 = one 16B unit per row
    # Non-swizzled (32, 8) tile: 8-row blocks are atom_K=8 elems (16B) apart,
    # but the group stride is 100 elems = 200B, not a multiple of 16B.
    s_full = TileLayout(S[(4, 8, C) : (100, C, 1)])
    t_full = TileLayout(S[(32, C) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])
    kernel = _make_cp_kernel(
        s_full,
        [32, C],
        [(0, 32), (0, C)],
        t_full,
        [32, C],
        [(0, 32), (0, C)],
        "bfloat16",
        "32x128b",
        "warpx4",
        C * bits // 32,
        32,
    )
    _assert_compile_raises(kernel, "not 16B-aligned")


def test_cp_rejects_non_canonical_swizzle_family():
    """A swizzled layout outside the canonical mma atom family (wrong
    per_element for the dtype) must be rejected."""
    bits = 16
    C = 128 // bits
    # per_element=2 is the f32 family; with a bf16 source it disagrees with the
    # descriptor permutation. Linear part is valid, so only family check rejects.
    s_full = ComposeLayout(2, 3, 3, TileLayout(S[(16, 8, 64) : (512, 64, 1)]))
    t_full = TileLayout(S[(128, C) : (1 @ TLane, 1 @ TCol)])
    kernel = _make_cp_kernel(
        s_full,
        [128, 64],
        [(0, 128), (0, C)],
        t_full,
        [128, C],
        [(0, 128), (0, C)],
        "bfloat16",
        "128x128b",
        None,
        C * bits // 32,
        32,
    )
    _assert_compile_raises(kernel, "swizzle family mismatch")


def test_cp_rejects_flipped_swizzle_inner():
    """A canonical-family swizzled layout with ``swizzle_inner=False`` must be
    rejected: the descriptor's hardware walk implements the
    ``swizzle_inner=True`` permutation ``x ^ ((x & outer_mask) >> atom_len)``
    (pinned bit-exactly on B200 by the round-trip tests above), while
    ``swizzle_inner=False`` selects the mirrored
    ``x ^ ((x & inner_mask) << atom_len)`` — a different permutation on any
    chunk whose swizzle bits are nonzero."""
    bits = 16
    C = 128 // bits
    # Same linear tiling and correct (per_element=3, atom_len=3) bf16 family;
    # only the permutation direction is flipped, so only swizzle_inner rejects.
    s_full = ComposeLayout(3, 3, 3, TileLayout(S[(16, 8, 64) : (512, 64, 1)]), swizzle_inner=False)
    t_full = TileLayout(S[(128, C) : (1 @ TLane, 1 @ TCol)])
    kernel = _make_cp_kernel(
        s_full,
        [128, 64],
        [(0, 128), (0, C)],
        t_full,
        [128, C],
        [(0, 128), (0, C)],
        "bfloat16",
        "128x128b",
        None,
        C * bits // 32,
        32,
    )
    _assert_compile_raises(kernel, "swizzle_inner")


# Legacy 32x128b.warpx4 tests (predate the generic planner)

# warpx4 requires the t buffer to declare the broadcast explicitly:
# R[4 : 32@TLane] — t.shape[lane] = 32 with replica 4 → 128 physical lanes.
T_LAY_BASIC = TileLayout(S[(32, 16) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])


def _make_2d_kernel(
    s_full,
    t_full,
    s_full_shape,
    t_full_shape,
    s_r0,
    s_r1,
    s_c0,
    s_c1,
    t_r0,
    t_r1,
    t_c0,
    t_c1,
    dtype,
    cta_group=1,
):
    """2D variant: SMEM/TMEM are both 2D; copy a rectangular sub-region."""
    n_tmem_cols_total = max(32, t_full_shape[-1])
    OUT_LANES = 32
    OUT_BYTES = 16

    @T.prim_func(check_well_formed=False)
    def kernel(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, s_full_shape, dtype)
        B = T.match_buffer(B_ptr, (OUT_LANES, OUT_BYTES), dtype)
        T.device_entry()
        warp_id = T.warp_id([4])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])
        lane_id = T.lane_id([32])
        A_smem = T.alloc_buffer(s_full_shape, dtype, scope="shared", layout=s_full, align=1024)
        tmem_addr = T.alloc_shared([1], "uint32")
        cp_mbar = T.alloc_shared([1], "uint64")
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(
                    T.address_of(tmem_addr),
                    n_cols=n_tmem_cols_total,
                    cta_group=cta_group,
                )
            if tid_in_wg == 0:
                T.ptx.mbarrier.init(cp_mbar.ptr_to([0]), 1)
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.cta_sync()
            Tx.cta.copy(A_smem[:, :], A[:, :])
            T.cuda.cta_sync()
            tmem = T.decl_buffer(
                t_full_shape,
                dtype,
                scope="tmem",
                allocated_addr=tmem_addr[0],
                layout=t_full,
            )
            if tid_in_wg == 0:
                Tx.copy_async(
                    tmem[t_r0:t_r1, t_c0:t_c1],
                    A_smem[s_r0:s_r1, s_c0:s_c1],
                    cta_group=cta_group,
                )
                T.ptx.tcgen05.commit(cp_mbar.ptr_to([0]), cta_group=cta_group)
            T.ptx.mbarrier.try_wait(cp_mbar.ptr_to([0]), 0)
            T.cuda.cta_sync()
            T.ptx.tcgen05.fence.after_thread_sync()
            if warp_id == 0:
                reg = T.alloc_buffer((4,), "uint32", scope="local")
                for i in range(4):
                    T.ptx.tcgen05.ld(
                        tmem.allocated_addr[0],
                        reg[i],
                        shape="32x32b",
                        num=1,
                        row=0,
                        col=i,
                    )
                T.ptx.tcgen05.wait.ld()
                B_bytes = reg.view(dtype)
                for i in range(OUT_BYTES):
                    B[lane_id, i] = B_bytes[i]
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=n_tmem_cols_total, cta_group=cta_group)

    return kernel


def _make_3d_4tile_kernel(s_full, t_full, s_full_shape, t_full_shape, dtype, cta_group=1):
    """3D variant: 4 stacked tiles (NVFP4-style multi-cp test)."""
    n_tmem_cols_total = max(32, t_full_shape[-1])

    @T.prim_func(check_well_formed=False)
    def kernel(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, s_full_shape, dtype)
        B = T.match_buffer(B_ptr, (32, 16), dtype)
        T.device_entry()
        warp_id = T.warp_id([4])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])
        lane_id = T.lane_id([32])
        A_smem = T.alloc_buffer(s_full_shape, dtype, scope="shared", layout=s_full, align=1024)
        tmem_addr = T.alloc_shared([1], "uint32")
        cp_mbar = T.alloc_shared([1], "uint64")
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(
                    T.address_of(tmem_addr),
                    n_cols=n_tmem_cols_total,
                    cta_group=cta_group,
                )
            if tid_in_wg == 0:
                T.ptx.mbarrier.init(cp_mbar.ptr_to([0]), 1)
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.cta_sync()
            Tx.cta.copy(A_smem[:, :, :], A[:, :, :])
            T.cuda.cta_sync()
            tmem = T.decl_buffer(
                t_full_shape,
                dtype,
                scope="tmem",
                allocated_addr=tmem_addr[0],
                layout=t_full,
            )
            if tid_in_wg == 0:
                Tx.copy_async(
                    tmem[:, :, :],
                    A_smem[:, :, :],
                    cta_group=cta_group,
                )
                T.ptx.tcgen05.commit(cp_mbar.ptr_to([0]), cta_group=cta_group)
            T.ptx.mbarrier.try_wait(cp_mbar.ptr_to([0]), 0)
            T.cuda.cta_sync()
            T.ptx.tcgen05.fence.after_thread_sync()
            if warp_id == 0:
                reg = T.alloc_buffer((4,), "uint32", scope="local")
                for i in range(4):
                    T.ptx.tcgen05.ld(
                        tmem.allocated_addr[0],
                        reg[i],
                        shape="32x32b",
                        num=1,
                        row=0,
                        col=i,
                    )
                T.ptx.tcgen05.wait.ld()
                B_bytes = reg.view(dtype)
                for i in range(16):
                    B[lane_id, i] = B_bytes[i]
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=n_tmem_cols_total, cta_group=cta_group)

    return kernel


def _run_2d(s_full, t_full, s_full_shape, s_region, dtype, A_init, expected):
    s_r0, s_r1 = s_region[0]
    s_c0, s_c1 = s_region[1]
    kernel = _make_2d_kernel(
        s_full, t_full, s_full_shape, [32, 16], s_r0, s_r1, s_c0, s_c1, 0, 32, 0, 16, dtype
    )
    return _execute(kernel, A_init, expected)


def _run_3d_4tile(s_full, t_full, s_full_shape, dtype, A_init, expected):
    kernel = _make_3d_4tile_kernel(s_full, t_full, s_full_shape, s_full_shape, dtype)
    return _execute(kernel, A_init, expected)


def _execute(kernel, A_init, expected):
    mod = _compile(kernel)
    dev = tvm.cuda(0)
    A = tvm.runtime.tensor(A_init, dev)
    B_np = np.zeros((32, 16), dtype=A_init.dtype)
    B = tvm.runtime.tensor(B_np, dev)
    mod(A, B)
    B_out = B.numpy()
    assert np.array_equal(B_out, expected), (
        f"mismatch:\nlane 0 expected={expected[0].tolist()}\n        got     ={B_out[0].tolist()}"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize(
    "name,s_full,s_full_shape,s_region",
    [
        ("sw0_plain_atom_aligned", TileLayout(S[(32, 16) : (16, 1)]), [32, 16], [(0, 32), (0, 16)]),
        (
            "sw1_32B_atom",
            mma_shared_layout("uint8", SwizzleMode.SWIZZLE_32B_ATOM, [32, 32]),
            [32, 32],
            [(0, 32), (0, 16)],
        ),
        (
            "sw2_64B_atom",
            mma_shared_layout("uint8", SwizzleMode.SWIZZLE_64B_ATOM, [32, 64]),
            [32, 64],
            [(0, 32), (0, 16)],
        ),
        (
            "sw3_128B_atom",
            mma_shared_layout("uint8", SwizzleMode.SWIZZLE_128B_ATOM, [32, 128]),
            [32, 128],
            [(0, 32), (0, 16)],
        ),
        (
            "sw3_64x128_corner",
            mma_shared_layout("uint8", SwizzleMode.SWIZZLE_128B_ATOM, [64, 128]),
            [64, 128],
            [(0, 32), (0, 16)],
        ),
        (
            "sw3_64x128_atom_row_8",
            mma_shared_layout("uint8", SwizzleMode.SWIZZLE_128B_ATOM, [64, 128]),
            [64, 128],
            [(8, 40), (0, 16)],
        ),
        (
            "sw2_32x256_col_64",
            mma_shared_layout("uint8", SwizzleMode.SWIZZLE_64B_ATOM, [32, 256]),
            [32, 256],
            [(0, 32), (64, 80)],
        ),
        (
            "sw0_M_atom_major_4_0",
            TileLayout(S[(8, 8, 2, 16) : (128, 16, 1024, 1)]),
            [64, 32],
            [(4, 36), (0, 16)],
        ),
    ],
)
def test_single_cp(name, s_full, s_full_shape, s_region):
    A_np = np.arange(int(np.prod(s_full_shape)), dtype=np.uint8).reshape(s_full_shape)
    r0, r1 = s_region[0]
    c0, c1 = s_region[1]
    expected = A_np[r0:r1, c0:c1]
    _run_2d(s_full, T_LAY_BASIC, s_full_shape, s_region, "uint8", A_np, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_multi_cp_sw0_4tiles():
    s_full = TileLayout(S[(4, 32, 16) : (512, 16, 1)])
    t_full = TileLayout(S[(4, 32, 16) : (16 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])
    A_np = (np.arange(4 * 32 * 16, dtype=np.int32) & 0xFF).astype(np.uint8).reshape(4, 32, 16)
    expected = A_np[0]
    _run_3d_4tile(s_full, t_full, [4, 32, 16], "uint8", A_np, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_align_middle_2_to_1_nvfp4_sfb():
    """SFB-style nvfp4 case: TMEM mid canonicalizes to single iter
    (16@TCol + 4@TCol merge), but SMEM mid stays as 2 iters
    (stride 512 + stride 2048 — outer/inner reversed so canon can't merge).
    Exercises ``_align_middles`` union-cut algorithm.

    Layout shapes mirror SFB nvfp4 with PIPE=1, SFB_n_chunks=2,
    MMA_K_BLOCKS=4, sf_mma_k=4.
    """
    # SMEM: (2, 4, 32, 4, 4) extents, strides (2048, 4, 16, 512, 1).
    # Mid post-canon = [(4, 512), (2, 2048)] — non-mergeable in this order.
    s_full = TileLayout(S[(2, 4, 32, 4, 4) : (2048, 4, 16, 512, 1)])
    # TMEM: SFB-style 5-axis layout. K_outer (4, 4@TCol) and N_chunk
    # (2, 16@TCol) merge into single mid iter (8, 4@TCol).
    t_full = TileLayout(
        S[(2, 4, 32, 4, 4) : (16 @ TCol, 4 @ TCol, 1 @ TLane, 32 @ TCol, 1 @ TCol)]
        + R[4 : 32 @ TLane]
    )
    s_full_shape = [256, 16]
    t_full_shape = [256, 16]
    n_tmem_cols_total = max(32, 32)  # SFB occupies 32 cols total (8*4 elements / 4 epc)

    @T.prim_func(check_well_formed=False)
    def kernel(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, s_full_shape, "uint8")
        B = T.match_buffer(B_ptr, (32, 16), "uint8")
        T.device_entry()
        warp_id = T.warp_id([4])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])
        lane_id = T.lane_id([32])
        A_smem = T.alloc_buffer(s_full_shape, "uint8", scope="shared", layout=s_full, align=1024)
        tmem_addr = T.alloc_shared([1], "uint32")
        cp_mbar = T.alloc_shared([1], "uint64")
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=n_tmem_cols_total, cta_group=1)
            if tid_in_wg == 0:
                T.ptx.mbarrier.init(cp_mbar.ptr_to([0]), 1)
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.cta_sync()
            Tx.cta.copy(A_smem[:, :], A[:, :])
            T.cuda.cta_sync()
            tmem = T.decl_buffer(
                t_full_shape,
                "uint8",
                scope="tmem",
                allocated_addr=tmem_addr[0],
                layout=t_full,
            )
            if tid_in_wg == 0:
                Tx.copy_async(tmem[:, :], A_smem[:, :], cta_group=1)
                T.ptx.tcgen05.commit(cp_mbar.ptr_to([0]), cta_group=1)
            T.ptx.mbarrier.try_wait(cp_mbar.ptr_to([0]), 0)
            T.cuda.cta_sync()
            T.ptx.tcgen05.fence.after_thread_sync()
            if warp_id == 0:
                reg = T.alloc_buffer((4,), "uint32", scope="local")
                for i in range(4):
                    T.ptx.tcgen05.ld(
                        tmem.allocated_addr[0],
                        reg[i],
                        shape="32x32b",
                        num=1,
                        row=0,
                        col=i,
                    )
                T.ptx.tcgen05.wait.ld()
                B_bytes = reg.view("uint8")
                for i in range(16):
                    B[lane_id, i] = B_bytes[i]
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=n_tmem_cols_total, cta_group=1)

    A_np = (np.arange(256 * 16, dtype=np.int32) & 0xFF).astype(np.uint8).reshape(256, 16)

    # Invert TMEM layout to map physical (TLane=L, TCol=p) → logical index, then
    # expected[L, b] = A[m, k]. i2 = L; TCol p: i1 = p // 4, i4 = p % 4 (i0=i3=0).
    expected = np.zeros((32, 16), dtype=np.uint8)
    for L in range(32):
        for p in range(16):
            i0 = 0
            i3 = 0
            i1 = p // 4
            i4 = p % 4
            i2 = L
            logical = i0 * (4 * 32 * 4 * 4) + i1 * (32 * 4 * 4) + i2 * (4 * 4) + i3 * 4 + i4
            m, k = divmod(logical, 16)
            expected[L, p] = A_np[m, k]

    _execute(kernel, A_np, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(
            (
                "sw3_mid_atom_row",
                mma_shared_layout("uint8", SwizzleMode.SWIZZLE_128B_ATOM, [64, 128]),
                [64, 128],
                [(4, 36), (0, 16)],
            ),
            id="sw3_mid_atom_row",
        ),
        pytest.param(
            (
                "sw2_mid_atom_col",
                mma_shared_layout("uint8", SwizzleMode.SWIZZLE_64B_ATOM, [32, 128]),
                [32, 128],
                [(0, 32), (32, 48)],
            ),
            id="sw2_mid_atom_col",
        ),
        pytest.param(
            ("sw0_row_stride_64", TileLayout(S[(64, 64) : (64, 1)]), [64, 64], [(4, 36), (0, 16)]),
            id="sw0_row_stride_64",
        ),
    ],
)
def test_dispatch_rejects_bad_inputs(bad):
    """Configurations where cp 32x128b cannot read the user's intended sub-tile.
    Compilation should fail with a clear ValueError from the dispatch."""
    name, s_full, s_full_shape, s_region = bad
    s_r0, s_r1 = s_region[0]
    s_c0, s_c1 = s_region[1]
    kernel = _make_2d_kernel(
        s_full, T_LAY_BASIC, s_full_shape, [32, 16], s_r0, s_r1, s_c0, s_c1, 0, 32, 0, 16, "uint8"
    )
    with pytest.raises(Exception):
        _compile(kernel)


def test_multi_cp_encodes_descriptor_once_and_patches_addr():
    """Compile-only regression for the shared-descriptor cp path.

    A multi-tile smem->tmem copy encodes ONE SMEM matrix descriptor template at
    SMEM base 0 (so the cache key no longer depends on the buffer identity) and
    patches its 14-bit address field per cp via ``cvta(addr) >> 4 & 0x3FFF``,
    instead of re-encoding a descriptor per tile. Verifies the 4-tile copy emits
    a single ``encode_matrix_descriptor`` reused across four
    ``tcgen05.cp.32x128b.warpx4`` issues, each with the address-field patch.
    """
    s_full = TileLayout(S[(4, 32, 16) : (512, 16, 1)])
    t_full = TileLayout(S[(4, 32, 16) : (16 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])
    kernel = _make_3d_4tile_kernel(s_full, t_full, [4, 32, 16], [4, 32, 16], "uint8")

    mod = _compile(kernel)
    src = mod.mod.imports[0].inspect_source()

    assert "tcgen05.cp.cta_group::1.32x128b.warpx4" in src, f"cp not emitted; src=\n{src}"
    # Descriptor encoded once (single matrix-descriptor encode call), then
    # reused with a per-cp 14-bit SMEM address patch (0x3FFF == 16383 mask).
    assert "16383" in src, "expected 14-bit SMEM address-field patch (0x3FFF mask)"
    assert src.count("cp_desc_ptr[0] &") == 4, (
        f"expected 4 address-patched cp's reusing one cp_desc; got "
        f"{src.count('cp_desc_ptr[0] &')}\nsrc=\n{src}"
    )


if __name__ == "__main__":
    tvm.testing.main()
