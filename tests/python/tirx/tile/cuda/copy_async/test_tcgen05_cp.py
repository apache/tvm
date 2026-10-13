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
"""Original tcgen05.cp layouts, regions, multicast and CTA-pair readbacks.

Instruction shape and descriptor fields are explicitly selected by each case.
The host references reconstruct the original physical TMEM cells and replicas,
independently of the caller's zero-copy view and native cp lowering.
"""

import itertools

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.backend.cuda.axis import TCol, TLane
from tvm.backend.cuda.layout import SwizzleMode, mma_shared_layout
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.layout import ComposeLayout, R, S, TileLayout

# Multicast replica lane offsets: (extent, stride) pairs on TLane. These are
# the warp-slab offsets receiving copies of the data (see module docstring).
_REPLICA_PATTERN = {
    None: [],
    "warpx4": [(4, 32)],
    "warpx2::02_13": [(2, 64)],
    "warpx2::01_23": [(2, 32)],
}

# The original shape/multicast matrix, selected explicitly in the opcode.
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
    """Destination TMEM layout + logical tensor shape for one cp footprint.

    The layout encodes the hardware row→lane mapping of ONE copy; the
    multicast replicas are declared via ``R[...]`` so the pattern can verify
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
        # Lane split lives in the iters; tensor stays the natural (64, C) tile.
        return (
            TileLayout(S[(2, 32, C) : (64 @ TLane, 1 @ TLane, 1 @ TCol)] + R[2 : 32 @ TLane]),
            [64, C],
        )
    raise ValueError(shape)


def _emit_cp(dst, src, t_region, s_region, shape, opcode, descriptor):
    """Expose fixed b32 atom regions selected by the caller's shape and bounds.

    These loops partition the original logical region without searching either
    operand's physical layout. Each region retains its original row/column start.
    """
    atom_rows, atom_bits = _shape_dims(shape)
    bits = tvm.DataType(src.dtype).bits
    words = atom_bits // 32
    s_row, s_col = s_region
    t_row, t_col = t_region
    source, destination = src.view("uint32"), dst.view("uint32")
    for row in range(0, t_row[1] - t_row[0], atom_rows):
        for col in range(0, (t_col[1] - t_col[0]) * bits // 32, words):
            sc, tc = s_col[0] * bits // 32 + col, t_col[0] * bits // 32 + col
            T.evaluate(
                T.cuda.tile.apply(
                    T.ptx[opcode],
                    destination[t_row[0] + row : t_row[0] + row + atom_rows, tc : tc + words],
                    source[s_row[0] + row : s_row[0] + row + atom_rows, sc : sc + words],
                    ldo_bytes=descriptor[0],
                    sdo_bytes=descriptor[1],
                    swizzle=descriptor[2],
                )
            )


def _cp_opcode(shape, multicast=None, cta_group=1, decompress=None):
    opcode = f"tcgen05.cp.cta_group::{cta_group}.{shape}"
    if multicast:
        opcode += "." + multicast
    if decompress:
        opcode += "." + decompress
    return opcode


def _cp_call_lines(ex, shape, multicast=None, cta_group=1):
    opcode = _cp_opcode(shape, multicast, cta_group)
    source = ex.mod.imports[0].inspect_source()
    name = "tvm_builtin_ptx_tcgen05_cp_cp_" + opcode.removeprefix("tcgen05.cp.").replace(
        "::", "__"
    ).replace(".", "_")
    return [line for line in source.splitlines() if line.strip().startswith(name + "(")]


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
    descriptor=(0, 1024, 3),
    pre_zero=False,
):
    s_sl = tuple(slice(a, b) for a, b in s_region)
    t_sl = tuple(slice(a, b) for a, b in t_region)
    s_full_sl = tuple(slice(0, e) for e in s_full_shape)
    cfg = {"shape": shape, "multicast": multicast}
    if extra_cfg:
        cfg.update(extra_cfg)
    opcode = _cp_opcode(**cfg)

    @T.function(check_well_formed=False)
    def kernel(A: T.Tensor(s_full_shape, dtype), B: T.Tensor((128, W32), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=4 * 32))
        warp_id = T.cuda.warp_id()
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()
        _lane = T.cuda.lane_id()
        A_smem = T.alloc_tensor(
            ty_args=[T.Tensor(s_full_shape, dtype, scope="shared", layout=s_full, align=1024)]
        )
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        cp_mbar = T.alloc_shared([1], "uint64", layout="default")
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    T.address_of(tmem_addr), T.uint32(n_tmem_cols)
                )
            if tid_in_wg == 0:
                T.ptx.mbarrier.init.shared.b64(cp_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.fence.proxy.async_.shared__cta()
            T.cuda.cta_sync()
            stage_src = A.sub[s_full_sl].view(-1)
            stage_dst = A_smem.sub[s_full_sl].view(-1)
            for step in range(T.ceildiv(stage_dst.shape[0], 128)):
                index = step * 128 + tid_in_wg
                if index < stage_dst.shape[0]:
                    stage_dst[index] = stage_src[index]
            T.cuda.cta_sync()
            tmem = T.cuda.decl_tmem(
                tmem_addr[0], ty_args=[T.Tensor(t_full_shape, dtype, scope="tmem", layout=t_full)]
            )
            if T.constexpr(pre_zero):
                zero_reg = T.alloc_tensor(
                    ty_args=[T.Tensor((W32,), "uint32", scope="local", layout="default")]
                )
                for i in range(W32):
                    zero_reg[i] = T.uint32(0)
                for i in range(W32):
                    T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](
                        T.cuda.get_tmem_addr(tmem_addr[0], 0, i), zero_reg[i]
                    )
                T.ptx.tcgen05.wait__st.sync.aligned()
                T.cuda.cta_sync()
                T.ptx.tcgen05.fence__after_thread_sync()
            if tid_in_wg == 0:
                _emit_cp(tmem, A_smem, t_region, s_region, shape, opcode, descriptor)
                T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                    cp_mbar.ptr_to([0])
                )
            T.cuda.mbarrier_wait(cp_mbar.ptr_to([0]), 0)
            T.cuda.cta_sync()
            T.ptx.tcgen05.fence__after_thread_sync()
            # Each of the 4 warps reads its own 32-lane slab (taddr lane 0 is
            # warp-slab-relative for .32x32b), covering all 128 TMEM lanes.
            reg = T.alloc_tensor(
                ty_args=[T.Tensor((W32,), "uint32", scope="local", layout="default")]
            )
            for i in range(W32):
                T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](
                    reg[i], T.cuda.get_tmem_addr(tmem_addr[0], 0, i)
                )
            T.ptx.tcgen05.wait__ld.sync.aligned()
            for i in range(W32):
                B[tid_in_wg, i] = reg[i]
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    tmem_addr[0], T.uint32(n_tmem_cols)
                )

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
    shape,
    multicast,
    sw,
    dtype,
    n_mid,
    s_row_off=0,
    t_col_off_e=0,
    extra_cfg=None,
    instruction_shape=None,
):
    """Assemble (kernel, host-check closure) for one cp configuration."""
    bits = tvm.runtime.DataType(dtype).bits
    rows, atom_bits = _shape_dims(shape)
    epa = atom_bits // bits  # elems per lane per cp instruction
    C = epa * n_mid  # copied cols (elems)
    t_C = C + t_col_off_e  # tmem tensor cols (elems)
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
        instruction_shape or shape,
        multicast,
        W32,
        n_tmem_cols,
        extra_cfg=extra_cfg,
        descriptor=(
            128 if sw == 0 and atom_bits == 256 else 0,
            256 if sw == 0 and atom_bits == 256 else 128 << sw,
            sw,
        ),
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
        return tvm.tirx.compile(tvm.IRModule({"main": kernel}), target=target, pipeline="tirx")


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
    """Bit-exact shared-to-TMEM transfer using the requested instruction."""
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
    helper_refs = src.count("tvm_builtin_ptx_tcgen05_cp_cp_cta_group__1_4x256b")
    assert helper_refs - 1 == 2, f"expected 2 cp calls, got {helper_refs - 1}; src=\n{src}"


def test_cp_explicit_shape_and_descriptor():
    kernel, _ = _build_case("128x256b", None, 3, "bfloat16", 4)
    ex = _compile(kernel)
    assert len(_cp_call_lines(ex, "128x256b")) == 4
    source = ex.mod.imports[0].inspect_source()
    assert "16383" in source  # descriptor's 14-bit start-address field


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("shape,multicast", [sm for sm in _SHAPE_MULTICAST if sm[0] != "128x128b"])
def test_cp_requested_shape_is_preserved(shape, multicast):
    # These layouts used to select a shape automatically. The caller now
    # chooses it, and all original layouts still produce the same output.
    kernel, check = _build_case(shape, multicast, 2, "bfloat16", 2)
    ex = _compile(kernel)
    assert len(_cp_call_lines(ex, shape, multicast)) == 2
    check(ex)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_cp_128x128b_layout_accepts_explicit_256b_atom():
    kernel, check = _build_case("128x128b", None, 2, "bfloat16", 2, instruction_shape="128x256b")
    ex = _compile(kernel)
    assert len(_cp_call_lines(ex, "128x256b")) == 1
    check(ex)


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
        "4x256b",
        None,
        W32,
        32,
        descriptor=(0, 256, 1),
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

    shape = cfg["shape"]
    descriptor = cfg["descriptor"]
    opcode = _cp_opcode(shape, cfg.get("multicast"), cfg["cta_group"], cfg.get("decompress"))
    s_region = [(0, d) for d in s_shape]
    t_region = [(0, d) for d in t_shape]

    @T.function(check_well_formed=False)
    def kernel(A: T.Tensor((2, *s_shape), dtype), B: T.Tensor((256, W32), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(2,), block=4 * 32, cluster=(2, 1)))
        warp_id = T.cuda.warp_id()
        cbx, cby = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
        cta_id = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()
        A_smem = T.alloc_tensor(
            ty_args=[T.Tensor(s_shape, dtype, scope="shared", layout=s_full, align=1024)]
        )
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        cp_mbar = T.alloc_shared([1], "uint64", layout="default")
        if tid_in_wg == 0:
            T.ptx.mbarrier.init.shared.b64(cp_mbar.ptr_to([0]), T.uint32(1))
        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(n_cols)
            )
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        T.cuda.cluster_sync()
        tmem = T.cuda.decl_tmem(
            tmem_addr[0], ty_args=[T.Tensor(t_shape, dtype, scope="tmem", layout=t_full)]
        )
        # Pre-zero both CTAs: decompression checks include untouched lanes.
        zero_reg = T.alloc_tensor(
            ty_args=[T.Tensor((W32,), "uint32", scope="local", layout="default")]
        )
        for i in range(W32):
            zero_reg[i] = T.uint32(0)
        for i in range(W32):
            T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](
                T.cuda.get_tmem_addr(tmem_addr[0], 0, i), zero_reg[i]
            )
        T.ptx.tcgen05.wait__st.sync.aligned()
        stage_src = A.sub[(cbx, *s_sl)].view(-1)
        stage_dst = A_smem.sub[s_sl].view(-1)
        for step in range(T.ceildiv(stage_dst.shape[0], 128)):
            index = step * 128 + tid_in_wg
            if index < stage_dst.shape[0]:
                stage_dst[index] = stage_src[index]
        T.cuda.cta_sync()
        T.cuda.cluster_sync()
        if cbx == 0:
            if tid_in_wg == 0:
                _emit_cp(tmem, A_smem, t_region, s_region, shape, opcode, descriptor)
                T.ptx.tcgen05.commit.cta_group__2.mbarrier__arrive__one.shared__cluster.multicast__cluster.b64(
                    cp_mbar.ptr_to([0]), T.uint16(3)
                )
        T.cuda.mbarrier_wait(cp_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence__after_thread_sync()
        reg = T.alloc_tensor(ty_args=[T.Tensor((W32,), "uint32", scope="local", layout="default")])
        for i in range(W32):
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](
                reg[i], T.cuda.get_tmem_addr(tmem_addr[0], 0, i)
            )
        T.ptx.tcgen05.wait__ld.sync.aligned()
        for i in range(W32):
            B[cbx * 128 + tid_in_wg, i] = reg[i]
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(tmem_addr[0], T.uint32(n_cols))

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
        s_full,
        [128, C],
        t_full,
        [128, C],
        {"shape": "128x256b", "cta_group": 2, "descriptor": (0, 256, 1)},
        C * 16 // 32,
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
        {
            "shape": "64x128b",
            "multicast": "warpx2::02_13",
            "cta_group": 2,
            "descriptor": (0, 128, 0),
        },
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
        "4x256b",
        None,
        W32,
        32,
        descriptor=(0, 256, 1),
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


def test_cp_explicit_32x128b_instruction_sequence():
    """The original four-tile input produces four explicit 32x128b.warpx4 calls.

    The selected descriptor is (ldo=0, sdo=128B, swizzle=0); native operations
    read shared bytes 0/512/1024/1536 into TMEM columns 0/4/8/12.
    """
    s_full = TileLayout(S[(4, 32, 16) : (512, 16, 1)])
    t_full = TileLayout(S[(4, 32, 16) : (16 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])

    @T.function(check_well_formed=False)
    def kernel(A: T.Tensor((4, 32, 16), "uint8")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=4 * 32))
        warp_id = T.cuda.warp_id()
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()
        _lane = T.cuda.lane_id()
        A_smem = T.alloc_tensor(
            ty_args=[T.Tensor((4, 32, 16), "uint8", scope="shared", layout=s_full, align=1024)]
        )
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    T.address_of(tmem_addr), T.uint32(32)
                )
            T.cuda.cta_sync()
            stage_src = A.sub[:, :, :].view(-1)
            stage_dst = A_smem.sub[:, :, :].view(-1)
            for step in range(T.ceildiv(stage_dst.shape[0], 128)):
                index = step * 128 + tid_in_wg
                if index < stage_dst.shape[0]:
                    stage_dst[index] = stage_src[index]
            T.cuda.cta_sync()
            tmem = T.cuda.decl_tmem(
                tmem_addr[0], ty_args=[T.Tensor((4, 32, 16), "uint8", scope="tmem", layout=t_full)]
            )
            if tid_in_wg == 0:
                # The original default shape is now written explicitly.
                T.cuda.tile.apply(
                    T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"],
                    tmem.view("uint32"),
                    A_smem.view("uint32"),
                    ldo_bytes=0,
                    sdo_bytes=128,
                )
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(32))

    mod = _compile(kernel)
    cp_lines = _cp_call_lines(mod, "32x128b", "warpx4")
    assert len(cp_lines) == 4
    # The host readback test checks the physical correspondence independently.
    for line in cp_lines:
        assert "16383" in line


# Invalid native opcode, ownership, alignment and descriptor contracts


def _assert_compile_raises(kernel, fragment):
    target = tvm.target.Target("cuda")
    with target:
        with pytest.raises(Exception, match=fragment):
            tvm.tirx.compile(tvm.IRModule({"main": kernel}), target=target, pipeline="tirx")


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
    _assert_compile_raises(kernel, "replicas must match")


def test_cp_rejects_illegal_shape_multicast_combo():
    with pytest.raises((ValueError, RuntimeError), match="takes no multicast|Invalid PTX"):
        _build_case("128x256b", None, 3, "bfloat16", 1, extra_cfg={"multicast": "warpx4"})


def test_cp_rejects_missing_multicast_for_64x128b():
    bits = 16
    C = 128 // bits
    t_full, t_full_shape = _tmem_layout_for("64x128b", "warpx2::02_13", C)
    s_full = mma_shared_layout("bfloat16", SwizzleMode(3), [64, 64])
    with pytest.raises((ValueError, RuntimeError), match="requires multicast|Invalid PTX"):
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
    _assert_compile_raises(kernel, "128-bit aligned")


def test_cp_rejects_unknown_shape():
    with pytest.raises((KeyError, ValueError, RuntimeError), match="96x128b|Invalid PTX"):
        _build_case("32x128b", "warpx4", 3, "bfloat16", 1, extra_cfg={"shape": "96x128b"})


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
    _assert_compile_raises(kernel, "128-lane")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("shape,multicast", _SHAPE_MULTICAST)
@pytest.mark.parametrize("sw", [0, 1, 2, 3])
@pytest.mark.parametrize("bits", [4, 6])
@pytest.mark.parametrize("cta_group", [1, 2])
def test_cp_native_decompression(shape, multicast, sw, bits, cta_group):
    """Native fp4/fp6 expansion, including padding, replicas and both CTAs.

    ISA 9.7.18.9.1: b4x16_p64 inserts two low zero bits; b6x16_p32
    inserts two high zero bits. Each source/destination block is 16 bytes.
    """
    rows, atom_bits = _shape_dims(shape)
    cols = atom_bits // 8
    s_rows = max(rows, 8)
    if sw == 0:
        if atom_bits == 128:
            s_shape = [s_rows, cols]
            s_layout = TileLayout(S[tuple(s_shape)])
            descriptor = (0, 128, 0)
        else:
            s_shape = [s_rows, cols]
            s_layout = TileLayout(S[(s_rows // 8, 8, 2, 16) : (256, 16, 128, 1)])
            descriptor = (128, 256, 0)
    else:
        s_shape = [s_rows, max(cols, 16 << sw)]
        s_layout = mma_shared_layout("uint8", sw, s_shape)
        descriptor = (0, 128 << sw, sw)
    t_layout, t_shape = _tmem_layout_for(shape, multicast, cols)
    s_region, t_region = [(0, rows), (0, cols)], [(0, rows), (0, cols)]
    fmt = "b8x16.b4x16_p64" if bits == 4 else "b8x16.b6x16_p32"
    width32 = cols // 4
    if cta_group == 1:
        kernel = _make_cp_kernel(
            s_layout,
            s_shape,
            s_region,
            t_layout,
            t_shape,
            t_region,
            "uint8",
            shape,
            multicast,
            width32,
            32,
            extra_cfg={"decompress": fmt},
            descriptor=descriptor,
            pre_zero=True,
        )
    else:
        kernel = _make_cp_kernel_cta2(
            s_layout,
            s_shape,
            t_layout,
            t_shape,
            "uint8",
            dict(
                shape=shape, multicast=multicast, cta_group=2, descriptor=descriptor, decompress=fmt
            ),
            width32,
            32,
        )
    ex = _compile(kernel)
    assert _cp_opcode(shape, multicast, cta_group, fmt) in ex.mod.imports[0].inspect_source()
    rng = np.random.default_rng(42)
    inputs = rng.integers(0, 256, size=(cta_group, *s_shape), dtype=np.uint8)
    expanded = np.zeros_like(inputs)
    for cta in range(cta_group):
        for row in range(rows):
            for col in range(0, cols, 16):
                values = rng.integers(0, 1 << bits, size=16, dtype=np.uint8)
                packed = sum(int(value) << (bits * i) for i, value in enumerate(values))
                # Leave the padding random; it must not enter the output payload.
                inputs[cta, row, col : col + 2 * bits] = np.frombuffer(
                    packed.to_bytes(2 * bits, "little"), dtype=np.uint8
                )
                expanded[cta, row, col : col + 16] = values << (2 if bits == 4 else 0)
    expected = []
    for cta in range(cta_group):
        values, _ = _expected_readback(
            expanded[cta], s_region, t_layout, t_shape, t_region, multicast, 8, width32
        )
        expected.append(values)
    out = tvm.runtime.tensor(np.zeros((cta_group * 128, width32), np.uint32), tvm.cuda())
    data = inputs[0] if cta_group == 1 else inputs
    ex(tvm.runtime.tensor(data, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), np.concatenate(expected))


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
    _assert_compile_raises(kernel, "descriptor")


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
    _assert_compile_raises(kernel, "descriptor address walk")


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
    _assert_compile_raises(kernel, "descriptor address walk")


# Original 32x128b.warpx4 byte-layout and scale-factor cases

# warpx4 requires the t tensor to declare the broadcast explicitly:
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
    descriptor=(0, 128, 0),
):
    """2D variant: SMEM/TMEM are both 2D; copy a rectangular sub-region."""
    n_tmem_cols_total = max(32, t_full_shape[-1])
    OUT_LANES = 32
    OUT_BYTES = 16

    @T.function(check_well_formed=False)
    def kernel(A: T.Tensor(s_full_shape, dtype), B: T.Tensor((OUT_LANES, OUT_BYTES), dtype)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=4 * 32))
        warp_id = T.cuda.warp_id()
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()
        lane_id = T.cuda.lane_id()
        A_smem = T.alloc_tensor(
            ty_args=[T.Tensor(s_full_shape, dtype, scope="shared", layout=s_full, align=1024)]
        )
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        cp_mbar = T.alloc_shared([1], "uint64", layout="default")
        if wg_id == 0:
            if warp_id == 0:
                T.ptx[f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"](
                    T.address_of(tmem_addr), T.uint32(n_tmem_cols_total)
                )
            if tid_in_wg == 0:
                T.ptx.mbarrier.init.shared.b64(cp_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.fence.proxy.async_.shared__cta()
            T.cuda.cta_sync()
            stage_src = A.sub[:, :].view(-1)
            stage_dst = A_smem.sub[:, :].view(-1)
            for step in range(T.ceildiv(stage_dst.shape[0], 128)):
                index = step * 128 + tid_in_wg
                if index < stage_dst.shape[0]:
                    stage_dst[index] = stage_src[index]
            T.cuda.cta_sync()
            tmem = T.cuda.decl_tmem(
                tmem_addr[0], ty_args=[T.Tensor(t_full_shape, dtype, scope="tmem", layout=t_full)]
            )
            if tid_in_wg == 0:
                _emit_cp(
                    tmem,
                    A_smem,
                    [(t_r0, t_r1), (t_c0, t_c1)],
                    [(s_r0, s_r1), (s_c0, s_c1)],
                    "32x128b",
                    _cp_opcode("32x128b", "warpx4", cta_group),
                    descriptor,
                )
                T.ptx[
                    f"tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one.shared::cluster.b64"
                ](cp_mbar.ptr_to([0]))
            T.cuda.mbarrier_wait(cp_mbar.ptr_to([0]), 0)
            T.cuda.cta_sync()
            T.ptx.tcgen05.fence__after_thread_sync()
            if warp_id == 0:
                reg = T.alloc_tensor(
                    ty_args=[T.Tensor((4,), "uint32", scope="local", layout="default")]
                )
                for i in range(4):
                    T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](
                        reg[i],
                        T.cuda.get_tmem_addr(
                            T.cast(T.reinterpret(tmem.data, ty="uint64"), "uint32"), 0, i
                        ),
                    )
                T.ptx.tcgen05.wait__ld.sync.aligned()
                B_bytes = reg.view(dtype)
                for i in range(OUT_BYTES):
                    B[lane_id, i] = B_bytes[i]
            if warp_id == 0:
                T.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned"]()
                T.ptx[f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32"](
                    tmem_addr[0], T.uint32(n_tmem_cols_total)
                )

    return kernel


def _make_3d_4tile_kernel(s_full, t_full, s_full_shape, t_full_shape, dtype, cta_group=1):
    """3D variant: 4 stacked tiles (NVFP4-style multi-cp test)."""
    n_tmem_cols_total = max(32, t_full_shape[-1])

    @T.function(check_well_formed=False)
    def kernel(A: T.Tensor(s_full_shape, dtype), B: T.Tensor((32, 16), dtype)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=4 * 32))
        warp_id = T.cuda.warp_id()
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()
        lane_id = T.cuda.lane_id()
        A_smem = T.alloc_tensor(
            ty_args=[T.Tensor(s_full_shape, dtype, scope="shared", layout=s_full, align=1024)]
        )
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        cp_mbar = T.alloc_shared([1], "uint64", layout="default")
        if wg_id == 0:
            if warp_id == 0:
                T.ptx[f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"](
                    T.address_of(tmem_addr), T.uint32(n_tmem_cols_total)
                )
            if tid_in_wg == 0:
                T.ptx.mbarrier.init.shared.b64(cp_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.fence.proxy.async_.shared__cta()
            T.cuda.cta_sync()
            stage_src = A.sub[:, :, :].view(-1)
            stage_dst = A_smem.sub[:, :, :].view(-1)
            for step in range(T.ceildiv(stage_dst.shape[0], 128)):
                index = step * 128 + tid_in_wg
                if index < stage_dst.shape[0]:
                    stage_dst[index] = stage_src[index]
            T.cuda.cta_sync()
            tmem = T.cuda.decl_tmem(
                tmem_addr[0], ty_args=[T.Tensor(t_full_shape, dtype, scope="tmem", layout=t_full)]
            )
            if tid_in_wg == 0:
                T.cuda.tile.apply(
                    T.ptx[_cp_opcode("32x128b", "warpx4", cta_group)],
                    tmem.view("uint32"),
                    A_smem.view("uint32"),
                    ldo_bytes=0,
                    sdo_bytes=128,
                )
                T.ptx[
                    f"tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one.shared::cluster.b64"
                ](cp_mbar.ptr_to([0]))
            T.cuda.mbarrier_wait(cp_mbar.ptr_to([0]), 0)
            T.cuda.cta_sync()
            T.ptx.tcgen05.fence__after_thread_sync()
            if warp_id == 0:
                reg = T.alloc_tensor(
                    ty_args=[T.Tensor((4,), "uint32", scope="local", layout="default")]
                )
                for i in range(4):
                    T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](
                        reg[i],
                        T.cuda.get_tmem_addr(
                            T.cast(T.reinterpret(tmem.data, ty="uint64"), "uint32"), 0, i
                        ),
                    )
                T.ptx.tcgen05.wait__ld.sync.aligned()
                B_bytes = reg.view(dtype)
                for i in range(16):
                    B[lane_id, i] = B_bytes[i]
            if warp_id == 0:
                T.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned"]()
                T.ptx[f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32"](
                    tmem_addr[0], T.uint32(n_tmem_cols_total)
                )

    return kernel


def _run_2d(s_full, t_full, s_full_shape, s_region, dtype, A_init, expected, descriptor):
    s_r0, s_r1 = s_region[0]
    s_c0, s_c1 = s_region[1]
    kernel = _make_2d_kernel(
        s_full,
        t_full,
        s_full_shape,
        [32, 16],
        s_r0,
        s_r1,
        s_c0,
        s_c1,
        0,
        32,
        0,
        16,
        dtype,
        descriptor=descriptor,
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
    descriptor = {
        "sw0_plain_atom_aligned": (0, 128, 0),
        "sw1_32B_atom": (0, 256, 1),
        "sw2_64B_atom": (0, 512, 2),
        "sw3_128B_atom": (0, 1024, 3),
        "sw3_64x128_corner": (0, 1024, 3),
        "sw3_64x128_atom_row_8": (0, 1024, 3),
        "sw2_32x256_col_64": (0, 512, 2),
        "sw0_M_atom_major_4_0": (0, 128, 0),
    }[name]
    _run_2d(s_full, T_LAY_BASIC, s_full_shape, s_region, "uint8", A_np, expected, descriptor)


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
    """Keep the original SFB arrays and expose their common row/word atom.

    The caller permutes the two repeat axes explicitly. Their different shared
    and TMEM strides remain intact; no middle-axis alignment solver is needed.
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

    @T.function(check_well_formed=False)
    def kernel(A: T.Tensor(s_full_shape, "uint8"), B: T.Tensor((32, 16), "uint8")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=4 * 32))
        warp_id = T.cuda.warp_id()
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()
        lane_id = T.cuda.lane_id()
        A_smem = T.alloc_tensor(
            ty_args=[T.Tensor(s_full_shape, "uint8", scope="shared", layout=s_full, align=1024)]
        )
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        cp_mbar = T.alloc_shared([1], "uint64", layout="default")
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    T.address_of(tmem_addr), T.uint32(n_tmem_cols_total)
                )
            if tid_in_wg == 0:
                T.ptx.mbarrier.init.shared.b64(cp_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.fence.proxy.async_.shared__cta()
            T.cuda.cta_sync()
            stage_src = A.sub[:, :].view(-1)
            stage_dst = A_smem.sub[:, :].view(-1)
            for step in range(T.ceildiv(stage_dst.shape[0], 128)):
                index = step * 128 + tid_in_wg
                if index < stage_dst.shape[0]:
                    stage_dst[index] = stage_src[index]
            T.cuda.cta_sync()
            tmem = T.cuda.decl_tmem(
                tmem_addr[0], ty_args=[T.Tensor(t_full_shape, "uint8", scope="tmem", layout=t_full)]
            )
            if tid_in_wg == 0:
                T.cuda.tile.apply(
                    T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"],
                    tmem.view(2, 4, 32, 4, 4).rearrange("a b r c v -> a c r (b v)").view("uint32"),
                    A_smem.view(2, 4, 32, 4, 4)
                    .rearrange("a b r c v -> a c r (b v)")
                    .view("uint32"),
                    ldo_bytes=0,
                    sdo_bytes=128,
                )
                T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                    cp_mbar.ptr_to([0])
                )
            T.cuda.mbarrier_wait(cp_mbar.ptr_to([0]), 0)
            T.cuda.cta_sync()
            T.ptx.tcgen05.fence__after_thread_sync()
            if warp_id == 0:
                reg = T.alloc_tensor(
                    ty_args=[T.Tensor((4,), "uint32", scope="local", layout="default")]
                )
                for i in range(4):
                    T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](
                        reg[i],
                        T.cuda.get_tmem_addr(
                            T.cast(T.reinterpret(tmem.data, ty="uint64"), "uint32"), 0, i
                        ),
                    )
                T.ptx.tcgen05.wait__ld.sync.aligned()
                B_bytes = reg.view("uint8")
                for i in range(16):
                    B[lane_id, i] = B_bytes[i]
            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    tmem_addr[0], T.uint32(n_tmem_cols_total)
                )

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
    "case",
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
def test_cp_subregion_descriptor_contract(case):
    """Two former planner restrictions are legal explicit descriptor regions.

    A 64-byte unswizzled row stride still disagrees with the native 16-byte
    row step; the other two cases are checked against an independent readback.
    """
    name, s_full, s_full_shape, s_region = case
    s_r0, s_r1 = s_region[0]
    s_c0, s_c1 = s_region[1]
    descriptor = {
        "sw3_mid_atom_row": (0, 1024, 3),
        "sw2_mid_atom_col": (0, 512, 2),
        "sw0_row_stride_64": (0, 128, 0),
    }[name]
    kernel = _make_2d_kernel(
        s_full,
        T_LAY_BASIC,
        s_full_shape,
        [32, 16],
        s_r0,
        s_r1,
        s_c0,
        s_c1,
        0,
        32,
        0,
        16,
        "uint8",
        descriptor=descriptor,
    )
    if name == "sw0_row_stride_64":
        _assert_compile_raises(kernel, "descriptor address walk")
    else:
        source = np.random.default_rng(16).integers(0, 256, s_full_shape, dtype=np.uint8)
        _execute(kernel, source, source[s_r0:s_r1, s_c0:s_c1])


def test_multi_cp_reuses_explicit_descriptor_fields():
    s_full = TileLayout(S[(4, 32, 16) : (512, 16, 1)])
    t_full = TileLayout(S[(4, 32, 16) : (16 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])
    kernel = _make_3d_4tile_kernel(s_full, t_full, [4, 32, 16], [4, 32, 16], "uint8")
    ex = _compile(kernel)
    calls = _cp_call_lines(ex, "32x128b", "warpx4")
    assert len(calls) == 4
    assert all("16383" in line for line in calls)


if __name__ == "__main__":
    tvm.testing.main()
