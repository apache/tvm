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
"""End-to-end tests for the smem->tmem (tcgen05.cp.32x128b.warpx4) dispatch.

The new dispatch requires the user to declare the t buffer with an
explicit ``R[4 : 32@TLane]`` indicating warpx4 broadcast — i.e., t.shape[lane] = 32
with replica 4 → 128 physical lanes.

Run with: pytest test_smem_tmem_dispatch.py -n 8 -v
"""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tirx.layout import R, S, TCol, TileLayout, TLane
from tvm.tirx.operator.tile_primitive.cuda.tma_utils import SwizzleMode, mma_shared_layout

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

    @Tx.prim_func(check_well_formed=False)
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, s_full_shape, dtype)
        B = Tx.match_buffer(B_ptr, (OUT_LANES, OUT_BYTES), dtype)
        Tx.device_entry()
        warp_id = Tx.warp_id([4])
        wg_id = Tx.warpgroup_id([1])
        tid_in_wg = Tx.thread_id_in_wg([128])
        lane_id = Tx.lane_id([32])
        A_smem = Tx.alloc_buffer(s_full_shape, dtype, scope="shared", layout=s_full, align=1024)
        tmem_addr = Tx.alloc_shared([1], "uint32")
        cp_mbar = Tx.alloc_shared([1], "uint64")
        if wg_id == 0:
            with Tx.warpgroup():
                if warp_id == 0:
                    with Tx.warp():
                        Tx.ptx.tcgen05.alloc(
                            Tx.address_of(tmem_addr),
                            n_cols=n_tmem_cols_total,
                            cta_group=cta_group,
                        )
                if tid_in_wg == 0:
                    with Tx.thread():
                        Tx.ptx.mbarrier.init(cp_mbar.ptr_to([0]), 1)
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.cta_sync()
                with Tx.cta():
                    Tx.copy(A_smem[:, :], A[:, :])
                Tx.cuda.cta_sync()
                tmem = Tx.decl_buffer(
                    t_full_shape,
                    dtype,
                    scope="tmem",
                    allocated_addr=tmem_addr[0],
                    layout=t_full,
                )
                if tid_in_wg == 0:
                    with Tx.thread():
                        Tx.copy_async(
                            tmem[t_r0:t_r1, t_c0:t_c1],
                            A_smem[s_r0:s_r1, s_c0:s_c1],
                            cta_group=cta_group,
                        )
                        Tx.ptx.tcgen05.commit(cp_mbar.ptr_to([0]), cta_group=cta_group)
                Tx.ptx.mbarrier.try_wait(cp_mbar.ptr_to([0]), 0)
                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()
                if warp_id == 0:
                    with Tx.warp():
                        reg = Tx.alloc_buffer((4,), "uint32", scope="local")
                        for i in range(4):
                            Tx.ptx.tcgen05.ld(
                                tmem.allocated_addr[0],
                                reg[i],
                                shape="32x32b",
                                num=1,
                                row=0,
                                col=i,
                            )
                        Tx.ptx.tcgen05.wait.ld()
                        B_bytes = reg.view(dtype)
                        for i in range(OUT_BYTES):
                            B[lane_id, i] = B_bytes[i]
                if warp_id == 0:
                    with Tx.warp():
                        Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
                        Tx.ptx.tcgen05.dealloc(
                            tmem_addr[0], n_cols=n_tmem_cols_total, cta_group=cta_group
                        )

    return kernel


def _make_3d_4tile_kernel(s_full, t_full, s_full_shape, t_full_shape, dtype, cta_group=1):
    """3D variant: 4 stacked tiles (NVFP4-style multi-cp test)."""
    n_tmem_cols_total = max(32, t_full_shape[-1])

    @Tx.prim_func(check_well_formed=False)
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, s_full_shape, dtype)
        B = Tx.match_buffer(B_ptr, (32, 16), dtype)
        Tx.device_entry()
        warp_id = Tx.warp_id([4])
        wg_id = Tx.warpgroup_id([1])
        tid_in_wg = Tx.thread_id_in_wg([128])
        lane_id = Tx.lane_id([32])
        A_smem = Tx.alloc_buffer(s_full_shape, dtype, scope="shared", layout=s_full, align=1024)
        tmem_addr = Tx.alloc_shared([1], "uint32")
        cp_mbar = Tx.alloc_shared([1], "uint64")
        if wg_id == 0:
            with Tx.warpgroup():
                if warp_id == 0:
                    with Tx.warp():
                        Tx.ptx.tcgen05.alloc(
                            Tx.address_of(tmem_addr),
                            n_cols=n_tmem_cols_total,
                            cta_group=cta_group,
                        )
                if tid_in_wg == 0:
                    with Tx.thread():
                        Tx.ptx.mbarrier.init(cp_mbar.ptr_to([0]), 1)
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.cta_sync()
                with Tx.cta():
                    Tx.copy(A_smem[:, :, :], A[:, :, :])
                Tx.cuda.cta_sync()
                tmem = Tx.decl_buffer(
                    t_full_shape,
                    dtype,
                    scope="tmem",
                    allocated_addr=tmem_addr[0],
                    layout=t_full,
                )
                if tid_in_wg == 0:
                    with Tx.thread():
                        Tx.copy_async(
                            tmem[:, :, :],
                            A_smem[:, :, :],
                            cta_group=cta_group,
                        )
                        Tx.ptx.tcgen05.commit(cp_mbar.ptr_to([0]), cta_group=cta_group)
                Tx.ptx.mbarrier.try_wait(cp_mbar.ptr_to([0]), 0)
                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()
                if warp_id == 0:
                    with Tx.warp():
                        reg = Tx.alloc_buffer((4,), "uint32", scope="local")
                        for i in range(4):
                            Tx.ptx.tcgen05.ld(
                                tmem.allocated_addr[0],
                                reg[i],
                                shape="32x32b",
                                num=1,
                                row=0,
                                col=i,
                            )
                        Tx.ptx.tcgen05.wait.ld()
                        B_bytes = reg.view(dtype)
                        for i in range(16):
                            B[lane_id, i] = B_bytes[i]
                if warp_id == 0:
                    with Tx.warp():
                        Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
                        Tx.ptx.tcgen05.dealloc(
                            tmem_addr[0], n_cols=n_tmem_cols_total, cta_group=cta_group
                        )

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
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
    dev = tvm.cuda(0)
    A = tvm.runtime.tensor(A_init, dev)
    B_np = np.zeros((32, 16), dtype=A_init.dtype)
    B = tvm.runtime.tensor(B_np, dev)
    mod(A, B)
    B_out = B.numpy()
    assert np.array_equal(B_out, expected), (
        f"mismatch:\nlane 0 expected={expected[0].tolist()}\n        got     ={B_out[0].tolist()}"
    )


@tvm.testing.requires_cuda_compute_version(10)
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


@tvm.testing.requires_cuda_compute_version(10)
def test_multi_cp_sw0_4tiles():
    s_full = TileLayout(S[(4, 32, 16) : (512, 16, 1)])
    t_full = TileLayout(S[(4, 32, 16) : (16 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])
    A_np = (np.arange(4 * 32 * 16, dtype=np.int32) & 0xFF).astype(np.uint8).reshape(4, 32, 16)
    expected = A_np[0]
    _run_3d_4tile(s_full, t_full, [4, 32, 16], "uint8", A_np, expected)


@tvm.testing.requires_cuda_compute_version(10)
def test_align_middle_2_to_1_nvfp4_sfb():
    """SFB-style nvfp4 case: TMEM mid canonicalizes to single iter
    (16@TCol + 4@TCol merge), but SMEM mid stays as 2 iters
    (stride 512 + stride 2048 — outer/inner reversed so canon can't merge).
    Exercises ``_align_middles`` union-cut algorithm.

    Layout shapes mirror SFB nvfp4 with PIPE=1, SFB_n_chunks=2,
    MMA_K_BLOCKS=4, sf_mma_k=4.
    """
    # SMEM: (2, 4, 32, 4, 4) extents, strides (2048, 4, 16, 512, 1)
    # — N_chunk outer (stride 2048), then sub-warp tile (4, stride 4), lane
    # (32, stride 16), K_block (4, stride 512), sf_mma_k (4, stride 1).
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

    @Tx.prim_func(check_well_formed=False)
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, s_full_shape, "uint8")
        B = Tx.match_buffer(B_ptr, (32, 16), "uint8")
        Tx.device_entry()
        warp_id = Tx.warp_id([4])
        wg_id = Tx.warpgroup_id([1])
        tid_in_wg = Tx.thread_id_in_wg([128])
        lane_id = Tx.lane_id([32])
        A_smem = Tx.alloc_buffer(s_full_shape, "uint8", scope="shared", layout=s_full, align=1024)
        tmem_addr = Tx.alloc_shared([1], "uint32")
        cp_mbar = Tx.alloc_shared([1], "uint64")
        if wg_id == 0:
            with Tx.warpgroup():
                if warp_id == 0:
                    with Tx.warp():
                        Tx.ptx.tcgen05.alloc(
                            Tx.address_of(tmem_addr), n_cols=n_tmem_cols_total, cta_group=1
                        )
                if tid_in_wg == 0:
                    with Tx.thread():
                        Tx.ptx.mbarrier.init(cp_mbar.ptr_to([0]), 1)
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.cta_sync()
                with Tx.cta():
                    Tx.copy(A_smem[:, :], A[:, :])
                Tx.cuda.cta_sync()
                tmem = Tx.decl_buffer(
                    t_full_shape,
                    "uint8",
                    scope="tmem",
                    allocated_addr=tmem_addr[0],
                    layout=t_full,
                )
                if tid_in_wg == 0:
                    with Tx.thread():
                        Tx.copy_async(tmem[:, :], A_smem[:, :], cta_group=1)
                        Tx.ptx.tcgen05.commit(cp_mbar.ptr_to([0]), cta_group=1)
                Tx.ptx.mbarrier.try_wait(cp_mbar.ptr_to([0]), 0)
                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()
                if warp_id == 0:
                    with Tx.warp():
                        reg = Tx.alloc_buffer((4,), "uint32", scope="local")
                        for i in range(4):
                            Tx.ptx.tcgen05.ld(
                                tmem.allocated_addr[0],
                                reg[i],
                                shape="32x32b",
                                num=1,
                                row=0,
                                col=i,
                            )
                        Tx.ptx.tcgen05.wait.ld()
                        B_bytes = reg.view("uint8")
                        for i in range(16):
                            B[lane_id, i] = B_bytes[i]
                if warp_id == 0:
                    with Tx.warp():
                        Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                        Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=n_tmem_cols_total, cta_group=1)

    A_np = (np.arange(256 * 16, dtype=np.int32) & 0xFF).astype(np.uint8).reshape(256, 16)

    # Compute expected: for each (lane=L in 0..32, byte b in 0..15), the
    # tcgen05.ld reads physical (TLane=L, TCol=b). We must invert the TMEM
    # layout to find which logical (m, k) is at that physical position, then
    # expected[L, b] = A[m, k].
    # Layout shard iters (i0..i4) with extents (2, 4, 32, 4, 4) and TMEM
    # strides (16, 4, 1@TLane, 32, 1) — only TLane and TCol contribute.
    # For (TLane=L, TCol=p) with L in 0..32, replica r=0:
    #   i2 = L; remaining iters (i0, i1, i3, i4) contribute to TCol:
    #   p = 16*i0 + 4*i1 + 32*i3 + i4
    # For p in 0..15 only i1 and i4 vary (i0 = i3 = 0):
    #   i1 = p // 4, i4 = p % 4
    # Logical buffer index: rev row-major over iter coords following shard order.
    # Shard order outer→inner: (i0, i1, i2, i3, i4) with extents (2, 4, 32, 4, 4).
    # Logical buffer index = i0*(4*32*4*4) + i1*(32*4*4) + i2*(4*4) + i3*4 + i4
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


@tvm.testing.requires_cuda_compute_version(10)
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
        target = tvm.target.Target("cuda")
        with target:
            tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")


if __name__ == "__main__":
    tvm.testing.main()
