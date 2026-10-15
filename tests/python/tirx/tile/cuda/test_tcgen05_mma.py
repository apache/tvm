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

"""Native tcgen05.mma instructions with explicit operand views."""

import functools
import operator

import numpy as np
import pytest

import tvm
import tvm.testing
from tirx.layout_fixtures import scale_factor_tmem_layout, tcgen05_register_layout
from tvm.backend.cuda.axis import TCol, TLane
from tvm.backend.cuda.axis import tid_in_wg as axis_tid_in_wg
from tvm.backend.cuda.layout import (
    mma_atom_layout,
    mma_atom_shape,
    mma_shared_layout,
    tmem_datapath_layout,
)
from tvm.ir import PrimType, PtrType
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.layout import R, S, TileLayout

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


def next_power_of_2(x):
    return 1 << max(0, x - 1).bit_length()


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize(
    "task",
    [
        (
            ((128, 512), "float32", [(0, 128), (256, 384)]),  # C
            ((3, 128, 64), "float16", [(1, 2), (0, 128), (0, 64)], 3),  # A
            ((3, 128, 64), "float16", [(2, 3), (0, 128), (0, 64)], 3),  # B
            False,  # transA
            False,  # transB
        )
    ],
)
def test_gemm_tcgen05_cta_group_1(task):
    (
        (C_shape, C_dtype, C_region),
        (A_shape, A_dtype, A_region, A_swizzle_mode),
        (B_shape, B_dtype, B_region, B_swizzle_mode),
        transA,
        transB,
    ) = task
    width = C_region[1][1] - C_region[1][0]
    assert C_shape[0] == 128
    assert C_region[0] == (0, 128)
    assert len(C_shape) == 2
    A_elem_bytes = tvm.runtime.DataType(A_dtype).bits // 8
    B_elem_bytes = tvm.runtime.DataType(B_dtype).bits // 8
    C_elem_bytes = tvm.runtime.DataType(C_dtype).bits // 8
    C_elem_32b = 4 // C_elem_bytes
    cols_alloc = max(32, next_power_of_2(C_shape[1] // C_elem_32b))
    A_layout = mma_shared_layout(A_dtype, A_swizzle_mode, A_shape)
    B_layout = mma_shared_layout(B_dtype, B_swizzle_mode, B_shape)

    r_gmem_A = list(slice(0, A_shape[i]) for i in range(len(A_shape)))
    r_gmem_B = list(slice(0, B_shape[i]) for i in range(len(B_shape)))
    total_bytes = (
        functools.reduce(operator.mul, A_shape, 1) * A_elem_bytes
        + functools.reduce(operator.mul, B_shape, 1) * B_elem_bytes
    )

    r_tmem_C = list(slice(C_region[i][0], C_region[i][1]) for i in range(len(C_shape)))
    r_smem_A = list(slice(A_region[i][0], A_region[i][1]) for i in range(len(A_shape)))
    r_smem_B = list(slice(B_region[i][0], B_region[i][1]) for i in range(len(B_shape)))

    @T.function
    def gemm_async(
        A: T.Tensor(A_shape, A_dtype), B: T.Tensor(B_shape, B_dtype), C: T.Tensor(C_shape, C_dtype)
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 4 * 32))
        warp_id = T.cuda.warp_id()
        cta_id = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()

        A_smem = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    A_shape,
                    A_dtype,
                    scope="shared",
                    layout=A_layout,
                    align=1024,
                )
            ],
        )
        B_smem = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    B_shape,
                    B_dtype,
                    scope="shared",
                    layout=B_layout,
                    align=1024,
                )
            ],
        )
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        tma_mbar = T.alloc_shared([1], "uint64", layout="default")
        mma_mbar = T.alloc_shared([1], "uint64", layout="default")

        if tid_in_wg == 0:
            T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.cuda.cta_sync()
        tmem = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    (128, C_shape[1]),
                    C_dtype,
                    scope="tmem",
                    layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]),
                )
            ],
        )

        if tid_in_wg == 0:
            tma_args = T.meta_var(
                {
                    "box": tuple(A_shape),
                    "coords": (0, 0, 0),
                    "swizzle": 3,
                    "mbar": tma_mbar.ptr_to([0]),
                }
            )
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                A_smem[tuple(r_gmem_A)],
                A[tuple(r_gmem_A)],
                **tma_args,
            )
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                B_smem[tuple(r_gmem_B)],
                B[tuple(r_gmem_B)],
                **tma_args,
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        if tid_in_wg == 0:
            dv = tmem.sub[tuple(r_tmem_C)].rearrange("(mr m) (nr n) -> mr nr m n", m=128, n=64)
            av = (
                A_smem.sub[tuple(r_smem_A)]
                .view(128, 64)
                .rearrange("(mr m) (kr k) -> mr kr m k", m=128, k=16)
            )
            bv = (
                B_smem.sub[tuple(r_smem_B)]
                .view(128, 64)
                .rearrange("(nr n) (kr k) -> nr kr n k", n=64, k=16)
            )
            T.cuda.tile.apply(
                T.ptx["tcgen05.mma.cta_group::1.kind::f16"],
                dv,
                av,
                bv,
                instruction_shape=(128, 64, 16),
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                mma_mbar.ptr_to([0])
            )
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        T.ptx.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(width, dtype=C_dtype, layout="default")
        C_view = C_reg.view(
            128, width, layout=TileLayout(S[(128, width) : (1 @ axis_tid_in_wg, 1)])
        )
        if wg_id == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.ld.sync.aligned.32x32b.x128.b32"],
                C_view.view(4, 32, 128),
                tmem.sub[tuple(r_tmem_C)].view(4, 32, 128),
                scope="warpgroup",
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        for col in T.serial(width):
            C[tid_in_wg, C_region[1][0] + col] = C_reg[col]

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))

    np.random.seed(0)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.IRModule({"main": gemm_async})
        # mod.show()
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        # print(mod.mod.imports[0].inspect_source())

        A_np = np.random.randn(*A_shape).astype(A_dtype)
        B_np = np.random.randn(*B_shape).astype(B_dtype)
        C_np = np.zeros(C_shape, dtype=C_dtype)
        C_ref = np.zeros(C_shape, dtype=C_dtype)
        A_ref = np.squeeze(A_np[tuple(r_smem_A)] if not transA else A_np[tuple(r_smem_A)].T)
        B_ref = np.squeeze(B_np[tuple(r_smem_B)] if transB else B_np[tuple(r_smem_B)].T)
        C_ref[tuple(r_tmem_C)] = A_ref @ B_ref

        def run_and_check():
            dev = tvm.cuda(0)
            A_tvm = tvm.runtime.tensor(A_np, dev)
            B_tvm = tvm.runtime.tensor(B_np, dev)
            C_tvm = tvm.runtime.tensor(C_np, dev)
            mod["main"](A_tvm, B_tvm, C_tvm)
            np.testing.assert_allclose(C_tvm.numpy(), C_ref, atol=1e-3, rtol=1e-3)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_gemm_tcgen05_cta_group_1_layout_f_m64():
    """Keep the original Layout F allocation and its 16x256b readback."""
    M, N, K = 64, 64, 64
    A_dtype, B_dtype, C_dtype = "float16", "float16", "float32"
    A_shape, B_shape, C_shape = (M, K), (N, K), (M, N)
    A_layout = mma_shared_layout(A_dtype, 3, A_shape)
    B_layout = mma_shared_layout(B_dtype, 3, B_shape)

    from tvm.backend.cuda.layout import tmem_datapath_layout

    c_layout = tmem_datapath_layout("F", 64, N)

    @T.function
    def gemm_layout_f(
        A: T.Tensor(A_shape, A_dtype), B: T.Tensor(B_shape, B_dtype), C: T.Tensor(C_shape, C_dtype)
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=4 * 32))
        warp_id = T.cuda.warp_id()
        cta_id = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()
        lane_id = T.cuda.lane_id()

        A_smem = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    A_shape,
                    A_dtype,
                    scope="shared",
                    layout=A_layout,
                    align=1024,
                )
            ],
        )
        B_smem = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    B_shape,
                    B_dtype,
                    scope="shared",
                    layout=B_layout,
                    align=1024,
                )
            ],
        )
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        tma_mbar = T.alloc_shared([1], "uint64", layout="default")
        mma_mbar = T.alloc_shared([1], "uint64", layout="default")

        if tid_in_wg == 0:
            T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(64)
            )
        T.cuda.cta_sync()
        # Layout F C operand — the path under test.
        tmem = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    (64, N),
                    C_dtype,
                    scope="tmem",
                    layout=c_layout,
                )
            ],
        )

        if tid_in_wg == 0:
            tma_args = T.meta_var(
                {"box": (64, 64), "coords": (0, 0), "swizzle": 3, "mbar": tma_mbar.ptr_to([0])}
            )
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                A_smem[:, :],
                A[:, :],
                **tma_args,
            )
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                B_smem[:, :],
                B[:, :],
                **tma_args,
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                tma_mbar.ptr_to([0]), T.uint32((M * K + N * K) * 2)
            )
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        if tid_in_wg == 0:
            dv = tmem.view(1, 1, 64, 64)
            av = A_smem.rearrange("(mr m) (kr k) -> mr kr m k", m=64, k=16)
            bv = B_smem.rearrange("(nr n) (kr k) -> nr kr n k", n=64, k=16)
            T.cuda.tile.apply(
                T.ptx["tcgen05.mma.cta_group::1.kind::f16"],
                dv,
                av,
                bv,
                instruction_shape=(64, 64, 16),
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                mma_mbar.ptr_to([0])
            )
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence__after_thread_sync()

        # Read back via .16x256b M=64 (the canonical pairing).
        reg = T.alloc_local(32, dtype="float32", layout="default")
        reg_view = reg.view(64, N, layout=tcgen05_register_layout("16x256b", (64, N), "float32"))
        if wg_id == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"],
                reg_view.view(4, 16, 64),
                tmem.view(4, 16, 64),
                scope="warpgroup",
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()

        # Per-(reg -> row, col) decomposition for .16x256b M=64 fp32 (BT=64 -> rep=8):
        #   r = v0p + 2*va + 4*vb,   v0p in {0,1}, va in {0,1}, vb in [0, 8)
        #   row = (lane_id >> 2) + 8*va + 16*warp_id
        #   col = v0p + ((lane_id & 3) << 1) + 8*vb
        for vb in T.unroll(8):
            for va in T.unroll(2):
                for v0p in T.unroll(2):
                    r: T.let = v0p + 2 * va + 4 * vb
                    row: T.let = (lane_id >> 2) + 8 * va + 16 * warp_id
                    col: T.let = v0p + ((lane_id & 3) << 1) + 8 * vb
                    C[row, col] = reg[r]

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(64))

    np.random.seed(0)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.tirx.compile(
            tvm.IRModule({"main": gemm_layout_f}), target=target, pipeline="tirx"
        )

    A_np = np.random.randn(*A_shape).astype(A_dtype)
    B_np = np.random.randn(*B_shape).astype(B_dtype)
    C_np = np.zeros(C_shape, dtype=C_dtype)
    C_ref = A_np.astype(np.float32) @ B_np.astype(np.float32).T

    def run_and_check():
        dev = tvm.cuda(0)
        A_tvm = tvm.runtime.tensor(A_np, dev)
        B_tvm = tvm.runtime.tensor(B_np, dev)
        C_tvm = tvm.runtime.tensor(C_np, dev)
        mod["main"](A_tvm, B_tvm, C_tvm)
        np.testing.assert_allclose(C_tvm.numpy(), C_ref, atol=1e-2, rtol=1e-2)

    tvm.testing.run_with_gpu_lock(run_and_check)


def _mapa(ptr, rank):
    """`mapa.u64` into a declared register, returned as a value."""
    mapped = T.alloc_local([1], "uint64", layout="default")
    T.evaluate(T.ptx.mapa.u64(mapped[0], ptr, T.uint32(rank)))
    return mapped[0]


def get_shape_per_cta(shape, trans):
    result = list(shape)
    result[-1 if trans else -2] //= 2
    return result


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize(
    "task",
    [
        (
            ((256, 512), "float32", [(0, 128), (128, 256)]),  # C
            ((3, 256, 64), "float16", [(1, 2), (0, 128), (0, 64)], 3),  # A
            ((3, 128, 64), "float16", [(2, 3), (0, 64), (0, 64)], 3),  # B
            False,  # transA
            False,  # transB
        )
    ],
)
def test_gemm_tcgen05_cta_group_2(task):
    (
        (C_shape, C_dtype, C_region),
        (A_shape, A_dtype, A_region, A_swizzle_mode),
        (B_shape, B_dtype, B_region, B_swizzle_mode),
        transA,
        transB,
    ) = task
    width = C_region[1][1] - C_region[1][0]
    assert C_shape[0] == 256
    assert C_region[0] == (0, 128)
    assert len(C_shape) == 2
    A_elem_bytes = tvm.runtime.DataType(A_dtype).bits // 8
    B_elem_bytes = tvm.runtime.DataType(B_dtype).bits // 8
    C_elem_bytes = tvm.runtime.DataType(C_dtype).bits // 8
    C_elem_32b = 4 // C_elem_bytes
    cols_alloc = max(32, next_power_of_2(C_shape[1] // C_elem_32b))

    A_shape_per_cta = get_shape_per_cta(A_shape, transA)
    B_shape_per_cta = get_shape_per_cta(B_shape, transB)
    A_layout = mma_shared_layout(A_dtype, A_swizzle_mode, A_shape_per_cta)
    B_layout = mma_shared_layout(B_dtype, B_swizzle_mode, B_shape_per_cta)

    r_smem_A_in = list(slice(0, A_shape_per_cta[i]) for i in range(len(A_shape_per_cta)))
    r_smem_B_in = list(slice(0, B_shape_per_cta[i]) for i in range(len(B_shape_per_cta)))
    total_bytes = (
        functools.reduce(operator.mul, A_shape, 1) * A_elem_bytes
        + functools.reduce(operator.mul, B_shape, 1) * B_elem_bytes
    )

    r_tmem_C = list(slice(C_region[i][0], C_region[i][1]) for i in range(len(C_shape)))
    r_smem_A = list(slice(A_region[i][0], A_region[i][1]) for i in range(len(A_shape)))
    r_smem_B = list(slice(B_region[i][0], B_region[i][1]) for i in range(len(B_shape)))

    @T.function
    def gemm_async(
        A: T.Tensor(A_shape, A_dtype), B: T.Tensor(B_shape, B_dtype), C: T.Tensor(C_shape, C_dtype)
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(2,), block=1 * 4 * 32, cluster=(2, 1)))
        warp_id = T.cuda.warp_id()
        cbx, cby = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
        cta_id = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()

        A_smem = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    A_shape_per_cta,
                    A_dtype,
                    scope="shared",
                    layout=A_layout,
                    align=1024,
                )
            ],
        )
        B_smem = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    B_shape_per_cta,
                    B_dtype,
                    scope="shared",
                    layout=B_layout,
                    align=1024,
                )
            ],
        )
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        tma_mbar = T.alloc_shared([1], "uint64", layout="default")
        mma_mbar = T.alloc_shared([1], "uint64", layout="default")

        ptr: T.let[T.Var(name="ptr", ty=PtrType(PrimType("uint64"), "shared"))] = T.reinterpret(
            _mapa(tma_mbar.ptr_to([0]), 0), ty=PtrType(PrimType("uint64"), "shared")
        )
        tma_mbar_cta_0 = T.decl_tensor(
            ptr,
            ty_args=[
                T.Tensor(
                    [1],
                    "uint64",
                    scope="shared",
                    layout="default",
                )
            ],
        )

        if tid_in_wg == 0:
            T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))

        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        T.cuda.cluster_sync()
        tmem = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    (128, C_shape[1]),
                    C_dtype,
                    scope="tmem",
                    layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]),
                )
            ],
        )

        tma_args = T.meta_var({"mbar": tma_mbar_cta_0.ptr_to([0]), "swizzle": 3})
        if tid_in_wg == 0:
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
                ],
                A_smem,
                A,
                box=tuple(A_shape_per_cta),
                coords=(0, cbx * 128, 0),
                **tma_args,
            )
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
                ],
                B_smem,
                B,
                box=tuple(B_shape_per_cta),
                coords=(0, cbx * 64, 0),
                **tma_args,
            )
            if cbx == 0:
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                    tma_mbar.ptr_to([0]), T.uint32(total_bytes)
                )

        if cbx == 0:
            T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
            T.ptx.tcgen05.fence__after_thread_sync()
            T.cuda.cta_sync()
            if tid_in_wg == 0:
                dv = tmem.sub[tuple(r_tmem_C)].view(1, 1, 128, 128)
                av = (
                    A_smem.sub[tuple(r_smem_A)]
                    .view(128, 64)
                    .rearrange("(mr m) (kr k) -> mr kr m k", m=128, k=16)
                )
                bv = (
                    B_smem.sub[tuple(r_smem_B)]
                    .view(64, 64)
                    .rearrange("(nr n) (kr k) -> nr kr n k", n=64, k=16)
                )
                T.cuda.tile.apply(
                    T.ptx["tcgen05.mma.cta_group::2.kind::f16"],
                    dv,
                    av,
                    bv,
                    instruction_shape=(256, 128, 16),
                )
                T.ptx[
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                    ".shared::cluster.multicast::cluster.b64"
                ](mma_mbar.ptr_to([0]), T.uint16(3))  # signal cta 1's mbarrier
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)  # both cta 0 and cta 1 have done mma
        T.ptx.tcgen05.fence__after_thread_sync()
        T.cuda.cta_sync()

        C_reg = T.alloc_local(width, dtype=C_dtype, layout="default")
        C_view = C_reg.view(
            128, width, layout=TileLayout(S[(128, width) : (1 @ axis_tid_in_wg, 1)])
        )
        if wg_id == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.ld.sync.aligned.32x32b.x128.b32"],
                C_view.view(4, 32, 128),
                tmem.sub[tuple(r_tmem_C)].view(4, 32, 128),
                scope="warpgroup",
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        for col in T.serial(width):
            C[cbx * 128 + tid_in_wg, C_region[1][0] + col] = C_reg[col]
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))

    np.random.seed(0)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.IRModule({"main": gemm_async})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        # print(mod.mod.imports[0].inspect_source())

        A_np = np.random.randn(*A_shape).astype(A_dtype)
        B_np = np.random.randn(*B_shape).astype(B_dtype)
        C_np = np.zeros(C_shape, dtype=C_dtype)
        C_ref = np.zeros(C_shape, dtype=C_dtype)
        A_ref = np.squeeze(
            A_np[tuple(r_smem_A[:-2])] if not transA else A_np[tuple(r_smem_A[:-2])].T
        )
        B_ref = np.squeeze(B_np[tuple(r_smem_B[:-2])] if transB else B_np[tuple(r_smem_B[:-2])].T)
        C_ref[:, C_region[1][0] : C_region[1][0] + width] = A_ref @ B_ref

        def run_and_check():
            dev = tvm.cuda(0)
            A_tvm = tvm.runtime.tensor(A_np, dev)
            B_tvm = tvm.runtime.tensor(B_np, dev)
            C_tvm = tvm.runtime.tensor(C_np, dev)
            mod["main"](A_tvm, B_tvm, C_tvm)
            np.testing.assert_allclose(C_tvm.numpy(), C_ref, atol=1e-3, rtol=1e-3)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_gemm_tcgen05_cta_group_2_layout_b():
    """Test cta_group=2 with Layout B (2x2 datapath, M=128 total, 64 per CTA).

    TMEM uses the 2x2 layout: logical (64, N) with shard (64, 2, N//2):(1@TLane, 64@TLane, 1@TCol).
    Physical readback via a (128, N//2) tensor aliasing the same TMEM allocation.
    """
    M_per_cta = 64
    N_logical = 128
    N_half = N_logical // 2
    K = 64
    A_dtype = "float16"
    B_dtype = "float16"
    C_dtype = "float32"
    swizzle_mode = 3

    A_shape = (M_per_cta, K)
    B_shape = (N_half, K)  # per CTA: N_logical // cta_group
    C_shape = (M_per_cta * 2, N_logical)  # global output

    A_elem_bytes = tvm.runtime.DataType(A_dtype).bits // 8
    B_elem_bytes = tvm.runtime.DataType(B_dtype).bits // 8
    C_elem_32b = 4 // (tvm.runtime.DataType(C_dtype).bits // 8)
    cols_alloc = max(32, next_power_of_2(N_half // C_elem_32b))

    A_layout = mma_shared_layout(A_dtype, swizzle_mode, A_shape)
    B_layout = mma_shared_layout(B_dtype, swizzle_mode, B_shape)

    # Both CTAs issue TMA copies; mbarrier expects total from both CTAs.
    per_cta_bytes = (
        functools.reduce(operator.mul, A_shape, 1) * A_elem_bytes
        + functools.reduce(operator.mul, B_shape, 1) * B_elem_bytes
    )
    total_bytes = per_cta_bytes * 2

    @T.function
    def gemm_async(
        A: T.Tensor((M_per_cta * 2, K), A_dtype),
        B: T.Tensor((N_logical, K), B_dtype),
        C: T.Tensor(C_shape, C_dtype),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(2,), block=1 * 4 * 32, cluster=(2, 1)))
        warp_id = T.cuda.warp_id()
        cbx, cby = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
        cta_id = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()

        A_smem = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    A_shape,
                    A_dtype,
                    scope="shared",
                    layout=A_layout,
                    align=1024,
                )
            ],
        )
        B_smem = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    B_shape,
                    B_dtype,
                    scope="shared",
                    layout=B_layout,
                    align=1024,
                )
            ],
        )
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        tma_mbar = T.alloc_shared([1], "uint64", layout="default")
        mma_mbar = T.alloc_shared([1], "uint64", layout="default")

        ptr: T.let[T.Var(name="ptr", ty=PtrType(PrimType("uint64"), "shared"))] = T.reinterpret(
            _mapa(tma_mbar.ptr_to([0]), 0), ty=PtrType(PrimType("uint64"), "shared")
        )
        tma_mbar_cta_0 = T.decl_tensor(
            ptr,
            ty_args=[
                T.Tensor(
                    [1],
                    "uint64",
                    scope="shared",
                    layout="default",
                )
            ],
        )

        if tid_in_wg == 0:
            T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))

        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        T.cuda.cluster_sync()
        tmem = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    (M_per_cta, N_logical),
                    C_dtype,
                    scope="tmem",
                    layout=TileLayout(
                        S[(M_per_cta, 2, N_half) : (1 @ TLane, 64 @ TLane, 1 @ TCol)]
                    ),
                )
            ],
        )
        # Physical TMEM view for readback: (128, N_half) standard layout
        tmem_phys = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    (128, N_half),
                    C_dtype,
                    scope="tmem",
                    layout=TileLayout(S[(128, N_half) : (1 @ TLane, 1 @ TCol)]),
                )
            ],
        )

        tma_args = T.meta_var(
            {
                "mbar": tma_mbar_cta_0.ptr_to([0]),
                "swizzle": 3,
                "box": (64, 64),
                "coords": (cbx * 64, 0),
            }
        )
        if tid_in_wg == 0:
            # CTA cbx loads its portion of A and B
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
                ],
                A_smem,
                A,
                **tma_args,
            )
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
                ],
                B_smem,
                B,
                **tma_args,
            )
            if cbx == 0:
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                    tma_mbar.ptr_to([0]), T.uint32(total_bytes)
                )

        if cbx == 0:
            T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
            T.ptx.tcgen05.fence__after_thread_sync()
            T.cuda.cta_sync()
            if tid_in_wg == 0:
                dv = tmem.view(1, 1, 64, 128)
                av = A_smem.rearrange("(mr m) (kr k) -> mr kr m k", m=64, k=16)
                bv = B_smem.rearrange("(nr n) (kr k) -> nr kr n k", n=64, k=16)
                T.cuda.tile.apply(
                    T.ptx["tcgen05.mma.cta_group::2.kind::f16"],
                    dv,
                    av,
                    bv,
                    instruction_shape=(128, 128, 16),
                )
                T.ptx[
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                    ".shared::cluster.multicast::cluster.b64"
                ](mma_mbar.ptr_to([0]), T.uint16(3))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        T.cuda.cta_sync()

        # Readback from physical TMEM view (128 rows x N_half cols)
        # Warps 0,1 (rows 0-63): first N half for M rows 0-63
        # Warps 2,3 (rows 64-127): second N half for M rows 0-63
        C_reg = T.alloc_local(N_half, dtype=C_dtype, layout="default")
        C_view = C_reg.view(
            128, N_half, layout=TileLayout(S[(128, N_half) : (1 @ axis_tid_in_wg, 1)])
        )
        if wg_id == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.ld.sync.aligned.32x32b.x64.b32"],
                C_view.view(4, 32, 64),
                tmem_phys.view(4, 32, 64),
                scope="warpgroup",
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        n_off = (tid_in_wg // 64) * N_half
        for col in T.serial(N_half):
            C[cbx * M_per_cta + tid_in_wg % 64, n_off + col] = C_reg[col]
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))

    np.random.seed(0)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.IRModule({"main": gemm_async})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        A_np = np.random.randn(M_per_cta * 2, K).astype(A_dtype)
        B_np = np.random.randn(N_logical, K).astype(B_dtype)
        C_np = np.zeros(C_shape, dtype=C_dtype)
        # Reference: C = A @ B.T
        C_ref = A_np.astype(np.float32) @ B_np.astype(np.float32).T

        def run_and_check():
            dev = tvm.cuda(0)
            A_tvm = tvm.runtime.tensor(A_np, dev)
            B_tvm = tvm.runtime.tensor(B_np, dev)
            C_tvm = tvm.runtime.tensor(C_np, dev)
            mod["main"](A_tvm, B_tvm, C_tvm)
            np.testing.assert_allclose(C_tvm.numpy(), C_ref, atol=1e-3, rtol=1e-3)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_gemm_tcgen05_cta_group_2_datapath_b_readback():
    """A cta_group=2 GEMM writes and reads a first-class datapath B tensor."""
    m_per_cta = 64
    n_logical = 128
    n_per_cta = n_logical // 2
    k = 64
    input_dtype = "float32"
    a_dtype = "float16"
    b_dtype = "float16"
    c_dtype = "float32"

    a_shape = (m_per_cta, k)
    b_shape = (n_per_cta, k)
    c_shape = (m_per_cta * 2, n_logical)
    a_layout = mma_shared_layout(a_dtype, 3, a_shape)
    b_layout = mma_shared_layout(b_dtype, 3, b_shape)

    @T.function
    def gemm_async(
        A: T.Tensor((m_per_cta * 2, k), input_dtype),
        B: T.Tensor((n_logical, k), input_dtype),
        C: T.Tensor(c_shape, c_dtype),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(2,), block=4 * 32, cluster=(2, 1)))
        warp_id = T.cuda.warp_id()
        cbx, cby = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
        wg_id = T.cuda.warpgroup_id()
        tid = T.cuda.thread_in_warpgroup()

        A_smem = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    a_shape,
                    a_dtype,
                    scope="shared",
                    layout=a_layout,
                    align=1024,
                )
            ],
        )
        B_smem = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    b_shape,
                    b_dtype,
                    scope="shared",
                    layout=b_layout,
                    align=1024,
                )
            ],
        )
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        mma_mbar = T.alloc_shared([1], "uint64", layout="default")

        if tid == 0:
            T.ptx.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(n_per_cta)
            )
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()

        tmem = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    (m_per_cta, n_logical),
                    c_dtype,
                    scope="tmem",
                    layout=tmem_datapath_layout("B", m_per_cta, n_logical),
                )
            ],
        )

        # Use ordinary shared-memory stores here so this test is independent
        # of the TMA/remote-mbarrier path exercised by the older Layout B test.
        for i in range(m_per_cta * k // 128):
            A_smem[(tid + i * 128) // k, (tid + i * 128) % k] = T.Cast(
                A[cbx * m_per_cta + (tid + i * 128) // k, (tid + i * 128) % k], a_dtype
            )
        for i in range(n_per_cta * k // 128):
            B_smem[(tid + i * 128) // k, (tid + i * 128) % k] = T.Cast(
                B[cbx * n_per_cta + (tid + i * 128) // k, (tid + i * 128) % k], b_dtype
            )
        T.cuda.cta_sync()
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cluster_sync()

        if cbx == 0:
            T.ptx.tcgen05.fence__after_thread_sync()
            T.cuda.cta_sync()
            if tid == 0:
                dv = tmem.view(1, 1, 64, 128)
                av = A_smem.rearrange("(mr m) (kr k) -> mr kr m k", m=64, k=16)
                bv = B_smem.rearrange("(nr n) (kr k) -> nr kr n k", n=64, k=16)
                T.cuda.tile.apply(
                    T.ptx["tcgen05.mma.cta_group::2.kind::f16"],
                    dv,
                    av,
                    bv,
                    instruction_shape=(128, 128, 16),
                )
                T.ptx[
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                    ".shared::cluster.multicast::cluster.b64"
                ](mma_mbar.ptr_to([0]), T.uint16(3))

        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        T.cuda.cta_sync()

        frag = T.alloc_local(((m_per_cta) * (n_logical) // 128,), c_dtype).view(
            m_per_cta,
            n_logical,
            layout=tcgen05_register_layout("32x32b", (m_per_cta, n_logical), c_dtype),
        )
        if wg_id == 0:
            rv = frag.rearrange("(wh r) (wl c) -> (wl wh) r c", wh=2, r=32, wl=2)
            tv = tmem.rearrange("(wh r) (wl c) -> (wl wh) r c", wh=2, r=32, wl=2)
            T.cuda.tile.apply(
                T.ptx["tcgen05.ld.sync.aligned.32x32b.x64.b32"], rv, tv, scope="warpgroup"
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()

        frag_local = frag.view(n_per_cta, layout="default")
        for i in range(n_per_cta):
            C[
                cbx * m_per_cta + tid % m_per_cta,
                (tid // m_per_cta) * n_per_cta + i,
            ] = frag_local[i]
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(tmem_addr[0], T.uint32(n_per_cta))

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.tirx.compile(tvm.IRModule({"main": gemm_async}), target=target, pipeline="tirx")

    np.random.seed(0)
    a_np = np.random.randn(m_per_cta * 2, k).astype(input_dtype)
    b_np = np.random.randn(n_logical, k).astype(input_dtype)
    c_np = np.zeros(c_shape, dtype=c_dtype)
    c_ref = a_np.astype(a_dtype).astype(np.float32) @ b_np.astype(b_dtype).astype(np.float32).T

    def run_and_check():
        dev = tvm.cuda(0)
        a_tvm = tvm.runtime.tensor(a_np, dev)
        b_tvm = tvm.runtime.tensor(b_np, dev)
        c_tvm = tvm.runtime.tensor(c_np, dev)
        mod["main"](a_tvm, b_tvm, c_tvm)
        np.testing.assert_allclose(c_tvm.numpy(), c_ref, atol=1e-3, rtol=1e-3)

    tvm.testing.run_with_gpu_lock(run_and_check)


def per_row_quantize_fp8(mat):
    """Quantize each row to fp8_e4m3fn with per-row power-of-2 scales."""
    row_max = np.max(np.abs(mat), axis=-1)
    row_max = np.maximum(row_max, 1e-12)
    log_scale = np.ceil(np.log2(row_max / 448.0))
    scale = np.power(2.0, log_scale)
    mat_fp8 = (mat / scale[..., None]).astype(ml_dtypes.float8_e4m3fn)
    exp_uint8 = (log_scale.astype(np.int32) + 127).astype(np.uint8)
    return mat_fp8, scale, exp_uint8


def pack_scale_uint32(exp_uint8, n_total=128):
    """Pack uint8 scale exponents into uint32 (replicate 4x)."""
    padded = np.full(n_total, 127, dtype=np.uint8)  # 127 = 2^0 = 1.0
    padded[: len(exp_uint8)] = exp_uint8
    packed = padded.astype(np.uint32)
    packed = packed | (packed << 8) | (packed << 16) | (packed << 24)
    return packed


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.skipif(ml_dtypes is None, reason="Requires ml_dtypes")
@pytest.mark.parametrize(
    "task",
    [
        (
            ((128, 512), "float32", [(0, 128), (0, 32)]),  # C
            ((128, 128), "float8_e4m3fn", [(0, 128), (0, 128)], 3),  # A
            ((32, 128), "float8_e4m3fn", [(0, 32), (0, 128)], 3),  # B
            "float8_e8m0fnu",  # scale factor dtype
            False,  # transA
            False,  # transB
        )
    ],
)
def test_gemm_block_scaled_fp8_cta_group_1(task):
    """Test block-scaled fp8 GEMM with cta_group=1 with an explicit native pattern.

    Uses random per-row quantization with float8_e8m0fnu scale factors
    loaded via tcgen05.cp. Reference: C = dequant(A) @ dequant(B).T.
    """
    (
        (C_shape, C_dtype, C_region),
        (A_shape, A_dtype, A_region, A_swizzle_mode),
        (B_shape, B_dtype, B_region, B_swizzle_mode),
        SF_dtype,
        transA,
        transB,
    ) = task

    M, K = A_shape
    N = B_shape[0]
    width = C_region[1][1] - C_region[1][0]
    assert C_shape[0] == 128
    assert C_region[0] == (0, 128)
    assert len(C_shape) == 2

    A_elem_bytes = max(1, tvm.runtime.DataType(A_dtype).bits // 8)
    B_elem_bytes = max(1, tvm.runtime.DataType(B_dtype).bits // 8)
    C_elem_bytes = tvm.runtime.DataType(C_dtype).bits // 8
    C_elem_32b = 4 // C_elem_bytes
    cols_alloc = max(32, next_power_of_2(C_shape[1] // C_elem_32b))

    A_layout = mma_shared_layout(A_dtype, A_swizzle_mode, A_shape)
    B_layout = mma_shared_layout(B_dtype, B_swizzle_mode, B_shape)

    r_gmem_A = list(slice(0, A_shape[i]) for i in range(len(A_shape)))
    r_gmem_B = list(slice(0, B_shape[i]) for i in range(len(B_shape)))
    total_bytes = (
        functools.reduce(operator.mul, A_shape, 1) * A_elem_bytes
        + functools.reduce(operator.mul, B_shape, 1) * B_elem_bytes
    )

    r_tmem_C = list(slice(C_region[i][0], C_region[i][1]) for i in range(len(C_shape)))
    r_smem_A = list(slice(A_region[i][0], A_region[i][1]) for i in range(len(A_shape)))
    r_smem_B = list(slice(B_region[i][0], B_region[i][1]) for i in range(len(B_shape)))

    sf_mma_k = 1  # fp8: 1 scale factor per MMA iteration
    sfa_layout = scale_factor_tmem_layout(M, SF_K=sf_mma_k * 1, sf_per_mma=sf_mma_k)
    sfb_layout = scale_factor_tmem_layout(N, SF_K=sf_mma_k * 1, sf_per_mma=sf_mma_k)
    sf_epc = 32 // tvm.runtime.DataType(SF_dtype).bits
    SFA_TMEM_SPACING = (int(sfa_layout.span("TCol")) + sf_epc - 1) // sf_epc
    SFA_TMEM_START = width
    SFB_TMEM_START = SFA_TMEM_START + SFA_TMEM_SPACING

    # Every K issue explicitly reuses the same per-row scale bytes.
    sfa_repeat = TileLayout(
        S[(1, 4, 4, 32, 1) : (0, 0, 4 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]
    )
    sfb_repeat = TileLayout(
        S[(1, 4, 1, 32, 1) : (0, 0, 4 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]
    )

    F32_BYTES = 4
    F128_BYTES = 16
    SF_smem_layout = TileLayout(S[(4, 32) : (32, 1)])
    SF_smem_post_layout = TileLayout(S[(4, 32) : (1, 4)])

    @T.function
    def gemm_async_fn(
        A: T.Tensor(A_shape, A_dtype),
        B: T.Tensor(B_shape, B_dtype),
        C: T.Tensor(C_shape, C_dtype),
        SFA_in: T.Tensor((128,), "uint32"),
        SFB_in: T.Tensor((128,), "uint32"),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 4 * 32))
        warp_id = T.cuda.warp_id()
        cta_id = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()

        A_smem = T.alloc_tensor(
            ty_args=[T.Tensor(A_shape, A_dtype, scope="shared", layout=A_layout, align=1024)]
        )
        B_smem = T.alloc_tensor(
            ty_args=[T.Tensor(B_shape, B_dtype, scope="shared", layout=B_layout, align=1024)]
        )
        SFA_smem = T.alloc_tensor(
            ty_args=[T.Tensor((4, 32), "uint32", scope="shared", layout=SF_smem_layout)]
        )
        SFB_smem = T.alloc_tensor(
            ty_args=[T.Tensor((4, 32), "uint32", scope="shared", layout=SF_smem_layout)]
        )
        SFA_smem_post = SFA_smem.view(4, 32, layout=SF_smem_post_layout)
        SFB_smem_post = SFB_smem.view(4, 32, layout=SF_smem_post_layout)
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        tma_mbar = T.alloc_shared([1], "uint64", layout="default")
        mma_mbar = T.alloc_shared([1], "uint64", layout="default")
        descSFA = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (1,),
                    "uint64",
                    scope="local",
                    layout="default",
                )
            ],
        )
        descSFB = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (1,),
                    "uint64",
                    scope="local",
                    layout="default",
                )
            ],
        )

        if tid_in_wg == 0:
            T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.cuda.cta_sync()

        tmem = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    (128, C_shape[1]),
                    C_dtype,
                    scope="tmem",
                    layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]),
                )
            ],
        )
        sfa_tmem = T.cuda.decl_tmem(
            SFA_TMEM_START,
            ty_args=[T.Tensor((M, sf_mma_k), SF_dtype, scope="tmem", layout=sfa_layout)],
        )
        sfb_tmem = T.cuda.decl_tmem(
            SFB_TMEM_START,
            ty_args=[T.Tensor((N, sf_mma_k), SF_dtype, scope="tmem", layout=sfb_layout)],
        )

        # TMA load A and B from global to shared
        if tid_in_wg == 0:
            tma_args = T.meta_var({"coords": (0, 0), "swizzle": 3, "mbar": tma_mbar.ptr_to([0])})
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                A_smem[tuple(r_gmem_A)],
                A,
                box=tuple(A_shape),
                **tma_args,
            )
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                B_smem[tuple(r_gmem_B)],
                B,
                box=tuple(B_shape),
                **tma_args,
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        SFA_smem[tid_in_wg // 32, tid_in_wg % 32] = SFA_in[tid_in_wg]
        SFB_smem[tid_in_wg // 32, tid_in_wg % 32] = SFB_in[tid_in_wg]
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        # Transpose scale factors in shared memory
        if warp_id == 0:
            SFA_transpose_regs = T.alloc_local((4,), "uint32", layout="default")
            # Read the entire tile before overwriting the aliased shared storage.
            for sf_row in T.unroll(4):
                SFA_transpose_regs[sf_row] = SFA_smem[
                    sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32
                ]
            T.cuda.warp_sync()
            for sf_row in T.unroll(4):
                SFA_smem_post[sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32] = (
                    SFA_transpose_regs[sf_row]
                )
            T.cuda.warp_sync()
            SFB_transpose_regs = T.alloc_local((4,), "uint32", layout="default")
            # Read the entire tile before overwriting the aliased shared storage.
            for sf_row in T.unroll(4):
                SFB_transpose_regs[sf_row] = SFB_smem[
                    sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32
                ]
            T.cuda.warp_sync()
            for sf_row in T.unroll(4):
                SFB_smem_post[sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32] = (
                    SFB_transpose_regs[sf_row]
                )
            T.cuda.warp_sync()
        T.cuda.cta_sync()

        # Copy SFA/SFB from shared to TMEM via tcgen05.cp, then issue MMA
        if tid_in_wg == 0:
            T.cuda.tcgen05.encode_matrix_descriptor(
                descSFA.data,
                SFA_smem.ptr_to([0] * len(SFA_smem.shape)),
                ldo=16,
                sdo=8 * 4 * F32_BYTES // F128_BYTES,
                swizzle=0,
            )
            T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](T.uint32(SFA_TMEM_START), descSFA[0])
            T.cuda.tcgen05.encode_matrix_descriptor(
                descSFB.data,
                SFB_smem.ptr_to([0] * len(SFB_smem.shape)),
                ldo=16,
                sdo=8 * 4 * F32_BYTES // F128_BYTES,
                swizzle=0,
            )
            T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](T.uint32(SFB_TMEM_START), descSFB[0])

            dv = tmem.sub[tuple(r_tmem_C)].view(1, 1, 128, 32)
            av = A_smem.rearrange("(mr m) (kr k) -> mr kr m k", m=128, k=32)
            bv = B_smem.rearrange("(nr n) (kr k) -> nr kr n k", n=32, k=32)
            sa = sfa_tmem.view(1, 4, 128, 1, layout=sfa_repeat)
            sb = sfb_tmem.view(1, 4, 32, 1, layout=sfb_repeat)
            T.cuda.tile.apply(
                T.ptx["tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"],
                dv,
                av,
                bv,
                sa,
                sb,
                instruction_shape=(128, 32, 32),
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                mma_mbar.ptr_to([0])
            )
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        # Copy result from tmem to global
        T.ptx.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(width, dtype=C_dtype, layout="default")
        C_view = C_reg.view(
            128, width, layout=TileLayout(S[(128, width) : (1 @ axis_tid_in_wg, 1)])
        )
        if wg_id == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"],
                C_view.view(4, 32, 32),
                tmem.sub[tuple(r_tmem_C)].view(4, 32, 32),
                scope="warpgroup",
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        for col in T.serial(width):
            C[tid_in_wg, C_region[1][0] + col] = C_reg[col]

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))

    np.random.seed(0)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.IRModule({"main": gemm_async_fn})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        # Generate random float32 data and quantize per-row
        A_f32 = np.random.randn(*A_shape).astype(np.float32)
        B_f32 = np.random.randn(*B_shape).astype(np.float32)
        A_fp8, sfa_scale, sfa_exp = per_row_quantize_fp8(A_f32)
        B_fp8, sfb_scale, sfb_exp = per_row_quantize_fp8(B_f32)

        sfa_packed = pack_scale_uint32(sfa_exp.ravel(), 128)
        sfb_packed = pack_scale_uint32(sfb_exp.ravel(), 128)

        C_np = np.zeros(C_shape, dtype=C_dtype)
        # Reference: C = dequant(A) @ dequant(B).T
        A_dq = A_fp8[tuple(r_smem_A)].astype(np.float32) * sfa_scale[..., None]
        B_dq = B_fp8[tuple(r_smem_B)].astype(np.float32) * sfb_scale[..., None]
        C_ref = np.zeros(C_shape, dtype=C_dtype)
        C_ref[tuple(r_tmem_C)] = A_dq @ B_dq.T

        def run_and_check():
            dev = tvm.cuda(0)
            A_tvm = tvm.runtime.tensor(A_fp8, dev)
            B_tvm = tvm.runtime.tensor(B_fp8, dev)
            C_tvm = tvm.runtime.tensor(C_np, dev)
            sfa_tvm = tvm.runtime.tensor(sfa_packed, dev)
            sfb_tvm = tvm.runtime.tensor(sfb_packed, dev)
            mod["main"](A_tvm, B_tvm, C_tvm, sfa_tvm, sfb_tvm)
            np.testing.assert_allclose(C_tvm.numpy(), C_ref, atol=1.0, rtol=0.15)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.skipif(ml_dtypes is None, reason="Requires ml_dtypes")
@pytest.mark.parametrize(
    "task",
    [
        (
            (
                (256, 512),
                "float32",
                [(0, 128), (0, 128)],
            ),  # C (cta_group=2, first 128 rows per CTA)
            ((3, 256, 128), "float8_e4m3fn", [(1, 2), (0, 128), (0, 128)], 3),  # A
            ((3, 128, 128), "float8_e4m3fn", [(2, 3), (0, 64), (0, 128)], 3),  # B
            "float8_e8m0fnu",  # scale factor dtype
            False,  # transA
            False,  # transB
        )
    ],
)
def test_gemm_block_scaled_fp8_cta_group_2(task):
    """Test block-scaled fp8 GEMM with cta_group=2 with an explicit native pattern.

    Uses random per-row SFA quantization (256 rows, indexed by cbx per CTA)
    and uniform SFB. Reference: C = dequant(A) @ dequant(B).T.
    """
    (
        (C_shape, C_dtype, C_region),
        (A_shape, A_dtype, A_region, A_swizzle_mode),
        (B_shape, B_dtype, B_region, B_swizzle_mode),
        SF_dtype,
        transA,
        transB,
    ) = task

    A_shape[-1]
    M_total = A_shape[-2]  # 256, split across 2 CTAs
    width = C_region[1][1] - C_region[1][0]
    assert C_shape[0] == 256
    assert C_region[0] == (0, 128)
    assert len(C_shape) == 2

    A_elem_bytes = max(1, tvm.runtime.DataType(A_dtype).bits // 8)
    B_elem_bytes = max(1, tvm.runtime.DataType(B_dtype).bits // 8)
    C_elem_bytes = tvm.runtime.DataType(C_dtype).bits // 8
    C_elem_32b = 4 // C_elem_bytes
    cols_alloc = max(32, next_power_of_2(C_shape[1] // C_elem_32b))

    A_shape_per_cta = get_shape_per_cta(A_shape, transA)
    B_shape_per_cta = get_shape_per_cta(B_shape, transB)
    A_layout = mma_shared_layout(A_dtype, A_swizzle_mode, A_shape_per_cta)
    B_layout = mma_shared_layout(B_dtype, B_swizzle_mode, B_shape_per_cta)

    r_smem_A_in = list(slice(0, A_shape_per_cta[i]) for i in range(len(A_shape_per_cta)))
    r_smem_B_in = list(slice(0, B_shape_per_cta[i]) for i in range(len(B_shape_per_cta)))
    total_bytes = (
        functools.reduce(operator.mul, A_shape, 1) * A_elem_bytes
        + functools.reduce(operator.mul, B_shape, 1) * B_elem_bytes
    )

    r_tmem_C = list(slice(C_region[i][0], C_region[i][1]) for i in range(len(C_shape)))
    r_smem_A = list(slice(A_region[i][0], A_region[i][1]) for i in range(len(A_shape)))
    r_smem_B = list(slice(B_region[i][0], B_region[i][1]) for i in range(len(B_shape)))

    sf_mma_k = 1  # fp8: 1 scale factor per MMA iteration
    sf_layout = scale_factor_tmem_layout(128, SF_K=sf_mma_k * 1, sf_per_mma=sf_mma_k)
    sf_epc = 32 // tvm.runtime.DataType(SF_dtype).bits
    SF_TMEM_SPACING = (int(sf_layout.span("TCol")) + sf_epc - 1) // sf_epc
    N_cols = C_region[1][1] - C_region[1][0]
    SFA_TMEM_START = N_cols
    SFB_TMEM_START = SFA_TMEM_START + SF_TMEM_SPACING

    sf_repeat = TileLayout(
        S[(1, 4, 4, 32, 1) : (0, 0, 4 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]
    )

    F32_BYTES = 4
    F128_BYTES = 16
    SF_smem_layout = TileLayout(S[(4, 32) : (32, 1)])
    SF_smem_post_layout = TileLayout(S[(4, 32) : (1, 4)])

    @T.function
    def gemm_async_fn(
        A: T.Tensor(A_shape, A_dtype),
        B: T.Tensor(B_shape, B_dtype),
        C: T.Tensor(C_shape, C_dtype),
        SFA_in: T.Tensor((M_total,), "uint32"),
        SFB_in: T.Tensor((128,), "uint32"),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(2,), block=1 * 4 * 32, cluster=(2, 1)))
        warp_id = T.cuda.warp_id()
        cbx, cby = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
        cta_id = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()

        A_smem = T.alloc_tensor(
            ty_args=[
                T.Tensor(A_shape_per_cta, A_dtype, scope="shared", layout=A_layout, align=1024)
            ]
        )
        B_smem = T.alloc_tensor(
            ty_args=[
                T.Tensor(B_shape_per_cta, B_dtype, scope="shared", layout=B_layout, align=1024)
            ]
        )
        SFA_smem = T.alloc_tensor(
            ty_args=[T.Tensor((4, 32), "uint32", scope="shared", layout=SF_smem_layout)]
        )
        SFB_smem = T.alloc_tensor(
            ty_args=[T.Tensor((4, 32), "uint32", scope="shared", layout=SF_smem_layout)]
        )
        SFA_smem_post = SFA_smem.view(4, 32, layout=SF_smem_post_layout)
        SFB_smem_post = SFB_smem.view(4, 32, layout=SF_smem_post_layout)
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        tma_mbar = T.alloc_shared([1], "uint64", layout="default")
        mma_mbar = T.alloc_shared([1], "uint64", layout="default")
        descSFA = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (1,),
                    "uint64",
                    scope="local",
                    layout="default",
                )
            ],
        )
        descSFB = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (1,),
                    "uint64",
                    scope="local",
                    layout="default",
                )
            ],
        )

        ptr: T.let[T.Var(name="ptr", ty=PtrType(PrimType("uint64"), "shared"))] = T.reinterpret(
            _mapa(tma_mbar.ptr_to([0]), 0), ty=PtrType(PrimType("uint64"), "shared")
        )
        tma_mbar_cta_0 = T.decl_tensor(
            ptr,
            ty_args=[
                T.Tensor(
                    [1],
                    "uint64",
                    scope="shared",
                    layout="default",
                )
            ],
        )

        if tid_in_wg == 0:
            T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))

        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        T.cuda.cluster_sync()
        tmem = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    (128, C_shape[1]),
                    C_dtype,
                    scope="tmem",
                    layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]),
                )
            ],
        )

        sfa_tmem = T.cuda.decl_tmem(
            SFA_TMEM_START,
            ty_args=[T.Tensor((128, sf_mma_k), SF_dtype, scope="tmem", layout=sf_layout)],
        )
        sfb_tmem = T.cuda.decl_tmem(
            SFB_TMEM_START,
            ty_args=[T.Tensor((128, sf_mma_k), SF_dtype, scope="tmem", layout=sf_layout)],
        )

        # TMA load A and B (both CTAs issue with multicast)
        tma_args = T.meta_var({"mbar": tma_mbar_cta_0.ptr_to([0]), "swizzle": 3})
        if tid_in_wg == 0:
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
                ],
                A_smem,
                A,
                box=tuple(A_shape_per_cta),
                coords=(0, cbx * 128, 0),
                **tma_args,
            )
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
                ],
                B_smem,
                B,
                box=tuple(B_shape_per_cta),
                coords=(0, cbx * 64, 0),
                **tma_args,
            )
            if cbx == 0:
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                    tma_mbar.ptr_to([0]), T.uint32(total_bytes)
                )
        SFA_smem[tid_in_wg // 32, tid_in_wg % 32] = SFA_in[cbx * 128 + tid_in_wg]
        SFB_smem[tid_in_wg // 32, tid_in_wg % 32] = SFB_in[tid_in_wg]
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        # Transpose scale factors (both CTAs)
        if warp_id == 0:
            SFA_transpose_regs = T.alloc_local((4,), "uint32", layout="default")
            # Read the entire tile before overwriting the aliased shared storage.
            for sf_row in T.unroll(4):
                SFA_transpose_regs[sf_row] = SFA_smem[
                    sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32
                ]
            T.cuda.warp_sync()
            for sf_row in T.unroll(4):
                SFA_smem_post[sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32] = (
                    SFA_transpose_regs[sf_row]
                )
            T.cuda.warp_sync()
            SFB_transpose_regs = T.alloc_local((4,), "uint32", layout="default")
            # Read the entire tile before overwriting the aliased shared storage.
            for sf_row in T.unroll(4):
                SFB_transpose_regs[sf_row] = SFB_smem[
                    sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32
                ]
            T.cuda.warp_sync()
            for sf_row in T.unroll(4):
                SFB_smem_post[sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32] = (
                    SFB_transpose_regs[sf_row]
                )
            T.cuda.warp_sync()
        T.cuda.cta_sync()

        # Copy SFA/SFB from shared to TMEM via tcgen05.cp (both CTAs, cta_group=2)
        if tid_in_wg == 0:
            T.cuda.tcgen05.encode_matrix_descriptor(
                descSFA.data,
                SFA_smem.ptr_to([0] * len(SFA_smem.shape)),
                ldo=16,
                sdo=8 * 4 * F32_BYTES // F128_BYTES,
                swizzle=0,
            )
            T.ptx["tcgen05.cp.cta_group::2.32x128b.warpx4"](T.uint32(SFA_TMEM_START), descSFA[0])
            T.cuda.tcgen05.encode_matrix_descriptor(
                descSFB.data,
                SFB_smem.ptr_to([0] * len(SFB_smem.shape)),
                ldo=16,
                sdo=8 * 4 * F32_BYTES // F128_BYTES,
                swizzle=0,
            )
            T.ptx["tcgen05.cp.cta_group::2.32x128b.warpx4"](T.uint32(SFB_TMEM_START), descSFB[0])
        T.cuda.cta_sync()
        T.cuda.cluster_sync()

        if cbx == 0:
            T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
            T.ptx.tcgen05.fence__after_thread_sync()
            T.cuda.cta_sync()
            if tid_in_wg == 0:
                dv = tmem.sub[tuple(r_tmem_C)].view(1, 1, 128, 128)
                av = (
                    A_smem.sub[tuple(r_smem_A)]
                    .view(128, 128)
                    .rearrange("(mr m) (kr k) -> mr kr m k", m=128, k=32)
                )
                bv = (
                    B_smem.sub[tuple(r_smem_B)]
                    .view(64, 128)
                    .rearrange("(nr n) (kr k) -> nr kr n k", n=64, k=32)
                )
                sa = sfa_tmem.view(1, 4, 128, 1, layout=sf_repeat)
                sb = sfb_tmem.view(1, 4, 128, 1, layout=sf_repeat)
                T.cuda.tile.apply(
                    T.ptx["tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X"],
                    dv,
                    av,
                    bv,
                    sa,
                    sb,
                    instruction_shape=(256, 128, 32),
                )
                T.ptx[
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                    ".shared::cluster.multicast::cluster.b64"
                ](mma_mbar.ptr_to([0]), T.uint16(3))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        T.cuda.cta_sync()

        # Copy result from tmem to global
        C_reg = T.alloc_local(width, dtype=C_dtype, layout="default")
        C_view = C_reg.view(
            128, width, layout=TileLayout(S[(128, width) : (1 @ axis_tid_in_wg, 1)])
        )
        if wg_id == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.ld.sync.aligned.32x32b.x128.b32"],
                C_view.view(4, 32, 128),
                tmem.sub[tuple(r_tmem_C)].view(4, 32, 128),
                scope="warpgroup",
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        for col in T.serial(width):
            C[cbx * 128 + tid_in_wg, C_region[1][0] + col] = C_reg[col]
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))

    np.random.seed(0)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.IRModule({"main": gemm_async_fn})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        # Generate random float32 data and quantize
        A_f32 = np.random.randn(*A_shape).astype(np.float32)
        B_f32 = np.random.randn(*B_shape).astype(np.float32)

        # Per-row quantize A's active slice (256 rows)
        A_active = np.squeeze(A_f32[tuple(r_smem_A[:-2])])  # (256, 128)
        A_fp8_active, sfa_scale, sfa_exp = per_row_quantize_fp8(A_active)

        # Per-block quantize B's active slice (uniform scale)
        B_active = np.squeeze(B_f32[tuple(r_smem_B[:-2])])  # (128, 128)
        b_max = max(np.max(np.abs(B_active)), 1e-12)
        b_log = np.ceil(np.log2(b_max / 448.0))
        b_scale = np.power(2.0, b_log)
        B_fp8_active = (B_active / b_scale).astype(ml_dtypes.float8_e4m3fn)
        sfb_exp_val = int(b_log) + 127

        # Put quantized data back into full arrays
        A_fp8 = np.zeros(A_shape, dtype=ml_dtypes.float8_e4m3fn)
        B_fp8 = np.zeros(B_shape, dtype=ml_dtypes.float8_e4m3fn)
        A_fp8[tuple(r_smem_A[:-2])] = A_fp8_active[np.newaxis]
        B_fp8[tuple(r_smem_B[:-2])] = B_fp8_active[np.newaxis]

        # Pack scale factors
        sfa_packed = pack_scale_uint32(sfa_exp.ravel(), M_total)
        sfb_packed = pack_scale_uint32(np.full(128, sfb_exp_val, dtype=np.uint8), 128)

        C_np = np.zeros(C_shape, dtype=C_dtype)
        # Reference: C = dequant(A) @ dequant(B).T
        A_dq = A_fp8_active.astype(np.float32) * sfa_scale[:, None]
        B_dq = B_fp8_active.astype(np.float32) * b_scale
        C_ref = np.zeros(C_shape, dtype=C_dtype)
        C_ref[:, C_region[1][0] : C_region[1][0] + width] = A_dq @ B_dq.T

        def run_and_check():
            dev = tvm.cuda(0)
            A_tvm = tvm.runtime.tensor(A_fp8, dev)
            B_tvm = tvm.runtime.tensor(B_fp8, dev)
            C_tvm = tvm.runtime.tensor(C_np, dev)
            sfa_tvm = tvm.runtime.tensor(sfa_packed, dev)
            sfb_tvm = tvm.runtime.tensor(sfb_packed, dev)
            mod["main"](A_tvm, B_tvm, C_tvm, sfa_tvm, sfb_tvm)
            np.testing.assert_allclose(C_tvm.numpy(), C_ref, atol=1.0, rtol=0.15)

        tvm.testing.run_with_gpu_lock(run_and_check)


def per_row_quantize_nvfp4(mat):
    """Quantize per row: scale = max(|row|) / 6.0 as float8_e4m3fn."""
    row_max = np.max(np.abs(mat), axis=-1)
    row_max = np.maximum(row_max, 1e-12)
    raw_scale = row_max / 6.0
    scale_fp8 = raw_scale.astype(ml_dtypes.float8_e4m3fn)
    scale_f32 = scale_fp8.astype(np.float32)
    scale_f32 = np.maximum(scale_f32, 1e-12)
    mat_fp4 = (mat / scale_f32[..., None]).astype(ml_dtypes.float4_e2m1fn)
    return mat_fp4, scale_fp8, scale_f32


def pack_fp4_to_uint8(fp4_arr):
    """Pack float4_e2m1fn to uint8 matching TVM convention (even=high nibble)."""
    raw = fp4_arr.view(np.uint8)
    even = raw[..., 0::2] & 0x0F
    odd = raw[..., 1::2] & 0x0F
    return ((even << 4) | odd).astype(np.uint8)


def pack_sf_fp8_uint32(sf_uint8, n_total=128):
    """Pack float8_e4m3fn per-row scales into uint32 (replicate 4x)."""
    padded = np.full(n_total, 0x38, dtype=np.uint8)  # 0x38 = float8_e4m3fn(1.0)
    padded[: len(sf_uint8)] = sf_uint8
    packed = padded.astype(np.uint32)
    packed = packed | (packed << 8) | (packed << 16) | (packed << 24)
    return packed


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.skipif(ml_dtypes is None, reason="Requires ml_dtypes")
def test_gemm_block_scaled_nvfp4_cta_group_1():
    """Test block-scaled nvfp4 GEMM with cta_group=1.

    Uses float4_e2m1fn A/B with float8_e4m3fn per-row scale factors.
    Reference: C = dequant(A) @ dequant(B).T.
    """
    M, N, K = 128, 32, 256
    C_shape = (128, 512)
    width = N
    SF_dtype = "float8_e4m3fn"
    C_dtype = "float32"

    A_packed_shape = (M, K // 2)
    B_packed_shape = (N, K // 2)
    A_fp4_shape = (M, K)
    B_fp4_shape = (N, K)

    C_elem_bytes = tvm.runtime.DataType(C_dtype).bits // 8
    C_elem_32b = 4 // C_elem_bytes
    cols_alloc = max(32, next_power_of_2(C_shape[1] // C_elem_32b))

    A_uint8_layout = mma_shared_layout("uint8", 3, A_packed_shape)
    B_uint8_layout = mma_shared_layout("uint8", 3, B_packed_shape)
    A_fp4_layout = mma_shared_layout("float4_e2m1fn", 3, A_fp4_shape)
    B_fp4_layout = mma_shared_layout("float4_e2m1fn", 3, B_fp4_shape)

    total_bytes = M * (K // 2) + N * (K // 2)

    sf_mma_k = 4  # nvfp4: 4 scale factors per MMA iteration (MMA_K=64, SF_VEC=16)
    sfa_layout = scale_factor_tmem_layout(M, SF_K=sf_mma_k * 1, sf_per_mma=sf_mma_k)
    sfb_layout = scale_factor_tmem_layout(N, SF_K=sf_mma_k * 1, sf_per_mma=sf_mma_k)
    sf_epc = 32 // tvm.runtime.DataType(SF_dtype).bits
    SFA_TMEM_SPACING = (int(sfa_layout.span("TCol")) + sf_epc - 1) // sf_epc
    SFA_TMEM_START = width
    SFB_TMEM_START = SFA_TMEM_START + SFA_TMEM_SPACING

    sfa_repeat = TileLayout(
        S[(1, 4, 4, 32, 4) : (0, 0, 4 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]
    )
    sfb_repeat = TileLayout(
        S[(1, 4, 1, 32, 4) : (0, 0, 4 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]
    )

    F32_BYTES = 4
    F128_BYTES = 16
    SF_smem_layout = TileLayout(S[(4, 32) : (32, 1)])
    SF_smem_post_layout = TileLayout(S[(4, 32) : (1, 4)])

    @T.function
    def gemm_async_fn(
        A_packed: T.Tensor(A_packed_shape, "uint8"),
        B_packed: T.Tensor(B_packed_shape, "uint8"),
        C: T.Tensor(C_shape, C_dtype),
        SFA_in: T.Tensor((128,), "uint32"),
        SFB_in: T.Tensor((128,), "uint32"),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 4 * 32))
        warp_id = T.cuda.warp_id()
        cta_id = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()

        A_smem_packed = T.alloc_tensor(
            ty_args=[
                T.Tensor(A_packed_shape, "uint8", scope="shared", layout=A_uint8_layout, align=1024)
            ]
        )
        B_smem_packed = T.alloc_tensor(
            ty_args=[
                T.Tensor(B_packed_shape, "uint8", scope="shared", layout=B_uint8_layout, align=1024)
            ]
        )
        A_smem = T.decl_tensor(
            A_smem_packed.data,
            ty_args=[
                T.Tensor(
                    A_fp4_shape, "float4_e2m1fn", scope="shared", layout=A_fp4_layout, align=1024
                )
            ],
        )
        B_smem = T.decl_tensor(
            B_smem_packed.data,
            ty_args=[
                T.Tensor(
                    B_fp4_shape, "float4_e2m1fn", scope="shared", layout=B_fp4_layout, align=1024
                )
            ],
        )

        SFA_smem = T.alloc_tensor(
            ty_args=[T.Tensor((4, 32), "uint32", scope="shared", layout=SF_smem_layout)]
        )
        SFB_smem = T.alloc_tensor(
            ty_args=[T.Tensor((4, 32), "uint32", scope="shared", layout=SF_smem_layout)]
        )
        SFA_smem_post = SFA_smem.view(4, 32, layout=SF_smem_post_layout)
        SFB_smem_post = SFB_smem.view(4, 32, layout=SF_smem_post_layout)
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        tma_mbar = T.alloc_shared([1], "uint64", layout="default")
        mma_mbar = T.alloc_shared([1], "uint64", layout="default")
        descSFA = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (1,),
                    "uint64",
                    scope="local",
                    layout="default",
                )
            ],
        )
        descSFB = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (1,),
                    "uint64",
                    scope="local",
                    layout="default",
                )
            ],
        )

        if tid_in_wg == 0:
            T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.cuda.cta_sync()

        tmem = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    (128, C_shape[1]),
                    C_dtype,
                    scope="tmem",
                    layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]),
                )
            ],
        )
        sfa_tmem = T.cuda.decl_tmem(
            SFA_TMEM_START,
            ty_args=[T.Tensor((M, sf_mma_k), SF_dtype, scope="tmem", layout=sfa_layout)],
        )
        sfb_tmem = T.cuda.decl_tmem(
            SFB_TMEM_START,
            ty_args=[T.Tensor((N, sf_mma_k), SF_dtype, scope="tmem", layout=sfb_layout)],
        )

        # TMA load A and B as uint8
        if tid_in_wg == 0:
            tma_args = T.meta_var({"coords": (0, 0), "swizzle": 3, "mbar": tma_mbar.ptr_to([0])})
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                A_smem_packed,
                A_packed,
                box=tuple(A_packed_shape),
                **tma_args,
            )
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                B_smem_packed,
                B_packed,
                box=tuple(B_packed_shape),
                **tma_args,
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        SFA_smem[tid_in_wg // 32, tid_in_wg % 32] = SFA_in[tid_in_wg]
        SFB_smem[tid_in_wg // 32, tid_in_wg % 32] = SFB_in[tid_in_wg]
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        # Transpose scale factors in shared memory
        if warp_id == 0:
            SFA_transpose_regs = T.alloc_local((4,), "uint32", layout="default")
            # Read the entire tile before overwriting the aliased shared storage.
            for sf_row in T.unroll(4):
                SFA_transpose_regs[sf_row] = SFA_smem[
                    sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32
                ]
            T.cuda.warp_sync()
            for sf_row in T.unroll(4):
                SFA_smem_post[sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32] = (
                    SFA_transpose_regs[sf_row]
                )
            T.cuda.warp_sync()
            SFB_transpose_regs = T.alloc_local((4,), "uint32", layout="default")
            # Read the entire tile before overwriting the aliased shared storage.
            for sf_row in T.unroll(4):
                SFB_transpose_regs[sf_row] = SFB_smem[
                    sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32
                ]
            T.cuda.warp_sync()
            for sf_row in T.unroll(4):
                SFB_smem_post[sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32] = (
                    SFB_transpose_regs[sf_row]
                )
            T.cuda.warp_sync()
        T.cuda.cta_sync()

        # Copy SFA/SFB from shared to TMEM via tcgen05.cp, then issue MMA
        if tid_in_wg == 0:
            T.cuda.tcgen05.encode_matrix_descriptor(
                descSFA.data,
                SFA_smem.ptr_to([0] * len(SFA_smem.shape)),
                ldo=16,
                sdo=8 * 4 * F32_BYTES // F128_BYTES,
                swizzle=0,
            )
            T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](T.uint32(SFA_TMEM_START), descSFA[0])
            T.cuda.tcgen05.encode_matrix_descriptor(
                descSFB.data,
                SFB_smem.ptr_to([0] * len(SFB_smem.shape)),
                ldo=16,
                sdo=8 * 4 * F32_BYTES // F128_BYTES,
                swizzle=0,
            )
            T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](T.uint32(SFB_TMEM_START), descSFB[0])

            dv = tmem.sub[0:128, 0:width].view(1, 1, 128, width)
            av = A_smem.rearrange("(mr m) (kr k) -> mr kr m k", m=128, k=64)
            bv = B_smem.rearrange("(nr n) (kr k) -> nr kr n k", n=32, k=64)
            sa = sfa_tmem.view(1, 4, 128, 4, layout=sfa_repeat)
            sb = sfb_tmem.view(1, 4, width, 4, layout=sfb_repeat)
            T.cuda.tile.apply(
                T.ptx["tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"],
                dv,
                av,
                bv,
                sa,
                sb,
                instruction_shape=(128, 32, 64),
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                mma_mbar.ptr_to([0])
            )
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        # Copy result from tmem to global
        T.ptx.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(width, dtype=C_dtype, layout="default")
        C_view = C_reg.view(
            128, width, layout=TileLayout(S[(128, width) : (1 @ axis_tid_in_wg, 1)])
        )
        if wg_id == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"],
                C_view.view(4, 32, width),
                tmem.sub[0:128, 0:width].view(4, 32, width),
                scope="warpgroup",
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        for col in T.serial(width):
            C[tid_in_wg, col] = C_reg[col]

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))

    np.random.seed(0)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.IRModule({"main": gemm_async_fn})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        # Generate random float32 data and quantize per-row
        A_f32 = np.random.randn(M, K).astype(np.float32)
        B_f32 = np.random.randn(N, K).astype(np.float32)
        A_fp4, sfa_fp8, sfa_f32 = per_row_quantize_nvfp4(A_f32)
        B_fp4, sfb_fp8, sfb_f32 = per_row_quantize_nvfp4(B_f32)

        # Pack fp4 to uint8 using TVM's convention (even→high nibble, odd→low nibble)
        A_packed = pack_fp4_to_uint8(A_fp4)
        B_packed = pack_fp4_to_uint8(B_fp4)

        sfa_packed = pack_sf_fp8_uint32(sfa_fp8.view(np.uint8).ravel(), 128)
        sfb_packed = pack_sf_fp8_uint32(sfb_fp8.view(np.uint8).ravel(), 128)

        C_np = np.zeros(C_shape, dtype=C_dtype)
        # Reference: C = dequant(A) @ dequant(B).T
        A_dq = A_fp4.astype(np.float32) * sfa_f32[..., None]
        B_dq = B_fp4.astype(np.float32) * sfb_f32[..., None]
        C_ref = np.zeros(C_shape, dtype=C_dtype)
        C_ref[0:128, 0:N] = A_dq @ B_dq.T

        def run_and_check():
            dev = tvm.cuda(0)
            A_tvm = tvm.runtime.tensor(A_packed, dev)
            B_tvm = tvm.runtime.tensor(B_packed, dev)
            C_tvm = tvm.runtime.tensor(C_np, dev)
            sfa_tvm = tvm.runtime.tensor(sfa_packed, dev)
            sfb_tvm = tvm.runtime.tensor(sfb_packed, dev)
            mod["main"](A_tvm, B_tvm, C_tvm, sfa_tvm, sfb_tvm)
            np.testing.assert_allclose(C_tvm.numpy(), C_ref, atol=1.0, rtol=0.15)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.skipif(ml_dtypes is None, reason="Requires ml_dtypes")
def test_gemm_block_scaled_nvfp4_cta_group_2():
    """Test block-scaled nvfp4 GEMM with cta_group=2.

    A: (256, 256) float4_e2m1fn, split M across 2 CTAs (128 each).
    B: (64, 256) float4_e2m1fn, split N across 2 CTAs (32 each).
    Per-row SFA, uniform SFB.
    Reference: C = dequant(A) @ dequant(B).T.
    """
    M_total, N_per_cta, K = 256, 32, 256
    N_total = N_per_cta * 2  # 64
    M_per_cta = M_total // 2  # 128
    C_shape = (M_total, 512)
    width = N_total  # output width per CTA in cta_group=2
    SF_dtype = "float8_e4m3fn"
    C_dtype = "float32"

    # Per-CTA shapes (fp4 element count and uint8 packed)
    A_packed_per_cta = (M_per_cta, K // 2)  # (128, 128)
    B_packed_per_cta = (N_per_cta, K // 2)  # (32, 128)
    A_fp4_per_cta = (M_per_cta, K)  # (128, 256)
    B_fp4_per_cta = (N_per_cta, K)  # (32, 256)

    # Full shapes
    A_packed_shape = (M_total, K // 2)  # (256, 128)
    B_packed_shape = (N_total, K // 2)  # (64, 128)

    C_elem_bytes = tvm.runtime.DataType(C_dtype).bits // 8
    C_elem_32b = 4 // C_elem_bytes
    cols_alloc = max(32, next_power_of_2(C_shape[1] // C_elem_32b))

    A_uint8_layout = mma_shared_layout("uint8", 3, A_packed_per_cta)
    B_uint8_layout = mma_shared_layout("uint8", 3, B_packed_per_cta)
    A_fp4_layout = mma_shared_layout("float4_e2m1fn", 3, A_fp4_per_cta)
    B_fp4_layout = mma_shared_layout("float4_e2m1fn", 3, B_fp4_per_cta)

    total_bytes = M_total * (K // 2) + N_total * (K // 2)

    sf_mma_k = 4  # nvfp4: 4 scale factors per MMA iteration
    sfa_layout = scale_factor_tmem_layout(M_per_cta, SF_K=sf_mma_k * 1, sf_per_mma=sf_mma_k)
    sfb_layout = scale_factor_tmem_layout(N_total, SF_K=sf_mma_k * 1, sf_per_mma=sf_mma_k)
    sf_epc = 32 // tvm.runtime.DataType(SF_dtype).bits
    SFA_TMEM_SPACING = (int(sfa_layout.span("TCol")) + sf_epc - 1) // sf_epc
    (int(sfb_layout.span("TCol")) + sf_epc - 1) // sf_epc
    SFA_TMEM_START = width
    SFB_TMEM_START = SFA_TMEM_START + SFA_TMEM_SPACING

    sfa_repeat = TileLayout(
        S[(1, 4, 4, 32, 4) : (0, 0, 4 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]
    )
    sfb_repeat = TileLayout(
        S[(1, 4, 2, 32, 4) : (0, 0, 4 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]
    )

    F32_BYTES = 4
    F128_BYTES = 16
    SF_smem_layout = TileLayout(S[(4, 32) : (32, 1)])
    SF_smem_post_layout = TileLayout(S[(4, 32) : (1, 4)])

    @T.function
    def gemm_async_fn(
        A_packed: T.Tensor(A_packed_shape, "uint8"),
        B_packed: T.Tensor(B_packed_shape, "uint8"),
        C: T.Tensor(C_shape, C_dtype),
        SFA_in: T.Tensor((M_total,), "uint32"),
        SFB_in: T.Tensor((128,), "uint32"),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(2,), block=1 * 4 * 32, cluster=(2, 1)))
        warp_id = T.cuda.warp_id()
        cbx, cby = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
        cta_id = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()

        A_smem_packed = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    A_packed_per_cta, "uint8", scope="shared", layout=A_uint8_layout, align=1024
                )
            ]
        )
        B_smem_packed = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    B_packed_per_cta, "uint8", scope="shared", layout=B_uint8_layout, align=1024
                )
            ]
        )
        A_smem = T.decl_tensor(
            A_smem_packed.data,
            ty_args=[
                T.Tensor(
                    A_fp4_per_cta, "float4_e2m1fn", scope="shared", layout=A_fp4_layout, align=1024
                )
            ],
        )
        B_smem = T.decl_tensor(
            B_smem_packed.data,
            ty_args=[
                T.Tensor(
                    B_fp4_per_cta, "float4_e2m1fn", scope="shared", layout=B_fp4_layout, align=1024
                )
            ],
        )

        SFA_smem = T.alloc_tensor(
            ty_args=[T.Tensor((4, 32), "uint32", scope="shared", layout=SF_smem_layout)]
        )
        SFB_smem = T.alloc_tensor(
            ty_args=[T.Tensor((4, 32), "uint32", scope="shared", layout=SF_smem_layout)]
        )
        SFA_smem_post = SFA_smem.view(4, 32, layout=SF_smem_post_layout)
        SFB_smem_post = SFB_smem.view(4, 32, layout=SF_smem_post_layout)
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        tma_mbar = T.alloc_shared([1], "uint64", layout="default")
        mma_mbar = T.alloc_shared([1], "uint64", layout="default")
        descSFA = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (1,),
                    "uint64",
                    scope="local",
                    layout="default",
                )
            ],
        )
        descSFB = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (1,),
                    "uint64",
                    scope="local",
                    layout="default",
                )
            ],
        )

        ptr: T.let[T.Var(name="ptr", ty=PtrType(PrimType("uint64"), "shared"))] = T.reinterpret(
            _mapa(tma_mbar.ptr_to([0]), 0), ty=PtrType(PrimType("uint64"), "shared")
        )
        tma_mbar_cta_0 = T.decl_tensor(
            ptr,
            ty_args=[
                T.Tensor(
                    [1],
                    "uint64",
                    scope="shared",
                    layout="default",
                )
            ],
        )

        if tid_in_wg == 0:
            T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))

        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        T.cuda.cluster_sync()
        tmem = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    (128, C_shape[1]),
                    C_dtype,
                    scope="tmem",
                    layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]),
                )
            ],
        )

        sfa_tmem = T.cuda.decl_tmem(
            SFA_TMEM_START,
            ty_args=[T.Tensor((M_per_cta, sf_mma_k), SF_dtype, scope="tmem", layout=sfa_layout)],
        )
        sfb_tmem = T.cuda.decl_tmem(
            SFB_TMEM_START,
            ty_args=[T.Tensor((N_total, sf_mma_k), SF_dtype, scope="tmem", layout=sfb_layout)],
        )

        # TMA load A and B with multicast (each CTA loads its portion)
        tma_args = T.meta_var({"swizzle": 3, "mbar": tma_mbar_cta_0.ptr_to([0])})
        if tid_in_wg == 0:
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
                ],
                A_smem_packed,
                A_packed,
                box=tuple(A_packed_per_cta),
                coords=(cbx * 128, 0),
                **tma_args,
            )
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
                ],
                B_smem_packed,
                B_packed,
                box=tuple(B_packed_per_cta),
                coords=(cbx * 32, 0),
                **tma_args,
            )
            if cbx == 0:
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                    tma_mbar.ptr_to([0]), T.uint32(total_bytes)
                )
        SFA_smem[tid_in_wg // 32, tid_in_wg % 32] = SFA_in[cbx * M_per_cta + tid_in_wg]
        SFB_smem[tid_in_wg // 32, tid_in_wg % 32] = SFB_in[tid_in_wg]
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        # Transpose scale factors
        if warp_id == 0:
            SFA_transpose_regs = T.alloc_local((4,), "uint32", layout="default")
            # Read the entire tile before overwriting the aliased shared storage.
            for sf_row in T.unroll(4):
                SFA_transpose_regs[sf_row] = SFA_smem[
                    sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32
                ]
            T.cuda.warp_sync()
            for sf_row in T.unroll(4):
                SFA_smem_post[sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32] = (
                    SFA_transpose_regs[sf_row]
                )
            T.cuda.warp_sync()
            SFB_transpose_regs = T.alloc_local((4,), "uint32", layout="default")
            # Read the entire tile before overwriting the aliased shared storage.
            for sf_row in T.unroll(4):
                SFB_transpose_regs[sf_row] = SFB_smem[
                    sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32
                ]
            T.cuda.warp_sync()
            for sf_row in T.unroll(4):
                SFB_smem_post[sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32] = (
                    SFB_transpose_regs[sf_row]
                )
            T.cuda.warp_sync()
        T.cuda.cta_sync()

        # Copy SFA/SFB from shared to TMEM via tcgen05.cp
        if tid_in_wg == 0:
            T.cuda.tcgen05.encode_matrix_descriptor(
                descSFA.data,
                SFA_smem.ptr_to([0] * len(SFA_smem.shape)),
                ldo=16,
                sdo=8 * 4 * F32_BYTES // F128_BYTES,
                swizzle=0,
            )
            T.ptx["tcgen05.cp.cta_group::2.32x128b.warpx4"](T.uint32(SFA_TMEM_START), descSFA[0])
            T.cuda.tcgen05.encode_matrix_descriptor(
                descSFB.data,
                SFB_smem.ptr_to([0] * len(SFB_smem.shape)),
                ldo=16,
                sdo=8 * 4 * F32_BYTES // F128_BYTES,
                swizzle=0,
            )
            T.ptx["tcgen05.cp.cta_group::2.32x128b.warpx4"](T.uint32(SFB_TMEM_START), descSFB[0])
        T.cuda.cta_sync()
        T.cuda.cluster_sync()

        if cbx == 0:
            T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
            T.ptx.tcgen05.fence__after_thread_sync()
            T.cuda.cta_sync()
            if tid_in_wg == 0:
                dv = tmem.sub[0:128, 0:width].view(1, 1, 128, width)
                av = A_smem.rearrange("(mr m) (kr k) -> mr kr m k", m=128, k=64)
                bv = B_smem.rearrange("(nr n) (kr k) -> nr kr n k", n=32, k=64)
                sa = sfa_tmem.view(1, 4, 128, 4, layout=sfa_repeat)
                sb = sfb_tmem.view(1, 4, width, 4, layout=sfb_repeat)
                T.cuda.tile.apply(
                    T.ptx["tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X"],
                    dv,
                    av,
                    bv,
                    sa,
                    sb,
                    instruction_shape=(256, 64, 64),
                )
                T.ptx[
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                    ".shared::cluster.multicast::cluster.b64"
                ](mma_mbar.ptr_to([0]), T.uint16(3))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        T.cuda.cta_sync()

        # Copy result from tmem to global
        C_reg = T.alloc_local(width, dtype=C_dtype, layout="default")
        C_view = C_reg.view(
            128, width, layout=TileLayout(S[(128, width) : (1 @ axis_tid_in_wg, 1)])
        )
        if wg_id == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.ld.sync.aligned.32x32b.x64.b32"],
                C_view.view(4, 32, width),
                tmem.sub[0:128, 0:width].view(4, 32, width),
                scope="warpgroup",
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        for col in T.serial(width):
            C[cbx * M_per_cta + tid_in_wg, col] = C_reg[col]
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))

    np.random.seed(0)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.IRModule({"main": gemm_async_fn})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        # Generate random float32 data
        A_f32 = np.random.randn(M_total, K).astype(np.float32)
        B_f32 = np.random.randn(N_total, K).astype(np.float32)

        # Per-row quantize A
        A_fp4, sfa_fp8, sfa_f32 = per_row_quantize_nvfp4(A_f32)

        # Uniform quantize B (same scale for all rows)
        b_max = max(np.max(np.abs(B_f32)), 1e-12)
        b_raw_scale = b_max / 6.0
        b_scale_fp8 = np.float64(b_raw_scale).astype(ml_dtypes.float8_e4m3fn)
        b_scale_f32 = max(float(b_scale_fp8), 1e-12)
        B_fp4 = (B_f32 / b_scale_f32).astype(ml_dtypes.float4_e2m1fn)

        # Pack fp4 to uint8
        A_packed = pack_fp4_to_uint8(A_fp4)
        B_packed = pack_fp4_to_uint8(B_fp4)

        # Pack SFA (per-row fp8 scales)
        sfa_packed = pack_sf_fp8_uint32(sfa_fp8.view(np.uint8).ravel(), M_total)

        # Pack SFB (uniform, replicate across 128 entries)
        sfb_exp = b_scale_fp8.view(np.uint8)
        sfb_packed = pack_sf_fp8_uint32(np.full(128, sfb_exp, dtype=np.uint8), 128)

        C_np = np.zeros(C_shape, dtype=C_dtype)
        # Reference: C = dequant(A) @ dequant(B).T
        A_dq = A_fp4.astype(np.float32) * sfa_f32[..., None]
        B_dq = B_fp4.astype(np.float32) * b_scale_f32
        C_ref = np.zeros(C_shape, dtype=C_dtype)
        C_ref[0:M_total, 0:N_total] = A_dq @ B_dq.T

        def run_and_check():
            dev = tvm.cuda(0)
            A_tvm = tvm.runtime.tensor(A_packed, dev)
            B_tvm = tvm.runtime.tensor(B_packed, dev)
            C_tvm = tvm.runtime.tensor(C_np, dev)
            sfa_tvm = tvm.runtime.tensor(sfa_packed, dev)
            sfb_tvm = tvm.runtime.tensor(sfb_packed, dev)
            mod["main"](A_tvm, B_tvm, C_tvm, sfa_tvm, sfb_tvm)
            np.testing.assert_allclose(C_tvm.numpy(), C_ref, atol=1.0, rtol=0.15)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.skipif(ml_dtypes is None, reason="Requires ml_dtypes")
def test_gemm_block_scaled_fp8_sf_id():
    """Test explicit K-repeat scale-byte views for fp8 block-scaled MMA.

    Per-block quantization (block_size=32) with 4 K-blocks per row, each
    with a different scale factor. The 4 scales are packed into different
    bytes of the uint32 TMEM column. The caller view places successive repeats at sf_id=0,1,2,3
    for each ki iteration, reading the correct byte. Without sf_id rotation,
    only byte 0 would be used for all blocks, giving wrong results.
    """
    M, N, K = 128, 32, 128  # 4 ki iterations (K/MMA_K = 128/32 = 4)
    MMA_K = 32
    num_blocks = K // MMA_K  # 4

    A_dtype = "float8_e4m3fn"
    B_dtype = "float8_e4m3fn"
    C_dtype = "float32"
    SF_dtype = "float8_e8m0fnu"

    C_shape = (128, 512)
    A_shape = (M, K)
    B_shape = (N, K)

    A_elem_bytes = max(1, tvm.runtime.DataType(A_dtype).bits // 8)
    B_elem_bytes = max(1, tvm.runtime.DataType(B_dtype).bits // 8)
    C_elem_bytes = tvm.runtime.DataType(C_dtype).bits // 8
    C_elem_32b = 4 // C_elem_bytes
    cols_alloc = max(32, next_power_of_2(C_shape[1] // C_elem_32b))

    A_layout = mma_shared_layout(A_dtype, 3, A_shape)
    B_layout = mma_shared_layout(B_dtype, 3, B_shape)

    total_bytes = (
        functools.reduce(operator.mul, A_shape, 1) * A_elem_bytes
        + functools.reduce(operator.mul, B_shape, 1) * B_elem_bytes
    )

    sf_mma_k = 1  # fp8: 1 scale factor per MMA iteration
    num_ki = K // MMA_K  # 4: distinct SF positions per call
    sfa_layout = scale_factor_tmem_layout(M, SF_K=sf_mma_k * num_ki, sf_per_mma=sf_mma_k)
    sfb_layout = scale_factor_tmem_layout(N, SF_K=sf_mma_k * num_ki, sf_per_mma=sf_mma_k)
    sf_epc = 32 // tvm.runtime.DataType(SF_dtype).bits
    SFA_TMEM_SPACING = (int(sfa_layout.span("TCol")) + sf_epc - 1) // sf_epc
    SFA_TMEM_START = N
    SFB_TMEM_START = SFA_TMEM_START + SFA_TMEM_SPACING

    F32_BYTES = 4
    F128_BYTES = 16
    SF_smem_layout = TileLayout(S[(4, 32) : (32, 1)])
    SF_smem_post_layout = TileLayout(S[(4, 32) : (1, 4)])

    @T.function
    def gemm_async_fn(
        A: T.Tensor(A_shape, A_dtype),
        B: T.Tensor(B_shape, B_dtype),
        C: T.Tensor(C_shape, C_dtype),
        SFA_in: T.Tensor((128,), "uint32"),
        SFB_in: T.Tensor((128,), "uint32"),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 4 * 32))
        warp_id = T.cuda.warp_id()
        cta_id = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()

        A_smem = T.alloc_tensor(
            ty_args=[T.Tensor(A_shape, A_dtype, scope="shared", layout=A_layout, align=1024)]
        )
        B_smem = T.alloc_tensor(
            ty_args=[T.Tensor(B_shape, B_dtype, scope="shared", layout=B_layout, align=1024)]
        )
        SFA_smem = T.alloc_tensor(
            ty_args=[T.Tensor((4, 32), "uint32", scope="shared", layout=SF_smem_layout)]
        )
        SFB_smem = T.alloc_tensor(
            ty_args=[T.Tensor((4, 32), "uint32", scope="shared", layout=SF_smem_layout)]
        )
        SFA_smem_post = SFA_smem.view(4, 32, layout=SF_smem_post_layout)
        SFB_smem_post = SFB_smem.view(4, 32, layout=SF_smem_post_layout)
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        tma_mbar = T.alloc_shared([1], "uint64", layout="default")
        mma_mbar = T.alloc_shared([1], "uint64", layout="default")
        descSFA = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (1,),
                    "uint64",
                    scope="local",
                    layout="default",
                )
            ],
        )
        descSFB = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (1,),
                    "uint64",
                    scope="local",
                    layout="default",
                )
            ],
        )

        if tid_in_wg == 0:
            T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.cuda.cta_sync()

        tmem = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    C_shape,
                    C_dtype,
                    scope="tmem",
                    layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]),
                )
            ],
        )
        sfa_tmem = T.cuda.decl_tmem(
            SFA_TMEM_START,
            ty_args=[T.Tensor((M, sf_mma_k * num_ki), SF_dtype, scope="tmem", layout=sfa_layout)],
        )
        sfb_tmem = T.cuda.decl_tmem(
            SFB_TMEM_START,
            ty_args=[T.Tensor((N, sf_mma_k * num_ki), SF_dtype, scope="tmem", layout=sfb_layout)],
        )

        # TMA load A and B from global to shared
        if tid_in_wg == 0:
            tma_args = T.meta_var({"coords": (0, 0), "swizzle": 3, "mbar": tma_mbar.ptr_to([0])})
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                A_smem,
                A,
                box=tuple(A_shape),
                **tma_args,
            )
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                B_smem,
                B,
                box=tuple(B_shape),
                **tma_args,
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        SFA_smem[tid_in_wg // 32, tid_in_wg % 32] = SFA_in[tid_in_wg]
        SFB_smem[tid_in_wg // 32, tid_in_wg % 32] = SFB_in[tid_in_wg]
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        # Transpose scale factors in shared memory
        if warp_id == 0:
            SFA_transpose_regs = T.alloc_local((4,), "uint32", layout="default")
            # Read the entire tile before overwriting the aliased shared storage.
            for sf_row in T.unroll(4):
                SFA_transpose_regs[sf_row] = SFA_smem[
                    sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32
                ]
            T.cuda.warp_sync()
            for sf_row in T.unroll(4):
                SFA_smem_post[sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32] = (
                    SFA_transpose_regs[sf_row]
                )
            T.cuda.warp_sync()
            SFB_transpose_regs = T.alloc_local((4,), "uint32", layout="default")
            # Read the entire tile before overwriting the aliased shared storage.
            for sf_row in T.unroll(4):
                SFB_transpose_regs[sf_row] = SFB_smem[
                    sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32
                ]
            T.cuda.warp_sync()
            for sf_row in T.unroll(4):
                SFB_smem_post[sf_row ^ ((tid_in_wg % 32) >> 3), tid_in_wg % 32] = (
                    SFB_transpose_regs[sf_row]
                )
            T.cuda.warp_sync()
        T.cuda.cta_sync()

        # Copy SF to TMEM, then single MMA call (caller views select sf_id per ki)
        if tid_in_wg == 0:
            T.cuda.tcgen05.encode_matrix_descriptor(
                descSFA.data,
                SFA_smem.ptr_to([0] * len(SFA_smem.shape)),
                ldo=16,
                sdo=8 * 4 * F32_BYTES // F128_BYTES,
                swizzle=0,
            )
            T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](T.uint32(SFA_TMEM_START), descSFA[0])
            T.cuda.tcgen05.encode_matrix_descriptor(
                descSFB.data,
                SFB_smem.ptr_to([0] * len(SFB_smem.shape)),
                ldo=16,
                sdo=8 * 4 * F32_BYTES // F128_BYTES,
                swizzle=0,
            )
            T.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](T.uint32(SFB_TMEM_START), descSFB[0])

            # Each explicit K repeat selects its own packed scale byte.
            dv = tmem.sub[0:128, 0:N].view(1, 1, 128, 32)
            av = A_smem.rearrange("(mr m) (kr k) -> mr kr m k", m=128, k=32)
            bv = B_smem.rearrange("(nr n) (kr k) -> nr kr n k", n=32, k=32)
            sa = sfa_tmem.rearrange("(mr m) (kr s) -> mr kr m s", m=128, s=1)
            sb = sfb_tmem.rearrange("(nr n) (kr s) -> nr kr n s", n=32, s=1)
            T.cuda.tile.apply(
                T.ptx["tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"],
                dv,
                av,
                bv,
                sa,
                sb,
                instruction_shape=(128, 32, 32),
            )

            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                mma_mbar.ptr_to([0])
            )
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        # Copy result from tmem to global
        T.ptx.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(N, dtype=C_dtype, layout="default")
        C_view = C_reg.view(128, N, layout=TileLayout(S[(128, N) : (1 @ axis_tid_in_wg, 1)]))
        if wg_id == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"],
                C_view.view(4, 32, 32),
                tmem.sub[0:128, 0:N].view(4, 32, 32),
                scope="warpgroup",
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        for col in T.serial(N):
            C[tid_in_wg, col] = C_reg[col]

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))

    def per_block_quantize_fp8(mat, block_size=32):
        """Quantize per block to fp8_e4m3fn with per-block power-of-2 scales."""
        rows, cols = mat.shape
        n_blocks = cols // block_size
        blocks = mat.reshape(rows, n_blocks, block_size)
        block_max = np.max(np.abs(blocks), axis=-1)
        block_max = np.maximum(block_max, 1e-12)
        log_scale = np.ceil(np.log2(block_max / 448.0))
        scale = np.power(2.0, log_scale)  # (rows, n_blocks)
        mat_fp8 = (blocks / scale[..., None]).astype(ml_dtypes.float8_e4m3fn)
        mat_fp8 = mat_fp8.reshape(rows, cols)
        exp_uint8 = (log_scale.astype(np.int32) + 127).astype(np.uint8)  # (rows, n_blocks)
        return mat_fp8, scale, exp_uint8

    np.random.seed(42)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.IRModule({"main": gemm_async_fn})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        # Create data with very different per-block ranges to ensure sf_id matters
        A_f32 = np.random.randn(M, K).astype(np.float32)
        B_f32 = np.random.randn(N, K).astype(np.float32)
        # Scale blocks to have different ranges
        A_f32[:, 0:32] *= 0.01
        A_f32[:, 32:64] *= 100.0
        A_f32[:, 64:96] *= 1.0
        A_f32[:, 96:128] *= 10.0
        B_f32[:, 0:32] *= 0.01
        B_f32[:, 32:64] *= 100.0
        B_f32[:, 64:96] *= 1.0
        B_f32[:, 96:128] *= 10.0

        A_fp8, A_scale, A_exp = per_block_quantize_fp8(A_f32, block_size=MMA_K)
        B_fp8, B_scale, B_exp = per_block_quantize_fp8(B_f32, block_size=MMA_K)

        # Pack 4 per-block scales into uint32: byte i = scale for block i
        sfa_packed = np.zeros(128, dtype=np.uint32)
        for i in range(num_blocks):
            sfa_packed |= A_exp[:, i].astype(np.uint32) << (8 * i)

        sfb_packed = np.full(128, 0x7F7F7F7F, dtype=np.uint32)  # 127 in all bytes
        sfb_base = np.zeros(N, dtype=np.uint32)
        for i in range(num_blocks):
            sfb_base |= B_exp[:, i].astype(np.uint32) << (8 * i)
        sfb_packed[:N] = sfb_base

        C_np = np.zeros(C_shape, dtype=C_dtype)
        # Reference: per-block dequantize and accumulate
        C_ref = np.zeros(C_shape, dtype=C_dtype)
        for i in range(num_blocks):
            A_block = (
                A_fp8[:, i * MMA_K : (i + 1) * MMA_K].astype(np.float32) * A_scale[:, i : i + 1]
            )
            B_block = (
                B_fp8[:, i * MMA_K : (i + 1) * MMA_K].astype(np.float32) * B_scale[:, i : i + 1]
            )
            C_ref[:M, :N] += A_block @ B_block.T

        def run_and_check():
            dev = tvm.cuda(0)
            A_tvm = tvm.runtime.tensor(A_fp8, dev)
            B_tvm = tvm.runtime.tensor(B_fp8, dev)
            C_tvm = tvm.runtime.tensor(C_np, dev)
            sfa_tvm = tvm.runtime.tensor(sfa_packed, dev)
            sfb_tvm = tvm.runtime.tensor(sfb_packed, dev)
            mod["main"](A_tvm, B_tvm, C_tvm, sfa_tvm, sfb_tvm)
            np.testing.assert_allclose(C_tvm.numpy(), C_ref, atol=1.0, rtol=0.15)

        tvm.testing.run_with_gpu_lock(run_and_check)

        # Sanity: blocks must have different scales (test is meaningless if uniform)
        for i in range(1, num_blocks):
            assert not np.allclose(A_scale[:, 0], A_scale[:, i], atol=1e-6), (
                f"Test requires A blocks 0 and {i} to have different scales"
            )


def _mid_stage_layout(dtype, swizzle_mode, shape):
    """Build SMEM layout for shape (D0, stages, D1) where the middle dim
    (stages) has the highest stride and the [D0, D1] subspace uses the
    standard swizzle atom.  E.g. shape=(128, 3, 64) → stages stride 8192."""
    base_2d = mma_shared_layout(dtype, swizzle_mode, (shape[0], shape[-1]))
    return base_2d.tile_to(shape, [shape[0], 1, shape[-1]])


def _mn_major_layout(dtype, swizzle_mode, shape):
    """Construct MN-major (column-major) SMEM layout: penultimate dim contiguous within atom.

    For shape (..., M, K), the standard K-major atom is [8, T*s] with K contiguous.
    MN-major swaps this: atom becomes [T*s, 8] with M contiguous.
    This is achieved by composing the swizzle with a stride-reversed TileLayout.
    """
    from tvm.tirx.layout import ComposeLayout

    swizzle_atom = mma_atom_layout(dtype, swizzle_mode)
    base_shape = mma_atom_shape(dtype, swizzle_mode)  # 2D: [8, T*s]
    swapped = [base_shape[1], base_shape[0]]  # [T*s, 8]
    # Stride-reversed tile: first dim (T*s) contiguous, second dim (8) has stride T*s
    mn_tile = TileLayout(S[tuple(swapped) : (1, swapped[0])])
    mn_atom = ComposeLayout(
        swizzle_atom.per_element,
        swizzle_atom.swizzle_len,
        swizzle_atom.atom_len,
        mn_tile,
        swizzle_atom.swizzle_inner,
    )
    # Tile up: first expand penultimate dim, then full shape
    tile_step = [1] * (len(shape) - 2) + [shape[-2], swapped[1]]
    atom_nd = [1] * (len(shape) - 2) + swapped
    return mn_atom.tile_to(tile_step, atom_nd).tile_to(shape, tile_step).canonicalize()


def _col_major_layout(shape):
    """Simple column-major layout: penultimate dim contiguous, last dim strided.

    For shape (..., M, K): physical order has M stride=1, K stride=M.
    Leading dims cover the full inner block.
    """
    strides = [0] * len(shape)
    strides[-2] = 1  # M contiguous
    strides[-1] = shape[-2]  # K stride = M
    inner_size = shape[-2] * shape[-1]
    for i in range(len(shape) - 3, -1, -1):
        strides[i] = inner_size
        inner_size *= shape[i]
    return TileLayout(S[tuple(shape) : tuple(strides)])


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize(
    "task,copy_perm,a_major,b_major",
    [
        (task, *recipe)
        for task, recipe in zip(
            [
                # B00005 fix: fp16 K=128 (K > swizzle atom width 64), K_iters=8
                (
                    ((128, 128), "float32", [(0, 128), (0, 128)]),  # C
                    ((3, 128, 128), "float16", [(1, 2), (0, 128), (0, 128)], 3),  # A
                    ((3, 128, 128), "float16", [(2, 3), (0, 128), (0, 128)], 3),  # B
                    False,  # transA
                    False,  # transB
                    1,  # cta_group
                ),
                # B00005 fix: fp16 K=128 with N=64 (different output width), K_iters=8
                (
                    ((128, 64), "float32", [(0, 128), (0, 64)]),  # C
                    ((3, 128, 128), "float16", [(1, 2), (0, 128), (0, 128)], 3),  # A
                    ((3, 64, 128), "float16", [(2, 3), (0, 64), (0, 128)], 3),  # B
                    False,  # transA
                    False,  # transB
                    1,  # cta_group
                ),
                # Transposed B: B stored as [K, N] instead of [N, K]
                (
                    ((128, 128), "float32", [(0, 128), (0, 128)]),  # C
                    ((3, 128, 64), "float16", [(1, 2), (0, 128), (0, 64)], 3),  # A: [stages, M, K]
                    ((3, 64, 128), "float16", [(2, 3), (0, 64), (0, 128)], 3),  # B: [stages, K, N]
                    False,  # transA
                    True,  # transB
                    1,  # cta_group
                ),
                # Transposed A: A stored as [K, M] instead of [M, K]
                (
                    ((128, 128), "float32", [(0, 128), (0, 128)]),  # C
                    ((3, 64, 128), "float16", [(1, 2), (0, 64), (0, 128)], 3),  # A: [stages, K, M]
                    ((3, 128, 64), "float16", [(2, 3), (0, 128), (0, 64)], 3),  # B: [stages, N, K]
                    True,  # transA
                    False,  # transB
                    1,  # cta_group
                ),
                # Both transposed + K=128 (combines B00005 fix with transpose)
                (
                    ((128, 128), "float32", [(0, 128), (0, 128)]),  # C
                    (
                        (3, 128, 128),
                        "float16",
                        [(1, 2), (0, 128), (0, 128)],
                        3,
                    ),  # A: [stages, K=128, M=128]
                    (
                        (3, 128, 128),
                        "float16",
                        [(2, 3), (0, 128), (0, 128)],
                        3,
                    ),  # B: [stages, K=128, N=128]
                    True,  # transA
                    True,  # transB
                    1,  # cta_group
                ),
                # Unit dim in middle: A stored as [M, stages, K] with stages as middle dim
                (
                    ((128, 128), "float32", [(0, 128), (0, 128)]),  # C
                    (
                        (128, 3, 64),
                        "float16",
                        [(0, 128), (1, 2), (0, 64)],  # A: [M, stages, K], stage 1
                        _mid_stage_layout("float16", 3, (128, 3, 64)),
                    ),  # custom layout
                    ((3, 128, 64), "float16", [(2, 3), (0, 128), (0, 64)], 3),  # B: [stages, N, K]
                    False,  # transA
                    False,  # transB
                    1,  # cta_group
                ),
                # MN-major A: both global and SMEM use MN-major (M contiguous).
                # Square inner dims (M=K=128) so column-major reinterpretation = clean transpose.
                (
                    ((128, 128), "float32", [(0, 128), (0, 128)]),  # C: [M=128, N=128]
                    (
                        (3, 128, 128),
                        "float16",
                        [(1, 2), (0, 128), (0, 128)],  # A: [stages, M=128, K=128]
                        _mn_major_layout("float16", 3, (3, 128, 128)),  # SMEM: swizzled MN-major
                        _col_major_layout((3, 128, 128)),  # global: column-major
                        (0, 2, 1),
                    ),  # ref_perm: transpose inner dims for reference
                    (
                        (3, 128, 128),
                        "float16",
                        [(2, 3), (0, 128), (0, 128)],
                        3,
                    ),  # B: [stages, N=128, K=128]
                    False,  # transA
                    False,  # transB
                    1,  # cta_group
                ),
                # transA + K-major SMEM: A is [K, M] with K (penultimate) contiguous in SMEM.
                # Exercises transposed K-major ldo/sdo swap (is_mn_major=F, is_transposed=T).
                (
                    ((128, 128), "float32", [(0, 128), (0, 128)]),  # C: [M=128, N=128]
                    (
                        (3, 128, 128),
                        "float16",
                        [(1, 2), (0, 128), (0, 128)],  # A: [stages, K=128, M=128]
                        _mn_major_layout(
                            "float16", 3, (3, 128, 128)
                        ),  # SMEM: K (penultimate) contiguous
                        _col_major_layout((3, 128, 128)),  # global: column-major (K contiguous)
                        (0, 2, 1),
                    ),  # ref_perm: transpose inner dims for reference
                    (
                        (3, 128, 128),
                        "float16",
                        [(2, 3), (0, 128), (0, 128)],
                        3,
                    ),  # B: [stages, N=128, K=128]
                    True,  # transA
                    False,  # transB
                    1,  # cta_group
                ),
            ],
            [
                ((0, 1, 2), "k", "k"),
                ((0, 1, 2), "k", "k"),
                ((0, 1, 2), "k", "mn"),
                ((0, 1, 2), "mn", "k"),
                ((0, 1, 2), "mn", "mn"),
                ((1, 0, 2), "k", "k"),
                ((0, 2, 1), "mn", "k"),
                ((0, 2, 1), "k", "k"),
            ],
        )
    ],
    ids=[
        "fp16_K128",
        "fp16_K128_N64",
        "transB",
        "transA",
        "transAB_K128",
        "unit_dim_middle",
        "mn_major",
        "transA_kmajor_smem",
    ],
)
def test_gemm_tcgen05_arbitrary_tiles(task, copy_perm, a_major, b_major):
    """Preserve the original arbitrary tiles through fixed caller views.

    Validates B00005 fix (K > atom width) and M/N decomposition.

    A/B spec tuples: (shape, dtype, region, smem_layout_or_swizzle[, gmem_layout[, ref_perm]]).
    gmem_layout: optional global memory layout (default: row-major).
    ref_perm: optional numpy axis permutation for reference data. When the global
      layout is column-major, row-major numpy bytes are reinterpreted by the kernel,
      so the reference must transpose accordingly (e.g. (0, 2, 1) for inner transpose).
    """
    ((C_shape, C_dtype, C_region), A_spec, B_spec, transA, transB, cta_group) = task
    A_shape, A_dtype, A_region, A_swizzle_mode = A_spec[:4]
    A_gmem_layout = A_spec[4] if len(A_spec) > 4 else None
    A_ref_perm = A_spec[5] if len(A_spec) > 5 else None
    B_shape, B_dtype, B_region, B_swizzle_mode = B_spec[:4]
    B_gmem_layout = B_spec[4] if len(B_spec) > 4 else None
    B_ref_perm = B_spec[5] if len(B_spec) > 5 else None
    M = C_region[0][1] - C_region[0][0]
    N = C_region[1][1] - C_region[1][0]
    C_elem_bytes = tvm.runtime.DataType(C_dtype).bits // 8
    C_elem_32b = 4 // C_elem_bytes
    cols_alloc = max(32, next_power_of_2(C_shape[1] // C_elem_32b))
    A_elem_bytes = tvm.runtime.DataType(A_dtype).bits // 8
    B_elem_bytes = tvm.runtime.DataType(B_dtype).bits // 8
    # Accept either swizzle mode (int) or pre-built layout
    A_layout = (
        A_swizzle_mode
        if not isinstance(A_swizzle_mode, int)
        else mma_shared_layout(A_dtype, A_swizzle_mode, A_shape)
    )
    B_layout = (
        B_swizzle_mode
        if not isinstance(B_swizzle_mode, int)
        else mma_shared_layout(B_dtype, B_swizzle_mode, B_shape)
    )

    r_gmem_A = list(slice(0, A_shape[i]) for i in range(len(A_shape)))
    r_gmem_B = list(slice(0, B_shape[i]) for i in range(len(B_shape)))
    total_bytes = (
        functools.reduce(operator.mul, A_shape, 1) * A_elem_bytes
        + functools.reduce(operator.mul, B_shape, 1) * B_elem_bytes
    )

    r_tmem_C = list(slice(C_region[i][0], C_region[i][1]) for i in range(len(C_shape)))
    r_smem_A = list(slice(A_region[i][0], A_region[i][1]) for i in range(len(A_shape)))
    r_smem_B = list(slice(B_region[i][0], B_region[i][1]) for i in range(len(B_shape)))

    a_active = tuple(hi - lo for lo, hi in A_region if hi - lo != 1)
    K = a_active[0] if transA else a_active[1]
    copy_a_shape = tuple(A_shape[i] for i in copy_perm)
    copy_b_shape = B_shape

    A_gmem_kw = {"layout": A_gmem_layout} if A_gmem_layout is not None else {}
    B_gmem_kw = {"layout": B_gmem_layout} if B_gmem_layout is not None else {}

    @T.function
    def gemm_async(
        A: T.Tensor(A_shape, A_dtype, **A_gmem_kw),
        B: T.Tensor(B_shape, B_dtype, **B_gmem_kw),
        C: T.Tensor(C_shape, C_dtype),
    ) -> None:
        ag = A.permute(*copy_perm)
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 4 * 32))
        warp_id = T.cuda.warp_id()
        cta_id = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()

        A_smem = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    A_shape,
                    A_dtype,
                    scope="shared",
                    layout=A_layout,
                    align=1024,
                )
            ],
        )
        B_smem = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    B_shape,
                    B_dtype,
                    scope="shared",
                    layout=B_layout,
                    align=1024,
                )
            ],
        )
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        tma_mbar = T.alloc_shared([1], "uint64", layout="default")
        mma_mbar = T.alloc_shared([1], "uint64", layout="default")

        if tid_in_wg == 0:
            T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx[f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"](
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.cuda.cta_sync()
        tmem = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    (M, C_shape[1]),
                    C_dtype,
                    scope="tmem",
                    layout=TileLayout(S[(M, C_shape[1]) : (1 @ TLane, 1 @ TCol)]),
                )
            ],
        )

        ash = A_smem.permute(*copy_perm)
        # Each issue is one physical swizzle atom. The outer coordinates
        # explicitly tile the original stage, row and column dimensions.
        acopy = ash.rearrange("stage (rg r) (cg c) -> stage rg cg r c", r=8, c=64).view(
            copy_a_shape[0], copy_a_shape[1] // 8, copy_a_shape[2] // 64, 1, 8, 64
        )
        bcopy = B_smem.rearrange("stage (rg r) (cg c) -> stage rg cg r c", r=8, c=64).view(
            copy_b_shape[0], copy_b_shape[1] // 8, copy_b_shape[2] // 64, 1, 8, 64
        )
        if tid_in_wg == 0:
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                acopy,
                ag,
                box=(1, 8, 64),
                coords=(0, 0, 0),
                coord_steps=((1, 0, 0), (0, 8, 0), (0, 0, 64)),
                swizzle=3,
                mbar=tma_mbar.ptr_to([0]),
            )
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                bcopy,
                B,
                box=(1, 8, 64),
                coords=(0, 0, 0),
                coord_steps=((1, 0, 0), (0, 8, 0), (0, 0, 64)),
                swizzle=3,
                mbar=tma_mbar.ptr_to([0]),
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        if T.constexpr(transA):
            amatrix = A_smem.sub[tuple(r_smem_A)].view(K, M).permute(1, 0)
        else:
            amatrix = A_smem.sub[tuple(r_smem_A)].view(M, K)
        if T.constexpr(transB):
            bmatrix = B_smem.sub[tuple(r_smem_B)].view(K, N).permute(1, 0)
        else:
            bmatrix = B_smem.sub[tuple(r_smem_B)].view(N, K)
        av = amatrix.rearrange("(mr m) (kr k) -> mr kr m k", m=128, k=16)
        bv = bmatrix.rearrange("(nr n) (kr k) -> nr kr n k", n=64, k=16)
        dv = tmem.sub[tuple(r_tmem_C)].rearrange("(mr m) (nr n) -> mr nr m n", m=128, n=64)
        if tid_in_wg == 0:
            # The second call must overwrite the first product. Its internal
            # K repeats still accumulate after the first instruction.
            T.cuda.tile.apply(
                T.ptx["tcgen05.mma.cta_group::1.kind::f16"],
                dv,
                av,
                bv,
                instruction_shape=(128, 64, 16),
                a_major=a_major,
                b_major=b_major,
                accum=False,
            )
            T.cuda.tile.apply(
                T.ptx["tcgen05.mma.cta_group::1.kind::f16"],
                dv,
                av,
                bv,
                instruction_shape=(128, 64, 16),
                a_major=a_major,
                b_major=b_major,
            )
            T.ptx[
                f"tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one.shared::cluster.b64"
            ](mma_mbar.ptr_to([0]))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        T.ptx.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(N, dtype=C_dtype, layout="default")
        C_view = C_reg.view(M, N, layout=TileLayout(S[(M, N) : (1 @ axis_tid_in_wg, 1)]))
        if wg_id == 0:
            T.cuda.tile.apply(
                T.ptx[f"tcgen05.ld.sync.aligned.32x32b.x{N}.b32"],
                C_view.view(4, 32, N),
                tmem.sub[tuple(r_tmem_C)].view(4, 32, N),
                scope="warpgroup",
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        for col in T.serial(N):
            C[tid_in_wg, C_region[1][0] + col] = C_reg[col]

        if warp_id == 0:
            T.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned"]()
            T.ptx[f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32"](
                tmem_addr[0], T.uint32(cols_alloc)
            )

    np.random.seed(0)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.IRModule({"main": gemm_async})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        A_np = np.random.randn(*A_shape).astype(A_dtype)
        B_np = np.random.randn(*B_shape).astype(B_dtype)
        C_np = np.zeros(C_shape, dtype=C_dtype)
        C_ref = np.zeros(C_shape, dtype=C_dtype)
        # Apply ref_perm: when global layout differs from row-major, the kernel
        # reinterprets the flat bytes, so the reference must transpose accordingly.
        # Permute both the numpy array and the region indices.
        if A_ref_perm is not None:
            A_np_ref = A_np.transpose(A_ref_perm)
            r_smem_A_ref = [r_smem_A[i] for i in A_ref_perm]
        else:
            A_np_ref, r_smem_A_ref = A_np, r_smem_A
        if B_ref_perm is not None:
            B_np_ref = B_np.transpose(B_ref_perm)
            r_smem_B_ref = [r_smem_B[i] for i in B_ref_perm]
        else:
            B_np_ref, r_smem_B_ref = B_np, r_smem_B
        A_ref = np.squeeze(
            A_np_ref[tuple(r_smem_A_ref)] if not transA else A_np_ref[tuple(r_smem_A_ref)].T
        )
        B_ref = np.squeeze(
            B_np_ref[tuple(r_smem_B_ref)] if transB else B_np_ref[tuple(r_smem_B_ref)].T
        )
        C_ref[tuple(r_tmem_C)] = A_ref @ B_ref

        def run_and_check():
            dev = tvm.cuda(0)
            A_tvm = tvm.runtime.tensor(A_np, dev)
            B_tvm = tvm.runtime.tensor(B_np, dev)
            C_tvm = tvm.runtime.tensor(C_np, dev)
            mod["main"](A_tvm, B_tvm, C_tvm)
            np.testing.assert_allclose(C_tvm.numpy(), C_ref, atol=1e-3, rtol=1e-3)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("k_lo,k_hi", [(0, 16), (0, 32), (16, 32), (16, 48), (32, 64)])
def test_gemm_tcgen05_contiguous_kslice_partial_k(k_lo, k_hi):
    """A slice on the *contiguous* (K) axis of a swizzled gemm_async operand must
    compute the correct partial-K product, not silently use full K.

    The operand tensor is 128B-swizzled (contiguous atom = 64 elems for fp16) and
    the gemm operand is sliced to K=[lo:hi] on that axis. The descriptor is
    anchored on the tensor's physical swizzle while K_iters covers only the slice,
    so the MMA accumulates exactly k in [lo, hi) -- enabling fine K-major split-K.
    Any MMA_K(16)-aligned [lo:hi] is supported.
    """
    M, N, K_alloc = 128, 128, 64
    dtype = "float16"
    A_shape, B_shape, C_shape = (M, K_alloc), (N, K_alloc), (M, N)
    A_layout = mma_shared_layout(dtype, 3, A_shape)
    B_layout = mma_shared_layout(dtype, 3, B_shape)
    total_bytes = (M * K_alloc + N * K_alloc) * 2

    @T.function
    def gemm_async(
        A: T.Tensor(A_shape, dtype), B: T.Tensor(B_shape, dtype), C: T.Tensor(C_shape, "float32")
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=4 * 32))
        warp_id = T.cuda.warp_id()
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()
        A_smem = T.alloc_tensor(
            ty_args=[T.Tensor(A_shape, dtype, scope="shared", layout=A_layout, align=1024)]
        )
        B_smem = T.alloc_tensor(
            ty_args=[T.Tensor(B_shape, dtype, scope="shared", layout=B_layout, align=1024)]
        )
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        tma_mbar = T.alloc_shared([1], "uint64", layout="default")
        mma_mbar = T.alloc_shared([1], "uint64", layout="default")
        if tid_in_wg == 0:
            T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(128)
            )
        T.cuda.cta_sync()
        tmem = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    (128, N),
                    "float32",
                    scope="tmem",
                    layout=TileLayout(S[(128, N) : (1 @ TLane, 1 @ TCol)]),
                )
            ],
        )
        if tid_in_wg == 0:
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                A_smem,
                A,
                box=(M, K_alloc),
                coords=(0, 0),
                swizzle=3,
                mbar=tma_mbar.ptr_to([0]),
            )
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                B_smem,
                B,
                box=(N, K_alloc),
                coords=(0, 0),
                swizzle=3,
                mbar=tma_mbar.ptr_to([0]),
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if tid_in_wg == 0:
            # Contiguous-axis K slice [k_lo:k_hi] -> must accumulate only that K range.
            av = A_smem.sub[:, k_lo:k_hi].rearrange("(mr m) (kr k) -> mr kr m k", m=M, k=16)
            bv = B_smem.sub[:, k_lo:k_hi].rearrange("(nr n) (kr k) -> nr kr n k", n=N, k=16)
            T.cuda.tile.apply(
                T.ptx["tcgen05.mma.cta_group::1.kind::f16"],
                tmem.view(1, 1, M, N),
                av,
                bv,
                instruction_shape=(128, 128, 16),
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                mma_mbar.ptr_to([0])
            )
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(N, dtype="float32", layout="default")
        C_view = C_reg.view(128, N, layout=TileLayout(S[(128, N) : (1 @ axis_tid_in_wg, 1)]))
        if wg_id == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.ld.sync.aligned.32x32b.x128.b32"],
                C_view.view(4, 32, N),
                tmem.view(4, 32, N),
                scope="warpgroup",
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        for col in T.serial(N):
            C[tid_in_wg, col] = C_reg[col]
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(128))

    np.random.seed(0)
    with tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}):
        mod = tvm.tirx.compile(
            tvm.IRModule({"main": gemm_async}),
            target={"kind": "cuda", "arch": "sm_100a"},
            pipeline="tirx",
        )
    A_np = np.random.randn(*A_shape).astype(dtype)
    B_np = np.random.randn(*B_shape).astype(dtype)
    C_np = np.zeros(C_shape, "float32")
    # Reference: accumulate only k in [k_lo, k_hi).
    C_ref = A_np[:, k_lo:k_hi].astype("float32") @ B_np[:, k_lo:k_hi].astype("float32").T

    def run_and_check():
        dev = tvm.cuda(0)
        A_t, B_t, C_t = (tvm.runtime.tensor(x, dev) for x in (A_np, B_np, C_np))
        mod["main"](A_t, B_t, C_t)
        np.testing.assert_allclose(C_t.numpy(), C_ref, atol=1e-2, rtol=1e-2)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.skipif(ml_dtypes is None, reason="Requires ml_dtypes for fp8")
def test_gemm_dense_fp8():
    _run_dense_gemm("float8_e4m3fn", "float8_e4m3fn", "float32", 128, atol=2.0, rtol=0.15)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_gemm_tf32_with_tfloat32_tma():
    _run_dense_gemm(
        "float32",
        "float32",
        "float32",
        64,
        is_AB_tf32=True,
        tma_dtype_B="tf32",
        atol=2e-2,
        rtol=2e-2,
    )


def _run_dense_gemm(
    A_dtype, B_dtype, C_dtype, K, *, is_AB_tf32=False, tma_dtype_B=None, atol=1e-3, rtol=1e-3
):
    M, N = 128, 128
    A_shape = (M, K)
    B_shape = (N, K)
    C_shape = (M, N)
    A_swizzle, B_swizzle = 3, 3
    A_layout = mma_shared_layout(A_dtype, A_swizzle, A_shape)
    B_layout = mma_shared_layout(B_dtype, B_swizzle, B_shape)
    C_elem_32b = 4 // (tvm.runtime.DataType(C_dtype).bits // 8)
    cols_alloc = max(32, next_power_of_2(N // C_elem_32b))
    total_bytes = functools.reduce(operator.mul, A_shape, 1) * (
        tvm.runtime.DataType(A_dtype).bits // 8
    ) + functools.reduce(operator.mul, B_shape, 1) * (tvm.runtime.DataType(B_dtype).bits // 8)
    instruction_k = 8 if is_AB_tf32 else 32
    kind = "tf32" if is_AB_tf32 else "f8f6f4"
    atom_width = 32 if is_AB_tf32 else 128
    b_format = tma_dtype_B if tma_dtype_B is not None else B_dtype
    a_format = "tf32" if is_AB_tf32 else A_dtype
    mma_b_format = "tf32" if is_AB_tf32 else B_dtype

    @T.function
    def gemm_async(
        A: T.Tensor(A_shape, A_dtype), B: T.Tensor(B_shape, B_dtype), C: T.Tensor(C_shape, C_dtype)
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=4 * 32))
        warp_id = T.cuda.warp_id()
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()
        A_smem = T.alloc_tensor(
            ty_args=[T.Tensor(A_shape, A_dtype, scope="shared", layout=A_layout, align=1024)]
        )
        B_smem = T.alloc_tensor(
            ty_args=[T.Tensor(B_shape, B_dtype, scope="shared", layout=B_layout, align=1024)]
        )
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        tma_mbar = T.alloc_shared([1], "uint64", layout="default")
        mma_mbar = T.alloc_shared([1], "uint64", layout="default")
        if tid_in_wg == 0:
            T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.cuda.cta_sync()
        tmem = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    (128, N),
                    C_dtype,
                    scope="tmem",
                    layout=TileLayout(S[(128, N) : (1 @ TLane, 1 @ TCol)]),
                )
            ],
        )
        if tid_in_wg == 0:
            acopy = A_smem.rearrange("m (kr k) -> kr m k", k=atom_width)
            bcopy = B_smem.rearrange("n (kr k) -> kr n k", k=atom_width)
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                acopy,
                A,
                box=(M, atom_width),
                coords=(0, 0),
                coord_steps=((0, atom_width),),
                swizzle=3,
                mbar=tma_mbar.ptr_to([0]),
            )
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                bcopy,
                B,
                box=(N, atom_width),
                coords=(0, 0),
                coord_steps=((0, atom_width),),
                swizzle=3,
                format=b_format,
                mbar=tma_mbar.ptr_to([0]),
            )
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if tid_in_wg == 0:
            av = A_smem.rearrange("(mr m) (kr k) -> mr kr m k", m=M, k=instruction_k)
            bv = B_smem.rearrange("(nr n) (kr k) -> nr kr n k", n=N, k=instruction_k)
            T.cuda.tile.apply(
                T.ptx[f"tcgen05.mma.cta_group::1.kind::{kind}"],
                tmem.view(1, 1, M, N),
                av,
                bv,
                instruction_shape=(M, N, instruction_k),
                a_format=a_format,
                b_format=mma_b_format,
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                mma_mbar.ptr_to([0])
            )
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(N, dtype=C_dtype, layout="default")
        C_view = C_reg.view(128, N, layout=TileLayout(S[(128, N) : (1 @ axis_tid_in_wg, 1)]))
        if wg_id == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.ld.sync.aligned.32x32b.x128.b32"],
                C_view.view(4, 32, N),
                tmem.view(4, 32, N),
                scope="warpgroup",
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        for col in T.serial(N):
            C[tid_in_wg, col] = C_reg[col]
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))

    dev = tvm.cuda(0)
    np.random.seed(0)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.tirx.compile(tvm.IRModule({"main": gemm_async}), target=target, pipeline="tirx")

    def _rand(shape, dtype):
        f = np.random.randn(*shape).astype("float32")
        return f.astype(dtype) if ml_dtypes is not None or "float8" not in dtype else f

    A_np = _rand(A_shape, A_dtype)
    B_np = _rand(B_shape, B_dtype)
    C_np = np.zeros(C_shape, dtype=C_dtype)
    A_t, B_t, C_t = (tvm.runtime.tensor(x, dev) for x in (A_np, B_np, C_np))
    mod["main"](A_t, B_t, C_t)
    C_ref = A_np.astype("float32") @ B_np.astype("float32").T
    np.testing.assert_allclose(C_t.numpy().astype("float32"), C_ref, atol=atol, rtol=rtol)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.skipif(ml_dtypes is None, reason="Requires ml_dtypes")
def test_gemm_tcgen05_no_swizzle_col_major_a_ws_local_idesc():
    """Keep the FlashMLA packed bytes and make their operand view explicit.

    The original column-major declaration is an alias over packed K-major
    storage. A no-copy view exposes that storage to the fixed PTX pattern.
    The expected instruction descriptor and GPU result remain unchanged.
    """
    M, K, N = 64, 64, 256
    dtype = "bfloat16"
    A_layout = TileLayout(S[(M, K) : (1, M)])  # column-major, unswizzled
    B_layout = mma_shared_layout(dtype, 3, (K, N))

    @T.function
    def gemm_ws(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((128, N // 2), "float32"),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=4 * 32))
        warp_id = T.cuda.warp_id()
        wg_id = T.cuda.warpgroup_id()
        tid_in_wg = T.cuda.thread_in_warpgroup()
        A_smem = T.alloc_tensor(
            ty_args=[T.Tensor((M, K), dtype, scope="shared", layout=A_layout, align=1024)]
        )
        B_smem = T.alloc_tensor(
            ty_args=[T.Tensor((K, N), dtype, scope="shared", layout=B_layout, align=1024)]
        )
        tmem_addr = T.alloc_shared([1], "uint32", layout="default")
        mma_mbar = T.alloc_shared([1], "uint64", layout="default")
        if tid_in_wg == 0:
            T.ptx.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(128)
            )
        T.cuda.cta_sync()
        # M=64 .ws accumulates via datapath E (FlashMLA head64's tmem_o layout):
        # lane = m + 64*(n >= N/2), col = n % (N/2).
        tmem = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    (M, N),
                    "float32",
                    scope="tmem",
                    layout=TileLayout(S[(M, 2, N // 2) : (1 @ TLane, 64 @ TLane, 1 @ TCol)]),
                )
            ],
        )
        # Identity overlay of the physical 128x128 TMEM footprint for readback.
        tmem_ldst = T.cuda.decl_tmem(
            tmem_addr[0],
            ty_args=[
                T.Tensor(
                    (128, N // 2),
                    "float32",
                    scope="tmem",
                    layout=TileLayout(S[(128, N // 2) : (1 @ TLane, 1 @ TCol)]),
                )
            ],
        )
        # Plain generic-proxy stores; A's bytes follow the FlashMLA S-tile ABI
        # (phys = 8*m + 8*M*(k//8) + k%8), col-major A_smem is descriptor fiction.
        for i in range(M * K // 128):
            a_idx = i * 128 + tid_in_wg
            a_m = a_idx % M
            a_k = a_idx // M
            a_phys = 8 * a_m + 8 * M * (a_k // 8) + (a_k % 8)
            A_smem[a_phys % M, a_phys // M] = A[a_m, a_k]
        for i in range(K * N // 128):
            b_idx = i * 128 + tid_in_wg
            B_smem[b_idx // N, b_idx % N] = B[b_idx // N, b_idx % N]
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if tid_in_wg == 0:
            apacked = A_smem.permute(1, 0).view(K // 8, M, 8).permute(1, 0, 2).view(M, K)
            av = apacked.rearrange("(mr m) (kr k) -> mr kr m k", m=M, k=16)
            bv = B_smem.permute(1, 0).rearrange("(nr n) (kr k) -> nr kr n k", n=N, k=16)
            T.cuda.tile.apply(
                T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"],
                tmem.view(1, 1, M, N),
                av,
                bv,
                instruction_shape=(M, N, 16),
                a_major="k",
                b_major="mn",
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                mma_mbar.ptr_to([0])
            )
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(N // 2, dtype="float32", layout="default")
        C_view = C_reg.view(
            128, N // 2, layout=TileLayout(S[(128, N // 2) : (1 @ axis_tid_in_wg, 1)])
        )
        if wg_id == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.ld.sync.aligned.32x32b.x128.b32"],
                C_view.view(4, 32, N // 2),
                tmem_ldst.view(4, 32, N // 2),
                scope="warpgroup",
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        for col in T.serial(N // 2):
            C[tid_in_wg, col] = C_reg[col]
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(128))

    with tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}):
        mod = tvm.tirx.compile(
            tvm.IRModule({"main": gemm_ws}),
            target={"kind": "cuda", "arch": "sm_100a"},
            pipeline="tirx",
        )

    src = mod.mod.imports[0].inspect_source()
    # Descriptor construction: no-swizzle col-major A -> (ldo=64, sdo=8, swizzle=0).
    assert "encode_instr_descriptor" not in src
    assert "tcgen05.mma.ws.cta_group::1.kind::f16" in src
    # idesc literal: M=64, N=256, f32 <- bf16 x bf16, a_major=0 (K-major, bit15
    # clear), b_major=1 — hand-encoded value FlashMLA head64 validated bit-exactly.
    assert str(0x04410490) in src, "expected K-major (bit15=0) idesc literal"
    assert str(0x04418490) not in src, "MN-major idesc (bit15=1) mis-pairs the K-major descA"

    dev = tvm.cuda(0)
    np.random.seed(0)
    A_np = np.random.randn(M, K).astype(ml_dtypes.bfloat16)
    B_np = np.random.randn(K, N).astype(ml_dtypes.bfloat16)
    C_np = np.zeros((128, N // 2), "float32")
    A_t, B_t, C_t = (tvm.runtime.tensor(x, dev) for x in (A_np, B_np, C_np))
    mod["main"](A_t, B_t, C_t)
    C_ref = A_np.astype("float32") @ B_np.astype("float32")
    C_out = C_t.numpy()
    # Datapath E: lanes 0-63 hold columns [0, N/2), lanes 64-127 hold [N/2, N).
    np.testing.assert_allclose(C_out[:M], C_ref[:, : N // 2], atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(C_out[M:], C_ref[:, N // 2 :], atol=1e-2, rtol=1e-2)
