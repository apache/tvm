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
# pylint: disable=missing-function-docstring
import copy
import functools
import operator

import numpy as np
import pytest

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None

import tvm
import tvm.testing
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx.cuda.operator.tile_primitive.gemm_async import sf_tmem_layout
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import (
    mma_atom_layout,
    mma_atom_shape,
    mma_shared_layout,
)
from tvm.tirx.layout import S, TCol, TileLayout, TLane, tcgen05_atom_layout
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def next_power_of_2(x):
    """Return the smallest power of 2 greater than or equal to x."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


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
    This is achieved by composing the SwizzleLayout with a stride-reversed TileLayout.
    """
    from tvm.tirx.layout import ComposeLayout

    swizzle_atom = mma_atom_layout(dtype, swizzle_mode)
    base_shape = mma_atom_shape(dtype, swizzle_mode)  # 2D: [8, T*s]
    swapped = [base_shape[1], base_shape[0]]  # [T*s, 8]
    # Stride-reversed tile: first dim (T*s) contiguous, second dim (8) has stride T*s
    mn_tile = TileLayout(S[tuple(swapped) : (1, swapped[0])])
    mn_atom = ComposeLayout(swizzle_atom, mn_tile)
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


def cta_split_dim(trans):
    """Return the axis index that is split across CTAs in a cta_group=2 setup."""
    return -1 if trans else -2


def get_shape_per_cta(shape, trans):
    """Halve the split dimension for per-CTA shapes (cta_group=2)."""
    shape_per_cta = copy.deepcopy(list(shape))
    shape_per_cta[cta_split_dim(trans)] //= 2
    return shape_per_cta


def get_global_region(shape, trans, cbx):
    """Return the global memory region for CTA *cbx* (cta_group=2)."""
    r = list(slice(0, shape[i]) for i in range(len(shape)))
    d = cta_split_dim(trans)
    r[d] = slice(cbx * shape[d], (cbx + 1) * shape[d])
    return r


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

    # fmt: off
    @T.prim_func
    def gemm_async(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, A_shape, A_dtype)
        B = T.match_buffer(B_ptr, B_shape, B_dtype)
        C = T.match_buffer(C_ptr, C_shape, C_dtype)

        T.device_entry()
        warp_id = T.warp_id([(1) * 4])
        cta_id = T.cta_id([1])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])

        A_smem = T.alloc_buffer(A_shape, A_dtype, scope="shared", layout=A_layout)
        B_smem = T.alloc_buffer(B_shape, B_dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        tma_mbar = T.alloc_shared([1], "uint64")
        mma_mbar = T.alloc_shared([1], "uint64")

        if tid_in_wg == 0:
            T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=cols_alloc, cta_group=1)
        T.cuda.cta_sync()
        tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501

        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem[tuple(r_gmem_A)], A[tuple(r_gmem_A)], **tma_args)
            Tx.copy_async(B_smem[tuple(r_gmem_B)], B[tuple(r_gmem_B)], **tma_args)
            T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), total_bytes)
        T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        if tid_in_wg == 0:
            Tx.gemm_async(tmem[tuple(r_tmem_C)], A_smem[tuple(r_smem_A)], B_smem[tuple(r_smem_B)], dispatch="tcgen05")  # noqa: E501
            T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=1)
        T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        T.ptx.tcgen05.fence.after_thread_sync()
        C_reg = T.alloc_local(width, dtype=C_dtype)
        C_view = C_reg.view(128, width, layout=TileLayout(S[(128, width) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[tuple(r_tmem_C)])
            T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, C_region[1][0]:C_region[1][1]], C_reg[:])

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=cols_alloc, cta_group=1)
        # fmt: on

    np.random.seed(0)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": gemm_async})
        # mod.show()
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
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
    """M=64 MMA with C operand allocated as Layout F (datapath="F").

    Exercises the new ``gemm_async`` path that accepts C buffers tagged
    Layout F — written by an M=64 MMA in their canonical scattered
    row->lane mapping (PTX ISA §9.7.16.10.5), read back via the
    ``.16x256b`` M=64 atom (one PTX issue covering all 64 logical rows
    densely). Without the dispatch change this kernel fails to compile
    because the C-operand layout check asserts Layout D identity.
    """
    M, N, K = 64, 64, 64
    A_dtype, B_dtype, C_dtype = "float16", "float16", "float32"
    A_shape, B_shape, C_shape = (M, K), (N, K), (M, N)
    A_layout = mma_shared_layout(A_dtype, 3, A_shape)
    B_layout = mma_shared_layout(B_dtype, 3, B_shape)

    # The C TMEM buffer carries Layout F over its full (64, N) shape; that's
    # what gemm_async structurally matches against to accept the M=64 write.
    from tvm.tirx.layout import tmem_datapath_layout

    c_layout = tmem_datapath_layout("F", 64, N)

    # fmt: off
    @T.prim_func
    def gemm_layout_f(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, A_shape, A_dtype)
        B = T.match_buffer(B_ptr, B_shape, B_dtype)
        C = T.match_buffer(C_ptr, C_shape, C_dtype)

        T.device_entry()
        warp_id = T.warp_id([4])
        cta_id  = T.cta_id([1])
        wg_id   = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])
        lane_id = T.lane_id([32])

        A_smem = T.alloc_buffer(A_shape, A_dtype, scope="shared", layout=A_layout)
        B_smem = T.alloc_buffer(B_shape, B_dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        tma_mbar = T.alloc_shared([1], "uint64")
        mma_mbar = T.alloc_shared([1], "uint64")

        if tid_in_wg == 0:
            T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=64, cta_group=1)
        T.cuda.cta_sync()
        # Layout F C operand — the path under test.
        tmem = T.decl_buffer((64, N), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=c_layout)  # noqa: E501

        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem[:, :], A[:, :], **tma_args)
            Tx.copy_async(B_smem[:, :], B[:, :], **tma_args)
            T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), (M * K + N * K) * 2)
        T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        if tid_in_wg == 0:
            Tx.gemm_async(tmem[0:64, 0:N], A_smem[:, :], B_smem[:, :], dispatch="tcgen05")
            T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=1)
        T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence.after_thread_sync()

        # Read back via .16x256b M=64 (the canonical pairing).
        reg = T.alloc_local(32, dtype="float32")
        reg_view = reg.view(64, N, layout=tcgen05_atom_layout("16x256b", (64, N), "float32"))
        if wg_id == 0:
            Tx.wg.copy_async(reg_view[:, :], tmem[0:64, 0:N])
            T.ptx.tcgen05.wait.ld()
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
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=64, cta_group=1)
    # fmt: on

    np.random.seed(0)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": gemm_layout_f}), target=target, tir_pipeline="tirx")

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

    # fmt: off
    @T.prim_func
    def gemm_async(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, A_shape, A_dtype)
        B = T.match_buffer(B_ptr, B_shape, B_dtype)
        C = T.match_buffer(C_ptr, C_shape, C_dtype)

        T.device_entry()
        warp_id = T.warp_id([(1) * 4])
        cbx, cby = T.cta_id_in_cluster([2, 1])
        cta_id = T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])

        A_smem = T.alloc_buffer(A_shape_per_cta, A_dtype, scope="shared", layout=A_layout)
        B_smem = T.alloc_buffer(B_shape_per_cta, B_dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        tma_mbar = T.alloc_shared([1], "uint64")
        mma_mbar = T.alloc_shared([1], "uint64")

        ptr: T.let[T.Var(name_hint="ptr", ty=PointerType(PrimType("uint64")))] = T.reinterpret("handle", T.ptx.map_shared_rank(tma_mbar.ptr_to([0]), 0))  # noqa: E501
        tma_mbar_cta_0 = T.decl_buffer([1], "uint64", data=ptr, scope="shared")

        if tid_in_wg == 0:
            T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)

        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=cols_alloc, cta_group=2)
        tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
        T.ptx.fence.mbarrier_init()
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        T.cuda.cluster_sync()

        tma_args = T.meta_var({"dispatch": "tma", "mbar": tma_mbar_cta_0.ptr_to([0]), "cta_group": 2})  # noqa: E501
        if tid_in_wg == 0:
            Tx.copy_async(A_smem[tuple(r_smem_A_in)], A[tuple(get_global_region(A_shape_per_cta, transA, cbx))], **tma_args)  # noqa: E501
            Tx.copy_async(B_smem[tuple(r_smem_B_in)], B[tuple(get_global_region(B_shape_per_cta, transB, cbx))], **tma_args)  # noqa: E501
            if cbx == 0:
                T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), total_bytes)

        if cbx == 0:
            T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
            T.ptx.tcgen05.fence.after_thread_sync()
            T.cuda.cta_sync()
            if tid_in_wg == 0:
                Tx.gemm_async(tmem[tuple(r_tmem_C)], A_smem[tuple(r_smem_A)], B_smem[tuple(r_smem_B)], dispatch="tcgen05", cta_group=2)  # noqa: E501
                T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=2, cta_mask=3) # signal cta 1's mbarrier  # noqa: E501
        T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0) # both cta 0 and cta 1 have done mma
        T.ptx.tcgen05.fence.after_thread_sync()
        T.cuda.cta_sync()

        C_reg = T.alloc_local(width , dtype=C_dtype)
        C_view = C_reg.view(128, width, layout=TileLayout(S[(128, width) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[C_region[0][0]:C_region[0][1], C_region[1][0]:C_region[1][0] + width])  # noqa: E501
            T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.copy(C[cbx * 128 +tid_in_wg, C_region[1][0]:C_region[1][0] + width], C_reg[:])
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=cols_alloc, cta_group=2)
        # fmt: on

    np.random.seed(0)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": gemm_async})
        mod.show()
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
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
    Physical readback via a (128, N//2) buffer aliasing the same TMEM allocation.
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

    # fmt: off
    @T.prim_func
    def gemm_async(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (M_per_cta * 2, K), A_dtype)
        B = T.match_buffer(B_ptr, (N_logical, K), B_dtype)
        C = T.match_buffer(C_ptr, C_shape, C_dtype)

        T.device_entry()
        warp_id = T.warp_id([(1) * 4])
        cbx, cby = T.cta_id_in_cluster([2, 1])
        cta_id = T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])

        A_smem = T.alloc_buffer(A_shape, A_dtype, scope="shared", layout=A_layout)
        B_smem = T.alloc_buffer(B_shape, B_dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        tma_mbar = T.alloc_shared([1], "uint64")
        mma_mbar = T.alloc_shared([1], "uint64")

        ptr: T.let[T.Var(name_hint="ptr", ty=PointerType(PrimType("uint64")))] = T.reinterpret("handle", T.ptx.map_shared_rank(tma_mbar.ptr_to([0]), 0))  # noqa: E501
        tma_mbar_cta_0 = T.decl_buffer([1], "uint64", data=ptr, scope="shared")

        if tid_in_wg == 0:
            T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)

        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=cols_alloc, cta_group=2)
        tmem = T.decl_buffer((M_per_cta, N_logical), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(M_per_cta, 2, N_half) : (1 @ TLane, 64 @ TLane, 1 @ TCol)]))  # noqa: E501
                # Physical TMEM view for readback: (128, N_half) standard layout
        tmem_phys = T.decl_buffer((128, N_half), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, N_half) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
        T.ptx.fence.mbarrier_init()
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        T.cuda.cluster_sync()

        tma_args = T.meta_var({"dispatch": "tma", "mbar": tma_mbar_cta_0.ptr_to([0]), "cta_group": 2})  # noqa: E501
        if tid_in_wg == 0:
                    # CTA cbx loads its portion of A and B
            Tx.copy_async(A_smem[0:M_per_cta, 0:K], A[cbx * M_per_cta:(cbx + 1) * M_per_cta, 0:K], **tma_args)  # noqa: E501
            Tx.copy_async(B_smem[0:N_half, 0:K], B[cbx * N_half:(cbx + 1) * N_half, 0:K], **tma_args)  # noqa: E501
            if cbx == 0:
                T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), total_bytes)

        if cbx == 0:
            T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
            T.ptx.tcgen05.fence.after_thread_sync()
            T.cuda.cta_sync()
            if tid_in_wg == 0:
                Tx.gemm_async(tmem[0:M_per_cta, 0:N_logical], A_smem[0:M_per_cta, 0:K], B_smem[0:N_half, 0:K], dispatch="tcgen05", cta_group=2)  # noqa: E501
                T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=2, cta_mask=3)
        T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0)
        T.ptx.tcgen05.fence.after_thread_sync()
        T.cuda.cta_sync()

                # Readback from physical TMEM view (128 rows x N_half cols)
                # Warps 0,1 (rows 0-63): first N half for M rows 0-63
                # Warps 2,3 (rows 64-127): second N half for M rows 0-63
        C_reg = T.alloc_local(N_half, dtype=C_dtype)
        C_view = C_reg.view(128, N_half, layout=TileLayout(S[(128, N_half) : (1 @ axis_tid_in_wg, 1)]))  # noqa: E501
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem_phys[0:128, 0:N_half])
            T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        n_off = (tid_in_wg // 64) * N_half
        Tx.copy(C[cbx * M_per_cta + tid_in_wg % 64, n_off : n_off + N_half], C_reg[:])
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=cols_alloc, cta_group=2)
        # fmt: on

    np.random.seed(0)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": gemm_async})
        mod.show()
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

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
    """Test block-scaled fp8 GEMM with cta_group=1 using gemm_async op.

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
    sfa_layout = sf_tmem_layout(M, SF_K=sf_mma_k * 1, sf_per_mma=sf_mma_k)
    sfb_layout = sf_tmem_layout(N, SF_K=sf_mma_k * 1, sf_per_mma=sf_mma_k)
    sf_epc = 32 // tvm.runtime.DataType(SF_dtype).bits
    SFA_TMEM_SPACING = (int(sfa_layout.span("TCol")) + sf_epc - 1) // sf_epc
    SFA_TMEM_START = width
    SFB_TMEM_START = SFA_TMEM_START + SFA_TMEM_SPACING

    F32_BYTES = 4
    F128_BYTES = 16
    SF_smem_layout = TileLayout(S[(4, 32) : (32, 1)])
    SF_smem_post_layout = TileLayout(S[(4, 32) : (1, 4)])

    # fmt: off
    @T.prim_func
    def gemm_async_fn(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle, SFA_ptr: T.handle, SFB_ptr: T.handle) -> None:  # noqa: E501
        A = T.match_buffer(A_ptr, A_shape, A_dtype)
        B = T.match_buffer(B_ptr, B_shape, B_dtype)
        C = T.match_buffer(C_ptr, C_shape, C_dtype)
        SFA_in = T.match_buffer(SFA_ptr, (128,), "uint32")
        SFB_in = T.match_buffer(SFB_ptr, (128,), "uint32")

        T.device_entry()
        warp_id = T.warp_id([(1) * 4])
        cta_id = T.cta_id([1])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])

        A_smem = T.alloc_buffer(A_shape, A_dtype, scope="shared", layout=A_layout)
        B_smem = T.alloc_buffer(B_shape, B_dtype, scope="shared", layout=B_layout)
        SFA_smem = T.alloc_buffer((4, 32), "uint32", scope="shared", layout=SF_smem_layout)
        SFB_smem = T.alloc_buffer((4, 32), "uint32", scope="shared", layout=SF_smem_layout)
        SFA_smem_post = SFA_smem.view(4, 32, layout=SF_smem_post_layout)
        SFB_smem_post = SFB_smem.view(4, 32, layout=SF_smem_post_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        tma_mbar = T.alloc_shared([1], "uint64")
        mma_mbar = T.alloc_shared([1], "uint64")
        descSFA = T.alloc_buffer((1,), "uint64", scope="local")
        descSFB = T.alloc_buffer((1,), "uint64", scope="local")

        if tid_in_wg == 0:
            T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=cols_alloc, cta_group=1)
        T.cuda.cta_sync()

        tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
        sfa_tmem = T.decl_buffer((M, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFA_TMEM_START, layout=sfa_layout)  # noqa: E501
        sfb_tmem = T.decl_buffer((N, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFB_TMEM_START, layout=sfb_layout)  # noqa: E501

                # TMA load A and B from global to shared
        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem[tuple(r_gmem_A)], A[tuple(r_gmem_A)], **tma_args)
            Tx.copy_async(B_smem[tuple(r_gmem_B)], B[tuple(r_gmem_B)], **tma_args)
            T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), total_bytes)
        T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        SFA_smem[tid_in_wg // 32, tid_in_wg % 32] = SFA_in[tid_in_wg]
        SFB_smem[tid_in_wg // 32, tid_in_wg % 32] = SFB_in[tid_in_wg]
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

                # Transpose scale factors in shared memory
        if warp_id == 0:
            Tx.warp.permute_layout(SFA_smem_post[:, :], SFA_smem[:, :])
            Tx.warp.permute_layout(SFB_smem_post[:, :], SFB_smem[:, :])
        T.cuda.cta_sync()

                # Copy SFA/SFB from shared to TMEM via tcgen05.cp, then issue MMA
        if tid_in_wg == 0:
            T.ptx.tcgen05.encode_matrix_descriptor(descSFA.data, SFA_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptx.tcgen05.cp(SFA_TMEM_START, descSFA[0], shape="32x128b", cta_group=1, multicast="warpx4")  # noqa: E501
            T.ptx.tcgen05.encode_matrix_descriptor(descSFB.data, SFB_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptx.tcgen05.cp(SFB_TMEM_START, descSFB[0], shape="32x128b", cta_group=1, multicast="warpx4")  # noqa: E501

            Tx.gemm_async(tmem[tuple(r_tmem_C)], A_smem[tuple(r_smem_A)], B_smem[tuple(r_smem_B)], SFA=sfa_tmem[0:M, 0:sf_mma_k], SFB=sfb_tmem[0:N, 0:sf_mma_k], dispatch="tcgen05")  # noqa: E501
            T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=1)
        T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

                # Copy result from tmem to global
        T.ptx.tcgen05.fence.after_thread_sync()
        C_reg = T.alloc_local(width, dtype=C_dtype)
        C_view = C_reg.view(128, width, layout=TileLayout(S[(128, width) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[tuple(r_tmem_C)])
            T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, C_region[1][0]:C_region[1][1]], C_reg[:])

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=cols_alloc, cta_group=1)
        # fmt: on

    np.random.seed(0)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": gemm_async_fn})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

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
    """Test block-scaled fp8 GEMM with cta_group=2 using gemm_async op.

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
    sf_layout = sf_tmem_layout(128, SF_K=sf_mma_k * 1, sf_per_mma=sf_mma_k)
    sf_epc = 32 // tvm.runtime.DataType(SF_dtype).bits
    SF_TMEM_SPACING = (int(sf_layout.span("TCol")) + sf_epc - 1) // sf_epc
    N_cols = C_region[1][1] - C_region[1][0]
    SFA_TMEM_START = N_cols
    SFB_TMEM_START = SFA_TMEM_START + SF_TMEM_SPACING

    F32_BYTES = 4
    F128_BYTES = 16
    SF_smem_layout = TileLayout(S[(4, 32) : (32, 1)])
    SF_smem_post_layout = TileLayout(S[(4, 32) : (1, 4)])

    # fmt: off
    @T.prim_func
    def gemm_async_fn(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle, SFA_ptr: T.handle, SFB_ptr: T.handle) -> None:  # noqa: E501
        A = T.match_buffer(A_ptr, A_shape, A_dtype)
        B = T.match_buffer(B_ptr, B_shape, B_dtype)
        C = T.match_buffer(C_ptr, C_shape, C_dtype)
        SFA_in = T.match_buffer(SFA_ptr, (M_total,), "uint32")
        SFB_in = T.match_buffer(SFB_ptr, (128,), "uint32")

        T.device_entry()
        warp_id = T.warp_id([(1) * 4])
        cbx, cby = T.cta_id_in_cluster([2, 1])
        cta_id = T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])

        A_smem = T.alloc_buffer(A_shape_per_cta, A_dtype, scope="shared", layout=A_layout)
        B_smem = T.alloc_buffer(B_shape_per_cta, B_dtype, scope="shared", layout=B_layout)
        SFA_smem = T.alloc_buffer((4, 32), "uint32", scope="shared", layout=SF_smem_layout)
        SFB_smem = T.alloc_buffer((4, 32), "uint32", scope="shared", layout=SF_smem_layout)
        SFA_smem_post = SFA_smem.view(4, 32, layout=SF_smem_post_layout)
        SFB_smem_post = SFB_smem.view(4, 32, layout=SF_smem_post_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        tma_mbar = T.alloc_shared([1], "uint64")
        mma_mbar = T.alloc_shared([1], "uint64")
        descSFA = T.alloc_buffer((1,), "uint64", scope="local")
        descSFB = T.alloc_buffer((1,), "uint64", scope="local")

        ptr: T.let[T.Var(name_hint="ptr", ty=PointerType(PrimType("uint64")))] = T.reinterpret("handle", T.ptx.map_shared_rank(tma_mbar.ptr_to([0]), 0))  # noqa: E501
        tma_mbar_cta_0 = T.decl_buffer([1], "uint64", data=ptr, scope="shared")

        if tid_in_wg == 0:
            T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)

        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=cols_alloc, cta_group=2)
        tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501

        sfa_tmem = T.decl_buffer((128, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFA_TMEM_START, layout=sf_layout)  # noqa: E501
        sfb_tmem = T.decl_buffer((128, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFB_TMEM_START, layout=sf_layout)  # noqa: E501

        T.ptx.fence.mbarrier_init()
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        T.cuda.cluster_sync()

                # TMA load A and B (both CTAs issue with multicast)
        tma_args = T.meta_var({"dispatch": "tma", "mbar": tma_mbar_cta_0.ptr_to([0]), "cta_group": 2})  # noqa: E501
        if tid_in_wg == 0:
            Tx.copy_async(A_smem[tuple(r_smem_A_in)], A[tuple(get_global_region(A_shape_per_cta, transA, cbx))], **tma_args)  # noqa: E501
            Tx.copy_async(B_smem[tuple(r_smem_B_in)], B[tuple(get_global_region(B_shape_per_cta, transB, cbx))], **tma_args)  # noqa: E501
            if cbx == 0:
                T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), total_bytes)
        SFA_smem[tid_in_wg // 32, tid_in_wg % 32] = SFA_in[cbx * 128 + tid_in_wg]
        SFB_smem[tid_in_wg // 32, tid_in_wg % 32] = SFB_in[tid_in_wg]
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

                # Transpose scale factors (both CTAs)
        if warp_id == 0:
            Tx.warp.permute_layout(SFA_smem_post[:, :], SFA_smem[:, :])
            Tx.warp.permute_layout(SFB_smem_post[:, :], SFB_smem[:, :])
        T.cuda.cta_sync()

                # Copy SFA/SFB from shared to TMEM via tcgen05.cp (both CTAs, cta_group=2)
        if tid_in_wg == 0:
            T.ptx.tcgen05.encode_matrix_descriptor(descSFA.data, SFA_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptx.tcgen05.cp(SFA_TMEM_START, descSFA[0], shape="32x128b", cta_group=2, multicast="warpx4")  # noqa: E501
            T.ptx.tcgen05.encode_matrix_descriptor(descSFB.data, SFB_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptx.tcgen05.cp(SFB_TMEM_START, descSFB[0], shape="32x128b", cta_group=2, multicast="warpx4")  # noqa: E501
        T.cuda.cta_sync()
        T.cuda.cluster_sync()

        if cbx == 0:
            T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
            T.ptx.tcgen05.fence.after_thread_sync()
            T.cuda.cta_sync()
            if tid_in_wg == 0:
                Tx.gemm_async(tmem[tuple(r_tmem_C)], A_smem[tuple(r_smem_A)], B_smem[tuple(r_smem_B)], SFA=sfa_tmem[0:128, 0:sf_mma_k], SFB=sfb_tmem[0:128, 0:sf_mma_k], dispatch="tcgen05", cta_group=2)  # noqa: E501
                T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=2, cta_mask=3)
        T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0)
        T.ptx.tcgen05.fence.after_thread_sync()
        T.cuda.cta_sync()

                # Copy result from tmem to global
        C_reg = T.alloc_local(width, dtype=C_dtype)
        C_view = C_reg.view(128, width, layout=TileLayout(S[(128, width) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[C_region[0][0]:C_region[0][1], C_region[1][0]:C_region[1][0] + width])  # noqa: E501
            T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.copy(C[cbx * 128 + tid_in_wg, C_region[1][0]:C_region[1][0] + width], C_reg[:])
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=cols_alloc, cta_group=2)
        # fmt: on

    np.random.seed(0)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": gemm_async_fn})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

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
    sfa_layout = sf_tmem_layout(M, SF_K=sf_mma_k * 1, sf_per_mma=sf_mma_k)
    sfb_layout = sf_tmem_layout(N, SF_K=sf_mma_k * 1, sf_per_mma=sf_mma_k)
    sf_epc = 32 // tvm.runtime.DataType(SF_dtype).bits
    SFA_TMEM_SPACING = (int(sfa_layout.span("TCol")) + sf_epc - 1) // sf_epc
    SFA_TMEM_START = width
    SFB_TMEM_START = SFA_TMEM_START + SFA_TMEM_SPACING

    F32_BYTES = 4
    F128_BYTES = 16
    SF_smem_layout = TileLayout(S[(4, 32) : (32, 1)])
    SF_smem_post_layout = TileLayout(S[(4, 32) : (1, 4)])

    # fmt: off
    @T.prim_func
    def gemm_async_fn(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle, SFA_ptr: T.handle, SFB_ptr: T.handle) -> None:  # noqa: E501
        A_packed = T.match_buffer(A_ptr, A_packed_shape, "uint8")
        B_packed = T.match_buffer(B_ptr, B_packed_shape, "uint8")
        C = T.match_buffer(C_ptr, C_shape, C_dtype)
        SFA_in = T.match_buffer(SFA_ptr, (128,), "uint32")
        SFB_in = T.match_buffer(SFB_ptr, (128,), "uint32")

        T.device_entry()
        warp_id = T.warp_id([(1) * 4])
        cta_id = T.cta_id([1])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])

        A_smem_packed = T.alloc_buffer(A_packed_shape, "uint8", scope="shared", layout=A_uint8_layout)  # noqa: E501
        B_smem_packed = T.alloc_buffer(B_packed_shape, "uint8", scope="shared", layout=B_uint8_layout)  # noqa: E501
        A_smem = T.decl_buffer(A_fp4_shape, "float4_e2m1fn", data=A_smem_packed.data, scope="shared", layout=A_fp4_layout)  # noqa: E501
        B_smem = T.decl_buffer(B_fp4_shape, "float4_e2m1fn", data=B_smem_packed.data, scope="shared", layout=B_fp4_layout)  # noqa: E501

        SFA_smem = T.alloc_buffer((4, 32), "uint32", scope="shared", layout=SF_smem_layout)
        SFB_smem = T.alloc_buffer((4, 32), "uint32", scope="shared", layout=SF_smem_layout)
        SFA_smem_post = SFA_smem.view(4, 32, layout=SF_smem_post_layout)
        SFB_smem_post = SFB_smem.view(4, 32, layout=SF_smem_post_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        tma_mbar = T.alloc_shared([1], "uint64")
        mma_mbar = T.alloc_shared([1], "uint64")
        descSFA = T.alloc_buffer((1,), "uint64", scope="local")
        descSFB = T.alloc_buffer((1,), "uint64", scope="local")

        if tid_in_wg == 0:
            T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=cols_alloc, cta_group=1)
        T.cuda.cta_sync()

        tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
        sfa_tmem = T.decl_buffer((M, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFA_TMEM_START, layout=sfa_layout)  # noqa: E501
        sfb_tmem = T.decl_buffer((N, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFB_TMEM_START, layout=sfb_layout)  # noqa: E501

                # TMA load A and B as uint8
        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem_packed[:, :], A_packed[:, :], **tma_args)
            Tx.copy_async(B_smem_packed[:, :], B_packed[:, :], **tma_args)
            T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), total_bytes)
        T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        SFA_smem[tid_in_wg // 32, tid_in_wg % 32] = SFA_in[tid_in_wg]
        SFB_smem[tid_in_wg // 32, tid_in_wg % 32] = SFB_in[tid_in_wg]
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

                # Transpose scale factors in shared memory
        if warp_id == 0:
            Tx.warp.permute_layout(SFA_smem_post[:, :], SFA_smem[:, :])
            Tx.warp.permute_layout(SFB_smem_post[:, :], SFB_smem[:, :])
        T.cuda.cta_sync()

                # Copy SFA/SFB from shared to TMEM via tcgen05.cp, then issue MMA
        if tid_in_wg == 0:
            T.ptx.tcgen05.encode_matrix_descriptor(descSFA.data, SFA_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptx.tcgen05.cp(SFA_TMEM_START, descSFA[0], shape="32x128b", cta_group=1, multicast="warpx4")  # noqa: E501
            T.ptx.tcgen05.encode_matrix_descriptor(descSFB.data, SFB_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptx.tcgen05.cp(SFB_TMEM_START, descSFB[0], shape="32x128b", cta_group=1, multicast="warpx4")  # noqa: E501

            Tx.gemm_async(tmem[0:128, 0:N], A_smem[:, :], B_smem[:, :], SFA=sfa_tmem[0:M, 0:sf_mma_k], SFB=sfb_tmem[0:N, 0:sf_mma_k], dispatch="tcgen05")  # noqa: E501
            T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=1)
        T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

                # Copy result from tmem to global
        T.ptx.tcgen05.fence.after_thread_sync()
        C_reg = T.alloc_local(width, dtype=C_dtype)
        C_view = C_reg.view(128, width, layout=TileLayout(S[(128, width) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[0:128, 0:N])
            T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, 0:N], C_reg[:])

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=cols_alloc, cta_group=1)
        # fmt: on

    np.random.seed(0)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": gemm_async_fn})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

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
    sfa_layout = sf_tmem_layout(M_per_cta, SF_K=sf_mma_k * 1, sf_per_mma=sf_mma_k)
    sfb_layout = sf_tmem_layout(N_total, SF_K=sf_mma_k * 1, sf_per_mma=sf_mma_k)
    sf_epc = 32 // tvm.runtime.DataType(SF_dtype).bits
    SFA_TMEM_SPACING = (int(sfa_layout.span("TCol")) + sf_epc - 1) // sf_epc
    (int(sfb_layout.span("TCol")) + sf_epc - 1) // sf_epc
    SFA_TMEM_START = width
    SFB_TMEM_START = SFA_TMEM_START + SFA_TMEM_SPACING

    F32_BYTES = 4
    F128_BYTES = 16
    SF_smem_layout = TileLayout(S[(4, 32) : (32, 1)])
    SF_smem_post_layout = TileLayout(S[(4, 32) : (1, 4)])

    # fmt: off
    @T.prim_func
    def gemm_async_fn(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle, SFA_ptr: T.handle, SFB_ptr: T.handle) -> None:  # noqa: E501
        A_packed = T.match_buffer(A_ptr, A_packed_shape, "uint8")
        B_packed = T.match_buffer(B_ptr, B_packed_shape, "uint8")
        C = T.match_buffer(C_ptr, C_shape, C_dtype)
        SFA_in = T.match_buffer(SFA_ptr, (M_total,), "uint32")
        SFB_in = T.match_buffer(SFB_ptr, (128,), "uint32")

        T.device_entry()
        warp_id = T.warp_id([(1) * 4])
        cbx, cby = T.cta_id_in_cluster([2, 1])
        cta_id = T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])

        A_smem_packed = T.alloc_buffer(A_packed_per_cta, "uint8", scope="shared", layout=A_uint8_layout)  # noqa: E501
        B_smem_packed = T.alloc_buffer(B_packed_per_cta, "uint8", scope="shared", layout=B_uint8_layout)  # noqa: E501
        A_smem = T.decl_buffer(A_fp4_per_cta, "float4_e2m1fn", data=A_smem_packed.data, scope="shared", layout=A_fp4_layout)  # noqa: E501
        B_smem = T.decl_buffer(B_fp4_per_cta, "float4_e2m1fn", data=B_smem_packed.data, scope="shared", layout=B_fp4_layout)  # noqa: E501

        SFA_smem = T.alloc_buffer((4, 32), "uint32", scope="shared", layout=SF_smem_layout)
        SFB_smem = T.alloc_buffer((4, 32), "uint32", scope="shared", layout=SF_smem_layout)
        SFA_smem_post = SFA_smem.view(4, 32, layout=SF_smem_post_layout)
        SFB_smem_post = SFB_smem.view(4, 32, layout=SF_smem_post_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        tma_mbar = T.alloc_shared([1], "uint64")
        mma_mbar = T.alloc_shared([1], "uint64")
        descSFA = T.alloc_buffer((1,), "uint64", scope="local")
        descSFB = T.alloc_buffer((1,), "uint64", scope="local")

        ptr: T.let[T.Var(name_hint="ptr", ty=PointerType(PrimType("uint64")))] = T.reinterpret("handle", T.ptx.map_shared_rank(tma_mbar.ptr_to([0]), 0))  # noqa: E501
        tma_mbar_cta_0 = T.decl_buffer([1], "uint64", data=ptr, scope="shared")

        if tid_in_wg == 0:
            T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)

        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=cols_alloc, cta_group=2)
        tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501

        sfa_tmem = T.decl_buffer((M_per_cta, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFA_TMEM_START, layout=sfa_layout)  # noqa: E501
        sfb_tmem = T.decl_buffer((N_total, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFB_TMEM_START, layout=sfb_layout)  # noqa: E501

        T.ptx.fence.mbarrier_init()
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        T.cuda.cluster_sync()

                # TMA load A and B with multicast (each CTA loads its portion)
        tma_args = T.meta_var({"dispatch": "tma", "mbar": tma_mbar_cta_0.ptr_to([0]), "cta_group": 2})  # noqa: E501
        if tid_in_wg == 0:
            Tx.copy_async(A_smem_packed[:, :], A_packed[cbx * M_per_cta:(cbx + 1) * M_per_cta, :], **tma_args)  # noqa: E501
            Tx.copy_async(B_smem_packed[:, :], B_packed[cbx * N_per_cta:(cbx + 1) * N_per_cta, :], **tma_args)  # noqa: E501
            if cbx == 0:
                T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), total_bytes)
        SFA_smem[tid_in_wg // 32, tid_in_wg % 32] = SFA_in[cbx * M_per_cta + tid_in_wg]
        SFB_smem[tid_in_wg // 32, tid_in_wg % 32] = SFB_in[tid_in_wg]
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

                # Transpose scale factors
        if warp_id == 0:
            Tx.warp.permute_layout(SFA_smem_post[:, :], SFA_smem[:, :])
            Tx.warp.permute_layout(SFB_smem_post[:, :], SFB_smem[:, :])
        T.cuda.cta_sync()

                # Copy SFA/SFB from shared to TMEM via tcgen05.cp
        if tid_in_wg == 0:
            T.ptx.tcgen05.encode_matrix_descriptor(descSFA.data, SFA_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptx.tcgen05.cp(SFA_TMEM_START, descSFA[0], shape="32x128b", cta_group=2, multicast="warpx4")  # noqa: E501
            T.ptx.tcgen05.encode_matrix_descriptor(descSFB.data, SFB_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptx.tcgen05.cp(SFB_TMEM_START, descSFB[0], shape="32x128b", cta_group=2, multicast="warpx4")  # noqa: E501
        T.cuda.cta_sync()
        T.cuda.cluster_sync()

        if cbx == 0:
            T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
            T.ptx.tcgen05.fence.after_thread_sync()
            T.cuda.cta_sync()
            if tid_in_wg == 0:
                Tx.gemm_async(tmem[0:128, 0:N_total], A_smem[:, :], B_smem[:, :], SFA=sfa_tmem[0:128, 0:sf_mma_k], SFB=sfb_tmem[0:N_total, 0:sf_mma_k], dispatch="tcgen05", cta_group=2)  # noqa: E501
                T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=2, cta_mask=3)
        T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0)
        T.ptx.tcgen05.fence.after_thread_sync()
        T.cuda.cta_sync()

                # Copy result from tmem to global
        C_reg = T.alloc_local(width, dtype=C_dtype)
        C_view = C_reg.view(128, width, layout=TileLayout(S[(128, width) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[0:128, 0:width])
            T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.copy(C[cbx * M_per_cta + tid_in_wg, 0:width], C_reg[:])
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=cols_alloc, cta_group=2)
        # fmt: on

    np.random.seed(0)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": gemm_async_fn})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

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
    """Test sf_id auto-derivation from layout for fp8 block-scaled MMA.

    Per-block quantization (block_size=32) with 4 K-blocks per row, each
    with a different scale factor. The 4 scales are packed into different
    bytes of the uint32 TMEM column. The schedule auto-derives sf_id=0,1,2,3
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
    sfa_layout = sf_tmem_layout(M, SF_K=sf_mma_k * num_ki, sf_per_mma=sf_mma_k)
    sfb_layout = sf_tmem_layout(N, SF_K=sf_mma_k * num_ki, sf_per_mma=sf_mma_k)
    sf_epc = 32 // tvm.runtime.DataType(SF_dtype).bits
    SFA_TMEM_SPACING = (int(sfa_layout.span("TCol")) + sf_epc - 1) // sf_epc
    SFA_TMEM_START = N
    SFB_TMEM_START = SFA_TMEM_START + SFA_TMEM_SPACING

    F32_BYTES = 4
    F128_BYTES = 16
    SF_smem_layout = TileLayout(S[(4, 32) : (32, 1)])
    SF_smem_post_layout = TileLayout(S[(4, 32) : (1, 4)])

    # fmt: off
    @T.prim_func
    def gemm_async_fn(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle, SFA_ptr: T.handle, SFB_ptr: T.handle) -> None:  # noqa: E501
        A = T.match_buffer(A_ptr, A_shape, A_dtype)
        B = T.match_buffer(B_ptr, B_shape, B_dtype)
        C = T.match_buffer(C_ptr, C_shape, C_dtype)
        SFA_in = T.match_buffer(SFA_ptr, (128,), "uint32")
        SFB_in = T.match_buffer(SFB_ptr, (128,), "uint32")

        T.device_entry()
        warp_id = T.warp_id([(1) * 4])
        cta_id = T.cta_id([1])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])

        A_smem = T.alloc_buffer(A_shape, A_dtype, scope="shared", layout=A_layout)
        B_smem = T.alloc_buffer(B_shape, B_dtype, scope="shared", layout=B_layout)
        SFA_smem = T.alloc_buffer((4, 32), "uint32", scope="shared", layout=SF_smem_layout)
        SFB_smem = T.alloc_buffer((4, 32), "uint32", scope="shared", layout=SF_smem_layout)
        SFA_smem_post = SFA_smem.view(4, 32, layout=SF_smem_post_layout)
        SFB_smem_post = SFB_smem.view(4, 32, layout=SF_smem_post_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        tma_mbar = T.alloc_shared([1], "uint64")
        mma_mbar = T.alloc_shared([1], "uint64")
        descSFA = T.alloc_buffer((1,), "uint64", scope="local")
        descSFB = T.alloc_buffer((1,), "uint64", scope="local")

        if tid_in_wg == 0:
            T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=cols_alloc, cta_group=1)
        T.cuda.cta_sync()

        tmem = T.decl_buffer(C_shape, C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
        sfa_tmem = T.decl_buffer((M, sf_mma_k * num_ki), SF_dtype, scope="tmem", allocated_addr=SFA_TMEM_START, layout=sfa_layout)  # noqa: E501
        sfb_tmem = T.decl_buffer((N, sf_mma_k * num_ki), SF_dtype, scope="tmem", allocated_addr=SFB_TMEM_START, layout=sfb_layout)  # noqa: E501

                # TMA load A and B from global to shared
        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem[0:M, 0:K], A[0:M, 0:K], **tma_args)
            Tx.copy_async(B_smem[0:N, 0:K], B[0:N, 0:K], **tma_args)
            T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), total_bytes)
        T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        SFA_smem[tid_in_wg // 32, tid_in_wg % 32] = SFA_in[tid_in_wg]
        SFB_smem[tid_in_wg // 32, tid_in_wg % 32] = SFB_in[tid_in_wg]
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

                # Transpose scale factors in shared memory
        if warp_id == 0:
            Tx.warp.permute_layout(SFA_smem_post[:, :], SFA_smem[:, :])
            Tx.warp.permute_layout(SFB_smem_post[:, :], SFB_smem[:, :])
        T.cuda.cta_sync()

                # Copy SF to TMEM, then single MMA call (schedule auto-derives sf_id per ki)
        if tid_in_wg == 0:
            T.ptx.tcgen05.encode_matrix_descriptor(descSFA.data, SFA_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptx.tcgen05.cp(SFA_TMEM_START, descSFA[0], shape="32x128b", cta_group=1, multicast="warpx4")  # noqa: E501
            T.ptx.tcgen05.encode_matrix_descriptor(descSFB.data, SFB_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptx.tcgen05.cp(SFB_TMEM_START, descSFB[0], shape="32x128b", cta_group=1, multicast="warpx4")  # noqa: E501

                    # Single call with K=128: schedule auto-encodes descI and
                    # rotates sf_id=0,1,2,3 for each of the 4 ki iterations.
                    # SFA/SFB region covers all 4 ki positions (num_ki elements)
                    # so the schedule knows sf_id should rotate.
            Tx.gemm_async(tmem[0:128, 0:N], A_smem[0:M, 0:K], B_smem[0:N, 0:K], SFA=sfa_tmem[0:M, 0:sf_mma_k * num_ki], SFB=sfb_tmem[0:N, 0:sf_mma_k * num_ki], dispatch="tcgen05")  # noqa: E501

            T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=1)
        T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

                # Copy result from tmem to global
        T.ptx.tcgen05.fence.after_thread_sync()
        C_reg = T.alloc_local(N, dtype=C_dtype)
        C_view = C_reg.view(128, N, layout=TileLayout(S[(128, N) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[0:128, 0:N])
            T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, 0:N], C_reg[:])

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=cols_alloc, cta_group=1)
        # fmt: on

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

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": gemm_async_fn})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

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


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize(
    "task",
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
                _mn_major_layout("float16", 3, (3, 128, 128)),  # SMEM: K (penultimate) contiguous
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
def test_gemm_tcgen05_arbitrary_tiles(task):
    """Test arbitrary tile decomposition for tcgen05 gemm_async.

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

    A_gmem_kw = {"layout": A_gmem_layout} if A_gmem_layout is not None else {}
    B_gmem_kw = {"layout": B_gmem_layout} if B_gmem_layout is not None else {}

    # fmt: off
    @T.prim_func
    def gemm_async(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, A_shape, A_dtype, **A_gmem_kw)
        B = T.match_buffer(B_ptr, B_shape, B_dtype, **B_gmem_kw)
        C = T.match_buffer(C_ptr, C_shape, C_dtype)

        T.device_entry()
        warp_id = T.warp_id([(1) * 4])
        cta_id = T.cta_id([1])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])

        A_smem = T.alloc_buffer(A_shape, A_dtype, scope="shared", layout=A_layout, align=1024)
        B_smem = T.alloc_buffer(B_shape, B_dtype, scope="shared", layout=B_layout, align=1024)
        tmem_addr = T.alloc_shared([1], "uint32")
        tma_mbar = T.alloc_shared([1], "uint64")
        mma_mbar = T.alloc_shared([1], "uint64")

        if tid_in_wg == 0:
            T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.alloc(
                T.address_of(tmem_addr), n_cols=cols_alloc, cta_group=cta_group
            )
        T.cuda.cta_sync()
        tmem = T.decl_buffer((M, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(M, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501

        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem[tuple(r_gmem_A)], A[tuple(r_gmem_A)], **tma_args)
            Tx.copy_async(B_smem[tuple(r_gmem_B)], B[tuple(r_gmem_B)], **tma_args)
            T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), total_bytes)
        T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        if tid_in_wg == 0:
            Tx.gemm_async(tmem[tuple(r_tmem_C)], A_smem[tuple(r_smem_A)], B_smem[tuple(r_smem_B)], transA=transA, transB=transB, dispatch="tcgen05", cta_group=cta_group)  # noqa: E501
            T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=cta_group)
        T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        T.ptx.tcgen05.fence.after_thread_sync()
        C_reg = T.alloc_local(N, dtype=C_dtype)
        C_view = C_reg.view(M, N, layout=TileLayout(S[(M, N) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[tuple(r_tmem_C)])
            T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, C_region[1][0]:C_region[1][1]], C_reg[:])

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=cols_alloc, cta_group=cta_group)
        # fmt: on

    np.random.seed(0)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": gemm_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

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

    The operand buffer is 128B-swizzled (contiguous atom = 64 elems for fp16) and
    the gemm operand is sliced to K=[lo:hi] on that axis. The descriptor is
    anchored on the buffer's physical swizzle while K_iters covers only the slice,
    so the MMA accumulates exactly k in [lo, hi) -- enabling fine K-major split-K.
    Any MMA_K(16)-aligned [lo:hi] is supported.
    """
    from tvm.tirx.cuda.operator.tile_primitive.tma_utils import SwizzleMode

    M, N, K_alloc = 128, 128, 64
    dtype = "float16"
    A_shape, B_shape, C_shape = (M, K_alloc), (N, K_alloc), (M, N)
    A_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, A_shape)
    B_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, B_shape)
    total_bytes = (M * K_alloc + N * K_alloc) * 2

    # fmt: off
    @T.prim_func
    def gemm_async(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, A_shape, dtype)
        B = T.match_buffer(B_ptr, B_shape, dtype)
        C = T.match_buffer(C_ptr, C_shape, "float32")
        T.device_entry()
        warp_id = T.warp_id([4])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])
        A_smem = T.alloc_buffer(A_shape, dtype, scope="shared", layout=A_layout)
        B_smem = T.alloc_buffer(B_shape, dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        tma_mbar = T.alloc_shared([1], "uint64")
        mma_mbar = T.alloc_shared([1], "uint64")
        if tid_in_wg == 0:
            T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=128, cta_group=1)
        T.cuda.cta_sync()
        tmem = T.decl_buffer((128, N), "float32", scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, N) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem[0:M, 0:K_alloc], A[0:M, 0:K_alloc], **tma_args)
            Tx.copy_async(B_smem[0:N, 0:K_alloc], B[0:N, 0:K_alloc], **tma_args)
            T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), total_bytes)
        T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if tid_in_wg == 0:
            # Contiguous-axis K slice [k_lo:k_hi] -> must accumulate only that K range.
            Tx.gemm_async(tmem[0:128, 0:N], A_smem[0:M, k_lo:k_hi], B_smem[0:N, k_lo:k_hi], dispatch="tcgen05")  # noqa: E501
            T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=1)
        T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence.after_thread_sync()
        C_reg = T.alloc_local(N, dtype="float32")
        C_view = C_reg.view(128, N, layout=TileLayout(S[(128, N) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[0:128, 0:N])
            T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, 0:N], C_reg[:])
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=128, cta_group=1)
    # fmt: on

    np.random.seed(0)
    with tvm.target.Target("cuda"):
        mod = tvm.compile(tvm.IRModule({"main": gemm_async}), target="cuda", tir_pipeline="tirx")
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
    gemm_kw = {"dispatch": "tcgen05"}
    if is_AB_tf32:
        gemm_kw["is_AB_tf32"] = True
    b_tma_kw = {"dispatch": "tma"}
    if tma_dtype_B is not None:
        b_tma_kw["tma_dtype"] = tma_dtype_B

    @T.prim_func
    def gemm_async(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, A_shape, A_dtype)
        B = T.match_buffer(B_ptr, B_shape, B_dtype)
        C = T.match_buffer(C_ptr, C_shape, C_dtype)
        T.device_entry()
        warp_id = T.warp_id([4])
        T.cta_id([1])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])
        A_smem = T.alloc_buffer(A_shape, A_dtype, scope="shared", layout=A_layout)
        B_smem = T.alloc_buffer(B_shape, B_dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        tma_mbar = T.alloc_shared([1], "uint64")
        mma_mbar = T.alloc_shared([1], "uint64")
        if tid_in_wg == 0:
            T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=cols_alloc, cta_group=1)
        T.cuda.cta_sync()
        tmem = T.decl_buffer(
            (128, N),
            C_dtype,
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, N) : (1 @ TLane, 1 @ TCol)]),
        )
        if tid_in_wg == 0:
            Tx.copy_async(A_smem[:, :], A[:, :], dispatch="tma", mbar=tma_mbar.ptr_to([0]))
            Tx.copy_async(B_smem[:, :], B[:, :], mbar=tma_mbar.ptr_to([0]), **b_tma_kw)
            T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), total_bytes)
        T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if tid_in_wg == 0:
            Tx.gemm_async(tmem[:, :], A_smem[:, :], B_smem[:, :], **gemm_kw)
            T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=1)
        T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence.after_thread_sync()
        C_reg = T.alloc_local(N, dtype=C_dtype)
        C_view = C_reg.view(128, N, layout=TileLayout(S[(128, N) : (1 @ axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[:, :])
            T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, 0:N], C_reg[:])
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=cols_alloc, cta_group=1)

    np.random.seed(0)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": gemm_async}), target=target, tir_pipeline="tirx")

    def _rand(shape, dtype):
        f = np.random.randn(*shape).astype("float32")
        return f.astype(dtype) if ml_dtypes is not None or "float8" not in dtype else f

    A_np = _rand(A_shape, A_dtype)
    B_np = _rand(B_shape, B_dtype)
    C_np = np.zeros(C_shape, dtype=C_dtype)
    C_ref = A_np.astype("float32") @ B_np.astype("float32").T

    def run_and_check():
        dev = tvm.cuda(0)
        A_t, B_t, C_t = (tvm.runtime.tensor(x, dev) for x in (A_np, B_np, C_np))
        mod["main"](A_t, B_t, C_t)
        np.testing.assert_allclose(C_t.numpy().astype("float32"), C_ref, atol=atol, rtol=rtol)

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


def _build_smem_desc_kernel(smem_desc):
    """Minimal cta_group=1 fp16 gemm_async kernel parametrized on ``smem_desc``."""
    C_shape, C_dtype, C_region = (128, 512), "float32", [(0, 128), (256, 384)]
    A_shape, A_dtype, A_sw = (3, 128, 64), "float16", 3
    B_shape, B_dtype, B_sw = (3, 128, 64), "float16", 3
    width = C_region[1][1] - C_region[1][0]
    A_layout = mma_shared_layout(A_dtype, A_sw, A_shape)
    B_layout = mma_shared_layout(B_dtype, B_sw, B_shape)
    r_gmem_A = [slice(0, A_shape[i]) for i in range(len(A_shape))]
    r_gmem_B = [slice(0, B_shape[i]) for i in range(len(B_shape))]
    total_bytes = (
        functools.reduce(operator.mul, A_shape, 1) * 2
        + functools.reduce(operator.mul, B_shape, 1) * 2
    )
    r_tmem_C = [slice(C_region[i][0], C_region[i][1]) for i in range(len(C_shape))]
    r_smem_A = [slice(1, 2), slice(0, 128), slice(0, 64)]
    r_smem_B = [slice(2, 3), slice(0, 128), slice(0, 64)]

    # fmt: off
    @T.prim_func
    def gemm_async(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, A_shape, A_dtype)
        B = T.match_buffer(B_ptr, B_shape, B_dtype)
        C = T.match_buffer(C_ptr, C_shape, C_dtype)
        T.device_entry()
        warp_id = T.warp_id([4])
        cta_id = T.cta_id([1])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])
        A_smem = T.alloc_buffer(A_shape, A_dtype, scope="shared", layout=A_layout)
        B_smem = T.alloc_buffer(B_shape, B_dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        tma_mbar = T.alloc_shared([1], "uint64")
        mma_mbar = T.alloc_shared([1], "uint64")
        if tid_in_wg == 0:
            T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=128, cta_group=1)
        T.cuda.cta_sync()
        tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem[tuple(r_gmem_A)], A[tuple(r_gmem_A)], **tma_args)
            Tx.copy_async(B_smem[tuple(r_gmem_B)], B[tuple(r_gmem_B)], **tma_args)
            T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), total_bytes)
        T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if tid_in_wg == 0:
            Tx.gemm_async(tmem[tuple(r_tmem_C)], A_smem[tuple(r_smem_A)], B_smem[tuple(r_smem_B)], dispatch="tcgen05", smem_desc=smem_desc)  # noqa: E501
            T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=1)
        T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence.after_thread_sync()
        C_reg = T.alloc_local(width, dtype=C_dtype)
        C_view = C_reg.view(128, width, layout=TileLayout(S[(128, width) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[tuple(r_tmem_C)])
            T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, C_region[1][0]:C_region[1][1]], C_reg[:])
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=128, cta_group=1)
        # fmt: on

    return gemm_async


@pytest.mark.parametrize("smem_desc", ["hoist", "recompute"])
@pytest.mark.gpu
def test_gemm_smem_desc_hoist_vs_recompute(smem_desc):
    """Compile-only: the SMEM matrix descriptor is built per-MMA from the buffer
    base address, selected by the ``smem_desc`` config.

    - ``hoist`` (default): allocate + encode one warp-uniform descriptor per
      operand (``descA`` / ``descB`` + ``smem_desc_make_lo_uniform``) and add the
      per-MMA 16B offset via ``smem_desc_add_16B_offset``.
    - ``recompute``: build the full descriptor inline per MMA (``_uniform_desc``)
      with no allocated/encoded descriptor cell — trades a few ALU ops for one
      fewer live register on the hot path.

    Both must emit the MMA; the descriptor-construction fingerprints differ.
    """
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": _build_smem_desc_kernel(smem_desc)}),
            target=target,
            tir_pipeline="tirx",
        )
    src = mod.mod.imports[0].inspect_source()
    assert "tcgen05.mma" in src, f"mma not emitted; src=\n{src}"

    if smem_desc == "hoist":
        assert "smem_desc_make_lo_uniform" in src, "hoist mode must encode a uniform descriptor"
        assert "smem_desc_add_16B_offset" in src, "hoist mode must add the per-MMA 16B offset"
    else:
        assert "smem_desc_make_lo_uniform" not in src, "recompute mode must not hoist a descriptor"
        assert "smem_desc_add_16B_offset" not in src, "recompute mode must not add a 16B offset"
        assert "encode_matrix_descriptor" not in src, "recompute mode must not encode a descriptor"


if __name__ == "__main__":
    tvm.testing.main()
