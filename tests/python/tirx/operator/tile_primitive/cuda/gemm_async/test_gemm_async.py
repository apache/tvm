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
import re

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
from tvm.tirx.cuda.tile_primitive.gemm_async import sf_tmem_layout
from tvm.tirx.cuda.tile_primitive.tma_utils import (
    SwizzleMode,
    mma_atom_layout,
    mma_atom_shape,
    mma_shared_layout,
)
from tvm.tirx.layout import (
    R,
    S,
    TCol,
    TileLayout,
    TLane,
    tcgen05_atom_layout,
    tmem_datapath_layout,
)
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _mapa(ptr, rank):
    """`mapa.u64` into a declared register, returned as a value."""
    mapped = T.alloc_local([1], "uint64")
    T.evaluate(T.ptxd.mapa.u64(mapped[0], ptr, T.uint32(rank)))
    return mapped[0]


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
            T.ptxd.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.cuda.cta_sync()
        tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501

        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma_auto", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem[tuple(r_gmem_A)], A[tuple(r_gmem_A)], **tma_args)
            Tx.copy_async(B_smem[tuple(r_gmem_B)], B[tuple(r_gmem_B)], **tma_args)
            T.ptxd.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        if tid_in_wg == 0:
            Tx.gemm_async(tmem[tuple(r_tmem_C)], A_smem[tuple(r_smem_A)], B_smem[tuple(r_smem_B)], dispatch="tcgen05", mma_m=128, mma_n=64)  # noqa: E501
            T.ptxd.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(mma_mbar.ptr_to([0]))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        T.ptxd.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(width, dtype=C_dtype)
        C_view = C_reg.view(128, width, layout=TileLayout(S[(128, width) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[tuple(r_tmem_C)])
            T.ptxd.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, C_region[1][0]:C_region[1][1]], C_reg[:])

        if warp_id == 0:
            T.ptxd.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptxd.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))
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
    """M=64 MMA with C operand allocated as Layout F.

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
            T.ptxd.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(64)
            )
        T.cuda.cta_sync()
        # Layout F C operand — the path under test.
        tmem = T.decl_buffer((64, N), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=c_layout)  # noqa: E501

        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma_auto", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem[:, :], A[:, :], **tma_args)
            Tx.copy_async(B_smem[:, :], B[:, :], **tma_args)
            T.ptxd.mbarrier.arrive.expect_tx.shared.b64(
                tma_mbar.ptr_to([0]), T.uint32((M * K + N * K) * 2)
            )
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        if tid_in_wg == 0:
            Tx.gemm_async(tmem[0:64, 0:N], A_smem[:, :], B_smem[:, :], dispatch="tcgen05")
            T.ptxd.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(mma_mbar.ptr_to([0]))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptxd.tcgen05.fence__after_thread_sync()

        # Read back via .16x256b M=64 (the canonical pairing).
        reg = T.alloc_local(32, dtype="float32")
        reg_view = reg.view(64, N, layout=tcgen05_atom_layout("16x256b", (64, N), "float32"))
        if wg_id == 0:
            Tx.wg.copy_async(reg_view[:, :], tmem[0:64, 0:N])
            T.ptxd.tcgen05.wait__ld.sync.aligned()
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
            T.ptxd.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptxd.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(64))
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

        ptr: T.let[T.Var(name="ptr", ty=PointerType(PrimType("uint64"), "shared"))] = T.reinterpret(PointerType(PrimType("uint64"), "shared"), _mapa(tma_mbar.ptr_to([0]), 0))  # noqa: E501
        tma_mbar_cta_0 = T.decl_buffer([1], "uint64", data=ptr, scope="shared")

        if tid_in_wg == 0:
            T.ptxd.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))

        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
        T.ptxd.fence.mbarrier_init.release.cluster()
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        T.cuda.cluster_sync()

        tma_args = T.meta_var({"dispatch": "tma_auto", "mbar": tma_mbar_cta_0.ptr_to([0]), "cta_group": 2})  # noqa: E501
        if tid_in_wg == 0:
            Tx.copy_async(A_smem[tuple(r_smem_A_in)], A[tuple(get_global_region(A_shape_per_cta, transA, cbx))], **tma_args)  # noqa: E501
            Tx.copy_async(B_smem[tuple(r_smem_B_in)], B[tuple(get_global_region(B_shape_per_cta, transB, cbx))], **tma_args)  # noqa: E501
            if cbx == 0:
                T.ptxd.mbarrier.arrive.expect_tx.shared.b64(
                    tma_mbar.ptr_to([0]), T.uint32(total_bytes)
                )

        if cbx == 0:
            T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
            T.ptxd.tcgen05.fence__after_thread_sync()
            T.cuda.cta_sync()
            if tid_in_wg == 0:
                Tx.gemm_async(tmem[tuple(r_tmem_C)], A_smem[tuple(r_smem_A)], B_smem[tuple(r_smem_B)], dispatch="tcgen05", cta_group=2, mma_m=256, mma_n=128)  # noqa: E501
                T.ptxd[
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                    ".shared::cluster.multicast::cluster.b64"
                ](mma_mbar.ptr_to([0]), T.uint16(3)) # signal cta 1's mbarrier
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0) # both cta 0 and cta 1 have done mma
        T.ptxd.tcgen05.fence__after_thread_sync()
        T.cuda.cta_sync()

        C_reg = T.alloc_local(width , dtype=C_dtype)
        C_view = C_reg.view(128, width, layout=TileLayout(S[(128, width) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[C_region[0][0]:C_region[0][1], C_region[1][0]:C_region[1][0] + width])  # noqa: E501
            T.ptxd.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        Tx.copy(C[cbx * 128 +tid_in_wg, C_region[1][0]:C_region[1][0] + width], C_reg[:])
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptxd.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            T.ptxd.tcgen05.dealloc.cta_group__2.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))
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

        ptr: T.let[T.Var(name="ptr", ty=PointerType(PrimType("uint64"), "shared"))] = T.reinterpret(PointerType(PrimType("uint64"), "shared"), _mapa(tma_mbar.ptr_to([0]), 0))  # noqa: E501
        tma_mbar_cta_0 = T.decl_buffer([1], "uint64", data=ptr, scope="shared")

        if tid_in_wg == 0:
            T.ptxd.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))

        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        tmem = T.decl_buffer((M_per_cta, N_logical), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(M_per_cta, 2, N_half) : (1 @ TLane, 64 @ TLane, 1 @ TCol)]))  # noqa: E501
                # Physical TMEM view for readback: (128, N_half) standard layout
        tmem_phys = T.decl_buffer((128, N_half), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, N_half) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
        T.ptxd.fence.mbarrier_init.release.cluster()
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        T.cuda.cluster_sync()

        tma_args = T.meta_var({"dispatch": "tma_auto", "mbar": tma_mbar_cta_0.ptr_to([0]), "cta_group": 2})  # noqa: E501
        if tid_in_wg == 0:
                    # CTA cbx loads its portion of A and B
            Tx.copy_async(A_smem[0:M_per_cta, 0:K], A[cbx * M_per_cta:(cbx + 1) * M_per_cta, 0:K], **tma_args)  # noqa: E501
            Tx.copy_async(B_smem[0:N_half, 0:K], B[cbx * N_half:(cbx + 1) * N_half, 0:K], **tma_args)  # noqa: E501
            if cbx == 0:
                T.ptxd.mbarrier.arrive.expect_tx.shared.b64(
                    tma_mbar.ptr_to([0]), T.uint32(total_bytes)
                )

        if cbx == 0:
            T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
            T.ptxd.tcgen05.fence__after_thread_sync()
            T.cuda.cta_sync()
            if tid_in_wg == 0:
                Tx.gemm_async(tmem[0:M_per_cta, 0:N_logical], A_smem[0:M_per_cta, 0:K], B_smem[0:N_half, 0:K], dispatch="tcgen05", cta_group=2, mma_m=128, mma_n=128)  # noqa: E501
                T.ptxd[
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                    ".shared::cluster.multicast::cluster.b64"
                ](mma_mbar.ptr_to([0]), T.uint16(3))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.ptxd.tcgen05.fence__after_thread_sync()
        T.cuda.cta_sync()

                # Readback from physical TMEM view (128 rows x N_half cols)
                # Warps 0,1 (rows 0-63): first N half for M rows 0-63
                # Warps 2,3 (rows 64-127): second N half for M rows 0-63
        C_reg = T.alloc_local(N_half, dtype=C_dtype)
        C_view = C_reg.view(128, N_half, layout=TileLayout(S[(128, N_half) : (1 @ axis_tid_in_wg, 1)]))  # noqa: E501
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem_phys[0:128, 0:N_half])
            T.ptxd.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        n_off = (tid_in_wg // 64) * N_half
        Tx.copy(C[cbx * M_per_cta + tid_in_wg % 64, n_off : n_off + N_half], C_reg[:])
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptxd.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            T.ptxd.tcgen05.dealloc.cta_group__2.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))
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
def test_gemm_tcgen05_cta_group_2_datapath_b_readback():
    """A cta_group=2 GEMM writes and reads a first-class datapath B buffer."""
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

    # fmt: off
    @T.prim_func
    def gemm_async(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (m_per_cta * 2, k), input_dtype)
        B = T.match_buffer(B_ptr, (n_logical, k), input_dtype)
        C = T.match_buffer(C_ptr, c_shape, c_dtype)

        T.device_entry()
        warp_id = T.warp_id([4])
        cbx, cby = T.cta_id_in_cluster([2, 1])
        T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        tid = T.thread_id_in_wg([128])

        A_smem = T.alloc_buffer(a_shape, a_dtype, scope="shared", layout=a_layout)
        B_smem = T.alloc_buffer(b_shape, b_dtype, scope="shared", layout=b_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        mma_mbar = T.alloc_shared([1], "uint64")

        if tid == 0:
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(n_per_cta)
            )
        T.ptxd.fence.mbarrier_init.release.cluster()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (m_per_cta, n_logical),
            c_dtype,
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=tmem_datapath_layout("B", m_per_cta, n_logical),
        )

        # Use ordinary shared-memory stores here so this test is independent
        # of the TMA/remote-mbarrier path exercised by the older Layout B test.
        for i in range(m_per_cta * k // 128):
            A_smem[(tid + i * 128) // k, (tid + i * 128) % k] = T.Cast(a_dtype, A[
                cbx * m_per_cta + (tid + i * 128) // k, (tid + i * 128) % k
            ])
        for i in range(n_per_cta * k // 128):
            B_smem[(tid + i * 128) // k, (tid + i * 128) % k] = T.Cast(b_dtype, B[
                cbx * n_per_cta + (tid + i * 128) // k, (tid + i * 128) % k
            ])
        T.cuda.cta_sync()
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cluster_sync()

        if cbx == 0:
            T.ptxd.tcgen05.fence__after_thread_sync()
            T.cuda.cta_sync()
            if tid == 0:
                Tx.gemm_async(
                    tmem[:, :],
                    A_smem[:, :],
                    B_smem[:, :],
                    dispatch="tcgen05",
                    cta_group=2,
                )
                T.ptxd[
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                    ".shared::cluster.multicast::cluster.b64"
                ](mma_mbar.ptr_to([0]), T.uint16(3))

        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.ptxd.tcgen05.fence__after_thread_sync()
        T.cuda.cta_sync()

        frag = T.alloc_tcgen05_ldst_frag(
            "32x32b", (m_per_cta, n_logical), c_dtype
        )
        if wg_id == 0:
            Tx.wg.copy_async(frag[:, :], tmem[:, :])
            T.ptxd.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()

        frag_local = frag.local()
        for i in range(n_per_cta):
            C[
                cbx * m_per_cta + tid % m_per_cta,
                (tid // m_per_cta) * n_per_cta + i,
            ] = frag_local[i]
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptxd.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            T.ptxd.tcgen05.dealloc.cta_group__2.sync.aligned.b32(tmem_addr[0], T.uint32(n_per_cta))
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": gemm_async}), target=target, tir_pipeline="tirx"
        )

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
            T.ptxd.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.cuda.cta_sync()

        tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
        sfa_tmem = T.decl_buffer((M, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFA_TMEM_START, layout=sfa_layout)  # noqa: E501
        sfb_tmem = T.decl_buffer((N, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFB_TMEM_START, layout=sfb_layout)  # noqa: E501

                # TMA load A and B from global to shared
        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma_auto", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem[tuple(r_gmem_A)], A[tuple(r_gmem_A)], **tma_args)
            Tx.copy_async(B_smem[tuple(r_gmem_B)], B[tuple(r_gmem_B)], **tma_args)
            T.ptxd.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        SFA_smem[tid_in_wg // 32, tid_in_wg % 32] = SFA_in[tid_in_wg]
        SFB_smem[tid_in_wg // 32, tid_in_wg % 32] = SFB_in[tid_in_wg]
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

                # Transpose scale factors in shared memory
        if warp_id == 0:
            Tx.warp.permute_layout(SFA_smem_post[:, :], SFA_smem[:, :])
            Tx.warp.permute_layout(SFB_smem_post[:, :], SFB_smem[:, :])
        T.cuda.cta_sync()

                # Copy SFA/SFB from shared to TMEM via tcgen05.cp, then issue MMA
        if tid_in_wg == 0:
            T.cuda.tcgen05.encode_matrix_descriptor(descSFA.data, SFA_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptxd["tcgen05.cp.cta_group::1.32x128b.warpx4"](T.uint32(SFA_TMEM_START), descSFA[0])
            T.cuda.tcgen05.encode_matrix_descriptor(descSFB.data, SFB_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptxd["tcgen05.cp.cta_group::1.32x128b.warpx4"](T.uint32(SFB_TMEM_START), descSFB[0])

            Tx.gemm_async(tmem[tuple(r_tmem_C)], A_smem[tuple(r_smem_A)], B_smem[tuple(r_smem_B)], SFA=sfa_tmem[0:M, 0:sf_mma_k], SFB=sfb_tmem[0:N, 0:sf_mma_k], dispatch="tcgen05")  # noqa: E501
            T.ptxd.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(mma_mbar.ptr_to([0]))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

                # Copy result from tmem to global
        T.ptxd.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(width, dtype=C_dtype)
        C_view = C_reg.view(128, width, layout=TileLayout(S[(128, width) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[tuple(r_tmem_C)])
            T.ptxd.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, C_region[1][0]:C_region[1][1]], C_reg[:])

        if warp_id == 0:
            T.ptxd.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptxd.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))
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

        ptr: T.let[T.Var(name="ptr", ty=PointerType(PrimType("uint64"), "shared"))] = T.reinterpret(PointerType(PrimType("uint64"), "shared"), _mapa(tma_mbar.ptr_to([0]), 0))  # noqa: E501
        tma_mbar_cta_0 = T.decl_buffer([1], "uint64", data=ptr, scope="shared")

        if tid_in_wg == 0:
            T.ptxd.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))

        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501

        sfa_tmem = T.decl_buffer((128, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFA_TMEM_START, layout=sf_layout)  # noqa: E501
        sfb_tmem = T.decl_buffer((128, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFB_TMEM_START, layout=sf_layout)  # noqa: E501

        T.ptxd.fence.mbarrier_init.release.cluster()
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        T.cuda.cluster_sync()

                # TMA load A and B (both CTAs issue with multicast)
        tma_args = T.meta_var({"dispatch": "tma_auto", "mbar": tma_mbar_cta_0.ptr_to([0]), "cta_group": 2})  # noqa: E501
        if tid_in_wg == 0:
            Tx.copy_async(A_smem[tuple(r_smem_A_in)], A[tuple(get_global_region(A_shape_per_cta, transA, cbx))], **tma_args)  # noqa: E501
            Tx.copy_async(B_smem[tuple(r_smem_B_in)], B[tuple(get_global_region(B_shape_per_cta, transB, cbx))], **tma_args)  # noqa: E501
            if cbx == 0:
                T.ptxd.mbarrier.arrive.expect_tx.shared.b64(
                    tma_mbar.ptr_to([0]), T.uint32(total_bytes)
                )
        SFA_smem[tid_in_wg // 32, tid_in_wg % 32] = SFA_in[cbx * 128 + tid_in_wg]
        SFB_smem[tid_in_wg // 32, tid_in_wg % 32] = SFB_in[tid_in_wg]
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

                # Transpose scale factors (both CTAs)
        if warp_id == 0:
            Tx.warp.permute_layout(SFA_smem_post[:, :], SFA_smem[:, :])
            Tx.warp.permute_layout(SFB_smem_post[:, :], SFB_smem[:, :])
        T.cuda.cta_sync()

                # Copy SFA/SFB from shared to TMEM via tcgen05.cp (both CTAs, cta_group=2)
        if tid_in_wg == 0:
            T.cuda.tcgen05.encode_matrix_descriptor(descSFA.data, SFA_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptxd["tcgen05.cp.cta_group::2.32x128b.warpx4"](T.uint32(SFA_TMEM_START), descSFA[0])
            T.cuda.tcgen05.encode_matrix_descriptor(descSFB.data, SFB_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptxd["tcgen05.cp.cta_group::2.32x128b.warpx4"](T.uint32(SFB_TMEM_START), descSFB[0])
        T.cuda.cta_sync()
        T.cuda.cluster_sync()

        if cbx == 0:
            T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
            T.ptxd.tcgen05.fence__after_thread_sync()
            T.cuda.cta_sync()
            if tid_in_wg == 0:
                Tx.gemm_async(tmem[tuple(r_tmem_C)], A_smem[tuple(r_smem_A)], B_smem[tuple(r_smem_B)], SFA=sfa_tmem[0:128, 0:sf_mma_k], SFB=sfb_tmem[0:128, 0:sf_mma_k], dispatch="tcgen05", cta_group=2)  # noqa: E501
                T.ptxd[
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                    ".shared::cluster.multicast::cluster.b64"
                ](mma_mbar.ptr_to([0]), T.uint16(3))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.ptxd.tcgen05.fence__after_thread_sync()
        T.cuda.cta_sync()

                # Copy result from tmem to global
        C_reg = T.alloc_local(width, dtype=C_dtype)
        C_view = C_reg.view(128, width, layout=TileLayout(S[(128, width) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[C_region[0][0]:C_region[0][1], C_region[1][0]:C_region[1][0] + width])  # noqa: E501
            T.ptxd.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        Tx.copy(C[cbx * 128 + tid_in_wg, C_region[1][0]:C_region[1][0] + width], C_reg[:])
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptxd.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            T.ptxd.tcgen05.dealloc.cta_group__2.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))
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
            T.ptxd.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.cuda.cta_sync()

        tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
        sfa_tmem = T.decl_buffer((M, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFA_TMEM_START, layout=sfa_layout)  # noqa: E501
        sfb_tmem = T.decl_buffer((N, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFB_TMEM_START, layout=sfb_layout)  # noqa: E501

                # TMA load A and B as uint8
        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma_auto", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem_packed[:, :], A_packed[:, :], **tma_args)
            Tx.copy_async(B_smem_packed[:, :], B_packed[:, :], **tma_args)
            T.ptxd.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        SFA_smem[tid_in_wg // 32, tid_in_wg % 32] = SFA_in[tid_in_wg]
        SFB_smem[tid_in_wg // 32, tid_in_wg % 32] = SFB_in[tid_in_wg]
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

                # Transpose scale factors in shared memory
        if warp_id == 0:
            Tx.warp.permute_layout(SFA_smem_post[:, :], SFA_smem[:, :])
            Tx.warp.permute_layout(SFB_smem_post[:, :], SFB_smem[:, :])
        T.cuda.cta_sync()

                # Copy SFA/SFB from shared to TMEM via tcgen05.cp, then issue MMA
        if tid_in_wg == 0:
            T.cuda.tcgen05.encode_matrix_descriptor(descSFA.data, SFA_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptxd["tcgen05.cp.cta_group::1.32x128b.warpx4"](T.uint32(SFA_TMEM_START), descSFA[0])
            T.cuda.tcgen05.encode_matrix_descriptor(descSFB.data, SFB_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptxd["tcgen05.cp.cta_group::1.32x128b.warpx4"](T.uint32(SFB_TMEM_START), descSFB[0])

            Tx.gemm_async(tmem[0:128, 0:N], A_smem[:, :], B_smem[:, :], SFA=sfa_tmem[0:M, 0:sf_mma_k], SFB=sfb_tmem[0:N, 0:sf_mma_k], dispatch="tcgen05")  # noqa: E501
            T.ptxd.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(mma_mbar.ptr_to([0]))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

                # Copy result from tmem to global
        T.ptxd.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(width, dtype=C_dtype)
        C_view = C_reg.view(128, width, layout=TileLayout(S[(128, width) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[0:128, 0:N])
            T.ptxd.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, 0:N], C_reg[:])

        if warp_id == 0:
            T.ptxd.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptxd.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))
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

        ptr: T.let[T.Var(name="ptr", ty=PointerType(PrimType("uint64"), "shared"))] = T.reinterpret(PointerType(PrimType("uint64"), "shared"), _mapa(tma_mbar.ptr_to([0]), 0))  # noqa: E501
        tma_mbar_cta_0 = T.decl_buffer([1], "uint64", data=ptr, scope="shared")

        if tid_in_wg == 0:
            T.ptxd.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))

        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501

        sfa_tmem = T.decl_buffer((M_per_cta, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFA_TMEM_START, layout=sfa_layout)  # noqa: E501
        sfb_tmem = T.decl_buffer((N_total, sf_mma_k), SF_dtype, scope="tmem", allocated_addr=SFB_TMEM_START, layout=sfb_layout)  # noqa: E501

        T.ptxd.fence.mbarrier_init.release.cluster()
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        T.cuda.cluster_sync()

                # TMA load A and B with multicast (each CTA loads its portion)
        tma_args = T.meta_var({"dispatch": "tma_auto", "mbar": tma_mbar_cta_0.ptr_to([0]), "cta_group": 2})  # noqa: E501
        if tid_in_wg == 0:
            Tx.copy_async(A_smem_packed[:, :], A_packed[cbx * M_per_cta:(cbx + 1) * M_per_cta, :], **tma_args)  # noqa: E501
            Tx.copy_async(B_smem_packed[:, :], B_packed[cbx * N_per_cta:(cbx + 1) * N_per_cta, :], **tma_args)  # noqa: E501
            if cbx == 0:
                T.ptxd.mbarrier.arrive.expect_tx.shared.b64(
                    tma_mbar.ptr_to([0]), T.uint32(total_bytes)
                )
        SFA_smem[tid_in_wg // 32, tid_in_wg % 32] = SFA_in[cbx * M_per_cta + tid_in_wg]
        SFB_smem[tid_in_wg // 32, tid_in_wg % 32] = SFB_in[tid_in_wg]
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

                # Transpose scale factors
        if warp_id == 0:
            Tx.warp.permute_layout(SFA_smem_post[:, :], SFA_smem[:, :])
            Tx.warp.permute_layout(SFB_smem_post[:, :], SFB_smem[:, :])
        T.cuda.cta_sync()

                # Copy SFA/SFB from shared to TMEM via tcgen05.cp
        if tid_in_wg == 0:
            T.cuda.tcgen05.encode_matrix_descriptor(descSFA.data, SFA_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptxd["tcgen05.cp.cta_group::2.32x128b.warpx4"](T.uint32(SFA_TMEM_START), descSFA[0])
            T.cuda.tcgen05.encode_matrix_descriptor(descSFB.data, SFB_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptxd["tcgen05.cp.cta_group::2.32x128b.warpx4"](T.uint32(SFB_TMEM_START), descSFB[0])
        T.cuda.cta_sync()
        T.cuda.cluster_sync()

        if cbx == 0:
            T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
            T.ptxd.tcgen05.fence__after_thread_sync()
            T.cuda.cta_sync()
            if tid_in_wg == 0:
                Tx.gemm_async(tmem[0:128, 0:N_total], A_smem[:, :], B_smem[:, :], SFA=sfa_tmem[0:128, 0:sf_mma_k], SFB=sfb_tmem[0:N_total, 0:sf_mma_k], dispatch="tcgen05", cta_group=2)  # noqa: E501
                T.ptxd[
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                    ".shared::cluster.multicast::cluster.b64"
                ](mma_mbar.ptr_to([0]), T.uint16(3))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.ptxd.tcgen05.fence__after_thread_sync()
        T.cuda.cta_sync()

                # Copy result from tmem to global
        C_reg = T.alloc_local(width, dtype=C_dtype)
        C_view = C_reg.view(128, width, layout=TileLayout(S[(128, width) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[0:128, 0:width])
            T.ptxd.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        Tx.copy(C[cbx * M_per_cta + tid_in_wg, 0:width], C_reg[:])
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptxd.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            T.ptxd.tcgen05.dealloc.cta_group__2.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))
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
            T.ptxd.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.cuda.cta_sync()

        tmem = T.decl_buffer(C_shape, C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
        sfa_tmem = T.decl_buffer((M, sf_mma_k * num_ki), SF_dtype, scope="tmem", allocated_addr=SFA_TMEM_START, layout=sfa_layout)  # noqa: E501
        sfb_tmem = T.decl_buffer((N, sf_mma_k * num_ki), SF_dtype, scope="tmem", allocated_addr=SFB_TMEM_START, layout=sfb_layout)  # noqa: E501

                # TMA load A and B from global to shared
        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma_auto", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem[0:M, 0:K], A[0:M, 0:K], **tma_args)
            Tx.copy_async(B_smem[0:N, 0:K], B[0:N, 0:K], **tma_args)
            T.ptxd.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        SFA_smem[tid_in_wg // 32, tid_in_wg % 32] = SFA_in[tid_in_wg]
        SFB_smem[tid_in_wg // 32, tid_in_wg % 32] = SFB_in[tid_in_wg]
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

                # Transpose scale factors in shared memory
        if warp_id == 0:
            Tx.warp.permute_layout(SFA_smem_post[:, :], SFA_smem[:, :])
            Tx.warp.permute_layout(SFB_smem_post[:, :], SFB_smem[:, :])
        T.cuda.cta_sync()

                # Copy SF to TMEM, then single MMA call (schedule auto-derives sf_id per ki)
        if tid_in_wg == 0:
            T.cuda.tcgen05.encode_matrix_descriptor(descSFA.data, SFA_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptxd["tcgen05.cp.cta_group::1.32x128b.warpx4"](T.uint32(SFA_TMEM_START), descSFA[0])
            T.cuda.tcgen05.encode_matrix_descriptor(descSFB.data, SFB_smem.access_ptr("r", offset=0), ldo=16, sdo=8 * 4 * F32_BYTES // F128_BYTES, swizzle=0)  # noqa: E501
            T.ptxd["tcgen05.cp.cta_group::1.32x128b.warpx4"](T.uint32(SFB_TMEM_START), descSFB[0])

                    # Single call with K=128: schedule auto-encodes descI and
                    # rotates sf_id=0,1,2,3 for each of the 4 ki iterations.
                    # SFA/SFB region covers all 4 ki positions (num_ki elements)
                    # so the schedule knows sf_id should rotate.
            Tx.gemm_async(tmem[0:128, 0:N], A_smem[0:M, 0:K], B_smem[0:N, 0:K], SFA=sfa_tmem[0:M, 0:sf_mma_k * num_ki], SFB=sfb_tmem[0:N, 0:sf_mma_k * num_ki], dispatch="tcgen05")  # noqa: E501

            T.ptxd.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(mma_mbar.ptr_to([0]))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

                # Copy result from tmem to global
        T.ptxd.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(N, dtype=C_dtype)
        C_view = C_reg.view(128, N, layout=TileLayout(S[(128, N) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[0:128, 0:N])
            T.ptxd.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, 0:N], C_reg[:])

        if warp_id == 0:
            T.ptxd.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptxd.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))
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
            T.ptxd.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptxd[f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"](
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.cuda.cta_sync()
        tmem = T.decl_buffer((M, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(M, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501

        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma_auto", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem[tuple(r_gmem_A)], A[tuple(r_gmem_A)], **tma_args)
            Tx.copy_async(B_smem[tuple(r_gmem_B)], B[tuple(r_gmem_B)], **tma_args)
            T.ptxd.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        if tid_in_wg == 0:
            Tx.gemm_async(tmem[tuple(r_tmem_C)], A_smem[tuple(r_smem_A)], B_smem[tuple(r_smem_B)], transA=transA, transB=transB, dispatch="tcgen05", cta_group=cta_group)  # noqa: E501
            T.ptxd[f"tcgen05.commit.cta_group::{cta_group}.mbarrier::arrive::one.shared::cluster.b64"](mma_mbar.ptr_to([0]))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        T.ptxd.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(N, dtype=C_dtype)
        C_view = C_reg.view(M, N, layout=TileLayout(S[(M, N) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[tuple(r_tmem_C)])
            T.ptxd.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, C_region[1][0]:C_region[1][1]], C_reg[:])

        if warp_id == 0:
            T.ptxd[f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned"]()
            T.ptxd[f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32"](
                tmem_addr[0], T.uint32(cols_alloc)
            )
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


@pytest.mark.parametrize("a_layout_kind", ["column_major", "packed_16b"])
def test_gemm_tcgen05_no_swizzle_smem_descriptor_codegen(a_layout_kind):
    M, K, N = 64, 64, 256
    B_N = N // 2
    dtype = "bfloat16"
    if a_layout_kind == "column_major":
        A_layout = _col_major_layout((M, K))
    else:
        A_layout = TileLayout(S[(M, K // 8, 8) : (8, M * 8, 1)])
    B_layout = _mn_major_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (K, B_N))
    C_layout = TileLayout(S[(M, 2, N // 2) : (1 @ TLane, 64 @ TLane, 1 @ TCol)])

    @T.prim_func
    def gemm_async_no_swizzle(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (M, K), dtype, layout=A_layout)
        B = T.match_buffer(B_ptr, (K, B_N), dtype, layout=B_layout)
        T.device_entry()
        warp_id = T.warp_id([4])
        T.thread_id([128])
        tid_in_wg = T.thread_id_in_wg([128])
        A_smem = T.alloc_buffer((M, K), dtype, scope="shared", layout=A_layout)
        B_smem = T.alloc_buffer((K, B_N), dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr[0]), T.uint32(256)
            )
        T.cuda.cta_sync()
        tmem = T.decl_buffer(
            (M, N),
            "float32",
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=C_layout,
        )
        if tid_in_wg == 0:
            Tx.gemm_async(
                tmem[:, :],
                A_smem[:, :],
                B_smem[:, :],
                transB=True,
                dispatch="tcgen05",
                cta_group=2,
            )

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": gemm_async_no_swizzle}), target=target, tir_pipeline="tirx"
        )

    src = mod.mod.imports[0].inspect_source()
    assert "descA" in src
    assert ", 64, 8, 0)" in src
    assert "encode_instr_descriptor" not in src


def test_gemm_tcgen05_cta_group_2_accepts_replicated_tmem_a_codegen():
    """A-in-TMEM for cta_group::2 may declare the physical +64 lane mirror.

    FlashMLA head128 copies Qt with 64x128b.warpx2::02_13, which writes rows
    0..63 and mirrors them at lane +64.  The QK GEMM still addresses the anchor
    tile, but the A buffer layout should be allowed to make that mirror explicit.
    """

    M = 64
    N = 128
    N_half = N // 2
    K = 128
    dtype = "bfloat16"
    C_layout = TileLayout(S[(M, 2, N_half) : (1 @ TLane, 64 @ TLane, 1 @ TCol)])
    A_layout = TileLayout(S[(M, K) : (1 @ TLane, 1 @ TCol)] + R[2 : 64 @ TLane])
    B_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (N_half, K))

    @T.prim_func
    def gemm_async_replicated_a() -> None:
        T.device_entry()
        warp_id = T.warp_id([4])
        T.thread_id([128])
        tid_in_wg = T.thread_id_in_wg([128])
        B_smem = T.alloc_buffer((N_half, K), dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr[0]), T.uint32(128)
            )
        T.cuda.cta_sync()
        C_tmem = T.decl_buffer(
            (M, N),
            "float32",
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=C_layout,
        )
        A_tmem = T.decl_buffer(
            (M, K),
            dtype,
            scope="tmem",
            allocated_addr=tmem_addr[0] + T.uint32(64),
            layout=A_layout,
        )
        if tid_in_wg == 0:
            Tx.gemm_async(
                C_tmem[:, :],
                A_tmem[:, :],
                B_smem[:, :],
                dispatch="tcgen05",
                cta_group=2,
            )

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": gemm_async_replicated_a}),
            target=target,
            tir_pipeline="tirx",
        )

    src = mod.mod.imports[0].inspect_source()
    assert "tcgen05.mma.cta_group::2" in src
    assert "tcgen05.mma.ws" not in src
    assert "tvm_builtin_ptxd_tcgen05_mma_ts_" in src


def test_gemm_tcgen05_cta_group_2_rejects_flat_tmem_a_codegen():
    """cta_group::2 Layout-B A-in-TMEM must declare its +64-lane footprint."""

    M = 64
    N = 128
    N_half = N // 2
    K = 128
    dtype = "bfloat16"
    C_layout = TileLayout(S[(M, 2, N_half) : (1 @ TLane, 64 @ TLane, 1 @ TCol)])
    A_layout = TileLayout(S[(M, K) : (1 @ TLane, 1 @ TCol)])
    B_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (N_half, K))

    @T.prim_func
    def gemm_async_flat_a() -> None:
        T.device_entry()
        warp_id = T.warp_id([4])
        T.thread_id([128])
        tid_in_wg = T.thread_id_in_wg([128])
        B_smem = T.alloc_buffer((N_half, K), dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr[0]), T.uint32(128)
            )
        T.cuda.cta_sync()
        C_tmem = T.decl_buffer(
            (M, N),
            "float32",
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=C_layout,
        )
        A_tmem = T.decl_buffer(
            (M, K),
            dtype,
            scope="tmem",
            allocated_addr=tmem_addr[0] + T.uint32(64),
            layout=A_layout,
        )
        if tid_in_wg == 0:
            Tx.gemm_async(
                C_tmem[:, :],
                A_tmem[:, :],
                B_smem[:, :],
                dispatch="tcgen05",
                cta_group=2,
            )

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        with pytest.raises(Exception, match="TMEM A layout"):
            tvm.compile(
                tvm.IRModule({"main": gemm_async_flat_a}),
                target=target,
                tir_pipeline="tirx",
            )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.skipif(ml_dtypes is None, reason="Requires ml_dtypes")
def test_gemm_tcgen05_no_swizzle_col_major_a_ws_local_idesc():
    """A column-major-viewed unswizzled SMEM A under kind::f16 + ws.

    This is the FlashMLA head64 O-GEMM: A = S tile [M=64, K=64] bf16 whose
    GEMM operand view is column-major (M, K):(1, M) with no swizzle.  The view
    is a stride fiction: the S tile is physically stored 16B-line packed along
    K (elem offset 8*m + 8*M*(k//8) + k%8), and the view's strides reproduce
    that K-major layout's byte offsets.  The dispatcher's no-swizzle
    descriptor (ldo=64, sdo=8) is therefore a K-major encoding: hardware
    consumes it with instruction-descriptor bit15 (a_major) CLEAR.
    Historically callers masked this by hand-passing ``descI`` with
    trans_a=0; with the locally encoded descI the dispatcher's returned
    majorness must itself be consistent with the descriptor it constructed.

    Asserts (a) the generated idesc literal equals the hand-validated
    0x04410490 (bit15=0) and not the MN-major mis-encoding 0x04418490, with
    the unchanged descA fields (64, 8, 0); and (b) the GEMM is numerically
    correct on GPU.
    """
    M, K, N = 64, 64, 256
    dtype = "bfloat16"
    A_layout = TileLayout(S[(M, K) : (1, M)])  # column-major, unswizzled
    B_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (K, N))

    # fmt: off
    @T.prim_func
    def gemm_ws(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (M, K), dtype)
        B = T.match_buffer(B_ptr, (K, N), dtype)
        C = T.match_buffer(C_ptr, (128, N // 2), "float32")
        T.device_entry()
        warp_id = T.warp_id([4])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])
        A_smem = T.alloc_buffer((M, K), dtype, scope="shared", layout=A_layout)
        B_smem = T.alloc_buffer((K, N), dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        mma_mbar = T.alloc_shared([1], "uint64")
        if tid_in_wg == 0:
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(128)
            )
        T.cuda.cta_sync()
        # M=64 .ws accumulates via datapath E (FlashMLA head64's tmem_o layout):
        # lane = m + 64*(n >= N/2), col = n % (N/2).
        tmem = T.decl_buffer((M, N), "float32", scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(M, 2, N // 2) : (1 @ TLane, 64 @ TLane, 1 @ TCol)]))  # noqa: E501
        # Identity overlay of the physical 128x128 TMEM footprint for readback.
        tmem_ldst = T.decl_buffer((128, N // 2), "float32", scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, N // 2) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
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
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if tid_in_wg == 0:
            Tx.gemm_async(tmem[:, :], A_smem[:, :], B_smem[:, :], transB=True, dispatch="tcgen05", cta_group=1, weight_stationary=True)  # noqa: E501
            T.ptxd.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(mma_mbar.ptr_to([0]))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptxd.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(N // 2, dtype="float32")
        C_view = C_reg.view(128, N // 2, layout=TileLayout(S[(128, N // 2) : (1@axis_tid_in_wg, 1)]))  # noqa: E501
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem_ldst[:, :])
            T.ptxd.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, 0 : N // 2], C_reg[:])
        if warp_id == 0:
            T.ptxd.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptxd.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(128))
    # fmt: on

    with tvm.target.Target("cuda"):
        mod = tvm.compile(tvm.IRModule({"main": gemm_ws}), target="cuda", tir_pipeline="tirx")

    src = mod.mod.imports[0].inspect_source()
    # Descriptor construction: no-swizzle col-major A -> (ldo=64, sdo=8, swizzle=0).
    assert ", 64, 8, 0)" in src
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
    from tvm.tirx.cuda.tile_primitive.tma_utils import SwizzleMode

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
            T.ptxd.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(128)
            )
        T.cuda.cta_sync()
        tmem = T.decl_buffer((128, N), "float32", scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, N) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma_auto", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem[0:M, 0:K_alloc], A[0:M, 0:K_alloc], **tma_args)
            Tx.copy_async(B_smem[0:N, 0:K_alloc], B[0:N, 0:K_alloc], **tma_args)
            T.ptxd.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if tid_in_wg == 0:
            # Contiguous-axis K slice [k_lo:k_hi] -> must accumulate only that K range.
            Tx.gemm_async(tmem[0:128, 0:N], A_smem[0:M, k_lo:k_hi], B_smem[0:N, k_lo:k_hi], dispatch="tcgen05")  # noqa: E501
            T.ptxd.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(mma_mbar.ptr_to([0]))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptxd.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(N, dtype="float32")
        C_view = C_reg.view(128, N, layout=TileLayout(S[(128, N) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[0:128, 0:N])
            T.ptxd.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, 0:N], C_reg[:])
        if warp_id == 0:
            T.ptxd.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptxd.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(128))
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
    b_tma_kw = {"dispatch": "tma_auto"}
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
            T.ptxd.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.cuda.cta_sync()
        tmem = T.decl_buffer(
            (128, N),
            C_dtype,
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, N) : (1 @ TLane, 1 @ TCol)]),
        )
        if tid_in_wg == 0:
            Tx.copy_async(A_smem[:, :], A[:, :], dispatch="tma_auto", mbar=tma_mbar.ptr_to([0]))
            Tx.copy_async(B_smem[:, :], B[:, :], mbar=tma_mbar.ptr_to([0]), **b_tma_kw)
            T.ptxd.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if tid_in_wg == 0:
            Tx.gemm_async(tmem[:, :], A_smem[:, :], B_smem[:, :], **gemm_kw)
            T.ptxd.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                mma_mbar.ptr_to([0])
            )
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptxd.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(N, dtype=C_dtype)
        C_view = C_reg.view(128, N, layout=TileLayout(S[(128, N) : (1 @ axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[:, :])
            T.ptxd.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, 0:N], C_reg[:])
        if warp_id == 0:
            T.ptxd.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptxd.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))

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
    b_tma_kw = {"dispatch": "tma_auto"}
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
            T.ptxd.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(cols_alloc)
            )
        T.cuda.cta_sync()
        tmem = T.decl_buffer(
            (128, N),
            C_dtype,
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, N) : (1 @ TLane, 1 @ TCol)]),
        )
        if tid_in_wg == 0:
            Tx.copy_async(A_smem[:, :], A[:, :], dispatch="tma_auto", mbar=tma_mbar.ptr_to([0]))
            Tx.copy_async(B_smem[:, :], B[:, :], mbar=tma_mbar.ptr_to([0]), **b_tma_kw)
            T.ptxd.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if tid_in_wg == 0:
            Tx.gemm_async(tmem[:, :], A_smem[:, :], B_smem[:, :], **gemm_kw)
            T.ptxd.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                mma_mbar.ptr_to([0])
            )
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptxd.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(N, dtype=C_dtype)
        C_view = C_reg.view(128, N, layout=TileLayout(S[(128, N) : (1 @ axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[:, :])
            T.ptxd.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, 0:N], C_reg[:])
        if warp_id == 0:
            T.ptxd.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptxd.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(cols_alloc))

    dev = tvm.cuda(0)
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
    A_t, B_t, C_t = (tvm.runtime.tensor(x, dev) for x in (A_np, B_np, C_np))
    mod["main"](A_t, B_t, C_t)
    C_ref = A_np.astype("float32") @ B_np.astype("float32").T
    np.testing.assert_allclose(C_t.numpy().astype("float32"), C_ref, atol=atol, rtol=rtol)


def _build_smem_desc_kernel(smem_desc, weight_stationary=False, pass_descI=False, mma_config=None):
    """Minimal cta_group=1 fp16 gemm_async kernel parametrized on ``smem_desc``."""
    mma_cfg = {} if mma_config is None else mma_config
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
            T.ptxd.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptxd.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptxd.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(128)
            )
        T.cuda.cta_sync()
        tmem = T.decl_buffer((128, C_shape[1]), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, C_shape[1]) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501
        if tid_in_wg == 0:
            tma_args = T.meta_var({"dispatch": "tma_auto", "mbar": tma_mbar.ptr_to([0])})
            Tx.copy_async(A_smem[tuple(r_gmem_A)], A[tuple(r_gmem_A)], **tma_args)
            Tx.copy_async(B_smem[tuple(r_gmem_B)], B[tuple(r_gmem_B)], **tma_args)
            T.ptxd.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if tid_in_wg == 0:
            if pass_descI:
                desc_i: T.uint32
                T.cuda.tcgen05.encode_instr_descriptor(
                    T.address_of(desc_i),  # noqa: F821
                    d_dtype=C_dtype,
                    a_dtype=A_dtype,
                    b_dtype=B_dtype,
                    M=128,
                    N=width,
                    K=16,
                    trans_a=False,
                    trans_b=False,
                    n_cta_groups=1,
                )
                Tx.gemm_async(tmem[tuple(r_tmem_C)], A_smem[tuple(r_smem_A)], B_smem[tuple(r_smem_B)], dispatch="tcgen05", smem_desc=smem_desc, weight_stationary=weight_stationary, descI=desc_i, **mma_cfg)  # noqa: E501, F821
            else:
                Tx.gemm_async(tmem[tuple(r_tmem_C)], A_smem[tuple(r_smem_A)], B_smem[tuple(r_smem_B)], dispatch="tcgen05", smem_desc=smem_desc, weight_stationary=weight_stationary, **mma_cfg)  # noqa: E501
            T.ptxd.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(mma_mbar.ptr_to([0]))
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptxd.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(width, dtype=C_dtype)
        C_view = C_reg.view(128, width, layout=TileLayout(S[(128, width) : (1@axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[tuple(r_tmem_C)])
            T.ptxd.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, C_region[1][0]:C_region[1][1]], C_reg[:])
        if warp_id == 0:
            T.ptxd.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptxd.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(128))
        # fmt: on

    return gemm_async


def _build_explicit_cta2_dense_kernel(M_per_cta, mma_m):
    """Minimal cta_group=2 fp16 kernel with an explicit descriptor M."""
    N, K = 64, 16
    A_shape = (M_per_cta, K)
    B_shape = (N // 2, K)
    A_layout = mma_shared_layout("float16", SwizzleMode.SWIZZLE_NONE, A_shape)
    B_layout = mma_shared_layout("float16", SwizzleMode.SWIZZLE_NONE, B_shape)
    if M_per_cta == 64:
        C_layout = TileLayout(S[(M_per_cta, 2, N // 2) : (1 @ TLane, 64 @ TLane, 1 @ TCol)])
    else:
        C_layout = TileLayout(S[(M_per_cta, N) : (1 @ TLane, 1 @ TCol)])

    # fmt: off
    @T.prim_func
    def kernel() -> None:
        T.device_entry()
        cta_id = T.cta_id([2])
        cbx, cby = T.cta_id_in_cluster([2, 1])
        thread_id = T.thread_id([128])
        tid = T.thread_id_in_wg([128])
        A_smem = T.alloc_buffer(A_shape, "float16", scope="shared", layout=A_layout)
        B_smem = T.alloc_buffer(B_shape, "float16", scope="shared", layout=B_layout)
        C_tmem = T.decl_buffer((M_per_cta, N), "float32", scope="tmem", allocated_addr=0, layout=C_layout)  # noqa: E501
        if tid == 0:
            Tx.gemm_async(C_tmem[:, :], A_smem[:, :], B_smem[:, :], dispatch="tcgen05", cta_group=2, mma_m=mma_m, mma_n=N)  # noqa: E501
        # fmt: on

    return kernel


def _build_explicit_block_scaled_split_n_kernel():
    """Minimal cta_group=2 block-scaled fp8 kernel split into two N instructions."""
    M, N, K = 128, 128, 32
    A_shape = (M, K)
    B_shape = (N // 2, K)
    A_layout = mma_shared_layout("float8_e4m3fn", SwizzleMode.SWIZZLE_32B_ATOM, A_shape)
    B_layout = mma_shared_layout("float8_e4m3fn", SwizzleMode.SWIZZLE_32B_ATOM, B_shape)
    C_layout = TileLayout(S[(M, N) : (1 @ TLane, 1 @ TCol)])
    sf_layout = sf_tmem_layout(M, SF_K=1, sf_per_mma=1)

    # fmt: off
    @T.prim_func
    def kernel() -> None:
        T.device_entry()
        cta_id = T.cta_id([2])
        cbx, cby = T.cta_id_in_cluster([2, 1])
        thread_id = T.thread_id([128])
        tid = T.thread_id_in_wg([128])
        A_smem = T.alloc_buffer(A_shape, "float8_e4m3fn", scope="shared", layout=A_layout)
        B_smem = T.alloc_buffer(B_shape, "float8_e4m3fn", scope="shared", layout=B_layout)
        C_tmem = T.decl_buffer((M, N), "float32", scope="tmem", allocated_addr=0, layout=C_layout)
        SFA_tmem = T.decl_buffer((M, 1), "float8_e8m0fnu", scope="tmem", allocated_addr=N, layout=sf_layout)  # noqa: E501
        SFB_tmem = T.decl_buffer((N, 1), "float8_e8m0fnu", scope="tmem", allocated_addr=N + 4, layout=sf_layout)  # noqa: E501
        if tid == 0:
            Tx.gemm_async(C_tmem[:, :], A_smem[:, :], B_smem[:, :], SFA=SFA_tmem[:, :], SFB=SFB_tmem[:, :], dispatch="tcgen05", cta_group=2, mma_m=256, mma_n=64)  # noqa: E501
        # fmt: on

    return kernel


def _compile_cuda_source(func):
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": func}),
            target=target,
            tir_pipeline="tirx",
        )
    return mod.mod.imports[0].inspect_source()


def _dense_mma_call_lines(src):
    helper = "tvm_builtin_ptxd_tcgen05_mma_ss_mma_cta_group__1_kind__f16("
    return [line for line in src.splitlines() if line.lstrip().startswith(helper)]


def _mma_descriptor_values(lines):
    # descI is the argument right before the all-zero disable-output-lane vector.
    return [
        int(re.search(r", \(uint\)(\d+), \(uint\)0, \(uint\)0", line).group(1)) for line in lines
    ]


def test_gemm_tcgen05_explicit_mma_tile_changes_physical_instructions():
    default_src = _compile_cuda_source(_build_smem_desc_kernel("hoist"))
    explicit_src = _compile_cuda_source(
        _build_smem_desc_kernel("hoist", mma_config={"mma_m": 128, "mma_n": 64})
    )

    default_calls = _dense_mma_call_lines(default_src)
    explicit_calls = _dense_mma_call_lines(explicit_src)
    assert len(default_calls) == 4
    assert len(explicit_calls) == 8

    assert {(desc >> 17) & 0x3F for desc in _mma_descriptor_values(default_calls)} == {128 // 8}
    assert {(desc >> 17) & 0x3F for desc in _mma_descriptor_values(explicit_calls)} == {64 // 8}


@pytest.mark.parametrize(
    ("M_per_cta", "mma_m"),
    [
        (64, 128),
        (128, 256),
    ],
)
def test_gemm_tcgen05_explicit_mma_m_is_cluster_total_for_cta_group_2(M_per_cta, mma_m):
    src = _compile_cuda_source(_build_explicit_cta2_dense_kernel(M_per_cta, mma_m))
    assert "tcgen05.mma.cta_group::2.kind::f16" in src
    helper = "tvm_builtin_ptxd_tcgen05_mma_ss_mma_cta_group__2_kind__f16("
    calls = [line for line in src.splitlines() if line.lstrip().startswith(helper)]
    assert len(calls) == 1
    assert {(desc >> 24) & 0x1F for desc in _mma_descriptor_values(calls)} == {mma_m // 16}


def test_gemm_tcgen05_block_scaled_explicit_mma_tile_splits_n():
    src = _compile_cuda_source(_build_explicit_block_scaled_split_n_kernel())
    helper = "tvm_builtin_ptxd_tcgen05_mma_block_scale_ss_mma_cta_group__2_kind__mxf8f6f4"
    calls = [line for line in src.splitlines() if line.lstrip().startswith(helper)]
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("mma_config", "message"),
    [
        ({"mma_m": 128}, "must be provided together"),
        ({"mma_n": 64}, "must be provided together"),
        ({"mma_m": 0, "mma_n": 64}, "mma_m must be a positive integer"),
        ({"mma_m": 128, "mma_n": -1}, "mma_n must be a positive integer"),
        ({"mma_m": True, "mma_n": 64}, "mma_m must be a positive integer"),
        ({"mma_m": 128.0, "mma_n": 64}, "mma_m must be a positive integer"),
        ({"mma_m": 64, "mma_n": 64}, "explicit mma_m must match"),
        ({"mma_m": 128, "mma_n": 96}, "explicit mma_n must divide"),
        ({"mma_m": 128, "mma_n": 8}, "Invalid matrix shape"),
    ],
)
def test_gemm_tcgen05_explicit_mma_tile_rejects_invalid_config(mma_config, message):
    target = tvm.target.Target("cuda")
    with target:
        with pytest.raises(Exception, match=message):
            tvm.compile(
                tvm.IRModule(
                    {
                        "main": _build_smem_desc_kernel(
                            "hoist",
                            mma_config=mma_config,
                        )
                    }
                ),
                target=target,
                tir_pipeline="tirx",
            )


@pytest.mark.parametrize("smem_desc", ["hoist", "local_hoist", "encode", "recompute"])
def test_gemm_smem_desc_modes_codegen(smem_desc):
    """Compile-only: the SMEM matrix descriptor is built per-MMA from the buffer
    base address, selected by the ``smem_desc`` config.

    - ``hoist`` (default): allocate + encode one descriptor per operand
      (``descA`` / ``descB``), then add the per-MMA 16B offset via
      ``smem_desc_add_16B_offset``.  This builder uses a single-thread scope,
      where the encoding thread is also the consumer and no warp shuffle is
      valid or needed.
    - ``recompute``: build the full descriptor inline per MMA (``_uniform_desc``)
      with no allocated/encoded descriptor cell — trades a few ALU ops for one
      fewer live register on the hot path.
    - ``local_hoist``: encode the descriptor at the ``gemm_async`` call site,
      inside the caller's elected-thread control flow, then use 16B-offset adds
      like hoist mode without an extra warp shuffle.
    - ``encode``: encode an exact shared pointer for every MMA, then pass the
      resulting descriptor directly to the instruction.

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
        assert "encode_matrix_descriptor" in src, "hoist mode must encode a descriptor"
        assert "smem_desc_make_lo_uniform" not in src, "hoist mode must not warp-shuffle"
        assert "smem_desc_add_16B_offset" in src, "hoist mode must add the per-MMA 16B offset"
    elif smem_desc == "local_hoist":
        assert "descA_local" in src and "descB_local" in src
        assert "smem_desc_add_16B_offset" in src
        assert "encode_matrix_descriptor" in src
    elif smem_desc == "encode":
        assert "encode_matrix_descriptor" in src
        assert "smem_desc_add_16B_offset" not in src
    else:
        assert "smem_desc_make_lo_uniform" not in src, "recompute mode must not hoist a descriptor"
        assert "smem_desc_add_16B_offset" not in src, "recompute mode must not add a 16B offset"
        assert "encode_matrix_descriptor" not in src, "recompute mode must not encode a descriptor"


def test_gemm_tcgen05_weight_stationary_codegen():
    """FlashMLA head64 requires the tcgen05.mma.ws PTX form."""

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": _build_smem_desc_kernel("hoist", weight_stationary=True)}),
            target=target,
            tir_pipeline="tirx",
        )
    src = mod.mod.imports[0].inspect_source()
    assert "tcgen05.mma.ws.cta_group::1.kind::f16" in src
    # No runtime address composition on this path. Match a call rather than the
    # name: the helper's definition can be pulled in by another site in the
    # same module, and a definition is not a use.
    assert not re.search(r"[^ ]tvm_builtin_cuda_get_tmem_addr\(", src)


def test_gemm_tcgen05_dense_descI_rejected():
    """Dense ``descI=`` was removed: the dispatcher self-encodes.

    A hand-passed dense descI performed zero cross-checks against the
    dispatcher-constructed descA/descB majorness — historically this masked
    the col-major-view majorness desync. Passing it must now raise loudly
    (block-scaled gemm_async still accepts descI for the hoisted-encode +
    per-ki sf_id rotation pattern; see deepgemm/mqa_logits_fp4).
    """

    target = tvm.target.Target("cuda")
    with target:
        with pytest.raises(Exception, match="descI was removed"):
            tvm.compile(
                tvm.IRModule(
                    {
                        "main": _build_smem_desc_kernel(
                            "local_hoist", weight_stationary=True, pass_descI=True
                        )
                    }
                ),
                target=target,
                tir_pipeline="tirx",
            )


def _build_cta1_m64_packed_c_kernel(weight_stationary=None, mma_config=None):
    """cta_group::1 M=64 GEMM whose C uses the packed (M, 2, N//2) TMEM layout.

    ``weight_stationary=None`` omits the flag entirely (the dispatch infers
    .ws from the Layout-E C); True/False pass it explicitly.
    """
    ws_cfg = {} if weight_stationary is None else {"weight_stationary": weight_stationary}
    mma_cfg = {} if mma_config is None else mma_config

    M, N, K = 64, 128, 16
    A_dtype = B_dtype = "bfloat16"
    C_dtype = "float32"
    B_layout = mma_shared_layout(B_dtype, SwizzleMode.SWIZZLE_32B_ATOM, (N, K))
    C_layout = TileLayout(S[(M, 2, N // 2) : (1 @ TLane, 64 @ TLane, 1 @ TCol)])

    @T.prim_func
    def gemm_packed_c(B_ptr: T.handle) -> None:
        B = T.match_buffer(B_ptr, (N, K), B_dtype)
        T.device_entry()
        warp_id = T.warp_id([4])
        T.thread_id([128])
        tid = T.thread_id_in_wg([128])
        B_smem = T.alloc_buffer((N, K), B_dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr[0]), T.uint32(512)
            )
        T.cuda.cta_sync()
        # A-in-TMEM .ws reads A from both 64-lane halves, so A is declared in
        # the honest batched A[2, M, K] fold (the A-side).
        A_tmem = T.decl_buffer(
            (2, M, K),
            A_dtype,
            scope="tmem",
            allocated_addr=256,
            layout=TileLayout(S[(2, M, K) : (64 @ TLane, 1 @ TLane, 1 @ TCol)]),
        )
        C_tmem = T.decl_buffer(
            (M, N),
            C_dtype,
            scope="tmem",
            allocated_addr=400,
            layout=C_layout,
        )
        if tid == 0:
            Tx.copy(B_smem[:, :], B[:, :])
            Tx.gemm_async(
                C_tmem[:, :],
                A_tmem[:, :, :],
                B_smem[:, :],
                dispatch="tcgen05",
                cta_group=1,
                **ws_cfg,
                **mma_cfg,
            )

    return gemm_packed_c


def test_gemm_tcgen05_cta1_m64_accepts_packed_c_layout_ws():
    """FlashMLA head64 stores logical N=128 in 64 physical TMEM columns.

    The packed (M, 2, N//2):(1@TLane, 64@TLane, 1@TCol) C layout is the M=64
    ``.ws`` datapath organization (PTX ISA 8.8 §9.7.16.10.5 Layout E,
    cta_group::1), so it is accepted with ``weight_stationary=True``."""

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": _build_cta1_m64_packed_c_kernel(weight_stationary=True)}),
            target=target,
            tir_pipeline="tirx",
        )

    src = mod.mod.imports[0].inspect_source()
    assert "tcgen05.mma.ws.cta_group::1.kind::f16" in src
    assert "tvm_builtin_ptxd_tcgen05_mma_ws_ts_mma_ws_cta_group__1_kind__f16((uint)400," in src
    assert "get_tmem_addr(400, 0, 0)" not in src
    assert "get_tmem_addr(400, 0, 64)" not in src


def test_gemm_tcgen05_explicit_mma_tile_preserves_inferred_ws_datapath():
    src = _compile_cuda_source(
        _build_cta1_m64_packed_c_kernel(
            mma_config={"mma_m": 64, "mma_n": 64},
        )
    )
    assert "tcgen05.mma.ws.cta_group::1.kind::f16" in src
    helper = "tvm_builtin_ptxd_tcgen05_mma_ws_ts_mma_ws_cta_group__1_kind__f16("
    calls = [line for line in src.splitlines() if line.lstrip().startswith(helper)]
    assert len(calls) == 2


def _build_cta1_m64_batched_c_kernel():
    """cta_group::1 M=64 GEMM whose C is the honest batched form C[2, M, N//2]
    (the leading dim is the Layout-E lane fold), instead of the packed
    C[M, 2, N//2]."""

    M, N, K = 64, 128, 16
    B_dtype = "bfloat16"
    B_layout = mma_shared_layout(B_dtype, SwizzleMode.SWIZZLE_32B_ATOM, (N, K))
    C_layout = TileLayout(S[(2, M, N // 2) : (64 @ TLane, 1 @ TLane, 1 @ TCol)])

    @T.prim_func
    def gemm_batched_c(B_ptr: T.handle) -> None:
        B = T.match_buffer(B_ptr, (N, K), B_dtype)
        T.device_entry()
        warp_id = T.warp_id([4])
        T.thread_id([128])
        tid = T.thread_id_in_wg([128])
        B_smem = T.alloc_buffer((N, K), B_dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr[0]), T.uint32(512)
            )
        T.cuda.cta_sync()
        # Honest batched A[2, M, K] fold (A-side), matching the
        # batched C below: both banks of the M=64 .ws are explicit.
        A_tmem = T.decl_buffer(
            (2, M, K),
            B_dtype,
            scope="tmem",
            allocated_addr=256,
            layout=TileLayout(S[(2, M, K) : (64 @ TLane, 1 @ TLane, 1 @ TCol)]),
        )
        C_tmem = T.decl_buffer(
            (2, M, N // 2),
            "float32",
            scope="tmem",
            allocated_addr=400,
            layout=C_layout,
        )
        if tid == 0:
            Tx.copy(B_smem[:, :], B[:, :])
            Tx.gemm_async(
                # No weight_stationary: the dispatch infers .ws from the
                # batched C[2, M, N] fold layout.
                C_tmem[:, :, :],
                A_tmem[:, :, :],
                B_smem[:, :],
                dispatch="tcgen05",
                cta_group=1,
            )

    return gemm_batched_c


def _build_cta1_m64_identity_c_ws_kernel():
    """M=64 cta_group::1 gemm forcing .ws but declaring C in the identity
    (Layout-D) layout instead of the Layout-E fold — the dual
    error the dispatch must reject."""

    M, N, K = 64, 128, 16
    B_dtype = "bfloat16"
    B_layout = mma_shared_layout(B_dtype, SwizzleMode.SWIZZLE_32B_ATOM, (N, K))

    @T.prim_func
    def gemm_identity_c(B_ptr: T.handle) -> None:
        B = T.match_buffer(B_ptr, (N, K), B_dtype)
        T.device_entry()
        warp_id = T.warp_id([4])
        T.thread_id([128])
        tid = T.thread_id_in_wg([128])
        B_smem = T.alloc_buffer((N, K), B_dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr[0]), T.uint32(512)
            )
        T.cuda.cta_sync()
        A_tmem = T.decl_buffer(
            (M, K),
            B_dtype,
            scope="tmem",
            allocated_addr=256,
            layout=TileLayout(S[(M, K) : (1 @ TLane, 1 @ TCol)]),
        )
        C_tmem = T.decl_buffer(
            (M, N),
            "float32",
            scope="tmem",
            allocated_addr=400,
            layout=TileLayout(S[(M, N) : (1 @ TLane, 1 @ TCol)]),
        )
        if tid == 0:
            Tx.copy(B_smem[:, :], B[:, :])
            Tx.gemm_async(
                C_tmem[:, :],
                A_tmem[:, :],
                B_smem[:, :],
                dispatch="tcgen05",
                cta_group=1,
                weight_stationary=True,
            )

    return gemm_identity_c


def test_gemm_tcgen05_cta1_m64_ws_requires_layout_e_c():
    """A forced .ws (M=64, cta_group::1) with an identity/Layout-F C is
    rejected: .ws writes the Layout-E fold, so an identity C would read the
    two banks from the wrong lanes (dual error)."""

    target = tvm.target.Target("cuda")
    with target:
        with pytest.raises(Exception, match="Layout-E"):
            tvm.compile(
                tvm.IRModule({"main": _build_cta1_m64_identity_c_ws_kernel()}),
                target=target,
                tir_pipeline="tirx",
            )


def _build_cta1_m64_flat_a_ws_kernel():
    """M=64 cta_group::1 .ws with A in TMEM declared as a flat 2D [M, K]
    identity (lanes 0-63 only) instead of the batched A[2, M, K] fold — the
    A-side error the dispatch must reject (the A-side dual of the
    identity-C reject above)."""

    M, N, K = 64, 128, 16
    B_dtype = "bfloat16"
    B_layout = mma_shared_layout(B_dtype, SwizzleMode.SWIZZLE_32B_ATOM, (N, K))
    C_layout = TileLayout(S[(M, 2, N // 2) : (1 @ TLane, 64 @ TLane, 1 @ TCol)])

    @T.prim_func
    def gemm_flat_a(B_ptr: T.handle) -> None:
        B = T.match_buffer(B_ptr, (N, K), B_dtype)
        T.device_entry()
        warp_id = T.warp_id([4])
        T.thread_id([128])
        tid = T.thread_id_in_wg([128])
        B_smem = T.alloc_buffer((N, K), B_dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr[0]), T.uint32(512)
            )
        T.cuda.cta_sync()
        A_tmem = T.decl_buffer(
            (M, K),
            B_dtype,
            scope="tmem",
            allocated_addr=256,
            layout=TileLayout(S[(M, K) : (1 @ TLane, 1 @ TCol)]),
        )
        C_tmem = T.decl_buffer(
            (M, N),
            "float32",
            scope="tmem",
            allocated_addr=400,
            layout=C_layout,
        )
        if tid == 0:
            Tx.copy(B_smem[:, :], B[:, :])
            Tx.gemm_async(
                C_tmem[:, :],
                A_tmem[:, :],
                B_smem[:, :],
                dispatch="tcgen05",
                cta_group=1,
                weight_stationary=True,
            )

    return gemm_flat_a


def test_gemm_tcgen05_cta1_m64_ws_requires_batched_a():
    """A forced .ws (M=64, cta_group::1) with A in TMEM declared as a flat 2D
    [M, K] identity is rejected: the .ws reads A from both 64-lane halves, so a
    flat A (lanes 0-63 only) cannot express — nor let the dispatch verify — the
    second-half occupancy. A must use the batched A[2, M, K] fold."""

    target = tvm.target.Target("cuda")
    with target:
        with pytest.raises(Exception, match="both 64-lane halves"):
            tvm.compile(
                tvm.IRModule({"main": _build_cta1_m64_flat_a_ws_kernel()}),
                target=target,
                tir_pipeline="tirx",
            )


def _build_m128_batched_a_kernel():
    """M=128 cta_group::1 NON-ws gemm whose A is (illegitimately) declared in the
    batched A[2, M, K] fold. The batched fold is defined *only* for the M=64
    cta_group::1 .ws datapath, so it must be rejected here (the converse of the
    requires-batched-A reject: batched A only for M=64 .ws)."""

    M, N, K = 128, 128, 16
    B_dtype = "bfloat16"
    B_layout = mma_shared_layout(B_dtype, SwizzleMode.SWIZZLE_32B_ATOM, (N, K))

    @T.prim_func
    def gemm_m128_batched_a(B_ptr: T.handle) -> None:
        B = T.match_buffer(B_ptr, (N, K), B_dtype)
        T.device_entry()
        warp_id = T.warp_id([4])
        T.thread_id([128])
        tid = T.thread_id_in_wg([128])
        B_smem = T.alloc_buffer((N, K), B_dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        if warp_id == 0:
            T.ptxd.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr[0]), T.uint32(512)
            )
        T.cuda.cta_sync()
        A_tmem = T.decl_buffer(
            (2, M, K),
            B_dtype,
            scope="tmem",
            allocated_addr=256,
            layout=TileLayout(S[(2, M, K) : (64 @ TLane, 1 @ TLane, 1 @ TCol)]),
        )
        C_tmem = T.decl_buffer(
            (M, N),
            "float32",
            scope="tmem",
            allocated_addr=400,
            layout=TileLayout(S[(M, N) : (1 @ TLane, 1 @ TCol)]),
        )
        if tid == 0:
            Tx.copy(B_smem[:, :], B[:, :])
            Tx.gemm_async(
                C_tmem[:, :],
                A_tmem[:, :, :],
                B_smem[:, :],
                dispatch="tcgen05",
                cta_group=1,
            )

    return gemm_m128_batched_a


def test_gemm_tcgen05_batched_a_rejects_unproven_datapath():
    """A batched A[2, M, K] fold outside Layout E / Layout B is rejected."""

    target = tvm.target.Target("cuda")
    with target:
        with pytest.raises(Exception, match="only valid for Layout E"):
            tvm.compile(
                tvm.IRModule({"main": _build_m128_batched_a_kernel()}),
                target=target,
                tir_pipeline="tirx",
            )


def test_gemm_tcgen05_cta1_m64_accepts_batched_c_layout_ws():
    """The honest batched C[2, M, N//2] .ws output form is accepted and emits
    byte-identically to the packed C[M, 2, N//2] form: the two
    describe the same physical Layout-E tile."""

    target = tvm.target.Target("cuda")

    def _compile(kernel):
        with target:
            mod = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
        return mod.mod.imports[0].inspect_source()

    batched_src = _compile(_build_cta1_m64_batched_c_kernel())
    packed_src = _compile(_build_cta1_m64_packed_c_kernel(weight_stationary=True))
    assert "tcgen05.mma.ws.cta_group::1.kind::f16" in batched_src

    # identical up to the kernel entry name (gemm_batched_c vs gemm_packed_c)
    def norm(s):
        return s.replace("gemm_batched_c", "K").replace("gemm_packed_c", "K")

    assert norm(batched_src) == norm(packed_src)


def test_gemm_tcgen05_cta1_m64_packed_c_infers_weight_stationary():
    """The packed (M, 2, N//2):(1@TLane, 64@TLane, 1@TCol) Layout-E C is
    uniquely the M=64 .ws datapath (PTX ISA 8.8 §9.7.16.10.5), so .ws is
    inferred with no weight_stationary flag; and an explicit
    weight_stationary=False contradicts the layout and is rejected."""

    target = tvm.target.Target("cuda")
    with target:
        # omitted flag -> .ws inferred, compiles to a .ws MMA
        mod = tvm.compile(
            tvm.IRModule({"main": _build_cta1_m64_packed_c_kernel()}),
            target=target,
            tir_pipeline="tirx",
        )
        assert "tcgen05.mma.ws.cta_group::1.kind::f16" in mod.mod.imports[0].inspect_source()
        # explicit False contradicts the Layout-E fold -> rejected
        with pytest.raises(Exception, match="weight_stationary=False"):
            tvm.compile(
                tvm.IRModule({"main": _build_cta1_m64_packed_c_kernel(weight_stationary=False)}),
                target=target,
                tir_pipeline="tirx",
            )


# Dispatch-level regression tests: call gemm_async_tcgen05_impl directly on a
# constructed TilePrimitiveCall to pin rejection paths without full compilation.


def _make_gemm_tcgen05_call(
    M,
    N,
    K,
    dtype,
    A_layout,
    B_layout,
    transA=False,
    transB=True,
    config=None,
    scope_kind="thread",
    return_context=False,
    C_layout=None,
    C_allocated_addr=0,
    A_scope="shared.dyn",
    A_allocated_addr=0,
):
    """Construct a GemmAsync TilePrimitiveCall and run the tcgen05 dispatch.

    Buffer-shape convention follows the dispatcher: transA=False -> A is
    [M, K]; transB=True -> B is [K, N], transB=False -> B is [N, K].
    C is a full-region (M, N) float32 TMEM buffer with the identity
    (1@TLane, 1@TCol) layout.
    """
    from tvm.ir import Range
    from tvm.tirx.cuda.tile_primitive.gemm_async.tcgen05 import (
        gemm_async_tcgen05_impl,
    )
    from tvm.tirx.exec_scope import ExecScope
    from tvm.tirx.operator.tile_primitive.ops import GemmAsync
    from tvm.tirx.stmt import BufferRegion
    from tvm.tirx.tile_primitive import DispatchContext

    def full_region(buf):
        return BufferRegion(buf, [Range.from_min_extent(0, s) for s in buf.shape])

    A_shape = (M, K) if not transA else (K, M)
    B_shape = (K, N) if transB else (N, K)
    A_buf = tvm.tirx.decl_buffer(A_shape, dtype, "A", scope=A_scope, layout=A_layout)
    if A_scope == "tmem":
        A_buf = A_buf.with_allocated_addr([tvm.tirx.IntImm("uint32", A_allocated_addr)])
    B_buf = tvm.tirx.decl_buffer(B_shape, dtype, "B_smem", scope="shared.dyn", layout=B_layout)
    if C_layout is None:
        C_layout = TileLayout(S[(M, N) : (1 @ TLane, 1 @ TCol)])
    C_buf = tvm.tirx.decl_buffer((M, N), "float32", "C_tmem", scope="tmem", layout=C_layout)
    C_buf = C_buf.with_allocated_addr([tvm.tirx.IntImm("uint32", C_allocated_addr)])
    call = GemmAsync(
        full_region(C_buf),
        full_region(A_buf),
        full_region(B_buf),
        transA,
        transB,
        False,
        config=dict(config or {}),
    )
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    sctx = DispatchContext(target, ExecScope(scope_kind), {}, {}, scope_kind=scope_kind)
    impl = gemm_async_tcgen05_impl(call, sctx)
    return (impl, sctx) if return_context else impl


def test_gemm_tcgen05_preserves_explicit_tmem_lane_bases():
    """D and TMEM-A row offsets are part of their physical taddr operands."""
    M, N, K = 64, 64, 16
    dtype = "bfloat16"
    B_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_32B_ATOM, (K, N))

    d_base = tmem_datapath_layout("F", M, N)
    d_offset = TileLayout.from_iters(d_base.shard, d_base.replica, {TLane: 1})
    A_smem_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_32B_ATOM, (M, K))
    d_impl = _make_gemm_tcgen05_call(
        M,
        N,
        K,
        dtype,
        A_smem_layout,
        B_layout,
        C_layout=d_offset,
        C_allocated_addr=400,
        config={"mma_m": M, "mma_n": N},
    )
    assert "T.cuda.get_tmem_addr(T.uint32(400), 1, ni * 64)" in d_impl.script()

    a_base = TileLayout(S[(M, K) : (1 @ TLane, 1 @ TCol)])
    a_offset = TileLayout.from_iters(a_base.shard, a_base.replica, {TLane: 1})
    a_impl = _make_gemm_tcgen05_call(
        M,
        N,
        K,
        dtype,
        a_offset,
        B_layout,
        C_layout=d_base,
        C_allocated_addr=400,
        A_scope="tmem",
        A_allocated_addr=256,
        config={"mma_m": M, "mma_n": N},
    )
    assert "T.cuda.get_tmem_addr(T.uint32(256), 1, ki * 8)" in a_impl.script()


def test_gemm_tcgen05_preserves_block_scale_tmem_lane_bases():
    """SFA/SFB row offsets must reach the encoded TMEM address operands."""
    from tvm.ir import Range
    from tvm.tirx.cuda.tile_primitive.gemm_async.tcgen05 import (
        gemm_async_tcgen05_impl,
    )
    from tvm.tirx.exec_scope import ExecScope
    from tvm.tirx.operator.tile_primitive.ops import GemmAsync
    from tvm.tirx.stmt import BufferRegion
    from tvm.tirx.tile_primitive import DispatchContext

    def full_region(buf):
        return BufferRegion(buf, [Range.from_min_extent(0, s) for s in buf.shape])

    M, N, K = 128, 64, 64
    data_dtype = "float4_e2m1fn"
    sf_dtype = "float8_e4m3fn"
    A = tvm.tirx.decl_buffer(
        (M, K),
        data_dtype,
        "A_smem",
        scope="shared.dyn",
        layout=mma_shared_layout(data_dtype, SwizzleMode.SWIZZLE_32B_ATOM, (M, K)),
    )
    B = tvm.tirx.decl_buffer(
        (N, K),
        data_dtype,
        "B_smem",
        scope="shared.dyn",
        layout=mma_shared_layout(data_dtype, SwizzleMode.SWIZZLE_32B_ATOM, (N, K)),
    )
    C = tvm.tirx.decl_buffer(
        (M, N),
        "float32",
        "C_tmem",
        scope="tmem",
        layout=tmem_datapath_layout("D", M, N),
    ).with_allocated_addr([tvm.tirx.IntImm("uint32", 0)])

    sf_per_mma = 4
    sfa_base = sf_tmem_layout(M, SF_K=sf_per_mma, sf_per_mma=sf_per_mma)
    sfb_base = sf_tmem_layout(N, SF_K=sf_per_mma, sf_per_mma=sf_per_mma)
    sfa_layout = TileLayout.from_iters(sfa_base.shard, sfa_base.replica, {TLane: 1})
    sfb_layout = TileLayout.from_iters(sfb_base.shard, sfb_base.replica, {TLane: 2})
    SFA = tvm.tirx.decl_buffer(
        (M, sf_per_mma), sf_dtype, "SFA_tmem", scope="tmem", layout=sfa_layout
    ).with_allocated_addr([tvm.tirx.IntImm("uint32", 256)])
    SFB = tvm.tirx.decl_buffer(
        (N, sf_per_mma), sf_dtype, "SFB_tmem", scope="tmem", layout=sfb_layout
    ).with_allocated_addr([tvm.tirx.IntImm("uint32", 320)])

    call = GemmAsync(
        full_region(C),
        full_region(A),
        full_region(B),
        full_region(SFA),
        full_region(SFB),
        False,
        False,
        False,
        config={"cta_group": 1, "mma_m": M, "mma_n": N},
    )
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    sctx = DispatchContext(target, ExecScope("thread"), {}, {}, scope_kind="thread")
    script = gemm_async_tcgen05_impl(call, sctx).script()

    assert "T.cuda.get_tmem_addr(T.uint32(256), 1, 0)" in script
    assert "T.cuda.get_tmem_addr(T.uint32(320), 2, 0)" in script


@pytest.mark.parametrize(
    ("scope_kind", "expect_uniform"),
    [("thread", False), ("warp", True)],
)
def test_gemm_tcgen05_hoisted_descriptor_uniformization(scope_kind, expect_uniform):
    M, N, K = 64, 256, 64
    dtype = "bfloat16"
    A_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (M, K))
    B_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (K, N))
    impl, sctx = _make_gemm_tcgen05_call(
        M,
        N,
        K,
        dtype,
        A_layout,
        B_layout,
        config={"smem_desc": "hoist"},
        scope_kind=scope_kind,
        return_context=True,
    )

    callback_text = str(sctx.callbacks["post_buffer_def_stmt"])
    assert "encode_matrix_descriptor" in callback_text
    assert ("smem_desc_make_lo_uniform" in callback_text) == expect_uniform
    assert ("elect_sync" in impl.script()) == expect_uniform
    if expect_uniform:
        assert callback_text.index("smem_desc_make_lo_uniform") < callback_text.index(
            "tvm_kernel_replace_point"
        )


def test_gemm_tcgen05_no_swizzle_col_major_rejects_non_128B_contiguous():
    """the col-major-view no-swizzle branch encodes the SBO field as
    the literal 8 (128B row-group pitch), which only matches the packed ABI
    when the contiguous dim spans exactly 128B (64 bf16 elements). A larger
    contiguous dim (M=128 here) used to be accepted with a silently wrong
    ``sdo = shape // elem_per_16B = 16`` — it must now be rejected."""
    M, N, K = 128, 256, 64
    dtype = "bfloat16"
    A_layout = TileLayout(S[(M, K) : (1, M)])  # column-major view, contiguous dim 128
    B_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (K, N))
    with pytest.raises(ValueError, match="span exactly 128B"):
        _make_gemm_tcgen05_call(M, N, K, dtype, A_layout, B_layout)

    # In-domain sanity: M=64 col-major view still dispatches.
    A_ok = TileLayout(S[(64, K) : (1, 64)])
    impl = _make_gemm_tcgen05_call(64, N, K, dtype, A_ok, B_layout)
    assert impl is not None


def test_gemm_tcgen05_no_swizzle_packed_16b_rejects_non_16bit_dtype():
    """the packed-16B no-swizzle branch used to encode
    ``sdo = elem_per_16B``, which equals the true SBO field (8 = 128B row
    pitch, PTX ISA §9.7.16.3.2) only for 16-bit dtypes. An fp8 packed-16B
    layout (elem_per_16B = 16) used to be accepted with sdo=16 — it must now
    be rejected (that domain is not hardware-validated)."""
    M, N, K = 64, 256, 64
    dtype = "float8_e4m3fn"  # 16 elements per 16B line
    A_layout = TileLayout(S[(M, K // 16, 16) : (16, M * 16, 1)])
    B_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (K, N))
    with pytest.raises(ValueError, match="16-bit dtypes"):
        _make_gemm_tcgen05_call(M, N, K, dtype, A_layout, B_layout)

    # In-domain sanity: the bf16 packed-16B layout still dispatches.
    bf16_A = TileLayout(S[(M, K // 8, 8) : (8, M * 8, 1)])
    bf16_B = mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_128B_ATOM, (K, N))
    impl = _make_gemm_tcgen05_call(M, N, K, "bfloat16", bf16_A, bf16_B)
    assert impl is not None


def test_gemm_tcgen05_instr_desc_fold_mirrors_runtime_shape_rules():
    """the compile-time descI fold must run the runtime encoder's
    shape validation. cta_group=1 with descriptor M=128 requires N % 16 == 0;
    the tile chooser alone only guarantees N % 8, so M=128/N=24 used to fold
    a descriptor the runtime encoder would reject."""
    M, N, K = 128, 24, 64
    dtype = "bfloat16"
    A_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (M, K))
    B_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (N, K))
    with pytest.raises(ValueError, match="Invalid matrix shape"):
        _make_gemm_tcgen05_call(M, N, K, dtype, A_layout, B_layout, transB=False)

    # N=32 (divisible by 16) with the same M=128 tile is accepted.
    B_ok = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (32, K))
    impl = _make_gemm_tcgen05_call(M, 32, K, dtype, A_layout, B_ok, transB=False)
    assert impl is not None


def test_gemm_tcgen05_rejects_tf32_mn_major():
    """tf32 MN-major operands are PTX-illegal without the
    128B-32B-atomicity swizzle, which the atom matcher never produces. A
    tf32 (is_AB_tf32) B operand matching MN-major used to be silently
    encoded; it must now be rejected."""
    M, N, K = 64, 256, 32
    dtype = "float32"
    A_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (M, K))
    # transB=True means B is [K, N]; a row-major swizzled layout then has the
    # MN (=N) dim contiguous, i.e. it matches as MN-major.
    B_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (K, N))
    with pytest.raises(ValueError, match="PTX-illegal"):
        _make_gemm_tcgen05_call(
            M, N, K, dtype, A_layout, B_layout, transB=True, config={"is_AB_tf32": True}
        )

    # MN-major stays accepted for a dtype where it is PTX-legal (bf16).
    bf16_A = mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_128B_ATOM, (M, 64))
    bf16_B = mma_shared_layout("bfloat16", SwizzleMode.SWIZZLE_128B_ATOM, (64, N))
    impl = _make_gemm_tcgen05_call(M, N, 64, "bfloat16", bf16_A, bf16_B, transB=True)
    assert impl is not None


def test_gemm_tcgen05_rejects_non_uniform_atom_grid():
    """``_try_atom`` reads only the innermost iter of each tiler
    dimension for the LBO/SBO fields, so each dimension must group into a
    single iter. A swizzled layout whose M-direction atom tiling has a
    stride gap (atoms (2,4) with outer stride 4096 instead of 2048) used to
    match with fields taken from the local inner stride, silently dropping
    the gap; it must now be rejected."""
    from tvm.tirx.layout import ComposeLayout

    M, N, K = 64, 256, 64
    dtype = "bfloat16"
    exotic_A = ComposeLayout(3, 3, 3, TileLayout(S[(2, 4, 8, 64) : (4096, 512, 64, 1)]))
    B_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (K, N))
    with pytest.raises(ValueError, match="no MMA SMEM descriptor matches"):
        _make_gemm_tcgen05_call(M, N, K, dtype, exotic_A, B_layout)

    # The uniform version of the same tiling (outer stride 2048) is accepted.
    uniform_A = ComposeLayout(3, 3, 3, TileLayout(S[(2, 4, 8, 64) : (2048, 512, 64, 1)]))
    impl = _make_gemm_tcgen05_call(M, N, K, dtype, uniform_A, B_layout)
    assert impl is not None


def test_gemm_tcgen05_dense_descI_rejected_at_dispatch():
    """Dispatch-level twin of the compile-path test above."""
    M, N, K = 64, 256, 64
    dtype = "bfloat16"
    A_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (M, K))
    B_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (K, N))
    with pytest.raises(ValueError, match="descI was removed"):
        _make_gemm_tcgen05_call(
            M,
            N,
            K,
            dtype,
            A_layout,
            B_layout,
            config={"descI": tvm.tirx.const(0x04410490, "uint32")},
        )


if __name__ == "__main__":
    tvm.testing.main()
