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
"""Tests for the TMEM copy_async dispatch (tcgen05-based tmem<->reg and smem<->tmem)."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx.layout import S, TCol, TileLayout, TLane
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("width_32b", [4, 8, 16, 32])
def test_copy_tmem2reg_async(dtype, width_32b):
    """Test async tmem<->local copy using copy_async instead of copy.

    This tests the new copy_async dispatch for tmem<->local that doesn't
    immediately wait after the operation, allowing for pipelining.
    """

    def next_power_of_2(x):
        """Return the smallest power of 2 greater than or equal to x."""
        if x <= 1:
            return 1
        return 1 << (x - 1).bit_length()

    bits = tvm.runtime.DataType(dtype).bits
    if 128 % bits != 0 or 32 % bits != 0:
        pytest.skip(f"dtype {dtype} is not supported")

    WIDTH = width_32b * (32 // bits)
    VEC_LEN = 128 // bits
    if WIDTH % VEC_LEN != 0:
        pytest.skip(f"dtype {dtype} + width {width_32b} is not supported")

    g_layout = TileLayout(S[(128, WIDTH // VEC_LEN, VEC_LEN) : (WIDTH, VEC_LEN, 1)])
    local_view = TileLayout(S[(128, WIDTH) : (1 @ axis_tid_in_wg, 1)])

    # fmt: off
    @T.prim_func
    def copy_async_test(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, WIDTH), dtype)
        B = T.match_buffer(B_ptr, (128, WIDTH), dtype)

        A_flat = A.view(-1)
        B_flat = B.view(-1)

        T.device_entry()
        warp_id = T.warp_id([(128) // 32])
        cta_id = T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        warp_id_in_wg = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
        tid_in_wg = T.thread_id([128])

        tmem_addr = T.alloc_shared([1], "uint32")

        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)  # noqa: E501

            T.tvm_storage_sync("shared")

            tmem = T.decl_buffer((128, WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],
                                 layout=TileLayout(S[(128, WIDTH) : (1 @ TLane, 1 @ TCol)]))

            A_reg = T.alloc_local((WIDTH), dtype)
            B_reg = T.alloc_local((WIDTH), dtype)
            A_local = A_reg.view(128, WIDTH, layout=local_view)
            B_local = B_reg.view(128, WIDTH, layout=local_view)
            for i in range(WIDTH // VEC_LEN):
                g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                Tx.copy(A_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN], A_flat[g_offset: g_offset + VEC_LEN])  # noqa: E501
            for i in range(WIDTH):
                B_reg[i] = T.cast(0, dtype)
            T.cuda.cta_sync()

                    # A_local -> tmem (async)
            Tx.wg.copy_async(tmem[:, :], A_local[:, :])
            T.ptx.tcgen05.wait.st()  # explicit wait
            T.cuda.cta_sync()

                    # tmem -> B_local (async)
            Tx.wg.copy_async(B_local[:, :], tmem[:, :])
            T.ptx.tcgen05.wait.ld()  # explicit wait
            T.cuda.cta_sync()
            for i in range(WIDTH // VEC_LEN):
                g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                Tx.copy(B_flat[g_offset: g_offset + VEC_LEN], B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])  # noqa: E501

            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)  # noqa: E501
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_async_test})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, WIDTH))
        B_np = np.zeros((128, WIDTH), dtype=dtype)

        def run_test():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            dev.sync()
            np.testing.assert_allclose(B.numpy(), A_np)

        tvm.testing.run_with_gpu_lock(run_test)


# ----------------------------------------------------------------------------
# Migrated from test_copy_sync.py: tmem<->reg round-trip via T.copy_async
# (the kernels themselves are the actual async tmem dispatch tests; the
# G↔L copies bookending them just stage data).
# ----------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("dtype", ["uint8", "float16", "float32"])
@pytest.mark.parametrize("width_32b", [2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("offset_32b", [0, 3, 10])
def test_copy_tmem2reg(dtype, width_32b, offset_32b):
    def next_power_of_2(x):
        if x <= 1:
            return 1
        return 1 << (x - 1).bit_length()

    bits = tvm.runtime.DataType(dtype).bits
    if 128 % bits != 0 or 32 % bits != 0:
        pytest.skip(f"dtype {dtype} is not supported")

    WIDTH = width_32b * (32 // bits)
    OFFSET = offset_32b * (32 // bits)
    VEC_LEN = 128 // bits
    if WIDTH % VEC_LEN != 0:
        pytest.skip(f"dtype {dtype} + width {width_32b} is not supported")

    g_layout = TileLayout(S[(128, WIDTH // VEC_LEN, VEC_LEN) : (WIDTH, VEC_LEN, 1)])
    local_view = TileLayout(S[(128, WIDTH) : (1 @ axis_tid_in_wg, 1)])

    # fmt: off
    @T.prim_func
    def copy_sync(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, WIDTH), dtype)
        B = T.match_buffer(B_ptr, (128, WIDTH), dtype)

        A_flat = A.view(-1)
        B_flat = B.view(-1)

        T.device_entry()
        warp_id = T.warp_id([(128) // 32])
        T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        tid_in_wg = T.thread_id([128])

        tmem_addr = T.alloc_shared([1], "uint32")

        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=max(32, next_power_of_2(offset_32b + width_32b)), cta_group=1)  # noqa: E501

            T.tvm_storage_sync("shared")

            tmem = T.decl_buffer((128, OFFSET + WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],  # noqa: E501
                                 layout=TileLayout(S[(128, OFFSET + WIDTH) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501

            A_reg = T.alloc_local((WIDTH), dtype)
            B_reg = T.alloc_local((WIDTH), dtype)
            A_local = A_reg.view(128, WIDTH, layout=local_view)
            B_local = B_reg.view(128, WIDTH, layout=local_view)
            for i in range(WIDTH // VEC_LEN):
                g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                Tx.copy(A_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN], A_flat[g_offset: g_offset + VEC_LEN])  # noqa: E501
            for i in range(WIDTH):
                B_reg[i] = T.cast(0, dtype)
            T.cuda.cta_sync()

                    # A_local -> tmem
            Tx.wg.copy_async(tmem[:, OFFSET: OFFSET + WIDTH], A_local[:, :])
            T.ptx.tcgen05.wait.st()
            T.cuda.cta_sync()

                    # tmem -> B_local
            Tx.wg.copy_async(B_local[:, :], tmem[:, OFFSET: OFFSET + WIDTH])
            T.ptx.tcgen05.wait.ld()
            T.cuda.cta_sync()
            for i in range(WIDTH // VEC_LEN):
                g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                Tx.copy(B_flat[g_offset: g_offset + VEC_LEN], B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])  # noqa: E501

            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=max(32, next_power_of_2(offset_32b + width_32b)), cta_group=1)  # noqa: E501
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_sync})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, WIDTH))
        B_np = np.zeros((128, WIDTH), dtype=dtype)

        def run_test():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            dev.sync()
            np.testing.assert_allclose(B.numpy(), A_np)

        tvm.testing.run_with_gpu_lock(run_test)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("width_32b", [4, 8, 16, 32])
@pytest.mark.parametrize("local_offset_32b", [0, 2, 4])
def test_copy_tmem2reg_sliced_local(dtype, width_32b, local_offset_32b):
    """tmem<->local copy with a sliced local buffer region."""

    def next_power_of_2(x):
        if x <= 1:
            return 1
        return 1 << (x - 1).bit_length()

    bits = tvm.runtime.DataType(dtype).bits
    if 128 % bits != 0 or 32 % bits != 0:
        pytest.skip(f"dtype {dtype} is not supported")

    WIDTH = width_32b * (32 // bits)
    LOCAL_OFFSET = local_offset_32b * (32 // bits)
    TOTAL_LOCAL_WIDTH = WIDTH + LOCAL_OFFSET
    VEC_LEN = 128 // bits
    if WIDTH % VEC_LEN != 0 or TOTAL_LOCAL_WIDTH % VEC_LEN != 0:
        pytest.skip(
            f"dtype {dtype} + width {width_32b} + offset {local_offset_32b} is not supported"
        )

    g_layout = TileLayout(S[(128, WIDTH // VEC_LEN, VEC_LEN) : (WIDTH, VEC_LEN, 1)])
    local_view = TileLayout(S[(128, TOTAL_LOCAL_WIDTH) : (1 @ axis_tid_in_wg, 1)])

    # fmt: off
    @T.prim_func
    def copy_sync(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, WIDTH), dtype)
        B = T.match_buffer(B_ptr, (128, WIDTH), dtype)

        A_flat = A.view(-1)
        B_flat = B.view(-1)

        T.device_entry()
        warp_id = T.warp_id([(128) // 32])
        T.cta_id([2])
        wg_id = T.warpgroup_id([1])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        tid_in_wg = T.thread_id([128])

        tmem_addr = T.alloc_shared([1], "uint32")

        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)  # noqa: E501

            T.tvm_storage_sync("shared")

            tmem = T.decl_buffer((128, WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],
                                 layout=TileLayout(S[(128, WIDTH) : (1 @ TLane, 1 @ TCol)]))

            A_reg = T.alloc_local((TOTAL_LOCAL_WIDTH), dtype)
            B_reg = T.alloc_local((TOTAL_LOCAL_WIDTH), dtype)
            A_local = A_reg.view(128, TOTAL_LOCAL_WIDTH, layout=local_view)
            B_local = B_reg.view(128, TOTAL_LOCAL_WIDTH, layout=local_view)
            for i in range(WIDTH // VEC_LEN):
                g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                Tx.copy(A_reg[LOCAL_OFFSET + i * VEC_LEN: LOCAL_OFFSET + i * VEC_LEN + VEC_LEN], A_flat[g_offset: g_offset + VEC_LEN])  # noqa: E501
            for i in range(TOTAL_LOCAL_WIDTH):
                B_reg[i] = T.cast(0, dtype)
            T.cuda.cta_sync()

                    # A_local[sliced] -> tmem (use sliced region)
            Tx.wg.copy_async(tmem[:, 0:WIDTH], A_local[:, LOCAL_OFFSET:LOCAL_OFFSET + WIDTH])
            T.ptx.tcgen05.wait.st()
            T.cuda.cta_sync()

                    # tmem -> B_local[sliced] (use sliced region)
            Tx.wg.copy_async(B_local[:, LOCAL_OFFSET:LOCAL_OFFSET + WIDTH], tmem[:, 0:WIDTH])
            T.ptx.tcgen05.wait.ld()
            T.cuda.cta_sync()
            for i in range(WIDTH // VEC_LEN):
                g_offset = T.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                Tx.copy(B_flat[g_offset: g_offset + VEC_LEN], B_reg[LOCAL_OFFSET + i * VEC_LEN: LOCAL_OFFSET + i * VEC_LEN + VEC_LEN])  # noqa: E501

            if warp_id == 0:
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)  # noqa: E501
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_sync})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, WIDTH))
        B_np = np.zeros((128, WIDTH), dtype=dtype)

        def run_test():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            dev.sync()
            np.testing.assert_allclose(B.numpy(), A_np)

        tvm.testing.run_with_gpu_lock(run_test)


if __name__ == "__main__":
    tvm.testing.main()
