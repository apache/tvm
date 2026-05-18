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
from tvm.script import tirx as Tx
from tvm.tirx.layout import S, TCol, TileLayout, TLane
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg


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
    @Tx.prim_func
    def copy_async_test(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128, WIDTH), dtype)
        B = Tx.match_buffer(B_ptr, (128, WIDTH), dtype)

        A_flat = A.view(-1)
        B_flat = B.view(-1)

        with Tx.kernel():
            warp_id = Tx.warp_id([(128) // 32])
            cta_id = Tx.cta_id([2])
            wg_id = Tx.warpgroup_id([1])
            warp_id_in_wg = Tx.warp_id_in_wg([4])
            lane_id = Tx.lane_id([32])
            tid_in_wg = Tx.thread_id([128])

            tmem_addr = Tx.alloc_shared([1], "uint32")

            if Tx.filter(wg_id, 0, 1):
                with Tx.warpgroup():
                    if Tx.filter(warp_id, 0, 1):
                        with Tx.warp():
                            Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)  # noqa: E501

                    Tx.tvm_storage_sync("shared")

                    tmem = Tx.decl_buffer((128, WIDTH), dtype, scope="tmem", allocated_addr=tmem_addr[0],  # noqa: E501
                                         layout=TileLayout(S[(128, WIDTH) : (1 @ TLane, 1 @ TCol)]))

                    A_reg = Tx.alloc_local((WIDTH), dtype)
                    B_reg = Tx.alloc_local((WIDTH), dtype)
                    A_local = A_reg.view(128, WIDTH, layout=local_view)
                    B_local = B_reg.view(128, WIDTH, layout=local_view)

                    # A -> A_local
                    with Tx.thread():
                        for i in range(WIDTH // VEC_LEN):
                            g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                            Tx.copy(A_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN], A_flat[g_offset: g_offset + VEC_LEN])  # noqa: E501
                        for i in range(WIDTH):
                            B_reg[i] = Tx.cast(0, dtype)
                    Tx.cuda.cta_sync()

                    # A_local -> tmem (async)
                    Tx.copy_async(tmem[:, :], A_local[:, :])
                    Tx.ptx.tcgen05.wait.st()  # explicit wait
                    Tx.cuda.cta_sync()

                    # tmem -> B_local (async)
                    Tx.copy_async(B_local[:, :], tmem[:, :])
                    Tx.ptx.tcgen05.wait.ld()  # explicit wait
                    Tx.cuda.cta_sync()

                    # B_local -> B
                    with Tx.thread():
                        for i in range(WIDTH // VEC_LEN):
                            g_offset = Tx.meta_var(g_layout.apply(tid_in_wg, i, 0)["m"])
                            Tx.copy(B_flat[g_offset: g_offset + VEC_LEN], B_reg[i * VEC_LEN: i * VEC_LEN + VEC_LEN])  # noqa: E501

                    if Tx.filter(warp_id, 0, 1):
                        with Tx.warp():
                            Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                            Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=max(32, next_power_of_2(width_32b)), cta_group=1)  # noqa: E501
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_async_test})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        A_np = tvm.testing.generate_random_array(dtype, (128, WIDTH))
        B_np = np.zeros((128, WIDTH), dtype=dtype)
        DEV = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, DEV)
        B = tvm.runtime.tensor(B_np, DEV)
        mod(A, B)
        np.testing.assert_allclose(B.numpy(), A_np)


if __name__ == "__main__":
    tvm.testing.main()
