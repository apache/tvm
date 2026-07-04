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
"""Tests for the non-bulk CTA-level copy_async dispatch (vectorized load)."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx.layout import S, TileLayout


@pytest.mark.parametrize(
    "task",
    [
        ################ A[0:128, 0:32] -> A_smem[0:128, 0:32] -> B[0:128, 0:32] ################
        (
            (128, 32),  # g_shape
            (128, 32),  # s_shape
            (0, 0),  # g_st
            (128, 32),  # g_extent
            32,  # thread_cnt
            TileLayout(S[128, 32]),  # layoutA
            TileLayout(S[128, 32]),  # layoutB
            TileLayout(S[128, 32]),  # layoutS
        ),
        ################ A[32:64, 32:64] -> A_smem[0:32, 0:32] -> B[32:64, 32:64] ################
        (
            (64, 64),  # g_shape
            (32, 32),  # s_shape
            (32, 0),  # g_st
            (32, 32),  # g_extent
            32,  # thread_cnt
            TileLayout(S[64, 64]),  # layoutA
            TileLayout(S[64, 64]),  # layoutB
            TileLayout(S[32, 32]),  # layoutS
        ),
        ################ A[0:1, 0:32, 0:32] -> A_smem[0:32, 0:32] -> B[0:1, 0:32, 0:32] ################  # noqa: E501
        (
            (4, 32, 32),  # g_shape
            (32, 32),  # s_shape
            (0, 0, 0),  # g_st
            (1, 32, 32),  # g_extent
            32,  # thread_cnt
            TileLayout(S[4, 32, 32]),  # layoutA
            TileLayout(S[4, 32, 32]),  # layoutB
            TileLayout(S[32, 32]),  # layoutS
        ),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize(
    "dtype", ["int8", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32"]
)
def test_copy_g2s_s2g_cta_vec_load(task, dtype):
    g_shape, s_shape, g_st, g_extent, thread_cnt, layoutA, layoutB, layoutS = task

    r_smem = list(slice(None) for i in range(len(s_shape)))
    r_gmem = list(slice(g_st[i], g_st[i] + g_extent[i]) for i in range(len(g_shape)))

    # fmt: off
    @T.prim_func
    def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([thread_cnt])
        A_smem = T.alloc_buffer(s_shape, dtype, scope="shared", layout=layoutS)

        Tx.cta.copy_async(A_smem[tuple(r_smem)], A[tuple(r_gmem)], dispatch="ldgsts")
        T.ptx.cp_async.commit_group()
        T.ptx.cp_async.wait_group()
        T.cuda.cta_sync()
        Tx.cta.copy(B[tuple(r_gmem)], A_smem[tuple(r_smem)])
        # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.tirx.transform.LowerTIRx()(mod)
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(np_dtype)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        B_ref = B_np.copy()
        B_ref[tuple(r_gmem)] = A_np[tuple(r_gmem)]

        def run_test():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            dev.sync()
            np.testing.assert_allclose(B_ref, B.numpy())

        tvm.testing.run_with_gpu_lock(run_test)


if __name__ == "__main__":
    tvm.testing.main()
