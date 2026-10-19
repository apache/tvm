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

import math

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env
from tvm_tirx.tirx.layout import S, TileLayout


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

    @T.function
    def copy_async(
        A: T.Tensor(g_shape, dtype, layout=layoutA), B: T.Tensor(g_shape, dtype, layout=layoutB)
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(thread_cnt,)))
        cta_id = T.cuda.block_idx("x")
        tid = T.cuda.thread_idx("x")
        A_smem = T.alloc_tensor(ty_args=[T.Tensor(s_shape, dtype, scope="shared", layout=layoutS)])

        S_atom = (
            A_smem.sub[tuple(r_smem)]
            .view(-1)
            .view("uint32")
            .rearrange("(r w t v) -> w t r v", w=1, t=32, v=4)
        )
        G_atom = (
            A.sub[tuple(r_gmem)]
            .view(-1)
            .view("uint32")
            .rearrange("(r w t v) -> w t r v", w=1, t=32, v=4)
        )
        T.cuda.tile.apply(
            T.ptx["cp.async.cg.shared.global"], S_atom, G_atom, chunk_bytes=16, scope="cta"
        )
        T.ptx.cp.async_.commit_group()
        T.ptx.cp.async_.wait_group(0)
        T.cuda.cta_sync()
        copy_src_1 = T.meta_var(A_smem[tuple(r_smem)])
        copy_src_tensor_1 = T.meta_var(copy_src_1.source)
        copy_dst_1 = T.meta_var(B[tuple(r_gmem)])
        copy_dst_tensor_1 = T.meta_var(copy_dst_1.source)
        for copy_step_1 in T.serial(
            T.ceildiv(math.prod([int(r.extent) for r in copy_src_1.region]), thread_cnt)
        ):
            copy_index_1 = copy_step_1 * (thread_cnt) + (tid)
            if copy_index_1 < math.prod([int(r.extent) for r in copy_src_1.region]):
                copy_value_1 = copy_src_tensor_1[
                    tuple(
                        [
                            copy_src_1.region[k].begin
                            + copy_index_1
                            // math.prod([int(s.extent) for s in copy_src_1.region[k + 1 :]])
                            % copy_src_1.region[k].extent
                            for k in range(len(copy_src_1.region))
                        ]
                    )
                ]
                copy_dst_index_1 = T.meta_var(
                    tuple(
                        [
                            copy_dst_1.region[k].begin
                            + copy_index_1
                            // math.prod([int(s.extent) for s in copy_dst_1.region[k + 1 :]])
                            % copy_dst_1.region[k].extent
                            for k in range(len(copy_dst_1.region))
                        ]
                    )
                )
                copy_dst_tensor_1[copy_dst_index_1] = copy_value_1

    np_dtype = tvm_tirx.testing.np_dtype_from_str(dtype)
    target = tvm_tirx.target.Target("cuda")
    with target:
        mod = tvm_tirx.IRModule({"main": copy_async})
        mod = tvm_tirx.ir.transform.Sequential(
            [tvm_tirx.tirx.transform.TileDispatch(), tvm_tirx.tirx.transform.LowerTensorLayouts()]
        )(mod)
        mod = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")

        np.random.seed(0)
        A_np = np.random.rand(*g_shape).astype(np_dtype)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        B_ref = B_np.copy()
        B_ref[tuple(r_gmem)] = A_np[tuple(r_gmem)]

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            A = tvm_tirx.runtime.tensor(A_np, dev)
            B = tvm_tirx.runtime.tensor(B_np, dev)
            mod(A, B)
            np.testing.assert_allclose(B_ref, B.numpy())

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def test_copy_ldgsts_predicate_zero_fill_codegen():
    """ldgsts direct mode forwards predicate/zero-fill/prefetch without partition temps."""

    @T.function
    def copy_async(A: T.Tensor((32, 16), "uint8", layout=TileLayout(S[32, 16]))) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tid = T.cuda.thread_idx("x")
        A_smem = T.alloc_tensor(
            ty_args=[T.Tensor((32, 16), "uint8", scope="shared", layout=TileLayout(S[32, 16]))]
        )

        T.cuda.tile.apply(
            T.ptx["cp.async.cg.shared.global.L2::128B"],
            A_smem.sub[tid, :],
            A.sub[tid, :],
            chunk_bytes=16,
            src_size=T.if_then_else(tid < 16, T.uint32(16), T.uint32(0)),
        )

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm_tirx.tirx.compile(
            tvm_tirx.IRModule({"main": copy_async}), target=target, pipeline="tirx"
        )
    src = mod.mod.imports[0].inspect_source()
    assert "cp.async.cg.shared.global.L2::128B" in src
    # the src-size arity: four operands, the last the zero-fill boundary
    assert "cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;" in src
    assert "condval" in src
    assert "s_ptr_ptr" not in src
    assert "g_ptr_ptr" not in src


if __name__ == "__main__":
    tvm_tirx.testing.main()
