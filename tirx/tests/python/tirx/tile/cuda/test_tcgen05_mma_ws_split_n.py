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

"""Split native N while preserving M64 banked-A and packed-C layouts."""

import numpy as np
import pytest

import tvm_tirx
from tvm_tirx.backend.cuda.axis import TCol, TLane, tid_in_wg
from tvm_tirx.backend.cuda.layout import mma_shared_layout
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env
from tvm_tirx.tirx.layout import S, TileLayout


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="Blackwell required")
def test_gemm_tcgen05_explicit_mma_tile_preserves_ws_datapath():
    """Two explicit N64 issues preserve the original N128 Layout-E output.

    Each N64 instruction splits its columns 32/32 between the two TMEM banks.
    The caller stages B in that order and permutes the logical C view without
    changing its physical storage. The independent A banks have different data.
    """
    ml_dtypes = pytest.importorskip("ml_dtypes")
    M, N, K = 64, 128, 16
    bl = mma_shared_layout("bfloat16", 1, (N, K))
    prepared_layout = mma_shared_layout("bfloat16", 1, (2, 64, K))
    al = TileLayout(S[(2, M, K) : (64 @ TLane, 1 @ TLane, 1 @ TCol)])
    dl = TileLayout(S[(M, 2, N // 2) : (1 @ TLane, 64 @ TLane, 1 @ TCol)])
    input_layout = TileLayout(S[(128, 8) : (1 @ tid_in_wg, 1)])

    @T.function
    def kernel(
        A: T.Tensor((2, M, K), "bfloat16"),
        B: T.Tensor((N, K), "bfloat16"),
        Output: T.Tensor((M, N), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        B_smem = T.alloc_shared((N, K), "bfloat16", layout=bl, align=1024)
        B_prepared = T.alloc_shared((2, 64, K), "bfloat16", layout=prepared_layout, align=1024)
        address = T.alloc_shared((1,), "uint32", layout="default")
        barrier = T.alloc_shared((1,), "uint64", layout="default")
        if tid == 0:
            T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), T.uint32(1))
        if warp == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                address.ptr_to([0]), T.uint32(512)
            )
        for j in T.serial(K):
            B_smem[tid, j] = B[tid, j]
        T.cuda.cta_sync()
        for j in T.serial(K):
            # split, row-in-N64 -> original bank, column-in-bank.
            B_prepared[tid // 64, tid % 64, j] = B_smem[
                (tid % 64) // 32 * 64 + (tid // 64) * 32 + tid % 32, j
            ]
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        A_tmem = T.cuda.decl_tmem(
            address[0] + T.uint32(256),
            ty_args=[T.Tensor((2, M, K), "bfloat16", scope="tmem", layout=al)],
        )
        D_tmem = T.cuda.decl_tmem(
            address[0] + T.uint32(400),
            ty_args=[T.Tensor((M, N), "float32", scope="tmem", layout=dl)],
        )
        ar = T.alloc_local((8,), "uint32", layout="default")
        ag = A.view("uint32").view(128, 8)
        for j in T.serial(8):
            ar[j] = ag[tid, j]
        av = ar.view(128, 8, layout=input_layout).view(4, 32, 8)
        at = A_tmem.view("uint32").view(4, 32, 8)
        T.cuda.tile.apply(T.ptx["tcgen05.st.sync.aligned.32x32b.x8.b32"], at, av, scope="warpgroup")
        T.ptx.tcgen05.wait__st.sync.aligned()
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence__after_thread_sync()
        d = D_tmem.view(M, 2, 2, 32).permute(2, 0, 1, 3).view(1, 2, M, 64)
        if tid == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.mma.ws.cta_group::1.kind::f16"],
                d,
                A_tmem.view(1, 1, 2, M, K),
                B_prepared.view(2, 1, 64, K),
                instruction_shape=(M, 64, K),
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                barrier.ptr_to([0])
            )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        dr = T.alloc_local((64,), "float32", layout="default")
        dv = dr.view(128, 64, layout=TileLayout(S[(128, 64) : (1 @ tid_in_wg, 1)])).view(4, 32, 64)
        physical = D_tmem.view(M, 2, 64).permute(1, 0, 2).view(4, 32, 64)
        T.cuda.tile.apply(
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x64.b32"], dv, physical, scope="warpgroup"
        )
        T.ptx.tcgen05.wait__ld.sync.aligned()
        for j in T.serial(64):
            Output[tid % 64, tid // 64 * 64 + j] = dr[j]
        T.cuda.cta_sync()
        if warp == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], T.uint32(512))

    f = tvm_tirx.tirx.compile(kernel, target={"kind": "cuda", "arch": "sm_100a"}, pipeline="tirx")
    calls = [
        line
        for line in f.mod.imports[0].inspect_source().splitlines()
        if line.strip().startswith("tvm_builtin_ptx_tcgen05_mma_ws_ts_")
    ]
    assert len(calls) == 2
    rng = np.random.default_rng(2026)
    a = rng.uniform(-1, 1, (2, M, K)).astype(ml_dtypes.bfloat16)
    b = rng.uniform(-1, 1, (N, K)).astype(ml_dtypes.bfloat16)
    result = tvm_tirx.runtime.tensor(np.zeros((M, N), "float32"), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(a, tvm_tirx.cuda()), tvm_tirx.runtime.tensor(b, tvm_tirx.cuda()), result)
    ref = np.concatenate(
        [
            a[bank].astype("float32") @ b[bank * 64 : (bank + 1) * 64].astype("float32").T
            for bank in range(2)
        ],
        axis=1,
    )
    np.testing.assert_allclose(result.numpy(), ref, atol=1e-2, rtol=1e-2)
