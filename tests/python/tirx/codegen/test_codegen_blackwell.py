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
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env


def _get_source(func: tvm.tirx.PrimFunc) -> str:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return src, mod


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_tmem_alloc_dealloc_relinquish():
    N_COLS = 512
    cta_group = 1

    # fmt: off
    @T.prim_func
    def test_tmem(A: T.Buffer((16, 16), "float16")):
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([4])
        lane_id = T.lane_id([32])
        tid = T.thread_id([128])
        # tmem_addr = T.alloc_buffer((1,), "uint32", scope="shared", align=8)
        tmem_addr = T.shared_scalar("uint32")

        # alloc TMEM
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=N_COLS, cta_group=cta_group)
        T.cuda.cta_sync()

        # dealloc TMEM
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
            T.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=cta_group)
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        src, _ = _get_source(test_tmem)
        assert f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32" in src
        assert f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32" in src
        assert f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned" in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_mbarrier_try_wait_once_codegen():
    # fmt: off
    @T.prim_func
    def test_try_wait_once(A: T.Buffer((16, 16), "float16")):
        T.device_entry()
        T.cta_id([1])
        T.thread_id([128])
        bar = T.shared_scalar("uint64")
        T.evaluate(T.ptx.mbarrier.try_wait_once(T.address_of(bar), 0, 0))
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        src, _ = _get_source(test_try_wait_once)
        assert "mbarrier.try_wait.parity.shared::cta.b64" in src
        assert "selp.u32" in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_fence_before_after_thread_sync():
    # fmt: off
    @T.prim_func
    def test_fence(A: T.Buffer((16, 16), "float16")):
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([4])
        lane_id = T.lane_id([32])
        tid = T.thread_id([128])
        T.ptx.tcgen05.fence.before_thread_sync()
        T.ptx.bar.sync(0, 32)
        T.ptx.tcgen05.fence.after_thread_sync()
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        src, _ = _get_source(test_fence)
        assert "tcgen05.fence::after_thread_sync" in src
        assert "tcgen05.fence::before_thread_sync" in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_tcgen05_ld_st_roundtrip():
    HEIGHT = 128
    WIDTH = 256
    N_COLS = 512
    REPEAT_NUM = 1
    cta_group = 1

    # fmt: off
    @T.prim_func
    def test_ld_st(A: T.Buffer((HEIGHT, WIDTH), "float32"), B: T.Buffer((HEIGHT, WIDTH), "float32")):  # noqa: E501
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([4])
        lane_id = T.lane_id([32])
        tx = T.thread_id([128])
        reg = T.alloc_buffer((WIDTH,), "float32", scope="local")
        # tmem_addr = T.alloc_buffer((1,), "uint32", scope="shared", align=8)
        tmem_addr = T.shared_scalar("uint32")

        # alloc TMEM
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=N_COLS, cta_group=cta_group)
        T.cuda.cta_sync()
        # GMEM -> RF
        for i in range(WIDTH):
            reg[i] = A[tx, i]
        # RF -> TMEM
        for i in range(WIDTH):
            T.ptx.tcgen05.st(tmem_addr, reg[i], shape="32x32b", num=REPEAT_NUM, row=warp_id * 32, col=i)  # noqa: E501
        T.ptx.tcgen05.wait.st()
        T.cuda.cta_sync()
        # reset RF
        for i in range(WIDTH):
            reg[i] = 0.0
        T.cuda.cta_sync()
        # TMEM -> RF
        T.ptx.tcgen05.fence.after_thread_sync()
        for i in range(WIDTH):
            T.ptx.tcgen05.ld(tmem_addr, reg[i], shape="32x32b", num=REPEAT_NUM, row=warp_id * 32, col=i)  # noqa: E501
        T.ptx.tcgen05.wait.ld()
        # RF -> GMEM
        for i in range(WIDTH):
            B[tx, i] = reg[i]

        # dealloc TMEM
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
            T.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=cta_group)
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        src, mod = _get_source(test_ld_st)
        assert "tcgen05.ld.sync.aligned.32x32b.x1.b32" in src
        assert "tcgen05.st.sync.aligned.32x32b.x1.b32" in src

    def run():
        dev = tvm.cuda(0)
        A_np = np.random.randn(HEIGHT, WIDTH).astype("float32")
        B_np = np.zeros((HEIGHT, WIDTH), dtype="float32")
        A = tvm.runtime.tensor(A_np, device=dev)
        B = tvm.runtime.tensor(B_np, device=dev)
        mod(A, B)
        np.testing.assert_allclose(A.numpy(), B.numpy())

    tvm.testing.run_with_gpu_lock(run)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_tcgen05_cp_ld_roundtrip():
    dtype = "float32"
    dtype_bits = tvm.DataType(dtype).bits
    HEIGHT = 128
    WIDTH = 64
    N_COLS = 512
    REPEAT_NUM = 1
    SWIZZLE = 0
    A_layout = T.TileLayout(T.S[(HEIGHT, WIDTH // 4, 4) : (4, HEIGHT * 4, 1)])
    ldo, sdo = 128, 8
    cta_group = 1

    # fmt: off
    @T.prim_func
    def test_cp_ld(A: T.Buffer((HEIGHT, WIDTH), dtype, layout=T.TileLayout(T.S[(HEIGHT, WIDTH // 4, 4) : (4, HEIGHT * 4, 1)])),  # noqa: E501
                   B: T.Buffer((HEIGHT, WIDTH), dtype, layout=T.TileLayout(T.S[(HEIGHT, WIDTH // 4, 4) : (4, HEIGHT * 4, 1)]))):  # noqa: E501
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([4])
        lane_id = T.lane_id([32])
        tx = T.thread_id([128])
        A_smem = T.alloc_buffer((HEIGHT, WIDTH), dtype, scope="shared", layout=A_layout)
        reg = T.alloc_buffer((WIDTH,), dtype, scope="local")
        # tmem_addr = T.alloc_buffer((1,), "uint32", scope="shared", align=8)
        tmem_addr = T.shared_scalar("uint32")
        descA = T.alloc_buffer((1,), "uint64", scope="local")
        bar = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
        phase = T.alloc_buffer((1,), "int32", scope="local")

        # alloc TMEM
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=N_COLS, cta_group=cta_group)
        T.cuda.cta_sync()
        Tx.cta.copy(A_smem[:, :], A[:, :])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        # reset RF
        for i in range(WIDTH):
            reg[i] = 0.0
        # SMEM -> TMEM (cp)
        phase[0] = 0
        if tx == 0:
            T.ptx.mbarrier.init(bar.data, 1)
            for k in range(dtype_bits * WIDTH // 256):
                T.ptx.tcgen05.encode_matrix_descriptor(descA.data, A_smem.access_ptr("r", offset=A_smem.elem_offset_of([0, k * 8])), ldo=ldo, sdo=sdo, swizzle=SWIZZLE)  # noqa: E501
                T.ptx.tcgen05.cp(tmem_addr, descA[0], shape="128x256b", cta_group=cta_group, col=k * 256 // 32)  # noqa: E501
            T.ptx.tcgen05.commit(bar.data, cta_group)
        T.ptx.mbarrier.try_wait(bar.data, phase[0])
        phase[0] = phase[0] ^ 1
        T.cuda.cta_sync()
        # TMEM -> RF (ld)
        T.ptx.tcgen05.fence.after_thread_sync()
        for i in range(WIDTH):
            T.ptx.tcgen05.ld(tmem_addr, reg[i], shape="32x32b", num=REPEAT_NUM, row=warp_id * 32, col=i)  # noqa: E501
        T.ptx.tcgen05.wait.ld()
        # RF -> GMEM
        for i in range(WIDTH):
            B[tx, i] = reg[i]

        # dealloc TMEM
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
            T.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=cta_group)
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        src, mod = _get_source(test_cp_ld)
        assert "tcgen05.cp.cta_group::1.128x256b" in src
        assert "tcgen05.ld.sync.aligned.32x32b.x1.b32" in src

    def run():
        dev = tvm.cuda(0)
        A_np = np.random.randn(HEIGHT, WIDTH).astype(dtype)
        B_np = np.zeros((HEIGHT, WIDTH), dtype=dtype)
        A = tvm.runtime.tensor(A_np, device=dev)
        B = tvm.runtime.tensor(B_np, device=dev)
        mod(A, B)
        np.testing.assert_allclose(A.numpy(), B.numpy())

    tvm.testing.run_with_gpu_lock(run)


@pytest.mark.parametrize("swizzle", [0, 1, 2, 3])
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_tcgen05_mma_ss_no_tma(swizzle):
    d_type, a_type, b_type = "float32", "float16", "float16"
    M, N, K = 128, 128, 64
    MMA_K = 16
    N_COLS = 512
    REPEAT_NUM = 1
    SWIZZLE = swizzle
    cta_group = 1

    if SWIZZLE == 0:
        A_layout = T.TileLayout(T.S[(M, K // 8, 8) : (8, M * 8, 1)])
        B_layout = T.TileLayout(T.S[(N, K // 8, 8) : (8, N * 8, 1)])
        ldo, sdo = 128, 8
    elif SWIZZLE == 1:
        A_layout = T.ComposeLayout(
            T.SwizzleLayout(3, 1, 3, swizzle_inner=True),
            T.TileLayout(T.S[(M, K // 16, 16) : (16, M * 16, 1)]),
        )
        B_layout = T.ComposeLayout(
            T.SwizzleLayout(3, 1, 3, swizzle_inner=True),
            T.TileLayout(T.S[(N, K // 16, 16) : (16, N * 16, 1)]),
        )
        ldo, sdo = 256, 16
    elif SWIZZLE == 2:
        A_layout = T.ComposeLayout(
            T.SwizzleLayout(3, 2, 3, swizzle_inner=True),
            T.TileLayout(T.S[(M, K // 32, 32) : (32, M * 32, 1)]),
        )
        B_layout = T.ComposeLayout(
            T.SwizzleLayout(3, 2, 3, swizzle_inner=True),
            T.TileLayout(T.S[(N, K // 32, 32) : (32, N * 32, 1)]),
        )
        ldo, sdo = 512, 32
    elif SWIZZLE == 3:
        A_layout = T.ComposeLayout(
            T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
            T.TileLayout(T.S[(M, 1, 64) : (64, M * 64, 1)]),
        )
        B_layout = T.ComposeLayout(
            T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
            T.TileLayout(T.S[(N, 1, 64) : (64, N * 64, 1)]),
        )
        ldo, sdo = 1, 64
    else:
        raise ValueError(f"Invalid swizzle: {SWIZZLE}")

    dyn_smem_bytes = 1024 + (M * K + N * K) * 2

    # fmt: off
    @T.prim_func
    def test_mma_ss_no_tma(A: T.Buffer((M, K), a_type, layout=T.TileLayout(T.S[M, K])),
                           B: T.Buffer((N, K), b_type, layout=T.TileLayout(T.S[N, K])),
                           C: T.Buffer((M, N), d_type)):
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([4])
        lane_id = T.lane_id([32])
        tx = T.thread_id([128])
        dyn = T.alloc_buffer((dyn_smem_bytes,), "uint8", scope="shared")
        tmem_addr = T.decl_scalar("uint32", dyn.data, scope="shared", elem_offset=0)
        A_smem = T.decl_buffer((M, K), a_type, dyn.data, elem_offset=256, layout=A_layout)
        B_smem = T.decl_buffer((N, K), b_type, dyn.data, elem_offset=256 + M*K, layout=B_layout)
        bar = T.decl_buffer((1,), "uint64", dyn.data, scope="shared", elem_offset=8)

        reg = T.alloc_buffer((N,), d_type, scope="local")
        descA = T.alloc_buffer((1,), "uint64", scope="local")
        descB = T.alloc_buffer((1,), "uint64", scope="local")
        descI = T.alloc_buffer((1,), "uint32", scope="local")
        phase = T.alloc_buffer((1,), "int32", scope="local")

        # alloc TMEM
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=N_COLS, cta_group=cta_group)
        T.cuda.cta_sync()
        for i in range(N):
            reg[i] = 0.0
        Tx.cta.copy(A_smem[:, :], A[:, :])
        Tx.cta.copy(B_smem[:, :], B[:, :])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        # MMA
        phase[0] = 0
        if tx == 0:
            T.ptx.mbarrier.init(bar.data, 1)
            T.ptx.tcgen05.encode_instr_descriptor(descI.data, d_dtype=d_type, a_dtype=a_type, b_dtype=b_type, M=M, N=N, K=MMA_K, trans_a=False, trans_b=False, n_cta_groups=cta_group)  # noqa: E501
            for k in range(K // MMA_K):
                T.ptx.tcgen05.encode_matrix_descriptor(descA.data, A_smem.access_ptr("r", offset=A_smem.elem_offset_of([0, k * MMA_K])), ldo=ldo, sdo=sdo, swizzle=SWIZZLE)  # noqa: E501
                T.ptx.tcgen05.encode_matrix_descriptor(descB.data, B_smem.access_ptr("r", offset=B_smem.elem_offset_of([0, k * MMA_K])), ldo=ldo, sdo=sdo, swizzle=SWIZZLE)  # noqa: E501
                if k == 0:
                    T.ptx.tcgen05.mma(tmem_addr, descA[0], descB[0], descI[0], d_dtype=d_type, a_dtype=a_type, b_dtype=b_type, use_a_tmem=False, cta_group=cta_group, enable_input_d=0)  # noqa: E501
                else:
                    T.ptx.tcgen05.mma(tmem_addr, descA[0], descB[0], descI[0], d_dtype=d_type, a_dtype=a_type, b_dtype=b_type, use_a_tmem=False, cta_group=cta_group, enable_input_d=1)  # noqa: E501
            T.ptx.tcgen05.commit(bar.data, cta_group)
        T.ptx.mbarrier.try_wait(bar.data, phase[0])
        phase[0] = phase[0] ^ 1
        T.cuda.cta_sync()

        # TMEM -> RF
        T.ptx.tcgen05.fence.after_thread_sync()
        for i in range(N):
            T.ptx.tcgen05.ld(tmem_addr, reg[i], shape="32x32b", num=REPEAT_NUM, row=warp_id * 32, col=i)  # noqa: E501
        T.ptx.tcgen05.wait.ld()
        # RF -> GMEM
        for i in range(N):
            C[tx, i] = reg[i]

        # dealloc TMEM
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
            T.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=cta_group)
    # fmt: on

    import torch

    torch.manual_seed(42)
    target = tvm.target.Target("cuda")
    with target:
        src, mod = _get_source(test_mma_ss_no_tma)
        print(src)
        assert "tcgen05.mma.cta_group::1.kind::f16" in src
        assert "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64" in src
        assert "tcgen05.ld.sync.aligned.32x32b.x1.b32" in src
        assert "tcgen05.wait::ld.sync.aligned" in src

    def run():
        dev = tvm.cuda(0)
        A_torch = torch.rand((M, K), dtype=torch.float16)
        B_torch = torch.rand((N, K), dtype=torch.float16)
        C_torch = torch.zeros((M, N), dtype=torch.float32)
        A = tvm.runtime.tensor(A_torch, device=dev)
        B = tvm.runtime.tensor(B_torch, device=dev)
        C = tvm.runtime.tensor(C_torch, device=dev)
        mod(A, B, C)
        ref = torch.matmul(A_torch, B_torch.T)
        np.testing.assert_allclose(C.numpy(), ref.numpy(), rtol=1e-3, atol=1e-2)

    tvm.testing.run_with_gpu_lock(run)


if __name__ == "__main__":
    tvm.testing.main()
