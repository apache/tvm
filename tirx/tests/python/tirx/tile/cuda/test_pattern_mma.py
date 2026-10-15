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

"""Explicit warp MMA coordinates and repeated M/N/K execution."""

import numpy as np
import pytest

import tvm_tirx
from tvm_tirx.backend.cuda.axis import laneid
from tvm_tirx.backend.cuda.layout import tmem_mma_operand_layout
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env
from tvm_tirx.tirx.layout import ComposeLayout, S, TileLayout


@pytest.mark.skipif(not env.has_cuda_compute(8), reason="Ampere required")
@pytest.mark.parametrize("k", [8, 16])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("initial", [0, 3])
def test_mma(k, dtype, initial):
    M, N, K = 2, 2, 3
    la = TileLayout(S[(M, K, 2, 8, k // 8, 4) : (K * k // 4, k // 4, 1, 4 @ laneid, 2, 1 @ laneid)])
    lb = TileLayout(S[(N, K, 8, k // 8, 4) : (K * k // 8, k // 8, 4 @ laneid, 1, 1 @ laneid)])
    ld = TileLayout(S[(M, N, 2, 8, 4, 2) : (N * 4, 4, 2, 4 @ laneid, 1 @ laneid, 1)])
    elem = "f16" if dtype == "float16" else "bf16"
    instruction = f"mma.sync.aligned.m16n8k{k}.row.col.f32.{elem}.{elem}.f32"

    @T.function
    def kernel(
        A: T.Tensor((M, K, 16, k // 2), "uint32"),
        B: T.Tensor((N, K, 8, k // 2), "uint32"),
        Output: T.Tensor((M * 16, N * 8), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        ar = T.alloc_local((M * K * k // 4,), "uint32", layout="default")
        br = T.alloc_local((N * K * k // 8,), "uint32", layout="default")
        dr = T.alloc_local((M * N * 4,), "float32", layout="default")
        av = ar.view(M, K, 16, k // 2, layout=la)
        bv = br.view(N, K, 8, k // 2, layout=lb)
        dv = dr.view(M, N, 16, 8, layout=ld)
        for m, kk, r in T.grid(M, K, k // 4):
            ar[(m * K + kk) * (k // 4) + r] = A[
                m, kk, lane // 4 + 8 * (r % 2), lane % 4 + 4 * (r // 2)
            ]
        for n, kk, r in T.grid(N, K, k // 8):
            br[(n * K + kk) * (k // 8) + r] = B[n, kk, lane // 4, lane % 4 + 4 * r]
        T.cuda.tile.apply(T.ptx[instruction], dv, av, bv, T.float32(initial), scope="warp")
        for m, n, r in T.grid(M, N, 4):
            Output[m * 16 + lane // 4 + 8 * (r // 2), n * 8 + 2 * (lane % 4) + r % 2] = dr[
                (m * N + n) * 4 + r
            ]

    f = tvm_tirx.tirx.compile(kernel, target="cuda", pipeline="tirx")
    rng = np.random.default_rng(7)
    a = rng.uniform(-1, 1, (M * 16, K * k)).astype(dtype)
    b = rng.uniform(-1, 1, (N * 8, K * k)).astype(dtype)
    aa = np.ascontiguousarray(a.reshape(M, 16, K, k).transpose(0, 2, 1, 3)).view(np.uint32)
    bb = np.ascontiguousarray(b.reshape(N, 8, K, k).transpose(0, 2, 1, 3)).view(np.uint32)
    output = tvm_tirx.runtime.tensor(np.zeros((M * 16, N * 8), "float32"), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(aa, tvm_tirx.cuda()), tvm_tirx.runtime.tensor(bb, tvm_tirx.cuda()), output)
    np.testing.assert_allclose(
        output.numpy(), a.astype("float32") @ b.astype("float32").T + initial, rtol=2e-4, atol=2e-4
    )


@pytest.mark.skipif(not env.has_cuda_compute(10), reason="Blackwell required")
@pytest.mark.parametrize("m,ws", [(128, False), (128, True), (64, False), (64, True)])
@pytest.mark.parametrize(
    "dtype,k,kind,acc_dtype",
    [
        ("float16", 16, "f16", "float32"),
        ("float16", 16, "f16", "float16"),
        ("bfloat16", 16, "f16", "float32"),
        ("float8_e4m3fn", 32, "f8f6f4", "float32"),
        ("float32", 8, "tf32", "float32"),
        ("int8", 32, "i8", "int32"),
    ],
)
@pytest.mark.parametrize("major", ["k", "mn"])
def test_tcgen05_mma(m, ws, dtype, k, kind, acc_dtype, major, swizzle=3):
    if major == "mn" and kind == "tf32" and swizzle != 4:
        pytest.skip("TF32 MN-major uses the separately tested 32B-atomicity descriptor")
    n, nk = 64, 3
    packed = 32 // tvm_tirx.DataType(acc_dtype).bits
    bits = tvm_tirx.DataType(dtype).bits
    row_bytes = 128 if swizzle == 4 else 16 << swizzle
    swizzle_len = 2 if swizzle == 4 else swizzle
    atom_len = 2 if swizzle == 4 else 3
    row_elems = row_bytes * 8 // bits
    per_element = ((256 if swizzle == 4 else 128) // bits).bit_length() - 1
    if major == "mn":
        al = ComposeLayout(
            per_element,
            swizzle_len,
            atom_len,
            TileLayout(
                S[
                    (1, nk, m // min(m, row_elems), min(m, row_elems), k) : (
                        0,
                        k * row_elems,
                        nk * k * row_elems,
                        1,
                        row_elems,
                    )
                ]
            ),
        )
        bl = ComposeLayout(
            per_element,
            swizzle_len,
            atom_len,
            TileLayout(
                S[
                    (1, nk, n // min(n, row_elems), min(n, row_elems), k) : (
                        0,
                        k * row_elems,
                        nk * k * row_elems,
                        1,
                        row_elems,
                    )
                ]
            ),
        )
    else:
        inner = min(k, row_elems)
        al = ComposeLayout(
            per_element,
            swizzle_len,
            atom_len,
            TileLayout(
                S[
                    (1, nk, m, k // inner, inner) : (
                        0,
                        m * max(k, row_elems),
                        row_elems,
                        m * row_elems,
                        1,
                    )
                ]
            ),
        )
        bl = ComposeLayout(
            per_element,
            swizzle_len,
            atom_len,
            TileLayout(
                S[
                    (1, nk, n, k // inner, inner) : (
                        0,
                        n * max(k, row_elems),
                        row_elems,
                        n * row_elems,
                        1,
                    )
                ]
            ),
        )
    dl = tmem_mma_operand_layout("D", (m, n), acc_dtype, M=m, cta_group=1, ws=ws)
    data_format = "tf32" if kind == "tf32" else dtype
    instruction = f"tcgen05.mma{'.ws' if ws else ''}.cta_group::1.kind::{kind}"

    @T.function
    def kernel(
        A: T.Tensor((m, nk * k), dtype),
        B: T.Tensor((n, nk * k), dtype),
        Output: T.Tensor((128, n), "uint32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        a = T.alloc_tensor(
            ty_args=[T.Tensor((1, nk, m, k), dtype, scope="shared", layout=al, align=1024)]
        )
        b = T.alloc_tensor(
            ty_args=[T.Tensor((1, nk, n, k), dtype, scope="shared", layout=bl, align=1024)]
        )
        address = T.alloc_shared((1,), "uint32", layout="default")
        barrier = T.alloc_shared((1,), "uint64", layout="default")
        if warp == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(address), T.uint32(n)
            )
        if tid == 0:
            T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), T.uint32(1))
            for ki, i, j in T.grid(nk, m, k):
                a[0, ki, i, j] = A[i, ki * k + j]
            for ki, i, j in T.grid(nk, n, k):
                b[0, ki, i, j] = B[i, ki * k + j]
        T.cuda.cta_sync()
        T.ptx.fence.proxy.async_.shared__cta()
        d = T.cuda.decl_tmem(
            address[0], ty_args=[T.Tensor((1, 1, m, n), acc_dtype, scope="tmem", layout=dl)]
        )
        if tid == 0:
            T.cuda.tile.apply(
                T.ptx[instruction],
                d,
                a,
                b,
                instruction_shape=(m, n, k),
                accum=False,
                a_major=major,
                b_major=major,
                a_format=data_format,
                b_format=data_format,
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                barrier.ptr_to([0])
            )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        word = T.alloc_local((1,), "uint32", layout="default")
        for j in range(n):
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](word[0], address[0] + T.uint32(j))
            T.ptx.tcgen05.wait__ld.sync.aligned()
            Output[tid, j] = word[0]
        T.cuda.cta_sync()
        if warp == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], T.uint32(n))

    f = tvm_tirx.tirx.compile(kernel, target="cuda", pipeline="tirx")
    rng = np.random.default_rng(4)
    a = (rng.integers(-4, 5, (m, nk * k)) / 4).astype(dtype)
    b = (rng.integers(-4, 5, (n, nk * k)) / 4).astype(dtype)
    if kind == "i8":
        a = rng.integers(-4, 5, (m, nk * k)).astype(dtype)
        b = rng.integers(-4, 5, (n, nk * k)).astype(dtype)
    out = tvm_tirx.runtime.tensor(np.zeros((128, n), "uint32"), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(a, tvm_tirx.cuda()), tvm_tirx.runtime.tensor(b, tvm_tirx.cuda()), out)
    actual = out.numpy().view(acc_dtype)[:, ::packed]
    ref = a.astype("float32") @ b.astype("float32").T
    for i in range(m):
        for j in range(n):
            if m == 128:
                row, col = i, j
            elif ws:
                row, col = i + 64 * (j // (n // 2)), j % (n // 2)
            else:
                row, col = 32 * (i // 16) + i % 16, j
            np.testing.assert_allclose(actual[row, col], ref[i, j], rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not env.has_cuda_compute(10), reason="Blackwell required")
def test_tcgen05_block_scale_repeated_k():
    from tvm_tirx.backend.cuda.axis import TCol, TLane
    from tvm_tirx.tirx.layout import R

    m, n, k, nk = 128, 64, 32, 4
    dtype = "float8_e4m3fn"
    al = ComposeLayout(4, 3, 3, TileLayout(S[(1, nk, m // 8, 8, k) : (0, k, 1024, 128, 1)]))
    bl = ComposeLayout(4, 3, 3, TileLayout(S[(1, nk, n // 8, 8, k) : (0, k, 1024, 128, 1)]))
    dl = tmem_mma_operand_layout("D", (m, n), "float32", M=m, cta_group=1)
    sfal = TileLayout(
        S[(1, nk, 4, 32, 1) : (0, 1 @ TCol, 4 @ TCol, 1 @ TLane, 0)] + R[4 : 32 @ TLane]
    )
    sfbl = TileLayout(
        S[(1, nk, 2, 32, 1) : (0, 1 @ TCol, 4 @ TCol, 1 @ TLane, 0)] + R[4 : 32 @ TLane]
    )

    @T.function
    def kernel(
        A: T.Tensor((m, nk * k), dtype),
        B: T.Tensor((n, nk * k), dtype),
        Output: T.Tensor((m, n), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        a = T.alloc_tensor(
            ty_args=[T.Tensor((1, nk, m, k), dtype, scope="shared", layout=al, align=1024)]
        )
        b = T.alloc_tensor(
            ty_args=[T.Tensor((1, nk, n, k), dtype, scope="shared", layout=bl, align=1024)]
        )
        address = T.alloc_shared((1,), "uint32", layout="default")
        barrier = T.alloc_shared((1,), "uint64", layout="default")
        if warp == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(address), T.uint32(128)
            )
        if tid == 0:
            T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), T.uint32(1))
            for ki, i, j in T.grid(nk, m, k):
                a[0, ki, i, j] = A[i, ki * k + j]
            for ki, i, j in T.grid(nk, n, k):
                b[0, ki, i, j] = B[i, ki * k + j]
        T.cuda.cta_sync()
        T.ptx.fence.proxy.async_.shared__cta()
        word = T.alloc_local((1,), "uint32", layout="default")
        word[0] = T.uint32(2189525119)
        for j in range(4):
            T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](address[0] + T.uint32(64 + j), word[0])
        word[0] = T.uint32(2139062143)
        for j in range(2):
            T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](address[0] + T.uint32(68 + j), word[0])
        T.ptx.tcgen05.wait__st.sync.aligned()
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence__after_thread_sync()
        d = T.cuda.decl_tmem(
            address[0], ty_args=[T.Tensor((1, 1, m, n), "float32", scope="tmem", layout=dl)]
        )
        sfa = T.cuda.decl_tmem(
            address[0] + T.uint32(64),
            ty_args=[T.Tensor((1, nk, m, 1), "float8_e8m0fnu", scope="tmem", layout=sfal)],
        )
        sfb = T.cuda.decl_tmem(
            address[0] + T.uint32(68),
            ty_args=[T.Tensor((1, nk, n, 1), "float8_e8m0fnu", scope="tmem", layout=sfbl)],
        )
        if tid == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"],
                d,
                a,
                b,
                sfa,
                sfb,
                instruction_shape=(m, n, k),
                accum=False,
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                barrier.ptr_to([0])
            )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        result = T.alloc_local((1,), "float32", layout="default")
        for j in range(n):
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](result[0], address[0] + T.uint32(j))
            T.ptx.tcgen05.wait__ld.sync.aligned()
            Output[tid, j] = result[0]
        T.cuda.cta_sync()
        if warp == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], T.uint32(128))

    f = tvm_tirx.tirx.compile(kernel, target="cuda", pipeline="tirx")
    rng = np.random.default_rng(5)
    a = rng.uniform(-1, 1, (m, nk * k)).astype(dtype)
    b = rng.uniform(-1, 1, (n, nk * k)).astype(dtype)
    out = tvm_tirx.runtime.tensor(np.zeros((m, n), "float32"), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(a, tvm_tirx.cuda()), tvm_tirx.runtime.tensor(b, tvm_tirx.cuda()), out)
    af = a.astype("float32")
    for ki in range(nk):
        af[:, ki * k : (ki + 1) * k] *= 2**ki
    np.testing.assert_allclose(out.numpy(), af @ b.astype("float32").T, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not env.has_cuda_compute(10), reason="Blackwell required")
@pytest.mark.parametrize("m", [128, 256])
def test_tcgen05_mma_cta_pair(m):
    n, k = 128, 16
    rows = m // 2
    al = ComposeLayout(3, 3, 3, TileLayout(S[(rows, k) : (64, 1)]))
    bl = ComposeLayout(3, 3, 3, TileLayout(S[(64, k) : (64, 1)]))
    dl = tmem_mma_operand_layout("D", (rows, n), "float32", M=m, cta_group=2)

    @T.function
    def kernel(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((n, k), "float16"),
        Output: T.Tensor((m, n), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=128, cluster=2))
        cta = T.cuda.cluster_cta_id("x")
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        a = T.alloc_tensor(
            ty_args=[T.Tensor((1, 1, rows, k), "float16", scope="shared", layout=al, align=1024)]
        )
        b = T.alloc_tensor(
            ty_args=[T.Tensor((1, 1, 64, k), "float16", scope="shared", layout=bl, align=1024)]
        )
        address = T.alloc_shared((1,), "uint32", layout="default")
        barrier = T.alloc_shared((1,), "uint64", layout="default")
        if warp == 0:
            T.ptx.tcgen05.alloc.cta_group__2.sync.aligned.shared__cta.b32(
                T.address_of(address), T.uint32(128)
            )
        if tid == 0:
            T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), T.uint32(1))
            for i, j in T.grid(rows, k):
                a[0, 0, i, j] = A[cta * rows + i, j]
            for i, j in T.grid(64, k):
                b[0, 0, i, j] = B[cta * 64 + i, j]
        T.cuda.cta_sync()
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cluster_sync()
        T.ptx.tcgen05.fence__after_thread_sync()
        d = T.cuda.decl_tmem(
            address[0], ty_args=[T.Tensor((1, 1, rows, n), "float32", scope="tmem", layout=dl)]
        )
        if cta == 0:
            if tid == 0:
                T.cuda.tile.apply(
                    T.ptx["tcgen05.mma.cta_group::2.kind::f16"],
                    d,
                    a,
                    b,
                    instruction_shape=(m, n, k),
                )
                T.ptx[
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
                ](barrier.ptr_to([0]), T.uint16(3))
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        reg = T.alloc_local((1,), "float32", layout="default")
        for j in range(n * rows // 128):
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](reg[0], address[0] + T.uint32(j))
            T.ptx.tcgen05.wait__ld.sync.aligned()
            Output[cta * rows + tid % rows, tid // rows * (n // 2) + j] = reg[0]
        T.cuda.cluster_sync()
        if warp == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__2.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__2.sync.aligned.b32(address[0], T.uint32(128))

    f = tvm_tirx.tirx.compile(kernel, target="cuda", pipeline="tirx")
    rng = np.random.default_rng(12)
    a = rng.uniform(-1, 1, (m, k)).astype("float16")
    b = rng.uniform(-1, 1, (n, k)).astype("float16")
    out = tvm_tirx.runtime.tensor(np.zeros((m, n), "float32"), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(a, tvm_tirx.cuda()), tvm_tirx.runtime.tensor(b, tvm_tirx.cuda()), out)
    np.testing.assert_allclose(
        out.numpy(), a.astype("float32") @ b.astype("float32").T, rtol=2e-4, atol=2e-4
    )


@pytest.mark.skipif(not env.has_cuda_compute(10), reason="Blackwell required")
def test_tcgen05_tmem_a_repeated_mnk():
    """TMEM A and D occupy separate column ranges, with independent M/N/K repeats."""
    from tvm_tirx.backend.cuda.axis import TCol, TLane

    mr, nr, kr, m, n, k = 2, 2, 2, 128, 64, 16
    al = TileLayout(S[(mr, kr, m, k) : (kr * k @ TCol, k @ TCol, 1 @ TLane, 1 @ TCol)])
    dl = TileLayout(S[(mr, nr, m, n) : (nr * n @ TCol, n @ TCol, 1 @ TLane, 1 @ TCol)])
    bl = ComposeLayout(3, 3, 3, TileLayout(S[(nr, kr, n, k) : (n * 64, k, 64, 1)]))

    @T.function
    def kernel(
        A: T.Tensor((mr, kr, m, k // 2), "uint32"),
        B: T.Tensor((nr, kr, n, k), "float16"),
        Output: T.Tensor((mr * m, nr * n), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        address = T.alloc_shared((1,), "uint32", layout="default")
        barrier = T.alloc_shared((1,), "uint64", layout="default")
        b = T.alloc_tensor(
            ty_args=[T.Tensor((nr, kr, n, k), "float16", scope="shared", layout=bl, align=1024)]
        )
        if warp == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(address), T.uint32(512)
            )
        if tid == 0:
            T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), T.uint32(1))
            for ni, ki, i, j in T.grid(nr, kr, n, k):
                b[ni, ki, i, j] = B[ni, ki, i, j]
        T.cuda.cta_sync()
        word = T.alloc_local((1,), "uint32", layout="default")
        for mi, ki, j in T.grid(mr, kr, k // 2):
            word[0] = A[mi, ki, tid, j]
            T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](
                address[0] + T.uint32(256 + (mi * kr + ki) * (k // 2) + j), word[0]
            )
        T.ptx.tcgen05.wait__st.sync.aligned()
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence__after_thread_sync()
        T.ptx.fence.proxy.async_.shared__cta()
        a = T.cuda.decl_tmem(
            address[0] + T.uint32(256),
            ty_args=[T.Tensor((mr, kr, m, k), "float16", scope="tmem", layout=al)],
        )
        d = T.cuda.decl_tmem(
            address[0], ty_args=[T.Tensor((mr, nr, m, n), "float32", scope="tmem", layout=dl)]
        )
        if tid == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.mma.cta_group::1.kind::f16"],
                d,
                a,
                b,
                instruction_shape=(m, n, k),
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                barrier.ptr_to([0])
            )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        result = T.alloc_local((1,), "float32", layout="default")
        for mi, ni, j in T.grid(mr, nr, n):
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](
                result[0], address[0] + T.uint32((mi * nr + ni) * n + j)
            )
            T.ptx.tcgen05.wait__ld.sync.aligned()
            Output[mi * m + tid, ni * n + j] = result[0]
        T.cuda.cta_sync()
        if warp == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], T.uint32(512))

    f = tvm_tirx.tirx.compile(kernel, target="cuda", pipeline="tirx")
    rng = np.random.default_rng(21)
    a = rng.uniform(-1, 1, (mr * m, kr * k)).astype("float16")
    b = rng.uniform(-1, 1, (nr * n, kr * k)).astype("float16")
    aa = np.ascontiguousarray(a.reshape(mr, m, kr, k).transpose(0, 2, 1, 3)).view("uint32")
    bb = np.ascontiguousarray(b.reshape(nr, n, kr, k).transpose(0, 2, 1, 3))
    out = tvm_tirx.runtime.tensor(np.zeros((mr * m, nr * n), "float32"), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(aa, tvm_tirx.cuda()), tvm_tirx.runtime.tensor(bb, tvm_tirx.cuda()), out)
    np.testing.assert_allclose(
        out.numpy(), a.astype("float32") @ b.astype("float32").T, rtol=2e-4, atol=2e-4
    )


@pytest.mark.skipif(not env.has_cuda_compute(10), reason="Blackwell required")
@pytest.mark.parametrize("cta_group", [1, 2])
@pytest.mark.parametrize(
    "scale_format,vec,kind", [("float8_e4m3fn", 4, "mxf4nvf4"), ("float8_e8m0fnu", 2, "mxf4")]
)
def test_tcgen05_fp4_scale_vectors(cta_group, scale_format, vec, kind):
    from tvm_tirx.backend.cuda.axis import TCol, TLane
    from tvm_tirx.tirx.layout import R

    rows, cols, k, kr = 128, 64, 64, 2
    m, n = rows * cta_group, cols * cta_group
    packed_a = ComposeLayout(4, 3, 3, TileLayout(S[(1, kr, rows, k // 2) : (0, k // 2, 128, 1)]))
    packed_b = ComposeLayout(4, 3, 3, TileLayout(S[(1, kr, cols, k // 2) : (0, k // 2, 128, 1)]))
    al = ComposeLayout(5, 3, 3, TileLayout(S[(1, kr, rows, k) : (0, k, 256, 1)]))
    bl = ComposeLayout(5, 3, 3, TileLayout(S[(1, kr, cols, k) : (0, k, 256, 1)]))
    dl = tmem_mma_operand_layout("D", (rows, n), "float32", M=m, cta_group=cta_group)
    sfal = TileLayout(
        S[(1, kr, 4, 32, vec) : (0, 16 @ TCol, 4 @ TCol, 1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane]
    )
    sfbl = TileLayout(
        S[(1, kr, n // 32, 32, vec) : (0, (n // 8) @ TCol, 4 @ TCol, 1 @ TLane, 1 @ TCol)]
        + R[4 : 32 @ TLane]
    )
    opcode = f"tcgen05.mma.cta_group::{cta_group}.kind::{kind}.block_scale.scale_vec::{vec}X"

    @T.function
    def kernel(
        A: T.Tensor((m, kr * k // 2), "uint8"),
        B: T.Tensor((n, kr * k // 2), "uint8"),
        Scales: T.Tensor((kr,), "uint32"),
        Output: T.Tensor((m, n), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=cta_group, block=128, cluster=cta_group))
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        cta = T.cuda.cluster_cta_id("x")
        ap = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (1, kr, rows, k // 2), "uint8", scope="shared", layout=packed_a, align=1024
                )
            ]
        )
        bp = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (1, kr, cols, k // 2), "uint8", scope="shared", layout=packed_b, align=1024
                )
            ]
        )
        a = T.decl_tensor(
            ap.data,
            ty_args=[
                T.Tensor((1, kr, rows, k), "float4_e2m1fn", scope="shared", layout=al, align=1024)
            ],
        )
        b = T.decl_tensor(
            bp.data,
            ty_args=[
                T.Tensor((1, kr, cols, k), "float4_e2m1fn", scope="shared", layout=bl, align=1024)
            ],
        )
        address = T.alloc_shared((1,), "uint32", layout="default")
        barrier = T.alloc_shared((1,), "uint64", layout="default")
        if warp == 0:
            T.ptx[f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"](
                T.address_of(address), T.uint32(256)
            )
        if tid == 0:
            T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), T.uint32(1))
            for ki, i, j in T.grid(kr, rows, k // 2):
                ap[0, ki, i, j] = A[cta * rows + i, ki * (k // 2) + j]
            for ki, i, j in T.grid(kr, cols, k // 2):
                bp[0, ki, i, j] = B[cta * cols + i, ki * (k // 2) + j]
        T.cuda.cta_sync()
        word = T.alloc_local((1,), "uint32", layout="default")
        for ki in range(kr):
            word[0] = Scales[ki]
            for j in range(4):
                T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](
                    address[0] + T.uint32(128 + ki * 4 + j), word[0]
                )
            for j in range(n // 32):
                T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](
                    address[0] + T.uint32(144 + ki * (n // 32) + j), word[0]
                )
        T.ptx.tcgen05.wait__st.sync.aligned()
        T.cuda.cta_sync()
        T.ptx.fence.mbarrier_init.release.cluster()
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cluster_sync()
        T.ptx.tcgen05.fence__after_thread_sync()
        d = T.cuda.decl_tmem(
            address[0], ty_args=[T.Tensor((1, 1, rows, n), "float32", scope="tmem", layout=dl)]
        )
        sfa = T.cuda.decl_tmem(
            address[0] + T.uint32(128),
            ty_args=[T.Tensor((1, kr, rows, vec), scale_format, scope="tmem", layout=sfal)],
        )
        sfb = T.cuda.decl_tmem(
            address[0] + T.uint32(144),
            ty_args=[T.Tensor((1, kr, n, vec), scale_format, scope="tmem", layout=sfbl)],
        )
        if cta == 0:
            if tid == 0:
                T.cuda.tile.apply(
                    T.ptx[opcode],
                    d,
                    a,
                    b,
                    sfa,
                    sfb,
                    instruction_shape=(m, n, k),
                )
                if T.constexpr(cta_group == 2):
                    T.ptx[
                        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
                    ](barrier.ptr_to([0]), T.uint16(3))
                else:
                    T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                        barrier.ptr_to([0])
                    )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        result = T.alloc_local((1,), "float32", layout="default")
        for j in range(n):
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](result[0], address[0] + T.uint32(j))
            T.ptx.tcgen05.wait__ld.sync.aligned()
            Output[cta * rows + tid, j] = result[0]
        T.cuda.cluster_sync()
        if warp == 0:
            T.ptx[f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned"]()
            T.ptx[f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32"](
                address[0], T.uint32(256)
            )

    f = tvm_tirx.tirx.compile(kernel, target="cuda", pipeline="tirx")
    rng = np.random.default_rng(22)
    a = rng.uniform(-4, 4, (m, kr * k)).astype("float4_e2m1fn")
    b = rng.uniform(-4, 4, (n, kr * k)).astype("float4_e2m1fn")

    def pack(x):
        raw = x.view("uint8")
        return (raw[:, ::2] << 4) | raw[:, 1::2]

    scale = np.tile(np.arange(1, vec + 1, dtype="float32"), (kr, 1))
    scale[1] *= 2
    raw_scale = np.zeros((kr, 4), "uint8")
    raw_scale[:, :vec] = scale.astype(scale_format).view("uint8")
    scale = raw_scale[:, :vec].copy().view(scale_format).astype("float32")
    out = tvm_tirx.runtime.tensor(np.zeros((m, n), "float32"), tvm_tirx.cuda())
    f(
        tvm_tirx.runtime.tensor(pack(a), tvm_tirx.cuda()),
        tvm_tirx.runtime.tensor(pack(b), tvm_tirx.cuda()),
        tvm_tirx.runtime.tensor(raw_scale.copy().view("uint32").reshape(kr), tvm_tirx.cuda()),
        out,
    )
    scale = np.repeat(scale, k // vec, axis=1).reshape(kr * k)
    reference = (a.astype("float32") * scale) @ (b.astype("float32") * scale).T
    np.testing.assert_allclose(out.numpy(), reference, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not env.has_cuda_compute(10), reason="Blackwell required")
@pytest.mark.parametrize("swizzle", [0, 1, 2])
@pytest.mark.parametrize("major", ["k", "mn"])
def test_tcgen05_descriptor_modes(swizzle, major):
    test_tcgen05_mma(128, False, "float16", 16, "f16", "float32", major, swizzle)


@pytest.mark.skipif(not env.has_cuda_compute(10), reason="Blackwell required")
@pytest.mark.parametrize("m,ws", [(64, False), (64, True), (128, False), (128, True)])
def test_tcgen05_tf32_32b_atomicity(m, ws):
    test_tcgen05_mma(m, ws, "float32", 8, "tf32", "float32", "mn", swizzle=4)


@pytest.mark.skipif(not env.has_cuda_compute(10), reason="Blackwell required")
@pytest.mark.parametrize("major", ["k", "mn"])
def test_tcgen05_fp8_no_swizzle(major):
    test_tcgen05_mma(64, False, "float8_e4m3fn", 32, "f8f6f4", "float32", major, swizzle=0)


@pytest.mark.skipif(not env.has_cuda_compute(10), reason="Blackwell required")
@pytest.mark.parametrize(
    "dtype,k,kind,major,message",
    [
        ("float32", 8, "tf32", "k", "no matching shared matrix descriptor"),
        ("float16", 16, "f16", "mn", "no matching shared matrix descriptor"),
    ],
)
def test_tcgen05_32b_atomicity_rejects_incompatible_operand(dtype, k, kind, major, message):
    with pytest.raises(Exception, match=message):
        test_tcgen05_mma(128, False, dtype, k, kind, "float32", major, swizzle=4)
