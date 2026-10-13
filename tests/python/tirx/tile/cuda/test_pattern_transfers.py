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

"""Instruction-specific transfer images and explicit synchronization."""

import numpy as np
import pytest

import tvm
from tvm.backend.cuda.axis import TCol, TLane, laneid, wid_in_wg
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.layout import ComposeLayout, R, S, TileLayout


def _build(f):
    return tvm.tirx.compile(f, target="cuda", pipeline="tirx")


@pytest.mark.skipif(not env.has_cuda_compute(10), reason="Blackwell required")
@pytest.mark.parametrize("shape", ["32x32b", "16x64b", "16x128b", "16x256b"])
@pytest.mark.parametrize("count", [1, 4, 32])
@pytest.mark.parametrize("store", [False, True])
def test_tmem(shape, count, store):
    if shape == "32x32b":
        rows, cols, nregs = 32, count, count
        reg_layout = TileLayout(S[(4, 32, count) : (1 @ wid_in_wg, 1 @ laneid, 1)])
    elif shape == "16x64b":
        rows, cols, nregs = 16, 2 * count, count
        reg_layout = TileLayout(
            S[(4, 2, 8, count, 2) : (1 @ wid_in_wg, 1 @ laneid, 4 @ laneid, 1, 2 @ laneid)]
        )
    elif shape == "16x128b":
        rows, cols, nregs = 16, 4 * count, 2 * count
        reg_layout = TileLayout(
            S[(4, 2, 8, count, 4) : (1 @ wid_in_wg, 1, 4 @ laneid, 2, 1 @ laneid)]
        )
    else:
        rows, cols, nregs = 16, 8 * count, 4 * count
        reg_layout = TileLayout(
            S[(4, 2, 8, count, 4, 2) : (1 @ wid_in_wg, 2, 4 @ laneid, 4, 1 @ laneid, 1)]
        )
    tmem_layout = TileLayout(S[(4, rows, cols) : (32 @ TLane, 1 @ TLane, 1 @ TCol)])
    ld = f"tcgen05.ld.sync.aligned.{shape}.x{count}.b32"
    st = f"tcgen05.st.sync.aligned.{shape}.x{count}.b32"
    width = max(32, cols)

    @T.function
    def kernel(A: T.Tensor((128, cols), "uint32"), Output: T.Tensor((128, cols), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        address = T.alloc_shared((1,), "uint32", layout="default")
        if warp == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(address), T.uint32(width)
            )
        T.cuda.cta_sync()
        mem = T.cuda.decl_tmem(
            address[0],
            ty_args=[T.Tensor((4, rows, cols), "uint32", scope="tmem", layout=tmem_layout)],
        )
        raw = T.alloc_local((nregs,), "uint32", layout="default")
        frag = raw.view(4, rows, cols, layout=reg_layout)
        word = T.alloc_local((1,), "uint32", layout="default")
        if T.constexpr(store):
            for i in range(nregs):
                raw[i] = A[tid, i]
            T.cuda.tile.apply(T.ptx[st], mem, frag, scope="warpgroup")
            T.ptx.tcgen05.wait__st.sync.aligned()
            for i in range(cols):
                T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](word[0], address[0] + T.uint32(i))
                T.ptx.tcgen05.wait__ld.sync.aligned()
                Output[tid, i] = word[0]
        else:
            for i in range(cols):
                word[0] = A[tid, i]
                T.ptx["tcgen05.st.sync.aligned.32x32b.x1.b32"](address[0] + T.uint32(i), word[0])
            T.ptx.tcgen05.wait__st.sync.aligned()
            T.cuda.tile.apply(T.ptx[ld], frag, mem, scope="warpgroup")
            T.ptx.tcgen05.wait__ld.sync.aligned()
            for i in range(nregs):
                Output[tid, i] = raw[i]
        T.cuda.cta_sync()
        if warp == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], T.uint32(width))

    f = _build(kernel)
    x = np.arange(128 * cols, dtype="uint32").reshape(128, cols)
    out = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    f(tvm.runtime.tensor(x, tvm.cuda()), out)
    actual = out.numpy()
    for t in range(128):
        lane, warp = t % 32, t // 32
        for reg in range(nregs):
            # Independent host reconstruction of the documented PTX diagram.
            if shape == "32x32b":
                row, col = lane, reg
            elif shape == "16x64b":
                row, col = lane // 4 + 8 * (lane % 2), lane // 2 % 2 + 2 * reg
            elif shape == "16x128b":
                row, col = lane // 4 + 8 * (reg % 2), lane % 4 + 4 * (reg // 2)
            else:
                row, col = lane // 4 + 8 * (reg // 2 % 2), reg % 2 + 2 * (lane % 4) + 8 * (reg // 4)
            if store:
                assert actual[32 * warp + row, col] == x[t, reg]
            else:
                assert actual[t, reg] == x[32 * warp + row, col]


@pytest.mark.skipif(not env.has_cuda_compute(8), reason="Ampere required")
@pytest.mark.parametrize("ignore", [False, True])
def test_cp_async(ignore):
    @T.function
    def kernel(A: T.Tensor((3, 4), "uint32"), Output: T.Tensor((3, 4), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        shared = T.alloc_shared((3, 4), "uint32", layout="default")
        T.cuda.tile.apply(
            T.ptx["cp.async.cg.shared.global"], shared, A, chunk_bytes=16, ignore_src=T.bool(ignore)
        )
        T.ptx.cp.async_.commit_group()
        T.ptx.cp.async_.wait_group(0)
        for i, j in T.grid(3, 4):
            Output[i, j] = shared[i, j]

    f = _build(kernel)
    x = np.arange(12, dtype="uint32").reshape(3, 4)
    out = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    f(tvm.runtime.tensor(x, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), np.zeros_like(x) if ignore else x)


@pytest.mark.skipif(not env.has_cuda_compute(10), reason="Blackwell required")
@pytest.mark.parametrize(
    "shape,multicast",
    [
        ("128x256b", ""),
        ("128x128b", ""),
        ("4x256b", ""),
        ("64x128b", "warpx2::02_13"),
        ("64x128b", "warpx2::01_23"),
        ("32x128b", "warpx4"),
    ],
)
@pytest.mark.parametrize("swizzle", [0, 1, 2, 3])
def test_tmem_copy(shape, multicast, swizzle, ldo_override=None):
    rows, bits = shape.split("x")
    rows, cols = int(rows), int(bits[:-1]) // 32
    row_words = 4 << swizzle
    ldo = 128 if swizzle == 0 and cols == 8 else 16
    if ldo_override is not None:
        ldo = ldo_override
    sdo = 256 if swizzle == 0 and cols == 8 else 8 * row_words * 4
    if swizzle == 0:
        sl = TileLayout(
            S[(max(1, rows // 8), min(8, rows), cols // 4, 4) : (sdo // 4, 4, ldo // 4, 1)]
        )
    else:
        sl = ComposeLayout(
            2,
            swizzle,
            3,
            TileLayout(S[(max(1, rows // 8), min(8, rows), cols) : (sdo // 4, row_words, 1)]),
        )
    replicas = (
        R[4 : 32 @ TLane]
        if multicast == "warpx4"
        else R[2 : (64 if multicast == "warpx2::02_13" else 32) @ TLane]
        if multicast
        else []
    )
    if multicast == "warpx2::01_23":
        tl = TileLayout(S[(2, 32, cols) : (64 @ TLane, 1 @ TLane, 1 @ TCol)] + replicas)
    else:
        spec = S[(rows, cols) : ((32 if rows == 4 else 1) @ TLane, 1 @ TCol)]
        tl = TileLayout(spec + replicas if multicast else spec)
    opcode = f"tcgen05.cp.cta_group::1.{shape}" + (f".{multicast}" if multicast else "")

    @T.function
    def kernel(A: T.Tensor((rows, cols), "uint32"), Output: T.Tensor((128, cols), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        backing = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (max(8, rows) * max(row_words, cols),),
                    "uint32",
                    scope="shared",
                    layout="default",
                    align=1024,
                )
            ]
        )
        shared = backing.view(rows, cols, layout=sl)
        address = T.alloc_shared((1,), "uint32", layout="default")
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if warp == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(address), T.uint32(32)
            )
        if tid == 0:
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
            for r, c in T.grid(rows, cols):
                shared[r, c] = A[r, c]
        T.cuda.cta_sync()
        T.ptx.fence.proxy.async_.shared__cta()
        mem = T.cuda.decl_tmem(
            address[0], ty_args=[T.Tensor((rows, cols), "uint32", scope="tmem", layout=tl)]
        )
        if tid == 0:
            T.cuda.tile.apply(
                T.ptx[opcode],
                mem,
                shared,
                ldo_bytes=ldo,
                sdo_bytes=(0 if rows == 4 else sdo),
                swizzle=swizzle,
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                mbar.ptr_to([0])
            )
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        word = T.alloc_local((1,), "uint32", layout="default")
        for c in range(cols):
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](word[0], address[0] + T.uint32(c))
            T.ptx.tcgen05.wait__ld.sync.aligned()
            Output[tid, c] = word[0]
        T.cuda.cta_sync()
        if warp == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], T.uint32(32))

    f = _build(kernel)
    x = np.arange(rows * cols, dtype="uint32").reshape(rows, cols)
    output = tvm.runtime.tensor(np.zeros((128, cols), "uint32"), tvm.cuda())
    f(tvm.runtime.tensor(x, tvm.cuda()), output)
    actual = output.numpy()
    for r in range(rows):
        lanes = [32 * r] if rows == 4 else [r]
        if multicast == "warpx4":
            lanes = [r + 32 * i for i in range(4)]
        elif multicast == "warpx2::02_13":
            lanes = [r, r + 64]
        elif multicast == "warpx2::01_23":
            lanes = [r // 32 * 64 + r % 32, r // 32 * 64 + r % 32 + 32]
        for lane in lanes:
            np.testing.assert_array_equal(actual[lane], x[r])


@pytest.mark.skipif(not env.has_cuda_compute(9), reason="Hopper required")
@pytest.mark.parametrize("swizzle", [0, 1, 2, 3])
def test_tma(swizzle):
    cols = 4 << swizzle
    sl = ComposeLayout(2, swizzle, 3, TileLayout(S[(2, 8, cols)]))

    @T.function
    def kernel(A: T.Tensor((16, cols), "uint32"), Output: T.Tensor((16, cols), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        shared = T.alloc_tensor(
            ty_args=[T.Tensor((2, 8, cols), "uint32", scope="shared", layout=sl, align=1024)]
        )
        mbar = T.alloc_shared((1,), "uint64", layout="default")
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(16 * cols * 4))
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                shared,
                A,
                box=(8, cols),
                coords=(0, 0),
                coord_steps=((8, 0),),
                swizzle=swizzle,
                mbar=mbar.ptr_to([0]),
            )
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if lane == 0:
            for g, r, c in T.grid(2, 8, cols):
                Output[g * 8 + r, c] = shared[g, r, c]

    f = _build(kernel)
    x = np.arange(16 * cols, dtype="uint32").reshape(16, cols)
    output = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    f(tvm.runtime.tensor(x, tvm.cuda()), output)
    np.testing.assert_array_equal(output.numpy(), x)


@pytest.mark.skipif(not env.has_cuda_compute(9), reason="Hopper required")
@pytest.mark.parametrize("reduce", [False, True])
def test_tma_store(reduce):
    opcode = (
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group"
        if reduce
        else "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
    )

    @T.function
    def kernel(A: T.Tensor((16, 16), "float32"), Output: T.Tensor((16, 16), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        shared = T.alloc_shared((2, 8, 16), "float32", layout="default", align=128)
        if lane == 0:
            for i, j in T.grid(16, 16):
                shared[i // 8, i % 8, j] = A[i, j]
        T.cuda.cta_sync()
        T.ptx.fence.proxy.async_.shared__cta()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx[opcode], Output, shared, box=(8, 16), coords=(0, 0), coord_steps=((8, 0),)
            )
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group(0)

    f = _build(kernel)
    x = np.arange(256, dtype="float32").reshape(16, 16)
    out = tvm.runtime.tensor(np.ones_like(x), tvm.cuda())
    f(tvm.runtime.tensor(x, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), x + 1 if reduce else x)


@pytest.mark.skipif(not env.has_cuda_compute(10), reason="Blackwell required")
def test_tma_gather4():
    @T.function
    def kernel(A: T.Tensor((32, 32), "uint32"), Output: T.Tensor((4, 32), "uint32"), row: T.int32):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        shared = T.alloc_shared((4, 32), "uint32", layout="default", align=128)
        barrier = T.alloc_shared((1,), "uint64", layout="default")
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), T.uint32(512))
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes"
                ],
                shared,
                A,
                box=(1, 32),
                coords=(0, 0),
                gather4=(row, row + 3, row + 7, row + 11),
                mbar=barrier.ptr_to([0]),
            )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.cuda.cta_sync()
        for i in range(4):
            Output[i, lane] = shared[i, lane]

    f = _build(kernel)
    x = np.arange(1024, dtype="uint32").reshape(32, 32)
    out = tvm.runtime.tensor(np.zeros((4, 32), "uint32"), tvm.cuda())
    f(tvm.runtime.tensor(x, tvm.cuda()), out, 5)
    np.testing.assert_array_equal(out.numpy(), x[[5, 8, 12, 16]])


@pytest.mark.skipif(not env.has_cuda_compute(9), reason="Hopper required")
def test_dsmem_remote_addresses():
    sl = TileLayout(S[(3, 8) : (16, 1)])

    @T.function
    def kernel(A: T.Tensor((3, 8), "uint32"), Output: T.Tensor((3, 8), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=1, cluster=2))
        cta = T.cuda.cluster_cta_id("x")
        src = T.alloc_tensor(ty_args=[T.Tensor((3, 8), "uint32", scope="shared", layout=sl)])
        dst = T.alloc_tensor(ty_args=[T.Tensor((3, 8), "uint32", scope="shared", layout=sl)])
        barrier = T.alloc_shared((1,), "uint64", layout="default")
        T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), T.uint32(1))
        T.ptx.fence.mbarrier_init.release.cluster()
        if cta == 0:
            for i, j in T.grid(3, 8):
                src[i, j] = A[i, j]
            T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cluster_sync()
        if cta == 0:
            T.cuda.tile.apply(
                T.ptx["cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"],
                dst,
                src,
                chunk_bytes=32,
                mbar=barrier.ptr_to([0]),
                remote_cta_id=1,
            )
        else:
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), T.uint32(96))
            T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
            for i, j in T.grid(3, 8):
                Output[i, j] = dst[i, j]
        T.cuda.cluster_sync()

    f = _build(kernel)
    x = np.arange(24, dtype="uint32").reshape(3, 8)
    out = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    f(tvm.runtime.tensor(x, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), x)


@pytest.mark.skipif(not env.has_cuda_compute(9), reason="Hopper required")
@pytest.mark.parametrize("rank", range(1, 6))
def test_tma_rank_and_distinct_descriptors(rank):
    """Two identically shaped operands must keep independent tensor maps."""
    shape = (2,) * (rank - 1) + (16,)
    count = int(np.prod(shape))
    opcode = f"cp.async.bulk.tensor.{rank}d.shared::cta.global.tile.mbarrier::complete_tx::bytes"

    @T.function
    def kernel(
        A: T.Tensor(shape, "uint32"),
        B: T.Tensor(shape, "uint32"),
        Output: T.Tensor((count,), "uint32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        a = T.alloc_shared(shape, "uint32", layout="default", align=128)
        b = T.alloc_shared(shape, "uint32", layout="default", align=128)
        barrier = T.alloc_shared((1,), "uint64", layout="default")
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(barrier.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier.ptr_to([0]), T.uint32(count * 8))
            T.cuda.tile.apply(
                T.ptx[opcode], a, A, box=shape, coords=(0,) * rank, mbar=barrier.ptr_to([0])
            )
            T.cuda.tile.apply(
                T.ptx[opcode], b, B, box=shape, coords=(0,) * rank, mbar=barrier.ptr_to([0])
            )
        T.cuda.mbarrier_wait(barrier.ptr_to([0]), 0)
        T.cuda.cta_sync()
        aa = a.view(count, layout="default")
        bb = b.view(count, layout="default")
        if lane == 0:
            for i in range(count):
                Output[i] = aa[i] + bb[i]

    f = _build(kernel)
    a = np.arange(count, dtype="uint32").reshape(shape)
    b = (a * 3 + 7).copy()
    out = tvm.runtime.tensor(np.zeros(count, "uint32"), tvm.cuda())
    f(tvm.runtime.tensor(a, tvm.cuda()), tvm.runtime.tensor(b, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), (a + b).reshape(-1))


@pytest.mark.skipif(not env.has_cuda_compute(10), reason="Blackwell required")
def test_swizzled_tmem_copy_ignores_ldo():
    test_tmem_copy("128x256b", "", 3, ldo_override=32)
