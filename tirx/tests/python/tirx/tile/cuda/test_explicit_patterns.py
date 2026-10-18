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

"""Executable examples and contract tests for the explicit PTX tile prototype."""

from types import SimpleNamespace

import numpy as np
import pytest

import tvm_tirx
from tvm_tirx.backend.cuda.axis import laneid
from tvm_tirx.backend.cuda.ptx import instruction_name
from tvm_tirx.ir import Var, assert_structural_equal
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env
from tvm_tirx.tirx.layout import S, TileLayout
from tvm_tirx.tirx.tile.dispatch import run_dispatch

LD = "ld.global.v4.b32"
LDMATRIX = "ldmatrix.sync.aligned.m8n8.x2.shared.b16"


def global_kernel(gapped=False):
    original = TileLayout(S[(15, 2, 7, 2) : (4, 1, 60, 2)])
    dst_layout = TileLayout(S[(15, 2, 7, 2) : (8, 2, 120, 4)]) if gapped else original

    @T.function
    def kernel(
        A: T.Tensor((15, 2, 7, 2), "float32", layout=original, align=16),
        Out: T.Tensor((420,), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        R = T.alloc_tensor(
            ty_args=[T.Tensor((15, 2, 7, 2), "float32", scope="local", layout=dst_layout)]
        )
        A_std = A.rearrange("i b j d -> j i (d b)")
        R_std = R.rearrange("i b j d -> j i (d b)")
        T.cuda.tile.apply(T.ptx[LD], R_std, A_std, scope="thread")
        for j, i, d, b in T.grid(7, 15, 2, 2):
            Out[60 * j + 4 * i + 2 * d + b] = R[i, b, j, d]

    return kernel


def matrix_kernel(partial=False, block=32):
    shared = TileLayout(S[(16, 4, 2) : (8, 1, 128)])
    registers = TileLayout(S[(2, 2, 8, 4) : (8, 2, 4 @ laneid, 1 @ laneid)])

    @T.function
    def kernel(A: T.Tensor((16, 4, 2), "uint32"), Out: T.Tensor((32, 12), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=block))
        lane = T.cuda.lane_id()
        SM = T.alloc_tensor(
            ty_args=[T.Tensor((16, 4, 2), "uint32", scope="shared", layout=shared, align=16)]
        )
        for q in range(4):
            n = lane + 32 * q
            SM[n // 8, (n // 2) % 4, n % 2] = A[n // 8, (n // 2) % 4, n % 2]
        T.cuda.cta_sync()
        raw = T.alloc_tensor(ty_args=[T.Tensor((12,), "uint32", scope="local", layout="default")])
        for i in T.unroll(12):
            raw[i] = T.uint32(4294966519)  # bit pattern of int32(-777)
        R = T.decl_tensor(
            raw.ptr_to([0]),
            ty_args=[T.Tensor((2, 2, 8, 4), "uint32", scope="local", layout=registers)],
        )
        S_std = SM.rearrange("(r m) w k -> k m r w", r=8, m=2)
        if T.constexpr(partial):
            if lane < 16:
                T.cuda.tile.apply(T.ptx[LDMATRIX], R, S_std, scope="warp")
        else:
            T.cuda.tile.apply(T.ptx[LDMATRIX], R, S_std, scope="warp")
        for i in T.unroll(12):
            Out[lane, i] = raw[i]

    return kernel


def compile_kernel(kernel):
    target = tvm_tirx.target.Target(
        {"kind": "cuda", "backend_config": {"cuda": {"target_format": "ptx"}}}
    )
    with target:
        return tvm_tirx.tirx.compile(kernel, target=target, pipeline="tirx")


@pytest.mark.parametrize("factory", [global_kernel, matrix_kernel])
def test_roundtrip(factory):
    kernel = factory()
    assert_structural_equal(kernel, tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(kernel)))
    reparsed = tvm_tirx.script.from_source(kernel.script(), extra_vars={"T": T})
    assert_structural_equal(kernel, reparsed)


def test_instruction_name():
    assert instruction_name(T.ptx["ld.b32.v4.global"]) == LD
    assert instruction_name(T.ptx[LDMATRIX]) == LDMATRIX
    with pytest.raises(ValueError, match="not fully specified"):
        instruction_name(T.ptx.ld)


def _dispatch(
    src_shape=(2, 4),
    src_layout=None,
    dst_shape=None,
    dst_layout=None,
    opcode=LD,
    scope="thread",
    align=16,
    active=32,
):
    matrix = opcode.startswith("ldmatrix")
    src = Var(
        "src",
        T.Tensor(
            src_shape,
            "uint32",
            scope="shared" if matrix else "global",
            layout=src_layout or TileLayout(S[src_shape]),
            align=align,
        ),
    )
    dst_shape = dst_shape or src_shape
    dst = Var(
        "dst",
        T.Tensor(dst_shape, "uint32", scope="local", layout=dst_layout or TileLayout(S[dst_shape])),
    )
    ctx = SimpleNamespace(
        target=tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"}),
        scope_kind=scope,
        launch_params={},
        var_range_map={},
        intra={"laneid": (active, 0)} if matrix else {},
    )
    return run_dispatch(T.cuda.tile.apply(T.ptx[opcode], dst, src, scope=scope), ctx)


def test_contract_rejections():
    with pytest.raises(RuntimeError, match="consecutive elements"):
        _dispatch(src_layout=TileLayout(S[(2, 4) : (8, 2)]))
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        _dispatch(src_layout=TileLayout(S[(2, 4)] + 1))
    with pytest.raises(RuntimeError, match="known 16-byte-aligned base"):
        _dispatch(align=4)
    with pytest.raises(RuntimeError, match="Repeat shapes must agree"):
        _dispatch(dst_shape=(3, 4))
    with pytest.raises(RuntimeError, match="No explicit tile pattern"):
        _dispatch(opcode="barrier.sync.aligned")
    with pytest.raises(RuntimeError, match="nonoverlapping storage strides"):
        _dispatch(dst_layout=TileLayout(S[(2, 4) : (4, 0)]))


def test_matrix_contract_rejections():
    shape = (2, 2, 8, 4)
    reg = TileLayout(S[shape : (8, 2, 4 @ laneid, 1 @ laneid)])
    shared = TileLayout(S[shape : (128, 8, 16, 1)])
    kw = dict(src_shape=shape, src_layout=shared, dst_layout=reg, opcode=LDMATRIX, scope="warp")
    _dispatch(**kw)
    # Last word axis becomes offsets 0,2,1,3 after splitting into two bits.
    with pytest.raises(RuntimeError, match="consecutive elements"):
        _dispatch(**{**kw, "src_layout": TileLayout(S[(2, 2, 8, 2, 2) : (128, 8, 16, 1, 2)])})
    with pytest.raises(RuntimeError, match="laneid owner"):
        _dispatch(**{**kw, "dst_layout": TileLayout(S[shape : (8, 2, 1 @ laneid, 8 @ laneid)])})
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        _dispatch(**{**kw, "src_layout": TileLayout(S[shape : (128, 8, 17, 1)])})
    with pytest.raises(RuntimeError, match="all 32 lanes"):
        _dispatch(**{**kw, "active": 16})
    with pytest.raises(RuntimeError, match="all 32 lanes"):
        _dispatch(**{**kw, "scope": "thread"})


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(8), reason="CUDA compute required")
@pytest.mark.parametrize("gapped", [False, True])
def test_global_gpu(gapped):
    compiled = compile_kernel(global_kernel(gapped))
    ptx = compiled.mod.imports[0].inspect_source("ptx")
    assert ptx.count("ld.global.v4.b32") == 105
    a = np.arange(420, dtype="float32").reshape(15, 2, 7, 2) + 0.25
    out = tvm_tirx.runtime.tensor(np.zeros(420, dtype="float32"), tvm_tirx.cuda())
    compiled(tvm_tirx.runtime.tensor(a, tvm_tirx.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), a.ravel())


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(8), reason="CUDA compute required")
def test_matrix_gpu():
    compiled = compile_kernel(matrix_kernel())
    ptx = compiled.mod.imports[0].inspect_source("ptx")
    assert ptx.count(LDMATRIX) == 2
    values = np.arange(128, dtype="uint32").reshape(16, 4, 2)
    a = ((values + 1000) << 16) | (values + 1)
    expected = np.full((32, 12), 4294966519, dtype="uint32")
    for lane in range(32):
        for k in range(2):
            for m in range(2):
                expected[lane, 8 * k + 2 * m] = a[2 * (lane // 4) + m, lane % 4, k]
    out = tvm_tirx.runtime.tensor(np.zeros_like(expected), tvm_tirx.cuda())
    compiled(tvm_tirx.runtime.tensor(a, tvm_tirx.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), expected)


@pytest.mark.parametrize("block,partial", [(32, True), (1, False), (16, False), (31, False)])
def test_partial_warp_rejected_in_pipeline(block, partial):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target, pytest.raises((RuntimeError, ValueError), match="all 32 lanes"):
        tvm_tirx.tirx.transform.TileDispatch()(
            tvm_tirx.IRModule({"main": matrix_kernel(partial=partial, block=block)})
        )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(8), reason="CUDA compute required")
def test_region_starts_and_element_offsets_gpu():
    @T.function
    def kernel(A: T.Tensor((32,), "uint32"), Out: T.Tensor((32,), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        raw = T.alloc_tensor(ty_args=[T.Tensor((32,), "uint32", scope="local", layout="default")])
        for i in T.unroll(32):
            raw[i] = T.uint32(777)
        src = T.decl_tensor(
            A.data,
            ty_args=[T.Tensor((3, 8), "uint32", align=16, layout=T.TileLayout(T.S[(3, 8)] + (4)))],
        )
        dst = T.decl_tensor(
            raw.data,
            ty_args=[
                T.Tensor((3, 8), "uint32", scope="local", layout=T.TileLayout(T.S[(3, 8)] + (4)))
            ],
        )
        T.cuda.tile.apply(T.ptx[LD], dst[1:3, 2:6], src[0:2, 4:8])
        for i in T.unroll(32):
            Out[i] = raw[i]

    compiled = compile_kernel(kernel)
    a = np.arange(32, dtype="uint32") + 100
    expected = np.full(32, 777, dtype="uint32")
    expected[14:18] = a[8:12]
    expected[22:26] = a[16:20]
    out = tvm_tirx.runtime.tensor(np.zeros_like(expected), tvm_tirx.cuda())
    compiled(tvm_tirx.runtime.tensor(a, tvm_tirx.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), expected)


@pytest.mark.skipif(not env.has_cuda_compute(9), reason="Hopper required for stmatrix")
@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("store", [False, True])
def test_matrix_modes(count, transpose, store):
    shared_layout = TileLayout(S[(count, 8, 4) : (128, 8, 1)])
    register_layout = TileLayout(S[(count, 8, 4) : (3, 4 @ laneid, 1 @ laneid)])
    op = (
        ("stmatrix" if store else "ldmatrix")
        + f".sync.aligned.m8n8.x{count}"
        + (".trans" if transpose else "")
        + ".shared.b16"
    )

    @T.function
    def kernel(A: T.Tensor((count, 8, 4), "uint32"), Output: T.Tensor((count, 8, 4), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        shared = T.alloc_tensor(
            ty_args=[T.Tensor((count, 8, 4), "uint32", scope="shared", layout=shared_layout)]
        )
        raw = T.alloc_local((count * 3,), "uint32", layout="default")
        regs = raw.view(count, 8, 4, layout=register_layout)
        if T.constexpr(store):
            for m in range(count):
                raw[m * 3] = A[m, lane // 4, lane % 4]
            T.cuda.tile.apply(T.ptx[op], shared, regs, scope="warp")
            T.cuda.cta_sync()
            for m in range(count):
                Output[m, lane // 4, lane % 4] = shared[m, lane // 4, lane % 4]
        else:
            for m in range(count):
                shared[m, lane // 4, lane % 4] = A[m, lane // 4, lane % 4]
            T.cuda.cta_sync()
            T.cuda.tile.apply(T.ptx[op], regs, shared, scope="warp")
            for m in range(count):
                Output[m, lane // 4, lane % 4] = raw[m * 3]

    f = compile_kernel(kernel)
    x = np.arange(count * 64, dtype="uint16").reshape(count, 8, 8)
    words = x.view("uint32")
    out = tvm_tirx.runtime.tensor(np.zeros_like(words), tvm_tirx.cuda())
    f(tvm_tirx.runtime.tensor(words, tvm_tirx.cuda()), out)
    expected = x.transpose(0, 2, 1) if transpose else x
    np.testing.assert_array_equal(out.numpy().view("uint16"), expected)


def test_tma_rejects_under_aligned_shared_atom():
    @T.function
    def kernel(A: T.Tensor((16,), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        shared = T.alloc_shared((16,), "uint32", layout="default", align=64)
        barrier = T.alloc_shared((1,), "uint64", layout="default")
        T.cuda.tile.apply(
            T.ptx["cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes"],
            shared,
            A,
            box=(16,),
            coords=(0,),
            mbar=barrier.ptr_to([0]),
        )

    with tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"}):
        with pytest.raises(RuntimeError, match="known 128-byte-aligned base"):
            tvm_tirx.tirx.transform.TileDispatch()(tvm_tirx.IRModule({"main": kernel}))


def test_cluster_pattern_requires_launch_cluster():
    @T.function
    def kernel(A: T.Tensor((2, 1, 32, 4), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32))
        r = T.alloc_local((2, 1, 32, 4), "uint32", layout="default")
        T.cuda.tile.apply(T.ptx[LD], r, A, scope="cluster")

    with tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"}):
        with pytest.raises(RuntimeError, match="explicit LaunchConfig.cluster"):
            tvm_tirx.tirx.transform.TileDispatch()(tvm_tirx.IRModule({"main": kernel}))


def test_swizzled_tmem_copy_rejects_gapped_words():
    from tvm_tirx.backend.cuda.axis import TCol, TLane
    from tvm_tirx.tirx.layout import ComposeLayout

    shared_layout = ComposeLayout(2, 3, 3, TileLayout(S[(128, 2, 4) : (32, 8, 1)]))
    tmem_layout = TileLayout(S[(128, 8) : (1 @ TLane, 1 @ TCol)])

    @T.function
    def kernel():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        src = T.alloc_tensor(
            ty_args=[T.Tensor((128, 8), "uint32", scope="shared", layout=shared_layout, align=1024)]
        )
        dst = T.cuda.decl_tmem(
            T.uint32(0), ty_args=[T.Tensor((128, 8), "uint32", scope="tmem", layout=tmem_layout)]
        )
        T.cuda.tile.apply(
            T.ptx["tcgen05.cp.cta_group::1.128x256b"],
            dst,
            src,
        )

    with tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"}):
        with pytest.raises(RuntimeError, match="descriptor address walk"):
            tvm_tirx.tirx.transform.TileDispatch()(tvm_tirx.IRModule({"main": kernel}))


@pytest.mark.parametrize(
    "index,block,size",
    [
        (lambda: T.cuda.thread_idx("x"), (32, 2), 32),
        (lambda: T.cuda.thread_idx("y"), (32, 2), 2),
        (lambda: T.cuda.block_idx("x"), 32, 3),
        (T.cuda.linear_thread_id, (32, 2), 64),
        (T.cuda.lane_id, 64, 32),
        (T.cuda.warp_id, 64, 2),
        (T.cuda.thread_in_warpgroup, 256, 128),
        (T.cuda.warp_in_warpgroup, 256, 4),
        (T.cuda.warpgroup_id, 256, 2),
    ],
)
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_cuda_index_region_bounds(index, block, size, offset):
    @T.function
    def kernel(A: T.Tensor((size,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=3, block=block))
        r = T.alloc_local((1,), "float32", layout="default")
        start = index() + offset
        T.cuda.tile.apply(T.ptx["ld.global.f32"], r, A[start : start + 1])

    with tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"}):
        if offset == 0:
            tvm_tirx.tirx.transform.TileDispatch()(tvm_tirx.IRModule({"main": kernel}))
        else:
            with pytest.raises(RuntimeError, match="in bounds|exceeds tensor shape"):
                tvm_tirx.tirx.transform.TileDispatch()(tvm_tirx.IRModule({"main": kernel}))
