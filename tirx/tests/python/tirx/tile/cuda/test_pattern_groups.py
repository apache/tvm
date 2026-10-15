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

"""Explicit group axes, scalar callbacks, and native register operands."""

import numpy as np
import pytest

import tvm
from tvm.backend.cuda.axis import laneid, warpid
from tvm.ir import assert_structural_equal
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.layout import S, TileLayout


def build(func):
    target = tvm.target.Target(
        {"kind": "cuda", "backend_config": {"cuda": {"target_format": "ptx"}}}
    )
    return tvm.tirx.compile(func, target=target, pipeline="tirx")


@T.function(private=True)
def affine(x: T.float32, scale: T.float32, bias: T.float32) -> T.float32:
    value = x * scale
    return value + bias


def group_kernel(ewise=False):
    layout = TileLayout(S[(2, 32, 3, 4) : (1 @ warpid, 1 @ laneid, 8, 2)])

    @T.function
    def kernel(A: T.Tensor((2, 32, 3, 4), "float32"), Output: T.Tensor((2, 32, 3, 4), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        R = T.alloc_tensor(
            ty_args=[T.Tensor((2, 32, 3, 4), "float32", scope="local", layout=layout)]
        )
        T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], R, A, scope="cta")
        if T.constexpr(ewise):
            T.cuda.tile.ewise(affine, R, R, T.float32(2), T.float32(1), scope="cta")
        T.cuda.tile.apply(T.ptx["st.global.v4.b32"], Output, R, scope="cta")

    return kernel


@pytest.mark.parametrize("ewise", [False, True])
def test_group_roundtrip(ewise):
    f = group_kernel(ewise)
    assert_structural_equal(f, tvm.ir.load_json(tvm.ir.save_json(f)))
    assert_structural_equal(f, tvm.script.from_source(f.script(), extra_vars={"T": T, "I": I}))


@pytest.mark.skipif(not env.has_cuda_compute(8), reason="CUDA required")
@pytest.mark.parametrize("ewise", [False, True])
def test_group_gpu(ewise):
    kernel = build(group_kernel(ewise))
    ptx = kernel.mod.imports[0].inspect_source("ptx")
    assert ptx.count("ld.global.v4.b32") == 3
    assert ptx.count("st.global.v4.b32") == 3
    x = np.arange(768, dtype="float32").reshape(2, 32, 3, 4)
    out = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    kernel(tvm.runtime.tensor(x, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), x * 2 + 1 if ewise else x)


@pytest.mark.skipif(not env.has_cuda_compute(8), reason="CUDA required")
def test_explicit_scalar_instruction():
    @T.function
    def kernel(A: T.Tensor((8, 1), "float32"), Output: T.Tensor((8, 1), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        R = T.alloc_tensor(ty_args=[T.Tensor((8, 1), "float32", scope="local", layout="default")])
        T.cuda.tile.apply(T.ptx["ld.global.f32"], R, A)
        T.cuda.tile.apply(T.ptx["add.rn.f32"], R, R, T.float32(3))
        T.cuda.tile.apply(T.ptx["st.global.f32"], Output, R)

    f = build(kernel)
    x = np.arange(8, dtype="float32").reshape(8, 1)
    out = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    f(tvm.runtime.tensor(x, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), x + 3)


@pytest.mark.skipif(not env.has_cuda(), reason="CUDA required")
@pytest.mark.parametrize(
    "bits,count",
    [(8, 1), (16, 1), (32, 1), (64, 1), (16, 2), (16, 4), (32, 2), (32, 4), (64, 2), (32, 8)],
)
@pytest.mark.parametrize("shared", [False, True])
def test_explicit_memory_width(bits, count, shared):
    if count == 8 and not env.has_cuda_compute(10):
        pytest.skip("v8 requires Blackwell")
    if shared and count == 8:
        pytest.skip("PTX shared v8 is unavailable")
    dtype = f"uint{bits}"
    suffix = (f"v{count}." if count > 1 else "") + f"u{bits}"
    space = "shared" if shared else "global"
    sl = TileLayout(S[(3, count) : (count * 2, 1)])
    rl = TileLayout(S[(3, count) : (count * 3, 3)])

    @T.function
    def kernel(A: T.Tensor((3, count), dtype), Output: T.Tensor((3, count), dtype)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        r = T.alloc_tensor(ty_args=[T.Tensor((3, count), dtype, scope="local", layout=rl)])
        if T.constexpr(shared):
            scratch = T.alloc_tensor(
                ty_args=[T.Tensor((3, count), dtype, scope="shared", layout=sl)]
            )
            for i, j in T.grid(3, count):
                scratch[i, j] = A[i, j]
            T.cuda.tile.apply(T.ptx[f"ld.{space}.{suffix}"], r, scratch)
            T.cuda.tile.apply(T.ptx[f"st.{space}.{suffix}"], scratch, r)
            for i, j in T.grid(3, count):
                Output[i, j] = scratch[i, j]
        else:
            T.cuda.tile.apply(T.ptx[f"ld.{space}.{suffix}"], r, A)
            T.cuda.tile.apply(T.ptx[f"st.{space}.{suffix}"], Output, r)

    f = build(kernel)
    x = np.arange(3 * count, dtype=dtype).reshape(3, count)
    out = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    f(tvm.runtime.tensor(x, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), x)


@pytest.mark.skipif(not env.has_cuda(), reason="CUDA required")
@pytest.mark.parametrize("enabled", [False, True])
def test_predicated_vector_load(enabled):
    @T.function
    def kernel(A: T.Tensor((4,), "uint32"), Output: T.Tensor((4,), "uint32"), valid: T.int32):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        r = T.alloc_local((4,), "uint32", layout="default")
        for i in range(4):
            r[i] = T.uint32(7)
        T.cuda.tile.apply(T.ptx["ld.global.cg.v4.u32"], r, A, pred=valid != 0, preserve_dst=True)
        T.cuda.tile.apply(T.ptx["st.global.v4.u32"], Output, r)

    f = build(kernel)
    x = np.arange(4, dtype="uint32")
    out = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    f(tvm.runtime.tensor(x, tvm.cuda()), out, int(enabled))
    np.testing.assert_array_equal(out.numpy(), x if enabled else np.full_like(x, 7))


@pytest.mark.skipif(not env.has_cuda_compute(9), reason="Hopper clusters required")
def test_cluster_group_axis():
    from tvm.backend.cuda.axis import cbx

    layout = TileLayout(S[(2, 1, 32, 4) : (1 @ cbx, 1 @ warpid, 1 @ laneid, 1)])

    @T.function
    def kernel(A: T.Tensor((2, 1, 32, 4), "uint32"), Output: T.Tensor((2, 1, 32, 4), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=32, cluster=2))
        r = T.alloc_tensor(
            ty_args=[T.Tensor((2, 1, 32, 4), "uint32", scope="local", layout=layout)]
        )
        T.cuda.tile.apply(T.ptx["ld.global.v4.u32"], r, A, scope="cluster")
        T.cuda.tile.apply(T.ptx["st.global.v4.u32"], Output, r, scope="cluster")

    f = build(kernel)
    x = np.arange(256, dtype="uint32").reshape(2, 1, 32, 4)
    out = tvm.runtime.tensor(np.zeros_like(x), tvm.cuda())
    f(tvm.runtime.tensor(x, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), x)


@pytest.mark.skipif(not env.has_cuda(), reason="CUDA required")
def test_k_tiled_shared_strides():
    """Explicit copies retain the shared K-tile jump instead of flattening it."""
    shared = TileLayout(S[(4, 32, 8, 4) : (128, 4, 512, 1)])
    registers = TileLayout(S[(4, 32, 8, 4) : (1 @ warpid, 1 @ laneid, 4, 1)])

    @T.function
    def kernel(A: T.Tensor((4, 32, 8, 4), "uint32"), Output: T.Tensor((4096,), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        r = T.alloc_tensor(
            ty_args=[T.Tensor((4, 32, 8, 4), "uint32", scope="local", layout=registers)]
        )
        s = T.alloc_tensor(
            ty_args=[T.Tensor((4, 32, 8, 4), "uint32", scope="shared", layout=shared)]
        )
        T.cuda.tile.apply(T.ptx["ld.global.v4.u32"], r, A, scope="cta")
        T.cuda.tile.apply(T.ptx["st.shared.v4.u32"], s, r, scope="cta")
        T.cuda.cta_sync()
        raw = s.view(4096, layout="default")
        for i in range(32):
            Output[i * 128 + tid] = raw[i * 128 + tid]

    f = build(kernel)
    x = np.arange(4096, dtype="uint32").reshape(4, 32, 8, 4)
    out = tvm.runtime.tensor(np.zeros(4096, "uint32"), tvm.cuda())
    f(tvm.runtime.tensor(x, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), x.transpose(2, 0, 1, 3).reshape(-1))
