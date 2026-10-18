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

"""Native mov instructions for fills and register regions."""

import math

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.cuda.axis import laneid, warpid, wid_in_wg
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env
from tvm_tirx.tirx.layout import S, TileLayout


def _run(kernel, arrays, expected, output=0, atol=1e-3, rtol=1e-7, target="cuda"):
    with tvm_tirx.target.Target(target):
        module = tvm_tirx.tirx.compile(kernel, target=target, pipeline="tirx")

    def check():
        args = [tvm_tirx.runtime.tensor(a, tvm_tirx.cuda()) for a in arrays]
        module(*args)
        tvm_tirx.testing.assert_allclose(args[output].numpy(), expected, atol=atol, rtol=rtol)

    tvm_tirx.testing.run_with_gpu_lock(check)
    return module


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("thread_scope", ["warp", "warpgroup"])
def test_unary_op_shared_subcta_scope(thread_scope):
    warps = 4 if thread_scope == "warpgroup" else 1
    shape = (warps * 32, 8)
    group = (warps, 32, 8, 1) if thread_scope == "warpgroup" else (32, 8, 1)
    layout = (
        TileLayout(S[(4, 32, 8, 1) : (1 @ wid_in_wg, 1 @ laneid, 1, 1)])
        if thread_scope == "warpgroup"
        else TileLayout(S[(32, 8, 1) : (1 @ laneid, 1, 1)])
    )
    size = math.prod(shape)

    @T.function
    def kernel(A: T.Tensor(shape, "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=256))
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        wg = T.cuda.warpgroup_id()
        AS = T.alloc_shared(shape, "float16", layout="default")
        sf, af = AS.view(size), A.view(size)
        for i in T.serial(T.ceildiv(size, 256)):
            q = i * 256 + tid
            if q < size:
                sf[q] = af[q]
        T.cuda.cta_sync()
        r = T.alloc_local(group, "uint16", layout=layout)
        if T.constexpr(thread_scope == "warp"):
            if warp == 5:
                T.cuda.tile.apply(T.ptx["mov.b16"], r, T.uint16(0), scope="warp")
                T.cuda.tile.apply(
                    T.ptx["st.shared.b16"], AS.view("uint16").view(*group), r, scope="warp"
                )
        else:
            if wg == 1:
                T.cuda.tile.apply(T.ptx["mov.b16"], r, T.uint16(0), scope="warpgroup")
                T.cuda.tile.apply(
                    T.ptx["st.shared.b16"], AS.view("uint16").view(*group), r, scope="warpgroup"
                )
        T.cuda.cta_sync()
        for i in T.serial(T.ceildiv(size, 256)):
            q = i * 256 + tid
            if q < size:
                af[q] = sf[q]

    values = np.random.RandomState(0).rand(*shape).astype("float16")
    _run(kernel, [values], np.zeros_like(values))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("shape", [(128, 8), (128, 4, 16), (128, 5, 5)])
@pytest.mark.parametrize("op_type", ["fill"])
@pytest.mark.parametrize("thread_scope", ["thread", "cta"])
@pytest.mark.parametrize("storage_scope", ["local", "shared"])
def test_unary_op_vectorized(shape, op_type, thread_scope, storage_scope):
    n = math.prod(shape[1:])
    bits = int(np.float16(7.89).view("uint16"))
    group_layout = TileLayout(S[(4, 32, n, 1) : (1 @ warpid, 1 @ laneid, 1, 1)])

    @T.function
    def kernel(A: T.Tensor(shape, "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        r = T.alloc_local(shape[1:], "float16", layout="default")
        if T.constexpr(thread_scope == "cta"):
            values = r.view(4, 32, n, 1, layout=group_layout).view("uint16")
            T.cuda.tile.apply(T.ptx["mov.b16"], values, T.uint16(bits), scope="cta")
        else:
            T.cuda.tile.apply(T.ptx["mov.b16"], r.view("uint16").view(n, 1), T.uint16(bits))
        rf, af = r.view(n), A.view(128, n)
        if T.constexpr(storage_scope == "shared"):
            sm = T.alloc_shared(shape, "float16", layout="default")
            sf = sm.view(128, n)
            for i in range(n):
                sf[tid, i] = rf[i]
            T.cuda.cta_sync()
            for i in range(n):
                af[tid, i] = sf[tid, i]
        else:
            for i in range(n):
                af[tid, i] = rf[i]

    x = np.random.RandomState(0).rand(*shape).astype("float16")
    _run(kernel, [x], np.full(shape, np.float16(7.89)), atol=0, rtol=0)


def test_mov_handles_scalar_regions():
    @T.function
    def kernel(B: T.Tensor((1,), "float32")) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        src = T.alloc_local((1,), "float32", layout="default")
        dst = T.alloc_local((4,), "float32", layout="default")
        src[0] = T.cast(7, ty="float32")
        T.cuda.tile.apply(T.ptx["mov.b32"], dst[2:3], src[:])
        B[0] = dst[2]

    target = tvm_tirx.target.Target("cuda")
    with target:
        mod = tvm_tirx.IRModule({"main": kernel})
        ex = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")
        src = ex.mod.imports[0].inspect_source()

    assert "mov.b32" in src
    out = tvm_tirx.runtime.tensor(np.zeros(1, "float32"), tvm_tirx.cuda())
    ex(out)
    np.testing.assert_array_equal(out.numpy(), [7])


if __name__ == "__main__":
    tvm_tirx.testing.main()
