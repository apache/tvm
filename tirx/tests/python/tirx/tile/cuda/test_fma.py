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
"""Native fma instructions with scalar broadcasts and explicit register views."""

import re
import struct

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.cuda.axis import Axis as CUDAAxis
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env
from tvm_tirx.tirx.layout import S, TileLayout


def _splat_f32(value):
    """Prepare the two equal f32 fields of a native b64 packed operand."""
    if isinstance(value, tvm_tirx.ir.FloatImm):
        bits = struct.unpack("I", struct.pack("f", value.value))[0]
        return T.uint64(bits) | (T.uint64(bits) << T.uint64(32))
    bits = T.cast(T.reinterpret(value, ty="uint32"), ty="uint64")
    return bits | (bits << T.uint64(32))


def _get_sm_version():
    target = tvm_tirx.target.Target("cuda")
    arch = target.arch if hasattr(target, "arch") else ""
    if not arch.startswith("sm_"):
        return 0
    digits = "".join(ch for ch in arch.split("_", 1)[1] if ch.isdigit())
    return int(digits) if digits else 0


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_fma_scalar_scalar():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed fma requires sm_100+, got sm_{sm}")

    N = 128
    dtype = "float32"
    target = tvm_tirx.target.Target("cuda")

    scale_val = 0.5
    bias_val = -1.0

    @T.function
    def test_func(A: T.Tensor((N,), dtype, layout=TileLayout(S[N]))) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(N,)))
        _bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        buf = T.alloc_tensor(
            ty_args=[T.Tensor((1,), dtype, scope="local", layout=TileLayout(S[1]))]
        )
        T.cuda.tile.apply(T.ptx["ld.global.f32"], buf, A[tx : tx + 1])
        T.cuda.tile.apply(T.ptx["fma.rn.f32"], buf, buf, T.float32(scale_val), T.float32(bias_val))
        T.cuda.tile.apply(T.ptx["st.global.f32"], A[tx : tx + 1], buf)

    with target:
        A_np = np.random.rand(N).astype(dtype)
        mod = tvm_tirx.IRModule({"main": test_func})
        mod = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            A = tvm_tirx.runtime.tensor(A_np, dev)
            mod(A)
            expected = A_np * scale_val + bias_val
            tvm_tirx.testing.assert_allclose(expected, A.numpy(), atol=1e-3)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_fma_tensor_scale_scalar_bias():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed fma requires sm_100+, got sm_{sm}")

    N = 2
    dtype = "float32"
    target = tvm_tirx.target.Target("cuda")

    coeff = 0.695

    @T.function
    def test_func(
        A: T.Tensor((N,), dtype, layout=TileLayout(S[N])),
        B: T.Tensor((N,), dtype, layout=TileLayout(S[N])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        _bx = T.cuda.block_idx("x")
        _tx = T.cuda.thread_idx("x")
        acc = T.alloc_tensor(
            ty_args=[T.Tensor((N,), dtype, scope="local", layout=TileLayout(S[N]))]
        )
        frac = T.alloc_tensor(
            ty_args=[T.Tensor((N,), dtype, scope="local", layout=TileLayout(S[N]))]
        )
        T.cuda.tile.apply(T.ptx["ld.global.v2.f32"], acc, A[0:N])
        T.cuda.tile.apply(T.ptx["ld.global.v2.f32"], frac, B[0:N])
        T.cuda.tile.apply(
            T.ptx["fma.rn.ftz.f32x2"],
            acc.view("uint64"),
            acc.view("uint64"),
            frac.view("uint64"),
            _splat_f32(T.float32(coeff)),
        )
        T.cuda.tile.apply(T.ptx["st.global.v2.f32"], A[0:N], acc)

    with target:
        A_np = np.random.rand(N).astype(dtype)
        B_np = np.random.rand(N).astype(dtype)
        mod = tvm_tirx.IRModule({"main": test_func})
        mod = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            A = tvm_tirx.runtime.tensor(A_np, dev)
            B = tvm_tirx.runtime.tensor(B_np, dev)
            mod(A, B)
            expected = A_np * B_np + coeff
            tvm_tirx.testing.assert_allclose(expected, A.numpy(), atol=1e-3)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("layout", [None, "default"])
def test_fma_no_layout(layout):
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed fma requires sm_100+, got sm_{sm}")

    N = 4
    dtype = "float32"
    target = tvm_tirx.target.Target("cuda")

    scale_val = 2.0
    bias_val = 1.0

    @T.function
    def test_func(A: T.Tensor((N,), dtype, layout=TileLayout(S[N]))) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        _bx = T.cuda.block_idx("x")
        _tx = T.cuda.thread_idx("x")
        buf = T.alloc_local([N], dtype, layout=layout)
        for i in T.serial(N):
            buf[i] = A[i]
        # Explicitly describe the existing compact local storage, including
        # the layout=None allocation; this view does not rearrange data.
        packed = buf.view(N, layout="default").view("uint64").view(2, 1)
        T.cuda.tile.apply(
            T.ptx["fma.rn.ftz.f32x2"],
            packed,
            packed,
            _splat_f32(T.float32(scale_val)),
            _splat_f32(T.float32(bias_val)),
        )
        for i in T.serial(N):
            A[i] = buf[i]

    with target:
        A_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        mod = tvm_tirx.IRModule({"main": test_func})
        mod = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            A_dev = tvm_tirx.runtime.tensor(A_np, dev)
            mod(A_dev)
            expected = A_np * scale_val + bias_val
            tvm_tirx.testing.assert_allclose(expected, A_dev.numpy(), atol=1e-3)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_fma_warpgroup_thread_row_layout():
    rows, cols = 128, 8
    dtype = "float32"
    scale_val = 1.5
    bias_val = -0.25
    target = tvm_tirx.target.Target("cuda")

    @T.function
    def test_func(
        A: T.Tensor((rows, cols), dtype, layout=TileLayout(S[rows, cols])),
        B: T.Tensor((rows, cols), dtype, layout=TileLayout(S[rows, cols])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 128))
        _bx = T.cuda.block_idx("x")
        wg_id = T.cuda.warpgroup_id()
        tid = T.cuda.thread_in_warpgroup()

        reg = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (rows, cols),
                    dtype,
                    scope="local",
                    layout=TileLayout(S[(128, cols) : (1 @ CUDAAxis.tid_in_wg, 1)]),
                )
            ]
        )
        reg_row = reg.view(cols, layout="default")
        for i in T.serial(cols):
            reg_row[i] = A[tid, i]
        pairs = reg.view("uint64").view(4, 32, cols // 2, 1)
        T.cuda.tile.apply(
            T.ptx["fma.rn.ftz.f32x2"],
            pairs,
            pairs,
            _splat_f32(T.float32(scale_val)),
            _splat_f32(T.float32(bias_val)),
            scope="warpgroup",
        )
        reg_row_1 = reg.view(cols, layout="default")
        for i in T.serial(cols):
            B[tid, i] = reg_row_1[i]

    with target:
        np.random.seed(0)
        A_np = np.random.rand(rows, cols).astype(dtype)
        B_np = np.zeros((rows, cols), dtype=dtype)
        mod = tvm_tirx.IRModule({"main": test_func})
        mod = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm_tirx.cuda(0)
            A_dev = tvm_tirx.runtime.tensor(A_np, dev)
            B_dev = tvm_tirx.runtime.tensor(B_np, dev)
            mod(A_dev, B_dev)
            expected = A_np * scale_val + bias_val
            tvm_tirx.testing.assert_allclose(expected, B_dev.numpy(), atol=1e-5)

        tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def test_fma_f32_sm100_packed_f32x2_dispatch():
    """The caller explicitly chooses packed f32x2 on the original register arrays."""
    shape = (64, 32)
    lay = TileLayout(S[shape])

    @T.function
    def k(
        A: T.Tensor(shape, "float32", layout=lay),
        B: T.Tensor(shape, "float32", layout=lay),
        C: T.Tensor(shape, "float32", layout=lay),
        D: T.Tensor(shape, "float32", layout=lay),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(64,)))
        _bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        ra = T.alloc_tensor(
            ty_args=[T.Tensor(shape[1:], "float32", scope="local", layout=TileLayout(S[shape[1:]]))]
        )
        rb = T.alloc_tensor(
            ty_args=[T.Tensor(shape[1:], "float32", scope="local", layout=TileLayout(S[shape[1:]]))]
        )
        rc = T.alloc_tensor(
            ty_args=[T.Tensor(shape[1:], "float32", scope="local", layout=TileLayout(S[shape[1:]]))]
        )
        rd = T.alloc_tensor(
            ty_args=[T.Tensor(shape[1:], "float32", scope="local", layout=TileLayout(S[shape[1:]]))]
        )
        T.cuda.tile.apply(T.ptx["ld.global.v4.f32"], ra.view(8, 4), A.sub[tx, :].view(8, 4))
        T.cuda.tile.apply(T.ptx["ld.global.v4.f32"], rb.view(8, 4), B.sub[tx, :].view(8, 4))
        T.cuda.tile.apply(T.ptx["ld.global.v4.f32"], rc.view(8, 4), C.sub[tx, :].view(8, 4))
        T.cuda.tile.apply(
            T.ptx["fma.rn.ftz.f32x2"],
            rd.view("uint64").view(16, 1),
            ra.view("uint64").view(16, 1),
            rb.view("uint64").view(16, 1),
            rc.view("uint64").view(16, 1),
        )
        T.cuda.tile.apply(T.ptx["st.global.v4.f32"], D.sub[tx, :].view(8, 4), rd.view(8, 4))

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm_tirx.IRModule({"main": k})
        mod = tvm_tirx.tirx.compile(mod, target=target, pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
    assert re.search(r"fma\.[a-z]+\.ftz\.f32x2", src), (
        f"expected packed fma_f32x2; got:\n{src[:2000]}"
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
