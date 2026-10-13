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
"""Tests for FMA op dispatch, layout=None local dispatch, scalar broadcast,
and rounding mode support."""

import re
import struct

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.backend.cuda.layout import wg_local_layout
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.layout import S, TileLayout


def _splat_f32(value):
    """Prepare the two equal f32 fields of a native b64 packed operand."""
    if isinstance(value, tvm.ir.prim.FloatImm):
        bits = struct.unpack("I", struct.pack("f", value.value))[0]
        return T.uint64(bits) | (T.uint64(bits) << T.uint64(32))
    bits = T.cast(T.reinterpret(value, ty="uint32"), "uint64")
    return bits | (bits << T.uint64(32))


def _get_sm_version():
    target = tvm.target.Target("cuda")
    arch = target.arch if hasattr(target, "arch") else ""
    if not arch.startswith("sm_"):
        return 0
    digits = "".join(ch for ch in arch.split("_", 1)[1] if ch.isdigit())
    return int(digits) if digits else 0


# ---------------------------------------------------------------------------
# FMA op: scalar scale + scalar bias
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_fma_scalar_scalar():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed fma requires sm_100+, got sm_{sm}")

    N = 128
    dtype = "float32"
    target = tvm.target.Target("cuda")

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
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            mod(A)
            expected = A_np * scale_val + bias_val
            tvm.testing.assert_allclose(expected, A.numpy(), atol=1e-3)

        tvm.testing.run_with_gpu_lock(run_and_check)


# ---------------------------------------------------------------------------
# FMA op: tensor scale + scalar bias (Horner pattern)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_fma_tensor_scale_scalar_bias():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed fma requires sm_100+, got sm_{sm}")

    N = 2
    dtype = "float32"
    target = tvm.target.Target("cuda")

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
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            expected = A_np * B_np + coeff
            tvm.testing.assert_allclose(expected, A.numpy(), atol=1e-3)

        tvm.testing.run_with_gpu_lock(run_and_check)


# ---------------------------------------------------------------------------
# Binary op with scalar broadcast (Expr scalar, e.g. TensorLoad)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_mul_scalar_broadcast():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed mul requires sm_100+, got sm_{sm}")

    N = 16
    dtype = "float32"
    target = tvm.target.Target("cuda")

    @T.function
    def test_func(
        A: T.Tensor((N,), dtype, layout=TileLayout(S[N])),
        Scale: T.Tensor((1,), dtype, layout=TileLayout(S[1])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        _bx = T.cuda.block_idx("x")
        _tx = T.cuda.thread_idx("x")
        a_local = T.alloc_tensor(
            ty_args=[T.Tensor((N,), dtype, scope="local", layout=TileLayout(S[N]))]
        )
        s_local = T.alloc_tensor(
            ty_args=[T.Tensor((1,), dtype, scope="local", layout=TileLayout(S[1]))]
        )
        T.cuda.tile.apply(T.ptx["ld.global.v4.f32"], a_local.view(4, 4), A.view(4, 4))
        T.cuda.tile.apply(T.ptx["ld.global.f32"], s_local, Scale[0:1])
        T.cuda.tile.apply(
            T.ptx["mul.rn.ftz.f32x2"],
            a_local.view("uint64").view(8, 1),
            a_local.view("uint64").view(8, 1),
            _splat_f32(s_local[0]),
        )
        T.cuda.tile.apply(T.ptx["st.global.v4.f32"], A.view(4, 4), a_local.view(4, 4))

    with target:
        A_np = np.random.rand(N).astype(dtype)
        S_np = np.array([2.5], dtype=dtype)
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm.cuda(0)
            A_dev = tvm.runtime.tensor(A_np, dev)
            S_dev = tvm.runtime.tensor(S_np, dev)
            mod(A_dev, S_dev)
            expected = A_np * S_np[0]
            tvm.testing.assert_allclose(expected, A_dev.numpy(), atol=1e-3)

        tvm.testing.run_with_gpu_lock(run_and_check)


# ---------------------------------------------------------------------------
# Binary add with rounding mode
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_add_rounding_mode():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed add with rounding requires sm_100+, got sm_{sm}")

    N = 2
    dtype = "float32"
    target = tvm.target.Target("cuda")

    round_const = float(2**23 + 2**22)

    @T.function
    def test_func(A: T.Tensor((N,), dtype, layout=TileLayout(S[N]))) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        _bx = T.cuda.block_idx("x")
        _tx = T.cuda.thread_idx("x")
        buf = T.alloc_tensor(
            ty_args=[T.Tensor((N,), dtype, scope="local", layout=TileLayout(S[N]))]
        )
        T.cuda.tile.apply(T.ptx["ld.global.v2.f32"], buf, A[0:N])
        T.cuda.tile.apply(
            T.ptx["add.rm.ftz.f32x2"],
            buf.view("uint64"),
            buf.view("uint64"),
            _splat_f32(T.float32(round_const)),
        )
        T.cuda.tile.apply(T.ptx["st.global.v2.f32"], A[0:N], buf)

    with target:
        A_np = np.array([1.3, 2.7], dtype=dtype)
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        # Check that the PTX uses the rounding mode
        src = mod.mod.imports[0].inspect_source()
        assert re.search(r"add\.rm\.ftz\.f32x2", src), (
            f"Expected packed add with rm rounding in PTX:\n{src}"
        )

        def run_and_check():
            dev = tvm.cuda(0)
            A_dev = tvm.runtime.tensor(A_np, dev)
            mod(A_dev)
            # Both exact sums lie above these integers; rm rounds down.
            expected = np.array([round_const + 1, round_const + 2], dtype="float32")
            np.testing.assert_array_equal(expected, A_dev.numpy())

        tvm.testing.run_with_gpu_lock(run_and_check)


# ---------------------------------------------------------------------------
# FMA op: layout=None local tensor (no TileLayout)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("layout", [None, "default"])
def test_fma_no_layout(layout):
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed fma requires sm_100+, got sm_{sm}")

    N = 4
    dtype = "float32"
    target = tvm.target.Target("cuda")

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
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm.cuda(0)
            A_dev = tvm.runtime.tensor(A_np, dev)
            mod(A_dev)
            expected = A_np * scale_val + bias_val
            tvm.testing.assert_allclose(expected, A_dev.numpy(), atol=1e-3)

        tvm.testing.run_with_gpu_lock(run_and_check)


# ---------------------------------------------------------------------------
# Binary sub with rounding mode (tensor-tensor)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_sub_tensor_tensor_rounding():
    sm = _get_sm_version()
    if sm < 100:
        pytest.skip(f"packed sub with rounding requires sm_100+, got sm_{sm}")

    N = 2
    dtype = "float32"
    target = tvm.target.Target("cuda")

    @T.function
    def test_func(
        A: T.Tensor((N,), dtype, layout=TileLayout(S[N])),
        B: T.Tensor((N,), dtype, layout=TileLayout(S[N])),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        _bx = T.cuda.block_idx("x")
        _tx = T.cuda.thread_idx("x")
        a_buf = T.alloc_tensor(
            ty_args=[T.Tensor((N,), dtype, scope="local", layout=TileLayout(S[N]))]
        )
        b_buf = T.alloc_tensor(
            ty_args=[T.Tensor((N,), dtype, scope="local", layout=TileLayout(S[N]))]
        )
        T.cuda.tile.apply(T.ptx["ld.global.v2.f32"], a_buf, A[0:N])
        T.cuda.tile.apply(T.ptx["ld.global.v2.f32"], b_buf, B[0:N])
        T.cuda.tile.apply(
            T.ptx["sub.rn.ftz.f32x2"],
            a_buf.view("uint64"),
            a_buf.view("uint64"),
            b_buf.view("uint64"),
        )
        T.cuda.tile.apply(T.ptx["st.global.v2.f32"], A[0:N], a_buf)

    with target:
        A_np = np.array([3.14, 2.71], dtype=dtype)
        B_np = np.array([1.41, 0.57], dtype=dtype)
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        assert re.search(r"sub\.rn\.ftz\.f32x2", src), (
            f"Expected packed sub with rn rounding in PTX:\n{src}"
        )

        def run_and_check():
            dev = tvm.cuda(0)
            A_dev = tvm.runtime.tensor(A_np, dev)
            B_dev = tvm.runtime.tensor(B_np, dev)
            mod(A_dev, B_dev)
            expected = A_np - B_np
            tvm.testing.assert_allclose(expected, A_dev.numpy(), atol=1e-6)

        tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_fma_warpgroup_wg_local_layout():
    rows, cols = 128, 8
    dtype = "float32"
    scale_val = 1.5
    bias_val = -0.25
    target = tvm.target.Target("cuda")

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
            ty_args=[T.Tensor((rows, cols), dtype, scope="local", layout=wg_local_layout(cols))]
        )
        reg_row = reg.local(cols)
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
        reg_row_1 = reg.local(cols)
        for i in T.serial(cols):
            B[tid, i] = reg_row_1[i]

    with target:
        np.random.seed(0)
        A_np = np.random.rand(rows, cols).astype(dtype)
        B_np = np.zeros((rows, cols), dtype=dtype)
        mod = tvm.IRModule({"main": test_func})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        def run_and_check():
            dev = tvm.cuda(0)
            A_dev = tvm.runtime.tensor(A_np, dev)
            B_dev = tvm.runtime.tensor(B_np, dev)
            mod(A_dev, B_dev)
            expected = A_np * scale_val + bias_val
            tvm.testing.assert_allclose(expected, B_dev.numpy(), atol=1e-5)

        tvm.testing.run_with_gpu_lock(run_and_check)


# -----------------------------------------------------------------------------
# Dispatch codegen check (no GPU runtime — explicit target arch).
# Complements ``test_fma_warpgroup_wg_local_emits_packed_f32x2`` (which uses
# the host-detected ``Target("cuda")`` and skips when arch < sm_100).
# -----------------------------------------------------------------------------
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

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.IRModule({"main": k})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
    assert re.search(r"fma\.[a-z]+\.ftz\.f32x2", src), (
        f"expected packed fma_f32x2; got:\n{src[:2000]}"
    )


if __name__ == "__main__":
    tvm.testing.main()
