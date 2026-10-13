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
"""Synchronous global/shared copies with explicit ld/st widths.

Global/shared transfers use explicit load/store loops at the call site.
Width/stride/swizzle regressions exercise explicit PTX tile calls.
"""

import math

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.backend.cuda.axis import tx
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.layout import ComposeLayout, S, TileLayout


def _build_kernel(scope, n_threads, shape, dtype):
    s_layout = TileLayout(S[shape])
    total = math.prod(shape)

    @T.function
    def kernel(A: T.Tensor(shape, dtype), B: T.Tensor(shape, dtype)) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=n_threads))
        tid = T.cuda.thread_idx("x")
        A_smem = T.alloc_shared(shape, dtype, layout=s_layout)
        for step in range(total // n_threads):
            index = step * n_threads + tid
            A_smem[index // shape[1], index % shape[1]] = A[index // shape[1], index % shape[1]]
        T.cuda.cta_sync()
        for step in range(total // n_threads):
            index = step * n_threads + tid
            B[index // shape[1], index % shape[1]] = A_smem[index // shape[1], index % shape[1]]

    return kernel


# (scope, n_threads, shape) — shape chosen so total / T / vec_len > 1 with at
# least one outer round, and total is divisible by T*vec_len.
TASKS = [
    ("warp", 32, (32, 32)),  # 1024 total, T=32, vec 8 → outer 4
    ("warp", 32, (32, 64)),  # 2048 total, outer 8
    ("warpgroup", 128, (128, 32)),  # 4096, outer 4
    ("warpgroup", 128, (128, 64)),  # 8192, outer 8
    ("warpgroup", 128, (256, 16)),  # 4096, outer 4
    ("cta", 256, (256, 32)),  # 8192, T=256, vec 8 → outer 4
    ("cta", 256, (512, 16)),  # 8192, outer 4
]


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize(
    "scope,n_threads,shape",
    [pytest.param(*t, id=f"{t[0]}-{t[1]}-{'x'.join(map(str, t[2]))}") for t in TASKS],
)
@pytest.mark.parametrize("dtype", ["float16", "float32", "uint8"])
def test_gmem_smem_roundtrip(scope, n_threads, shape, dtype):
    kernel = _build_kernel(scope, n_threads, shape, dtype)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        compiled = tvm.tirx.compile(mod, target=target, pipeline="tirx")

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    A_np = tvm.testing.generate_random_array(dtype, shape)
    B_np = np.zeros(shape, dtype=np_dtype)

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        compiled(A, B)
        np.testing.assert_array_equal(B.numpy(), A_np)

    tvm.testing.run_with_gpu_lock(run_and_check)


# ----------------------------------------------------------------------------
# Synchronous global/shared transfers through explicit ld/st instructions.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task",
    [
        # A[0:128, 0:32] -> A_smem[0:128, 0:32] -> B[0:128, 0:32]
        (
            (128, 32),
            (128, 32),
            ((0, 128), (0, 32)),
            32,
            TileLayout(S[128, 32]),
            TileLayout(S[128, 32]),
            TileLayout(S[128, 32]),
        ),
        # A[32:64, 32:64] -> A_smem[0:32, 0:32] -> B[32:64, 32:64]
        (
            (64, 64),
            (32, 32),
            ((32, 64), (32, 64)),
            32,
            TileLayout(S[64, 64]),
            TileLayout(S[64, 64]),
            TileLayout(S[32, 32]),
        ),
        # A[0:1, 0:32, 0:32] -> A_smem[0:32, 0:32] -> B[0:1, 0:32, 0:32]
        (
            (4, 32, 32),
            (32, 32),
            ((0, 1), (0, 32), (0, 32)),
            32,
            TileLayout(S[4, 32, 32]),
            TileLayout(S[4, 32, 32]),
            TileLayout(S[32, 32]),
        ),
        # A[0:8, 0:8] -> A_smem[0:8, 0:8] -> B[0:8, 0:8]
        (
            (16, 16),
            (8, 8),
            ((0, 8), (0, 8)),
            32,
            TileLayout(S[16, 16]),
            TileLayout(S[16, 16]),
            TileLayout(S[8, 8]),
        ),
        # A[32:96, 256:512] -> A_smem[0:32, 0:256] -> B[32:96, 256:512] (swizzled)
        (
            (96, 512),
            (32, 256),
            ((16, 48), (256, 512)),
            32,
            TileLayout(S[96, 512]),
            TileLayout(S[96, 512]),
            ComposeLayout(3, 3, 3, TileLayout(S[8, 64]))
            .tile_to((16, 128), (8, 64))
            .tile_to((32, 256), (16, 128)),
        ),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize(
    "dtype", ["int8", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32"]
)
@pytest.mark.parametrize("scope", ["cta", "thread"])
def test_copy_g2s_s2g(task, dtype, scope):
    g_shape, s_shape, g_region, thread_cnt, layoutA, layoutB, layoutS = task

    r_gmem = tuple(slice(g_region[i][0], g_region[i][1]) for i in range(len(g_shape)))

    if scope == "thread":
        thread_cnt = 1

    total = math.prod(s_shape)

    @T.function
    def copy_sync(
        A: T.Tensor(g_shape, dtype, layout=layoutA), B: T.Tensor(g_shape, dtype, layout=layoutB)
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=thread_cnt))
        tid = T.cuda.thread_idx("x")
        A_smem = T.alloc_shared(s_shape, dtype, layout=layoutS)
        source = A.sub[r_gmem].view(*s_shape)
        destination = B.sub[r_gmem].view(*s_shape)
        for step in range(T.ceildiv(total, thread_cnt)):
            index = step * thread_cnt + tid
            if index < total:
                A_smem[index // s_shape[1], index % s_shape[1]] = source[
                    index // s_shape[1], index % s_shape[1]
                ]
        T.cuda.cta_sync()
        for step in range(T.ceildiv(total, thread_cnt)):
            index = step * thread_cnt + tid
            if index < total:
                destination[index // s_shape[1], index % s_shape[1]] = A_smem[
                    index // s_shape[1], index % s_shape[1]
                ]

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_sync})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        B_ref = B_np.copy()
        B_ref[r_gmem] = A_np[r_gmem]

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            np.testing.assert_allclose(B_ref, B.numpy())

        tvm.testing.run_with_gpu_lock(run_and_check)


def test_swizzled_smem_emit_must_be_swizzle_aware():
    shape = (128, 32)
    gl = TileLayout(S[shape])
    sl = ComposeLayout(3, 3, 3, TileLayout(S[shape]))
    kernel = _explicit_staging_kernel(shape, shape, gl, sl, 0, "uint32", 4)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    with target:
        ex = tvm.tirx.compile(kernel, target=target)
    source = ex.mod.imports[0].inspect_source()
    assert "^" in source
    assert "st.shared.v4.b32" in source


# ----------------------------------------------------------------------------
# Explicit per-thread memory instructions over the original physical layouts.
# ----------------------------------------------------------------------------


def _explicit_staging_kernel(g_shape, s_shape, g_layout, s_layout, start, word_dtype, words):
    """A fixed per-row instruction loop, with a scalar readback of shared memory."""
    rows, cols = s_shape
    lanes = min(rows, 32)
    reg_layout = TileLayout(S[s_shape : (1 @ tx, 1)])
    bits = tvm.DataType(word_dtype).bits
    suffix = f"v{words}.b{bits}" if words > 1 else f"b{bits}"
    load, store = "ld.global." + suffix, "st.shared." + suffix
    # The raw argument includes gaps in the original global layout.
    storage_size = int(g_layout.span())

    @T.function
    def kernel(A: T.Tensor((storage_size,), "float16"), B: T.Tensor(s_shape, "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=rows))
        tid = T.cuda.thread_idx("x")
        gmem = T.decl_tensor(A.data, ty_args=[T.Tensor(g_shape, "float16", layout=g_layout)])
        smem = T.alloc_shared(s_shape, "float16", layout=s_layout)
        reg = T.alloc_local(s_shape, "float16", layout=reg_layout)
        src = (
            gmem.sub[:, start : start + cols]
            .view(word_dtype)
            .rearrange("(w t) (r v) -> w t r v", t=lanes, v=words)
        )
        dst = smem.view(word_dtype).rearrange("(w t) (r v) -> w t r v", t=lanes, v=words)
        values = reg.view(word_dtype).rearrange("(w t) (r v) -> w t r v", t=lanes, v=words)
        T.cuda.tile.apply(T.ptx[load], values, src, scope="cta")
        T.cuda.tile.apply(T.ptx[store], dst, values, scope="cta")
        T.cuda.cta_sync()
        for col in range(cols):
            B[tid, col] = smem[tid, col]

    return kernel


def _check_explicit_staging(g_shape, s_shape, g_layout, s_layout, start, word_dtype, words):
    kernel = _explicit_staging_kernel(
        g_shape, s_shape, g_layout, s_layout, start, word_dtype, words
    )
    target = tvm.target.Target("cuda")
    with target:
        ex = tvm.tirx.compile(kernel, target=target)
    size = int(g_layout.span())
    source = (np.arange(size, dtype="float32") % 113 + 0.5).astype("float16")
    expected = np.empty(s_shape, "float16")
    for row in range(s_shape[0]):
        for col in range(s_shape[1]):
            offset = int(g_layout.apply(row, col + start, shape=g_shape)["m"])
            expected[row, col] = source[offset]
    out = tvm.runtime.tensor(np.zeros(s_shape, "float16"), tvm.cuda())
    ex(tvm.runtime.tensor(source, tvm.cuda()), out)
    np.testing.assert_array_equal(out.numpy(), expected)
    return ex


def _reject_explicit_staging(g_shape, s_shape, g_layout, s_layout, start, word_dtype, words):
    kernel = _explicit_staging_kernel(
        g_shape, s_shape, g_layout, s_layout, start, word_dtype, words
    )
    target = tvm.target.Target("cuda")
    with target, pytest.raises(RuntimeError, match="consecutive elements|byte.aligned"):
        tvm.tirx.transform.TileDispatch()(tvm.IRModule({"main": kernel}))


@pytest.mark.gpu
@pytest.mark.parametrize("per_element,expected_max_vec", [(2, 4), (1, 2), (0, 1)])
def test_swizzled_smem_requires_width_within_chunk(per_element, expected_max_vec):
    """An explicit width fits the original swizzle; a wider one is rejected."""
    shape = (32, 32)
    global_layout = TileLayout(S[shape])
    shared_layout = ComposeLayout(per_element, 3, 3, TileLayout(S[shape]))
    _check_explicit_staging(
        shape, shape, global_layout, shared_layout, 0, "uint16", expected_max_vec
    )
    # A larger instruction crosses the explicitly described contiguous chunk.
    if per_element == 0:
        _reject_explicit_staging(shape, shape, global_layout, shared_layout, 0, "uint16", 2)
    else:
        _reject_explicit_staging(shape, shape, global_layout, shared_layout, 0, "uint32", 4)


@pytest.mark.gpu
def test_strided_rows_reject_unaligned_vector():
    """The original 20-element row stride allows 64-bit, not 128-bit, copies."""
    shape = (2, 16)
    layout = TileLayout(S[(2, 16) : (20, 1)])
    _check_explicit_staging(shape, shape, layout, layout, 0, "uint32", 2)
    _reject_explicit_staging(shape, shape, layout, layout, 0, "uint32", 4)


@pytest.mark.gpu
def test_region_offset_requires_scalar_instruction():
    """The same columns [3,19) use scalar b16; a vector cannot align at col 3."""
    g_shape, s_shape = (4, 32), (4, 16)
    gl, sl = TileLayout(S[g_shape]), TileLayout(S[s_shape])
    _check_explicit_staging(g_shape, s_shape, gl, sl, 3, "uint16", 1)
    _reject_explicit_staging(g_shape, s_shape, gl, sl, 3, "uint16", 2)


@pytest.mark.gpu
def test_layout_permute_copy_preserves_smem_strides():
    """The original K-tiled shared layout keeps its 1024-element K-group jump."""
    shape = (128, 64)
    gl = TileLayout(S[shape])
    sl = TileLayout(S[(128, 8, 8) : (8, 1024, 1)])
    assert int(sl.apply(0, 8, shape=shape)["m"]) == 1024
    assert int(gl.apply(0, 8, shape=shape)["m"]) == 8
    ex = _check_explicit_staging(shape, shape, gl, sl, 0, "uint32", 4)
    source = ex.mod.imports[0].inspect_source()
    assert "ld.global.v4.b32" in source
    assert "st.shared.v4.b32" in source


if __name__ == "__main__":
    tvm.testing.main()
