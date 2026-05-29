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
"""Tests for T.cuda.cta_reduce / cta_sum / cta_max / cta_min intrinsics."""

import numpy as np
import pytest

import tvm
from tvm.script import tirx as Tx

DEV = tvm.cuda(0)
TARGET = tvm.target.Target("cuda")


def _build_and_run(func, n):
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=TARGET, tir_pipeline="tirx")
    out_np = np.zeros(n, dtype="float32")
    out = tvm.runtime.tensor(out_np, device=DEV)
    mod(out)
    return out.numpy(), mod


def test_cta_sum_4_warps():
    """CTA sum with 4 warps (128 threads): all threads get the same sum."""
    NUM_WARPS = 4
    N = NUM_WARPS * 32

    # fmt: off
    @Tx.prim_func
    def func(out_ptr: Tx.handle):
        out = Tx.match_buffer(out_ptr, (N,), "float32")
        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            warp_id = Tx.warp_id([NUM_WARPS])
            lane_id = Tx.lane_id([32])
            tid = Tx.thread_id([N])
            with Tx.cta():
                scratch = Tx.alloc_buffer((NUM_WARPS,), "float32", scope="shared")
                with Tx.thread():
                    val: Tx.f32 = Tx.float32(tid + 1)
                    val = Tx.cuda.cta_sum(val, NUM_WARPS, scratch.ptr_to([0]))
                    out[tid] = val
    # fmt: on

    result, mod = _build_and_run(func, N)
    expected = np.float32(N * (N + 1) / 2)  # sum(1..128)
    np.testing.assert_allclose(result, np.full(N, expected))
    assert "cta_reduce_sum_4" in mod.mod.imports[0].inspect_source()


def test_cta_sum_8_warps():
    """CTA sum with 8 warps (256 threads)."""
    NUM_WARPS = 8
    N = NUM_WARPS * 32

    # fmt: off
    @Tx.prim_func
    def func(out_ptr: Tx.handle):
        out = Tx.match_buffer(out_ptr, (N,), "float32")
        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            warp_id = Tx.warp_id([NUM_WARPS])
            lane_id = Tx.lane_id([32])
            tid = Tx.thread_id([N])
            with Tx.cta():
                scratch = Tx.alloc_buffer((NUM_WARPS,), "float32", scope="shared")
                with Tx.thread():
                    val: Tx.f32 = Tx.float32(tid + 1)
                    val = Tx.cuda.cta_sum(val, NUM_WARPS, scratch.ptr_to([0]))
                    out[tid] = val
    # fmt: on

    result, _ = _build_and_run(func, N)
    expected = np.float32(N * (N + 1) / 2)
    np.testing.assert_allclose(result, np.full(N, expected))


def test_cta_max_4_warps():
    """CTA max with 4 warps: all threads get the maximum value."""
    NUM_WARPS = 4
    N = NUM_WARPS * 32

    # fmt: off
    @Tx.prim_func
    def func(out_ptr: Tx.handle):
        out = Tx.match_buffer(out_ptr, (N,), "float32")
        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            warp_id = Tx.warp_id([NUM_WARPS])
            lane_id = Tx.lane_id([32])
            tid = Tx.thread_id([N])
            with Tx.cta():
                scratch = Tx.alloc_buffer((NUM_WARPS,), "float32", scope="shared")
                with Tx.thread():
                    val: Tx.f32 = Tx.float32(tid + 1)
                    val = Tx.cuda.cta_max(val, NUM_WARPS, scratch.ptr_to([0]))
                    out[tid] = val
    # fmt: on

    result, _ = _build_and_run(func, N)
    np.testing.assert_allclose(result, np.full(N, float(N)))


def test_cta_min_4_warps():
    """CTA min with 4 warps: all threads get the minimum value."""
    NUM_WARPS = 4
    N = NUM_WARPS * 32

    # fmt: off
    @Tx.prim_func
    def func(out_ptr: Tx.handle):
        out = Tx.match_buffer(out_ptr, (N,), "float32")
        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            warp_id = Tx.warp_id([NUM_WARPS])
            lane_id = Tx.lane_id([32])
            tid = Tx.thread_id([N])
            with Tx.cta():
                scratch = Tx.alloc_buffer((NUM_WARPS,), "float32", scope="shared")
                with Tx.thread():
                    val: Tx.f32 = Tx.float32(tid + 1)
                    val = Tx.cuda.cta_min(val, NUM_WARPS, scratch.ptr_to([0]))
                    out[tid] = val
    # fmt: on

    result, _ = _build_and_run(func, N)
    np.testing.assert_allclose(result, np.full(N, 1.0))


def test_cta_sum_1_warp():
    """CTA sum with 1 warp: degenerates to a pure warp reduce."""
    NUM_WARPS = 1
    N = 32

    # fmt: off
    @Tx.prim_func
    def func(out_ptr: Tx.handle):
        out = Tx.match_buffer(out_ptr, (N,), "float32")
        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            warp_id = Tx.warp_id([NUM_WARPS])
            lane_id = Tx.lane_id([32])
            tid = Tx.thread_id([N])
            with Tx.cta():
                scratch = Tx.alloc_buffer((NUM_WARPS,), "float32", scope="shared")
                with Tx.thread():
                    val: Tx.f32 = Tx.float32(tid + 1)
                    val = Tx.cuda.cta_sum(val, NUM_WARPS, scratch.ptr_to([0]))
                    out[tid] = val
    # fmt: on

    result, _ = _build_and_run(func, N)
    expected = np.float32(32 * 33 / 2)
    np.testing.assert_allclose(result, np.full(N, expected))


@pytest.mark.parametrize("num_warps", [1, 2, 4, 8, 16])
def test_cta_sum_all_warp_counts(num_warps):
    """Parametric test: cta_sum with various warp counts."""
    N = num_warps * 32

    # fmt: off
    @Tx.prim_func
    def func(out_ptr: Tx.handle):
        out = Tx.match_buffer(out_ptr, (N,), "float32")
        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            warp_id = Tx.warp_id([num_warps])
            lane_id = Tx.lane_id([32])
            tid = Tx.thread_id([N])
            with Tx.cta():
                scratch = Tx.alloc_buffer((num_warps,), "float32", scope="shared")
                with Tx.thread():
                    val: Tx.f32 = Tx.float32(tid + 1)
                    val = Tx.cuda.cta_sum(val, num_warps, scratch.ptr_to([0]))
                    out[tid] = val
    # fmt: on

    result, _ = _build_and_run(func, N)
    expected = np.float32(N * (N + 1) / 2)
    np.testing.assert_allclose(result, np.full(N, expected))
