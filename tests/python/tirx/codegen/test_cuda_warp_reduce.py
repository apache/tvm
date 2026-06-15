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
"""Tests for T.cuda.warp_reduce / warp_sum / warp_max / warp_min intrinsics."""

import numpy as np
import pytest

import tvm
from tvm.script import tirx as T
from tvm.testing import env

DEV = tvm.cuda(0)
TARGET = tvm.target.Target("cuda")


def _build_and_run(func, n=32):
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=TARGET, tir_pipeline="tirx")
    out_np = np.zeros(n, dtype="float32")
    out = tvm.runtime.tensor(out_np, device=DEV)
    mod(out)
    return out.numpy(), mod


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_warp_sum_full():
    """Full warp sum (width=32): each lane gets the sum of all 32 values."""

    # fmt: off
    @T.prim_func
    def func(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (32,), "float32")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane = T.lane_id([32])
        val: T.f32 = T.float32(lane + 1)
        val = T.cuda.warp_sum(val)
        out[lane] = val
        # fmt: on

    result, mod = _build_and_run(func)
    expected = np.float32(32 * 33 / 2)  # sum(1..32)
    np.testing.assert_allclose(result, np.full(32, expected))
    assert "warp_reduce_sum_32" in mod.mod.imports[0].inspect_source()


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_warp_sum_partial_8():
    """Partial warp sum (width=8): 4 groups of 8 lanes, each group sums independently."""

    # fmt: off
    @T.prim_func
    def func(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (32,), "float32")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane = T.lane_id([32])
        val: T.f32 = T.float32(lane + 1)
        val = T.cuda.warp_sum(val, width=8)
        out[lane] = val
        # fmt: on

    result, _ = _build_and_run(func)
    # Group 0: lanes 0-7 → sum(1..8) = 36
    # Group 1: lanes 8-15 → sum(9..16) = 100
    # Group 2: lanes 16-23 → sum(17..24) = 164
    # Group 3: lanes 24-31 → sum(25..32) = 228
    expected = np.zeros(32, dtype="float32")
    for g in range(4):
        group_sum = sum(range(g * 8 + 1, g * 8 + 9))
        expected[g * 8 : (g + 1) * 8] = group_sum
    np.testing.assert_allclose(result, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_warp_max_partial_4():
    """Partial warp max (width=4): 8 groups of 4 lanes."""

    # fmt: off
    @T.prim_func
    def func(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (32,), "float32")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane = T.lane_id([32])
        val: T.f32 = T.float32(lane + 1)
        val = T.cuda.warp_max(val, width=4)
        out[lane] = val
        # fmt: on

    result, _ = _build_and_run(func)
    expected = np.zeros(32, dtype="float32")
    for g in range(8):
        group_max = float(g * 4 + 4)
        expected[g * 4 : (g + 1) * 4] = group_max
    np.testing.assert_allclose(result, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_warp_min_full():
    """Full warp min (width=32)."""

    # fmt: off
    @T.prim_func
    def func(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (32,), "float32")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane = T.lane_id([32])
        val: T.f32 = T.float32(lane + 1)
        val = T.cuda.warp_min(val)
        out[lane] = val
        # fmt: on

    result, _ = _build_and_run(func)
    np.testing.assert_allclose(result, np.full(32, 1.0))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_warp_sum_partial_2():
    """Smallest partial warp sum (width=2): 16 pairs of adjacent lanes."""

    # fmt: off
    @T.prim_func
    def func(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (32,), "float32")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane = T.lane_id([32])
        val: T.f32 = T.float32(lane)
        val = T.cuda.warp_sum(val, width=2)
        out[lane] = val
        # fmt: on

    result, _ = _build_and_run(func)
    # Pairs: (0,1)→1, (2,3)→5, (4,5)→9, ...
    expected = np.zeros(32, dtype="float32")
    for i in range(16):
        pair_sum = float(2 * i + 2 * i + 1)
        expected[2 * i] = pair_sum
        expected[2 * i + 1] = pair_sum
    np.testing.assert_allclose(result, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("width", [2, 4, 8, 16, 32])
def test_warp_sum_all_widths(width):
    """Parametric test: warp_sum with every valid width."""

    # fmt: off
    @T.prim_func
    def func(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (32,), "float32")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane = T.lane_id([32])
        val: T.f32 = T.float32(lane)
        val = T.cuda.warp_sum(val, width=width)
        out[lane] = val
        # fmt: on

    result, _ = _build_and_run(func)
    expected = np.zeros(32, dtype="float32")
    num_groups = 32 // width
    for g in range(num_groups):
        group_sum = sum(range(g * width, (g + 1) * width))
        expected[g * width : (g + 1) * width] = float(group_sum)
    np.testing.assert_allclose(result, expected)
