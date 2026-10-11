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
# ruff: noqa: F821
"""CUDA mutable cell values and table-declared writable operands."""

import re

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.testing import env


@pytest.mark.skipif(not tvm.cuda().exist, reason="CUDA device is unavailable")
def test_mutable_cell_cuda_values_and_ptx_outputs():
    @T.function
    def kernel(out: T.Tensor((6,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        value: T.int32 = 1
        captured: T.let = value
        value = 9
        out[0] = captured
        T.ptx.mov.b32(value, T.int32(7))
        T.ptx.add.s32(value, value, T.int32(1))
        out[1] = value
        flag: T.bool = True
        if flag:
            value += 3
        out[4] = value
        vector = T.alloc_cell("int32x2", T.broadcast(2, 2), scope="local")
        old: T.let = vector
        vector = vector + T.broadcast(5, 2)
        out[T.ramp(2, 1, 2)] = old + vector
        for i in range(2):
            fresh: T.int32
            fresh += i
            out[5] = fresh

    target = tvm.target.Target({"kind": "cuda", "arch": env.cuda_arch() or "sm_90"})
    module = tvm.compile(kernel, target=target, tir_pipeline="tirx")
    source = module.mod.imports[0].inspect_source("cuda")
    assert re.search(r"int value\s*=\s*1;", source)
    assert not re.search(r"value\s*\[", source)

    def run():
        output = tvm.runtime.tensor(np.zeros(6, dtype="int32"), device=tvm.cuda(0))
        module(output)
        np.testing.assert_array_equal(output.numpy(), [1, 8, 9, 9, 11, 1])

    tvm.testing.run_with_gpu_lock(run)


@pytest.mark.skipif(not tvm.cuda().exist, reason="CUDA device is unavailable")
def test_wait_until_cell_condition_remains_live():
    @T.function
    def kernel(state: T.Tensor((1,), "int32"), out: T.Tensor((1,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        observed: T.int32
        T.cuda.wait_until(observed, state.ptr_to([0]), predicate=lambda v: v == 42, scope="gpu")
        out[0] = observed

    target = tvm.target.Target({"kind": "cuda", "arch": env.cuda_arch() or "sm_90"})
    module = tvm.compile(kernel, target=target, tir_pipeline="tirx")
    source = module.mod.imports[0].inspect_source("cuda")
    invocation = next(
        line
        for line in source.splitlines()
        if "tvm_builtin_cuda_wait_until" in line and "(observed," in line
    )
    assert "(observed == 42)" in invocation

    def run():
        state = tvm.runtime.tensor(np.array([42], dtype="int32"), device=tvm.cuda(0))
        output = tvm.runtime.tensor(np.zeros(1, dtype="int32"), device=tvm.cuda(0))
        module(state, output)
        np.testing.assert_array_equal(output.numpy(), [42])

    tvm.testing.run_with_gpu_lock(run)


@pytest.mark.skipif(not tvm.cuda().exist, reason="CUDA device is unavailable")
@pytest.mark.parametrize(
    "dtype,value",
    [
        ("uint128", 0),
        ("uint128", (1 << 64) + 7),
        ("uint128", (1 << 128) - 1),
        ("int128", 0),
        ("int128", -1),
        ("int128", -(1 << 127)),
        ("int128", (1 << 127) - 1),
    ],
)
def test_mutable_cell_cuda_wide_integer_initialization(dtype, value):
    @T.function
    def kernel(out: T.Tensor((4,), "uint64")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        cell = T.alloc_cell(dtype, value)
        captured: T.let = cell
        cell = T.cast(1, dtype)
        out[0] = T.cast(captured, "uint64")
        out[1] = T.cast(T.shift_right(T.cast(captured, "uint128"), 64), "uint64")
        out[2] = T.cast(cell, "uint64")
        out[3] = T.cast(T.shift_right(T.cast(cell, "uint128"), 64), "uint64")

    target = tvm.target.Target({"kind": "cuda", "arch": env.cuda_arch() or "sm_90"})
    module = tvm.compile(kernel, target=target, tir_pipeline="tirx")

    def run():
        output = tvm.runtime.tensor(np.zeros(4, dtype="uint64"), device=tvm.cuda(0))
        module(output)
        mask = (1 << 64) - 1
        expected = np.array([value & mask, (value >> 64) & mask, 1, 0], dtype="uint64")
        np.testing.assert_array_equal(output.numpy(), expected)

    tvm.testing.run_with_gpu_lock(run)
