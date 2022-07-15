#!/usr/bin/env python

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
"""Test script for tvm torch module"""
import numpy as np

import torch
import torch.nn

import tvm
from tvm.meta_schedule.tune import TuneConfig
import tvm.testing
from tvm.contrib.torch import as_torch
from tvm.script import tir as T


@as_torch
def matmul(M: int, N: int, K: int, dtype: str):
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [M, K], dtype=dtype)
        B = T.match_buffer(b, [N, K], dtype=dtype)
        C = T.match_buffer(c, [M, N], dtype=dtype)
        for i, j, k in T.grid(M, N, K):
            with T.block():
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    return main


@as_torch
@T.prim_func
def matmul_original(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i, j in T.grid(32, 32):
        with T.block("init"):
            vi, vj = T.axis.remap("SS", [i, j])
            for ii, jj in T.grid(4, 4):
                C[vi * 4 + ii, vj * 4 + jj] = T.float32(0)

        for k in range(0, 32):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                for ii, jj, kk in T.grid(4, 4, 4):
                    C[vi * 4 + ii, vj * 4 + jj] = (
                        C[vi * 4 + ii, vj * 4 + jj]
                        + A[vi * 4 + ii, vk * 4 + kk] * B[vj * 4 + jj, vk * 4 + kk]
                    )


@as_torch
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        # We exchange data between function by handles, which are similar to pointer.
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Create buffer from handles.
        A = T.match_buffer(a, (8,), dtype="float32")
        B = T.match_buffer(b, (8,), dtype="float32")
        for i in range(8):
            # A block is an abstraction for computation.
            with T.block("B"):
                # Define a spatial block iterator and bind it to value i.
                vi = T.axis.spatial(8, i)
                B[vi] = A[vi] + 1.0


@as_torch
@tvm.script.ir_module
class ModuleGPU:
    @T.prim_func
    def main(A: T.Buffer[8, "float32"], B: T.Buffer[8, "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i_0 in T.thread_binding(2, thread="blockIdx.x"):
            for i_2 in T.thread_binding(2, thread="threadIdx.x"):
                for i_1 in T.serial(2):
                    with T.block("B"):
                        vi = T.axis.spatial(8, i_0 * 4 + i_1 * 2 + i_2)
                        T.reads(A[vi])
                        T.writes(B[vi])
                        B[vi] = A[vi] + T.float32(1)


@as_torch
@T.prim_func
def func_with_part_access_region(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    with T.block():
        for i, j in T.grid(128, 128):
            with T.block("s1"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                B[vi, vj] = A[vi, vj] + T.float32(1)

        for i, j in T.grid(128, 128):
            with T.block("s2"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + T.float32(1)


config = TuneConfig(
    strategy="replay_trace",
    num_trials_per_iter=128,
    max_trials_per_task=128,
    max_trials_global=128,
)


@as_torch(config)
def softmax(M: int, N: int, dtype: str):
    @T.prim_func
    def f(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [M, N], dtype=dtype)
        B = T.match_buffer(b, [M, N], dtype=dtype)
        C = T.alloc_buffer((M), dtype=dtype, scope="local")
        for i in T.thread_binding(0, M, thread="threadIdx.x"):
            with T.block("row1"):
                for j in T.parallel(N):
                    with T.block("column1"):
                        C[i] = T.max(C[i], A[i, j])
        for i in T.thread_binding(0, M, thread="blockIdx.x"):
            with T.block("row2"):
                for j in T.thread_binding(0, N, thread="threadIdx.x"):
                    with T.block("column2"):
                        B[i, j] = tvm.tir.exp(A[i, j] - C[i])
        for i in T.thread_binding(0, M, thread="blockIdx.x"):
            with T.block("row3"):
                C[i] = 0
                for j in T.parallel(N):
                    with T.block("column3"):
                        C[i] = C[i] + B[i, j]
        for i in T.thread_binding(0, M, thread="blockIdx.x"):
            with T.block("row4"):
                for j in T.thread_binding(0, N, thread="threadIdx.x"):
                    with T.block("column4"):
                        B[i, j] = B[i, j] / C[i]

    return f


@as_torch(config)
@T.prim_func
def elementwise_with_root(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    with T.block():
        for i, j in T.grid(128, 128):
            with T.block("s1"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + T.float32(1)
        for i, j in T.grid(128, 128):
            with T.block("s2"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + T.float32(1)


class MinuesOnes(torch.nn.Module):
    def __init__(self):
        super(MinuesOnes, self).__init__()
        self.engine = MyModule

    def forward(self, *input):
        self.engine.forward(*input)
        return input[-1] - 1


def test_tvmscript_torch_matmul():
    s1 = np.random.rand(128, 128).astype("float32")
    s2 = np.random.rand(128, 128).astype("float32")
    s3 = np.random.rand(128, 128).astype("float32")

    q1 = torch.from_numpy(s1)
    q2 = torch.from_numpy(s2)
    q3 = torch.from_numpy(s3)

    numpy_result = np.matmul(s1, np.transpose(s2))

    nn_module = matmul(128, 128, 128, "float32")

    nn_module(q1, q2, q3)

    tvm.testing.assert_allclose(q3.numpy(), numpy_result, atol=1e-5, rtol=1e-5)


def test_tvmscript_torch_decorator():
    q1 = torch.arange(8).type(torch.float32)
    q2 = torch.zeros((8,), dtype=torch.float32)

    MyModule(q1, q2)

    tvm.testing.assert_allclose(q2.numpy(), (q1 + 1).numpy(), atol=1e-5, rtol=1e-5)


def test_tvmscript_torch_gpu():
    cuda0 = torch.device("cuda:0")
    q1 = torch.arange(8, device=cuda0).type(torch.float32)
    q2 = torch.zeros((8,), dtype=torch.float32, device=cuda0)

    ModuleGPU(q1, q2)

    tvm.testing.assert_allclose(q2.cpu().numpy(), (q1 + 1).cpu().numpy(), atol=1e-5, rtol=1e-5)


def test_torch_with_tvmscript():
    ref_result = np.arange(8).astype("float32")

    q1 = torch.arange(8).type(torch.float32)
    q2 = torch.zeros((8,), dtype=torch.float32)

    nn_module = MinuesOnes()

    ret = nn_module.forward(q1, q2)

    tvm.testing.assert_allclose(ret.numpy(), ref_result, atol=1e-5, rtol=1e-5)


def test_tvmscript_torch_func_with_part_access_region():
    a1 = torch.rand(128, 128)
    a2 = torch.zeros(128, 128)
    a3 = torch.zeros(128, 128)

    result = a1 + 2

    func_with_part_access_region(a1, a2, a3)

    tvm.testing.assert_allclose(a3.numpy(), result.numpy(), atol=1e-5, rtol=1e-5)


def test_tvmscript_torch_softmax():
    x = torch.rand(300, 200).cuda()
    y = torch.zeros(300, 200).cuda()

    result = torch.softmax(x, axis=1).cpu().numpy()

    func = softmax(300, 200, "float32")
    func(x, y)

    tvm.testing.assert_allclose(y.cpu().numpy(), result, atol=1e-5, rtol=1e-5)


def test_tvmscript_torch_elementwise_with_root():
    a1 = torch.rand(128, 128)
    a2 = torch.zeros(128, 128)
    a3 = torch.zeros(128, 128)

    result = a1 + 2

    elementwise_with_root(a1, a2, a3)

    tvm.testing.assert_allclose(a3.numpy(), result.numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_tvmscript_torch_matmul()
    test_tvmscript_torch_decorator()
    test_tvmscript_torch_gpu()
    test_torch_with_tvmscript()
    test_tvmscript_torch_func_with_part_access_region()
    test_tvmscript_torch_softmax()
    test_tvmscript_torch_elementwise_with_root()
