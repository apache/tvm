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
import tvm
import torch
from tvm.contrib.torch import as_torch
from tvm.script import tir as T
import numpy as np
import torch.nn
import tvm.testing


@as_torch
def matmul(M: int, N: int, K: int, dtype: str):
    @T.prim_func
    def f(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [M, K], dtype=dtype)
        B = T.match_buffer(b, [N, K], dtype=dtype)
        C = T.match_buffer(c, [M, N], dtype=dtype)
        for i, j, k in T.grid(M, N, K):
            with T.block():
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]
    return f


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

class MinuesOnes(torch.nn.Module):
    def __init__(self):
        super(MinuesOnes, self).__init__()
        self.engine = MyModule

    def forward(self, *input):
        self.engine.forward(*input)
        return input[-1] - 1


def test_tvmscript_torch_matmul():
    s1 = np.ones((128, 128)).astype("float32")
    s2 = np.ones((128, 128)).astype("float32")
    s3 = np.zeros((128, 128)).astype("float32")
    s1[0, 0] = 0
    s2[4, 4] = 0

    q1 = torch.from_numpy(s1)
    q2 = torch.from_numpy(s2)
    q3 = torch.from_numpy(s3)

    numpy_result = np.matmul(s1, s2)

    nn_module = matmul(128, 128, 128, "float32")

    nn_module(q1, q2, q3)

    tvm.testing.assert_allclose(q3.numpy(), numpy_result, atol=1e-5, rtol=1e-5)


def test_tvmscript_torch_decorator():
    q1 = torch.arange(8).type(torch.float32)
    q2 = torch.zeros((8,), dtype=torch.float32)

    MyModule(q1, q2)

    tvm.testing.assert_allclose(q2.numpy(), (q1+1).numpy(), atol=1e-5, rtol=1e-5)
    
def test_tvmscript_torch_gpu():
    cuda0 = torch.device('cuda:0')
    q1 = torch.arange(8, device=cuda0).type(torch.float32)
    q2 = torch.zeros((8,), dtype=torch.float32, device=cuda0)

    ModuleGPU(q1, q2)

    tvm.testing.assert_allclose(q2.cpu().numpy(), (q1+1).cpu().numpy(), atol=1e-5, rtol=1e-5)


def test_torch_with_tvmscript():
    ref_result = np.arange(8).astype("float32")

    q1 = torch.arange(8).type(torch.float32)
    q2 = torch.zeros((8,), dtype=torch.float32)

    nn_module = MinuesOnes()

    ret = nn_module.forward(q1, q2)

    tvm.testing.assert_allclose(ret.numpy(), ref_result, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_tvmscript_torch_matmul()
    test_tvmscript_torch_decorator()
    test_tvmscript_torch_gpu()
    test_torch_with_tvmscript()
