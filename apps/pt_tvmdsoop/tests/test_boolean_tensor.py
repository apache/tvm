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
"""Test script for boolean tensor support"""
import tempfile

import torch

import tvm
import tvm.testing
from tvm.contrib.torch import as_torch, optimize_torch
from tvm.script import tir as T


def negate(x):
    return x.logical_not()


def sum_up_tensor(x):
    return x.size(dim=0) - torch.sum(x.int())


def tensor_boolean_operation(x):
    arr1 = (x + 0.3).floor().bool()
    arr2 = (~((x + 0.7).int().bool())).bool()
    ret = ((arr1 & arr2).byte() + 0.5).half()
    return ~(ret.bool())


def test_bool_tensor_negate():
    input = torch.ones(1, dtype=torch.bool)
    optimized_negate = optimize_torch(
        negate,
        input,
    )
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        torch.save(optimized_negate, tmp.name)
        loaded_mod = torch.load(tmp.name)
        output = loaded_mod(negate(input))
    tvm.testing.assert_allclose(input.numpy(), output.numpy(), atol=1e-5, rtol=1e-5)


def test_sum_up_tensor():
    x = torch.randint(0, 2, (16,))
    y = x.bool()
    optimized_func = optimize_torch(
        sum_up_tensor,
        (y,),
    )
    ret1 = (x[x == 0]).size(dim=0)
    ret2 = optimized_func(y).numpy()
    tvm.testing.assert_allclose(ret1, ret2, atol=1e-5, rtol=1e-5)


def test_tensor_boolean_operation():
    input = torch.rand(200)
    model = optimize_torch(
        tensor_boolean_operation,
        input,
    )
    ret1 = tensor_boolean_operation(input)
    ret2 = model(input)
    tvm.testing.assert_allclose(ret1, ret2, atol=1e-5, rtol=1e-5)


@as_torch
@T.prim_func
def negate_tvmscript(
    X: T.Buffer((8, 8), "bool"),
    Y: T.Buffer((8, 8), "float32"),
    Z: T.Buffer((8, 8), "bool"),
    U: T.Buffer((8, 8), "float32"),
) -> None:
    for i, j in T.grid(8, 8):
        with T.block():
            if Y[i, j] > 0.0:
                Z[i, j] = X[i, j]
                U[i, j] = Y[i, j]
            else:
                Z[i, j] = not X[i, j]
                U[i, j] = 0.0 - Y[i, j]


def negate_vanila(x, y):
    z = torch.zeros(8, 8).bool()
    for i in range(8):
        for j in range(8):
            if y[i, j] > 0:
                z[i, j] = x[i, j]
            else:
                z[i, j] = ~x[i, j]
    return z


def test_tvmscript_torch_decorator():
    q1 = (torch.rand(8, 8) + 0.5).int().bool()
    q2 = torch.rand(8, 8) - 0.5
    q3 = torch.zeros(8, 8).bool()
    q4 = torch.zeros(8, 8)

    std1 = negate_vanila(q1, q2)
    std2 = torch.abs(q2)

    negate_tvmscript(q1, q2, q3, q4)

    tvm.testing.assert_allclose(std1.numpy(), q3.numpy(), atol=1e-5, rtol=1e-5)
    tvm.testing.assert_allclose(std2.numpy(), q4.numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_tvmscript_torch_decorator()
    test_bool_tensor_negate()
    test_sum_up_tensor()
    test_tensor_boolean_operation()
