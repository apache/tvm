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
"""Test code for relu activation"""
import sys
import os
import numpy as np
import tvm
from tvm import te
from tvm import topi
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple
from tvm.contrib.nvcc import have_fp16

import pytest
import tvm.testing


m, n, dtype = tvm.testing.parameters(
    (10, 128, "float32"),
    (128, 64, "float16"),
    # Commented due to weird killed
    # (1024 * 100, 512, "float32"),
)


def test_relu(target, dev, m, n, dtype):
    A = te.placeholder((m, n), name="A", dtype=dtype)
    B = topi.nn.relu(A)

    a_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = a_np * (a_np > 0)

    if dtype == "float16" and target == "cuda" and not have_fp16(tvm.cuda(0).compute_version):
        pytest.skip("Skip because %s does not have fp16 support" % target)

    print("Running on target: %s" % target)
    with tvm.target.Target(target):
        s = tvm.topi.testing.get_elemwise_schedule(target)(B)

    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
    # Building with the CSE pass disabled
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
        foo = tvm.build(s, [A, B], target, name="relu")
    foo(a, b)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)


size, alpha = tvm.testing.parameters((100, 0.1))


def test_leaky_relu(size, alpha):
    A = te.placeholder((size,), name="A")
    B = topi.nn.leaky_relu(A, alpha)
    s = te.create_schedule([B.op])

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = a_np * (a_np > 0) + a_np * (a_np < 0) * alpha
    dev = tvm.cpu(0)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
    # Building with the CSE pass disabled
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
        foo = tvm.build(s, [A, B], "llvm", name="leaky_relu")
    foo(a, b)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)


x, w, axis, weight_reshape = tvm.testing.parameters(
    ((1, 3, 2, 2), (3,), 1, (3, 1, 1)),
    ((1, 3, 2, 2), (2,), 2, (2, 1)),
    ((1, 3), (3,), 1, (3,)),
)


def test_prelu(x, w, axis, weight_reshape):
    X = te.placeholder((x), name="X")
    W = te.placeholder((w), name="W")
    x_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(X.shape)).astype(X.dtype)
    w_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(W.shape)).astype(W.dtype)

    def _prelu_numpy(x, W):
        return (x < 0) * (x * W.reshape(weight_reshape)) + (x >= 0) * x

    B = topi.nn.prelu(X, W, axis)
    s = te.create_schedule([B.op])

    dev = tvm.cpu(0)
    x_tvm = tvm.nd.array(x_np, dev)
    w_tvm = tvm.nd.array(w_np, dev)

    b = tvm.nd.array(np.zeros(get_const_tuple(X.shape), dtype=B.dtype), dev)
    # Building with the CSE pass disabled
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
        foo = tvm.build(s, [X, W, B], "llvm", name="prelu")
    foo(x_tvm, w_tvm, b)
    out_np = _prelu_numpy(x_np, w_np)
    tvm.testing.assert_allclose(b.numpy(), out_np, rtol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
