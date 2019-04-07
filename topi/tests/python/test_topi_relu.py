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
import os
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple

from common import get_all_backend

def verify_relu(m, n):
    A = tvm.placeholder((m, n), name='A')
    B = topi.nn.relu(A)

    a_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = a_np * (a_np > 0)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_elemwise(B)

        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        foo = tvm.build(s, [A, B], device, name="relu")
        foo(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in get_all_backend():
        check_device(device)


def verify_leaky_relu(m, alpha):
    A = tvm.placeholder((m,), name='A')
    B = topi.nn.leaky_relu(A, alpha)
    s = tvm.create_schedule([B.op])

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = a_np * (a_np > 0) + a_np * (a_np < 0) * alpha
    ctx = tvm.cpu(0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    foo = tvm.build(s, [A, B], "llvm", name="leaky_relu")
    foo(a, b)
    tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)


def verify_prelu(x, w, axis, weight_reshape):
    X = tvm.placeholder((x), name='X')
    W = tvm.placeholder((w), name='W')
    x_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(X.shape)).astype(X.dtype)
    w_np = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(W.shape)).astype(W.dtype)

    def _prelu_numpy(x, W):
        return (x < 0) * (x *W.reshape(weight_reshape)) + (x>=0) * x

    B = topi.nn.prelu(X, W, axis)
    s = tvm.create_schedule([B.op])

    ctx = tvm.cpu(0)
    x_tvm = tvm.nd.array(x_np, ctx)
    w_tvm = tvm.nd.array(w_np, ctx)

    b = tvm.nd.array(np.zeros(get_const_tuple(X.shape), dtype=B.dtype), ctx)
    foo = tvm.build(s, [X, W, B], "llvm", name="prelu")
    foo(x_tvm, w_tvm, b)
    out_np = _prelu_numpy(x_np, w_np)
    tvm.testing.assert_allclose(b.asnumpy(), out_np, rtol=1e-5)

def test_relu():
    verify_relu(10, 128)

def test_schedule_big_array():
    verify_relu(1024 * 100 , 512)


def test_leaky_relu():
    verify_leaky_relu(100, 0.1)

def test_prelu():
    verify_prelu((1, 3, 2, 2), (3,), 1, (3, 1, 1))
    verify_prelu((1, 3, 2, 2), (2,), 2, (2, 1))
    verify_prelu((1, 3), (3,), 1, (3, ))

if __name__ == "__main__":
    test_schedule_big_array()
    test_relu()
    test_leaky_relu()
    test_prelu()
