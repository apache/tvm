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
"""Configure pytest"""
# pylint: disable=invalid-name
import numpy as np
import tvm
import tvm.testing
from tvm import te
import tvm.topi.testing
from tvm.contrib import rocblas


@tvm.testing.requires_rocm
def test_matmul():
    """Tests matmul operation using roc"""
    n = 1024
    l = 128
    m = 235
    A = te.placeholder((n, l), name="A")
    B = te.placeholder((l, m), name="B")
    C = rocblas.matmul(A, B)
    s = te.create_schedule(C.op)

    def verify(target="rocm"):
        if not tvm.get_global_func("tvm.contrib.rocblas.matmul", True):
            print("skip because extern function is not available")
            return
        dev = tvm.rocm(0)
        f = tvm.build(s, [A, B, C], target)
        a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(l, m)).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), dev)
        f(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), np.dot(a.numpy(), b.numpy()), rtol=1e-5)

    verify()


def verify_batch_matmul(batch, m, k, n, lib, transa=False, transb=False, dtype="float32"):
    """Tests matmul operation in batch using roc"""
    ashape = (batch, k, m) if transa else (batch, m, k)
    bshape = (batch, n, k) if transb else (batch, k, n)
    A = te.placeholder(ashape, name="A", dtype=dtype)
    B = te.placeholder(bshape, name="B", dtype=dtype)
    C = lib.batch_matmul(A, B, transa, transb)
    s = te.create_schedule(C.op)

    def get_numpy(a, b, transa, transb):
        if transa:
            a = a.transpose(0, 2, 1)
        if not transb:
            b = b.transpose(0, 2, 1)
        return tvm.topi.testing.batch_matmul(a, b)

    def verify(target="rocm"):
        if not tvm.testing.device_enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func(lib.__name__ + ".batch_matmul", True):
            print("skip because extern function is not available")
            return
        dev = tvm.rocm(0)
        f = tvm.build(s, [A, B, C], target)
        a = tvm.nd.array(np.random.uniform(size=ashape).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=bshape).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros((batch, m, n), dtype=C.dtype), dev)
        f(a, b, c)
        tvm.testing.assert_allclose(
            c.numpy(), get_numpy(a.numpy(), b.numpy(), transa, transb), rtol=1e-5
        )

    verify()


@tvm.testing.requires_rocm
def test_batch_matmul():
    """Tests of matmul operation in batch using roc"""
    verify_batch_matmul(128, 64, 512, 512, rocblas, transa=False, transb=False)
    verify_batch_matmul(128, 64, 512, 512, rocblas, transa=False, transb=True)
    verify_batch_matmul(128, 64, 512, 512, rocblas, transa=True, transb=False)
    verify_batch_matmul(128, 64, 512, 512, rocblas, transa=True, transb=True)
    verify_batch_matmul(128, 512, 512, 64, rocblas, transa=False, transb=False)
    verify_batch_matmul(128, 512, 512, 64, rocblas, transa=False, transb=True)
    verify_batch_matmul(128, 512, 512, 64, rocblas, transa=True, transb=False)
    verify_batch_matmul(128, 512, 512, 64, rocblas, transa=True, transb=True)
    verify_batch_matmul(128, 512, 64, 512, rocblas, transa=False, transb=False)
    verify_batch_matmul(128, 512, 64, 512, rocblas, transa=False, transb=True)
    verify_batch_matmul(128, 512, 64, 512, rocblas, transa=True, transb=False)
    verify_batch_matmul(128, 512, 64, 512, rocblas, transa=True, transb=True)
    verify_batch_matmul(128, 64, 128, 128, rocblas, transa=False, transb=False)
    verify_batch_matmul(128, 64, 128, 128, rocblas, transa=False, transb=True)
    verify_batch_matmul(128, 64, 128, 128, rocblas, transa=True, transb=False)
    verify_batch_matmul(128, 64, 128, 128, rocblas, transa=True, transb=True)
    verify_batch_matmul(128, 128, 128, 64, rocblas, transa=False, transb=False)
    verify_batch_matmul(128, 128, 128, 64, rocblas, transa=False, transb=True)
    verify_batch_matmul(128, 128, 128, 64, rocblas, transa=True, transb=False)
    verify_batch_matmul(128, 128, 128, 64, rocblas, transa=True, transb=True)


if __name__ == "__main__":
    test_matmul()
    test_batch_matmul()
