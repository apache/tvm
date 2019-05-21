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
import tvm
import numpy as np
import topi.testing
from tvm.contrib import cblas

def verify_matmul_add(m, l, n, transa=False, transb=False, dtype=tvm.float32):
    bias = tvm.var('bias', dtype=dtype)
    ashape = (l, n) if transa else (n, l)
    bshape = (m, l) if transb else (l, m)
    A = tvm.placeholder(ashape, name='A', dtype=dtype)
    B = tvm.placeholder(bshape, name='B', dtype=dtype)
    C = cblas.matmul(A, B, transa, transb)
    D = tvm.compute(C.shape, lambda i, j: C[i,j] + bias, name="D")
    s = tvm.create_schedule(D.op)

    def get_numpy(a, b, bb, transa, transb):
        if transa:
            a = a.transpose()
        if transb:
            b = b.transpose()
        return np.dot(a, b) + bb

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.cblas.matmul", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A, B, D, bias], target)
        a = tvm.nd.array(np.random.uniform(size=ashape).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=bshape).astype(B.dtype), ctx)
        d = tvm.nd.array(np.zeros((n, m), dtype=D.dtype), ctx)
        bb = 10.0
        f(a, b, d, bb)
        tvm.testing.assert_allclose(
            d.asnumpy(), get_numpy(a.asnumpy(), b.asnumpy(), bb, transa, transb), rtol=1e-5)
    verify()

def test_matmul_add():
    verify_matmul_add(235, 128, 1024)
    verify_matmul_add(235, 128, 1024, True, False)
    verify_matmul_add(235, 128, 1024, False, True)
    verify_matmul_add(235, 128, 1024, True, True)
    verify_matmul_add(1, 16, 4)
    verify_matmul_add(1, 16, 3, True, False)
    verify_matmul_add(1, 16, 3, False, False)
    verify_matmul_add(1, 16, 3, True, True)

def verify_batch_matmul(batch, m, l, n, transa=False, transb=False, iterative=False, dtype=tvm.float32):
    ashape = (batch, l, n) if transa else (batch, n, l)
    bshape = (batch, m, l) if transb else (batch, l, m)
    A = tvm.placeholder(ashape, name='A', dtype=dtype)
    B = tvm.placeholder(bshape, name='B', dtype=dtype)
    C = cblas.batch_matmul(A, B, transa, transb)
    D = tvm.compute(C.shape, lambda k, i, j: C[k, i,j], name="D")
    s = tvm.create_schedule(D.op)

    def get_numpy(a, b, transa, transb):
        if transa:
            a = a.transpose(0, 2, 1)
        if not transb:
            b = b.transpose(0, 2, 1)
        return topi.testing.batch_matmul(a, b)

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.cblas.matmul", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A, B, D], target)
        a = tvm.nd.array(np.random.uniform(size=ashape).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=bshape).astype(B.dtype), ctx)
        d = tvm.nd.array(np.zeros((batch, n, m), dtype=D.dtype), ctx)
        f(a, b, d)
        tvm.testing.assert_allclose(
            d.asnumpy(), get_numpy(a.asnumpy(), b.asnumpy(), transa, transb), rtol=1e-5)
    verify()

def test_batch_matmul():
    verify_batch_matmul(16, 235, 128, 1024)
    verify_batch_matmul(16, 235, 128, 1024, True, False)
    verify_batch_matmul(16, 235, 128, 1024, False, True)
    verify_batch_matmul(16, 235, 128, 1024, True, True)
    verify_batch_matmul(1, 1, 16, 3)
    verify_batch_matmul(1, 1, 16, 3, True, False)
    verify_batch_matmul(1, 1, 16, 3, False, False)
    verify_batch_matmul(1, 1, 16, 3, True, True)
    verify_batch_matmul(1, 1, 16, 3, iterative=True)

if __name__ == "__main__":
    test_matmul_add()
    test_batch_matmul()
