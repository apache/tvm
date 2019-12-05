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
from tvm.contrib import cublas

def verify_matmul_add(in_dtype, out_dtype, rtol=1e-5):
    n = 1024
    l = 128
    m = 236
    A = tvm.placeholder((n, l), name='A', dtype=in_dtype)
    B = tvm.placeholder((l, m), name='B', dtype=in_dtype)
    C = cublas.matmul(A, B, dtype=out_dtype)
    s = tvm.create_schedule(C.op)

    def verify(target="cuda"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.cublas.matmul", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.gpu(0)
        f = tvm.build(s, [A, B, C], target)
        a = tvm.nd.array(np.random.uniform(0, 128, size=(n, l)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(0, 128, size=(l, m)).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
        f(a, b, c)
        tvm.testing.assert_allclose(
            c.asnumpy(), np.dot(a.asnumpy().astype(C.dtype), b.asnumpy().astype(C.dtype)), rtol=rtol)
    verify()

def verify_batch_matmul(in_dtype, out_dtype, rtol=1e-5):
    j = 16
    n = 1024
    l = 128
    m = 236
    A = tvm.placeholder((j, n, l), name='A', dtype=in_dtype)
    B = tvm.placeholder((j, l, m), name='B', dtype=in_dtype)
    C = cublas.batch_matmul(A, B, dtype=out_dtype)
    s = tvm.create_schedule(C.op)

    def verify(target="cuda"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.cublas.matmul", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.gpu(0)
        f = tvm.build(s, [A, B, C], target)
        a = tvm.nd.array(np.random.uniform(size=(j, n, l)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(j, l, m)).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros((j, n, m), dtype=C.dtype), ctx)
        f(a, b, c)
        tvm.testing.assert_allclose(
            c.asnumpy(), np.matmul(a.asnumpy().astype(C.dtype),
                                   b.asnumpy().astype(C.dtype)).astype(C.dtype), rtol=rtol)
    verify()

def test_matmul_add():
    verify_matmul_add('float', 'float')
    verify_matmul_add('float16', 'float')
    verify_matmul_add('float16', 'float16', rtol=1e-2)
    verify_matmul_add('int8', 'int32')

def test_batch_matmul():
    verify_batch_matmul('float', 'float')
    verify_batch_matmul('float16', 'float')
    verify_batch_matmul('float16', 'float16', rtol=1e-2)

if __name__ == "__main__":
    test_matmul_add()
    test_batch_matmul()

