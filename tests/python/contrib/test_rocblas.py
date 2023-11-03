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
    op_l = 128
    m = 235
    input_a = te.placeholder((n, op_l), name="input_a")
    input_b = te.placeholder((op_l, m), name="input_b")
    result_c = rocblas.matmul(input_a, input_b)
    s = te.create_schedule(result_c.op)

    def verify(target="rocm"):
        if not tvm.get_global_func("tvm.contrib.rocblas.matmul", True):
            print("skip because extern function is not available")
            return
        dev = tvm.rocm(0)
        f = tvm.build(s, [input_a, input_b, result_c], target)
        a = tvm.nd.array(np.random.uniform(size=(n, op_l)).astype(input_a.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(op_l, m)).astype(input_b.dtype), dev)
        c = tvm.nd.array(np.zeros((n, m), dtype=result_c.dtype), dev)
        f(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), np.dot(a.numpy(), b.numpy()), rtol=1e-5)

    verify()


def verify_batch_matmul(batch, m, k, n, lib, transa=False, transb=False, dtype="float32"):
    """Tests matmul operation in batch using roc"""
    ashape = (batch, k, m) if transa else (batch, m, k)
    bshape = (batch, n, k) if transb else (batch, k, n)
    input_a = te.placeholder(ashape, name="input_a", dtype=dtype)
    input_b = te.placeholder(bshape, name="input_b", dtype=dtype)
    result_c = lib.batch_matmul(input_a, input_b, transa, transb)
    s = te.create_schedule(result_c.op)

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
        f = tvm.build(s, [input_a, input_b, result_c], target)
        a = tvm.nd.array(np.random.uniform(size=ashape).astype(input_a.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=bshape).astype(input_b.dtype), dev)
        c = tvm.nd.array(np.zeros((batch, m, n), dtype=result_c.dtype), dev)
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
