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
import pytest
import tvm
from tvm import te
import numpy as np
import tvm.topi.testing
from tvm.contrib import cblas
from tvm.contrib import mkl
from tvm.contrib import dnnl
import tvm.testing


def verify_matmul_add(m, l, n, lib, transa=False, transb=False, dtype="float32"):
    bias = te.var("bias", dtype=dtype)
    ashape = (l, n) if transa else (n, l)
    bshape = (m, l) if transb else (l, m)
    A = te.placeholder(ashape, name="A", dtype=dtype)
    B = te.placeholder(bshape, name="B", dtype=dtype)
    C = lib.matmul(A, B, transa, transb)
    D = te.compute(C.shape, lambda i, j: C[i, j] + bias, name="D")
    s = te.create_schedule(D.op)

    def get_numpy(a, b, bb, transa, transb):
        if transa:
            a = a.transpose()
        if transb:
            b = b.transpose()
        return np.dot(a, b) + bb

    def compile(f, name="test_matmul_add", ext=".so"):
        path = name + ext
        f.export_library(path)
        mod = tvm.runtime.load_module(path)
        f = mod[name]
        return f

    def verify(target="llvm"):
        if not tvm.testing.device_enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func(lib.__name__ + ".matmul", True):
            print("skip because extern function is not available")
            return
        dev = tvm.cpu(0)
        name = "test_matmul_add"
        f = tvm.build(s, [A, B, D, bias], target, name=name)
        if target == "c":
            f = compile(f, name)
        a = tvm.nd.array(np.random.uniform(size=ashape).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=bshape).astype(B.dtype), dev)
        d = tvm.nd.array(np.zeros((n, m), dtype=D.dtype), dev)
        bb = 10.0
        f(a, b, d, bb)
        tvm.testing.assert_allclose(
            d.numpy(), get_numpy(a.numpy(), b.numpy(), bb, transa, transb), rtol=1e-5
        )

    verify("llvm")
    verify("c")


def test_matmul_add():
    verify_matmul_add(235, 128, 1024, cblas)
    verify_matmul_add(235, 128, 1024, cblas, True, False)
    verify_matmul_add(235, 128, 1024, cblas, False, True)
    verify_matmul_add(235, 128, 1024, cblas, True, True)
    verify_matmul_add(235, 128, 1024, mkl)
    verify_matmul_add(235, 128, 1024, mkl, True, False)
    verify_matmul_add(235, 128, 1024, mkl, False, True)
    verify_matmul_add(235, 128, 1024, mkl, True, True)
    verify_matmul_add(235, 128, 1024, dnnl)
    verify_matmul_add(235, 128, 1024, dnnl, True, False)
    verify_matmul_add(235, 128, 1024, dnnl, False, True)
    verify_matmul_add(235, 128, 1024, dnnl, True, True)
    verify_matmul_add(1, 16, 4, cblas)
    verify_matmul_add(1, 16, 3, cblas, True, False)
    verify_matmul_add(1, 16, 3, cblas, False, False)
    verify_matmul_add(1, 16, 3, cblas, True, True)
    verify_matmul_add(1, 16, 4, mkl)
    verify_matmul_add(1, 16, 3, mkl, True, False)
    verify_matmul_add(1, 16, 3, mkl, False, False)
    verify_matmul_add(1, 16, 3, mkl, True, True)
    verify_matmul_add(1, 16, 4, dnnl)
    verify_matmul_add(1, 16, 3, dnnl, True, False)
    verify_matmul_add(1, 16, 3, dnnl, False, False)
    verify_matmul_add(1, 16, 3, dnnl, True, True)


def verify_quantized_matmul_add(m, l, n, transa=False, transb=False):
    if not tvm.get_global_func("tvm.contrib.mkl.matmul_u8s8s32", True):
        pytest.skip("Quantized dense is supported only for MKL. TVM GPU CI uses openblas")
    data_dtype = "uint8"
    kernel_dtype = "int8"
    out_dtype = "int32"
    bias = te.var("bias", dtype=out_dtype)
    ashape = (l, n) if transa else (n, l)
    bshape = (m, l) if transb else (l, m)
    A = te.placeholder(ashape, name="A", dtype=data_dtype)
    B = te.placeholder(bshape, name="B", dtype=kernel_dtype)
    C = mkl.matmul_u8s8s32(A, B, transa, transb, dtype=out_dtype)
    D = te.compute(C.shape, lambda i, j: C[i, j] + bias, name="D")
    s = te.create_schedule(D.op)

    def get_numpy(a, b, bb, transa, transb):
        if transa:
            a = a.transpose()
        if transb:
            b = b.transpose()
        return np.dot(a, b) + bb

    def verify(target="llvm"):
        if not tvm.testing.device_enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.mkl.matmul_u8s8s32", True):
            print("skip because extern function is not available")
            return
        dev = tvm.cpu(0)
        f = tvm.build(s, [A, B, D, bias], target)
        a = tvm.nd.array(np.random.randint(low=0, high=50, size=ashape).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.randint(low=0, high=50, size=bshape).astype(B.dtype), dev)
        d = tvm.nd.array(np.zeros((n, m), dtype=D.dtype), dev)
        bb = 10
        f(a, b, d, bb)
        tvm.testing.assert_allclose(
            d.numpy(),
            get_numpy(a.numpy().astype("int32"), b.numpy().astype("int32"), bb, transa, transb),
            rtol=1e-5,
        )

    verify()


def test_quantized_matmul_add():
    verify_quantized_matmul_add(235, 128, 1024)
    verify_quantized_matmul_add(235, 128, 1024, True, False)
    verify_quantized_matmul_add(235, 128, 1024, False, True)
    verify_quantized_matmul_add(235, 128, 1024, True, True)
    verify_quantized_matmul_add(1, 16, 4)
    verify_quantized_matmul_add(1, 16, 3, True, False)
    verify_quantized_matmul_add(1, 16, 3, False, True)
    verify_quantized_matmul_add(1, 16, 3, True, True)


def verify_batch_matmul(
    batch_a, batch_b, m, l, n, lib, transa=False, transb=False, iterative=False, dtype="float32"
):
    batch = max(batch_a, batch_b)
    ashape = (batch_a, l, n) if transa else (batch_a, n, l)
    bshape = (batch_b, m, l) if transb else (batch_b, l, m)
    A = te.placeholder(ashape, name="A", dtype=dtype)
    B = te.placeholder(bshape, name="B", dtype=dtype)
    C = lib.batch_matmul(A, B, transa, transb)
    D = te.compute(C.shape, lambda k, i, j: C[k, i, j], name="D")
    s = te.create_schedule(D.op)

    def get_numpy(a, b, transa, transb):
        if transa:
            a = a.transpose(0, 2, 1)
        if not transb:
            b = b.transpose(0, 2, 1)
        return tvm.topi.testing.batch_matmul(a, b)

    def compile(f, name="test_batch_matmul", ext=".so"):
        path = name + ext
        f.export_library(path)
        mod = tvm.runtime.load_module(path)
        f = mod[name]
        return f

    def verify(target="llvm"):
        if not tvm.testing.device_enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func(lib.__name__ + ".matmul", True):
            print("skip because extern function is not available")
            return
        dev = tvm.cpu(0)
        name = "test_batch_matmul"
        f = tvm.build(s, [A, B, D], target, name=name)
        if target == "c":
            f = compile(f, name)
        a = tvm.nd.array(np.random.uniform(size=ashape).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=bshape).astype(B.dtype), dev)
        d = tvm.nd.array(np.zeros((batch, n, m), dtype=D.dtype), dev)
        f(a, b, d)
        tvm.testing.assert_allclose(
            d.numpy(), get_numpy(a.numpy(), b.numpy(), transa, transb), rtol=1e-5
        )

    verify("llvm")
    verify("c")


def test_batch_matmul():
    verify_batch_matmul(16, 16, 235, 128, 1024, cblas)
    verify_batch_matmul(16, 16, 235, 128, 1024, cblas, True, False)
    verify_batch_matmul(16, 16, 235, 128, 1024, cblas, False, True)
    verify_batch_matmul(16, 16, 235, 128, 1024, cblas, True, True)
    verify_batch_matmul(16, 16, 235, 128, 1024, mkl)
    verify_batch_matmul(16, 16, 235, 128, 1024, mkl, True, False)
    verify_batch_matmul(16, 16, 235, 128, 1024, mkl, False, True)
    verify_batch_matmul(16, 16, 235, 128, 1024, mkl, True, True)
    verify_batch_matmul(16, 1, 235, 128, 1024, cblas)
    verify_batch_matmul(1, 16, 235, 128, 1024, cblas)
    verify_batch_matmul(16, 1, 235, 128, 1024, cblas, iterative=True)
    verify_batch_matmul(1, 16, 235, 128, 1024, cblas, iterative=True)
    verify_batch_matmul(16, 1, 235, 128, 1024, mkl)
    verify_batch_matmul(1, 16, 235, 128, 1024, mkl)
    verify_batch_matmul(16, 1, 235, 128, 1024, mkl, iterative=True)
    verify_batch_matmul(1, 16, 235, 128, 1024, mkl, iterative=True)
    verify_batch_matmul(1, 1, 1, 16, 3, cblas)
    verify_batch_matmul(1, 1, 1, 16, 3, cblas, True, False)
    verify_batch_matmul(1, 1, 1, 16, 3, cblas, False, False)
    verify_batch_matmul(1, 1, 1, 16, 3, cblas, True, True)
    verify_batch_matmul(1, 1, 1, 16, 3, cblas, iterative=True)
    verify_batch_matmul(1, 1, 1, 16, 3, mkl)
    verify_batch_matmul(1, 1, 1, 16, 3, mkl, True, False)
    verify_batch_matmul(1, 1, 1, 16, 3, mkl, False, False)
    verify_batch_matmul(1, 1, 1, 16, 3, mkl, True, True)
    verify_batch_matmul(1, 1, 1, 16, 3, mkl, iterative=True)


if __name__ == "__main__":
    test_matmul_add()
    test_quantized_matmul_add()
    test_batch_matmul()
