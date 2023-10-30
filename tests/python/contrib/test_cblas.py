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
import pytest
import numpy as np
import tvm
from tvm import te
from tvm.contrib import cblas
from tvm.contrib import mkl
from tvm.contrib import dnnl
import tvm.testing
import tvm.topi.testing


def verify_matmul_add(
    matrix_m, matrix_l, matrix_n, lib, transa=False, transb=False, dtype="float32"
):
    """Tests matmul+add op"""
    bias = te.var("bias", dtype=dtype)
    ashape = (matrix_l, matrix_n) if transa else (matrix_n, matrix_l)
    bshape = (matrix_m, matrix_l) if transb else (matrix_l, matrix_m)
    input1_data = te.placeholder(ashape, name="input1_data", dtype=dtype)
    input2_data = te.placeholder(bshape, name="input2_data", dtype=dtype)
    matmul_result = lib.matmul(input1_data, input2_data, transa, transb)
    final_result = te.compute(
        matmul_result.shape, lambda i, j: matmul_result[i, j] + bias, name="final_result"
    )
    s = te.create_schedule(final_result.op)

    def get_numpy(a, b, matrix_bias, transa, transb):
        if transa:
            a = a.transpose()
        if transb:
            b = b.transpose()
        return np.dot(a, b) + matrix_bias

    def compiling(f, name="test_matmul_add", ext=".so"):
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
        f = tvm.build(s, [input1_data, input2_data, final_result, bias], target, name=name)
        if target == "c":
            f = compiling(f, name)
        matrix_input1 = tvm.nd.array(np.random.uniform(size=ashape).astype(input1_data.dtype), dev)
        matrix_input2 = tvm.nd.array(np.random.uniform(size=bshape).astype(input2_data.dtype), dev)
        matrix_result = tvm.nd.array(np.zeros((matrix_n, matrix_m), dtype=final_result.dtype), dev)
        matrix_bias = 10.0
        f(matrix_input1, matrix_input2, matrix_result, matrix_bias)
        tvm.testing.assert_allclose(
            matrix_result.numpy(),
            get_numpy(matrix_input1.numpy(), matrix_input2.numpy(), matrix_bias, transa, transb),
            rtol=1e-5,
        )

    verify("llvm")
    verify("c")


def test_matmul_add():
    """Tests of matmul+add op"""
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


def verify_quantized_matmul_add(matrix_m, matrix_l, matrix_n, transa=False, transb=False):
    """Tests quantized matmul+add op"""
    if not tvm.get_global_func("tvm.contrib.mkl.matmul_u8s8s32", True):
        pytest.skip("Quantized dense is supported only for MKL. TVM GPU CI uses openblas")
    data_dtype = "uint8"
    kernel_dtype = "int8"
    out_dtype = "int32"
    bias = te.var("bias", dtype=out_dtype)
    ashape = (matrix_l, matrix_n) if transa else (matrix_n, matrix_l)
    bshape = (matrix_m, matrix_l) if transb else (matrix_l, matrix_m)
    input1_data = te.placeholder(ashape, name="input1_data", dtype=data_dtype)
    input2_data = te.placeholder(bshape, name="input2_data", dtype=kernel_dtype)
    matmul_result = mkl.matmul_u8s8s32(input1_data, input2_data, transa, transb, dtype=out_dtype)
    final_result = te.compute(
        matmul_result.shape, lambda i, j: matmul_result[i, j] + bias, name="final_result"
    )
    s = te.create_schedule(final_result.op)

    def get_numpy(a, b, matrix_bias, transa, transb):
        if transa:
            a = a.transpose()
        if transb:
            b = b.transpose()
        return np.dot(a, b) + matrix_bias

    def verify(target="llvm"):
        if not tvm.testing.device_enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.mkl.matmul_u8s8s32", True):
            print("skip because extern function is not available")
            return
        dev = tvm.cpu(0)
        f = tvm.build(s, [input1_data, input2_data, final_result, bias], target)
        matrix_input1 = tvm.nd.array(
            np.random.randint(low=0, high=50, size=ashape).astype(input1_data.dtype), dev
        )
        matrix_input2 = tvm.nd.array(
            np.random.randint(low=0, high=50, size=bshape).astype(input2_data.dtype), dev
        )
        matrix_result = tvm.nd.array(np.zeros((matrix_n, matrix_m), dtype=final_result.dtype), dev)
        matrix_bias = 10
        f(matrix_input1, matrix_input2, matrix_result, matrix_bias)
        tvm.testing.assert_allclose(
            matrix_result.numpy(),
            get_numpy(
                matrix_input1.numpy().astype("int32"),
                matrix_input2.numpy().astype("int32"),
                matrix_bias,
                transa,
                transb,
            ),
            rtol=1e-5,
        )

    verify()


def test_quantized_matmul_add():
    """Tests of quantized matmul+add op"""
    verify_quantized_matmul_add(235, 128, 1024)
    verify_quantized_matmul_add(235, 128, 1024, True, False)
    verify_quantized_matmul_add(235, 128, 1024, False, True)
    verify_quantized_matmul_add(235, 128, 1024, True, True)
    verify_quantized_matmul_add(1, 16, 4)
    verify_quantized_matmul_add(1, 16, 3, True, False)
    verify_quantized_matmul_add(1, 16, 3, False, True)
    verify_quantized_matmul_add(1, 16, 3, True, True)


def verify_batch_matmul(
    batch_a,
    batch_b,
    matrix_m,
    matrix_l,
    matrix_n,
    lib,
    transa=False,
    transb=False,
    dtype="float32",
):
    """Tests matmul op where matrices are in batch"""
    batch = max(batch_a, batch_b)
    ashape = (batch_a, matrix_l, matrix_n) if transa else (batch_a, matrix_n, matrix_l)
    bshape = (batch_b, matrix_m, matrix_l) if transb else (batch_b, matrix_l, matrix_m)
    input1_data = te.placeholder(ashape, name="input1_data", dtype=dtype)
    input2_data = te.placeholder(bshape, name="input2_data", dtype=dtype)
    matmul_result = lib.batch_matmul(input1_data, input2_data, transa, transb)
    final_result = te.compute(
        matmul_result.shape, lambda k, i, j: matmul_result[k, i, j], name="final_result"
    )
    s = te.create_schedule(final_result.op)

    def get_numpy(a, b, transa, transb):
        if transa:
            a = a.transpose(0, 2, 1)
        if not transb:
            b = b.transpose(0, 2, 1)
        return tvm.topi.testing.batch_matmul(a, b)

    def compiling(f, name="test_batch_matmul", ext=".so"):
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
        f = tvm.build(s, [input1_data, input2_data, final_result], target, name=name)
        if target == "c":
            f = compiling(f, name)
        matrix_input1 = tvm.nd.array(np.random.uniform(size=ashape).astype(input1_data.dtype), dev)
        matrix_input2 = tvm.nd.array(np.random.uniform(size=bshape).astype(input2_data.dtype), dev)
        matrix_result = tvm.nd.array(
            np.zeros((batch, matrix_n, matrix_m), dtype=final_result.dtype), dev
        )
        f(matrix_input1, matrix_input2, matrix_result)
        tvm.testing.assert_allclose(
            matrix_result.numpy(),
            get_numpy(matrix_input1.numpy(), matrix_input2.numpy(), transa, transb),
            rtol=1e-5,
        )

    verify("llvm")
    verify("c")


def test_batch_matmul():
    """Tests of matmul op where matrices are in batch"""
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
    verify_batch_matmul(16, 1, 235, 128, 1024, cblas)
    verify_batch_matmul(1, 16, 235, 128, 1024, cblas)
    verify_batch_matmul(16, 1, 235, 128, 1024, mkl)
    verify_batch_matmul(1, 16, 235, 128, 1024, mkl)
    verify_batch_matmul(16, 1, 235, 128, 1024, mkl)
    verify_batch_matmul(1, 16, 235, 128, 1024, mkl)
    verify_batch_matmul(1, 1, 1, 16, 3, cblas)
    verify_batch_matmul(1, 1, 1, 16, 3, cblas, True, False)
    verify_batch_matmul(1, 1, 1, 16, 3, cblas, False, False)
    verify_batch_matmul(1, 1, 1, 16, 3, cblas, True, True)
    verify_batch_matmul(1, 1, 1, 16, 3, cblas)
    verify_batch_matmul(1, 1, 1, 16, 3, mkl)
    verify_batch_matmul(1, 1, 1, 16, 3, mkl, True, False)
    verify_batch_matmul(1, 1, 1, 16, 3, mkl, False, False)
    verify_batch_matmul(1, 1, 1, 16, 3, mkl, True, True)
    verify_batch_matmul(1, 1, 1, 16, 3, mkl)


if __name__ == "__main__":
    test_matmul_add()
    test_quantized_matmul_add()
    test_batch_matmul()
