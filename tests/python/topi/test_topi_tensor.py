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
"""Test code for tensor operator"""
import numpy as np
import tvm
from tvm import te
from tvm import topi
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.contrib.nvcc import have_fp16
import tvm.testing


def verify_elemwise_sum(num_args, dtype):
    shape = (3, 5, 4)

    tvm_placeholders = []
    for i in range(num_args):
        tvm_placeholders.append(te.placeholder(shape, name="data" + str(i), dtype=dtype))
    esum = topi.elemwise_sum(tvm_placeholders)
    s = te.create_schedule([esum.op])

    @memoize("topi.tests.test_topi_elemwise_sum")
    def get_ref_data():
        np_nd = [np.random.uniform(0, 10, size=shape).astype(dtype) for i in range(num_args)]
        return np_nd

    np_nd = get_ref_data()

    def check_target(target):
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return

        dev = tvm.device(target, 0)
        out = tvm.nd.array(np.zeros(shape, dtype=dtype), dev)
        f = tvm.build(s, tvm_placeholders + [esum], target, name="elemwise_sum")
        tvm_nd = [tvm.nd.array(nd, dev) for nd in np_nd] + [out]
        f(*tvm_nd)
        np_out = np.sum(np.array(np_nd), axis=0)
        tvm.testing.assert_allclose(out.numpy(), np_out, rtol=1e-5)

    for target in ["llvm"]:
        check_target(target)


def verify_full(shape, dtype, fill_value):
    A = te.placeholder(shape, dtype=dtype, name="A")
    B = topi.full_like(A, fill_value=fill_value)
    C = topi.full(shape=shape, dtype=dtype, fill_value=fill_value)
    s1 = te.create_schedule([B.op])
    s2 = te.create_schedule([C.op])

    @memoize("topi.tests.test_topi_full")
    def get_ref_data():
        return np.full(shape, fill_value, dtype)

    np_nd = get_ref_data()

    def check_target(target):
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return

        dev = tvm.device(target, 0)
        out = tvm.nd.array(np.zeros(shape, dtype=dtype), dev)
        f = tvm.build(s1, [A, B], target, name="full_like")
        f(tvm.nd.array(np.zeros(shape, dtype), dev), out)
        tvm.testing.assert_allclose(out.numpy(), np_nd, rtol=1e-5)

        f = tvm.build(s2, [C], target, name="full")
        f(out)
        tvm.testing.assert_allclose(out.numpy(), np_nd, rtol=1e-5)

    for target in ["llvm"]:
        check_target(target)


def verify_vectorization(n, m, dtype):
    def check_targeta(targeta):
        if not tvm.testing.device_enabled(targeta):
            print("Skip because %s is not enabled" % targeta)
            return
        if dtype == "float16" and targeta == "cuda" and not have_fp16(tvm.cuda(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return
        with tvm.target.Target(targeta):
            dev = tvm.device(targeta, 0)
            A = te.placeholder((n, m), name="A", dtype=dtype)
            B = te.compute((n, m), lambda i, j: A[i, j] + tvm.tir.const(1, A.dtype), name="B")
            S = tvm.topi.testing.get_elemwise_schedule(targeta)(B)

            fun = tvm.build(S, [A, B], targeta)
            np_A = tvm.nd.empty((n, m), A.dtype, dev).copyfrom(np.random.uniform(size=(n, m)))
            np_B = tvm.nd.empty((n, m), B.dtype, dev)
            fun(np_A, np_B)
            tvm.testing.assert_allclose(np_B.numpy(), np_A.numpy() + 1, rtol=1e-5)

    for targeta in ["cuda"]:
        check_targeta(targeta)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_vectorization():
    verify_vectorization(128, 64, "float16")


def test_elemwise_sum():
    verify_elemwise_sum(1, "float32")
    verify_elemwise_sum(5, "float32")
    verify_elemwise_sum(4, "int32")


def test_full():
    verify_full((3, 4, 5), "float32", 3.14)
    verify_full((10,), "int32", 7)


if __name__ == "__main__":
    test_elemwise_sum()
    test_full()
    test_vectorization()
