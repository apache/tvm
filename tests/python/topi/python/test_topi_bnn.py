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
"""Test code for binary neural network operators."""
import numpy as np
import tvm
import tvm.testing
from tvm import te
from tvm import topi
from tvm.topi.utils import get_const_tuple
from tvm.contrib.pickle_memoize import memoize


def verify_binary_dense(batch, in_dim, out_dim):
    A = te.placeholder((batch, in_dim), name="A")
    B = te.placeholder((out_dim, in_dim), name="B")
    bnn_A = topi.nn.binarize_pack(A)
    bnn_B = topi.nn.binarize_pack(B)
    # binary dense
    bnn_A1 = te.placeholder(bnn_A.shape, dtype=bnn_A.dtype)
    bnn_B1 = te.placeholder(bnn_B.shape, dtype=bnn_B.dtype)
    bnn_C = topi.nn.binary_dense(bnn_A1, bnn_B1)
    # schedule
    with tvm.target.Target("llvm"):
        s1 = topi.x86.schedule_binarize_pack(bnn_A)
        s2 = topi.x86.schedule_binarize_pack(bnn_B)
        s3 = topi.x86.schedule_binary_dense(bnn_C)

    dtype = A.dtype

    @memoize("topi.tests.test_topi_binary_dense")
    def get_ref_data():
        # generate random matrix of +1 or -1 value
        a_np = (np.random.randint(2, size=(batch, in_dim)) * 2 - 1).astype(dtype)
        b_np = (np.random.randint(2, size=(out_dim, in_dim)) * 2 - 1).astype(dtype)
        c_np = np.dot(a_np, b_np.T)
        return a_np, b_np, c_np

    a_np, b_np, c_np = get_ref_data()

    dev = tvm.cpu(0)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    bnn_a = tvm.nd.array(np.zeros(get_const_tuple(bnn_A.shape), dtype=bnn_A.dtype), dev)
    bnn_b = tvm.nd.array(np.zeros(get_const_tuple(bnn_B.shape), dtype=bnn_B.dtype), dev)
    bnn_c = tvm.nd.array(np.zeros(get_const_tuple(bnn_C.shape), dtype=bnn_C.dtype), dev)
    f1 = tvm.build(s1, [A, bnn_A], "llvm")
    f2 = tvm.build(s2, [B, bnn_B], "llvm")
    f3 = tvm.build(s3, [bnn_A1, bnn_B1, bnn_C], "llvm")
    f1(a, bnn_a)
    f2(b, bnn_b)
    f3(bnn_a, bnn_b, bnn_c)
    tvm.testing.assert_allclose(bnn_c.numpy(), c_np, rtol=1e-5)


def test_binary_dense():
    verify_binary_dense(1, 4096, 1024)
    verify_binary_dense(1, 1024, 1000)


if __name__ == "__main__":
    test_binary_dense()
