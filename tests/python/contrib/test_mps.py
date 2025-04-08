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
import tvm.testing
from tvm import te
import numpy as np
from tvm.contrib import mps


@tvm.testing.requires_metal
def test_matmul():
    n = 1024
    l = 128
    m = 256
    A = te.placeholder((n, l), name="A")
    B = te.placeholder((l, m), name="B")
    C = mps.matmul(A, B)

    def verify(A, B, C):
        if not tvm.get_global_func("tvm.contrib.mps.matmul", True):
            print("skip because extern function is not available")
            return
        dev = tvm.metal(0)
        f = tvm.compile(te.create_prim_func([A, B, C]), target="metal")
        a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(l, m)).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), dev)
        f(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), np.dot(a.numpy(), b.numpy()), rtol=1e-5)

    verify(A, B, C)


@tvm.testing.requires_metal
def test_conv2d():
    n = 1
    h = 14
    w = 14
    ci = 2
    co = 4
    kh = 3
    kw = 3
    stride = 2
    A = te.placeholder((n, h, w, ci), name="x")
    B = te.placeholder((co, kh, kw, ci), name="w")
    C = mps.conv2d(A, B, "SAME", 2)

    def verify(A, B, C, target="llvm"):
        if not tvm.get_global_func("tvm.contrib.mps.conv2d", True):
            print("skip because extern function is not available")
            return
        dev = tvm.metal(0)
        f = tvm.compile(te.create_prim_func([A, B, C]), target="metal")
        a = tvm.nd.array(np.random.uniform(size=(n, h, w, ci)).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(co, kh, kw, ci)).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros((n, h // stride, w // stride, co), dtype=C.dtype), dev)
        f(a, b, c)

    verify(A, B, C, s1)


if __name__ == "__main__":
    # test_matmul()
    test_conv2d()
