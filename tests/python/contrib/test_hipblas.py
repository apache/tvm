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
import numpy as np

import tvm
import tvm.testing
from tvm import te
from tvm.contrib import hipblas


def verify_matmul_add(in_dtype, out_dtype, rtol=1e-5):
    n = 1024
    l = 128
    m = 236
    A = te.placeholder((n, l), name="A", dtype=in_dtype)
    B = te.placeholder((l, m), name="B", dtype=in_dtype)
    C = hipblas.matmul(A, B, dtype=out_dtype)
    s = te.create_schedule(C.op)

    def verify(target="rocm"):
        if not tvm.get_global_func("tvm.contrib.hipblas.matmul", True):
            print("skip because extern function is not available")
            return
        dev = tvm.rocm(0)
        f = tvm.build(s, [A, B, C], target)
        a = tvm.nd.array(np.random.uniform(0, 128, size=(n, l)).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(0, 128, size=(l, m)).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), dev)
        f(a, b, c)
        tvm.testing.assert_allclose(
            c.numpy(), np.dot(a.numpy().astype(C.dtype), b.numpy().astype(C.dtype)), rtol=rtol
        )

    verify()


def roundoff(v, d):
    return int(np.floor((v + d - 1) / d) * d)


def verify_batch_matmul(Ashape, Bshape, Cshape, in_dtype, out_dtype, rtol=1e-5):
    A = te.placeholder(Ashape, name="A", dtype=in_dtype)
    B = te.placeholder(Bshape, name="B", dtype=in_dtype)
    C = hipblas.batch_matmul(A, B, dtype=out_dtype)
    s = te.create_schedule(C.op)

    dev = tvm.rocm(0)
    f = tvm.build(s, [A, B, C], "rocm")

    if "int" in in_dtype:
        a = tvm.nd.array(np.random.uniform(1, 10, size=Ashape).astype(in_dtype), dev)
        b = tvm.nd.array(np.random.uniform(1, 10, size=Bshape).astype(in_dtype), dev)
    else:
        a = tvm.nd.array(np.random.uniform(size=Ashape).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=Bshape).astype(B.dtype), dev)

    c = tvm.nd.array(np.zeros(Cshape, dtype=C.dtype), dev)
    f(a, b, c)
    tvm.testing.assert_allclose(
        c.numpy(),
        np.matmul(a.numpy().astype(C.dtype), b.numpy().astype(C.dtype)).astype(C.dtype),
        rtol=rtol,
    )


@tvm.testing.requires_rocm
def test_matmul_add():
    verify_matmul_add("float", "float", rtol=1e-3)
    verify_matmul_add("float16", "float")
    verify_matmul_add("float16", "float16", rtol=1e-2)
    verify_matmul_add("int8", "int32")


@tvm.testing.requires_rocm
def test_batch_matmul():
    if not tvm.get_global_func("tvm.contrib.hipblas.batch_matmul", True):
        print("skip because extern function is not available")
        return

    verify_batch_matmul((16, 1024, 128), (16, 128, 236), (16, 1024, 236), "float", "float")
    verify_batch_matmul((16, 1024, 128), (1, 128, 236), (16, 1024, 236), "float", "float")
    verify_batch_matmul((16, 1024, 128), (16, 128, 236), (16, 1024, 236), "float16", "float")
    verify_batch_matmul((16, 1024, 128), (1, 128, 236), (16, 1024, 236), "float16", "float")
    verify_batch_matmul(
        (16, 1024, 128), (16, 128, 236), (16, 1024, 236), "float16", "float16", rtol=1e-2
    )
    verify_batch_matmul(
        (16, 1024, 128), (1, 128, 236), (16, 1024, 236), "float16", "float16", rtol=1e-2
    )

    verify_batch_matmul((16, 1024, 128), (16, 128, 236), (16, 1024, 236), "int8", "int32")


if __name__ == "__main__":
    tvm.testing.main()
