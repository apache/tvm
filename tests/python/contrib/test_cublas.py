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
from tvm import te
import numpy as np
from tvm.contrib import cublas
from tvm.contrib import cublaslt
import tvm.testing


def verify_matmul_add(in_dtype, out_dtype, rtol=1e-5):
    n = 1024
    l = 128
    m = 236
    A = te.placeholder((n, l), name="A", dtype=in_dtype)
    B = te.placeholder((l, m), name="B", dtype=in_dtype)
    C = cublas.matmul(A, B, dtype=out_dtype)
    s = te.create_schedule(C.op)

    def verify(target="cuda"):
        if not tvm.get_global_func("tvm.contrib.cublas.matmul", True):
            print("skip because extern function is not available")
            return
        dev = tvm.cuda(0)
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


def verify_matmul_add_igemm(in_dtype, out_dtype, rtol=1e-5):
    n = 1024
    l = 1024
    m = 1024
    L = roundoff(l, 32)
    N = roundoff(n, 8)
    N_out = roundoff(n, 32)

    A = te.placeholder((N, L), name="A", dtype=in_dtype)
    B = te.placeholder((m, L), name="B", dtype=in_dtype)
    # C has CUBLASLT_ORDER_COL32 layout, thus a different shape
    C = cublaslt.matmul(A, B, False, True, m, N_out, dtype=out_dtype)
    s = te.create_schedule(C.op)

    def verify(target="cuda"):
        if not tvm.get_global_func("tvm.contrib.cublaslt.matmul", True):
            print("skip because extern function is not available")
            return
        dev = tvm.cuda(0)
        f = tvm.build(s, [A, B, C], target)
        a_old = np.random.uniform(0, 128, size=(n, l))
        b_old = np.random.uniform(0, 128, size=(l, m))

        # Transform a to become CUBLASLT_ORDER_COL4_4R2_8C layout
        a_new = np.hstack((a_old.astype(A.dtype), np.zeros([n, L - l])))
        a_new = np.vstack((a_new.astype(A.dtype), np.zeros([N - n, L])))
        a_even = np.vsplit(a_new[::2], N / 8)
        a_odd = np.vsplit(a_new[1::2], N / 8)
        a_new = [None] * (len(a_even) + len(a_odd))
        a_new[::2] = a_even
        a_new[1::2] = a_odd
        a_new = np.vstack(a_new)
        a_new = np.vstack(
            np.vstack(np.vstack(np.hsplit(i, 8)).reshape([4, 32]) for i in np.vsplit(j, N / 4))
            for j in np.hsplit(a_new, L / 32)
        )
        a_new = a_new.reshape([N, L])
        # Transform b to become CUBLASLT_ORDER_COL32 layout
        b_new = np.vstack(
            np.hsplit(np.hstack((b_old.T.astype(B.dtype), np.zeros([m, L - l]))), L / 32)
        )
        b_new = b_new.reshape([m, L])

        a = tvm.nd.array(a_new.astype(A.dtype), dev)
        b = tvm.nd.array(b_new.astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros((m, N_out), dtype=C.dtype), dev)
        f(a, b, c)
        # Transform output c from layout CUBLASLT_ORDER_COL32 to row major layout
        c_out = c.numpy()
        c_out = c_out.reshape([int(m * N_out / 32), 32])
        c_out = np.hstack(np.vsplit(c_out, int(N_out / 32)))
        c_out = c_out[:, :n]
        c_out = c_out.T
        tvm.testing.assert_allclose(
            c_out, np.dot(a_old.astype(C.dtype), b_old.astype(C.dtype)), rtol=rtol
        )

    verify()


def verify_batch_matmul(Ashape, Bshape, Cshape, in_dtype, out_dtype, rtol=1e-5):
    A = te.placeholder(Ashape, name="A", dtype=in_dtype)
    B = te.placeholder(Bshape, name="B", dtype=in_dtype)
    C = cublas.batch_matmul(A, B, dtype=out_dtype)
    s = te.create_schedule(C.op)

    dev = tvm.cuda(0)
    f = tvm.build(s, [A, B, C], "cuda")
    a = tvm.nd.array(np.random.uniform(size=Ashape).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=Bshape).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(Cshape, dtype=C.dtype), dev)
    f(a, b, c)
    tvm.testing.assert_allclose(
        c.numpy(),
        np.matmul(a.numpy().astype(C.dtype), b.numpy().astype(C.dtype)).astype(C.dtype),
        rtol=rtol,
    )


@tvm.testing.requires_cuda
def test_matmul_add():
    verify_matmul_add("float", "float", rtol=1e-3)
    verify_matmul_add("float16", "float")
    verify_matmul_add("float16", "float16", rtol=1e-2)
    verify_matmul_add("int8", "int32")


@tvm.testing.requires_cuda
def test_matmul_add_igemm():
    verify_matmul_add_igemm("int8", "int32")


@tvm.testing.requires_cuda
def test_batch_matmul():
    if not tvm.get_global_func("tvm.contrib.cublas.matmul", True):
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


if __name__ == "__main__":
    test_matmul_add()
    test_batch_matmul()
    test_matmul_add_igemm()
