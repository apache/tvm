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
from tvm import testing


@tvm.register_func("tvm.test_matmul")
def my_matmul(a, b, c):
    c.copyfrom(np.dot(a.asnumpy(), b.asnumpy()))


def test_packed_func(parallel=True, target="llvm"):
    M, K, N = 4, 4, 2

    A = te.placeholder((M, K), name="A", dtype="float64")
    B = te.placeholder((K, N), name="B", dtype="float64")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    bn = 2
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    s[C].reorder(xo, yo, xi, yi, k)
    if parallel:
        s[C].parallel(xo)

    def intrin_libxsmm(m, k, n):
        a = te.placeholder((m, k), name="a", dtype="float64")
        b = te.placeholder((k, n), name="b", dtype="float64")
        k = te.reduce_axis((0, k), name="k")
        c = te.compute((m, n), lambda i, j: te.sum(a[i, k] * b[k, j], axis=k), name="c")
        a_buffer = tvm.tir.decl_buffer(
            a.shape, a.dtype, name="a_buffer", offset_factor=1, strides=[te.var("s1"), 1]
        )
        b_buffer = tvm.tir.decl_buffer(
            b.shape, b.dtype, name="b_buffer", offset_factor=1, strides=[te.var("s2"), 1]
        )
        c_buffer = tvm.tir.decl_buffer(
            c.shape, c.dtype, name="c_buffer", offset_factor=1, strides=[te.var("s3"), 1]
        )

        def intrin_func(ins, outs):
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_packed("tvm.test_matmul", ins[0], ins[1], outs[0]))
            return ib.get()

        return te.decl_tensor_intrin(
            c.op, intrin_func, binds={a: a_buffer, b: b_buffer, c: c_buffer}
        )

    micro_kernel = intrin_libxsmm(bn, K, bn)
    s[C].tensorize(xi, micro_kernel)
    func = tvm.build(s, [A, B, C], target=target)
    ctx = tvm.cpu(0)
    a = tvm.nd.array(np.random.uniform(size=(M, K)).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(K, N)).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros((M, N), dtype=C.dtype), ctx)
    func(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), rtol=1e-5)


if __name__ == "__main__":
    # Test cases for issue: https://github.com/apache/tvm/issues/7246
    test_packed_func(True, "llvm")
    test_packed_func(False, "llvm")
    test_packed_func(True, "stackvm")
    test_packed_func(False, "stackvm")
