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
import re
import numpy as np


@tvm.testing.requires_vulkan
def test_vector_comparison():
    target = "vulkan"

    def check_correct_assembly(dtype):
        n = (1024,)
        A = te.placeholder(n, dtype=dtype, name="A")
        B = te.compute(
            A.shape,
            lambda i: tvm.tir.Select(
                A[i] >= 0, A[i] + tvm.tir.const(1, dtype), tvm.tir.const(0, dtype)
            ),
            name="B",
        )
        s = te.create_schedule(B.op)

        (bx, tx) = s[B].split(s[B].op.axis[0], factor=128)
        (tx, vx) = s[B].split(tx, factor=4)
        s[B].bind(bx, te.thread_axis("blockIdx.x"))
        s[B].bind(tx, te.thread_axis("threadIdx.x"))
        s[B].vectorize(vx)
        f = tvm.build(s, [A, B], target)

        # Verify we generate the boolx4 type declaration and the OpSelect
        # v4{float,half,int} instruction
        assembly = f.imported_modules[0].get_source()
        matches = re.findall("%v4bool = OpTypeVector %bool 4", assembly)
        assert len(matches) == 1
        matches = re.findall("OpSelect %v4.*", assembly)
        assert len(matches) == 1

    check_correct_assembly("float32")
    check_correct_assembly("int32")
    check_correct_assembly("float16")


tx = te.thread_axis("threadIdx.x")
bx = te.thread_axis("blockIdx.x")


@tvm.testing.requires_vulkan
def test_vulkan_copy():
    def check_vulkan(dtype, n):
        A = te.placeholder((n,), name="A", dtype=dtype)
        ctx = tvm.vulkan(0)
        a_np = np.random.uniform(size=(n,)).astype(A.dtype)
        a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(a_np)
        b_np = a.asnumpy()
        tvm.testing.assert_allclose(a_np, b_np)
        tvm.testing.assert_allclose(a_np, a.asnumpy())

    for _ in range(100):
        dtype = np.random.choice(["float32", "float16", "int8", "int32"])
        logN = np.random.randint(1, 15)
        peturb = np.random.uniform(low=0.5, high=1.5)
        check_vulkan(dtype, int(peturb * (2 ** logN)))


@tvm.testing.requires_vulkan
def test_vulkan_vectorize_add():
    num_thread = 8

    def check_vulkan(dtype, n, lanes):
        A = te.placeholder((n,), name="A", dtype="%sx%d" % (dtype, lanes))
        B = te.compute((n,), lambda i: A[i] + tvm.tir.const(1, A.dtype), name="B")
        s = te.create_schedule(B.op)
        xo, xi = s[B].split(B.op.axis[0], factor=num_thread)
        s[B].bind(xo, bx)
        s[B].bind(xi, tx)
        fun = tvm.build(s, [A, B], "vulkan")
        ctx = tvm.vulkan(0)
        a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(np.random.uniform(size=(n, lanes)))
        c = tvm.nd.empty((n,), B.dtype, ctx)
        fun(a, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + 1)

    check_vulkan("float32", 64, 2)
    check_vulkan("float16", 64, 2)


@tvm.testing.requires_vulkan
def test_vulkan_stress():
    """
    Launch a randomized test with multiple kernels per stream, multiple uses of
    kernels per stream, over multiple threads.
    """
    import random
    import threading

    n = 1024
    num_thread = 64

    def run_stress():
        def worker():
            A = te.placeholder((n,), name="A", dtype="float32")
            B = te.placeholder((n,), name="B", dtype="float32")
            functions = [
                (
                    lambda: te.compute((n,), lambda i: 2 * A[i] + 3 * B[i]),
                    lambda a, b: 2 * a + 3 * b,
                ),
                (lambda: te.compute((n,), lambda i: A[i] + B[i]), lambda a, b: a + b),
                (lambda: te.compute((n,), lambda i: A[i] + 2 * B[i]), lambda a, b: a + 2 * b),
            ]

            def build_f(f_ref):
                (C_f, ref) = f_ref
                C = C_f()
                s = te.create_schedule(C.op)
                xo, xi = s[C].split(C.op.axis[0], factor=num_thread)
                s[C].bind(xo, bx)
                s[C].bind(xi, tx)
                fun = tvm.build(s, [A, B, C], "vulkan")
                return (fun, ref)

            fs = [
                build_f(random.choice(functions)) for _ in range(np.random.randint(low=1, high=10))
            ]
            ctx = tvm.vulkan(0)
            a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(np.random.uniform(size=(n,)))
            b = tvm.nd.empty((n,), B.dtype, ctx).copyfrom(np.random.uniform(size=(n,)))
            cs = [tvm.nd.empty((n,), A.dtype, ctx) for _ in fs]
            for ((f, _), c) in zip(fs, cs):
                f(a, b, c)

            for ((_, ref), c) in zip(fs, cs):
                tvm.testing.assert_allclose(c.asnumpy(), ref(a.asnumpy(), b.asnumpy()))

        ts = [threading.Thread(target=worker) for _ in range(np.random.randint(1, 10))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

    run_stress()


if __name__ == "__main__":
    test_vector_comparison()
    test_vulkan_copy()
    test_vulkan_vectorize_add()
    test_vulkan_stress()
