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

import re
import sys

import numpy as np

import tvm
import tvm.testing
from tvm import relay, te
from tvm.topi.math import cast


def check_mod(mod, x_np, res_np):
    target = "vulkan"
    dev = tvm.device(target, 0)
    ex = relay.create_executor("vm", mod=mod, device=dev, target=target)
    res = ex.evaluate()(x_np).numpy()
    tvm.testing.assert_allclose(res, res_np, atol=1e-5)


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
        dev = tvm.vulkan(0)
        a_np = np.random.uniform(size=(n,)).astype(A.dtype)
        a = tvm.nd.empty((n,), A.dtype, dev).copyfrom(a_np)
        b_np = a.numpy()
        tvm.testing.assert_allclose(a_np, b_np)
        tvm.testing.assert_allclose(a_np, a.numpy())

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
        dev = tvm.vulkan(0)
        a = tvm.nd.empty((n,), A.dtype, dev).copyfrom(np.random.uniform(size=(n, lanes)))
        c = tvm.nd.empty((n,), B.dtype, dev)
        fun(a, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + 1)

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
            dev = tvm.vulkan(0)
            a = tvm.nd.empty((n,), A.dtype, dev).copyfrom(np.random.uniform(size=(n,)))
            b = tvm.nd.empty((n,), B.dtype, dev).copyfrom(np.random.uniform(size=(n,)))
            cs = [tvm.nd.empty((n,), A.dtype, dev) for _ in fs]
            for ((f, _), c) in zip(fs, cs):
                f(a, b, c)

            for ((_, ref), c) in zip(fs, cs):
                tvm.testing.assert_allclose(c.numpy(), ref(a.numpy(), b.numpy()))

        ts = [threading.Thread(target=worker) for _ in range(np.random.randint(1, 10))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

    run_stress()


@tvm.testing.requires_vulkan
def test_vulkan_bool_load():
    def do_copy(A, B, n):
        ib = tvm.tir.ir_builder.create()
        A = ib.buffer_ptr(A)
        B = ib.buffer_ptr(B)

        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")

        max_threads = 32
        ib.scope_attr(bx, "thread_extent", tvm.tir.indexdiv(n + max_threads - 1, max_threads))
        ib.scope_attr(tx, "thread_extent", max_threads)
        tid = bx * max_threads + tx

        with ib.if_scope(tid < n):
            B[tid] = cast(A[tid], "int32")

        return ib.get()

    n = 1024
    A = te.placeholder((n,), name="A", dtype="bool")
    B = te.placeholder((n,), name="B", dtype="int32")

    target = "vulkan"

    B = te.extern(
        A.shape,
        [A],
        lambda ins, outs: do_copy(ins[0], outs[0], n),
        name="bool_copy_ir",
        dtype="int32",
    )
    s = te.create_schedule(B.op)

    with tvm.transform.PassContext(opt_level=3):
        func = tvm.build(s, [A, B], target)

    dev = tvm.device(target, 0)
    a_np = np.random.uniform(size=n) > 0.5
    b_np = np.zeros((n,), dtype="int32")
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    func(a, b)
    ref = a_np.astype(np.int32)
    tvm.testing.assert_allclose(b.numpy(), ref)


@tvm.testing.requires_vulkan
def test_vulkan_pushconstants():
    # Three 32 bit pushconstants: any_dim, stride, stride
    dtype = "float32"
    x = relay.var("x", shape=(relay.Any(),), dtype=dtype)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], relay.sqrt(x))
    x_np = np.random.uniform(size=(10,)).astype(dtype)
    res_np = np.sqrt(x_np)

    check_mod(mod, x_np, res_np)

    # One 64 bit and one 32 bit constants
    dtype = "int32"
    x = relay.var("x", shape=(relay.Any(),), dtype=dtype)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], relay.argsort(x))
    x_np = np.random.randint(0, high=10, size=(10,)).astype(dtype)
    res_np = np.argsort(x_np)

    check_mod(mod, x_np, res_np)

    # One 64 bit and one 32 bit constants
    dtype = "int32"
    x = relay.var("x", shape=(relay.Any(),), dtype=dtype)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], relay.cumsum(x))
    x_np = np.random.randint(0, high=10, size=(10,)).astype(dtype)
    res_np = np.cumsum(x_np)

    check_mod(mod, x_np, res_np)


@tvm.testing.requires_vulkan
def test_vulkan_unique():
    dtype = "int32"
    x = relay.var("x", shape=(relay.Any(),), dtype=dtype)
    mod = tvm.IRModule()
    [unique, _, _, num_unique] = relay.unique(x, is_sorted=True)
    mod["main"] = relay.Function([x], relay.op.strided_slice(unique, begin=[0], end=num_unique))
    x_np = np.random.randint(0, high=10, size=(10,)).astype(dtype)
    res_np = np.unique(x_np)
    check_mod(mod, x_np, res_np)


@tvm.testing.requires_vulkan
def test_vulkan_constant_passing():
    target = "vulkan"

    def test_scalar_params(num_int_params):
        n = te.var("n")
        scalars = [te.var("scale{}".format(i)) for i in range(num_int_params)]
        scalar_sum = scalars[0]
        for s in scalars[1:]:
            scalar_sum += s

        A = te.placeholder((n,), name="A")
        B = te.compute(A.shape, lambda i: scalar_sum + A[i], name="B")

        s = te.create_schedule(B.op)
        xo, xi = s[B].split(B.op.axis[0], factor=64)
        s[B].bind(xo, bx)
        s[B].bind(xi, tx)
        f_add = tvm.build(s, scalars + [A, B], target)

        n = 1024
        scalars = [1 for _ in scalars]
        dev = tvm.vulkan(0)
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        f_add(*scalars, a, b)

        tvm.testing.assert_allclose(a.numpy() + sum(scalars), b.numpy())

    # f_add has 3+num_int_params scalar parameters.  The other three
    # are length_n, stride1, and stride2.

    # 4 params, 32 bytes.  Within 128-byte spec-guaranteed size of
    # push constants.  Uses push constants.
    test_scalar_params(1)

    # 24 params, 192 bytes.  Too big for push constants, uses uniform
    # buffer.
    test_scalar_params(20)

    # 2047 params, 16376 bytes, just below 16kB of uniform buffer
    # space guaranteed by the vulkan spec.
    test_scalar_params(2044)


@tvm.testing.parametrize_targets("vulkan")
def test_vulkan_while_if(target, dev):
    def do_compute(A, B, n):
        ib = tvm.tir.ir_builder.create()
        A = ib.buffer_ptr(A)
        B = ib.buffer_ptr(B)

        ib.scope_attr(te.thread_axis("blockIdx.x"), "thread_extent", 0)

        iterations = ib.allocate("int32", (1,), name="iterations", scope="local")
        iterations[0] = 0
        B[0] = 0

        # WhileNode's condition is re-evaluated every loop.  The
        # if_then_else block introduces additional labels/blocks that
        # must be kept separate from the WhileNode's block.
        loop_condition = iterations[0] < tvm.tir.if_then_else(A[0] > 0, 10, 20)
        with ib.while_loop(loop_condition):
            iterations[0] += 1
            B[0] += iterations[0]

        return ib.get()

    n = 1
    dtype = "int32"
    A = te.placeholder((n,), name="A", dtype=dtype)

    B = te.extern(
        A.shape,
        [A],
        lambda ins, outs: do_compute(ins[0], outs[0], n),
        dtype=dtype,
    )
    s = te.create_schedule(B.op)

    # Point of failure would be here, at tvm.build.
    with tvm.transform.PassContext(opt_level=3):
        func = tvm.build(s, [A, B], target)

    a = tvm.nd.array(np.array([5], dtype=A.dtype), dev)
    b = tvm.nd.array(np.zeros(n, dtype=A.dtype), dev)
    func(a, b)
    tvm.testing.assert_allclose(b.numpy(), [55])

    a = tvm.nd.array(np.array([-5], dtype=A.dtype), dev)
    b = tvm.nd.array(np.zeros(n, dtype=A.dtype), dev)
    func(a, b)
    tvm.testing.assert_allclose(b.numpy(), [210])


@tvm.testing.parametrize_targets("vulkan")
def test_vulkan_local_threadidx(target, dev):
    # To access the thread index, the vulkan runtime accesses a global
    # array of thread indices, storing the result in a local variable.
    # In CUDA, these are the built-in threadIdx.x variables, which are
    # globally accessible.  In vulkan, these local variables must be
    # defined inside a function, but are hoisted up to the function
    # header to mimic the global CUDA semantics.  Before this
    # hoisting, this test could trigger spvValidate errors for
    # potentially undeclared variables.

    def do_compute(A, B, n):
        ib = tvm.tir.ir_builder.create()
        A = ib.buffer_ptr(A)
        B = ib.buffer_ptr(B)

        # One single declaration of te.thread_axis.
        tx = te.thread_axis("threadIdx.x")

        with ib.for_range(0, 1):
            # Used inside a for-loop scope, defines local thread_id
            # variable.
            ib.scope_attr(tx, "thread_extent", 16)
            B[tx + 0] = A[tx + 0]

        with ib.for_range(0, 1):
            # Used in next scope.  If local variable defined at point
            # of use instead of function header, will fail spvValidate
            # for access of out-of-scope local variable.
            ib.scope_attr(tx, "thread_extent", 16)
            B[tx + 16] = A[tx + 16]

        return ib.get()

    n = te.var("n")
    A = te.placeholder((n,), name="A", dtype="int32")
    B = te.placeholder((n,), name="B", dtype="int32")

    B = te.extern(
        A.shape,
        [A],
        lambda ins, outs: do_compute(ins[0], outs[0], n),
        dtype="int32",
    )
    s = te.create_schedule(B.op)

    # Expected failure occurs at build step.
    func = tvm.build(s, [A, B], target)

    n = 32
    a_np = np.arange(n).astype(dtype=A.dtype)
    b_np = np.zeros((n,), dtype="int32")
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    func(a, b)
    tvm.testing.assert_allclose(b.numpy(), a_np)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
