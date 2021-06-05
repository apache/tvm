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
from tvm.contrib import nvcc
import numpy as np
import time
import tvm.testing


@tvm.testing.requires_gpu
def test_exp():
    # graph
    n = tvm.runtime.convert(1024)
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: te.exp(A(*i)), name="B")
    s = te.create_schedule(B.op)
    # create iter var and assign them tags.
    num_thread = 8
    bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
    s[B].bind(bx, te.thread_axis("blockIdx.x"))
    s[B].bind(tx, te.thread_axis("threadIdx.x"))

    # one line to build the function.
    def check_device(device, host="stackvm"):
        if not tvm.testing.device_enabled(host):
            return
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled.." % device)
            return
        fexp = tvm.build(s, [A, B], device, host, name="myexp")
        dev = tvm.device(device, 0)
        # launch the kernel.
        n = 1024
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        fexp(a, b)
        tvm.testing.assert_allclose(b.numpy(), np.exp(a.numpy()), rtol=1e-5)

    check_device("opencl -device=intel_graphics")
    check_device("cuda", "llvm")
    check_device("vulkan")


@tvm.testing.requires_gpu
def test_fmod():
    # graph
    def run(dtype):
        n = te.size_var("n")
        A = te.placeholder((n,), name="A", dtype=dtype)
        B = te.placeholder((n,), name="B", dtype=dtype)
        C = te.compute(A.shape, lambda *i: te.fmod(A(*i), B(*i)), name="C")
        s = te.create_schedule(C.op)
        # create iter var and assign them tags.
        num_thread = 8
        bx, tx = s[C].split(C.op.axis[0], factor=num_thread)

        def check_device(device):
            dev = tvm.device(device, 0)
            if not tvm.testing.device_enabled(device):
                print("skip because %s is not enabled.." % device)
                return
            target = tvm.target.Target(device)
            if "cpu" not in target.keys:
                s[C].bind(bx, te.thread_axis("blockIdx.x"))
                s[C].bind(tx, te.thread_axis("threadIdx.x"))
            fmod = tvm.build(s, [A, B, C], device, name="myfmod")

            # launch the kernel.
            n = 1024
            a_np = (np.random.uniform(size=n) * 256).astype(A.dtype)
            b_np = (np.random.uniform(size=n) * 256).astype(B.dtype)

            # "fix" the values in a and b to avoid the result being too small
            b_np += (b_np < 2.0) * 2
            a_np[np.abs(np.fmod(a_np, b_np)) < 1] += 1

            a = tvm.nd.array(a_np, dev)
            b = tvm.nd.array(b_np, dev)
            c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
            ftimer = fmod.time_evaluator(fmod.entry_name, dev, number=1)
            tcost = ftimer(a, b, c).mean
            # fmod(a, b, c)
            np.testing.assert_allclose(c.numpy(), np.mod(a.numpy(), b.numpy()), rtol=1e-5)

        check_device("cuda")
        check_device("opencl -device=intel_graphics")
        check_device("metal")

    run("float32")


@tvm.testing.requires_gpu
def test_multiple_cache_write():
    # graph
    n = tvm.runtime.convert(1024)
    A0 = te.placeholder((n,), name="A0", dtype="float32")
    A1 = te.placeholder((n,), name="A1", dtype="float32")
    B0, B1 = te.compute((n,), lambda *i: (A0(*i) + A1(*i), A0(*i) * A1(*i)), name="B")
    C = te.compute((n,), lambda *i: B0(*i) + B1(*i), name="C")
    s = te.create_schedule(C.op)
    # create iter var and assign them tags.
    num_thread = 8
    B0_cache, B1_cache = s.cache_write([B0, B1], "local")
    bx, tx = s[C].split(C.op.axis[0], factor=num_thread)
    s[B0].compute_at(s[C], bx)
    s[B0_cache].compute_at(s[C], bx)
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))
    # one line to build the function.
    def check_device(device, host="stackvm"):
        if not tvm.testing.device_enabled(host):
            return
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            return
        func = tvm.build(s, [A0, A1, C], device, host, name="multiple_cache_write")
        dev = tvm.device(device, 0)
        # launch the kernel.
        n = 1024
        a0 = tvm.nd.array(np.random.uniform(size=n).astype(A0.dtype), dev)
        a1 = tvm.nd.array(np.random.uniform(size=n).astype(A1.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        func(a0, a1, c)
        tvm.testing.assert_allclose(
            c.numpy(), a0.numpy() + a1.numpy() + (a0.numpy() * a1.numpy()), rtol=1e-5
        )

    check_device("cuda", "llvm")
    check_device("vulkan")
    check_device("opencl")


def test_log_pow_llvm():
    # graph
    n = te.size_var("n")
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: te.power(te.log(A(*i)), 2.0), name="B")
    s = te.create_schedule(B.op)
    # create iter var and assign them tags.
    bx, tx = s[B].split(B.op.axis[0], factor=32)
    # one line to build the function.
    if not tvm.testing.device_enabled("llvm"):
        return

    flog = tvm.build(s, [A, B], "llvm", name="mylog")
    dev = tvm.cpu(0)
    # launch the kernel.
    n = 1028
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
    repeat = 10
    ftimer = flog.time_evaluator(flog.entry_name, dev, number=1, repeat=repeat)
    res = ftimer(a, b)
    assert len(res.results) == repeat
    tvm.testing.assert_allclose(b.numpy(), np.power(np.log(a.numpy()), 2.0), rtol=1e-5)


@tvm.testing.uses_gpu
def test_popcount():
    def run(dtype):
        # graph
        n = tvm.runtime.convert(1024)
        A = te.placeholder((n,), name="A", dtype=dtype)
        B = te.compute(A.shape, lambda *i: tvm.tir.popcount(A(*i)), name="B")
        s = te.create_schedule(B.op)
        # simple schedule
        num_thread = 8
        bx, tx = s[B].split(B.op.axis[0], factor=num_thread)

        def check_device(device):
            dev = tvm.device(device, 0)
            if not tvm.testing.device_enabled(device):
                print("skip because %s is not enabled.." % device)
                return
            target = tvm.target.Target(device)
            if "cpu" not in target.keys:
                s[B].bind(bx, te.thread_axis("blockIdx.x"))
                s[B].bind(tx, te.thread_axis("threadIdx.x"))
            func = tvm.build(s, [A, B], device)
            # launch the kernel.
            n = 1024
            a = tvm.nd.array(np.random.randint(low=0, high=1000, size=n, dtype=A.dtype), dev)
            b = tvm.nd.array(np.zeros(shape=n, dtype=B.dtype), dev)
            func(a, b)
            tvm.testing.assert_allclose(
                b.numpy(), list(map(lambda x: bin(x).count("1"), a.numpy())), rtol=1e-5
            )

        check_device("llvm")
        check_device("cuda")
        check_device("opencl")
        if dtype == "uint32":
            check_device("metal")
            check_device("vulkan")

    run("uint32")
    run("uint64")


@tvm.testing.requires_gpu
def test_add():
    def run(dtype):
        # graph
        n = te.size_var("n")
        A = te.placeholder((n,), name="A", dtype=dtype)
        B = te.placeholder((n,), name="B", dtype=dtype)
        bias = te.var("bias", dtype=dtype)
        scale = te.var("scale", dtype=dtype)
        C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
        # schedule
        s = te.create_schedule(C.op)
        # create iter var and assign them tags.
        num_thread = 16
        bx, x = s[C].split(C.op.axis[0], factor=num_thread * 4)
        tx, x = s[C].split(x, nparts=num_thread)
        _, x = s[C].split(x, factor=4)
        s[C].bind(bx, te.thread_axis("blockIdx.x"))
        s[C].bind(tx, te.thread_axis("threadIdx.x"))
        s[C].vectorize(x)

        # one line to build the function.
        def check_device(device):
            dev = tvm.device(device, 0)
            if not tvm.testing.device_enabled(device):
                print("skip because %s is not enabled.." % device)
                return
            fadd = tvm.build(s, [A, B, C], device, name="myadd")

            # launch the kernel.
            n = 1024
            a = tvm.nd.array((np.random.uniform(size=n) * 256).astype(A.dtype), dev)
            b = tvm.nd.array((np.random.uniform(size=n) * 256).astype(B.dtype), dev)
            c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
            ftimer = fadd.time_evaluator(fadd.entry_name, dev, number=1)
            tcost = ftimer(a, b, c).mean
            tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy(), rtol=1e-6)

        check_device("opencl")
        check_device("cuda")
        if dtype == "float32":
            check_device("metal")
            check_device("vulkan")

    run("float32")
    run("int32")
    run("int64")
    run("uint64")


@tvm.testing.requires_gpu
def try_warp_memory():
    """skip this in default test because it require higher arch"""
    m = 128
    A = te.placeholder((m,), name="A")
    B = te.compute((m,), lambda i: A[i] + 3, name="B")
    warp_size = 32
    s = te.create_schedule(B.op)
    AA = s.cache_read(A, "warp", [B])
    xo, xi = s[B].split(B.op.axis[0], warp_size * 2)
    xi0, xi1 = s[B].split(xi, factor=warp_size)
    tx = te.thread_axis("threadIdx.x")
    s[B].bind(xi1, tx)
    s[B].bind(xo, te.thread_axis("blockIdx.x"))
    s[AA].compute_at(s[B], xo)
    xo, xi = s[AA].split(s[AA].op.axis[0], warp_size)
    s[AA].bind(xi, tx)

    @tvm.register_func
    def tvm_callback_cuda_compile(code):
        ptx = nvcc.compile_cuda(code, target="ptx")
        return ptx

    # one line to build the function.
    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled.." % device)
            return
        f = tvm.build(s, [A, B], device)
        a = tvm.nd.array((np.random.uniform(size=m) * 256).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(m, dtype=B.dtype), dev)
        f(a, b)
        tvm.testing.assert_allclose(b.numpy(), a.numpy() + 3, rtol=1e-6)

    check_device("cuda")


if __name__ == "__main__":
    test_exp()
    try_warp_memory()
    test_multiple_cache_write()
    test_add()
    test_log_pow_llvm()
    test_popcount()
    test_fmod()
