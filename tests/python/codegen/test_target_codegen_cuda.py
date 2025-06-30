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
import pytest

import tvm
import tvm.testing
from tvm import te, topi
from tvm.contrib.nvcc import have_bf16, have_fp16, have_int8
from tvm.script import tir as T


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_vectorize_add():
    num_thread = 8

    def check_cuda(dtype, n, lanes):
        if dtype == "float16" and not have_fp16(tvm.cuda(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return
        if dtype == "int8" and not have_int8(tvm.cuda(0).compute_version):
            print("skip because gpu does not support int8")
            return
        A = te.placeholder((n,), name="A", dtype="%sx%d" % (dtype, lanes))
        B = te.compute((n,), lambda i: A[i] + tvm.tir.const(1, A.dtype), name="B")

        sch = tvm.tir.Schedule(te.create_prim_func([A, B]))
        xo, xi = sch.split(sch.get_loops("B")[0], factors=[None, num_thread])
        sch.bind(xo, "blockIdx.x")
        sch.bind(xi, "threadIdx.x")
        fun = tvm.compile(sch.mod, target="cuda")

        dev = tvm.cuda(0)
        a = tvm.nd.empty((n,), A.dtype, dev).copyfrom(np.random.uniform(size=(n, lanes)))
        c = tvm.nd.empty((n,), B.dtype, dev)
        fun(a, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + 1)

    check_cuda("float32", 64, 2)
    check_cuda("float32", 64, 3)
    check_cuda("float32", 64, 4)
    check_cuda("int8", 64, 2)
    check_cuda("int8", 64, 3)
    check_cuda("int8", 64, 4)
    check_cuda("uint8", 64, 2)
    check_cuda("uint8", 64, 3)
    check_cuda("uint8", 64, 4)
    check_cuda("float16", 64, 2)
    check_cuda("float16", 64, 4)
    check_cuda("float16", 64, 6)
    check_cuda("float16", 64, 8)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_bf16_vectorize_add():
    if not have_bf16(tvm.cuda(0).compute_version):
        print("skip because gpu does not support bf16")
        return
    num_thread = 8

    def np_float2np_bf16(arr):
        """Convert a numpy array of float to a numpy array
        of bf16 in uint16"""
        orig = arr.view("<u4")
        bias = np.bitwise_and(np.right_shift(orig, 16), 1) + 0x7FFF
        return np.right_shift(orig + bias, 16).astype("uint16")

    def np_bf162np_float(arr):
        """Convert a numpy array of bf16 (uint16) to a numpy array
        of float"""
        u32 = np.left_shift(arr.astype("uint32"), 16)
        return u32.view("<f4")

    def check_cuda(n, lanes):
        A = te.placeholder((n,), name="A", dtype="bfloat16x%d" % lanes)
        B = te.compute((n,), lambda i: A[i] + tvm.tir.const(1, A.dtype), name="B")

        sch = tvm.tir.Schedule(te.create_prim_func([A, B]))
        xo, xi = sch.split(sch.get_loops("B")[0], factors=[None, num_thread])
        sch.bind(xo, "blockIdx.x")
        sch.bind(xi, "threadIdx.x")
        with tvm.transform.PassContext(
            disabled_pass=["tir.BF16Promote", "tir.BF16CastElimination", "tir.BF16TypeLowering"]
        ):
            fun = tvm.compile(sch.mod, target="cuda")
        dev = tvm.cuda(0)
        np_a = np.random.uniform(size=(n, lanes)).astype("float32")
        np_a = np_bf162np_float(np_float2np_bf16(np_a))
        a = tvm.nd.empty((n,), A.dtype, dev).copyfrom(np_float2np_bf16(np_a))
        c = tvm.nd.empty((n,), B.dtype, dev)
        fun(a, c)
        c = tvm.nd.empty((n, lanes), "uint16", dev).copyfrom(c)
        tvm.testing.assert_allclose(c.numpy(), np_float2np_bf16(np_a + 1))

    check_cuda(64, 2)
    check_cuda(64, 4)
    check_cuda(64, 6)
    check_cuda(64, 8)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_multiply_add():
    num_thread = 8

    def check_cuda(dtype, n, lanes):
        if dtype == "int8" and not have_int8(tvm.cuda(0).compute_version):
            print("skip because gpu does not support int8")
            return
        A = te.placeholder((n,), name="A", dtype="%sx%d" % (dtype, lanes))
        B = te.placeholder((n,), name="B", dtype="%sx%d" % (dtype, lanes))
        C = te.placeholder((n,), name="C", dtype="int32")
        D = te.compute(
            (n,), lambda i: tvm.tir.call_pure_extern("int32", "__dp4a", A[i], B[i], C[i]), name="D"
        )
        sch = tvm.tir.Schedule(te.create_prim_func([A, B, C, D]))
        xo, xi = sch.split(sch.get_loops("D")[0], factors=[None, num_thread])
        sch.bind(xo, "blockIdx.x")
        sch.bind(xi, "threadIdx.x")
        fun = tvm.compile(sch.mod, target="cuda")

        np_a = np.random.randint(low=-128, high=127, size=(n, lanes))
        np_b = np.random.randint(low=-128, high=127, size=(n, lanes))
        np_c = np.random.randint(low=0, high=127, size=(n,))
        np_d = [sum(x * y) + z for x, y, z in zip(np_a, np_b, np_c)]
        dev = tvm.cuda(0)
        a = tvm.nd.empty((n,), A.dtype, dev).copyfrom(np_a)
        b = tvm.nd.empty((n,), B.dtype, dev).copyfrom(np_b)
        c = tvm.nd.empty((n,), C.dtype, dev).copyfrom(np_c)
        d = tvm.nd.empty((n,), D.dtype, dev)
        fun(a, b, c, d)
        tvm.testing.assert_allclose(d.numpy(), np_d)

    check_cuda("int8", 64, 4)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_vectorize_load():
    num_thread = 8

    def check_cuda(dtype, n, lanes):
        dev = tvm.cuda(0)
        A = te.placeholder((n,), name="A", dtype="%sx%d" % (dtype, lanes))
        B = te.compute((n,), lambda i: A[i], name="B")

        sch = tvm.tir.Schedule(te.create_prim_func([A, B]))
        xo, xi = sch.split(sch.get_loops("B")[0], factors=[None, num_thread])
        sch.bind(xo, "blockIdx.x")
        sch.bind(xi, "threadIdx.x")
        fun = tvm.compile(sch.mod, target="cuda")

        np_a = np.random.randint(low=-128, high=127, size=(n, lanes))
        a = tvm.nd.empty((n,), A.dtype, dev).copyfrom(np_a)
        b = tvm.nd.empty((n,), B.dtype, dev)
        fun(a, b)
        tvm.testing.assert_allclose(a.numpy(), b.numpy())

    check_cuda("int8", 64, 2)
    check_cuda("int8", 64, 3)
    check_cuda("int8", 64, 4)
    check_cuda("int8", 64, 8)
    check_cuda("int8", 64, 16)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_make_int8():
    def check_cuda(n, value, lanes):
        dtype = "int8"
        dev = tvm.cuda(0)
        A = te.compute((n, lanes), lambda i, j: tvm.tir.const(value, dtype=dtype), name="A")

        sch = tvm.tir.Schedule(te.create_prim_func([A]))
        y, x = sch.get_loops("A")
        sch.vectorize(x)
        sch.bind(y, "blockIdx.x")
        fun = tvm.compile(sch.mod, target="cuda")

        np_a = np.full((n, lanes), value, dtype=dtype)
        a = tvm.nd.empty(np_a.shape, dtype, dev)
        fun(a)
        np.testing.assert_equal(a.numpy(), np_a)

    check_cuda(64, np.int8(0xAB), 4)
    check_cuda(64, 0, 4)
    check_cuda(64, -3, 4)
    check_cuda(64, np.int8(0xAB), 3)
    check_cuda(64, 0, 3)
    check_cuda(64, -3, 3)
    check_cuda(64, np.int8(0xAB), 2)
    check_cuda(64, 0, 2)
    check_cuda(64, -3, 2)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_inf_nan():
    target = "cuda"

    def check_inf_nan(dev, n, value, dtype):
        A = te.placeholder((n,), name="A", dtype=dtype)
        inf_value = tvm.tir.const(value, dtype=dtype)
        C = te.compute((n,), lambda i: inf_value, name="C")

        sch = tvm.tir.Schedule(te.create_prim_func([A, C]))
        xo, xi = sch.split(sch.get_loops("C")[0], factors=[None, 8])
        sch.bind(xo, "blockIdx.x")
        sch.bind(xi, "threadIdx.x")
        fun = tvm.compile(sch.mod, target="cuda")

        a = tvm.nd.empty((n,), A.dtype, dev)
        c = tvm.nd.empty((n,), A.dtype, dev)
        # Only need to test compiling here
        fun(a, c)

    dev = tvm.device(target, 0)

    check_inf_nan(dev, 1, -float("inf"), "float32")
    check_inf_nan(dev, 1, -float("inf"), "float64")
    check_inf_nan(dev, 1, float("inf"), "float32")
    check_inf_nan(dev, 1, float("inf"), "float64")
    check_inf_nan(dev, 1, float("nan"), "float32")
    check_inf_nan(dev, 1, float("nan"), "float64")


@tvm.testing.parametrize_targets("cuda", "rocm")
def test_crossthread_reduction1(target, dev):
    n = te.var("n")
    m = te.var("m")
    A = te.placeholder((n, m), name="A")
    k = te.reduce_axis((0, m), "m")
    B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")

    def sched(nthd):
        sch = tvm.tir.Schedule(te.create_prim_func([A, B]))
        x, k = sch.get_loops("B")
        ko, _ = sch.split(k, factors=[nthd, None])
        sch.bind(ko, "threadIdx.x")
        sch.bind(x, "blockIdx.x")
        fun = tvm.compile(sch.mod, target="cuda")
        return fun

    def verify(nthd):
        func = sched(nthd)
        nn = 3
        # checks three typical cases
        vals = [nthd - 1, nthd, nthd + 1]
        for kk in [x for x in vals]:
            size = (nn, kk)
            a = tvm.nd.array(np.random.uniform(size=size).astype(A.dtype), dev)
            b = tvm.nd.array(np.zeros(nn, dtype=B.dtype), dev)
            func(a, b)
            tvm.testing.assert_allclose(b.numpy(), np.sum(a.numpy(), axis=1), rtol=1e-3)

    verify(16)
    verify(32)
    verify(64)


@tvm.testing.parametrize_targets("cuda", "rocm")
def test_crossthread_reduction2(target, dev):
    n = te.var("n")
    k0 = te.var("k0")
    k1 = te.var("k1")
    A = te.placeholder((n, k0, k1), name="A")
    k0 = te.reduce_axis((0, k0), "k0")
    k1 = te.reduce_axis((0, k1), "k1")
    B = te.compute((n,), lambda i: te.sum(A[i, k0, k1], axis=(k0, k1)), name="B")

    def sched(nthdx, nthdy):
        sch = tvm.tir.Schedule(te.create_prim_func([A, B]))
        x, k0, k1 = sch.get_loops("B")
        k0o, _ = sch.split(k0, factors=[nthdx, None])
        k1o, _ = sch.split(k1, factors=[nthdy, None])
        sch.bind(k0o, "threadIdx.x")
        sch.bind(k1o, "threadIdx.y")
        sch.bind(x, "blockIdx.x")
        func = tvm.compile(sch.mod, target="cuda")
        return func

    def verify(nthdx, nthdy):
        func = sched(nthdx, nthdy)
        nn = 3
        # checks three typical cases
        vx = [nthdx - 1, nthdx, nthdx + 1]
        vy = [nthdy - 1, nthdy, nthdy + 1]
        for kk0, kk1 in [(x, y) for x in vx for y in vy]:
            size = (nn, kk0, kk1)
            a = tvm.nd.array(np.random.uniform(size=size).astype(A.dtype), dev)
            b = tvm.nd.array(np.zeros(nn, dtype=B.dtype), dev)
            func(a, b)
            tvm.testing.assert_allclose(b.numpy(), np.sum(a.numpy(), axis=(1, 2)), rtol=1e-3)

    verify(16, 16)
    verify(32, 32)
    verify(16, 32)
    verify(32, 16)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_reduction_binding():
    k = te.reduce_axis((0, 32), "k")
    A = te.placeholder((96, 32), name="A")
    B = te.compute((96,), lambda m: te.sum(A[m, k], axis=k), name="B")

    sch = tvm.tir.Schedule(te.create_prim_func([A, B]))
    x, k = sch.get_loops("B")
    sch.reorder(k, x)
    mo, _ = sch.split(x, factors=[None, 32])
    sch.bind(mo, "blockIdx.x")
    func = tvm.compile(sch.mod, target="cuda")


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_const_float_to_half():
    # This import is required to use nvcc to perform code gen;
    # otherwise it is found that the code gen is done by nvrtc.

    shape = (2, 3, 4)
    a = te.placeholder(shape, dtype="float16", name="a")
    b = tvm.tir.const(0.5, dtype="float16")
    c = te.compute(shape, lambda i, j, k: a[i, j, k] > b, name="C")

    sch = tvm.tir.Schedule(te.create_prim_func([a, c]))
    xo, xi = sch.split(sch.fuse(*sch.get_loops("C")), factors=[None, 64])
    sch.bind(xo, "blockIdx.x")
    sch.bind(xi, "threadIdx.x")
    func = tvm.compile(sch.mod, target="cuda")

    dev = tvm.cuda(0)
    a_np = np.random.uniform(size=shape).astype(a.dtype)
    c_np = np.zeros(shape=shape, dtype=c.dtype)
    a = tvm.nd.array(a_np, dev)
    c = tvm.nd.array(c_np, dev)
    func(a, c)
    np.testing.assert_equal(c.numpy(), a_np > b.value)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_floordiv_with_vectorization():
    with tvm.target.cuda():
        # B[i] = A[floordiv(i, k)]
        n = 256
        k = 37
        A = te.placeholder((n,), name="A")
        B = te.compute((n,), lambda i: A[tvm.tir.floordiv(i, k)], name="B")

        sch = tvm.tir.Schedule(te.create_prim_func([A, B]))
        xo, xi = sch.split(sch.get_loops("B")[0], factors=[1, None])
        xio, xii = sch.split(xi, factors=[None, 4])
        sch.vectorize(xii)
        sch.bind(xo, "blockIdx.x")
        sch.bind(xio, "threadIdx.x")
        func = tvm.compile(sch.mod, target="cuda")

        dev = tvm.cuda(0)
        a_np = np.random.uniform(size=(n,)).astype(A.dtype)
        b_np = np.array([a_np[i // k] for i in range(0, n)])
        a_nd = tvm.nd.array(a_np, dev)
        b_nd = tvm.nd.array(np.zeros(b_np.shape, dtype=b_np.dtype), dev)
        func(a_nd, b_nd)
        tvm.testing.assert_allclose(b_nd.numpy(), b_np, rtol=1e-3)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_floormod_with_vectorization():
    with tvm.target.cuda():
        # B[i] = A[floormod(i, k)]
        n = 256
        k = 37
        A = te.placeholder((n,), name="A")
        B = te.compute((n,), lambda i: A[tvm.tir.floormod(i, k)], name="B")
        sch = tvm.tir.Schedule(te.create_prim_func([A, B]))
        xo, xi = sch.split(sch.get_loops("B")[0], factors=[1, None])
        xio, xii = sch.split(xi, factors=[None, 4])
        sch.vectorize(xii)
        sch.bind(xo, "blockIdx.x")
        sch.bind(xio, "threadIdx.x")
        func = tvm.compile(sch.mod, target="cuda")

        dev = tvm.cuda(0)
        a_np = np.random.uniform(size=(n,)).astype(A.dtype)
        b_np = np.array([a_np[i % k] for i in range(0, n)])
        a_nd = tvm.nd.array(a_np, dev)
        b_nd = tvm.nd.array(np.zeros(b_np.shape, dtype=b_np.dtype), dev)
        func(a_nd, b_nd)
        tvm.testing.assert_allclose(b_nd.numpy(), b_np, rtol=1e-3)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_vectorized_casts():
    def check(t0, t1, factor):
        if (t0 == "float16" or t1 == "float16") and not have_fp16(tvm.cuda(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return

        # compute
        n = 128
        A = te.placeholder((n,), dtype=t0, name="A")
        B = te.placeholder((n,), dtype=t1, name="B")
        C = te.compute((n,), lambda i: A[i] + topi.cast(B[i], A.dtype), name="C")

        # schedule
        sch = tvm.tir.Schedule(te.create_prim_func([A, B, C]))
        ob, ib = sch.split(sch.get_loops("C")[0], factors=[None, factor])
        sch.vectorize(ib)
        sch.bind(ob, "threadIdx.x")
        func = tvm.compile(sch.mod, target="cuda")

        # correctness
        dev = tvm.cuda(0)
        low, high = (0, 20) if t0.startswith("u") or t1.startswith("u") else (-10, 10)
        a_np = np.random.randint(low, high, size=n).astype(A.dtype)
        b_np = np.random.randint(low, high, size=n).astype(B.dtype)
        c_np = (a_np + b_np).astype(A.dtype)
        a_nd = tvm.nd.array(a_np, dev)
        b_nd = tvm.nd.array(b_np, dev)
        c_nd = tvm.nd.array(np.zeros(c_np.shape, dtype=c_np.dtype), dev)
        func(a_nd, b_nd, c_nd)
        tvm.testing.assert_allclose(c_nd.numpy(), c_np, rtol=1e-3)

    def skip(t0, t1):
        if t0 == t1:
            return True
        # CUDA does support cast between {u}int8 and fp16.
        skip_set = {"float16", "uint8", "int8"}
        if t0 in skip_set and t1 in skip_set:
            return True
        return False

    types_4 = [
        "float16",
        "float32",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "float64",
        "int64",
        "uint64",
    ]
    types_8 = ["float16", "float32", "int8", "uint8", "int16", "uint16", "int32", "uint32"]
    for t0, t1 in [(x, y) for x in types_4 for y in types_4 if not skip(x, y)]:
        check(t0, t1, 4)
    for t0, t1 in [(x, y) for x in types_8 for y in types_8 if not skip(x, y)]:
        check(t0, t1, 8)
    check("int8", "uint8", 16)
    check("uint8", "int8", 16)


def sched(A, B):
    # schedule
    sch = tvm.tir.Schedule(te.create_prim_func([A, B]))
    io, ii = sch.split(sch.get_loops("B")[0], factors=[1, None])
    iio, iii = sch.split(ii, factors=[32, None])
    _, iiii = sch.split(iii, factors=[None, 4])
    sch.vectorize(iiii)
    sch.bind(io, "blockIdx.x")
    sch.bind(iio, "threadIdx.x")
    return tvm.compile(sch.mod, target="cuda")


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_vectorized_intrin1():
    test_funcs = [
        (tvm.tir.floor, lambda x: np.floor(x)),
        (tvm.tir.ceil, lambda x: np.ceil(x)),
        (tvm.tir.trunc, lambda x: np.trunc(x)),
        (tvm.tir.abs, lambda x: np.fabs(x)),
        (tvm.tir.round, lambda x: np.round(x)),
        (tvm.tir.exp, lambda x: np.exp(x)),
        (tvm.tir.exp2, lambda x: np.exp2(x)),
        (tvm.tir.exp10, lambda x: np.power(10, x)),
        (tvm.tir.log, lambda x: np.log(x)),
        (tvm.tir.log2, lambda x: np.log2(x)),
        (tvm.tir.log10, lambda x: np.log10(x)),
        (tvm.tir.tan, lambda x: np.tan(x)),
        (tvm.tir.cos, lambda x: np.cos(x)),
        (tvm.tir.cosh, lambda x: np.cosh(x)),
        (tvm.tir.sin, lambda x: np.sin(x)),
        (tvm.tir.sinh, lambda x: np.sinh(x)),
        (tvm.tir.atan, lambda x: np.arctan(x)),
        (tvm.tir.tanh, lambda x: np.tanh(x)),
        (tvm.tir.sqrt, lambda x: np.sqrt(x)),
    ]

    def run_test(tvm_intrin, np_func, dtype):
        if dtype == "float16" and not have_fp16(tvm.cuda(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return
        # set of intrinsics does not support fp16 yet.
        skip_set = {
            tvm.tir.abs,
            tvm.tir.round,
            tvm.tir.tan,
            tvm.tir.atan,
            tvm.tir.tanh,
            tvm.tir.cosh,
            tvm.tir.sinh,
        }
        if dtype == "float16" and tvm_intrin in skip_set:
            print("Skip because '{0}' does not support fp16 yet".format(tvm_intrin.__name__))
            return

        n = 128
        A = te.placeholder((n,), dtype=dtype, name="A")
        B = te.compute((n,), lambda *i: tvm_intrin(A(*i)), name="B")
        f = sched(A, B)
        dev = tvm.cuda(0)
        a = tvm.nd.array(np.random.uniform(0, 1, size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(shape=(n,)).astype(A.dtype), dev)
        f(a, b)
        tvm.testing.assert_allclose(b.numpy(), np_func(a.numpy()), atol=1e-3, rtol=1e-3)

    for func in test_funcs:
        run_test(*func, "float32")
        run_test(*func, "float16")


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_vectorized_intrin2(dtype="float32"):
    c2 = tvm.tir.const(2, dtype=dtype)
    test_funcs = [
        (tvm.tir.power, lambda x: np.power(x, 2.0)),
        (tvm.tir.fmod, lambda x: np.fmod(x, 2.0)),
    ]

    def run_test(tvm_intrin, np_func):
        n = 128
        A = te.placeholder((n,), dtype=dtype, name="A")
        B = te.compute((n,), lambda i: tvm_intrin(A[i], c2), name="B")
        f = sched(A, B)
        dev = tvm.cuda(0)
        a = tvm.nd.array(np.random.uniform(0, 1, size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(shape=(n,)).astype(A.dtype), dev)
        f(a, b)
        tvm.testing.assert_allclose(b.numpy(), np_func(a.numpy()), atol=1e-3, rtol=1e-3)

    for func in test_funcs:
        run_test(*func)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_vectorized_popcount():
    def ref_popcount(x):
        cnt = 0
        while x:
            x -= x & -x
            cnt += 1
        return cnt

    def run_test(dtype):
        n = 128
        A = te.placeholder((n,), dtype=dtype, name="A")
        B = te.compute((n,), lambda i: tvm.tir.popcount(A[i]), name="B")
        f = sched(A, B)
        dev = tvm.cuda(0)
        a = tvm.nd.array(np.random.randint(0, 100000, size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(shape=(n,)).astype(B.dtype), dev)
        f(a, b)
        ref = np.vectorize(ref_popcount)(a.numpy())
        tvm.testing.assert_allclose(b.numpy(), ref)

    run_test("uint32")
    run_test("uint64")


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_vectorize_load_permute_pad():
    def check_cuda(dtype, n, l, padding, lanes):
        if dtype == "float16" and not have_fp16(tvm.cuda(0).compute_version):
            print("Skip because gpu does not have fp16 support")
            return

        dev = tvm.cuda(0)
        A = tvm.te.placeholder((n, l), name="A", dtype=dtype)
        B = tvm.te.compute(
            (n // lanes, l + 2 * padding, lanes),
            lambda i, j, k: tvm.te.if_then_else(
                tvm.te.any(j < padding, j >= l + padding),
                tvm.tir.const(0, dtype),
                A[i * lanes + k, j - padding],
            ),
            name="B",
        )

        sch = tvm.tir.Schedule(te.create_prim_func([A, B]))
        block, thread, vectorize = sch.get_loops("B")
        sch.bind(block, "blockIdx.x")
        sch.bind(thread, "threadIdx.x")
        sch.vectorize(vectorize)
        fun = tvm.compile(sch.mod, target="cuda")

        np_a = np.random.randint(low=-128, high=127, size=(n, l)).astype(A.dtype)
        a = tvm.nd.empty((n, l), A.dtype, dev).copyfrom(np_a)
        b = tvm.nd.empty((n // lanes, l + padding * 2, lanes), B.dtype, dev)
        fun(a, b)
        np_a_reshape = np_a.reshape(n // lanes, lanes, l).transpose(0, 2, 1)
        ref = np.pad(
            np_a_reshape, ((0, 0), (padding, padding), (0, 0)), mode="constant", constant_values=0
        )
        tvm.testing.assert_allclose(b.numpy(), ref)

    check_cuda("int8", 64, 16, 3, 2)
    check_cuda("uint8", 64, 16, 3, 2)
    check_cuda("int8", 64, 16, 3, 4)
    check_cuda("uint8", 64, 16, 3, 4)
    check_cuda("int32", 64, 16, 3, 4)
    check_cuda("float16", 64, 16, 3, 4)
    check_cuda("float32", 64, 16, 3, 4)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_try_unaligned_vector_load():
    def get_compute(N, C_N, offset):
        A = te.placeholder((N,), name="A", dtype="float16")
        C = te.compute((C_N,), lambda i: A[i + offset], name="C")
        return N, C_N, A, C

    def get_compute_unaligned():
        return get_compute(3, 2, 1)

    def get_compute_aligned():
        return get_compute(4, 2, 2)

    def build(A, C, N, C_N):
        sch = tvm.tir.Schedule(te.create_prim_func([A, C]))
        oi, ii = sch.split(sch.get_loops("C")[0], factors=[None, 2])
        sch.bind(oi, "threadIdx.x")
        sch.vectorize(ii)  # BUG: misalignment

        f = tvm.tir.build(sch.mod, target="cuda")

        kernel_source = f.imported_modules[0].get_source()
        dev = tvm.cuda()
        a_data = np.arange(0, N).astype(A.dtype)
        a = tvm.nd.array(a_data, dev)
        c = tvm.nd.array(np.zeros(C_N, dtype=C.dtype), dev)
        f(a, c)

        return a_data, c.numpy(), kernel_source

    N, C_N, A, C = get_compute_unaligned()
    a_data, c, kernel_source = build(A, C, N, C_N)
    # (uint1*)(A + (1)) is invalid
    assert "A + (1)" not in kernel_source

    expected = a_data[1 : C_N + 1]
    assert np.allclose(c, expected), f"expected={expected}\nactual={c}"

    N, C_N, A, C = get_compute_aligned()
    a_data, c, kernel_source = build(A, C, N, C_N)
    # (uint1*)(A + (2)) is a valid vector load
    assert "A + 2" in kernel_source

    expected = a_data[2 : C_N + 2]
    assert np.allclose(c, expected), f"expected={expected}\nactual={c}"


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_thread_sync_inside_condition():
    @T.prim_func
    def func1(A: T.Buffer((4, 4), "float32")) -> None:
        A_shared = T.alloc_buffer((4, 4), "float32", scope="shared")
        for bx in T.thread_binding(1, "blockIdx.x"):
            for tx in T.thread_binding(32, "threadIdx.x"):
                if A[0, 0] > 1.0:
                    for i, j in T.grid(4, 4):
                        A_shared[i, j] = A[i, j]
                    for i, j in T.grid(4, 4):
                        A[i, j] = A_shared[i, j] + 1.0

    @T.prim_func
    def func2(A: T.Buffer((4, 4), "float32")) -> None:
        A_shared = T.alloc_buffer((4, 4), "float32", scope="shared")
        for bx in T.thread_binding(1, "blockIdx.x"):
            for tx in T.thread_binding(32, "threadIdx.x"):
                if T.tvm_thread_invariant(A[0, 0] > 1.0):
                    for i, j in T.grid(4, 4):
                        A_shared[i, j] = A[i, j]
                    for i, j in T.grid(4, 4):
                        A[i, j] = A_shared[i, j] + 1.0

    @T.prim_func
    def func3(A: T.Buffer((4, 4), "float32")) -> None:
        A_shared = T.alloc_buffer((4, 4), "float32", scope="shared")
        for bx in T.thread_binding(1, "blockIdx.x"):
            for tx in T.thread_binding(32, "threadIdx.x"):
                while T.tvm_thread_invariant(A[0, 0] > 1.0):
                    for i, j in T.grid(4, 4):
                        A_shared[i, j] = A[i, j]
                    for i, j in T.grid(4, 4):
                        A[i, j] = A_shared[i, j] + 1.0

    mod = tvm.IRModule({"main": func1})
    with pytest.raises(tvm.error.InternalError):
        tvm.compile(mod, target="cuda")

    mod = tvm.IRModule({"main": func2})
    tvm.compile(mod, target="cuda")

    mod = tvm.IRModule({"main": func3})
    tvm.compile(mod, target="cuda")


@tvm.testing.requires_cuda
def test_invalid_reinterpret():
    @T.prim_func
    def func(A: T.Buffer((4,), "uint32"), B: T.Buffer((4,), "uint8")) -> None:
        for tx in T.thread_binding(4, "threadIdx.x"):
            B[tx] = T.call_intrin("uint8", "tir.reinterpret", A[tx])

    with pytest.raises(tvm.error.TVMError):
        tvm.compile(func, target="cuda")


@tvm.testing.requires_cuda
@tvm.testing.requires_cuda_compute_version(9)
def test_cuda_tensormap():
    # fmt: off
    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (16, 16), dtype="float32", align=16)

        A_map: T.handle("tensormap") = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapInit", A_map, "float32", 2, A.data,
                      16, 16, 64, 16, 16, 1, 1, 0, 0, 0, 0)

        for blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                if threadIdx == 0:
                    A[0, 0] = T.reinterpret("float64", A_map)
    # fmt: on

    mod = tvm.IRModule({"main": main})
    mod = tvm.compile(mod, target="cuda")
    assert (
        """
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ A, const __grid_constant__ CUtensorMap A_map) {
  if (((int)threadIdx.x) == 0) {
    A[0] = ((float)(*(double *)(&(A_map))));
  }
}""".strip()
        in mod.mod.imported_modules[0].get_source()
    )


if __name__ == "__main__":
    tvm.testing.main()
