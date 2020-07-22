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
import topi
from tvm.contrib import util, clang
import numpy as np
import ctypes
import math


def test_nearbyint():
    m = te.var("m",)
    A = te.placeholder((m,), name='A')
    A_rounded = te.compute((m,), lambda *i: tvm.tir.nearbyint(A(*i)), name='A')
    s = te.create_schedule(A_rounded.op)
    f = tvm.build(s, [A, A_rounded], "llvm")
    ctx = tvm.cpu(0)
    n = 10
    a = tvm.nd.array(np.random.uniform(high=100, size=n).astype(A.dtype), ctx)
    a_rounded = tvm.nd.array( \
            np.random.uniform(size=n).astype(A_rounded.dtype), ctx)
    f(a, a_rounded)
    # Note that numpys rint rounds to nearest integer with
    # ties to halfway is broken by rounding to even.
    # So that 1.5 and 2.5 will round 2.
    # This is the default rounding mode with libc as well.
    # However one can set a different rounding mode and in that
    # case numpy result might differ.
    tvm.testing.assert_allclose(
        a_rounded.asnumpy(), np.rint(a.asnumpy()))

def test_round_intrinsics_on_int():
    i = tvm.te.var("i", 'int32')
    for op in [tvm.tir.round, tvm.tir.trunc, tvm.tir.ceil,
                            tvm.tir.floor, tvm.tir.nearbyint]:
        assert op(tvm.tir.const(10,'int32')).value == 10
        assert op(tvm.tir.const(True,'bool')).value == True
        assert op(i).same_as(i)

    assert tvm.tir.isnan(tvm.tir.const(10, 'int32')).value == False


def test_unary_intrin():
    test_funcs = [
        (tvm.tir.exp10, lambda x : np.power(10, x)),
        (tvm.tir.log2, lambda x : np.log2(x)),
        (tvm.tir.log10, lambda x : np.log10(x)),
        (tvm.tir.sinh, lambda x : np.sinh(x)),
        (tvm.tir.cosh, lambda x : np.cosh(x)),
        (tvm.tir.log1p, lambda x : np.log1p(x)),
        (tvm.tir.asin, lambda x : np.arcsin(x)),
        (tvm.tir.acos, lambda x : np.arccos(x)),
        (tvm.tir.atan, lambda x : np.arctan(x)),
        (tvm.tir.asinh, lambda x : np.arcsinh(x)),
        (tvm.tir.acosh, lambda x : np.arccosh(x)),
        (tvm.tir.atanh, lambda x : np.arctanh(x)),
    ]
    def run_test(tvm_intrin, np_func):
        m = te.var("m",)
        A = te.placeholder((m,), name='A')
        B = te.compute((m,), lambda *i: tvm_intrin(A(*i)), name='B')
        s = te.create_schedule(B.op)
        f = tvm.build(s, [A, B], "llvm")
        ctx = tvm.cpu(0)
        n = 10
        a = tvm.nd.array(np.random.uniform(0.1, 0.5, size=n).astype(A.dtype), ctx)
        b = tvm.nd.array( \
            np.random.uniform(size=n).astype(A.dtype), ctx)
        f(a, b)
        tvm.testing.assert_allclose(
            b.asnumpy(), np_func(a.asnumpy()), atol=1e-5, rtol=1e-5)
    
    for func in test_funcs:
        run_test(*func)


def test_binary_intrin():
    test_funcs = [
        (tvm.tir.atan2, lambda x1, x2 : np.arctan2(x1, x2)),
        (tvm.tir.nextafter, lambda x1, x2 : np.nextafter(x1, x2)),
        (tvm.tir.copysign, lambda x1, x2 : np.copysign(x1, x2)),
        (tvm.tir.hypot, lambda x1, x2 : np.hypot(x1, x2)),
    ]
    def run_test(tvm_intrin, np_func):
        m = te.var("m",)
        A = te.placeholder((m,), name='A')
        B = te.placeholder((m,), name='B')
        C = te.compute((m,), lambda *i: tvm_intrin(A(*i), B(*i)), name='C')
        s = te.create_schedule(C.op)
        f = tvm.build(s, [A, B, C], "llvm")
        ctx = tvm.cpu(0)
        n = 10
        a = tvm.nd.array(np.random.uniform(0, 1, size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(0, 1, size=n).astype(B.dtype), ctx)
        c = tvm.nd.array( \
            np.random.uniform(size=n).astype(A.dtype), ctx)
        f(a, b, c)
        tvm.testing.assert_allclose(
            c.asnumpy(), np_func(a.asnumpy(), b.asnumpy()), atol=1e-5, rtol=1e-5)

    for func in test_funcs:
        run_test(*func)


def test_ldexp():
    m = te.var("m",)
    A = te.placeholder((m,), name='A')
    B = te.placeholder((m,), name='B', dtype="int32")
    C = te.compute((m,), lambda *i: tvm.tir.ldexp(A(*i), B(*i)), name='C')
    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], "llvm")
    ctx = tvm.cpu(0)
    n = 10
    a = tvm.nd.array(np.random.uniform(0, 1, size=n).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.randint(0, 5, size=n).astype(B.dtype), ctx)
    c = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
    f(a, b, c)
    tvm.testing.assert_allclose(
        c.asnumpy(), np.ldexp(a.asnumpy(), b.asnumpy()), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_nearbyint()
    test_unary_intrin()
    test_round_intrinsics_on_int()
    test_binary_intrin()
    test_ldexp()
