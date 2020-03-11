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


def test_unary_intrin():
    test_funcs = [
        (tvm.tir.exp10, lambda x : np.power(10, x)),
        (tvm.tir.log2, lambda x : np.log2(x)),
        (tvm.tir.log10, lambda x : np.log10(x)),
        (tvm.tir.sinh, lambda x : np.sinh(x)),
        (tvm.tir.cosh, lambda x : np.cosh(x)),
    ]
    def run_test(tvm_intrin, np_func):
        m = te.var("m",)
        A = te.placeholder((m,), name='A')
        B = te.compute((m,), lambda *i: tvm_intrin(A(*i)), name='B')
        s = te.create_schedule(B.op)
        f = tvm.build(s, [A, B], "llvm")
        ctx = tvm.cpu(0)
        n = 10
        a = tvm.nd.array(np.random.uniform(0, 1, size=n).astype(A.dtype), ctx)
        b = tvm.nd.array( \
            np.random.uniform(size=n).astype(A.dtype), ctx)
        f(a, b)
        tvm.testing.assert_allclose(
            b.asnumpy(), np_func(a.asnumpy()), atol=1e-5, rtol=1e-5)
    
    for func in test_funcs:
        run_test(*func);


if __name__ == "__main__":
    test_nearbyint()
    test_unary_intrin()
