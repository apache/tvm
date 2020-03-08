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


def test_intrinsic_argtypes():
    for intrinsic in [tvm.tir.round, tvm.tir.trunc, tvm.tir.ceil,
                            tvm.tir.floor, tvm.tir.nearbyint]:
        assert intrinsic(tvm.tir.const(10,'int32')).value == 10
        assert intrinsic(tvm.tir.const(True,'bool')).value == True

    assert tvm.tir.isnan(tvm.tir.const(10, 'int32')).value == False
    assert isinstance(tvm.tir.popcount(tvm.tir.const(10, 'int32')), tvm.tir.expr.Call)

    for intrinsic in [tvm.tir.exp, tvm.tir.erf, tvm.tir.tanh,
                            tvm.tir.sigmoid, tvm.tir.log,
                            tvm.tir.tan, tvm.tir.cos,
                            tvm.tir.sin, tvm.tir.atan,
                            tvm.tir.sqrt, tvm.tir.rsqrt]:
        try:
            intrinsic(tvm.tir.const(10,'int32'))
            assert False
        except RuntimeError:
            pass


if __name__ == "__main__":
    test_nearbyint()
    test_intrinsic_argtypes()