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
import topi
from tvm.contrib import util, clang
import numpy as np
import ctypes
import math


def test_nearbyint():
    m = tvm.var("m",)
    A = tvm.placeholder((m,), name='A')
    A_rounded = tvm.compute((m,), lambda *i: tvm.nearbyint(A(*i)), name='A')
    s = tvm.create_schedule(A_rounded.op)
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


if __name__ == "__main__":
    test_nearbyint()
