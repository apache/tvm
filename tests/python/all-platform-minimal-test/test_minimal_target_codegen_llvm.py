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
"""LLVM enablement tests."""

import tvm
import tvm.testing
from tvm import te
from tvm import topi
from tvm.contrib import utils
import numpy as np
import ctypes
import math
import re


@tvm.testing.requires_llvm
def test_llvm_add_pipeline():
    """all-platform-minimal-test: Check LLVM enablement."""
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    AA = te.compute((n,), lambda *i: A(*i), name="A")
    BB = te.compute((n,), lambda *i: B(*i), name="B")
    T = te.compute(A.shape, lambda *i: AA(*i) + BB(*i), name="T")
    C = te.compute(A.shape, lambda *i: T(*i), name="C")
    s = te.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    xo1, xo2 = s[C].split(xo, factor=13)
    s[C].parallel(xo2)
    s[C].pragma(xo1, "parallel_launch_point")
    s[C].pragma(xo2, "parallel_stride_pattern")
    s[C].pragma(xo2, "parallel_barrier_when_finish")
    s[C].vectorize(xi)

    def check_llvm():
        # Specifically allow offset to test codepath when offset is available
        Ab = tvm.tir.decl_buffer(
            A.shape, A.dtype, elem_offset=te.size_var("Aoffset"), offset_factor=8, name="A"
        )
        binds = {A: Ab}
        # BUILD and invoke the kernel.
        f = tvm.build(s, [A, B, C], "llvm", binds=binds)
        dev = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        f(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    check_llvm()
