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
    nn = 128
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    AA = te.compute((n,), lambda *i: A(*i), name="A")
    BB = te.compute((n,), lambda *i: B(*i), name="B")
    T = te.compute(A.shape, lambda *i: AA(*i) + BB(*i), name="T")
    C = te.compute(A.shape, lambda *i: T(*i), name="C")

    sch = tvm.tir.Schedule(te.create_prim_func([A, B, C]))
    xo, xi = sch.split(sch.get_loops("C")[0], factors=[None, 4])
    sch.parallel(xo)
    sch.vectorize(xi)

    def check_llvm():
        # BUILD and invoke the kernel.
        f = tvm.compile(sch.mod, target="llvm")
        dev = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        f(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    check_llvm()
