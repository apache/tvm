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
import numpy as np
from tvm.contrib import util

def test_add():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name='A')
    B = te.placeholder((n,), name='B')
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = te.create_schedule(C.op)

    def check_c():
        mhost = tvm.build(s, [A, B, C], "c", name="fadd")
        temp = util.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fadd = m['fadd']
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        fadd(a, b, c)
        tvm.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + b.asnumpy())
    check_c()


def test_add_pipeline():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name='A')
    B = te.placeholder((n,), name='B')
    AA = te.compute((n,), lambda *i: A(*i), name='A')
    BB = te.compute((n,), lambda *i: B(*i), name='B')
    T = te.compute(A.shape, lambda *i: AA(*i) + BB(*i), name='T')
    C = te.compute(A.shape, lambda *i: T(*i), name='C')
    s = te.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    xo1, xo2 = s[C].split(xo, factor=13)
    s[C].parallel(xo2)
    s[C].pragma(xo1, "parallel_launch_point")
    s[C].pragma(xo2, "parallel_stride_pattern")
    s[C].pragma(xo2, "parallel_barrier_when_finish")
    s[C].vectorize(xi)

    def check_c():
        # Specifically allow offset to test codepath when offset is available
        Ab = tvm.tir.decl_buffer(
            A.shape, A.dtype,
            elem_offset=te.size_var('Aoffset'),
            offset_factor=8,
            name='A')
        binds = {A : Ab}
        # BUILD and invoke the kernel.
        f1 = tvm.lower(s, [A,B,C], name="fadd_pipeline")
        mhost = tvm.build(f1, target="c")

        temp = util.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fadd = m["fadd_pipeline"]
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        fadd(a, b, c)
        tvm.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + b.asnumpy())

    check_c()


def test_reinterpret():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name='A', dtype="int32")
    B = te.compute(A.shape, lambda *i: tvm.tir.call_intrin("float32", "tir.reinterpret", 2 + A(*i)), name='B')
    s = te.create_schedule(B.op)

    def check_c():
        mhost = tvm.build(s, [A, B], "c", name="reinterpret")
        temp = util.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fadd = m['reinterpret']
        ctx = tvm.cpu(0)
        n = nn
        a = tvm.nd.array(np.random.randint(-2 ** 30, 2 ** 30, size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), ctx)
        fadd(a, b)
        tvm.testing.assert_allclose(
            b.asnumpy(), (2 + a.asnumpy()).view('float32'))
    check_c()


if __name__ == "__main__":
    test_add()
    test_add_pipeline()
    test_reinterpret()
