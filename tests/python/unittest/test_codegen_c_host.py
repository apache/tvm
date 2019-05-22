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
import numpy as np
from tvm.contrib import util

def test_add():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = tvm.create_schedule(C.op)

    def check_c():
        mhost = tvm.build(s, [A, B, C], "c", name="fadd")
        temp = util.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.module.load(path_dso)
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
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    AA = tvm.compute((n,), lambda *i: A(*i), name='A')
    BB = tvm.compute((n,), lambda *i: B(*i), name='B')
    T = tvm.compute(A.shape, lambda *i: AA(*i) + BB(*i), name='T')
    C = tvm.compute(A.shape, lambda *i: T(*i), name='C')
    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=4)
    xo1, xo2 = s[C].split(xo, factor=13)
    s[C].parallel(xo2)
    s[C].pragma(xo1, "parallel_launch_point")
    s[C].pragma(xo2, "parallel_stride_pattern")
    s[C].pragma(xo2, "parallel_barrier_when_finish")
    s[C].vectorize(xi)

    def check_c():
        if not tvm.module.enabled("llvm"):
            return
        # Specifically allow offset to test codepath when offset is available
        Ab = tvm.decl_buffer(
            A.shape, A.dtype,
            elem_offset=tvm.var('Aoffset'),
            offset_factor=8,
            name='A')
        binds = {A : Ab}
        # BUILD and invoke the kernel.
        f1 = tvm.lower(s, [A,B,C], name="fadd_pipeline")
        fsplits = [x for x in tvm.ir_pass.SplitHostDevice(f1)]
        fsplits[0] = tvm.ir_pass.LowerTVMBuiltin(fsplits[0])
        mhost = tvm.codegen.build_module(fsplits[0], "c")
        temp = util.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.module.load(path_dso)
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

    with tvm.build_config(offset_factor=4):
        check_c()

if __name__ == "__main__":
    test_add()
    test_add_pipeline()
