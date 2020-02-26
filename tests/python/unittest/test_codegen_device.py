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
from tvm.contrib import util
import numpy as np

def test_large_uint_imm():
    value =  (1 << 63) + 123
    other = tvm.const(3, "uint64")
    n = 12
    num_thread = 2

    A = tvm.compute((n,), lambda *i: tvm.const(value, "uint64") + other, name='A')
    s = tvm.create_schedule(A.op)
    xo, xi = s[A].split(A.op.axis[0], factor=num_thread)
    s[A].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[A].bind(xo, tvm.thread_axis("blockIdx.x"))

    def check_target(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            return
        f = tvm.build(s, [A], device)
        # launch the kernel.
        a = tvm.nd.empty((n, ), dtype=A.dtype, ctx=ctx)
        f(a)
        assert a.asnumpy()[0] == value + 3

    check_target("cuda")
    check_target("vulkan")


def test_add_pipeline():
    n = tvm.size_var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(), name='C')
    D = tvm.compute(A.shape, lambda *i: C(*i) + 1, name='D')
    s = tvm.create_schedule(D.op)

    # GPU schedule have to split by gridIdx and threadIdx
    num_thread = 256
    xo, xi = s[C].split(C.op.axis[0], factor=num_thread)
    s[C].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[C].bind(xo, tvm.thread_axis("blockIdx.x"))

    xo, xi = s[D].split(D.op.axis[0], factor=num_thread)
    s[D].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[D].bind(xo, tvm.thread_axis("blockIdx.x"))

    # compile to IR
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    Bb = tvm.decl_buffer(B.shape, B.dtype, name='B')
    Db = tvm.decl_buffer(D.shape, D.dtype, name='D')
    stmt = tvm.ir_pass.LoopPartition(stmt, False)
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B:Bb, D:Db}, 64)
    stmt = tvm.ir_pass.Simplify(stmt)
    fapi = tvm.ir_pass.MakeAPI(stmt, "myadd", [Ab, Bb, Db], 0, True)
    fsplits = [x for x in tvm.ir_pass.SplitHostDevice(fapi)]
    # lower the floordiv(use stackvm rules so it works for all targets)
    fsplits = [tvm.ir_pass.LowerIntrin(x, "stackvm") for x in fsplits]
    fsplits[0] = tvm.ir_pass.LowerTVMBuiltin(fsplits[0])

    def check_target(device, host="stackvm"):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            return
        if not tvm.runtime.enabled(host):
            return
        mhost = tvm.target.codegen.build_module(fsplits[0], host)
        mdev = tvm.target.codegen.build_module(fsplits[1:], device)
        mhost.import_module(mdev)
        code = mdev.get_source()
        f = mhost.entry_func
        # launch the kernel.
        n = 1027
        a = tvm.nd.array(np.random.uniform(size=n).astype(Ab.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=()).astype(Bb.dtype), ctx)
        d = tvm.nd.array(np.zeros(n, dtype=Db.dtype), ctx)
        f(a, b, d)
        tvm.testing.assert_allclose(
            d.asnumpy(), a.asnumpy() + b.asnumpy() + 1)

    def check_module_save(device, host="stackvm"):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            return
        if not tvm.runtime.enabled(host):
            return
        if device == "cuda":
            fmt = "ptx"
        elif device == "rocm":
            fmt = "hsaco"
        else:
            fmt = device
        mhost = tvm.target.codegen.build_module(fsplits[0], host)
        mdev = tvm.target.codegen.build_module(fsplits[1:], device)
        temp = util.tempdir()
        mpath = temp.relpath("test.%s" % fmt)
        mdev.save(mpath)
        mdev2 = tvm.runtime.load_module(mpath)
        mhost.import_module(mdev2)
        f = mhost.entry_func
        # launch the kernel.
        n = 1027
        a = tvm.nd.array(np.random.uniform(size=n).astype(Ab.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=()).astype(Bb.dtype), ctx)
        d = tvm.nd.array(np.zeros(n, dtype=Db.dtype), ctx)
        f(a, b, d)
        tvm.testing.assert_allclose(
            d.asnumpy(), a.asnumpy() + b.asnumpy() + 1)

    check_target("cuda", host="stackvm")
    check_target("cuda", host="llvm")
    check_module_save("cuda", host="stackvm")
    check_target("nvptx", host="llvm")
    check_target("vulkan", host="llvm")
    check_module_save("vulkan", host="stackvm")
    check_target("rocm", host="llvm")
    check_module_save("rocm", host="llvm")


if __name__ == "__main__":
    test_large_uint_imm()
    test_add_pipeline()
