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

def test_for():
    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    A = ib.allocate("float32", n, name="A", scope="global")
    with ib.for_range(0, n, name="i") as i:
        A[i] = A[i] + 1
        with ib.for_range(0, 10, name="j") as j:
            A[j] = A[j] + 2

    body = ib.get()
    print(body)
    assert isinstance(body, tvm.stmt.AttrStmt)
    body = body.body
    assert isinstance(body, tvm.stmt.Allocate)
    body = body.body
    assert isinstance(body, tvm.stmt.For)
    body = body.body
    assert isinstance(body, tvm.stmt.Block)
    assert isinstance(body.rest, tvm.stmt.For)

def test_if():
    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, n, name="i") as i:
        with ib.if_scope((i % 2) == 0):
            A[i] = A[i] + 1
        with ib.else_scope():
            A[0] = A[i] + 2

    body = ib.get()
    assert A == A
    assert isinstance(body, tvm.stmt.For)
    body = body.body
    assert isinstance(body, tvm.stmt.IfThenElse)
    assert isinstance(body.condition, tvm.expr.EQ)
    assert isinstance(body.then_case.index, tvm.expr.Var)
    assert body.else_case.index.value == 0

def test_prefetch():
    A = tvm.placeholder((10, 20), name="A")
    ib = tvm.ir_builder.create()
    n = tvm.var("n")

    with ib.for_range(0, n, name="i") as i:
        ib.emit(
            tvm.make.Prefetch(
                A.op, A.value_index, A.dtype,
                [tvm.make.range_by_min_extent(i+1, 2),
                 tvm.make.range_by_min_extent(0, 20)]))
    body = ib.get()
    assert body.body.bounds[0].extent.value == 2

def test_cpu():
    n = 1024
    dtype = "float32"
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    def test_device_ir(A, B, C):
        n = A.shape[0]
        max_threads = 8
        ib = tvm.ir_builder.create()
        Aptr = ib.buffer_ptr(A)
        Bptr = ib.buffer_ptr(B)
        Cptr = ib.buffer_ptr(C)
        with ib.for_range(0, n, name="i") as i:
            Cptr[i] = Aptr[i] + Bptr[i]
        body = ib.get()
        return body
    C = tvm.extern(A.shape, [A, B], lambda ins, outs: test_device_ir(ins[0], ins[1], outs[0]),
                   name="vector_add", dtype=dtype)
    s = tvm.create_schedule(C.op)
    def check_target(target):
        if not tvm.module.enabled(target):
            return
        # build and invoke the kernel.
        fadd = tvm.build(s, [A, B, C], target)
        ctx = tvm.context(target, 0)
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())
    check_target("llvm")

def test_gpu():
    n = tvm.var('n')
    dtype = "float32"
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    def test_device_ir(A, B, C):
        n = A.shape[0]
        max_threads = 32
        ib = tvm.ir_builder.create()
        bx = tvm.thread_axis("blockIdx.x")
        tx = tvm.thread_axis("threadIdx.x")
        ib.scope_attr(bx, "thread_extent", (n+max_threads-1) // max_threads)
        ib.scope_attr(tx, "thread_extent", max_threads)
        idx = bx.var * max_threads + tx.var
        Aptr = ib.buffer_ptr(A)
        Bptr = ib.buffer_ptr(B)
        Cptr = ib.buffer_ptr(C)
        with ib.if_scope(ib.likely(idx<n)):
            Cptr[idx] = Aptr[idx] + Bptr[idx]
        body = ib.get()
        return body
    C = tvm.extern(A.shape, [A, B], lambda ins, outs: test_device_ir(ins[0], ins[1], outs[0]),
                   name="vector_add", dtype=dtype)
    s = tvm.create_schedule(C.op)
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    def check_target(target):
        n = 1024
        if not tvm.module.enabled(target):
            return
        # build and invoke the kernel.
        fadd = tvm.build(s, [A, B, C], target)
        ctx = tvm.context(target, 0)
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())
    check_target("opencl")
    check_target("cuda")

if __name__ == "__main__":
    test_prefetch()
    test_if()
    test_for()
    test_cpu()
    test_gpu()
