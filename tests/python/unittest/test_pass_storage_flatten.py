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

def test_flatten2():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = tvm.create_schedule(A2.op)
    xo, xi = s[A2].split(A2.op.axis[0], 8)
    s[A1].compute_at(s[A2], xo)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)

    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    A2b = tvm.decl_buffer(A2.shape, A2.dtype, name='A2')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, A2: A2b}, 64)
    stmt = tvm.ir_pass.Simplify(stmt)

def test_flatten_prefetch():
    A = tvm.placeholder((25, 100, 4), name = 'A')
    _A= tvm.decl_buffer(A.shape, A.dtype, name = 'A');
    i = tvm.var('i')
    j = tvm.var('j')
    region = [tvm.make.range_by_min_extent(i[0], i[1]) for i in [(i, 2), (j, 8), (0, 4)]]
    stmt = tvm.make.Prefetch(A.op, 0, A.dtype, region)
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: _A}, 64)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert stmt.extent.value == 2
    assert isinstance(stmt.body, tvm.stmt.For)
    assert stmt.body.extent.value == 2


def test_flatten_storage_align():
    m = 8
    l = 16
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = tvm.create_schedule(A2.op)
    s[A1].storage_align(A1.op.axis[0], 2, 1)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    A2b = tvm.decl_buffer(A2.shape, A2.dtype, name='A2')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, A2: A2b}, 64)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert(stmt.body.extents[0].value == 17 * 8)

def test_flatten_double_buffer():
    dtype = 'int64'
    n = 100
    m = 4
    tx = tvm.thread_axis("threadIdx.x")
    ib = tvm.ir_builder.create()
    A = ib.pointer("float32", name="A")
    C = ib.pointer("float32", name="C")
    ib.scope_attr(tx, "thread_extent", 1)
    with ib.for_range(0, n) as i:
        B = ib.allocate("float32", m, name="B", scope="shared")
        with ib.new_scope():
            ib.scope_attr(B.asnode(), "double_buffer_scope", 1)
            with ib.for_range(0, m) as j:
                B[j] = A[i * 4 + j]
        with ib.for_range(0, m) as j:
            C[j] = B[j] + 1

    stmt = ib.get()
    stmt = tvm.ir_pass.StorageFlatten(stmt, {}, 64)
    stmt = tvm.ir_pass.InjectDoubleBuffer(stmt, 2)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert isinstance(stmt.body.body, tvm.stmt.Allocate)
    assert stmt.body.body.extents[0].value == 2
    f = tvm.ir_pass.MakeAPI(stmt, "db", [A.asnode(), C.asnode()], 2, True)
    f = tvm.ir_pass.ThreadSync(f, "shared")
    count = [0]
    def count_sync(op):
        if isinstance(op, tvm.expr.Call) and op.name == "tvm_storage_sync":
            count[0] += 1
    tvm.ir_pass.PostOrderVisit(f.body, count_sync)
    assert count[0] == 4

if __name__ == "__main__":
    test_flatten_storage_align()
    test_flatten2()
    test_flatten_prefetch()
    test_flatten_double_buffer()
