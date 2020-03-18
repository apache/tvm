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


def lower(sch, args):
    binds = {}
    arg_list = []
    for x in args:
        if isinstance(x, te.tensor.Tensor):
            buf = tvm.tir.decl_buffer(x.shape, dtype=x.dtype, name=x.name)
            assert x not in binds
            binds[x] = buf
            arg_list.append(buf)
        else:
            raise ValueError("args must be Tensor, Buffer or Var")
    bounds = te.schedule.InferBound(sch)
    stmt = te.schedule.ScheduleOps(sch, bounds)
    stmt = tvm.tir.ir_pass.StorageFlatten(stmt, binds, 64, False)
    stmt = tvm.tir.ir_pass.DataTypeRewrite(stmt)
    return stmt


def test_const():
    def test(m, n, dtype):
        A = te.placeholder((m, n), name='A')
        B = te.placeholder((m, n), name='B')
        C = te.compute((m, n), lambda *idx: A[idx] + B[idx])
        s = te.create_schedule(C.op)
        stmt = lower(s, [A, B, C])
        assert stmt.body.loop_var.dtype == dtype
        assert stmt.body.body.loop_var.dtype == dtype

    test(2, 2, "int32")
    # i32 + i32 is not promoted to i64 even in the case of overflow
    test(2**16, 2**16, "int32")
    test(tvm.tir.const(2, dtype='int64'), tvm.tir.const(2, dtype='int64'), "int32")
    test(tvm.tir.const(2**16, dtype='int64'), tvm.tir.const(2**16, dtype='int64'), "int64")


def test_symbolic():
    def test(m, n, dtype):
        A = te.placeholder((m, n), name='A')
        B = te.placeholder((m, n), name='B')
        C = te.compute((m, n), lambda *idx: A[idx] + B[idx])
        s = te.create_schedule(C.op)
        stmt = lower(s, [A, B, C])
        assert stmt.body.loop_var.dtype == dtype
        assert stmt.body.body.loop_var.dtype == dtype

    test(te.size_var(name='m', dtype='int32'), te.size_var(name='n', dtype='int32'), "int32")
    test(te.size_var(name='m', dtype='int64'), te.size_var(name='n', dtype='int64'), "int64")


def test_thread_axis():
    def test(m, n, k, dtype):
        A = te.placeholder((m, n, k), name='A')
        B = te.placeholder((m, n, k), name='B')
        C = te.compute((m, n, k), lambda *idx: A[idx] + B[idx])
        s = te.create_schedule(C.op)
        fused = s[C].fuse(*[axis for axis in C.op.axis])
        xo, xi = s[C].split(fused, factor=32)
        s[C].bind(xo, te.thread_axis("blockIdx.x"))
        s[C].bind(xi, te.thread_axis("threadIdx.x"))
        stmt = lower(s, [A, B, C])

    test(2, 2, 2, dtype='int32')
    # i32 + i32 is not promoted to i64 even in the case of overflow
    test(2**10, 2**11, 2**12, dtype='int32')
    test(tvm.tir.const(2, dtype='int64'),
         tvm.tir.const(2, dtype='int64'),
         tvm.tir.const(2, dtype='int64'),
         dtype='int32')
    test(tvm.tir.const(2**10, dtype='int64'),
         tvm.tir.const(2**11, dtype='int64'),
         tvm.tir.const(2**12, dtype='int64'),
         dtype='int64')


def test_multilanes():
    def test(m, lanes, dtype):
        A = te.placeholder((m,), name='A', dtype='float32x{}'.format(lanes))
        B = te.placeholder((m,), name='B', dtype='float32x{}'.format(lanes))
        C = te.compute((m,), lambda *idx: A[idx] + B[idx])
        s = te.create_schedule(C.op)
        stmt = lower(s, [A, B, C])
        assert stmt.body.loop_var.dtype == dtype

    test(tvm.tir.const(64, dtype='int32'), 2, 'int32')
    test(tvm.tir.const(2 ** 32, dtype='int64'), 2, 'int64')


def test_reduce():
    def test(m, dtype):
        A = te.placeholder((m,), name='A', dtype='float32')
        k = te.reduce_axis((0, m), "k")
        B = te.compute((), lambda *idx: te.sum(A[k], axis=k), name='B')
        s = te.create_schedule(B.op)
        stmt = lower(s, [A, B])
        assert stmt.body[1].loop_var.dtype == dtype

    test(tvm.tir.const(64, dtype='int32'), 'int32')
    test(tvm.tir.const(64, dtype='int64'), 'int32')
    test(te.var('n', dtype='int32'), 'int32')
    test(te.var('n', dtype='int64'), 'int64')


if __name__ == "__main__":
    test_const()
    test_symbolic()
    test_thread_axis()
    test_multilanes()
    test_reduce()
