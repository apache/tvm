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
from tvm.tir import const


def lower_stmt(params, stmt, target_bits):
    func = tvm.tir.PrimFunc(params, stmt)
    func = tvm.tir.transform.NarrowDataType(target_bits)(
        tvm.IRModule.from_expr(func))["main"]
    stmt = func.body
    return stmt


def lower_sch(sch, args, target_bits):
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

    func = tvm.te.schedule.SchedulePostProcToPrimFunc(args, stmt, None)
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.StorageFlatten(64)(mod)
    return tvm.tir.transform.NarrowDataType(target_bits)(mod)["main"].body


def test_basic():
    def check(m, n, target_bits, target_dtype):
        ib = tvm.tir.ir_builder.create()
        Ab = tvm.tir.decl_buffer((m, n), name='A')
        A = ib.buffer_ptr(Ab)
        Bb = tvm.tir.decl_buffer((m, n), name='B')
        B = ib.buffer_ptr(Bb)
        with ib.for_range(0, m, name='i') as i:
            with ib.for_range(0, n, name='j') as j:
                B[i * n + j] = A[i * n + j] + 1
        stmt = ib.get()
        stmt = lower_stmt([Ab, Bb], stmt, target_bits)
        assert stmt.loop_var.dtype == target_dtype
        assert stmt.body.loop_var.dtype == target_dtype

    # const shape
    # i32 -> i32
    check(2, 2, 32, "int32")
    check(2**16, 2**16, 32, "int32")  # i32 + i32 is not promoted to i64 even if overflow
    # i64 -> i32
    check(const(2, dtype='int64'), const(2, dtype='int64'), 32, "int32")
    check(const(2**16, dtype='int64'), const(2**16, dtype='int64'), 32, "int64")
    # i32 -> i16
    check(2, 2, 16, "int16")
    check(2**10, 2**10, 16, "int32")

    # symbolic shape
    check(te.size_var(name='m', dtype='int32'), te.size_var(name='n', dtype='int32'), 32, "int32")
    check(te.size_var(name='m', dtype='int64'), te.size_var(name='n', dtype='int64'), 32, "int64")


def test_thread_axis():
    def check(m, n, target_bits, target_dtype):
        ib = tvm.tir.ir_builder.create()
        Ab = tvm.tir.decl_buffer((m, n), name='A')
        A = ib.buffer_ptr(Ab)
        Bb = tvm.tir.decl_buffer((m, n), name='B')
        B = ib.buffer_ptr(Bb)
        bx = te.thread_axis("blockIdx.x")
        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(bx, "thread_extent", m)
        ib.scope_attr(tx, "thread_extent", n)
        B[bx * n + tx] = A[bx * n + tx] + 1
        stmt = ib.get()
        stmt = lower_stmt([Ab, Bb], stmt, target_bits)
        assert stmt.node.var.dtype == target_dtype
        assert stmt.body.node.var.dtype == target_dtype

    # i32 -> i32
    check(2, 32,
          target_bits=32, target_dtype='int32')
    check(2**30, 32,  # i32 + i32 is not promoted to i64 even in the case of overflow
          target_bits=32, target_dtype='int32')
    # i64 -> i32
    check(const(2, dtype='int64'),
          const(32, dtype='int64'),
          target_bits=32, target_dtype='int32')
    check(const(2**30, dtype='int64'),
          const(32, dtype='int64'),
          target_bits=32, target_dtype='int64')
    # i32 -> i16
    check(2, 32,
          target_bits=16, target_dtype='int16')
    check(2**14, 32,
          target_bits=16, target_dtype='int32')


def test_multilanes():
    def check(m, lanes, target_bits, target_dtype):
        ib = tvm.tir.ir_builder.create()
        Ab = tvm.tir.decl_buffer((m,), dtype='float32x{}'.format(lanes), name='A')
        A = ib.buffer_ptr(Ab)
        Bb = tvm.tir.decl_buffer((m,), dtype='float32x{}'.format(lanes), name='B')
        B = ib.buffer_ptr(Bb)
        with ib.for_range(0, m, name='i', dtype=m.dtype) as i:
            B[i] = A[i] + 1
        stmt = ib.get()
        stmt = lower_stmt([Ab, Bb], stmt, target_bits)
        assert stmt.loop_var.dtype == target_dtype

    # i32 -> i32
    check(const(2 ** 10, dtype='int32'), 2,
          target_bits=32, target_dtype='int32')
    check(const(2 ** 32, dtype='int32'), 2,
          target_bits=32, target_dtype='int32')
    # i64 -> i32
    check(const(2 ** 10, dtype='int64'), 2,
          target_bits=32, target_dtype='int32')
    check(const(2 ** 32, dtype='int64'), 2,
          target_bits=32, target_dtype='int64')
    # i32 -> i16
    check(const(2 ** 10, dtype='int32'), 2,
          target_bits=16, target_dtype='int16')
    check(const(2 ** 16, dtype='int32'), 2,
          target_bits=16, target_dtype='int32')


def test_reduce():
    def check(m, target_bits, target_dtype):
        A = te.placeholder((m,), name='A', dtype='float32')
        k = te.reduce_axis((0, m), "k")
        B = te.compute((), lambda *idx: te.sum(A[k], axis=k), name='B')
        s = te.create_schedule(B.op)
        stmt = lower_sch(s, [A, B], target_bits)
        assert stmt[1].loop_var.dtype == target_dtype

    # i32 -> i32
    check(const(64, dtype='int32'), 32, 'int32')
    # i64 -> i32
    check(const(64, dtype='int64'), 32, 'int32')
    # i32 -> i16
    check(const(64, dtype='int32'), 16, 'int16')
    check(const(2**16, dtype='int32'), 16, 'int32')
    # symbolic
    check(te.var('n', dtype='int32'), 32, 'int32')
    check(te.var('n', dtype='int64'), 32, 'int64')


def test_slice():
    def check(m, n, target_bits, target_dtype):
        # The index may overflow in B, while not in A
        ib = tvm.tir.ir_builder.create()
        Ab = tvm.tir.decl_buffer((m, n), name='A')
        A = ib.buffer_ptr(Ab)
        Bb = tvm.tir.decl_buffer((m, n * 2), name='B')
        B = ib.buffer_ptr(Bb)
        with ib.for_range(0, m, name='i') as i:
            with ib.for_range(0, n, name='j') as j:
                A[i * n + j] = B[i * 2 * n + 2 * j] + 1
        stmt = ib.get()
        stmt = lower_stmt([Ab, Bb], stmt, target_bits)
        assert stmt.loop_var.dtype == target_dtype
        assert stmt.body.loop_var.dtype == target_dtype

    # The maximum index is (2**15 * 2**15 - 1) * 2 <= 2**31 - 1
    check(const(2**15, 'int64'), const(2**15, 'int64'),
          target_bits=32, target_dtype='int32')
    # The maximum index is (2**15 * 2**15 - 1 + 2**15) * 2 > 2**31 - 1
    check(const(2**15, 'int64'), const((2**15 + 1), 'int64'),
          target_bits=32, target_dtype='int64')


if __name__ == "__main__":
    test_basic()
    test_thread_axis()
    test_multilanes()
    test_reduce()
    test_slice()
