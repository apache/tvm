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

def test_copy2d():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.compute((m, l), lambda i, j: A[i, j], name='B')
    s = tvm.create_schedule(B.op)
    s[B].pragma(B.op.axis[0], "memcpy")
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    Bb = tvm.decl_buffer(B.shape, B.dtype, name='B')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B: Bb}, 64)
    def cb(src, dst, pad_before, pad_after, pad_value):
        assert dst.strides[0] == l
        assert dst.strides[1].value == 1
        assert src.strides[0] == l
        assert tuple(src.shape) == (m, l)
        return tvm.make.Evaluate(0)
    stmt = tvm.ir_pass.InjectCopyIntrin(stmt, "memcpy", cb)

def test_copy_pad():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.compute((m + 2, l), lambda i, j:
                    tvm.if_then_else(tvm.all(i >= 1, i < m + 1),
                                     A[i - 1, j], 1.0), name='B')
    s = tvm.create_schedule(B.op)
    s[B].pragma(B.op.axis[0], "memcpy")
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    Bb = tvm.decl_buffer(B.shape, B.dtype, name='B')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B: Bb}, 64)
    def cb(src, dst, pad_before, pad_after, pad_value):
        assert tvm.ir_pass.Simplify(src.elem_offset).value == 0
        assert pad_before[0].value == 1
        assert pad_before[1].value == 0
        assert pad_after[0].value == 1
        assert pad_after[1].value == 0
        assert pad_value.value == 1.0
        return tvm.make.Evaluate(0)
    stmt = tvm.ir_pass.InjectCopyIntrin(stmt, "memcpy", cb)

def test_single_point_test():
    A = tvm.placeholder((1,), name='A')
    B = tvm.compute((1,), lambda i:
                    A[i], name='B')
    s = tvm.create_schedule(B.op)
    s[B].pragma(B.op.axis[0], "memcpy")
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    Bb = tvm.decl_buffer(B.shape, B.dtype, name='B')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B: Bb}, 64)
    def cb(src, dst, pad_before, pad_after, pad_value):
        assert tvm.ir_pass.Simplify(src.elem_offset).value == 0
        assert tvm.ir_pass.Simplify(dst.elem_offset).value == 0
        assert tvm.ir_pass.Simplify(src.strides[0]).value == 1
        assert tvm.ir_pass.Simplify(dst.strides[0]).value == 1
        return tvm.make.Evaluate(0)
    stmt = tvm.ir_pass.InjectCopyIntrin(stmt, "memcpy", cb)

def assert_expr_equal(a, b):
    assert tvm.ir_pass.Simplify(a - b).value == 0

def test_copy_pad_split():
    m = 4 * 3
    A = tvm.placeholder((m, ), name="A")
    Apad = tvm.compute((m + 2,), lambda i:
                       tvm.if_then_else(tvm.all(i >= 1, i <= m),
                                        A[i - 1], 0.0), "Apad")
    B = tvm.compute((m,), lambda i: Apad[i] + Apad[i + 1] + Apad[i + 2])
    s = tvm.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=4)
    s[Apad].compute_at(s[B], xo)
    s[Apad].pragma(s[Apad].op.axis[0], "memcpy")
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    Bb = tvm.decl_buffer(B.shape, B.dtype, name='B')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B: Bb}, 64)
    stmt = tvm.ir_pass.Simplify(stmt)
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    def cb(src, dst, pad_before, pad_after, pad_value):
        assert(dst.elem_offset.value == 0)
        assert_expr_equal(src.elem_offset, tvm.max(xo * 4, 1) - 1)

        rpad_before = tvm.max(1 - xo * 4, 0)
        rpad_after = tvm.max(xo * 4 - 7, 0)
        assert_expr_equal(pad_before[0], rpad_before)
        assert_expr_equal(pad_after[0], rpad_after)
        assert_expr_equal(src.shape[0], 6 - rpad_before - rpad_after)
        return tvm.make.Evaluate(0)
    stmt = tvm.ir_pass.InjectCopyIntrin(stmt, "memcpy", cb)


if __name__ == "__main__":
    test_copy2d()
    test_copy_pad()
    test_copy_pad_split()
    test_single_point_test()
