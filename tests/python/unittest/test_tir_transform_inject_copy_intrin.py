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
import tvm.testing
from tvm import te
from tvm.driver.build_module import schedule_to_module


def test_copy2d():
    m = te.var("m")
    l = te.var("l")
    A = te.placeholder((m, l), name="A")
    B = te.compute((m, l), lambda i, j: A[i, j], name="B")
    s = te.create_schedule(B.op)
    s[B].pragma(B.op.axis[0], "memcpy")
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)
    func = tvm.te.schedule.SchedulePostProcToPrimFunc([A, B], stmt, None)
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.StorageFlatten(64)(mod)

    def cb(src, dst, pad_before, pad_after, pad_value):
        assert dst.strides[0] == l
        assert dst.strides[1].value == 1
        assert src.strides[0] == l
        assert tuple(src.shape) == (m, l)
        return tvm.tir.Evaluate(0)

    stmt = tvm.tir.transform.InjectCopyIntrin("memcpy", cb)(mod)["main"].body


def test_copy_pad():
    m = te.var("m")
    l = te.var("l")
    A = te.placeholder((m, l), name="A")
    B = te.compute(
        (m + 2, l),
        lambda i, j: tvm.tir.if_then_else(tvm.tir.all(i >= 1, i < m + 1), A[i - 1, j], 1.0),
        name="B",
    )
    s = te.create_schedule(B.op)
    s[B].pragma(B.op.axis[0], "memcpy")
    mod = schedule_to_module(s, [A, B])
    mod = tvm.tir.transform.StorageFlatten(64)(mod)

    def cb(src, dst, pad_before, pad_after, pad_value):
        tvm.testing.assert_prim_expr_equal(src.elem_offset, 0)
        assert pad_before[0].value == 1
        assert pad_before[1].value == 0
        assert pad_after[0].value == 1
        assert pad_after[1].value == 0
        assert pad_value.value == 1.0
        return tvm.tir.Evaluate(0)

    stmt = tvm.tir.transform.InjectCopyIntrin("memcpy", cb)(mod)["main"].body


def test_single_point_test():
    A = te.placeholder((1,), name="A")
    B = te.compute((1,), lambda i: A[i], name="B")
    s = te.create_schedule(B.op)
    s[B].pragma(B.op.axis[0], "memcpy")
    mod = schedule_to_module(s, [A, B])
    mod = tvm.tir.transform.StorageFlatten(64)(mod)

    def cb(src, dst, pad_before, pad_after, pad_value):
        tvm.testing.assert_prim_expr_equal(src.elem_offset, 0)
        tvm.testing.assert_prim_expr_equal(dst.elem_offset, 0)
        tvm.testing.assert_prim_expr_equal(src.strides[0], 1)
        tvm.testing.assert_prim_expr_equal(dst.strides[0], 1)
        return tvm.tir.Evaluate(0)

    stmt = tvm.tir.transform.InjectCopyIntrin("memcpy", cb)(mod)["main"].body


def test_copy_pad_split():
    m = 4 * 3
    A = te.placeholder((m,), name="A")
    Apad = te.compute(
        (m + 2,), lambda i: tvm.tir.if_then_else(tvm.tir.all(i >= 1, i <= m), A[i - 1], 0.0), "Apad"
    )
    B = te.compute((m,), lambda i: Apad[i] + Apad[i + 1] + Apad[i + 2])
    s = te.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=4)
    s[Apad].compute_at(s[B], xo)
    s[Apad].pragma(s[Apad].op.axis[0], "memcpy")

    mod = schedule_to_module(s, [A, B])
    mod = tvm.tir.transform.StorageFlatten(64)(mod._move())
    mod = tvm.tir.transform.Simplify()(mod._move())

    def cb(src, dst, pad_before, pad_after, pad_value):
        assert dst.elem_offset.value == 0
        tvm.testing.assert_prim_expr_equal(src.elem_offset, tvm.te.max(xo * 4, 1) - 1)

        rpad_before = tvm.te.max(1 - xo * 4, 0)
        rpad_after = tvm.te.max(xo * 4 - 7, 0)
        tvm.testing.assert_prim_expr_equal(pad_before[0], rpad_before)
        tvm.testing.assert_prim_expr_equal(pad_after[0], rpad_after)
        tvm.testing.assert_prim_expr_equal(src.shape[0], 6 - rpad_before - rpad_after)
        return tvm.tir.Evaluate(0)

    stmt = tvm.tir.transform.InjectCopyIntrin("memcpy", cb)(mod)["main"].body


if __name__ == "__main__":
    test_copy2d()
    test_copy_pad()
    test_copy_pad_split()
    test_single_point_test()
