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
from tvm import s_tir
from tvm.s_tir.transform import HoistedConditionals, HoistedLetBindings
from tvm.script import tir as T


def _run_transform(before, hoisted_conditionals, hoisted_let_bindings):
    """Run HoistExpression transform with the given configuration."""
    before_mod = tvm.IRModule.from_expr(before)

    config = {
        "s_tir.HoistExpression": {
            "hoisted_conditionals": hoisted_conditionals.value,
            "hoisted_let_bindings": hoisted_let_bindings.value,
        }
    }

    with tvm.transform.PassContext(config=config):
        after_mod = tvm.s_tir.transform.HoistExpression()(before_mod)

    return after_mod["main"]


def test_hoist_to_top_if_else_stmt():
    @T.prim_func(private=True)
    def before(A: T.Buffer((16,), "float32"), n: T.int32):
        for i in T.serial(16):
            if n != 0:
                A[i] = 0.0

    @T.prim_func(private=True)
    def expected(A: T.Buffer((16,), "float32"), n: T.int32):
        if n != 0:
            for i in T.serial(16):
                A[i] = 0.0

    after = _run_transform(before, HoistedConditionals.IfElseStmt, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_hoist_to_top_all():
    @T.prim_func(private=True)
    def before(A: T.Buffer((16,), "float32"), n: T.int32):
        for i in T.serial(16):
            if n != 0:
                A[i] = 0.0

    @T.prim_func(private=True)
    def expected(A: T.Buffer((16,), "float32"), n: T.int32):
        if n != 0:
            for i in T.serial(16):
                A[i] = 0.0

    after = _run_transform(before, HoistedConditionals.All, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_suppress_hoist_if_else_never():
    @T.prim_func(private=True)
    def before(A: T.Buffer((16,), "float32"), n: T.int32):
        for i in T.serial(16):
            if n != 0:
                A[i] = 0.0

    expected = before

    after = _run_transform(before, HoistedConditionals.Never, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_suppress_hoist_if_else_expr_only():
    @T.prim_func(private=True)
    def before(A: T.Buffer((16,), "float32"), n: T.int32):
        for i in T.serial(16):
            if n != 0:
                A[i] = 0.0

    expected = before

    after = _run_transform(before, HoistedConditionals.IfElseExpr, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_hoist_block_var():
    @T.prim_func(private=True)
    def before(A: T.Buffer((128, 16), "float32"), n: T.int32):
        i = T.env_thread("threadIdx.x")
        T.launch_thread(i, 128)

        for j in T.serial(16):
            if i < 32:
                A[i, j] = 0.0

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128, 16), "float32"), n: T.int32):
        i = T.env_thread("threadIdx.x")
        T.launch_thread(i, 128)

        if i < 32:
            for j in T.serial(16):
                A[i, j] = 0.0

    after = _run_transform(before, HoistedConditionals.All, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_suppress_hoist_block_var():
    @T.prim_func(private=True)
    def before(A: T.Buffer((128, 16), "float32"), n: T.int32):
        thread_x = T.env_thread("threadIdx.x")
        T.launch_thread(thread_x, 128)

        for i in T.thread_binding(0, 128, thread="threadIdx.x"):
            if i < 32:
                for j in T.serial(16):
                    A[i, j] = 0.0

    expected = before

    after = _run_transform(
        before,
        HoistedConditionals.All & ~HoistedConditionals.UsingBlockVar,
        HoistedLetBindings.All,
    )
    tvm.ir.assert_structural_equal(after, expected)


def test_hoist_across_block_var():
    @T.prim_func(private=True)
    def before(A: T.Buffer((128, 16), "float32"), n: T.int32):
        thread_x = T.env_thread("threadIdx.x")
        T.launch_thread(thread_x, 128)

        for i in T.thread_binding(0, 128, thread="threadIdx.x"):
            if n == 0:
                for j in T.serial(16):
                    A[i, j] = 0.0

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128, 16), "float32"), n: T.int32):
        thread_x = T.env_thread("threadIdx.x")

        if n == 0:
            T.launch_thread(thread_x, 128)
            for i in T.thread_binding(0, 128, thread="threadIdx.x"):
                for j in T.serial(16):
                    A[i, j] = 0.0

    after = _run_transform(before, HoistedConditionals.All, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_suppress_hoist_across_block_var():
    @T.prim_func(private=True)
    def before(A: T.Buffer((128, 16), "float32"), n: T.int32):
        thread_x = T.env_thread("threadIdx.x")
        T.launch_thread(thread_x, 128)

        for i in T.thread_binding(0, 128, thread="threadIdx.x"):
            for j in T.serial(16):
                if n == 0:
                    A[i, j] = 0.0

    @T.prim_func(private=True)
    def expected(A: T.Buffer((128, 16), "float32"), n: T.int32):
        thread_x = T.env_thread("threadIdx.x")

        T.launch_thread(thread_x, 128)
        if n == 0:
            for i in T.thread_binding(0, 128, thread="threadIdx.x"):
                for j in T.serial(16):
                    A[i, j] = 0.0

    after = _run_transform(
        before,
        HoistedConditionals.All & ~HoistedConditionals.UsingBlockVar,
        HoistedLetBindings.All,
    )
    tvm.ir.assert_structural_equal(after, expected)


def test_hoist_to_middle():
    @T.prim_func(private=True)
    def before(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            for j in T.serial(4):
                if i < 3:
                    A[i, j] = 0.0

    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            if i < 3:
                for j in T.serial(4):
                    A[i, j] = 0.0

    after = _run_transform(before, HoistedConditionals.All, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_hoist_with_let():
    @T.prim_func(private=True)
    def before(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            for j in T.serial(4):
                condition = i < 3
                if condition:
                    A[i, j] = 0.0

    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            condition = i < 3
            if condition:
                for j in T.serial(4):
                    A[i, j] = 0.0

    after = _run_transform(before, HoistedConditionals.All, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_hoist_disable_let():
    """As test_hoist_with_let, but forbid hoisting of LetStmt

    Because the condition depends on the let binding, it should no
    longer be hoisted.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            for j in T.serial(4):
                condition = i < 3
                if condition:
                    A[i, j] = 0.0

    expected = before

    after = _run_transform(before, HoistedConditionals.All, HoistedLetBindings.Never)
    tvm.ir.assert_structural_equal(after, expected)


def test_hoist_if_else():
    @T.prim_func(private=True)
    def before(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            for j in T.serial(4):
                if i < 3:
                    A[i, j] = 0.0
                else:
                    A[i, j] = 1.0

    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            if i < 3:
                for j in T.serial(4):
                    A[i, j] = 0.0
            else:
                for j in T.serial(4):
                    A[i, j] = 1.0

    after = _run_transform(before, HoistedConditionals.All, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_hoist_sequential_assign():
    @T.prim_func(private=True)
    def before(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            for j in T.serial(4):
                if i < 3:
                    A[i, j] = 0.0
                    B[i, j] = 0.0
                else:
                    A[i, j] = 1.0
                    B[i, j] = 1.0

    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            if i < 3:
                for j in T.serial(4):
                    A[i, j] = 0.0
                    B[i, j] = 0.0
            else:
                for j in T.serial(4):
                    A[i, j] = 1.0
                    B[i, j] = 1.0

    after = _run_transform(before, HoistedConditionals.All, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_hoist_multi_if():
    @T.prim_func(private=True)
    def before(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            for j in T.serial(4):
                for k in T.serial(4):
                    if j < 3:
                        if i < 2:
                            A[i, j] = 0.0

    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            if i < 2:
                for j in T.serial(4):
                    if j < 3:
                        for k in T.serial(4):
                            A[i, j] = 0.0

    after = _run_transform(before, HoistedConditionals.All, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_hoist_complex_conditional():
    @T.prim_func(private=True)
    def before(A: T.Buffer((4, 4), "float32")):
        for i, j, k in T.grid(4, 4, 4):
            if j < 3 and i < 2:
                A[i, j] = 0.0

    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            if i < 2:
                for j in T.serial(4):
                    if j < 3:
                        for k in T.serial(4):
                            A[i, j] = 0.0

    after = _run_transform(before, HoistedConditionals.All, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_suppress_splitting_conditional():
    @T.prim_func(private=True)
    def before(A: T.Buffer((4, 4), "float32")):
        for i, j, k in T.grid(4, 4, 4):
            if j < 3 and i < 2:
                A[i, j] = 0.0

    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 4), "float32")):
        for i, j in T.grid(4, 4):
            if j < 3 and i < 2:
                for k in T.serial(4):
                    A[i, j] = 0.0

    after = _run_transform(
        before,
        HoistedConditionals.All & ~HoistedConditionals.BooleanExpression,
        HoistedLetBindings.All,
    )
    tvm.ir.assert_structural_equal(after, expected)


def test_hoist_multi_if_else():
    @T.prim_func(private=True)
    def before(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            for j in T.serial(4):
                for k in T.serial(4):
                    if j < 3:
                        if i < 2:
                            A[i, j] = 0.0
                        else:
                            A[i, j] = 1.0
                    else:
                        if i < 2:
                            A[i, j] = 2.0
                        else:
                            A[i, j] = 3.0

    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            if i < 2:
                for j in T.serial(4):
                    if j < 3:
                        for k in T.serial(4):
                            A[i, j] = 0.0
                    else:
                        for k in T.serial(4):
                            A[i, j] = 2.0
            else:
                for j in T.serial(4):
                    if j < 3:
                        for k in T.serial(4):
                            A[i, j] = 1.0
                    else:
                        for k in T.serial(4):
                            A[i, j] = 3.0

    after = _run_transform(before, HoistedConditionals.All, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_hoist_multi_if_else_different_branches():
    @T.prim_func(private=True)
    def before(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            for j in T.serial(4):
                for k in T.serial(4):
                    if j < 3:
                        if i < 2:
                            A[i, j] = 0.0
                        else:
                            A[i, j] = 1.0
                    else:
                        if i < 1:
                            A[i, j] = 2.0
                        else:
                            A[i, j] = 3.0

    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            if i < 2:
                if i < 1:
                    for j in T.serial(4):
                        if j < 3:
                            for k in T.serial(4):
                                A[i, j] = 0.0
                        else:
                            for k in T.serial(4):
                                A[i, j] = 2.0
                else:
                    for j in T.serial(4):
                        if j < 3:
                            for k in T.serial(4):
                                A[i, j] = 0.0
                        else:
                            for k in T.serial(4):
                                A[i, j] = 3.0
            else:
                for j in T.serial(4):
                    if j < 3:
                        for k in T.serial(4):
                            A[i, j] = 1.0
                    else:
                        for k in T.serial(4):
                            A[i, j] = 3.0

    after = _run_transform(before, HoistedConditionals.All, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_hoist_if_else_expr():
    @T.prim_func(private=True)
    def before(A: T.Buffer((4, 4), "float32")):
        for i, j in T.grid(4, 4):
            A[i, j] = T.if_then_else(i < 2, 1.0, 2.0, dtype="float32")

    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            if i < 2:
                for j in T.serial(4):
                    A[i, j] = 1.0
            else:
                for j in T.serial(4):
                    A[i, j] = 2.0

    after = _run_transform(before, HoistedConditionals.All, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_suppress_hoist_if_else_expr():
    @T.prim_func(private=True)
    def before(A: T.Buffer((4, 4), "float32")):
        for i, j in T.grid(4, 4):
            A[i, j] = T.if_then_else(i < 2, 1.0, 2.0, dtype="float32")

    expected = before

    after = _run_transform(
        before,
        HoistedConditionals.All & ~HoistedConditionals.IfElseExpr,
        HoistedLetBindings.All,
    )
    tvm.ir.assert_structural_equal(after, expected)


def test_hoist_let_expr():
    @T.prim_func(private=True)
    def before(A: T.Buffer((4, 4), "float32")):
        for i, j in T.grid(4, 4):
            x = T.float32()
            A[i, j] = T.Let(5.0 * x + T.cast(j, "float32"), where={x: T.cast(i + 1, "float32")})

    @T.prim_func(private=True)
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            x = T.cast(i + 1, "float32")
            for j in T.serial(4):
                A[i, j] = 5.0 * x + T.cast(j, "float32")

    after = _run_transform(before, HoistedConditionals.All, HoistedLetBindings.All)
    tvm.ir.assert_structural_equal(after, expected)


def test_suppress_hoist_let_expr():
    @T.prim_func(private=True)
    def before(A: T.Buffer((4, 4), "float32")):
        for i, j in T.grid(4, 4):
            x = T.float32()
            A[i, j] = T.Let(5.0 * x + T.cast(j, "float32"), where={x: T.cast(i + 1, "float32")})

    expected = before

    after = _run_transform(
        before,
        HoistedConditionals.All,
        HoistedLetBindings.All & ~HoistedLetBindings.LetExpr,
    )
    tvm.ir.assert_structural_equal(after, expected)


if __name__ == "__main__":
    tvm.testing.main()
