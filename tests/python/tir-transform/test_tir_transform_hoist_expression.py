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
from tvm import tir
from tvm.script import tir as T
from tvm.tir.transform import HoistedConditionals, HoistedLetBindings, HoistExpression


class BaseBeforeAfter:
    hoisted_conditionals = tvm.testing.parameter(HoistedConditionals.All)
    hoisted_let_bindings = tvm.testing.parameter(HoistedLetBindings.All)

    def test_hoist(self, hoisted_conditionals, hoisted_let_bindings):
        before = self.before
        before_mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))

        config = {
            "tir.HoistExpression": {
                "hoisted_conditionals": hoisted_conditionals.value,
                "hoisted_let_bindings": hoisted_let_bindings.value,
            }
        }

        with tvm.transform.PassContext(config=config):
            after_mod = tvm.tir.transform.HoistExpression()(before_mod)

        after = after_mod["main"]
        expected = self.expected.with_attr("global_symbol", "main")

        try:
            tvm.ir.assert_structural_equal(after, expected)
        except ValueError as err:
            script = tvm.IRModule({"expected": expected, "after": after, "before": before}).script()
            raise ValueError(
                f"Function after simplification did not match expected:\n{script}"
            ) from err


class TestHoistToTop(BaseBeforeAfter):
    hoisted_conditionals = tvm.testing.parameter(
        HoistedConditionals.IfElseStmt,
        HoistedConditionals.All,
    )

    @T.prim_func
    def before(A: T.Buffer((16,), "float32"), n: T.int32):
        for i in T.serial(16):
            if n != 0:
                A[i] = 0.0

    @T.prim_func
    def expected(A: T.Buffer((16,), "float32"), n: T.int32):
        if n != 0:
            for i in T.serial(16):
                A[i] = 0.0


class TestSuppressHoistIfElse(BaseBeforeAfter):
    hoisted_conditionals = tvm.testing.parameter(
        HoistedConditionals.Never,
        HoistedConditionals.IfElseExpr,
    )

    @T.prim_func
    def before(A: T.Buffer((16,), "float32"), n: T.int32):
        for i in T.serial(16):
            if n != 0:
                A[i] = 0.0

    expected = before


class TestHoistBlockVar(BaseBeforeAfter):
    @T.prim_func
    def before(A: T.Buffer((128, 16), "float32"), n: T.int32):
        i = T.env_thread("threadIdx.x")
        T.launch_thread(i, 128)

        for j in T.serial(16):
            if i < 32:
                A[i, j] = 0.0

    @T.prim_func
    def expected(A: T.Buffer((128, 16), "float32"), n: T.int32):
        i = T.env_thread("threadIdx.x")
        T.launch_thread(i, 128)

        if i < 32:
            for j in T.serial(16):
                A[i, j] = 0.0


class TestSuppressHoistBlockVar(BaseBeforeAfter):
    hoisted_conditionals = tvm.testing.parameter(
        HoistedConditionals.All & ~HoistedConditionals.UsingBlockVar
    )

    @T.prim_func
    def before(A: T.Buffer((128, 16), "float32"), n: T.int32):
        thread_x = T.env_thread("threadIdx.x")
        T.launch_thread(thread_x, 128)

        for i in T.thread_binding(0, 128, thread="threadIdx.x"):
            if i < 32:
                for j in T.serial(16):
                    A[i, j] = 0.0

    expected = before


class TestHoistAcrossBlockVar(BaseBeforeAfter):
    @T.prim_func
    def before(A: T.Buffer((128, 16), "float32"), n: T.int32):
        thread_x = T.env_thread("threadIdx.x")
        T.launch_thread(thread_x, 128)

        for i in T.thread_binding(0, 128, thread="threadIdx.x"):
            if n == 0:
                for j in T.serial(16):
                    A[i, j] = 0.0

    @T.prim_func
    def expected(A: T.Buffer((128, 16), "float32"), n: T.int32):
        thread_x = T.env_thread("threadIdx.x")

        if n == 0:
            T.launch_thread(thread_x, 128)
            for i in T.thread_binding(0, 128, thread="threadIdx.x"):
                for j in T.serial(16):
                    A[i, j] = 0.0


class TestSuppressHoistAcrossBlockVar(BaseBeforeAfter):
    hoisted_conditionals = tvm.testing.parameter(
        HoistedConditionals.All & ~HoistedConditionals.UsingBlockVar
    )

    @T.prim_func
    def before(A: T.Buffer((128, 16), "float32"), n: T.int32):
        thread_x = T.env_thread("threadIdx.x")
        T.launch_thread(thread_x, 128)

        for i in T.thread_binding(0, 128, thread="threadIdx.x"):
            for j in T.serial(16):
                if n == 0:
                    A[i, j] = 0.0

    @T.prim_func
    def expected(A: T.Buffer((128, 16), "float32"), n: T.int32):
        thread_x = T.env_thread("threadIdx.x")

        T.launch_thread(thread_x, 128)
        if n == 0:
            for i in T.thread_binding(0, 128, thread="threadIdx.x"):
                for j in T.serial(16):
                    A[i, j] = 0.0


class TestHoistToMiddle(BaseBeforeAfter):
    @T.prim_func
    def before(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            for j in T.serial(4):
                if i < 3:
                    A[i, j] = 0.0

    @T.prim_func
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            if i < 3:
                for j in T.serial(4):
                    A[i, j] = 0.0


class TestHoistWithLet(BaseBeforeAfter):
    @T.prim_func
    def before(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            for j in T.serial(4):
                condition = i < 3
                if condition:
                    A[i, j] = 0.0

    @T.prim_func
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            condition = i < 3
            if condition:
                for j in T.serial(4):
                    A[i, j] = 0.0


class TestHoistDisableLet(BaseBeforeAfter):
    """As TestHoistWithLet, but forbid hoisting of LetStmt

    Because the condition depends on the let binding, it should no
    longer be hoisted.
    """

    hoisted_let_bindings = tvm.testing.parameter(HoistedLetBindings.Never)

    @T.prim_func
    def before(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            for j in T.serial(4):
                condition = i < 3
                if condition:
                    A[i, j] = 0.0

    expected = before


class TestHoistIfElse(BaseBeforeAfter):
    @T.prim_func
    def before(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            for j in T.serial(4):
                if i < 3:
                    A[i, j] = 0.0
                else:
                    A[i, j] = 1.0

    @T.prim_func
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            if i < 3:
                for j in T.serial(4):
                    A[i, j] = 0.0
            else:
                for j in T.serial(4):
                    A[i, j] = 1.0


class TestHoistSequentialAssign(BaseBeforeAfter):
    @T.prim_func
    def before(A: T.Buffer((4, 4), "float32"), B: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            for j in T.serial(4):
                if i < 3:
                    A[i, j] = 0.0
                    B[i, j] = 0.0
                else:
                    A[i, j] = 1.0
                    B[i, j] = 1.0

    @T.prim_func
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


class TestHoistMultiIf(BaseBeforeAfter):
    @T.prim_func
    def before(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            for j in T.serial(4):
                for k in T.serial(4):
                    if j < 3:
                        if i < 2:
                            A[i, j] = 0.0

    @T.prim_func
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            if i < 2:
                for j in T.serial(4):
                    if j < 3:
                        for k in T.serial(4):
                            A[i, j] = 0.0


class TestHoistComplexConditional(BaseBeforeAfter):
    @T.prim_func
    def before(A: T.Buffer((4, 4), "float32")):
        for i, j, k in T.grid(4, 4, 4):
            if j < 3 and i < 2:
                A[i, j] = 0.0

    @T.prim_func
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            if i < 2:
                for j in T.serial(4):
                    if j < 3:
                        for k in T.serial(4):
                            A[i, j] = 0.0


class TestSuppressSplittingConditional(BaseBeforeAfter):
    hoisted_conditionals = tvm.testing.parameter(
        HoistedConditionals.All & ~HoistedConditionals.BooleanExpression
    )

    @T.prim_func
    def before(A: T.Buffer((4, 4), "float32")):
        for i, j, k in T.grid(4, 4, 4):
            if j < 3 and i < 2:
                A[i, j] = 0.0

    @T.prim_func
    def expected(A: T.Buffer((4, 4), "float32")):
        for i, j in T.grid(4, 4):
            if j < 3 and i < 2:
                for k in T.serial(4):
                    A[i, j] = 0.0


class TestHoistMultiIfElse(BaseBeforeAfter):
    @T.prim_func
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

    @T.prim_func
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


class TestHoistMultiIfElseDifferentBranches(BaseBeforeAfter):
    @T.prim_func
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

    @T.prim_func
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


class TestHoistIfElseExpr(BaseBeforeAfter):
    @T.prim_func
    def before(A: T.Buffer((4, 4), "float32")):
        for i, j in T.grid(4, 4):
            A[i, j] = T.if_then_else(i < 2, 1.0, 2.0, dtype="float32")

    @T.prim_func
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            if i < 2:
                for j in T.serial(4):
                    A[i, j] = 1.0
            else:
                for j in T.serial(4):
                    A[i, j] = 2.0


class TestSuppressHoistIfElseExpr(TestHoistIfElseExpr):
    hoisted_conditionals = tvm.testing.parameter(
        HoistedConditionals.All & ~HoistedConditionals.IfElseExpr
    )

    @T.prim_func
    def before(A: T.Buffer((4, 4), "float32")):
        for i, j in T.grid(4, 4):
            A[i, j] = T.if_then_else(i < 2, 1.0, 2.0, dtype="float32")

    expected = before


class TestHoistLetExpr(BaseBeforeAfter):
    @T.prim_func
    def before(A: T.Buffer((4, 4), "float32")):
        for i, j in T.grid(4, 4):
            x = T.float32()
            A[i, j] = T.Let(5.0 * x + T.cast(j, "float32"), where={x: T.cast(i + 1, "float32")})

    @T.prim_func
    def expected(A: T.Buffer((4, 4), "float32")):
        for i in T.serial(4):
            x = T.cast(i + 1, "float32")
            for j in T.serial(4):
                A[i, j] = 5.0 * x + T.cast(j, "float32")


class TestSuppressHoistLetExpr(BaseBeforeAfter):
    hoisted_let_bindings = tvm.testing.parameter(
        HoistedLetBindings.All & ~HoistedLetBindings.LetExpr
    )

    @T.prim_func
    def before(A: T.Buffer((4, 4), "float32")):
        for i, j in T.grid(4, 4):
            x = T.float32()
            A[i, j] = T.Let(5.0 * x + T.cast(j, "float32"), where={x: T.cast(i + 1, "float32")})

    expected = before


if __name__ == "__main__":
    tvm.testing.main()
