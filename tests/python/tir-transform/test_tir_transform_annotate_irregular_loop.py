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

"""Tests for AnnotateIrregularLoop"""

import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T


def test_handle_irrgular_unit_loop():
    """Dedicated testcase to check the unitloop with loop jump not simplified"""

    @T.prim_func
    def before(A: T.Buffer((10,), "int32")):
        for i in T.serial(1):
            if A[i] > 5:
                break
            A[i] = A[i] + 1
        for j in T.serial(1):
            if A[j] > 5:
                continue
            A[j] = A[j] + 1
        for k in T.serial(1):
            A[k] = A[k] + 1

    @T.prim_func
    def expected(A: T.Buffer((10,), "int32")):
        for i in T.serial(1, annotations={"irregular_loop_mark": 1}):
            if A[i] > 5:
                break
            A[i] = A[i] + 1
        for j in T.serial(1, annotations={"irregular_loop_mark": 1}):
            if A[j] > 5:
                continue
            A[j] = A[j] + 1
        A[0] = A[0] + 1

    mod = tvm.IRModule.from_expr(before)
    mod = tvm.tir.transform.AnnotateIrregularLoop()(mod)
    mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
    tvm.ir.assert_structural_equal(mod["before"].with_attr("global_symbol", "expected"), expected)


class BaseCompare(tvm.testing.CompareBeforeAfter):
    transform = tir.transform.AnnotateIrregularLoop()


class TestAnnotateLoopWithBreak(BaseCompare):
    """Test that loops containing break statements are annotated as irregular."""

    def before(A: T.Buffer((10,), "int32")):
        for i in T.serial(10):
            if A[i] > 5:
                break
            A[i] = A[i] + 1

    def expected(A: T.Buffer((10,), "int32")):
        for i in T.serial(10, annotations={"irregular_loop_mark": 1}):
            if A[i] > 5:
                break
            A[i] = A[i] + 1


class TestAnnotateLoopWithContinue(BaseCompare):
    """Test that loops containing continue statements are annotated as irregular."""

    def before(A: T.Buffer((10,), "int32")):
        for i in T.serial(10):
            if A[i] < 0:
                continue
            A[i] = A[i] * 2

    def expected(A: T.Buffer((10,), "int32")):
        for i in T.serial(10, annotations={"irregular_loop_mark": 1}):
            if A[i] < 0:
                continue
            A[i] = A[i] * 2


class TestNestedIrregularBothLoops(BaseCompare):
    """Test nested loops where both loops have break/continue."""

    def before(A: T.Buffer((10, 10), "int32")):
        for i in T.serial(10):
            if i > 7:
                break
            for j in T.serial(10):
                if A[i, j] < 0:
                    continue
                A[i, j] = A[i, j] + 1

    def expected(A: T.Buffer((10, 10), "int32")):
        for i in T.serial(10, annotations={"irregular_loop_mark": 1}):
            if i > 7:
                break
            for j in T.serial(10, annotations={"irregular_loop_mark": 1}):
                if A[i, j] < 0:
                    continue
                A[i, j] = A[i, j] + 1


class TestWhileLoopWithBreak(BaseCompare):
    """Test that while loops with break/continue are not annotated (while loops don't have annotations)."""

    def before(A: T.Buffer((10,), "int32")):
        i = T.int32(0)
        while i < 10:
            if A[i] > 5:
                break
            A[i] = A[i] + 1
            i = i + 1

    def expected(A: T.Buffer((10,), "int32")):
        i = T.int32(0)
        while i < 10:
            if A[i] > 5:
                break
            A[i] = A[i] + 1
            i = i + 1


class TestBreakInNestedConditional(BaseCompare):
    """Test break statement deeply nested in conditional blocks."""

    def before(A: T.Buffer((10,), "int32"), flag1: T.int32, flag2: T.int32):
        for i in T.serial(10):
            if flag1 > 0:
                if flag2 > 0:
                    if A[i] > 5:
                        break
            A[i] = A[i] + 1

    def expected(A: T.Buffer((10,), "int32"), flag1: T.int32, flag2: T.int32):
        for i in T.serial(10, annotations={"irregular_loop_mark": 1}):
            if flag1 > 0:
                if flag2 > 0:
                    if A[i] > 5:
                        break
            A[i] = A[i] + 1


class TestWhileLoopWithBreakStandalone(BaseCompare):
    """Test that while loops with break/continue are not annotated (while loops don't have annotations)."""

    def before(A: T.Buffer((10,), "int32")):
        i = T.int32(0)
        while i < 10:
            if A[i] > 5:
                break
            A[i] = A[i] + 1
            i = i + 1

    def expected(A: T.Buffer((10,), "int32")):
        i = T.int32(0)
        while i < 10:
            if A[i] > 5:
                break
            A[i] = A[i] + 1
            i = i + 1


class TestNestedIrregularLoopStandalone(BaseCompare):
    """Test deeply nested loops with irregular control flow only in innermost loop."""

    def before(A: T.Buffer((5, 5, 5), "int32")):
        for i in T.serial(5):
            for j in T.serial(5):
                for k in T.serial(5):
                    if A[i, j, k] > 10:
                        break
                    if A[i, j, k] < 0:
                        continue
                    A[i, j, k] = A[i, j, k] + 1

    def expected(A: T.Buffer((5, 5, 5), "int32")):
        for i in T.serial(5):
            for j in T.serial(5):
                for k in T.serial(5, annotations={"irregular_loop_mark": 1}):
                    if A[i, j, k] > 10:
                        break
                    if A[i, j, k] < 0:
                        continue
                    A[i, j, k] = A[i, j, k] + 1


if __name__ == "__main__":
    tvm.testing.main()
