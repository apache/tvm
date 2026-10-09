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
from tvm.script import ir as I
from tvm.script import tirx as T


def test_outermost_independent_loop():
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((4, 4, 4), "int32")):
            for i in T.serial(4):
                for j in T.serial(4):
                    for k in T.serial(4):
                        if i < 2:
                            A[i, j, k] = 1
                        else:
                            A[i, j, k] = 2

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((4, 4, 4), "int32")):
            for i in T.serial(4):
                if i < 2:
                    for j in T.serial(4):
                        for k in T.serial(4):
                            A[i, j, k] = 1
                else:
                    for j in T.serial(4):
                        for k in T.serial(4):
                            A[i, j, k] = 2

    After = tvm.tirx.transform.HoistIf()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_multiple_loop_dependencies():
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((4, 4, 4), "int32")):
            for i in T.serial(4):
                for j in T.serial(4):
                    for k in T.serial(4):
                        if i + j < 2:
                            A[i, j, k] = 1

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((4, 4, 4), "int32")):
            for i in T.serial(4):
                for j in T.serial(4):
                    if i + j < 2:
                        for k in T.serial(4):
                            A[i, j, k] = 1

    After = tvm.tirx.transform.HoistIf()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_nested_conditions():
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((4, 4), "int32"), n: T.int32):
            for i in T.serial(4):
                if i < 2:
                    for j in T.serial(4):
                        if 0 < n:
                            A[i, j] = 1

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((4, 4), "int32"), n: T.int32):
            if 0 < n:
                for i in T.serial(4):
                    if i < 2:
                        for j in T.serial(4):
                            A[i, j] = 1

    # One-sided conditions must not clone the loop before cleanup: a chain of
    # such conditions would otherwise produce exponentially many loop copies.
    inserted = tvm.tirx.transform.HoistIf().passes[0](Before)
    assert inserted["main"].body[0].else_case is None
    After = tvm.tirx.transform.HoistIf()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_dependent_condition():
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((4,), "int32")):
            for i in T.serial(4):
                if i < 2:
                    A[i] = 1

    Expected = Before
    After = tvm.tirx.transform.HoistIf()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_sibling_statements():
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((4, 4), "int32"), n: T.int32):
            for i in T.serial(4):
                A[i, 0] = 0
                for j in T.serial(4):
                    if 0 < n:
                        A[i, j] = 1

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((4, 4), "int32"), n: T.int32):
            for i in T.serial(4):
                A[i, 0] = 0
                if 0 < n:
                    for j in T.serial(4):
                        A[i, j] = 1

    After = tvm.tirx.transform.HoistIf()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_binding_scope():
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((4, 4), "int32")):
            for i in T.serial(4):
                x: T.let[T.int32] = A[i, 0]
                for j in T.serial(4):
                    if 0 < x:
                        A[i, j] = 1

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((4, 4), "int32")):
            for i in T.serial(4):
                x: T.let[T.int32] = A[i, 0]
                if 0 < x:
                    for j in T.serial(4):
                        A[i, j] = 1

    After = tvm.tirx.transform.HoistIf()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_memory_condition():
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((4,), "int32")):
            for i in T.serial(4):
                if 0 < A[0]:
                    A[i] = 1

    Expected = Before
    After = tvm.tirx.transform.HoistIf()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_effectful_condition():
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((4,), "int32")):
            for i in T.serial(4):
                if 0 < T.call_extern("predicate", ty="int32"):
                    A[i] = 1

    Expected = Before
    After = tvm.tirx.transform.HoistIf()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_thread_region_boundary():
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((4, 4), "int32"), n: T.int32):
            for i in T.serial(4):
                with T.launch_thread("threadIdx.x", 4) as tx:
                    for j in T.serial(4):
                        if 0 < n:
                            A[i, tx] = j

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((4, 4), "int32"), n: T.int32):
            for i in T.serial(4):
                with T.launch_thread("threadIdx.x", 4) as tx:
                    if 0 < n:
                        for j in T.serial(4):
                            A[i, tx] = j

    After = tvm.tirx.transform.HoistIf()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_guarded_division():
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((4,), "int32"), n: T.int32):
            for i in T.serial(4):
                if i < n:
                    if 1 < 100 // n:
                        A[i] = 1

    Expected = Before
    After = tvm.tirx.transform.HoistIf()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_zero_trip_division():
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((4,), "int32"), n: T.int32):
            for i in T.serial(n):
                if 1 < 100 // n:
                    A[i] = 1

    Expected = Before
    After = tvm.tirx.transform.HoistIf()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_enclosing_else_is_preserved():
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((4,), "int32"), n: T.int32):
            for i in T.serial(4):
                if i < 2:
                    if 0 < n:
                        A[i] = 1
                else:
                    A[i] = 2

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((4,), "int32"), n: T.int32):
            if 0 < n:
                for i in T.serial(4):
                    if i < 2:
                        A[i] = 1
                    else:
                        A[i] = 2
            else:
                for i in T.serial(4):
                    if not i < 2:
                        A[i] = 2

    After = tvm.tirx.transform.HoistIf()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
