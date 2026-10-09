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
from tvm.script import tirx as T


def check(before, after):
    actual = tvm.tirx.transform.HoistIf()(tvm.IRModule({"main": before}))
    tvm.ir.assert_structural_equal(actual["main"], after, map_free_vars=True)
    assert tvm.tirx.analysis.verify_well_formed(actual["main"])


def test_outermost_independent_loop():
    @T.function(private=True)
    def Before(A: T.Tensor((4, 4, 4), "int32")):
        for i in T.serial(4):
            for j in T.serial(4):
                for k in T.serial(4):
                    if i < 2:
                        A[i, j, k] = 1
                    else:
                        A[i, j, k] = 2

    @T.function(private=True)
    def After(A: T.Tensor((4, 4, 4), "int32")):
        for i in T.serial(4):
            if i < 2:
                for j in T.serial(4):
                    for k in T.serial(4):
                        A[i, j, k] = 1
            else:
                for j in T.serial(4):
                    for k in T.serial(4):
                        A[i, j, k] = 2

    check(Before, After)


def test_multiple_loop_dependencies():
    @T.function(private=True)
    def Before(A: T.Tensor((4, 4, 4), "int32")):
        for i in T.serial(4):
            for j in T.serial(4):
                for k in T.serial(4):
                    if i + j < 2:
                        A[i, j, k] = 1

    @T.function(private=True)
    def After(A: T.Tensor((4, 4, 4), "int32")):
        for i in T.serial(4):
            for j in T.serial(4):
                if i + j < 2:
                    for k in T.serial(4):
                        A[i, j, k] = 1

    check(Before, After)


def test_nested_conditions():
    @T.function(private=True)
    def Before(A: T.Tensor((4, 4), "int32"), n: T.int32):
        for i in T.serial(4):
            if i < 2:
                for j in T.serial(4):
                    if 0 < n:
                        A[i, j] = 1

    @T.function(private=True)
    def After(A: T.Tensor((4, 4), "int32"), n: T.int32):
        if 0 < n:
            for i in T.serial(4):
                if i < 2:
                    for j in T.serial(4):
                        A[i, j] = 1

    # One-sided conditions must not clone the loop before cleanup: a chain of
    # such conditions would otherwise produce exponentially many loop copies.
    inserted = tvm.tirx.transform.HoistIf().passes[0](tvm.IRModule({"main": Before}))
    assert inserted["main"].body[0].else_case is None
    check(Before, After)


def test_dependent_condition():
    @T.function(private=True)
    def Before(A: T.Tensor((4,), "int32")):
        for i in T.serial(4):
            if i < 2:
                A[i] = 1

    check(Before, Before)


def test_sibling_statements():
    @T.function(private=True)
    def Before(A: T.Tensor((4, 4), "int32"), n: T.int32):
        for i in T.serial(4):
            A[i, 0] = 0
            for j in T.serial(4):
                if 0 < n:
                    A[i, j] = 1

    @T.function(private=True)
    def After(A: T.Tensor((4, 4), "int32"), n: T.int32):
        for i in T.serial(4):
            A[i, 0] = 0
            if 0 < n:
                for j in T.serial(4):
                    A[i, j] = 1

    check(Before, After)


def test_binding_scope():
    @T.function(private=True)
    def Before(A: T.Tensor((4, 4), "int32")):
        for i in T.serial(4):
            x: T.let[T.int32] = A[i, 0]
            for j in T.serial(4):
                if 0 < x:
                    A[i, j] = 1

    @T.function(private=True)
    def After(A: T.Tensor((4, 4), "int32")):
        for i in T.serial(4):
            x: T.let[T.int32] = A[i, 0]
            if 0 < x:
                for j in T.serial(4):
                    A[i, j] = 1

    check(Before, After)


def test_memory_condition():
    @T.function(private=True)
    def Before(A: T.Tensor((4,), "int32")):
        for i in T.serial(4):
            if 0 < A[0]:
                A[i] = 1

    check(Before, Before)


def test_effectful_condition():
    @T.function(private=True)
    def Before(A: T.Tensor((4,), "int32")):
        for i in T.serial(4):
            if 0 < T.call_extern("predicate", ty="int32"):
                A[i] = 1

    check(Before, Before)


def test_thread_region_boundary():
    @T.function(private=True)
    def Before(A: T.Tensor((4, 4), "int32"), n: T.int32):
        for i in T.serial(4):
            with T.launch_thread("threadIdx.x", 4) as tx:
                for j in T.serial(4):
                    if 0 < n:
                        A[i, tx] = j

    @T.function(private=True)
    def After(A: T.Tensor((4, 4), "int32"), n: T.int32):
        for i in T.serial(4):
            with T.launch_thread("threadIdx.x", 4) as tx:
                if 0 < n:
                    for j in T.serial(4):
                        A[i, tx] = j

    check(Before, After)


def test_guarded_division():
    @T.function(private=True)
    def Before(A: T.Tensor((4,), "int32"), n: T.int32):
        for i in T.serial(4):
            if i < n:
                if 1 < 100 // n:
                    A[i] = 1

    check(Before, Before)


def test_zero_trip_division():
    @T.function(private=True)
    def Before(A: T.Tensor((4,), "int32"), n: T.int32):
        for i in T.serial(n):
            if 1 < 100 // n:
                A[i] = 1

    check(Before, Before)


def test_enclosing_else_is_preserved():
    @T.function(private=True)
    def Before(A: T.Tensor((4,), "int32"), n: T.int32):
        for i in T.serial(4):
            if i < 2:
                if 0 < n:
                    A[i] = 1
            else:
                A[i] = 2

    @T.function(private=True)
    def After(A: T.Tensor((4,), "int32"), n: T.int32):
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

    check(Before, After)


if __name__ == "__main__":
    tvm.testing.main()
