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

import pytest

import tvm
import tvm.testing
from tvm.script import ir as I, tir as T


def test_pass_simple():
    @T.prim_func
    def element_wise(
        A: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        B = T.alloc_buffer((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with T.block("C"):
                # It's a opaque block , so it can use outside variables
                C[i, j] = B[i, j] * 2.0

    assert tvm.tir.analysis.verify_well_formed(element_wise)
    assert tvm.tir.analysis.verify_well_formed(tvm.IRModule.from_expr(element_wise))


def test_fail_use_out_loop_var():
    @T.prim_func
    def element_wise(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
    ):
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                # we cannot use `i` since it's defined outside the block
                B[vi, vj] = A[i, vj] * 2.0

    assert not tvm.tir.analysis.verify_well_formed(element_wise, assert_mode=False)


def test_error_for_out_of_scope_usage():
    """A variable may not be used after its scope ends"""

    @T.prim_func
    def func():
        i = T.int32()
        with T.LetStmt(42, var=i):
            T.evaluate(i)
        T.evaluate(i)

    with pytest.raises(ValueError, match="Invalid use of variable i at .* no longer in-scope."):
        tvm.tir.analysis.verify_well_formed(func)


def test_error_for_nested_rebind_usage():
    """A variable may not be re-defined within the initial scope"""

    @T.prim_func
    def func():
        i = T.int32()
        with T.LetStmt(42, var=i):
            with T.LetStmt(42, var=i):
                T.evaluate(i)

    with pytest.raises(
        ValueError, match="ill-formed, due to multiple nested definitions of variable i"
    ):
        tvm.tir.analysis.verify_well_formed(func)


def test_error_for_repeated_binding():
    """A variable may not be re-defined after the scope ends"""

    @T.prim_func
    def func():
        i = T.int32()
        with T.LetStmt(42, var=i):
            T.evaluate(i)
        with T.LetStmt(17, var=i):
            T.evaluate(i)

    with pytest.raises(ValueError, match="multiple definitions of variable i"):
        tvm.tir.analysis.verify_well_formed(func)


def test_error_for_cross_function_reuse():
    """A variable may not be re-defined in another function"""

    i = tvm.tir.Var("i", "int32")

    @I.ir_module
    class mod:
        @T.prim_func
        def func1():
            with T.LetStmt(42, var=i):
                T.evaluate(i)

        @T.prim_func
        def func2():
            with T.LetStmt(42, var=i):
                T.evaluate(i)

    with pytest.raises(ValueError, match="multiple definitions of variable i"):
        tvm.tir.analysis.verify_well_formed(mod)


if __name__ == "__main__":
    tvm.testing.main()
