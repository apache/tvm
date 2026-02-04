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
from tvm.script import tir as T, ir_module


def test_before_after_prim_func():
    @T.prim_func(private=True)
    def before():
        T.evaluate(0)

    expected = before

    mod = tvm.IRModule.from_expr(before)
    # Identity transform (no-op)
    mod = (lambda x: x)(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_before_after_method():
    @T.prim_func(private=True)
    def before():
        T.evaluate(0)

    expected = before

    mod = tvm.IRModule.from_expr(before)
    # Identity transform (no-op)
    mod = (lambda x: x)(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_before_after_fixture():
    @T.prim_func(private=True)
    def before():
        T.evaluate(0)

    expected = before

    mod = tvm.IRModule.from_expr(before)
    # Identity transform (no-op)
    mod = (lambda x: x)(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_before_after_delayed_prim_func():
    @T.prim_func(private=True)
    def before():
        T.evaluate(0)

    expected = before

    mod = tvm.IRModule.from_expr(before)
    # Identity transform (no-op)
    mod = (lambda x: x)(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_before_after_parametrized_fixture():
    """Test with different buffer sizes"""
    for n in [1, 8, 16]:

        @T.prim_func(private=True)
        def before(A: T.Buffer(n, "float32")):
            for i in T.serial(n):
                A[i] = 0.0

        expected = before

        mod = tvm.IRModule.from_expr(before)
        # Identity transform (no-op)
        mod = (lambda x: x)(mod)
        tvm.ir.assert_structural_equal(mod["main"], expected)


def test_before_after_ir_module():
    """The preferred form for writing TIR unit tests

    All evaluation is done at test-time, with the minimal amount of
    additional lines.
    """

    @ir_module
    class before:
        @T.prim_func(private=True)
        def func_A(A: T.Buffer(16, "float32")):
            for i in T.serial(16):
                A[i] = 0.0

        @T.prim_func(private=True)
        def func_B(A: T.Buffer(16, "int32")):
            for i in T.serial(16):
                A[i] = 42

    expected = before

    # Identity transform (no-op)
    mod = (lambda x: x)(before)
    tvm.ir.assert_structural_equal(mod, expected)


def test_before_after_ir_module_explicit_fixture():
    """Like test_before_after_ir_module, but with an explicit fixture

    If the IRModule depends on additional fixtures, this form can be
    used.
    """

    @ir_module
    class before:
        @T.prim_func(private=True)
        def func_A(A: T.Buffer(16, "float32")):
            for i in T.serial(16):
                A[i] = 0.0

        @T.prim_func(private=True)
        def func_B(A: T.Buffer(16, "int32")):
            for i in T.serial(16):
                A[i] = 42

    expected = before

    # Identity transform (no-op)
    mod = (lambda x: x)(before)
    tvm.ir.assert_structural_equal(mod, expected)


if __name__ == "__main__":
    tvm.testing.main()
