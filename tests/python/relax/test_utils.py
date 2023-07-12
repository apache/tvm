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
from tvm import relax
from tvm.ir.base import assert_structural_equal
from tvm.script.parser import relax as R


def test_copy_with_new_vars():
    @R.function
    def before(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")):
        gv = R.add(x, y)
        return gv

    after = relax.utils.copy_with_new_vars(before)
    assert_structural_equal(after, before)

    assert len(after.params) == len(before.params)
    for before_var, after_var in zip(before.params, after.params):
        assert before_var != after_var


def test_copy_with_new_vars_copied_symbolic_vars():
    @R.function
    def before(x: R.Tensor(("m",), "float32"), y: R.Tensor(("m",), "float32")):
        gv = R.add(x, y)
        return gv

    after = relax.utils.copy_with_new_vars(before)
    assert_structural_equal(after, before)

    assert len(after.params) == len(before.params)
    for before_var, after_var in zip(before.params, after.params):
        assert before_var != after_var
        assert before_var.struct_info.shape[0] != after_var.struct_info.shape[0]


def test_copy_with_new_vars_on_ir_module():
    @tvm.script.ir_module
    class Actual:
        @R.function
        def func(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")):
            gv = R.add(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def func(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")):
            gv = R.add(x, y)
            return gv

        @R.function
        def func_copied(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")):
            gv = R.add(x, y)
            return gv

    Actual["func_copied"] = relax.utils.copy_with_new_vars(Actual["func"]).with_attr(
        "global_symbol", "func_copied"
    )

    # Assertion will fail if the f_copied contains the same VarNode that's used in
    # the original function, due to var mapping during structural equal.
    assert_structural_equal(Actual, Expected)


def test_copy_with_new_vars_on_ir_module_nested_function():
    @tvm.script.ir_module
    class Actual:
        @R.function
        def func(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")):
            @R.function
            def inner(x: R.Tensor((3,), "float32")) -> R.Tensor((3,), dtype="float32"):
                gv = R.add(x, x)
                return gv

            gv = R.add(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def func(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")):
            @R.function
            def inner(x: R.Tensor((3,), "float32")) -> R.Tensor((3,), dtype="float32"):
                gv = R.add(x, x)
                return gv

            gv = R.add(x, y)
            return gv

        @R.function
        def func_copied(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")):
            @R.function
            def inner(x: R.Tensor((3,), "float32")) -> R.Tensor((3,), dtype="float32"):
                gv = R.add(x, x)
                return gv

            gv = R.add(x, y)
            return gv

    Actual["func_copied"] = relax.utils.copy_with_new_vars(Actual["func"]).with_attr(
        "global_symbol", "func_copied"
    )

    assert_structural_equal(Actual, Expected)


if __name__ == "__main__":
    pytest.main([__file__])
