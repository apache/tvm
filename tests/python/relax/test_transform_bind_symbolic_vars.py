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
import tvm.script
import tvm.testing
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def test_bind_tensors():
    """Symbolic variables may occur in Tensor shapes"""

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(("batch", "m"), dtype="float32"),
            w0: R.Tensor(("m", "n"), dtype="float32"),
            w1: R.Tensor(("k", 10), dtype="float32"),
        ) -> R.Tensor(("batch", "k"), dtype="float32"):
            batch = T.Var("batch", "int64")
            n = T.Var("n", "int64")
            k = T.Var("k", "int64")
            with R.dataflow():
                lv0 = R.call_dps_packed(
                    "test0", (x, w0), out_sinfo=R.Tensor((batch, n), dtype="float32")
                )
                out = R.call_dps_packed(
                    "test1", (lv0, w1), out_sinfo=R.Tensor((batch, k), dtype="float32")
                )
                R.output(out)
            return out

    symvar_map = {"batch": 1, "k": 3}
    target_func_name = "main"
    After = relax.transform.BindSymbolicVars(symvar_map, target_func_name)(Before)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, "m"), dtype="float32"),
            w0: R.Tensor(("m", "n"), dtype="float32"),
            w1: R.Tensor((3, 10), dtype="float32"),
        ) -> R.Tensor((1, 3), dtype="float32"):
            n = T.int64()
            with R.dataflow():
                lv0 = R.call_dps_packed(
                    "test0", (x, w0), out_sinfo=R.Tensor((1, n), dtype="float32")
                )
                out = R.call_dps_packed(
                    "test1", (lv0, w1), out_sinfo=R.Tensor((1, 3), dtype="float32")
                )
                R.output(out)
            return out

    tvm.ir.assert_structural_equal(Expected, After)


def test_bind_shape():
    """Symbolic variables may occur in ShapeExpr"""

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Shape(("batch", "m")),
            w0: R.Shape(("m", "n")),
            w1: R.Shape(("k", 10)),
        ) -> R.Shape(("batch", "k")):
            batch = T.Var("batch", "int64")
            n = T.Var("n", "int64")
            k = T.Var("k", "int64")
            with R.dataflow():
                lv0 = R.call_dps_packed("test0", (x, w0), out_sinfo=R.Tensor((batch, n)))
                out = R.call_dps_packed("test1", (lv0, w1), out_sinfo=R.Tensor((batch, k)))
                R.output(out)
            return out

    symvar_map = {"batch": 1, "k": 3}
    target_func_name = "main"
    After = relax.transform.BindSymbolicVars(symvar_map, target_func_name)(Before)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Shape([1, "m"]), w0: R.Shape(["m", "n"]), w1: R.Shape([3, 10])
        ) -> R.Shape([1, 3]):
            n = T.int64()
            with R.dataflow():
                lv0 = R.call_dps_packed("test0", (x, w0), out_sinfo=R.Tensor((1, n)))
                out = R.call_dps_packed("test1", (lv0, w1), out_sinfo=R.Tensor((1, 3)))
                R.output(out)
            return out

    tvm.ir.assert_structural_equal(Expected, After)


def test_arith():
    """Symbolic shapes may use TIR arithmetic expressions"""

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(("batch", "m-1"), dtype="float32"),
            w0: R.Tensor(("m", "n"), dtype="float32"),
            w1: R.Tensor(("k", 10), dtype="float32"),
        ) -> R.Tensor(("batch", "k*m"), dtype="float32"):
            batch = T.Var("batch", "int64")
            m = T.Var("m", "int64")
            n = T.Var("n", "int64")
            k = T.Var("k", "int64")
            with R.dataflow():
                lv0 = R.call_dps_packed(
                    "test0",
                    (x, w0),
                    out_sinfo=R.Tensor((batch, m + n), dtype="float32"),
                )
                out = R.call_dps_packed(
                    "test1",
                    (lv0, w1),
                    out_sinfo=R.Tensor((batch, k + n), dtype="float32"),
                )
                R.output(out)
            return out

    symvar_map = {"batch": 1, "k": 2, "m": 3}
    target_func_name = "main"
    After = relax.transform.BindSymbolicVars(symvar_map, target_func_name)(Before)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 2), dtype="float32"),
            w0: R.Tensor((3, "n"), dtype="float32"),
            w1: R.Tensor((2, 10), dtype="float32"),
        ) -> R.Tensor((1, 6), dtype="float32"):
            n = T.int64()
            with R.dataflow():
                lv0 = R.call_dps_packed(
                    "test0", (x, w0), out_sinfo=R.Tensor((1, n + 3), dtype="float32")
                )
                out = R.call_dps_packed(
                    "test1", (lv0, w1), out_sinfo=R.Tensor((1, n + 2), dtype="float32")
                )
                R.output(out)
            return out

    tvm.ir.assert_structural_equal(Expected, After)


def test_bind_multiple_variables_by_name():
    """String names may be used to replace across multiple functions"""

    @tvm.script.ir_module
    class Before:
        @R.function
        def main_1(x: R.Tensor(("m", "n"), dtype="float32")):
            return x

        @R.function
        def main_2(x: R.Tensor(("m", "n"), dtype="float32")):
            return x

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main_1(x: R.Tensor(("m", 16), dtype="float32")):
            return x

        @R.function
        def main_2(x: R.Tensor(("m", 16), dtype="float32")):
            return x

    After = relax.transform.BindSymbolicVars({"n": 16})(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_bind_single_variable_by_identity():
    """TIR variables may be used to replace a specific var"""

    @tvm.script.ir_module
    class Before:
        @R.function
        def main_1(x: R.Tensor(("m", "n"), dtype="float32")):
            return x

        @R.function
        def main_2(x: R.Tensor(("m", "n"), dtype="float32")):
            return x

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main_1(x: R.Tensor(("m", 16), dtype="float32")):
            return x

        @R.function
        def main_2(x: R.Tensor(("m", "n"), dtype="float32")):
            return x

    main_1_n = Before["main_1"].params[0].struct_info.shape[1]
    After = relax.transform.BindSymbolicVars({main_1_n: 16})(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_bind_single_variable_by_function_name():
    """Variable name and function name may be used to replace a specific var"""

    @tvm.script.ir_module
    class Before:
        @R.function
        def main_1(x: R.Tensor(("m", "n"), dtype="float32")):
            return x

        @R.function
        def main_2(x: R.Tensor(("m", "n"), dtype="float32")):
            return x

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main_1(x: R.Tensor(("m", 16), dtype="float32")):
            return x

        @R.function
        def main_2(x: R.Tensor(("m", "n"), dtype="float32")):
            return x

    After = relax.transform.BindSymbolicVars({"n": 16}, "main_1")(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_error_for_unused_replacement():
    """Each replacement must be used"""

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor(("m", "n"), dtype="float32")):
            return x

    with pytest.raises(tvm.TVMError):
        relax.transform.BindSymbolicVars({"non_existing_var_name": 16})(Before)


if __name__ == "__main__":
    tvm.testing.main()
