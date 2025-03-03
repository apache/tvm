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
from tvm import relax
import tvm.script
from tvm.script import relax as R, tir as T, ir as I
from tvm.relax import transform
from tvm.ir.base import assert_structural_equal


def _check_equal(x, y):
    tvm.ir.assert_structural_equal(x, y)
    tvm.ir.assert_structural_equal(y, x)

    xhash = tvm.ir.structural_hash(x, map_free_vars=True)
    yhash = tvm.ir.structural_hash(y, map_free_vars=True)
    assert xhash == yhash


def _check_save_roundtrip(x):
    y = tvm.ir.load_json(tvm.ir.save_json(x))
    _check_equal(x, y)


def test_basic():
    """Functions can be listed from local bindings to the IRModule"""

    # the target IRModule
    @I.ir_module
    class Expected:
        @R.function(private=True)
        def main_inner(
            x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
            return s

        @R.function
        def main(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            gv1: R.Tensor((10, 5), "float32") = Expected.main_inner(x1, y1)
            return gv1

    @I.ir_module
    class Before:
        @R.function
        def main(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            @R.function
            def inner(
                x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
            ) -> R.Tensor((10, 5), "float32"):
                s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
                return s

            gv1: R.Tensor((10, 5), "float32") = inner(x1, y1)
            return gv1

    before = Before
    expected = Expected
    # Perform Lambda Lifting
    after = transform.LambdaLift()(before)
    assert len(after.functions) == 2
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_input_module_is_unmodified():
    """The input module may not be modified

    If the output requires new StructInfo, it must create a new relax
    variable.  It must not update the struct info of an existing relax
    variable, as that variable may be used by another IRModule.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
        ) -> R.Tensor((2, 3), "float32"):
            @R.function
            def outer_func(
                c1: R.Tensor((2, 3), "float32")
            ) -> R.Callable((R.Tensor((2, 3), "float32"),), R.Tensor((2, 3), "float32")):
                @R.function
                def inner_func(x1: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
                    s: R.Tensor((2, 3), "float32") = R.add(x1, c1)
                    return s

                return inner_func

            in_call = outer_func(x)
            res = in_call(y)
            return res

    before = Before
    copy_of_before = tvm.ir.load_json(tvm.ir.save_json(before))

    transform.LambdaLift()(before)

    tvm.ir.assert_structural_equal(before, copy_of_before)


def test_closure():
    """Lifting functions may require producing closures"""

    # the expected IRModule
    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
        ) -> R.Tensor((2, 3), "float32"):
            in_call = Expected.main_outer_func(x)
            res = R.invoke_pure_closure(
                in_call, (y,), sinfo_args=(R.Tensor((2, 3), dtype="float32"))
            )
            return res

        @R.function(private=True)
        def main_inner_func(x1: R.Tensor((2, 3), "float32"), c1: R.Tensor((2, 3), "float32")):
            r_1: R.Tensor((2, 3), "float32") = R.add(x1, c1)
            return r_1

        @R.function(private=True)
        def main_outer_func(y: R.Tensor((2, 3), "float32")) -> R.Object:
            inner_func = R.make_closure(Expected.main_inner_func, (y,))
            return inner_func

    # IRModule to perform Lambda Lifting
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
        ) -> R.Tensor((2, 3), "float32"):
            @R.function
            def outer_func(
                c1: R.Tensor((2, 3), "float32")
            ) -> R.Callable((R.Tensor((2, 3), "float32"),), R.Tensor((2, 3), "float32")):
                @R.function
                def inner_func(x1: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
                    s: R.Tensor((2, 3), "float32") = R.add(x1, c1)
                    return s

                return inner_func

            in_call = outer_func(x)
            res = in_call(y)
            return res

    before = Before
    after = transform.LambdaLift()(before)
    expected = Expected
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_recursive():
    """The lifted function may be recursively defined"""

    # the expected IRModule
    @I.ir_module
    class Expected:
        @R.function(private=True)
        def main_while_loop(
            i: R.Tensor((), "int32"), s: R.Tensor((2, 3), "float32"), x: R.Tensor((2, 3), "float32")
        ) -> R.Tensor((2, 3), "float32"):
            cond: R.Tensor((), "bool") = R.call_pure_packed(
                "test.vm.less", i, R.const(10), sinfo_args=(R.Tensor((), dtype="bool"))
            )
            c: R.Tensor((), "int32") = R.const(1, dtype="int32")
            if cond:
                new_i: R.Tensor((), "int32") = R.add(i, c)
                new_s: R.Tensor((2, 3), "float32") = R.add(s, x)
                new_r = Expected.main_while_loop(new_i, new_s, x)
                r = new_r
            else:
                r = s
            return r

        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), dtype="float32"):
            while_loop = R.make_closure(Expected.main_while_loop, (x,))
            gv: R.Tensor((2, 3), dtype="float32") = R.invoke_pure_closure(
                while_loop,
                (R.const(0), x),
                sinfo_args=(R.Tensor((2, 3), dtype="float32")),
            )
            return gv

    # the IRModule to apply lambda lifting
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor:
            @R.function
            def while_loop(
                i: R.Tensor((), "int32"), s: R.Tensor((2, 3), "float32")
            ) -> R.Tensor((2, 3), "float32"):
                cond: R.Tensor((), "bool") = R.call_pure_packed(
                    "test.vm.less", i, R.const(10), sinfo_args=(R.Tensor((), dtype="bool"))
                )
                c: R.Tensor((), "int32") = R.const(1, dtype="int32")
                if cond:
                    new_i: R.Tensor((), "int32") = R.add(i, c)
                    new_s: R.Tensor((2, 3), "float32") = R.add(s, x)
                    r: R.Tensor((2, 3), "float32") = while_loop(new_i, new_s)
                else:
                    r: R.Tensor((2, 3), "float32") = s
                return r

            gv: R.Tensor((2, 3), "float32") = while_loop(R.const(0), x)
            return gv

    before = Before
    expected = Expected
    # check well-formness of recursive call
    assert relax.analysis.well_formed(before)

    # Perform Lambda Lifting
    after = transform.LambdaLift()(before)
    assert len(after.functions) == 2

    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_multi_func():
    """Lifting may be required for multiple top-level functions

    De-duplication of GlobalVar names at the IRModule is done by
    appending the name of the function from which they were lifted.
    """

    # expected IRModule
    @I.ir_module
    class Expected:
        @R.function
        def glob_func_1(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv1: R.Tensor((10, 5), "float32") = Expected.glob_func_1_inner(x1, y1)
            return gv1

        @R.function
        def glob_func_2(
            x11: R.Tensor((10, 5), "float32"), y11: R.Tensor((10, 5), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv11: R.Tensor((10, 5), "float32") = Expected.glob_func_2_inner(x11, y11)
            return gv11

        @R.function(private=True)
        def glob_func_1_inner(
            x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
            return s

        @R.function(private=True)
        def glob_func_2_inner(
            x21: R.Tensor((10, 5), "float32"), y21: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            s1: R.Tensor((10, 5), "float32") = R.add(x21, y21)
            return s1

    # the IRModule to apply lambda lifting
    @I.ir_module
    class Before:
        @R.function
        def glob_func_1(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            @R.function
            def inner(
                x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
            ) -> R.Tensor((10, 5), "float32"):
                s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
                return s

            gv1: R.Tensor((10, 5), "float32") = inner(x1, y1)
            return gv1

        @R.function
        def glob_func_2(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            @R.function
            def inner(
                x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
            ) -> R.Tensor((10, 5), "float32"):
                s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
                return s

            gv1: R.Tensor((10, 5), "float32") = inner(x1, y1)
            return gv1

    before = Before
    expected = Expected
    # Perform Lambda Lifting
    after = transform.LambdaLift()(before)
    assert len(after.functions) == 4
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_no_local_func():
    @I.ir_module
    class Before:
        @T.prim_func
        def sub(
            A: T.Buffer((16, 16), "float32"),
            B: T.Buffer((16, 16), "float32"),
            C: T.Buffer((16, 16), "float32"),
        ) -> None:
            for i, j in T.grid(16, 16):
                with T.block("sub"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = A[vi, vj] - B[vi, vj]

        @R.function
        def before(c0: R.Tensor((16, 16), "float32"), x: R.Tensor(dtype="float32", ndim=2)):
            s = R.call_tir(Before.sub, (c0, x), R.Tensor((16, 16), dtype="float32"))
            return s

    before = Before
    # Perform lambda lifting
    after = transform.LambdaLift()(before)
    # No local functions are lifted
    assert_structural_equal(after, before, map_free_vars=True)
    _check_save_roundtrip(after)


def test_impure_function():
    @I.ir_module
    class Expected:
        @R.function(pure=False, private=True)
        def main_inner() -> R.Tuple:
            y = R.print(format="Wow!")
            return y

        @R.function(pure=False)
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            gv1 = Expected.main_inner()
            return x

    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            @R.function(pure=False)
            def inner() -> R.Tuple:
                y = R.print(format="Wow!")
                return y

            gv1 = inner()
            return x

    before = Before
    expected = Expected
    after = transform.LambdaLift()(before)
    assert len(after.functions) == 2
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_lambda_function_with_same_name_as_global():
    """Lifted lambda names may not conflict with previous names

    Like `test_basic`, but the module has an existing function
    `main_inner`, which has the same name as the LambdaLift's first
    choice of name for the hoisted function.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            @R.function
            def inner(
                x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
            ) -> R.Tensor((10, 5), "float32"):
                s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
                return s

            gv1: R.Tensor((10, 5), "float32") = inner(x1, y1)
            return gv1

        @R.function
        def main_inner():
            return R.tuple()

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            gv1: R.Tensor((10, 5), "float32") = Expected.main_inner_0(x1, y1)
            return gv1

        @R.function(private=True)
        def main_inner_0(
            x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
            return s

        @R.function
        def main_inner():
            return R.tuple()

    after = transform.LambdaLift()(Before)
    assert_structural_equal(Expected, after)


def test_symbolic_variable_defined_by_inner_func():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            @R.function
            def inner(x2: R.Tensor(("n", "m"), "float32"), y2: R.Tensor(("n", "m"), "float32")):
                sum_inner = R.add(x2, y2)
                return sum_inner

            sum_main = inner(x1, y1)
            return sum_main

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            sum_main = Expected.main_inner(x1, y1)
            return sum_main

        @R.function(private=True)
        def main_inner(
            x2: R.Tensor(("n", "m"), "float32"), y2: R.Tensor(("n", "m"), "float32")
        ) -> R.Tensor(("n", "m"), "float32"):
            sum_inner = R.add(x2, y2)
            return sum_inner

    After = transform.LambdaLift()(Before)
    assert_structural_equal(Expected, After)


def test_symbolic_variable_defined_by_outer_func():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x1: R.Tensor(("n", "m"), "float32"), y1: R.Tensor(("n", "m"), "float32")
        ) -> R.Tensor(("n", "m"), "float32"):
            n = T.int64()
            m = T.int64()

            @R.function
            def inner(x2: R.Tensor((n, m), "float32"), y2: R.Tensor((n, m), "float32")):
                sum_inner = R.add(x2, y2)
                return sum_inner

            sum_main = inner(x1, y1)
            return sum_main

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x1: R.Tensor(("n", "m"), "float32"), y1: R.Tensor(("n", "m"), "float32")
        ) -> R.Tensor(("n", "m"), "float32"):
            sum_main = Expected.main_inner(x1, y1)
            return sum_main

        @R.function(private=True)
        def main_inner(
            x2: R.Tensor(("n", "m"), "float32"), y2: R.Tensor(("n", "m"), "float32")
        ) -> R.Tensor(("n", "m"), "float32"):
            sum_inner = R.add(x2, y2)
            return sum_inner

    After = transform.LambdaLift()(Before)
    assert_structural_equal(Expected, After)


if __name__ == "__main__":
    tvm.testing.main()
