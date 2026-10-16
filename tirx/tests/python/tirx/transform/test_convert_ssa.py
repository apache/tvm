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
# ruff: noqa: F401

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx import ir, tirx
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_reuse_in_sequential_bind():
    """De-dup sequential variable bindings"""

    var = T.dynamic("var", "int32")

    @I.ir_module(check_well_formed=False)
    class Before:
        @T.function(private=True)
        def main():
            T.bind(16, var=var)
            T.evaluate(var)
            T.bind(32, var=var)
            T.evaluate(var)

    assert Before["main"].body[0].var.same_as(Before["main"].body[2].var)

    @T.function(private=True)
    def expected():
        var1 = T.bind(T.int32(16))
        T.evaluate(var1)
        var2 = T.bind(T.int32(32))
        T.evaluate(var2)

    After = tvm_tirx.tirx.transform.ConvertSSA()(Before)
    tvm_tirx.ir.assert_structural_equal(After["main"], expected)


def test_reuse_in_nested_bind():
    """Redefinition changes all subsequent uses, including the final sibling."""
    var = T.dynamic("var", "int32")

    @I.ir_module(check_well_formed=False)
    class Before:
        @T.function(private=True)
        def main():
            T.bind(32, var=var)
            T.evaluate(var)
            T.bind(16, var=var)
            T.evaluate(var)
            T.evaluate(var)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            var1 = T.bind(32)
            T.evaluate(var1)
            var2 = T.bind(16)
            T.evaluate(var2)
            T.evaluate(var2)

    assert Before["main"].body[0].var.same_as(Before["main"].body[2].var)
    After = tvm_tirx.tirx.transform.ConvertSSA()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_reused_var_across_module():
    """De-duplicate Var bindings across entire module"""

    @T.function(private=True)
    def func():
        var = T.bind(10)
        T.evaluate(var)

    @I.ir_module(check_well_formed=False)
    class before:
        func_a = func.with_attr("global_symbol", "func_a")
        func_b = func.with_attr("global_symbol", "func_b")

    assert before["func_a"].body.same_as(before["func_b"].body)

    @I.ir_module
    class expected:
        @T.function
        def func_a():
            var: T.let = T.int32(10)
            T.evaluate(var)

        @T.function
        def func_b():
            var: T.let = T.int32(10)
            T.evaluate(var)

    after = tvm_tirx.tirx.transform.ConvertSSA()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_reused_parameter():
    """De-duplicate Var usage in parameters

    In this test, the same `tirx.Var` instance is used for the
    parameter `n` in both functions.
    """

    @T.function(private=True)
    def func(n: T.int32):
        T.evaluate(n)

    @I.ir_module(check_well_formed=False)
    class before:
        func_a = func.with_attr("global_symbol", "func_a")
        func_b = func.with_attr("global_symbol", "func_b")

    assert before["func_a"].body.same_as(before["func_b"].body)

    @I.ir_module
    class expected:
        @T.function
        def func_a(n: T.int32):
            T.evaluate(n)

        @T.function
        def func_b(n: T.int32):
            T.evaluate(n)

    after = tvm_tirx.tirx.transform.ConvertSSA()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_reused_tensor_obj():
    """De-duplicate tensor usage across entire module"""

    @T.function(private=True)
    def func(a: T.handle("float32")):
        A = T.decl_tensor(a, ty_args=[T.Tensor(1, "float32", data=a)])
        T.evaluate(A[0])

    @I.ir_module(check_well_formed=False)
    class before:
        func_a = func.with_attr("global_symbol", "func_a")
        func_b = func.with_attr("global_symbol", "func_b")

    assert before["func_a"].body.same_as(before["func_b"].body)

    @I.ir_module
    class expected:
        @T.function
        def func_a(a: T.handle("float32")):
            A = T.decl_tensor(a, ty_args=[T.Tensor(1, "float32", data=a)])
            T.evaluate(A[0])

        @T.function
        def func_b(a: T.handle("float32")):
            A = T.decl_tensor(a, ty_args=[T.Tensor(1, "float32", data=a)])
            T.evaluate(A[0])

    after = tvm_tirx.tirx.transform.ConvertSSA()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_reused_tensor_parameter():
    """De-duplicate tensor parameters across the entire module."""

    @T.function(private=True)
    def func(A: T.Tensor(1, "float32")):
        T.evaluate(A[0])

    @I.ir_module(check_well_formed=False)
    class before:
        func_a = func.with_attr("global_symbol", "func_a")
        func_b = func.with_attr("global_symbol", "func_b")

    assert before["func_a"].body.same_as(before["func_b"].body)

    @I.ir_module
    class expected:
        @T.function
        def func_a(A: T.Tensor(1, "float32")):
            T.evaluate(A[0])

        @T.function
        def func_b(A: T.Tensor(1, "float32")):
            T.evaluate(A[0])

    after = tvm_tirx.tirx.transform.ConvertSSA()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_reused_compound_tensor_shape_var():
    """De-duplicate implicit Vars nested in tensor parameter shapes."""
    n = T.dynamic("n", "int32")

    @T.function(private=True)
    def func(A: T.Tensor((T.max(n, 1),), "float32", layout=None)):
        T.evaluate(n)

    @I.ir_module(check_well_formed=False)
    class before:
        func_a = func.with_attr("global_symbol", "func_a")
        func_b = func.with_attr("global_symbol", "func_b")

    assert before["func_a"].body.same_as(before["func_b"].body)

    n_a = T.dynamic("n", "int32")
    n_b = T.dynamic("n", "int32")

    @I.ir_module
    class Expected:
        @T.function
        def func_a(A: T.Tensor((T.max(n_a, 1),), "float32", layout=None)):
            T.evaluate(n_a)

        @T.function
        def func_b(A: T.Tensor((T.max(n_b, 1),), "float32", layout=None)):
            T.evaluate(n_b)

    after = tvm_tirx.tirx.transform.ConvertSSA()(before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)
    func_a = after["func_a"]
    func_b = after["func_b"]
    n_a = func_a.params[0].shape[0].a
    n_b = func_b.params[0].shape[0].a
    assert not n_a.same_as(n_b)
    assert n_a.same_as(func_a.body[0].value)
    assert n_b.same_as(func_b.body[0].value)


def test_no_change_if_already_ssa():
    """A module that is already SSA should be unchanged"""

    @I.ir_module
    class before:
        @T.function
        def func(A: T.Tensor(1, "float32")):
            T.evaluate(A[0])

    after = tvm_tirx.tirx.transform.ConvertSSA()(before)
    tvm_tirx.ir.assert_structural_equal(before, after)
    assert before.same_as(after)


def test_keep_duplicate_thread_idx_in_same_function():
    """Sibling launches bind independent lexical variables and are already SSA."""

    @I.ir_module
    class before:
        @T.function
        def main(A: T.Tensor([256], "float32")):
            with T.launch_thread("threadIdx.x", 256) as threadIdx_x:
                A[threadIdx_x] = A[threadIdx_x] + 1.0

            with T.launch_thread("threadIdx.x", 256) as threadIdx_x:
                A[threadIdx_x] = A[threadIdx_x] + 2.0

    after = tvm_tirx.tirx.transform.ConvertSSA()(before)
    tvm_tirx.ir.assert_structural_equal(after, before)


def test_de_duplicate_thread_idx_across_multiple_functions():
    """ConvertSSA separates an explicitly reused region parameter across functions.

    The generic region form deliberately bypasses the fresh-variable launch DSL
    so this fixture still exercises duplicate definitions."""

    threadIdx_x = T.dynamic("threadIdx_x", "int32")

    # threadIdx_x is defined outside
    @I.ir_module(check_well_formed=False)
    class before:
        @T.function
        def kernel_1(A: T.Tensor([256], "float32")):
            T.region(
                "tirx.launch_thread",
                ["threadIdx.x", 256],
                body_params=[threadIdx_x],
            )
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

        @T.function
        def kernel_2(A: T.Tensor([256], "float32")):
            T.region(
                "tirx.launch_thread",
                ["threadIdx.x", 256],
                body_params=[threadIdx_x],
            )
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

    kernel_1_threadIdx_x = T.dynamic("threadIdx_x", "int32")
    kernel_2_threadIdx_x = T.dynamic("threadIdx_x", "int32")

    @I.ir_module
    class expected:
        @T.function
        def kernel_1(A: T.Tensor([256], "float32")):
            T.region(
                "tirx.launch_thread",
                ["threadIdx.x", 256],
                body_params=[kernel_1_threadIdx_x],
            )
            A[kernel_1_threadIdx_x] = A[kernel_1_threadIdx_x] + T.float32(1)

        @T.function
        def kernel_2(A: T.Tensor([256], "float32")):
            T.region(
                "tirx.launch_thread",
                ["threadIdx.x", 256],
                body_params=[kernel_2_threadIdx_x],
            )
            A[kernel_2_threadIdx_x] = A[kernel_2_threadIdx_x] + T.float32(1)

    after = tvm_tirx.tirx.transform.ConvertSSA()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_thread_idx_reused_within_and_across_functions():
    """ConvertSSA separates reused parameters in sibling regions and functions.

    The old function-wide thread identity is retired: each region now defines
    its own lexical parameter, including sibling launches in one function."""

    threadIdx_x = T.dynamic("threadIdx_x", "int32")
    body_params = [threadIdx_x]

    # complaints of multiple definitions of threadIdx_x
    @I.ir_module(check_well_formed=False)
    class before:
        @T.function
        def kernel_1(A: T.Tensor([256], "float32")):
            with T.region(
                "tirx.launch_thread",
                ["threadIdx.x", 256],
                body_params=body_params,
            ):
                A[threadIdx_x] = A[threadIdx_x] + 1.0
            with T.region(
                "tirx.launch_thread",
                ["threadIdx.x", 256],
                body_params=body_params,
            ):
                A[threadIdx_x] = A[threadIdx_x] + 2.0

        @T.function
        def kernel_2(A: T.Tensor([256], "float32")):
            with T.region(
                "tirx.launch_thread",
                ["threadIdx.x", 256],
                body_params=body_params,
            ):
                A[threadIdx_x] = A[threadIdx_x] + 1.0
            with T.region(
                "tirx.launch_thread",
                ["threadIdx.x", 256],
                body_params=body_params,
            ):
                A[threadIdx_x] = A[threadIdx_x] + 2.0

    @I.ir_module
    class expected:
        @T.function
        def kernel_1(A: T.Tensor([256], "float32")):
            with T.launch_thread("threadIdx.x", 256) as threadIdx_x:
                A[threadIdx_x] = A[threadIdx_x] + 1.0
            with T.launch_thread("threadIdx.x", 256) as threadIdx_x:
                A[threadIdx_x] = A[threadIdx_x] + 2.0

        @T.function
        def kernel_2(A: T.Tensor([256], "float32")):
            with T.launch_thread("threadIdx.x", 256) as threadIdx_x:
                A[threadIdx_x] = A[threadIdx_x] + 1.0
            with T.launch_thread("threadIdx.x", 256) as threadIdx_x:
                A[threadIdx_x] = A[threadIdx_x] + 2.0

    after = tvm_tirx.tirx.transform.ConvertSSA()(before)
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_shared_shape_var_in_tensor_params_and_alloc_tensor():
    """Shape var shared across tensor params and AllocTensor should not be renamed.

    When the same Var (e.g., `n`) appears in multiple tensor parameter
    annotations (A and B both have shape [n]), ConvertSSA should not treat
    the second occurrence as a redefinition.  All uses of `n` in the
    function body (including AllocTensor shapes) must remain the same
    Var object so that MakePackedAPI can bind it from the DLTensor shape.
    """
    n = T.dynamic("n", "int32")

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
            C = T.alloc_tensor(ty_args=[T.Tensor((n,), "float32")])
            T.evaluate(1)

    After = tvm_tirx.tirx.transform.ConvertSSA()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Before)
    assert After.same_as(Before)


def test_reused_loop_var_in_decl_tensor_elem_offset():
    """Remap a tensor whose elem_offset depends on an SSA-renamed loop var."""

    @T.function(private=True)
    def template(data: T.handle("float32", "shared.dyn")):
        for i in range(128):
            tensor = T.decl_tensor(
                data,
                ty_args=[
                    T.Tensor((128,), "float32", data=data, elem_offset=i * 128, scope="shared.dyn")
                ],
            )
            T.evaluate(tensor[0])

    shared_loop = template.body[0]
    shared_data = template.params[0]

    @I.ir_module(check_well_formed=False)
    class Before:
        @T.function(private=True)
        def main(data: shared_data):
            T.add_to_parent(shared_loop)
            T.add_to_parent(shared_loop)
            T.add_to_parent(shared_loop)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(data: T.handle("float32", "shared.dyn")):
            for i in range(128):
                tensor = T.decl_tensor(
                    data,
                    ty_args=[
                        T.Tensor(
                            (128,), "float32", data=data, elem_offset=i * 128, scope="shared.dyn"
                        )
                    ],
                )
                T.evaluate(tensor[0])
            for j in range(128):
                tensor = T.decl_tensor(
                    data,
                    ty_args=[
                        T.Tensor(
                            (128,), "float32", data=data, elem_offset=j * 128, scope="shared.dyn"
                        )
                    ],
                )
                T.evaluate(tensor[0])
            for k in range(128):
                tensor = T.decl_tensor(
                    data,
                    ty_args=[
                        T.Tensor(
                            (128,), "float32", data=data, elem_offset=k * 128, scope="shared.dyn"
                        )
                    ],
                )
                T.evaluate(tensor[0])

    assert Before["main"].body[0].same_as(Before["main"].body[1])
    assert Before["main"].body[1].same_as(Before["main"].body[2])
    After = tvm_tirx.tirx.transform.ConvertSSA()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)
    tvm_tirx.tirx.analysis.verify_well_formed(After["main"], assert_mode=True)


if __name__ == "__main__":
    tvm_tirx.testing.main()
