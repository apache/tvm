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

import tvm
import tvm.testing
from tvm import ir, tirx
from tvm.script import ir as I
from tvm.script import tirx as T


def test_reuse_in_sequential_bind():
    """De-dup sequential variable bindings"""

    # Manually construct the Function body, as SSA violations are
    # not valid TIR, and may not be expressible in future versions
    # of TVMSCript.
    var = tirx.Var("var", "int32")
    sequential_bindings = tirx.SeqStmt(
        [
            tirx.Bind(var, 16),
            tirx.Evaluate(var),
            tirx.Bind(var, 32),
            tirx.Evaluate(var),
        ]
    )
    before = tirx.Function([], sequential_bindings)

    @T.function(private=True)
    def expected():
        var1 = T.bind(T.int32(16))
        T.evaluate(var1)
        var2 = T.bind(T.int32(32))
        T.evaluate(var2)

    mod = tvm.IRModule.from_expr(before)
    mod = tvm.tirx.transform.ConvertSSA()(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_reuse_in_nested_bind():
    """De-dup sequential bindings of the same variable.

    In the flat Bind model, all Binds are siblings in a SeqStmt. A second
    Bind of the same variable redefines it for all subsequent siblings.
    ConvertSSA should create a new variable for the second binding and
    update all subsequent uses to refer to the new variable.
    """

    # Manually construct the Function body, as SSA violations are
    # not valid TIR, and may not be expressible in future versions
    # of TVMScript.
    var = tirx.Var("var", "int32")
    # Note: nested SeqStmt is flattened by the IR builder, so the input
    # is actually a flat SeqStmt with 5 elements.
    inner_seq = tirx.SeqStmt(
        [
            tirx.Bind(var, 16),
            tirx.Evaluate(var),
        ]
    )
    outer_seq = tirx.SeqStmt(
        [
            tirx.Bind(var, 32),
            tirx.Evaluate(var),
            inner_seq,
            tirx.Evaluate(var),
        ]
    )
    before = tirx.Function([], outer_seq)

    # In the flat model, the second Bind(var, 16) redefines var for
    # ALL subsequent siblings including the last Evaluate.
    var1 = tirx.Var("var", "int32")
    var2 = tirx.Var("var", "int32")
    expected_body = tirx.SeqStmt(
        [
            tirx.Bind(var1, 32),
            tirx.Evaluate(var1),
            tirx.Bind(var2, 16),
            tirx.Evaluate(var2),
            tirx.Evaluate(var2),
        ]
    )
    expected = tirx.Function([], expected_body)

    mod = tvm.IRModule.from_expr(before)
    mod = tvm.tirx.transform.ConvertSSA()(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_reused_var_across_module():
    """De-duplicate Var bindings across entire module"""

    @T.function(private=True)
    def func():
        var = T.bind(10)
        T.evaluate(var)

    before = tvm.IRModule(
        {
            "func_a": func.with_attr("global_symbol", "func_a"),
            "func_b": func.with_attr("global_symbol", "func_b"),
        }
    )

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

    after = tvm.tirx.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_reused_parameter():
    """De-duplicate Var usage in parameters

    In this test, the same `tirx.Var` instance is used for the
    parameter `n` in both functions.
    """

    @T.function(private=True)
    def func(n: T.int32):
        T.evaluate(n)

    before = tvm.IRModule(
        {
            "func_a": func.with_attr("global_symbol", "func_a"),
            "func_b": func.with_attr("global_symbol", "func_b"),
        }
    )

    @I.ir_module
    class expected:
        @T.function
        def func_a(n: T.int32):
            T.evaluate(n)

        @T.function
        def func_b(n: T.int32):
            T.evaluate(n)

    after = tvm.tirx.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_reused_buffer_obj():
    """De-duplicate buffer usage across entire module"""

    @T.function(private=True)
    def func(a: T.handle("float32")):
        A = T.decl_tensor(shape=1, dtype="float32", data=a)
        T.evaluate(A[0])

    before = tvm.IRModule(
        {
            "func_a": func.with_attr("global_symbol", "func_a"),
            "func_b": func.with_attr("global_symbol", "func_b"),
        }
    )

    @I.ir_module
    class expected:
        @T.function
        def func_a(a: T.handle("float32")):
            A = T.decl_tensor(shape=1, dtype="float32", data=a)
            T.evaluate(A[0])

        @T.function
        def func_b(a: T.handle("float32")):
            A = T.decl_tensor(shape=1, dtype="float32", data=a)
            T.evaluate(A[0])

    after = tvm.tirx.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_reused_buffer_parameter():
    """De-duplicate buffer parameters across the entire module."""

    @T.function(private=True)
    def func(A: T.Tensor(1, "float32")):
        T.evaluate(A[0])

    before = tvm.IRModule(
        {
            "func_a": func.with_attr("global_symbol", "func_a"),
            "func_b": func.with_attr("global_symbol", "func_b"),
        }
    )

    @I.ir_module
    class expected:
        @T.function
        def func_a(A: T.Tensor(1, "float32")):
            T.evaluate(A[0])

        @T.function
        def func_b(A: T.Tensor(1, "float32")):
            T.evaluate(A[0])

    after = tvm.tirx.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_reused_compound_buffer_shape_var():
    """De-duplicate implicit Vars nested in buffer parameter shapes."""
    n = tirx.Var("n", "int32")
    A = tirx.decl_tensor((tirx.max(n, 1),), layout=None)
    func = tirx.Function([A], tirx.Evaluate(n))
    before = tvm.IRModule(
        {
            "func_a": func.with_attr("global_symbol", "func_a"),
            "func_b": func.with_attr("global_symbol", "func_b"),
        }
    )

    after = tvm.tirx.transform.ConvertSSA()(before)
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

    after = tvm.tirx.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(before, after)
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

    after = tvm.tirx.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, before)


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
                [tvm.ir.StringImm("threadIdx.x"), 256],
                body_params=[threadIdx_x],
            )
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

        @T.function
        def kernel_2(A: T.Tensor([256], "float32")):
            T.region(
                "tirx.launch_thread",
                [tvm.ir.StringImm("threadIdx.x"), 256],
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
                [tvm.ir.StringImm("threadIdx.x"), 256],
                body_params=[kernel_1_threadIdx_x],
            )
            A[kernel_1_threadIdx_x] = A[kernel_1_threadIdx_x] + T.float32(1)

        @T.function
        def kernel_2(A: T.Tensor([256], "float32")):
            T.region(
                "tirx.launch_thread",
                [tvm.ir.StringImm("threadIdx.x"), 256],
                body_params=[kernel_2_threadIdx_x],
            )
            A[kernel_2_threadIdx_x] = A[kernel_2_threadIdx_x] + T.float32(1)

    after = tvm.tirx.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_de_duplicate_thread_idx_iter_var_across_multiple_functions():
    """ConvertSSA separates a shared parameter list reused across functions.

    Launch regions no longer wrap their bindings in IterVars; this retains the
    existing shared-wrapper fixture using the explicit body-parameter list."""

    threadIdx_x = T.dynamic("threadIdx_x", "int32")
    body_params = [threadIdx_x]

    # complaints of multiple definitions for threadIdx_x
    @I.ir_module(check_well_formed=False)
    class before:
        @T.function
        def kernel_1(A: T.Tensor([256], "float32")):
            T.region(
                "tirx.launch_thread",
                [tvm.ir.StringImm("threadIdx.x"), 256],
                body_params=body_params,
            )
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

        @T.function
        def kernel_2(A: T.Tensor([256], "float32")):
            T.region(
                "tirx.launch_thread",
                [tvm.ir.StringImm("threadIdx.x"), 256],
                body_params=body_params,
            )
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

    kernel_1_threadIdx_x = T.dynamic("threadIdx_x", "int32")
    kernel_2_threadIdx_x = T.dynamic("threadIdx_x", "int32")

    @I.ir_module(check_well_formed=False)
    class expected:
        @T.function
        def kernel_1(A: T.Tensor([256], "float32")):
            T.region(
                "tirx.launch_thread",
                [tvm.ir.StringImm("threadIdx.x"), 256],
                body_params=[kernel_1_threadIdx_x],
            )
            A[kernel_1_threadIdx_x] = A[kernel_1_threadIdx_x] + T.float32(1)

        @T.function
        def kernel_2(A: T.Tensor([256], "float32")):
            T.region(
                "tirx.launch_thread",
                [tvm.ir.StringImm("threadIdx.x"), 256],
                body_params=[kernel_2_threadIdx_x],
            )
            A[kernel_2_threadIdx_x] = A[kernel_2_threadIdx_x] + T.float32(1)

    after = tvm.tirx.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, expected)


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
                [tvm.ir.StringImm("threadIdx.x"), 256],
                body_params=body_params,
            ):
                A[threadIdx_x] = A[threadIdx_x] + 1.0
            with T.region(
                "tirx.launch_thread",
                [tvm.ir.StringImm("threadIdx.x"), 256],
                body_params=body_params,
            ):
                A[threadIdx_x] = A[threadIdx_x] + 2.0

        @T.function
        def kernel_2(A: T.Tensor([256], "float32")):
            with T.region(
                "tirx.launch_thread",
                [tvm.ir.StringImm("threadIdx.x"), 256],
                body_params=body_params,
            ):
                A[threadIdx_x] = A[threadIdx_x] + 1.0
            with T.region(
                "tirx.launch_thread",
                [tvm.ir.StringImm("threadIdx.x"), 256],
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

    after = tvm.tirx.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_shared_shape_var_in_buffer_params_and_alloc_buffer():
    """Shape var shared across buffer params and AllocTensor should not be renamed.

    When the same Var (e.g., `n`) appears in multiple buffer parameter
    annotations (A and B both have shape [n]), ConvertSSA should not treat
    the second occurrence as a redefinition.  All uses of `n` in the
    function body (including AllocTensor shapes) must remain the same
    Var object so that MakePackedAPI can bind it from the DLTensor shape.
    """
    n = tirx.Var("n", "int32")
    A = tirx.decl_tensor((n,), "float32", "A")
    B = tirx.decl_tensor((n,), "float32", "B")

    # AllocTensor with shape [n] in the body (flat, no body)
    C = tirx.decl_tensor((n,), "float32", "C")
    body = tirx.SeqStmt(
        [
            tvm.tirx.Bind(
                C,
                tvm.ir.Call(
                    "tirx.alloc_tensor",
                    [
                        tvm.ir.Tuple(C.shape),
                        tvm.ir.DataTypeImm(tvm.DataType(C.dtype)),
                        tvm.ir.StringImm(C.scope()),
                    ],
                    attrs=tvm.ir.DictAttrs({}),
                    ty=C.ty,
                ),
            ),
            tirx.Evaluate(1),
        ]
    )

    before = tirx.Function([A, B], body)

    mod = tvm.IRModule.from_expr(before)
    after = tvm.tirx.transform.ConvertSSA()(mod)
    # The function is already SSA — ConvertSSA should not change it.
    tvm.ir.assert_structural_equal(after["main"], before)


def test_reused_loop_var_in_decl_buffer_elem_offset():
    """Remap a buffer whose elem_offset depends on an SSA-renamed loop var."""
    loop_var = tirx.Var("loop_var", "int32")
    buffer = tirx.decl_tensor(
        (128,),
        "float32",
        "buffer",
        elem_offset=loop_var * 128,
        scope="shared.dyn",
    )
    buffer_data = tirx.Var("buffer_data", buffer.data.ty)
    loop = tirx.For(
        loop_var,
        0,
        128,
        tirx.ForKind.DEFAULT,
        tirx.SeqStmt(
            [
                tirx.Bind(
                    buffer,
                    tvm.ir.Call(
                        "tirx.decl_tensor",
                        [
                            buffer_data,
                            tvm.ir.Tuple(buffer.shape),
                            tvm.ir.DataTypeImm(tvm.DataType(buffer.dtype)),
                            tvm.ir.StringImm(buffer.scope()),
                        ],
                        ty=buffer.ty,
                    ),
                ),
                tirx.Evaluate(tirx.TensorLoad(buffer, [0])),
            ]
        ),
    )
    func = tirx.Function([buffer_data], tirx.SeqStmt([loop, loop, loop]))

    after = tvm.tirx.transform.ConvertSSA()(tvm.IRModule.from_expr(func))

    tvm.tirx.analysis.verify_well_formed(after["main"], assert_mode=True)


if __name__ == "__main__":
    tvm.testing.main()
