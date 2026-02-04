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
from tvm import tir, ir
from tvm.script import tir as T, ir as I


def test_reuse_in_sequential_let_stmt():
    """De-dup sequential variable bindings"""

    # Manually construct the PrimFunc body, as SSA violations are
    # not valid TIR, and may not be expressible in future versions
    # of TVMSCript.
    var = tir.Var("var", "int32")
    sequential_bindings = tir.SeqStmt(
        [
            tir.LetStmt(var, 16, tir.Evaluate(var)),
            tir.LetStmt(var, 32, tir.Evaluate(var)),
        ]
    )
    before = tir.PrimFunc([], sequential_bindings)

    @T.prim_func(private=True)
    def expected():
        with T.LetStmt(T.int32(16)) as var1:
            T.evaluate(var1)
        with T.LetStmt(T.int32(32)) as var2:
            T.evaluate(var2)

    mod = tvm.IRModule.from_expr(before)
    mod = tvm.tir.transform.ConvertSSA()(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_reuse_in_nested_let_stmt():
    """De-dup nested bindings

    Use of a variable with nested bindings is de-duplicated to refer
    to the inner-most binding that contains the use site.
    """

    # Manually construct the PrimFunc body, as SSA violations are
    # not valid TIR, and may not be expressible in future versions
    # of TVMSCript.
    var = tir.Var("var", "int32")
    inner_let = tir.LetStmt(var, 16, tir.Evaluate(var))
    outer_let = tir.LetStmt(
        var,
        32,
        tir.SeqStmt(
            [
                tir.Evaluate(var),
                inner_let,
                tir.Evaluate(var),
            ]
        ),
    )
    before = tir.PrimFunc([], outer_let)

    @T.prim_func(private=True)
    def expected():
        with T.LetStmt(T.int32(32)) as outer:
            T.evaluate(outer)
            with T.LetStmt(T.int32(16)) as inner:
                T.evaluate(inner)
            T.evaluate(outer)

    mod = tvm.IRModule.from_expr(before)
    mod = tvm.tir.transform.ConvertSSA()(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_reused_var_across_module():
    """De-duplicate Var bindings across entire module"""

    @T.prim_func(private=True)
    def func():
        with T.LetStmt(10) as var:
            T.evaluate(var)

    before = tvm.IRModule(
        {
            "func_a": func.with_attr("global_symbol", "func_a"),
            "func_b": func.with_attr("global_symbol", "func_b"),
        }
    )

    @I.ir_module
    class expected:
        @T.prim_func
        def func_a():
            var = T.int32(10)
            T.evaluate(var)

        @T.prim_func
        def func_b():
            var = T.int32(10)
            T.evaluate(var)

    after = tvm.tir.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_reused_parameter():
    """De-duplicate Var usage in parameters

    In this test, the same `tir.Var` instance is used for the
    parameter `n` in both functions.
    """

    @T.prim_func(private=True)
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
        @T.prim_func
        def func_a(n: T.int32):
            T.evaluate(n)

        @T.prim_func
        def func_b(n: T.int32):
            T.evaluate(n)

    after = tvm.tir.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_reused_buffer_obj():
    """De-duplicate buffer usage across entire module"""

    @T.prim_func(private=True)
    def func(a: T.handle("float32")):
        A = T.Buffer(shape=1, dtype="float32", data=a)
        T.evaluate(A[0])

    before = tvm.IRModule(
        {
            "func_a": func.with_attr("global_symbol", "func_a"),
            "func_b": func.with_attr("global_symbol", "func_b"),
        }
    )

    @I.ir_module
    class expected:
        @T.prim_func
        def func_a(a: T.handle("float32")):
            A = T.Buffer(shape=1, dtype="float32", data=a)
            T.evaluate(A[0])

        @T.prim_func
        def func_b(a: T.handle("float32")):
            A = T.Buffer(shape=1, dtype="float32", data=a)
            T.evaluate(A[0])

    after = tvm.tir.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_reused_buffer_parameter():
    """De-duplicate buffer_map across entire module"""

    @T.prim_func(private=True)
    def func(A: T.Buffer(1, "float32")):
        T.evaluate(A[0])

    before = tvm.IRModule(
        {
            "func_a": func.with_attr("global_symbol", "func_a"),
            "func_b": func.with_attr("global_symbol", "func_b"),
        }
    )

    @I.ir_module
    class expected:
        @T.prim_func
        def func_a(A: T.Buffer(1, "float32")):
            T.evaluate(A[0])

        @T.prim_func
        def func_b(A: T.Buffer(1, "float32")):
            T.evaluate(A[0])

    after = tvm.tir.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_no_change_if_already_ssa():
    """A module that is already SSA should be unchanged"""

    @I.ir_module
    class before:
        @T.prim_func
        def func(A: T.Buffer(1, "float32")):
            T.evaluate(A[0])

    after = tvm.tir.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(before, after)
    assert before.same_as(after)


def test_keep_duplicate_thread_idx_in_same_function():
    """Environment threads are treated as being at function scope

    The `"thread_extent"` attribute has some unique semantics.  It
    serves as the definition of the `tir::Var` representing the
    environment thread (e.g. `threadIdx.x` in CUDA).  However,
    multiple `"thread_extent"` attributes may co-exist in the same
    PrimFunc.  For the purpose of variable scope, use of the
    `tir::Var` is only allowed within the body of the `AttrStmt`.
    However, for the purpose of well-formed-ness, all
    `"thread_extent"` attributes must use the same IterVar instance
    (e.g. `WarpIndexFinder` in `lower_warp_memory.cc` may throw an
    error if multiple IterVar instances occur).

    If there are multiple `AttrStmt` with key `"thread_extent"` in a
    single function (represented in TVMScript as `T.launch_thread`),
    these should be treated as a definition of a single variable at
    function scope, and should not be de-duplicated.
    """

    @I.ir_module
    class before:
        @T.prim_func
        def main(A: T.Buffer([256], "float32")):
            threadIdx_x = T.env_thread("threadIdx.x")
            with T.launch_thread(threadIdx_x, 256):
                A[threadIdx_x] = A[threadIdx_x] + 1.0

            with T.launch_thread(threadIdx_x, 256):
                A[threadIdx_x] = A[threadIdx_x] + 2.0

    after = tvm.tir.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, before)


def test_de_duplicate_thread_idx_across_multiple_functions():
    """Environment threads are treated as being at function scope

    See `test_keep_duplicate_thread_idx_in_same_function` for background
    information.

    If there are multiple functions in an IRModule, the `AttrStmt`
    with key `"thread_extent"` in a single function (represented in
    TVMScript as `T.launch_thread`), these should be treated as a
    definition of a single variable at function scope, and should not
    be de-duplicated.

    For this test case, the `AttrStmt` for `"thread_extent"` are
    written explicitly, without using the usual `T.env_thread` and
    `T.launch_thread`, as they cannot represent the duplciate
    Var/IterVar usage across the two PrimFuncs.
    """

    threadIdx_x = tvm.tir.Var("threadIdx_x", "int32")

    # threadIdx_x is defined outside
    @I.ir_module(check_well_formed=False)
    class before:
        @T.prim_func
        def kernel_1(A: T.Buffer([256], "float32")):
            T.attr(
                T.iter_var(threadIdx_x, T.Range(0, 256), "ThreadIndex", "threadIdx.x"),
                "thread_extent",
                256,
            )
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

        @T.prim_func
        def kernel_2(A: T.Buffer([256], "float32")):
            T.attr(
                T.iter_var(threadIdx_x, T.Range(0, 256), "ThreadIndex", "threadIdx.x"),
                "thread_extent",
                256,
            )
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

    @I.ir_module
    class expected:
        @T.prim_func
        def kernel_1(A: T.Buffer([256], "float32")):
            threadIdx_x = T.int32()
            T.attr(
                T.iter_var(threadIdx_x, T.Range(0, 256), "ThreadIndex", "threadIdx.x"),
                "thread_extent",
                256,
            )
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

        @T.prim_func
        def kernel_2(A: T.Buffer([256], "float32")):
            threadIdx_x = T.int32()
            T.attr(
                T.iter_var(threadIdx_x, T.Range(0, 256), "ThreadIndex", "threadIdx.x"),
                "thread_extent",
                256,
            )
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

    after = tvm.tir.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_de_duplicate_thread_idx_iter_var_across_multiple_functions():
    """Environment threads are treated as being at function scope

    Like `test_de_duplicate_thread_idx_across_multiple_functions`, except the
    `IterVar` for the environment thread is duplicated across multiple
    PrimFuncs, not just the `tir.Var` inside the `IterVar`.
    """

    threadIdx_x = tvm.tir.Var("threadIdx_x", "int32")
    iter_var = tvm.tir.IterVar(
        tvm.ir.Range(0, 256), threadIdx_x, tvm.tir.IterVar.ThreadIndex, "threadIdx.x"
    )

    # complaints of multiple definitions for threadIdx_x
    @I.ir_module(check_well_formed=False)
    class before:
        @T.prim_func
        def kernel_1(A: T.Buffer([256], "float32")):
            T.attr(iter_var, "thread_extent", 256)
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

        @T.prim_func
        def kernel_2(A: T.Buffer([256], "float32")):
            T.attr(iter_var, "thread_extent", 256)
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

    @I.ir_module(check_well_formed=False)
    class expected:
        @T.prim_func
        def kernel_1(A: T.Buffer([256], "float32")):
            threadIdx_x = T.int32()
            T.attr(
                T.iter_var(threadIdx_x, T.Range(0, 256), "ThreadIndex", "threadIdx.x"),
                "thread_extent",
                256,
            )
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

        @T.prim_func
        def kernel_2(A: T.Buffer([256], "float32")):
            threadIdx_x = T.int32()
            T.attr(
                T.iter_var(threadIdx_x, T.Range(0, 256), "ThreadIndex", "threadIdx.x"),
                "thread_extent",
                256,
            )
            A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

    after = tvm.tir.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_thread_idx_reused_within_and_across_functions():
    """Environment threads are treated as being at function scope

    A combination of
    test_de_duplicate_thread_idx_iter_var_across_multiple_functions and
    test_keep_duplicate_thread_idx_in_same_function.  The re-use within a
    function should be maintained, while re-use across functions is
    de-duplicated.
    """

    threadIdx_x = tvm.tir.Var("threadIdx_x", "int32")
    iter_var = tvm.tir.IterVar(
        tvm.ir.Range(0, 256), threadIdx_x, tvm.tir.IterVar.ThreadIndex, "threadIdx.x"
    )

    # complaints of multiple definitions of threadIdx_x
    @I.ir_module(check_well_formed=False)
    class before:
        @T.prim_func
        def kernel_1(A: T.Buffer([256], "float32")):
            with T.attr(iter_var, "thread_extent", 256):
                A[threadIdx_x] = A[threadIdx_x] + 1.0
            with T.attr(iter_var, "thread_extent", 256):
                A[threadIdx_x] = A[threadIdx_x] + 2.0

        @T.prim_func
        def kernel_2(A: T.Buffer([256], "float32")):
            with T.attr(iter_var, "thread_extent", 256):
                A[threadIdx_x] = A[threadIdx_x] + 1.0
            with T.attr(iter_var, "thread_extent", 256):
                A[threadIdx_x] = A[threadIdx_x] + 2.0

    @I.ir_module
    class expected:
        @T.prim_func
        def kernel_1(A: T.Buffer([256], "float32")):
            threadIdx_x = T.env_thread("threadIdx.x")
            with T.launch_thread(threadIdx_x, 256):
                A[threadIdx_x] = A[threadIdx_x] + 1.0
            with T.launch_thread(threadIdx_x, 256):
                A[threadIdx_x] = A[threadIdx_x] + 2.0

        @T.prim_func
        def kernel_2(A: T.Buffer([256], "float32")):
            threadIdx_x = T.env_thread("threadIdx.x")
            with T.launch_thread(threadIdx_x, 256):
                A[threadIdx_x] = A[threadIdx_x] + 1.0
            with T.launch_thread(threadIdx_x, 256):
                A[threadIdx_x] = A[threadIdx_x] + 2.0

    after = tvm.tir.transform.ConvertSSA()(before)
    tvm.ir.assert_structural_equal(after, expected)


def test_track_forward_declarations_in_attr_stmt():
    """T.attr statements may refer to a about-to-be-defined tir.Var"""

    # Generate the PrimFunc, which is already SSA
    #
    # This is constructed directly, rather than using TVMScript or
    # the `tvm.tir.ir_builder`.  This test case requires a
    # `tir.AttrStmt` that references a variable, followed by the
    # `tir.For` defining that variable.  This is not expressible in
    # either TVMScript or `tvm.tir.ir_builder`, as they only provide
    # the loop iterator within the body of the loop.
    i0_outer_outer = tir.Var("i0_outer_outer", "int32")
    i0_outer_inner = tir.Var("i0_outer_inner", "int32")
    i0_inner = tir.Var("i0_inner", "int32")

    A = tir.decl_buffer(1024, "float32", "A")
    B = tir.decl_buffer(1024, "float32", "B")

    index = i0_outer_outer * 52 + i0_outer_inner * 4 + i0_inner

    stmt = tir.BufferStore(B, tir.BufferLoad(A, [index]), [index])
    stmt = tir.IfThenElse(i0_outer_outer * 13 + i0_outer_inner < 256, stmt, None)
    stmt = tir.For(i0_inner, 0, 4, tir.ForKind.VECTORIZED, stmt)
    stmt = tir.For(i0_outer_inner, 0, 13, tir.ForKind.PARALLEL, stmt)
    stmt = tir.AttrStmt(
        T.iter_var(i0_outer_inner, None, "DataPar", ""),
        "pragma_parallal_barrier_when_finish",
        1,
        stmt,
    )
    stmt = tir.AttrStmt(
        T.iter_var(i0_outer_inner, None, "DataPar", ""),
        "pragma_parallal_stride_pattern",
        1,
        stmt,
    )
    stmt = tir.For(i0_outer_outer, 0, 20, tir.ForKind.SERIAL, stmt)
    stmt = tir.AttrStmt(
        T.iter_var(i0_outer_outer, None, "DataPar", ""),
        "pragma_parallal_launch_point",
        1,
        stmt,
    )

    A_handle = tir.Var("A_handle", "handle")
    B_handle = tir.Var("B_handle", "handle")

    before = tir.PrimFunc(
        [A_handle, B_handle],
        stmt,
        buffer_map={A_handle: A, B_handle: B},
    )

    mod = tvm.IRModule.from_expr(before)
    after = tvm.tir.transform.ConvertSSA()(mod)
    tvm.ir.assert_structural_equal(after["main"], before)


if __name__ == "__main__":
    tvm.testing.main()
