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
from tvm import tir
from tvm.script import tir as T, ir as I


class BaseBeforeAfter(tvm.testing.CompareBeforeAfter):
    transform = tvm.tir.transform.ConvertSSA()


class TestReuseInSequentialLetStmt(BaseBeforeAfter):
    """De-dup sequential variable bindings"""

    def before(self):
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
        func = tir.PrimFunc([], sequential_bindings)

        return func

    def expected(self):
        @T.prim_func
        def func():
            with T.LetStmt(T.int32(16)) as var1:
                T.evaluate(var1)
            with T.LetStmt(T.int32(32)) as var2:
                T.evaluate(var2)

        return func


class TestReuseInNestedLetStmt(BaseBeforeAfter):
    """De-dup nested bindings

    Use of a variable with nested bindings is de-duplicated to refer
    to the inner-most binding that contains the use site.
    """

    def before(self):
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
        func = tir.PrimFunc([], outer_let)

        return func

    def expected(self):
        @T.prim_func
        def func():
            with T.LetStmt(T.int32(32)) as outer:
                T.evaluate(outer)
                with T.LetStmt(T.int32(16)) as inner:
                    T.evaluate(inner)
                T.evaluate(outer)

        return func


class TestReusedVarAcrossModule(BaseBeforeAfter):
    """De-duplicate Var bindings across entire module"""

    def before(self):
        @T.prim_func
        def func():
            with T.LetStmt(10) as var:
                T.evaluate(var)

        return tvm.IRModule(
            {
                "func_a": func.with_attr("global_symbol", "func_a"),
                "func_b": func.with_attr("global_symbol", "func_b"),
            }
        )

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def func_a():
                var = T.int32(10)
                T.evaluate(var)

            @T.prim_func
            def func_b():
                var = T.int32(10)
                T.evaluate(var)

        return mod


class TestReusedParameter(BaseBeforeAfter):
    """De-duplicate Var usage in parameters

    In this test, the same `tir.Var` instance is used for the
    parameter `n` in both functions.
    """

    def before(self):
        @T.prim_func
        def func(n: T.int32):
            T.evaluate(n)

        return tvm.IRModule(
            {
                "func_a": func.with_attr("global_symbol", "func_a"),
                "func_b": func.with_attr("global_symbol", "func_b"),
            }
        )

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def func_a(n: T.int32):
                T.evaluate(n)

            @T.prim_func
            def func_b(n: T.int32):
                T.evaluate(n)

        return mod


class TestReusedBufferObj(BaseBeforeAfter):
    """De-duplicate buffer usage across entire module"""

    def before(self):
        @T.prim_func
        def func(a: T.handle("float32")):
            A = T.Buffer(shape=1, dtype="float32", data=a)
            T.evaluate(A[0])

        return tvm.IRModule(
            {
                "func_a": func.with_attr("global_symbol", "func_a"),
                "func_b": func.with_attr("global_symbol", "func_b"),
            }
        )

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def func_a(a: T.handle("float32")):
                A = T.Buffer(shape=1, dtype="float32", data=a)
                T.evaluate(A[0])

            @T.prim_func
            def func_b(a: T.handle("float32")):
                A = T.Buffer(shape=1, dtype="float32", data=a)
                T.evaluate(A[0])

        return mod


class TestReusedBufferParameter(BaseBeforeAfter):
    """De-duplicate buffer_map across entire module"""

    def before(self):
        @T.prim_func
        def func(A: T.Buffer(1, "float32")):
            T.evaluate(A[0])

        return tvm.IRModule(
            {
                "func_a": func.with_attr("global_symbol", "func_a"),
                "func_b": func.with_attr("global_symbol", "func_b"),
            }
        )

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def func_a(A: T.Buffer(1, "float32")):
                T.evaluate(A[0])

            @T.prim_func
            def func_b(A: T.Buffer(1, "float32")):
                T.evaluate(A[0])

        return mod


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


class TestDedupAutoBroadcastBuffer(BaseBeforeAfter):
    """De-dup auto-broadcast buffers

    Auto-broadcast buffers can define additional variables during the
    `Buffer::Buffer` constructor for the strides.  This is intended to
    be used for match buffers, where these variables are defined based
    on the argument being passed in.

    These additional variables can cause errors when copying a buffer
    with the `Buffer::Buffer` constructor.  If a buffer has non-empty
    shape, empty strides, and kAutoBroadcast type, then the resulting
    buffer will have additional strides defined.  Such a buffer can
    result from lowering of a scalar buffer, which will be flattened
    to a shape of [1].

    Previous implementations of ConvertSSA incorrectly handled this
    case, resulting in undefined stride variables.
    """

    def _make_func(self):
        @T.prim_func
        def func(a: T.handle):
            A = T.match_buffer(a, shape=(), dtype="float32", buffer_type="auto")
            A[()] = 1.0

        return tvm.lower(func)["main"]

    def before(self):
        func = self._make_func()
        return tvm.IRModule({"func_a": func, "func_b": func})

    def expected(self):
        return tvm.IRModule({"func_a": self._make_func(), "func_b": self._make_func()})


class TestKeepDuplicateThreadIdxInSameFunction(BaseBeforeAfter):
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

    def before(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def main(A: T.Buffer([256], "float32")):
                threadIdx_x = T.env_thread("threadIdx.x")
                with T.launch_thread(threadIdx_x, 256):
                    A[threadIdx_x] = A[threadIdx_x] + 1.0

                with T.launch_thread(threadIdx_x, 256):
                    A[threadIdx_x] = A[threadIdx_x] + 2.0

        return mod

    expected = before


class TestDeDuplicateThreadIdxAcrossMultipleFunctions(BaseBeforeAfter):
    """Environment threads are treated as being at function scope

    See `TestKeepDuplicateThreadIdxInSameFunction` for background
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

    def before(self):
        threadIdx_x = tvm.tir.Var("threadIdx_x", "int32")

        @I.ir_module
        class mod:
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

        return mod

    def expected(self):
        @I.ir_module
        class mod:
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

        return mod


class TestDeDuplicateThreadIdxIterVarAcrossMultipleFunctions(BaseBeforeAfter):
    """Environment threads are treated as being at function scope

    Like `TestDeDuplicateThreadIdxAcrossMultipleFunctions`, except the
    `IterVar` for the environment thread is duplicated across multiple
    PrimFuncs, not just the `tir.Var` inside the `IterVar`.
    """

    def before(self):
        threadIdx_x = tvm.tir.Var("threadIdx_x", "int32")
        iter_var = tvm.tir.IterVar(
            tvm.ir.Range(0, 256), threadIdx_x, tvm.tir.IterVar.ThreadIndex, "threadIdx.x"
        )

        @I.ir_module
        class mod:
            @T.prim_func
            def kernel_1(A: T.Buffer([256], "float32")):
                T.attr(iter_var, "thread_extent", 256)
                A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

            @T.prim_func
            def kernel_2(A: T.Buffer([256], "float32")):
                T.attr(iter_var, "thread_extent", 256)
                A[threadIdx_x] = A[threadIdx_x] + T.float32(1)

        return mod

    def expected(self):
        @I.ir_module
        class mod:
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

        return mod


class TestThreadIdxReusedWithinAndAcrossFunctions(BaseBeforeAfter):
    """Environment threads are treated as being at function scope

    A combination of
    TestDeDuplicateThreadIdxIterVarAcrossMultipleFunctions and
    TestKeepDuplicateThreadIdxInSameFunction.  The re-use within a
    function should be maintained, while re-use across functions is
    de-duplicated.
    """

    def before(self):
        threadIdx_x = tvm.tir.Var("threadIdx_x", "int32")
        iter_var = tvm.tir.IterVar(
            tvm.ir.Range(0, 256), threadIdx_x, tvm.tir.IterVar.ThreadIndex, "threadIdx.x"
        )

        @I.ir_module
        class mod:
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

        return mod

    def expected(self):
        @I.ir_module
        class mod:
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

        return mod


if __name__ == "__main__":
    tvm.testing.main()
