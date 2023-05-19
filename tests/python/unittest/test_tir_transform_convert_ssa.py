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
from tvm.script import tir as T, ir as I


class BaseBeforeAfter(tvm.testing.CompareBeforeAfter):
    transform = tvm.tir.transform.ConvertSSA()


class TestReuseInSequentialLetStmt(BaseBeforeAfter):
    """De-dup sequential variable bindings"""

    def before(self):
        @T.prim_func
        def func():
            i1 = T.handle("int32")
            with T.LetStmt(i1, var=i1):
                T.evaluate(0)
            with T.LetStmt(i1, var=i1):
                T.evaluate(0)
            buffer = T.Buffer((16,), "int32", data=i1)
            T.evaluate(buffer[0])

        return func

    def expected(self):
        @T.prim_func
        def func():
            i1 = T.handle("int32")
            with T.LetStmt(i1, var=i1):
                T.evaluate(0)
            with T.LetStmt(i1, T.handle("int32")) as i1_1:
                T.evaluate(0)
            buffer = T.Buffer((16,), "int32", data=i1)
            T.evaluate(buffer[0])

        return func


class TestReuseInNestedLetStmt(BaseBeforeAfter):
    """De-dup nested bindings

    Use of a variable with nested bindings is de-duplicated to refer
    to the inner-most binding that contains the use site.
    """

    def before(self):
        @T.prim_func
        def func():
            i = T.int32()
            with T.LetStmt(5, var=i):
                T.evaluate(i)
                with T.LetStmt(10, var=i):
                    T.evaluate(i)
                T.evaluate(i)

        return func

    def expected(self):
        @T.prim_func
        def func():
            i = T.int32()
            with T.LetStmt(5, var=i):
                T.evaluate(i)
                j = T.int32()
                with T.LetStmt(10, var=j):
                    T.evaluate(j)
                T.evaluate(i)

        return func


class TestReusedVarAcrossModule(BaseBeforeAfter):
    """De-duplicate Var bindings across entire module"""

    def before(self):
        var = T.int32()

        @I.ir_module
        class mod:
            @T.prim_func
            def func_a():
                with T.LetStmt(10, var=var):
                    T.evaluate(var)

            @T.prim_func
            def func_b():
                with T.LetStmt(42, var=var):
                    T.evaluate(var)

        return mod

    def expected(self):
        @I.ir_module
        class mod:
            @T.prim_func
            def func_a():
                var = T.int32(10)
                T.evaluate(var)

            @T.prim_func
            def func_b():
                var = T.int32(42)
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

        return tvm.IRModule({"func_a": func, "func_b": func})

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

        return tvm.IRModule({"func_a": func, "func_b": func})

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

        return tvm.IRModule({"func_a": func, "func_b": func})

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


if __name__ == "__main__":
    tvm.testing.main()
