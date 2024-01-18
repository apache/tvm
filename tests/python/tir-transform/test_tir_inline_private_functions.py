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

import tvm.testing
from tvm.script import ir as I, tir as T


class BaseTestCase:
    def test_well_formed(self):
        After = tvm.tir.transform.InlinePrivateFunctions()(self.Before)
        tvm.tir.analysis.verify_well_formed(After)

    def test_produces_expected(self):
        After = tvm.tir.transform.InlinePrivateFunctions()(self.Before)
        tvm.ir.assert_structural_equal(self.Expected, After)


class TestSimple(BaseTestCase):
    """Simple case directly acting on PrimFunc"""

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer([80, 16], "float32"), B: T.Buffer([64, 16], "float32")):
            for i in range(64):
                Before.subroutine(T.address_of(A[i, 0]), T.address_of(B[i, 0]))

        @T.prim_func(private=True)
        def subroutine(A_data: T.handle("float32"), B_data: T.handle("float32")):
            A = T.decl_buffer([16, 16], "float32", data=A_data)
            B = T.decl_buffer([16], "float32", data=B_data)
            for i in range(16):
                B[i] = 0.0
                for j in range(16):
                    B[i] = B[i] + A[i, j]

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer([80, 16], "float32"), B: T.Buffer([64, 16], "float32")):
            for i in range(64):
                A_view_data: T.handle("float32") = T.address_of(A[i, 0])
                Aview = T.decl_buffer([16, 16], "float32", data=A_view_data)
                B_view_data: T.handle("float32") = T.address_of(B[i, 0])
                Bview = T.decl_buffer([16], "float32", data=B_view_data)
                for j in range(16):
                    Bview[j] = 0.0
                    for k in range(16):
                        Bview[j] = Bview[j] + Aview[j, k]


class TestRetainCrossFunctionSubroutines(BaseTestCase):
    """Do not inline functions that cross device boundaries

    When lowering TIR, calls for which the callsite and callee have
    different targets are used at some stages, before being further
    lowered to explicit device kernel launches.  Since inlining the
    function would remove this cross-device information,
    InlinePrivateSubroutines should not inline these cases.
    """

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer([80, 16], "float32"), B: T.Buffer([64, 16], "float32")):
            T.func_attr({"target": T.target("llvm")})
            for i in range(64):
                Before.subroutine(T.address_of(A[i, 0]), T.address_of(B[i, 0]))

        @T.prim_func(private=True)
        def subroutine(A_data: T.handle("float32"), B_data: T.handle("float32")):
            T.func_attr({"target": T.target("cuda")})
            A = T.decl_buffer([16, 16], "float32", data=A_data)
            B = T.decl_buffer([16], "float32", data=B_data)
            for i in range(16):
                B[i] = 0.0
                for j in range(16):
                    B[i] = B[i] + A[i, j]

    Expected = Before


class TestRetainRecursiveSubroutines(BaseTestCase):
    """Do not inline recursive functions

    To avoid potentially infinite loops at compile-time, disable
    inlining of recursive functions.  If inlining of these functions
    would be useful, this restriction may be relaxed with improved
    analysis of the subroutine.
    """

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer(16, "float32")):
            Before.subroutine(T.address_of(A[0]), 16)

        @T.prim_func(private=True)
        def subroutine(A_data: T.handle("float32"), A_size: T.int32):
            A = T.decl_buffer(A_size, "float32", data=A_data)
            A[1] = A[0] + A[1]

            if A_size > 1:
                Before.subroutine(T.address_of(A[1]), A_size - 1)

    Expected = Before


class TestDeduplicateBlockName(BaseTestCase):
    """Block names must be de-duplicated after inlining"""

    @pytest.mark.xfail(reason="Inlining of schedulable TIR not yet supported")
    def test_produces_expected(self):
        super().test_produces_expected(self)

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer([2, 16], "float32"), B: T.Buffer([2, 16], "float32")):
            Before.subroutine(T.address_of(A[0, 0]), T.address_of(B[0, 0]))
            Before.subroutine(T.address_of(A[1, 0]), T.address_of(B[1, 0]))

        @T.prim_func(private=True)
        def subroutine(A_data: T.handle("float32"), B_data: T.handle("float32")):
            A = T.decl_buffer(16, "float32", data=A_data)
            B = T.decl_buffer(16, "float32", data=B_data)
            for i in range(16):
                with T.block("scalar_mul"):
                    B[i] = A[i] * 2.0

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer([80, 16], "float32"), B: T.Buffer([64, 16], "float32")):
            with T.LetStmt(T.address_of(A[0, 0]), var=T.handle("float32")) as A_data_1:
                A_1 = T.decl_buffer(16, "float32", data=A_data_1)
                B_data_1: T.handle("float32") = T.address_of(B[0, 0])
                B_1 = T.decl_buffer(16, "float32", data=B_data_1)
                for i in range(16):
                    with T.block("scalar_mul_1"):
                        B_1[i] = A_1[i] * 2.0

            with T.LetStmt(T.address_of(A[1, 0]), var=T.handle("float32")) as A_data_2:
                A_2 = T.decl_buffer(16, "float32", data=A_data_2)
                B_data_2: T.handle("float32") = T.address_of(B[1, 0])
                B_2 = T.decl_buffer(16, "float32", data=B_data_2)
                for i in range(16):
                    with T.block("scalar_mul_2"):
                        B_2[i] = A_2[i] * 2.0


class TestInlineCallOccurringInExpression(BaseTestCase):
    """Inline a Call node that is used in a function

    The current implementation only replaces `tir.Call` instances that
    occur in a `tir.Evaluate` context.  This is the primary use case,
    used in destination-passing style.

    This unit test is marked as xfail.  If/when the implementation
    supports inlining of function calls occurring as part of an
    expression, the annotation can be removed.
    """

    @pytest.mark.xfail(reason="Inlining of PrimFuncs outside of tir.Evaluate is not yet supported")
    def test_produces_expected(self):
        super().test_produces_expected(self)

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer(16, "float32")):
            for i in range(16):
                A[i] = Before.subroutine(i)

        @T.prim_func(private=True)
        def subroutine(i: T.int32) -> T.float32:
            cos = T.cos(T.cast(i, "float32"))
            sin = T.sin(T.cast(i, "float32"))
            retval = cos * cos + sin * sin
            T.ret(retval)

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer(16, "float32")):
            for i in range(16):
                cos = T.cos(T.cast(i, "float32"))
                sin = T.sin(T.cast(i, "float32"))
                retval = cos * cos + sin * sin
                A[i] = retval


class TestInlineFunctionWithBufferArguments(BaseTestCase):
    """Inline a function that accepts buffer arguments

    The current implementation does not support this usage.  This unit
    test is provided to display a possible user interaction, and is
    marked with `@pytest.mark.xfail`.  If/when the implementation
    supports inlining of function calls with buffer arguments, the
    annotation can be removed.
    """

    @pytest.mark.xfail(reason="Inlining of PrimFuncs with buffer arguments")
    def test_produces_expected(self):
        super().test_produces_expected(self)

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer(16, "float32")):
            Before.subroutine(
                T.tvm_stack_make_array(
                    A.data,
                    T.tvm_stack_make_shape(*A.shape, dtype="handle"),
                    0,
                    len(A.shape),
                    0.0,
                    A.elem_offset,
                    dtype="handle",
                )
            )

        @T.prim_func(private=True)
        def subroutine(A: T.Buffer(16, "float32")):
            for i in range(16):
                A[i] = A[i] * 2.0

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer(16, "float32")):
            for i in range(16):
                A[i] = A[i] * 2.0


if __name__ == "__main__":
    tvm.testing.main()
