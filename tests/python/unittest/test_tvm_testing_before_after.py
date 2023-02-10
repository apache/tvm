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


class BaseBeforeAfter(tvm.testing.CompareBeforeAfter):
    def transform(self):
        return lambda x: x


class TestBeforeAfterPrimFunc(BaseBeforeAfter):
    @T.prim_func
    def before():
        T.evaluate(0)

    expected = before


class TestBeforeAfterMethod(BaseBeforeAfter):
    def before(self):
        @T.prim_func
        def func():
            T.evaluate(0)

        return func

    expected = before


class TestBeforeAfterFixture(BaseBeforeAfter):
    @tvm.testing.fixture
    def before(self):
        @T.prim_func
        def func():
            T.evaluate(0)

        return func

    expected = before


class TestBeforeAfterDelayedPrimFunc(BaseBeforeAfter):
    def before():
        T.evaluate(0)

    expected = before


class TestBeforeAfterParametrizedFixture(BaseBeforeAfter):
    n = tvm.testing.parameter(1, 8, 16)

    @tvm.testing.fixture
    def before(self, n):
        @T.prim_func
        def func(A: T.Buffer(n, "float32")):
            for i in T.serial(n):
                A[i] = 0.0

        return func

    expected = before


class TestBeforeAfterIRModule(BaseBeforeAfter):
    """The preferred form for writing TIR unit tests

    All evaluation is done at test-time, with the minimal amount of
    additional lines.  The `@tvm.testing.fixture`, `@ir_module`, and
    `@T.prim_func` annotations are handled by
    `tvm.testing.CompareBeforeAfter`.
    """

    class before:
        def func_A(A: T.Buffer(16, "float32")):
            for i in T.serial(16):
                A[i] = 0.0

        def func_B(A: T.Buffer(16, "int32")):
            for i in T.serial(16):
                A[i] = 42

    expected = before


class TestBeforeAfterIRModuleExplicitFixture(BaseBeforeAfter):
    """Like TestBeforeAfterIRModule, but with an explicit fixture

    If the IRModule depends on additional fixtures, this form can be
    used.
    """

    @tvm.testing.fixture
    def before(self):
        @ir_module
        class mod:
            @T.prim_func
            def func_A(A: T.Buffer(16, "float32")):
                for i in T.serial(16):
                    A[i] = 0.0

            @T.prim_func
            def func_B(A: T.Buffer(16, "int32")):
                for i in T.serial(16):
                    A[i] = 42

        return mod

    expected = before


if __name__ == "__main__":
    tvm.testing.main()
