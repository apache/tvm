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
from tvm.script import tir as T


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
        def func(A: T.Buffer[n, "float32"]):
            for i in T.serial(n):
                A[i] = 0.0

        return func

    expected = before


if __name__ == "__main__":
    tvm.testing.main()
