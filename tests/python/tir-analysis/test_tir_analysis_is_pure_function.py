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
from tvm.script import tir as T

from tvm.tir.analysis import is_pure_function, assert_pure_function


class CheckPureFunction:
    def test_check_purity(self):
        assert is_pure_function(self.func)

    def test_assert_purity(self):
        assert_pure_function(self.func)


class CheckImpureFunction:
    def test_check_purity(self):
        assert not is_pure_function(self.func)

    def test_assert_purity(self):
        with pytest.raises(AssertionError):
            assert_pure_function(self.func)


class TestNoOp(CheckPureFunction):
    @T.prim_func
    def func():
        pass


class TestReturnValue(CheckPureFunction):
    @T.prim_func
    def func() -> T.int32:
        T.ret(42)


class TestComputeValueAndReturn(CheckPureFunction):
    @T.prim_func
    def func(N: T.int32, M: T.int32) -> T.int32:
        T.ret(N * M)


class TestReadBufferArgument(CheckPureFunction):
    @T.prim_func
    def func(A: T.Buffer(16, "float32")) -> T.float32:
        T.ret(A[0])


class TestWriteToBufferArgument(CheckImpureFunction):
    @T.prim_func
    def func(A: T.Buffer(16, "float32"), B: T.Buffer(16, "float32")):
        for i in range(16):
            B[i] = A[i]


class TestWriteToInternalAllocation(CheckPureFunction):
    @T.prim_func
    def func(A: T.Buffer([16, 16], "float32")) -> T.float32:
        Sum = T.decl_buffer([], "float32")
        Sum[()] = 0.0
        for i, j in T.grid(16, 16):
            Sum[()] = Sum[()] + A[i, j]

        T.ret(Sum[()])


class TestCallPureBuiltin(CheckPureFunction):
    @T.prim_func
    def func(x: T.float32) -> T.float32:
        T.ret(T.cos(x))


class TestCallPureExtern(CheckPureFunction):
    @T.prim_func
    def func():
        T.call_pure_extern("some_pure_extern_func_name", dtype="void")


class TestCallImpureExtern(CheckImpureFunction):
    @T.prim_func
    def func():
        T.call_extern("some_impure_extern_func_name", dtype="void")


if __name__ == "__main__":
    tvm.testing.main()
