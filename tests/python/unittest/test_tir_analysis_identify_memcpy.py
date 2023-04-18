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

import re

import pytest

import tvm
import tvm.testing
from tvm.tir import BufferRegion, StringImm

from tvm.script import tir as T

identify_memcpy = tvm.tir.analysis._ffi_api._identify_memcpy


class BaseTest:
    """Utility class for defining unit tests for memcpy"""

    def __init_subclass__(cls):
        cls.func = tvm.testing.CompareBeforeAfter._normalize_before(cls.func)
        cls.expected = pytest.fixture(cls.expected)

    def test_identify_memcpy(self, func, expected):
        results = identify_memcpy(func.body)

        if isinstance(expected, str) or (
            isinstance(expected, tuple) and isinstance(expected[0], BufferRegion)
        ):
            expected = [expected]

        assert len(expected) == len(results)
        for expected, result in zip(expected, results):
            if isinstance(expected, str):
                assert isinstance(result, StringImm)
                assert re.search(expected, result.value)
            else:
                tvm.ir.assert_structural_equal(result, expected)


class Test1D(BaseTest):
    """Simplest test case"""

    def func(A: T.Buffer[1024, "float32"], B: T.Buffer[1024, "float32"]):
        for i in T.serial(1024):
            B[i] = A[i]

    def expected(self, func):
        A, B = func.buffer_map.values()
        return A[0:1024], B[0:1024]


class Test1DCompute(BaseTest):
    """Like Test1D, but a computation prevents this being a memcpy"""

    def func(A: T.Buffer[1024, "float32"], B: T.Buffer[1024, "float32"]):
        for i in T.serial(1024):
            B[i] = A[i] + 1.0

    def expected(self, func):
        return "Expected BufferStore's value to be BufferLoad"


class Test1DConditional(BaseTest):
    """Like Test1D, but a conditionals prevents this being a memcpy"""

    def func(A: T.Buffer[1024, "float32"], B: T.Buffer[1024, "float32"]):
        for i in T.serial(1024):
            if i < 1024:
                B[i] = A[i]

    def expected(self, func):
        A, B = func.buffer_map.values()
        return "Expected innermost loop to have BufferStore body"


class Test1DStridedInput(BaseTest):
    """Like Test1D, but strided input prevents this being a memcpy"""

    def func(A: T.Buffer[2048, "float32"], B: T.Buffer[1024, "float32"]):
        for i in T.serial(1024):
            B[i] = A[i * 2]

    def expected(self, func):
        return "Mismatch between loop iterations (.*) and number of src indices"


class Test1DStridedOutput(BaseTest):
    """Like Test1D, but strided output prevents this being a memcpy"""

    def func(A: T.Buffer[1024, "float32"], B: T.Buffer[2048, "float32"]):
        for i in T.serial(1024):
            B[i * 2] = A[i]

    def expected(self, func):
        return "Mismatch between loop iterations (.*) and number of dst indices"


class Test1DInput2DOutputFusedLoop(BaseTest):
    """Like Test1D, but the output is written as a 2-d buffer"""

    def func(A: T.Buffer[1024, "float32"], B: T.Buffer[(32, 32), "float32"]):
        for i in T.serial(1024):
            B[i // 32, i % 32] = A[i]

    def expected(self, func):
        A, B = func.buffer_map.values()
        return A[0:1024], B[0:32, 0:32]


class Test2DInput1DOutputFusedLoop(BaseTest):
    """Like Test1D, but the input is written as a 2-d buffer"""

    def func(A: T.Buffer[(32, 32), "float32"], B: T.Buffer[1024, "float32"]):
        for i in T.serial(1024):
            B[i] = A[i // 32, i % 32]

    def expected(self, func):
        A, B = func.buffer_map.values()
        return A[0:32, 0:32], B[0:1024]


class Test1DInput1DOutputNestedLoop(BaseTest):
    """Like Test1D, but the iterator is written as a nested loop

    In test cases with more than one loop, each loop is checked to see
    if could be written as a memcpy.  The C++ utility function
    operates on individual loops, but for unit testing in Python, it
    is more convenient to return the results for all loops.
    """

    def func(A: T.Buffer[1024, "float32"], B: T.Buffer[1024, "float32"]):
        for i, j in T.grid(32, 32):
            B[i * 32 + j] = A[i * 32 + j]

    def expected(self, func):
        A, B = func.buffer_map.values()
        i = func.body.loop_var
        return [
            (A[0:1024], B[0:1024]),
            (A[i * 32 : i * 32 + 32], B[i * 32 : i * 32 + 32]),
        ]


class Test1DInput1DOutputNestedLoopEquivalentExpressions(BaseTest):
    """Like Test1DInput1DOutputNestedLoop, but with equivalent indices

    If the expressions are not identical, the loops may still be
    recognizable as a memcpy, so long as the expressions are
    equivalent.
    """

    def func(A: T.Buffer[1024, "float32"], B: T.Buffer[1024, "float32"]):
        for i, j in T.grid(32, 32):
            B[i * 32 + j] = A[j + i * 32]

    def expected(self, func):
        A, B = func.buffer_map.values()
        i = func.body.loop_var
        return [
            (A[0:1024], B[0:1024]),
            (A[i * 32 : i * 32 + 32], B[i * 32 : i * 32 + 32]),
        ]


class Test1DInput2DOutputNestedLoop(BaseTest):
    """Like Test1DInput1DOutputNestedLoop, but with a 2-d output buffer"""

    def func(A: T.Buffer[1024, "float32"], B: T.Buffer[(32, 32), "float32"]):
        for i, j in T.grid(32, 32):
            B[i, j] = A[i * 32 + j]

    def expected(self, func):
        A, B = func.buffer_map.values()
        i = func.body.loop_var
        return [
            (A[0:1024], B[0:32, 0:32]),
            (A[i * 32 : i * 32 + 32], B[i, 0:32]),
        ]


class Test2DInput1DOutputNestedLoop(BaseTest):
    """Like Test1DInput1DOutputNestedLoop, but with a 2-d input buffer"""

    def func(A: T.Buffer[(32, 32), "float32"], B: T.Buffer[1024, "float32"]):
        for i, j in T.grid(32, 32):
            B[i * 32 + j] = A[i, j]

    def expected(self, func):
        A, B = func.buffer_map.values()
        i = func.body.loop_var
        return [
            (A[0:32, 0:32], B[0:1024]),
            (A[i, 0:32], B[i * 32 : i * 32 + 32]),
        ]


class Test2DInput2DOutputNestedLoop(BaseTest):
    """Like Test1DInput1DOutputNestedLoop, but with 2-d input/output buffers"""

    def func(A: T.Buffer[(32, 32), "float32"], B: T.Buffer[(32, 32), "float32"]):
        for i, j in T.grid(32, 32):
            B[i, j] = A[i, j]

    def expected(self, func):
        A, B = func.buffer_map.values()
        i = func.body.loop_var
        return [
            (A[0:32, 0:32], B[0:32, 0:32]),
            (A[i, 0:32], B[i, 0:32]),
        ]


class Test2DInput2DOutputTransposeOutput(BaseTest):
    """Test2DInput2DOutputNestedLoop, but with a transposed output

    This is not recognized as a memcpy, because it results in a transpose.
    """

    def func(A: T.Buffer[(32, 32), "float32"], B: T.Buffer[(32, 32), "float32"]):
        for i, j in T.grid(32, 32):
            B[j, i] = A[i, j]

    def expected(self, func):
        return [
            "different source",
            "Mismatch .* number of dst indices touched",
        ]


class Test2DInput2DOutputTransposeInput(BaseTest):
    """Test2DInput2DOutputNestedLoop, but with a transposed input

    This is not recognized as a memcpy, because it results in a transpose.
    """

    def func(A: T.Buffer[(32, 32), "float32"], B: T.Buffer[(32, 32), "float32"]):
        for i, j in T.grid(32, 32):
            B[i, j] = A[j, i]

    def expected(self, func):
        return [
            "different source",
            "Mismatch .* number of src indices touched",
        ]


class Test2DInput2DOutputTransposeBoth(BaseTest):
    """Test2DInput2DOutputNestedLoop, but with a transposed input

    The inner loop is not recognized as a memcpy, because it has
    strided access of both the input and output buffers.  However, the
    outer loop is still recognized as a memcpy, because the full
    region has been copied over, even though it occurs out of order.
    """

    def func(A: T.Buffer[(32, 32), "float32"], B: T.Buffer[(32, 32), "float32"]):
        for i, j in T.grid(32, 32):
            B[j, i] = A[j, i]

    def expected(self, func):
        A, B = func.buffer_map.values()
        return [
            (A[0:32, 0:32], B[0:32, 0:32]),
            "Mismatch .* number of src indices touched",
        ]


class TestCacheRead(BaseTest):
    """Like Test2DInput2DOutputNestedLoop, but with a 1-d

    The inner loop is a memcpy of a single row at a time.  This
    pattern would appear when B is a read cache of A.
    """

    def func(A: T.Buffer[(32, 32), "float32"], B: T.Buffer[32, "float32"]):
        for i, j in T.grid(32, 32):
            B[j] = A[i, j]

    def expected(self, func):
        A, B = func.buffer_map.values()
        i = func.body.loop_var
        return [
            "does not form a bijective transform",
            (A[i, 0:32], B[0:32]),
        ]


class TestCacheWrite(BaseTest):
    """Like Test2DInput2DOutputNestedLoop, but with a 1-d

    The inner loop is a memcpy of a single row at a time.  This
    pattern would appear when A is a write cache of B.
    """

    def func(A: T.Buffer[32, "float32"], B: T.Buffer[(32, 32), "float32"]):
        for i, j in T.grid(32, 32):
            B[i, j] = A[j]

    def expected(self, func):
        A, B = func.buffer_map.values()
        i = func.body.loop_var
        return [
            "does not form a bijective transform",
            (A[0:32], B[i, 0:32]),
        ]


if __name__ == "__main__":
    tvm.testing.main()
