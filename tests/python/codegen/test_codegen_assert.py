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
"""AssertStmt codegen tests: verify kind and message_parts produce correct exceptions."""

import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T

codegen_target = tvm.testing.parameter("llvm", "c")


def test_assert_runtime_error(codegen_target):
    """AssertStmt with RuntimeError kind produces RuntimeError."""

    @T.prim_func
    def func(x: T.int32):
        assert x > 0, ("RuntimeError", ["Expected non-null input"])

    lib = tvm.compile(func, target=codegen_target)
    with pytest.raises(RuntimeError, match="Expected non-null input"):
        lib(0)


def test_assert_value_error(codegen_target):
    """AssertStmt with ValueError kind produces ValueError."""

    @T.prim_func
    def func(x: T.int32):
        assert x > 0, ("ValueError", ["Shape mismatch: expected 4 got 8"])

    lib = tvm.compile(func, target=codegen_target)
    with pytest.raises(ValueError, match="Shape mismatch"):
        lib(0)


def test_assert_type_error(codegen_target):
    """AssertStmt with TypeError kind produces TypeError."""

    @T.prim_func
    def func(x: T.int32):
        assert x > 0, ("TypeError", ["Expected Tensor but got int"])

    lib = tvm.compile(func, target=codegen_target)
    with pytest.raises(TypeError, match="Expected Tensor but got int"):
        lib(0)


def test_assert_multi_part_message(codegen_target):
    """Multi-part messages are correctly concatenated at runtime."""

    @T.prim_func
    def func(x: T.int32):
        assert x > 0, ("ValueError", ["Expected shape ", "4", " but got ", "8"])

    lib = tvm.compile(func, target=codegen_target)
    with pytest.raises(ValueError, match="Expected shape 4 but got 8"):
        lib(0)


def test_assert_passing_condition(codegen_target):
    """Passing assertion does not raise."""

    @T.prim_func
    def func(x: T.int32):
        assert x > 0, ("RuntimeError", ["This should not be raised"])

    lib = tvm.compile(func, target=codegen_target)
    lib(1)  # should pass without error


def test_assert_many_parts(codegen_target):
    """Assertion with 8 parts concatenated correctly."""

    @T.prim_func
    def func(x: T.int32):
        assert x > 0, ("RuntimeError", ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"])

    lib = tvm.compile(func, target=codegen_target)
    with pytest.raises(RuntimeError, match="p0p1p2p3p4p5p6p7"):
        lib(0)


def test_tvmscript_assert_preserves_kind(codegen_target):
    """Regression: TVMScript structured assert preserves kind at runtime."""

    @T.prim_func
    def func(x: T.int32):
        assert x > 0, ("ValueError", ["x must be positive"])

    lib = tvm.compile(func, target=codegen_target)
    with pytest.raises(ValueError, match="x must be positive"):
        lib(0)


def test_tvmscript_assert_preserves_parts(codegen_target):
    """Regression: TVMScript structured assert with separate parts."""

    @T.prim_func
    def func(x: T.int32):
        assert x > 0, ("ValueError", ["x must be ", "positive"])

    lib = tvm.compile(func, target=codegen_target)
    with pytest.raises(ValueError, match="x must be positive"):
        lib(0)


if __name__ == "__main__":
    tvm.testing.main()
