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
"""Test error handling codegen with AssertStmt kind and message_parts."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import tir
from tvm.script import ir as I
from tvm.script import tir as T


@tvm.testing.requires_llvm
def test_assert_runtime_error_llvm():
    """Test that AssertStmt with RuntimeError kind produces correct error via LLVM."""
    target = tvm.target.Target("llvm")

    # Build a function with an always-failing assertion
    x = tir.Var("x", "int32")
    assert_stmt = tir.AssertStmt(
        tir.StringImm("RuntimeError"),
        tir.const(False, "bool"),
        [tir.StringImm("Expected non-null input")],
    )
    body = tir.SeqStmt([assert_stmt, tir.Evaluate(0)])
    func = tir.PrimFunc([x], body).with_attr(
        {"target": target, "global_symbol": "test_assert_runtime_error"}
    )

    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.MakePackedAPI()(mod)
    lib = tvm.build(mod, target=target)

    with pytest.raises(RuntimeError, match="Expected non-null input"):
        lib["test_assert_runtime_error"](0)


@tvm.testing.requires_llvm
def test_assert_value_error_llvm():
    """Test that AssertStmt with ValueError kind produces ValueError via LLVM."""
    target = tvm.target.Target("llvm")

    x = tir.Var("x", "int32")
    assert_stmt = tir.AssertStmt(
        tir.StringImm("ValueError"),
        tir.const(False, "bool"),
        [tir.StringImm("Shape mismatch: expected 4 got 8")],
    )
    body = tir.SeqStmt([assert_stmt, tir.Evaluate(0)])
    func = tir.PrimFunc([x], body).with_attr(
        {"target": target, "global_symbol": "test_assert_value_error"}
    )

    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.MakePackedAPI()(mod)
    lib = tvm.build(mod, target=target)

    with pytest.raises(ValueError, match="Shape mismatch"):
        lib["test_assert_value_error"](0)


@tvm.testing.requires_llvm
def test_assert_type_error_llvm():
    """Test that AssertStmt with TypeError kind produces TypeError via LLVM."""
    target = tvm.target.Target("llvm")

    x = tir.Var("x", "int32")
    assert_stmt = tir.AssertStmt(
        tir.StringImm("TypeError"),
        tir.const(False, "bool"),
        [tir.StringImm("Expected Tensor but got int")],
    )
    body = tir.SeqStmt([assert_stmt, tir.Evaluate(0)])
    func = tir.PrimFunc([x], body).with_attr(
        {"target": target, "global_symbol": "test_assert_type_error"}
    )

    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.MakePackedAPI()(mod)
    lib = tvm.build(mod, target=target)

    with pytest.raises(TypeError, match="Expected Tensor but got int"):
        lib["test_assert_type_error"](0)


@tvm.testing.requires_llvm
def test_assert_multi_part_message_llvm():
    """Test that multi-part messages are correctly concatenated via LLVM."""
    target = tvm.target.Target("llvm")

    x = tir.Var("x", "int32")
    assert_stmt = tir.AssertStmt(
        tir.StringImm("ValueError"),
        tir.const(False, "bool"),
        [
            tir.StringImm("Expected shape "),
            tir.StringImm("4"),
            tir.StringImm(" but got "),
            tir.StringImm("8"),
        ],
    )
    body = tir.SeqStmt([assert_stmt, tir.Evaluate(0)])
    func = tir.PrimFunc([x], body).with_attr(
        {"target": target, "global_symbol": "test_assert_multi_part"}
    )

    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.MakePackedAPI()(mod)
    lib = tvm.build(mod, target=target)

    with pytest.raises(ValueError, match="Expected shape 4 but got 8"):
        lib["test_assert_multi_part"](0)


@tvm.testing.requires_llvm
def test_assert_passing_condition_llvm():
    """Test that a passing assertion does not raise."""
    target = tvm.target.Target("llvm")

    x = tir.Var("x", "int32")
    assert_stmt = tir.AssertStmt(
        tir.StringImm("RuntimeError"),
        tir.const(True, "bool"),
        [tir.StringImm("This should not be raised")],
    )
    body = tir.SeqStmt([assert_stmt, tir.Evaluate(0)])
    func = tir.PrimFunc([x], body).with_attr(
        {"target": target, "global_symbol": "test_assert_passing"}
    )

    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.MakePackedAPI()(mod)
    lib = tvm.build(mod, target=target)

    # Should not raise
    lib["test_assert_passing"](0)


@tvm.testing.requires_llvm
def test_assert_many_parts_llvm():
    """Test assertion with more than 6 parts (triggers larger helper)."""
    target = tvm.target.Target("llvm")

    x = tir.Var("x", "int32")
    parts = [tir.StringImm(f"part{i}") for i in range(8)]
    assert_stmt = tir.AssertStmt(
        tir.StringImm("RuntimeError"),
        tir.const(False, "bool"),
        parts,
    )
    body = tir.SeqStmt([assert_stmt, tir.Evaluate(0)])
    func = tir.PrimFunc([x], body).with_attr(
        {"target": target, "global_symbol": "test_assert_many_parts"}
    )

    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.MakePackedAPI()(mod)
    lib = tvm.build(mod, target=target)

    with pytest.raises(RuntimeError, match="part0part1part2part3part4part5part6part7"):
        lib["test_assert_many_parts"](0)


def test_assert_ir_structure():
    """Test that AssertStmt IR nodes have correct structure."""
    kind = tir.StringImm("TypeError")
    cond = tir.const(True, "bool")
    parts = [tir.StringImm("msg1"), tir.StringImm("msg2")]
    stmt = tir.AssertStmt(kind, cond, parts)

    assert stmt.kind.value == "TypeError"
    assert len(stmt.message_parts) == 2
    assert stmt.message_parts[0].value == "msg1"
    assert stmt.message_parts[1].value == "msg2"


if __name__ == "__main__":
    tvm.testing.main()
