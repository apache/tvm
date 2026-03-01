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
"""Phase 0: Basic AssertStmt codegen tests for kind and message_parts."""

import pytest

import tvm
import tvm.testing
from tvm import tir

# Phase 0 tests only need LLVM; they test basic AssertStmt codegen infrastructure
codegen_target = tvm.testing.parameter("llvm")


def _build_assert_func(tgt, kind, message_parts, func_name="test_func"):
    """Build and compile a PrimFunc with an always-failing assert."""
    target_obj = tvm.target.Target(tgt)
    x = tir.Var("x", "int32")
    assert_stmt = tir.AssertStmt(
        tir.StringImm(kind),
        tir.const(False, "bool"),
        [tir.StringImm(p) for p in message_parts],
    )
    body = tir.SeqStmt([assert_stmt, tir.Evaluate(0)])
    func = tir.PrimFunc([x], body).with_attr(
        {"target": target_obj, "global_symbol": func_name}
    )
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.MakePackedAPI()(mod)
    return tvm.build(mod, target=target_obj), func_name


def test_assert_runtime_error(codegen_target):
    """Test that AssertStmt with RuntimeError kind produces correct error."""
    lib, name = _build_assert_func(codegen_target, "RuntimeError", ["Expected non-null input"])
    with pytest.raises(RuntimeError, match="Expected non-null input"):
        lib[name](0)


def test_assert_value_error(codegen_target):
    """Test that AssertStmt with ValueError kind produces ValueError."""
    lib, name = _build_assert_func(
        codegen_target, "ValueError", ["Shape mismatch: expected 4 got 8"]
    )
    with pytest.raises(ValueError, match="Shape mismatch"):
        lib[name](0)


def test_assert_type_error(codegen_target):
    """Test that AssertStmt with TypeError kind produces TypeError."""
    lib, name = _build_assert_func(
        codegen_target, "TypeError", ["Expected Tensor but got int"]
    )
    with pytest.raises(TypeError, match="Expected Tensor but got int"):
        lib[name](0)


def test_assert_multi_part_message(codegen_target):
    """Test that multi-part messages are correctly concatenated."""
    lib, name = _build_assert_func(
        codegen_target, "ValueError", ["Expected shape ", "4", " but got ", "8"]
    )
    with pytest.raises(ValueError, match="Expected shape 4 but got 8"):
        lib[name](0)


def test_assert_passing_condition(codegen_target):
    """Test that a passing assertion does not raise."""
    target_obj = tvm.target.Target(codegen_target)
    x = tir.Var("x", "int32")
    assert_stmt = tir.AssertStmt(
        tir.StringImm("RuntimeError"),
        tir.const(True, "bool"),
        [tir.StringImm("This should not be raised")],
    )
    body = tir.SeqStmt([assert_stmt, tir.Evaluate(0)])
    func = tir.PrimFunc([x], body).with_attr(
        {"target": target_obj, "global_symbol": "test_passing"}
    )
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.MakePackedAPI()(mod)
    lib = tvm.build(mod, target=target_obj)
    lib["test_passing"](0)


def test_assert_many_parts(codegen_target):
    """Test assertion with more than 6 parts (triggers larger helper in LLVM)."""
    parts = [f"part{i}" for i in range(8)]
    lib, name = _build_assert_func(codegen_target, "RuntimeError", parts)
    with pytest.raises(RuntimeError, match="part0part1part2part3part4part5part6part7"):
        lib[name](0)


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
