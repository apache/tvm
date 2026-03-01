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


def _build_assert_func_c(kind, message_parts):
    """Helper to build and compile a PrimFunc with an always-failing assert via C codegen."""
    target = tvm.target.Target("c")
    x = tir.Var("x", "int32")
    assert_stmt = tir.AssertStmt(
        tir.StringImm(kind),
        tir.const(False, "bool"),
        [tir.StringImm(p) for p in message_parts],
    )
    body = tir.SeqStmt([assert_stmt, tir.Evaluate(0)])
    func = tir.PrimFunc([x], body).with_attr({"global_symbol": "test_c_codegen"})
    return tvm.compile(tvm.IRModule.from_expr(func), target=target)


def test_assert_value_error_c():
    """Test that C host codegen raises ValueError with correct message."""
    lib = _build_assert_func_c("ValueError", ["Shape mismatch: expected 4 got 8"])
    with pytest.raises(ValueError, match="Shape mismatch"):
        lib["test_c_codegen"](0)


def test_assert_type_error_c():
    """Test that C host codegen raises TypeError with correct message."""
    lib = _build_assert_func_c("TypeError", ["Expected Tensor but got int"])
    with pytest.raises(TypeError, match="Expected Tensor but got int"):
        lib["test_c_codegen"](0)


def test_assert_multi_part_c():
    """Test that C host codegen correctly concatenates multi-part messages."""
    lib = _build_assert_func_c(
        "ValueError",
        ["Expected shape ", "4", " but got ", "8"],
    )
    with pytest.raises(ValueError, match="Expected shape 4 but got 8"):
        lib["test_c_codegen"](0)


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


# ============================================================
# Phase 1: Integration tests for rich error messages
# ============================================================


@tvm.testing.requires_llvm
def test_wrong_argument_count_error():
    """Verify wrong argument count produces TypeError with function signature."""

    @T.prim_func
    def add_one(a: T.Buffer((8,), "float32"), b: T.Buffer((8,), "float32")):
        for i in range(8):
            b[i] = a[i] + T.float32(1)

    lib = tvm.compile(add_one, target="llvm")

    with pytest.raises(TypeError, match="Expected 2 arguments"):
        lib()

    with pytest.raises(TypeError, match="add_one"):
        lib()


@tvm.testing.requires_llvm
def test_type_mismatch_error():
    """Verify wrong type produces TypeError with function signature."""

    @T.prim_func
    def scale(a: T.Buffer((8,), "float32"), b: T.Buffer((8,), "float32")):
        for i in range(8):
            b[i] = a[i] * T.float32(2)

    lib = tvm.compile(scale, target="llvm")

    # Passing an integer instead of a tensor should produce a TypeError
    with pytest.raises(TypeError, match="Mismatched type on argument #0"):
        lib(42, 0)


@tvm.testing.requires_llvm
def test_ndim_mismatch_error():
    """Verify ndim mismatch produces ValueError with function signature."""

    @T.prim_func
    def func_2d(a: T.Buffer((4, 4), "float32"), b: T.Buffer((4, 4), "float32")):
        for i, j in T.grid(4, 4):
            b[i, j] = a[i, j]

    lib = tvm.compile(func_2d, target="llvm")

    # Pass a 1D array where 2D is expected
    a = tvm.runtime.tensor(np.zeros(4, dtype="float32"))
    b = tvm.runtime.tensor(np.zeros(4, dtype="float32"))

    with pytest.raises(ValueError, match="ndim"):
        lib(a, b)

    # Verify function signature appears in the error message
    with pytest.raises(ValueError, match="func_2d"):
        lib(a, b)


@tvm.testing.requires_llvm
def test_shape_mismatch_error():
    """Verify shape mismatch produces ValueError with function signature."""

    @T.prim_func
    def add(a: T.Buffer((8,), "float32"), b: T.Buffer((8,), "float32")):
        for i in range(8):
            b[i] = a[i] + T.float32(1)

    lib = tvm.compile(add, target="llvm")

    # Pass wrong shape
    a = tvm.runtime.tensor(np.zeros(16, dtype="float32"))
    b = tvm.runtime.tensor(np.zeros(16, dtype="float32"))

    with pytest.raises(ValueError, match="add"):
        lib(a, b)


@tvm.testing.requires_llvm
def test_dtype_mismatch_error():
    """Verify dtype mismatch produces TypeError with function signature."""

    @T.prim_func
    def copy_f32(a: T.Buffer((8,), "float32"), b: T.Buffer((8,), "float32")):
        for i in range(8):
            b[i] = a[i]

    lib = tvm.compile(copy_f32, target="llvm")

    # Pass int32 array where float32 is expected
    a = tvm.runtime.tensor(np.zeros(8, dtype="int32"))
    b = tvm.runtime.tensor(np.zeros(8, dtype="float32"))

    with pytest.raises(TypeError, match="dtype"):
        lib(a, b)

    with pytest.raises(TypeError, match="copy_f32"):
        lib(a, b)


@tvm.testing.requires_llvm
def test_make_packed_api_signature_in_asserts():
    """Verify MakePackedAPI generates structured error messages with signature."""

    @T.prim_func
    def add_one(a: T.Buffer((8,), "float32"), b: T.Buffer((8,), "float32")):
        T.func_attr(
            {
                "target": tvm.target.Target("llvm", host="llvm"),
                "global_symbol": "add_one",
            }
        )
        for i in range(8):
            b[i] = a[i] + T.float32(1)

    mod = tvm.IRModule.from_expr(add_one)
    mod = tvm.tir.transform.MakePackedAPI()(mod)
    func = mod["add_one"]

    # Walk the IR and find AssertStmt nodes
    asserts = []

    def visitor(stmt):
        if isinstance(stmt, tvm.tir.AssertStmt):
            asserts.append(stmt)

    tvm.tir.stmt_functor.post_order_visit(func.body, visitor)

    # Should have asserts
    assert len(asserts) > 0

    # Verify kind is set correctly (TypeError for type checks, ValueError for shape)
    kinds = {a.kind.value for a in asserts}
    assert "TypeError" in kinds

    # Verify signature fragment appears in at least one assert
    assert any(any("add_one" in part.value for part in a.message_parts) for a in asserts)

    # Verify signature contains buffer info
    assert any(any("Tensor" in part.value for part in a.message_parts) for a in asserts)


if __name__ == "__main__":
    tvm.testing.main()
