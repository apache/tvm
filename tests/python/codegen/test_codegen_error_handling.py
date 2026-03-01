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
"""Phase 1: Rich error message integration tests for MakePackedAPI + ArgBinder."""

import re

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T

# Parameterize over both LLVM and C backends
codegen_target = tvm.testing.parameter("llvm", "c")


def _make_add_one_shared_shape(tgt):
    """Create add_one(a: Tensor([n0], float32), b: Tensor([n0], float32))
    where a and b share shape variable n0."""
    n = tir.Var("n0", "int64")
    a_buf = tir.decl_buffer([n], "float32", name="a")
    b_buf = tir.decl_buffer([n], "float32", name="b")
    a_param = tir.Var("a_handle", "handle")
    b_param = tir.Var("b_handle", "handle")
    i = tir.Var("i", "int64")

    body = tir.For(
        i,
        0,
        n,
        tir.ForKind.SERIAL,
        tir.BufferStore(
            b_buf,
            tir.BufferLoad(a_buf, [i]) + tir.const(1.0, "float32"),
            [i],
        ),
    )
    func = tir.PrimFunc(
        [a_param, b_param], body, buffer_map={a_param: a_buf, b_param: b_buf}
    ).with_attr(
        {
            "global_symbol": "add_one",
            "target": tvm.target.Target(tgt, host=tgt),
        }
    )
    return func


def _make_add_one_aligned(tgt):
    """Create add_one with offset_factor=4 on buffer a (16-byte alignment)."""
    n = tir.Var("n0", "int64")
    a_buf = tir.decl_buffer([n], "float32", name="a", data_alignment=64, offset_factor=4)
    b_buf = tir.decl_buffer([n], "float32", name="b")
    a_param = tir.Var("a_handle", "handle")
    b_param = tir.Var("b_handle", "handle")
    i = tir.Var("i", "int64")

    body = tir.For(
        i,
        0,
        n,
        tir.ForKind.SERIAL,
        tir.BufferStore(
            b_buf,
            tir.BufferLoad(a_buf, [i]) + tir.const(1.0, "float32"),
            [i],
        ),
    )
    func = tir.PrimFunc(
        [a_param, b_param], body, buffer_map={a_param: a_buf, b_param: b_buf}
    ).with_attr(
        {
            "global_symbol": "add_one",
            "target": tvm.target.Target(tgt, host=tgt),
        }
    )
    return func


def test_type_mismatch_non_tensor(codegen_target):
    """Test: passing a non-tensor where a tensor is expected."""
    func = _make_add_one_shared_shape(codegen_target)
    lib = tvm.compile(tvm.IRModule.from_expr(func), target=codegen_target)
    a = tvm.runtime.tensor(np.zeros(128, dtype="float32"))

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Mismatched type on argument #1 when calling:\n"
            "  `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32))`,\n"
            "  expected Tensor"
        ),
    ):
        lib["add_one"](a, 1)


def test_shape_mismatch_shared_variable(codegen_target):
    """Test: b has different shape than a when they share variable n0."""
    func = _make_add_one_shared_shape(codegen_target)
    lib = tvm.compile(tvm.IRModule.from_expr(func), target=codegen_target)
    a = tvm.runtime.tensor(np.zeros(128, dtype="float32"))
    b_short = tvm.runtime.tensor(np.zeros(126, dtype="float32"))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Mismatched b.shape[0] on argument #1 when calling:\n"
            "  `add_one(a: Tensor([n0], float32), b: Tensor([n0], float32))`,\n"
            "  expected to match a.shape[0]"
        ),
    ):
        lib["add_one"](a, b_short)


def test_invalid_shape_fixed(codegen_target):
    """Test: passing a shape that doesn't match a fixed buffer dimension."""

    @T.prim_func
    def add(a: T.Buffer((128,), "float32"), b: T.Buffer((128,), "float32")):
        for i in range(128):
            b[i] = a[i] + T.float32(1)

    lib = tvm.compile(add, target=codegen_target)
    a_wrong = tvm.runtime.tensor(np.zeros(256, dtype="float32"))
    b_wrong = tvm.runtime.tensor(np.zeros(256, dtype="float32"))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid a.shape[0] on argument #0 when calling:\n"
            "  `add(a: Tensor([128], float32), b: Tensor([128], float32))`,\n"
            "  expected 128"
        ),
    ):
        lib(a_wrong, b_wrong)


def test_misaligned_tensor_data(codegen_target):
    """Test: misaligned tensor data when buffer requires alignment."""
    func = _make_add_one_aligned(codegen_target)
    mod = tvm.IRModule.from_expr(func)
    lowered = tvm.tir.transform.MakePackedAPI()(mod)
    lowered_func = lowered["add_one"]

    alignment_asserts = []

    def visitor(stmt):
        if isinstance(stmt, tvm.tir.AssertStmt):
            msg = "".join(p.value for p in stmt.message_parts)
            if "Misaligned" in msg:
                alignment_asserts.append(msg)

    tvm.tir.stmt_functor.post_order_visit(lowered_func.body, visitor)
    assert len(alignment_asserts) >= 1
    assert "Misaligned Tensor data on argument #0" in alignment_asserts[0]
    assert "expected data alignment=16 bytes" in alignment_asserts[0]
    assert "add_one(a: Tensor([n0], float32), b: Tensor([n0], float32))" in alignment_asserts[0]


def test_wrong_argument_count_error(codegen_target):
    """Verify wrong argument count produces TypeError with function signature."""
    func = _make_add_one_shared_shape(codegen_target)
    lib = tvm.compile(tvm.IRModule.from_expr(func), target=codegen_target)

    with pytest.raises(TypeError, match="Expected 2 arguments"):
        lib["add_one"]()

    with pytest.raises(TypeError, match=re.escape("add_one(")):
        lib["add_one"]()


def test_ndim_mismatch_error(codegen_target):
    """Verify ndim mismatch produces ValueError with function signature."""

    @T.prim_func
    def func_2d(a: T.Buffer((4, 4), "float32"), b: T.Buffer((4, 4), "float32")):
        for i, j in T.grid(4, 4):
            b[i, j] = a[i, j]

    lib = tvm.compile(func_2d, target=codegen_target)
    a = tvm.runtime.tensor(np.zeros(4, dtype="float32"))
    b = tvm.runtime.tensor(np.zeros(4, dtype="float32"))

    with pytest.raises(
        ValueError,
        match=re.escape("Mismatched a.ndim on argument #0"),
    ):
        lib(a, b)


def test_dtype_mismatch_error(codegen_target):
    """Verify dtype mismatch produces TypeError with function signature."""

    @T.prim_func
    def copy_f32(a: T.Buffer((8,), "float32"), b: T.Buffer((8,), "float32")):
        for i in range(8):
            b[i] = a[i]

    lib = tvm.compile(copy_f32, target=codegen_target)
    a = tvm.runtime.tensor(np.zeros(8, dtype="int32"))
    b = tvm.runtime.tensor(np.zeros(8, dtype="float32"))

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Mismatched a.dtype on argument #0 when calling:\n"
            "  `copy_f32(a: Tensor([8], float32), b: Tensor([8], float32))`,\n"
            "  expected float32"
        ),
    ):
        lib(a, b)


def test_make_packed_api_signature_in_asserts(codegen_target):
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

    asserts = []

    def visitor(stmt):
        if isinstance(stmt, tvm.tir.AssertStmt):
            asserts.append(stmt)

    tvm.tir.stmt_functor.post_order_visit(func.body, visitor)
    assert len(asserts) > 0
    kinds = {a.kind.value for a in asserts}
    assert "TypeError" in kinds
    assert any(any("add_one" in part.value for part in a.message_parts) for a in asserts)
    assert any(
        any("a: Tensor([8], float32)" in part.value for part in a.message_parts) for a in asserts
    )


def test_error_message_format_consistency(codegen_target):
    """Verify all rich error messages follow the standard format."""
    func = _make_add_one_shared_shape(codegen_target)
    mod = tvm.IRModule.from_expr(func)
    lowered = tvm.tir.transform.MakePackedAPI()(mod)
    lowered_func = lowered["add_one"]

    asserts = []

    def visitor(stmt):
        if isinstance(stmt, tvm.tir.AssertStmt):
            msg = "".join(p.value for p in stmt.message_parts)
            asserts.append((stmt.kind.value, msg))

    tvm.tir.stmt_functor.post_order_visit(lowered_func.body, visitor)

    sig = "add_one(a: Tensor([n0], float32), b: Tensor([n0], float32))"
    for kind, msg in asserts:
        if sig in msg:
            assert "when calling:" in msg, f"Missing 'when calling:' in: {msg}"
            assert f"`{sig}`" in msg, f"Missing backtick-wrapped signature in: {msg}"


if __name__ == "__main__":
    tvm.testing.main()
