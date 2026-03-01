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
"""Tests for tir.transform.MakePackedAPI TIR transform.

Tests verify the transform output using TVMScript before/after patterns.
Runtime error tests are in tests/python/codegen/test_codegen_error_handling.py.
"""

import pytest

import tvm
import tvm.testing
from tvm import tir
from tvm.script import ir as I
from tvm.script import tir as T


def _find_compute_scope(func):
    result = None

    def _visitor(stmt):
        if isinstance(stmt, tir.AttrStmt) and stmt.attr_key == "compute_scope":
            nonlocal result
            result = stmt

    tir.stmt_functor.post_order_visit(func.body, _visitor)

    return result


@pytest.mark.parametrize("use_global_symbol", [True, False])
def test_no_op_when_global_symbol_is_absent(use_global_symbol):
    func_attr = {"target": tvm.target.Target("llvm", host="llvm")}

    @T.prim_func(private=True)
    def before():
        T.func_attr(func_attr)
        T.evaluate(0)

    if use_global_symbol:
        before = before.with_attr("global_symbol", "main")

    after = tvm.tir.transform.MakePackedAPI()(tvm.IRModule.from_expr(before))["main"]
    if use_global_symbol:
        assert len(after.params) == 4
    else:
        tvm.ir.assert_structural_equal(before, after)


def test_target_host_removed():
    """After MakePackedAPI, host-side target should be the host

    MakePackedAPI is the last transform that requires both the device
    and the host.  After MakePackedAPI, the target attribute should
    only contain the host-side target.
    """

    host = tvm.target.Target("llvm")

    @I.ir_module
    class before:
        @T.prim_func
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"global_symbol": "main", "target": T.target("cuda", host=host)})
            T.evaluate(0)

    after = tvm.tir.transform.MakePackedAPI()(before)
    target_attr = after["main"].attrs["target"]
    assert str(host) == str(target_attr)


def test_internal_subroutine_call():
    """Internal subroutines should not use the PackedFunc API

    A subroutine without the "global_symbol" attribute is an internal
    subroutine, and is not directly exposed to a user of the generated
    `runtime.Module`.  Therefore, it doesn't need to follow the
    PackedFunc API.
    """

    @I.ir_module
    class before:
        @T.prim_func
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            before.subroutine(A.data)

        # this test fails if it's made public
        @T.prim_func(private=True)
        def subroutine(A_data: T.handle("float32")):
            T.func_attr({"target": T.target("llvm")})
            T.evaluate(A_data)

    after = tvm.tir.transform.MakePackedAPI()(before)
    tvm.ir.assert_structural_equal(before["subroutine"], after["subroutine"])

    compute_scope = _find_compute_scope(after["main"])
    subroutine_call_op = compute_scope.body.value.op
    assert isinstance(subroutine_call_op, tvm.ir.GlobalVar), (
        f"The main function's CallNode should use the subroutine's GlobalVar as the operation, "
        f"but instead has an operation of type {subroutine_call_op}"
    )


def test_subroutine_call_to_externally_visible_subroutine():
    """Externally-visible subroutines should use the PackedFunc API

    Because the subroutine may be called directly by a user, it must
    use the PackedFunc API.  Its signature should be updated to the
    PackedFunc signature, and call sites should be updated to use
    `T.tvm_call_cpacked`.
    """

    @I.ir_module
    class before:
        @T.prim_func
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"global_symbol": "main", "target": T.target("llvm", host="llvm")})
            before.subroutine(A.data)

        @T.prim_func
        def subroutine(A_data: T.handle("float32")):
            T.func_attr({"global_symbol": "subroutine", "target": T.target("llvm", host="llvm")})
            T.evaluate(A_data)

    after = tvm.tir.transform.MakePackedAPI()(before)

    main_compute_scope = _find_compute_scope(after["main"])
    assert main_compute_scope is not None
    subroutine_compute_scope = _find_compute_scope(after["subroutine"])
    assert subroutine_compute_scope is not None

    subroutine_call_op = main_compute_scope.body.value.op
    assert (
        isinstance(subroutine_call_op, tvm.ir.Op)
        and subroutine_call_op.name == "tir.tvm_call_cpacked"
    ), (
        f"The main function's CallNode should be lowered to the builtin 'tir.tvm_call_cpacked', "
        f"but instead has an operation of type {subroutine_call_op}"
    )


def _collect_asserts(func):
    """Collect all AssertStmt nodes from a function body."""
    asserts = []

    def _visitor(stmt):
        if isinstance(stmt, tir.AssertStmt):
            asserts.append(stmt)

    tir.stmt_functor.post_order_visit(func.body, _visitor)
    return asserts


def _assert_msg(assert_stmt):
    """Join message_parts of an AssertStmt into a single string."""
    return "".join(p.value for p in assert_stmt.message_parts)


def test_zero_arg_function():
    """Zero-arg function emits num_args check but no null-pointer check."""

    @I.ir_module
    class Before:
        @T.prim_func
        def func_without_arg() -> T.int64:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            return T.int64(42)

    After = tvm.tir.transform.MakePackedAPI()(Before)
    func = After["func_without_arg"]

    assert len(func.params) == 4
    assert func.attrs["calling_conv"] == 1
    assert func.attrs["global_symbol"] == "__tvm_ffi_func_without_arg"

    asserts = _collect_asserts(func)
    assert len(asserts) >= 1
    assert asserts[0].error_kind.value == "TypeError"
    assert "Expected 0 arguments" in _assert_msg(asserts[0])
    assert "func_without_arg()" in _assert_msg(asserts[0])


def test_int_parameter():
    """Int parameter emits type check accepting int or bool."""

    @I.ir_module
    class Before:
        @T.prim_func
        def main(arg: T.int32) -> T.int32:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            if arg > 0:
                return 10
            else:
                return 20

    After = tvm.tir.transform.MakePackedAPI()(Before)
    func = After["main"]

    asserts = _collect_asserts(func)
    assert len(asserts) >= 3  # num_args, null check, type check

    # Verify function signature in error messages
    assert any("main(arg: int32)" in _assert_msg(a) for a in asserts)

    # Verify type check with "expected int"
    type_checks = [a for a in asserts if a.error_kind.value == "TypeError"]
    assert any("expected int" in _assert_msg(tc) for tc in type_checks)


def test_bool_parameter():
    """Bool parameter emits type check accepting bool or int."""

    @I.ir_module
    class Before:
        @T.prim_func
        def main(arg: T.bool) -> T.int32:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            if arg:
                return 10
            else:
                return 20

    After = tvm.tir.transform.MakePackedAPI()(Before)
    func = After["main"]

    asserts = _collect_asserts(func)
    assert len(asserts) >= 3

    assert any("main(arg: bool)" in _assert_msg(a) for a in asserts)

    type_checks = [a for a in asserts if a.error_kind.value == "TypeError"]
    assert any("expected boolean" in _assert_msg(tc) for tc in type_checks)


def test_float_parameter():
    """Float parameter emits type check accepting float, int, or bool."""

    @I.ir_module
    class Before:
        @T.prim_func
        def main(arg: T.float32) -> T.int32:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            if arg > T.float32(0):
                return 10
            else:
                return 20

    After = tvm.tir.transform.MakePackedAPI()(Before)
    func = After["main"]

    asserts = _collect_asserts(func)
    assert len(asserts) >= 3

    assert any("main(arg: float32)" in _assert_msg(a) for a in asserts)

    type_checks = [a for a in asserts if a.error_kind.value == "TypeError"]
    assert any("expected float" in _assert_msg(tc) for tc in type_checks)


if __name__ == "__main__":
    tvm.testing.main()
