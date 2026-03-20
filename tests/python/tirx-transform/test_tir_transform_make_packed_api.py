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
"""Tests for tirx.transform.MakePackedAPI TIR transform.

Tests verify the transform output using TVMScript before/after patterns.
Runtime error tests are in tests/python/codegen/test_codegen_error_handling.py.
"""

import pytest

import tvm
import tvm.testing
from tvm import tirx
from tvm.script import ir as I
from tvm.script import tirx as T


def _find_compute_scope(func):
    result = None

    def _visitor(stmt):
        if isinstance(stmt, tirx.AttrStmt) and stmt.attr_key == "compute_scope":
            nonlocal result
            result = stmt

    tirx.stmt_functor.post_order_visit(func.body, _visitor)

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

    after = tvm.tirx.transform.MakePackedAPI()(tvm.IRModule.from_expr(before))["main"]
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

    after = tvm.tirx.transform.MakePackedAPI()(before)
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

    after = tvm.tirx.transform.MakePackedAPI()(before)
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

    after = tvm.tirx.transform.MakePackedAPI()(before)

    main_compute_scope = _find_compute_scope(after["main"])
    assert main_compute_scope is not None
    subroutine_compute_scope = _find_compute_scope(after["subroutine"])
    assert subroutine_compute_scope is not None

    subroutine_call_op = main_compute_scope.body.value.op
    assert (
        isinstance(subroutine_call_op, tvm.ir.Op)
        and subroutine_call_op.name == "tirx.tvm_call_cpacked"
    ), (
        f"The main function's CallNode should be lowered to the builtin 'tirx.tvm_call_cpacked', "
        f"but instead has an operation of type {subroutine_call_op}"
    )


def test_zero_arg_function():
    """Zero-arg function emits num_args check but no null-pointer check."""

    @I.ir_module
    class Before:
        @T.prim_func
        def func_without_arg() -> T.int64:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            return T.int64(42)

    @I.ir_module
    class Expected:
        @T.prim_func
        def func_without_arg(
            self_handle: T.handle,
            args: T.handle,
            num_args: T.int32,
            result: T.handle("void", "global"),
        ) -> T.int32:
            T.func_attr(
                {
                    "calling_conv": 1,
                    "global_symbol": "__tvm_ffi_func_without_arg",
                    "target": T.target("llvm"),
                }
            )
            assert num_args == 0, (
                "TypeError",
                ["Expected ", "0", " arguments", " when calling:\n  `", "func_without_arg()", "`"],
            )
            with T.attr(0, "compute_scope", "func_without_arg_compute_"):
                T.tvm_struct_set(result, 0, 13, 1)
                T.tvm_struct_set(result, 0, 14, 0)
                T.tvm_struct_set(result, 0, 15, T.Cast("int64", T.int64(42)))
                return 0
            return 0

    After = tvm.tirx.transform.MakePackedAPI()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


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

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(
            self_handle: T.handle,
            args: T.handle,
            num_args: T.int32,
            result: T.handle("void", "global"),
        ) -> T.int32:
            T.func_attr(
                {
                    "calling_conv": 1,
                    "global_symbol": "__tvm_ffi_main",
                    "target": T.target("llvm"),
                }
            )
            assert num_args == 1, (
                "TypeError",
                ["Expected ", "1", " arguments", " when calling:\n  `", "main(arg: int32)", "`"],
            )
            assert not T.isnullptr(args), (
                "TypeError",
                ["args pointer is NULL", " when calling:\n  `", "main(arg: int32)", "`"],
            )
            arg_type_index: T.int32 = T.tvm_struct_get(args, 0, 13, "int32")
            assert arg_type_index == 1 or arg_type_index == 2, (
                "TypeError",
                [
                    "Mismatched type on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(arg: int32)",
                    "`,\n  expected ",
                    "int",
                ],
            )
            arg: T.int32 = T.Cast("int32", T.tvm_struct_get(args, 0, 15, "int64"))
            with T.attr(0, "compute_scope", "main_compute_"):
                if arg > 0:
                    T.tvm_struct_set(result, 0, 13, 1)
                    T.tvm_struct_set(result, 0, 14, 0)
                    T.tvm_struct_set(result, 0, 15, T.Cast("int64", 10))
                    return 0
                else:
                    T.tvm_struct_set(result, 0, 13, 1)
                    T.tvm_struct_set(result, 0, 14, 0)
                    T.tvm_struct_set(result, 0, 15, T.Cast("int64", 20))
                    return 0
            return 0

    After = tvm.tirx.transform.MakePackedAPI()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


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

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(
            self_handle: T.handle,
            args: T.handle,
            num_args: T.int32,
            result: T.handle("void", "global"),
        ) -> T.int32:
            T.func_attr(
                {
                    "calling_conv": 1,
                    "global_symbol": "__tvm_ffi_main",
                    "target": T.target("llvm"),
                }
            )
            assert num_args == 1, (
                "TypeError",
                ["Expected ", "1", " arguments", " when calling:\n  `", "main(arg: bool)", "`"],
            )
            assert not T.isnullptr(args), (
                "TypeError",
                ["args pointer is NULL", " when calling:\n  `", "main(arg: bool)", "`"],
            )
            arg_type_index: T.int32 = T.tvm_struct_get(args, 0, 13, "int32")
            assert arg_type_index == 2 or arg_type_index == 1, (
                "TypeError",
                [
                    "Mismatched type on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(arg: bool)",
                    "`,\n  expected ",
                    "boolean",
                ],
            )
            arg: T.bool = T.Cast("bool", T.tvm_struct_get(args, 0, 15, "int64"))
            with T.attr(0, "compute_scope", "main_compute_"):
                if arg:
                    T.tvm_struct_set(result, 0, 13, 1)
                    T.tvm_struct_set(result, 0, 14, 0)
                    T.tvm_struct_set(result, 0, 15, T.Cast("int64", 10))
                    return 0
                else:
                    T.tvm_struct_set(result, 0, 13, 1)
                    T.tvm_struct_set(result, 0, 14, 0)
                    T.tvm_struct_set(result, 0, 15, T.Cast("int64", 20))
                    return 0
            return 0

    After = tvm.tirx.transform.MakePackedAPI()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


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

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(
            self_handle: T.handle,
            args: T.handle,
            num_args: T.int32,
            result: T.handle("void", "global"),
        ) -> T.int32:
            T.func_attr(
                {
                    "calling_conv": 1,
                    "global_symbol": "__tvm_ffi_main",
                    "target": T.target("llvm"),
                }
            )
            assert num_args == 1, (
                "TypeError",
                ["Expected ", "1", " arguments", " when calling:\n  `", "main(arg: float32)", "`"],
            )
            assert not T.isnullptr(args), (
                "TypeError",
                ["args pointer is NULL", " when calling:\n  `", "main(arg: float32)", "`"],
            )
            arg_type_index: T.int32 = T.tvm_struct_get(args, 0, 13, "int32")
            assert arg_type_index == 3 or arg_type_index == 1 or arg_type_index == 2, (
                "TypeError",
                [
                    "Mismatched type on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(arg: float32)",
                    "`,\n  expected ",
                    "float",
                ],
            )
            arg: T.float32 = T.Select(
                arg_type_index == 3,
                T.Cast("float32", T.tvm_struct_get(args, 0, 15, "float64")),
                T.Cast("float32", T.tvm_struct_get(args, 0, 15, "int64")),
            )
            with T.attr(0, "compute_scope", "main_compute_"):
                if arg > T.float32(0.0):
                    T.tvm_struct_set(result, 0, 13, 1)
                    T.tvm_struct_set(result, 0, 14, 0)
                    T.tvm_struct_set(result, 0, 15, T.Cast("int64", 10))
                    return 0
                else:
                    T.tvm_struct_set(result, 0, 13, 1)
                    T.tvm_struct_set(result, 0, 14, 0)
                    T.tvm_struct_set(result, 0, 15, T.Cast("int64", 20))
                    return 0
            return 0

    After = tvm.tirx.transform.MakePackedAPI()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_forward_reference_symbolic_variable():
    """MakePackedAPI succeeds when a symbolic variable is used before it is defined.

    When buffer A has shape (batch_size+1,) and buffer B has shape (batch_size,),
    batch_size is referenced (in A's shape check) before it is defined (from B's
    shape). The three-sequence separation (init_nest, asserts, decl_buffers)
    ensures all variable definitions precede all assertions.
    """

    @I.ir_module
    class Before:
        @T.prim_func
        def main(a: T.handle, b: T.handle):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            batch_size = T.int64()
            A = T.match_buffer(a, (batch_size + 1,), "int32")
            B = T.match_buffer(b, (batch_size,), "int32")
            for i in range(batch_size):
                B[i] = A[i] + A[i + 1]

    # Should not raise "variable batch_size has been used before definition"
    After = tvm.tirx.transform.MakePackedAPI()(Before)
    assert len(After["main"].params) == 4


if __name__ == "__main__":
    tvm.testing.main()
