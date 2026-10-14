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
Runtime error tests are in tests/python/codegen/test_argument_validation.py.
"""

import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T


def _is_tensor_binding(node, *op_names):
    return (
        isinstance(node, tvm.ir.Bind)
        and isinstance(node.value, tvm.ir.Call)
        and isinstance(node.value.op, tvm.ir.Op)
        and node.value.op.name in op_names
    )


def _find_compute_scope(func):
    result = None

    def _visitor(stmt):
        if isinstance(stmt, tvm.ir.RegionStmt) and stmt.op.same_as(
            tvm.ir.Op.get("tirx.compute_scope")
        ):
            nonlocal result
            result = stmt

    tvm_ffi.structural_walk(func.body, _visitor)

    return result


def test_no_op_when_global_symbol_is_absent():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            T.func_attr({"target": T.target("llvm", host="llvm")})
            T.evaluate(0)

    After = tvm.tirx.transform.MakePackedAPI()(Before)
    tvm.ir.assert_structural_equal(After, Before)


def test_target_host_removed():
    """After MakePackedAPI, host-side target should be the host

    MakePackedAPI is the last transform that requires both the device
    and the host.  After MakePackedAPI, the target attribute should
    only contain the host-side target.
    """

    host = tvm.target.Target("llvm")

    @I.ir_module
    class before:
        @T.function
        def main(A: T.Tensor(1, "float32")):
            T.func_attr({"global_symbol": "main", "target": T.target("cuda", host=host)})
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.function
        def main(
            self_handle: T.Ptr(T.void),
            args: T.Ptr(T.void),
            num_args: T.int32,
            result: T.Ptr(T.void),
        ) -> T.int32:
            T.func_attr({"global_symbol": "__tvm_ffi_main"})
            T.func_attr({"calling_conv": 1, "target": T.target("llvm")})
            assert num_args == 1, (
                "TypeError",
                [
                    "Expected ",
                    "1",
                    " arguments",
                    " when calling:\n  `",
                    "main(A: Tensor([1], float32))",
                    "`",
                ],
            )
            assert not T.isnullptr(args), (
                "TypeError",
                [
                    "args pointer is NULL",
                    " when calling:\n  `",
                    "main(A: Tensor([1], float32))",
                    "`",
                ],
            )
            A_type_index: T.let[T.int32] = T.abi_field_get(args, 0, 13, ty="int32")
            assert (
                A_type_index == 0 or A_type_index == 4 or A_type_index == 7 or A_type_index >= 64
            ), (
                "TypeError",
                [
                    "Mismatched type on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(A: Tensor([1], float32))",
                    "`,\n  expected ",
                    "Tensor",
                ],
            )
            A_handle: T.let[T.Ptr(T.void)] = T.if_then_else(
                A_type_index == 70,
                T.handle_add_byte_offset(T.abi_field_get(args, 0, 15, ty=T.Ptr(T.void)), 24),
                T.abi_field_get(args, 0, 15, ty=T.Ptr(T.void)),
            )
            assert not T.isnullptr(A_handle), (
                "TypeError",
                [
                    "Mismatched type on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(A: Tensor([1], float32))",
                    "`,\n  expected ",
                    "Tensor",
                ],
            )
            assert 1 == T.abi_field_get(A_handle, 0, 4, ty="int32"), (
                "ValueError",
                [
                    "Mismatched ",
                    "A",
                    ".ndim on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(A: Tensor([1], float32))",
                    "`,\n  expected ",
                    "1",
                ],
            )
            assert (
                T.abi_field_get(A_handle, 0, 5, ty="uint8") == T.uint8(2)
                and T.abi_field_get(A_handle, 0, 6, ty="uint8") == T.uint8(32)
                and T.abi_field_get(A_handle, 0, 7, ty="uint16") == T.uint16(1)
            ), (
                "TypeError",
                [
                    "Mismatched ",
                    "A",
                    ".dtype on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(A: Tensor([1], float32))",
                    "`,\n  expected ",
                    "float32",
                ],
            )
            main_A_shape: T.let[T.Ptr(T.PrimType("int64"))] = T.abi_field_get(
                A_handle, 0, 2, ty=T.Ptr(T.PrimType("int64"))
            )
            main_A_strides: T.let[T.Ptr(T.PrimType("int64"))] = T.abi_field_get(
                A_handle, 0, 3, ty=T.Ptr(T.PrimType("int64"))
            )
            assert T.abi_field_get(A_handle, 0, 10, ty="int32") == 2, (
                "ValueError",
                [
                    "Mismatched ",
                    "A",
                    ".device_type on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(A: Tensor([1], float32))",
                    "`,\n  expected ",
                    "cuda",
                ],
            )
            dev_id: T.let[T.int32] = T.abi_field_get(A_handle, 0, 9, ty="int32")
            with T.device_context(2, dev_id):
                if not T.isnullptr(main_A_strides):
                    assert T.bool(True), (
                        "ValueError",
                        [
                            "Mismatched ",
                            "A",
                            ".strides on argument #",
                            "0",
                            " when calling:\n  `",
                            "main(A: Tensor([1], float32))",
                            "`,\n  expected to be compact array",
                        ],
                    )
                assert not T.isnullptr(
                    T.reinterpret(
                        T.abi_field_get(A_handle, 0, 1, ty=T.Ptr(T.void)),
                        ty=T.Ptr(T.float32),
                    )
                ), (
                    "ValueError",
                    [
                        "A",
                        " data pointer is NULL on argument #",
                        "0",
                        " when calling:\n  `",
                        "main(A: Tensor([1], float32))",
                        "`,\n  expected non-NULL data pointer",
                    ],
                )
                assert T.truncmod(
                    T.reinterpret(
                        T.reinterpret(
                            T.reinterpret(
                                T.abi_field_get(A_handle, 0, 1, ty=T.Ptr(T.void)),
                                ty=T.Ptr(T.float32),
                            ),
                            ty=T.Ptr(T.void),
                        ),
                        ty="uint64",
                    ),
                    T.uint64(64),
                ) == T.uint64(0), (
                    "ValueError",
                    [
                        "Misaligned Tensor data on argument #",
                        "0",
                        " when calling:\n  `",
                        "main(A: Tensor([1], float32))",
                        "`,\n  expected data alignment=",
                        "64",
                        " bytes",
                    ],
                )
                assert T.Cast(T.abi_field_get(main_A_shape, 0, 17, ty="int64"), "int32") == 1, (
                    "ValueError",
                    [
                        "Invalid ",
                        "A.shape[0]",
                        " on argument #",
                        "0",
                        " when calling:\n  `",
                        "main(A: Tensor([1], float32))",
                        "`,\n  expected ",
                        "1",
                    ],
                )
                assert T.uint64(0) == T.abi_field_get(A_handle, 0, 8, ty="uint64"), (
                    "ValueError",
                    [
                        "Invalid ",
                        "A.byte_offset",
                        " on argument #",
                        "0",
                        " when calling:\n  `",
                        "main(A: Tensor([1], float32))",
                        "`,\n  expected ",
                        "0",
                    ],
                )
                A = T.decl_tensor(
                    T.reinterpret(
                        T.abi_field_get(A_handle, 0, 1, ty=T.Ptr(T.void)),
                        ty=T.Ptr(T.float32),
                    ),
                    ty_args=[T.Tensor((1,), T.float32)],
                )
                T.assume_aligned(A, 64)
                set_device_status: T.let[T.int32] = T.call_extern(
                    "TVMBackendSetDevice", 2, dev_id, ty="int32"
                )
                if set_device_status != 0:
                    T.throw_last_error(ty="int32")
                with T.compute_scope("main_compute_"):
                    pass
                return 0

    after = tvm.tirx.transform.MakePackedAPI()(before)
    tvm.ir.assert_structural_equal(after, Expected)
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
        @T.function
        def main(A: T.handle("float32")):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            before.subroutine(A)

        # this test fails if it's made public
        @T.function(private=True)
        def subroutine(A_data: T.handle("float32")):
            T.func_attr({"target": T.target("llvm")})
            T.evaluate(A_data)

    @I.ir_module
    class Expected:
        @T.function
        def main(
            self_handle: T.Ptr(T.void),
            args: T.Ptr(T.void),
            num_args: T.int32,
            result: T.Ptr(T.void),
        ) -> T.int32:
            T.func_attr({"global_symbol": "__tvm_ffi_main"})
            T.func_attr({"calling_conv": 1, "target": T.target("llvm")})
            assert num_args == 1, (
                "TypeError",
                [
                    "Expected ",
                    "1",
                    " arguments",
                    " when calling:\n  `",
                    "main(A: T.Ptr(T.float32))",
                    "`",
                ],
            )
            assert not T.isnullptr(args), (
                "TypeError",
                [
                    "args pointer is NULL",
                    " when calling:\n  `",
                    "main(A: T.Ptr(T.float32))",
                    "`",
                ],
            )
            A_type_index: T.let[T.int32] = T.abi_field_get(args, 0, 13, ty="int32")
            assert (
                A_type_index == 0 or A_type_index == 4 or A_type_index == 7 or A_type_index >= 64
            ), (
                "TypeError",
                [
                    "Mismatched type on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(A: T.Ptr(T.float32))",
                    "`,\n  expected ",
                    "pointer",
                ],
            )
            A: T.let[T.Ptr(T.float32)] = T.reinterpret(
                T.if_then_else(
                    A_type_index == 70,
                    T.handle_add_byte_offset(T.abi_field_get(args, 0, 15, ty=T.Ptr(T.void)), 24),
                    T.abi_field_get(args, 0, 15, ty=T.Ptr(T.void)),
                ),
                ty=T.Ptr(T.float32),
            )
            with T.compute_scope("main_compute_"):
                Expected.subroutine(A)
            return 0

        @T.function(private=True)
        def subroutine(A_data: T.Ptr(T.float32)):
            T.func_attr({"target": T.target("llvm")})
            T.evaluate(A_data)

    after = tvm.tirx.transform.MakePackedAPI()(before)
    tvm.ir.assert_structural_equal(after, Expected)
    tvm.ir.assert_structural_equal(before["subroutine"], after["subroutine"])

    compute_scope = _find_compute_scope(after["main"])
    subroutine_call_op = compute_scope.body[0].value.op
    assert isinstance(subroutine_call_op, tvm.ir.GlobalVar), (
        f"The main function's CallNode should use the subroutine's GlobalVar as the operation, "
        f"but instead has an operation of type {subroutine_call_op}"
    )


def test_subroutine_call_to_externally_visible_subroutine():
    """Externally-visible subroutines should use the PackedFunc API

    Because the subroutine may be called directly by a user, it must
    use the PackedFunc API.  Its signature should be updated to the
    PackedFunc signature, and call sites should be updated to use
    `T.call_cpacked`.
    """

    @I.ir_module
    class before:
        @T.function
        def main(A: T.handle("float32")):
            T.func_attr({"global_symbol": "main", "target": T.target("llvm", host="llvm")})
            before.subroutine(A)

        @T.function
        def subroutine(A_data: T.handle("float32")):
            T.func_attr({"global_symbol": "subroutine", "target": T.target("llvm", host="llvm")})
            T.evaluate(A_data)

    @I.ir_module
    class Expected:
        @T.function
        def main(
            self_handle: T.Ptr(T.void),
            args: T.Ptr(T.void),
            num_args: T.int32,
            result: T.Ptr(T.void),
        ) -> T.int32:
            T.func_attr({"global_symbol": "__tvm_ffi_main"})
            T.func_attr({"calling_conv": 1, "target": T.target("llvm")})
            assert num_args == 1, (
                "TypeError",
                [
                    "Expected ",
                    "1",
                    " arguments",
                    " when calling:\n  `",
                    "main(A: T.Ptr(T.float32))",
                    "`",
                ],
            )
            assert not T.isnullptr(args), (
                "TypeError",
                [
                    "args pointer is NULL",
                    " when calling:\n  `",
                    "main(A: T.Ptr(T.float32))",
                    "`",
                ],
            )
            A_type_index: T.let[T.int32] = T.abi_field_get(args, 0, 13, ty="int32")
            assert (
                A_type_index == 0 or A_type_index == 4 or A_type_index == 7 or A_type_index >= 64
            ), (
                "TypeError",
                [
                    "Mismatched type on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(A: T.Ptr(T.float32))",
                    "`,\n  expected ",
                    "pointer",
                ],
            )
            A: T.let[T.Ptr(T.float32)] = T.reinterpret(
                T.if_then_else(
                    A_type_index == 70,
                    T.handle_add_byte_offset(T.abi_field_get(args, 0, 15, ty=T.Ptr(T.void)), 24),
                    T.abi_field_get(args, 0, 15, ty=T.Ptr(T.void)),
                ),
                ty=T.Ptr(T.float32),
            )
            with T.compute_scope("main_compute_"):
                T.call_cpacked(
                    "subroutine",
                    A,
                    T.reinterpret(T.uint64(0), ty=T.Ptr(T.void)),
                    ty=T.Tuple(),
                )
            return 0

        @T.function
        def subroutine(
            self_handle: T.Ptr(T.void),
            args: T.Ptr(T.void),
            num_args: T.int32,
            result: T.Ptr(T.void),
        ) -> T.int32:
            T.func_attr({"global_symbol": "__tvm_ffi_subroutine"})
            T.func_attr({"calling_conv": 1, "target": T.target("llvm")})
            assert num_args == 1, (
                "TypeError",
                [
                    "Expected ",
                    "1",
                    " arguments",
                    " when calling:\n  `",
                    "subroutine(A_data: T.Ptr(T.float32))",
                    "`",
                ],
            )
            assert not T.isnullptr(args), (
                "TypeError",
                [
                    "args pointer is NULL",
                    " when calling:\n  `",
                    "subroutine(A_data: T.Ptr(T.float32))",
                    "`",
                ],
            )
            A_data_type_index: T.let[T.int32] = T.abi_field_get(args, 0, 13, ty="int32")
            assert (
                A_data_type_index == 0
                or A_data_type_index == 4
                or A_data_type_index == 7
                or A_data_type_index >= 64
            ), (
                "TypeError",
                [
                    "Mismatched type on argument #",
                    "0",
                    " when calling:\n  `",
                    "subroutine(A_data: T.Ptr(T.float32))",
                    "`,\n  expected ",
                    "pointer",
                ],
            )
            A_data: T.let[T.Ptr(T.float32)] = T.reinterpret(
                T.if_then_else(
                    A_data_type_index == 70,
                    T.handle_add_byte_offset(T.abi_field_get(args, 0, 15, ty=T.Ptr(T.void)), 24),
                    T.abi_field_get(args, 0, 15, ty=T.Ptr(T.void)),
                ),
                ty=T.Ptr(T.float32),
            )
            with T.compute_scope("subroutine_compute_"):
                T.evaluate(A_data)
            return 0

    after = tvm.tirx.transform.MakePackedAPI()(before)
    tvm.ir.assert_structural_equal(after, Expected)

    main_compute_scope = _find_compute_scope(after["main"])
    assert main_compute_scope is not None
    subroutine_compute_scope = _find_compute_scope(after["subroutine"])
    assert subroutine_compute_scope is not None

    subroutine_call_op = main_compute_scope.body[0].value.op
    assert (
        isinstance(subroutine_call_op, tvm.ir.Op) and subroutine_call_op.name == "tirx.call_cpacked"
    ), (
        f"The main function's CallNode should be lowered to the builtin 'tirx.call_cpacked', "
        f"but instead has an operation of type {subroutine_call_op}"
    )


def test_zero_arg_function():
    """Zero-arg function emits num_args check but no null-pointer check."""

    @I.ir_module
    class Before:
        @T.function
        def func_without_arg() -> T.int64:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            return T.int64(42)

    @I.ir_module
    class Expected:
        @T.function
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
            with T.compute_scope("func_without_arg_compute_"):
                T.abi_field_set(result, 0, 13, 1)
                T.abi_field_set(result, 0, 14, 0)
                T.abi_field_set(result, 0, 15, T.Cast(T.int64(42), "int64"))
                return 0
            return 0

    After = tvm.tirx.transform.MakePackedAPI()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_pointer_return():
    """Pointer returns are stored as an opaque pointer in TVMFFIAny."""

    @I.ir_module
    class Before:
        @T.function
        def main(arg: T.handle) -> T.handle:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            return arg

    @I.ir_module
    class Expected:
        @T.function
        def main(
            self_handle: T.Ptr(T.void),
            args: T.Ptr(T.void),
            num_args: T.int32,
            result: T.Ptr(T.void),
        ) -> T.int32:
            T.func_attr({"global_symbol": "__tvm_ffi_main"})
            T.func_attr({"calling_conv": 1, "target": T.target("llvm")})
            assert num_args == 1, (
                "TypeError",
                [
                    "Expected ",
                    "1",
                    " arguments",
                    " when calling:\n  `",
                    "main(arg: T.Ptr(T.void))",
                    "`",
                ],
            )
            assert not T.isnullptr(args), (
                "TypeError",
                [
                    "args pointer is NULL",
                    " when calling:\n  `",
                    "main(arg: T.Ptr(T.void))",
                    "`",
                ],
            )
            arg_type_index: T.let[T.int32] = T.abi_field_get(args, 0, 13, ty="int32")
            assert (
                arg_type_index == 0
                or arg_type_index == 4
                or arg_type_index == 7
                or arg_type_index >= 64
            ), (
                "TypeError",
                [
                    "Mismatched type on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(arg: T.Ptr(T.void))",
                    "`,\n  expected ",
                    "pointer",
                ],
            )
            arg: T.let[T.Ptr(T.void)] = T.reinterpret(
                T.if_then_else(
                    arg_type_index == 70,
                    T.handle_add_byte_offset(T.abi_field_get(args, 0, 15, ty=T.Ptr(T.void)), 24),
                    T.abi_field_get(args, 0, 15, ty=T.Ptr(T.void)),
                ),
                ty=T.Ptr(T.void),
            )
            with T.compute_scope("main_compute_"):
                T.abi_field_set(result, 0, 13, 4)
                T.abi_field_set(result, 0, 14, 0)
                T.abi_field_set(result, 0, 15, arg)
                return 0
            return 0

    after = tvm.tirx.transform.MakePackedAPI()(Before)
    tvm.ir.assert_structural_equal(after, Expected)
    after = after["main"]
    return_type_indices = []

    def collect(node):
        if not isinstance(node, tvm.ir.Call) or node.op.name != "tirx.abi_field_set":
            return
        field = node.args[2]
        if isinstance(field, tvm.tirx.IntImm) and int(field) == 13:
            return_type_indices.append(int(node.args[3]))

    tvm_ffi.structural_walk(after.body, collect)
    assert 4 in return_type_indices  # ffi::TypeIndex::kTVMFFIOpaquePtr


def test_return_from_parallel_scope_is_rejected():
    """A parallel loop cannot return from its enclosing function."""

    @I.ir_module
    class Before:
        @T.function
        def main() -> T.int32:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            for i in T.parallel(2):
                return i

    with pytest.raises(tvm.error.InternalError, match="Return cannot be used in parallel scope"):
        tvm.tirx.transform.MakePackedAPI()(Before)


def test_int_parameter():
    """Int parameter emits type check accepting int or bool."""

    @I.ir_module
    class Before:
        @T.function
        def main(arg: T.int32) -> T.int32:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            if arg > 0:
                return 10
            else:
                return 20

    @I.ir_module
    class Expected:
        @T.function
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
            arg_type_index: T.let[T.int32] = T.abi_field_get(args, 0, 13, ty="int32")
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
            arg: T.let[T.int32] = T.Cast(T.abi_field_get(args, 0, 15, ty="int64"), "int32")
            with T.compute_scope("main_compute_"):
                if arg > 0:
                    T.abi_field_set(result, 0, 13, 1)
                    T.abi_field_set(result, 0, 14, 0)
                    T.abi_field_set(result, 0, 15, T.Cast(10, "int64"))
                    return 0
                else:
                    T.abi_field_set(result, 0, 13, 1)
                    T.abi_field_set(result, 0, 14, 0)
                    T.abi_field_set(result, 0, 15, T.Cast(20, "int64"))
                    return 0
            return 0

    After = tvm.tirx.transform.MakePackedAPI()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_bool_parameter():
    """Bool parameter emits type check accepting bool or int."""

    @I.ir_module
    class Before:
        @T.function
        def main(arg: T.bool) -> T.int32:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            if arg:
                return 10
            else:
                return 20

    @I.ir_module
    class Expected:
        @T.function
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
            arg_type_index: T.let[T.int32] = T.abi_field_get(args, 0, 13, ty="int32")
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
            arg: T.let[T.bool] = T.Cast(T.abi_field_get(args, 0, 15, ty="int64"), "bool")
            with T.compute_scope("main_compute_"):
                if arg:
                    T.abi_field_set(result, 0, 13, 1)
                    T.abi_field_set(result, 0, 14, 0)
                    T.abi_field_set(result, 0, 15, T.Cast(10, "int64"))
                    return 0
                else:
                    T.abi_field_set(result, 0, 13, 1)
                    T.abi_field_set(result, 0, 14, 0)
                    T.abi_field_set(result, 0, 15, T.Cast(20, "int64"))
                    return 0
            return 0

    After = tvm.tirx.transform.MakePackedAPI()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_float_parameter():
    """Float parameter emits type check accepting float, int, or bool."""

    @I.ir_module
    class Before:
        @T.function
        def main(arg: T.float32) -> T.int32:
            T.func_attr({"target": T.target("llvm", host="llvm")})
            if arg > T.float32(0):
                return 10
            else:
                return 20

    @I.ir_module
    class Expected:
        @T.function
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
            arg_type_index: T.let[T.int32] = T.abi_field_get(args, 0, 13, ty="int32")
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
            arg: T.let[T.float32] = T.Select(
                arg_type_index == 3,
                T.Cast(T.abi_field_get(args, 0, 15, ty="float64"), "float32"),
                T.Cast(T.abi_field_get(args, 0, 15, ty="int64"), "float32"),
            )
            with T.compute_scope("main_compute_"):
                if arg > T.float32(0.0):
                    T.abi_field_set(result, 0, 13, 1)
                    T.abi_field_set(result, 0, 14, 0)
                    T.abi_field_set(result, 0, 15, T.Cast(10, "int64"))
                    return 0
                else:
                    T.abi_field_set(result, 0, 13, 1)
                    T.abi_field_set(result, 0, 14, 0)
                    T.abi_field_set(result, 0, 15, T.Cast(20, "int64"))
                    return 0
            return 0

    After = tvm.tirx.transform.MakePackedAPI()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_forward_reference_symbolic_variable():
    """MakePackedAPI succeeds when a symbolic variable is used before it is defined.

    When tensor A has shape (batch_size+1,) and tensor B has shape (batch_size,),
    batch_size is referenced (in A's shape check) before it is defined (from B's
    shape). The three-sequence separation (init_nest, asserts, decl_tensors)
    ensures all variable definitions precede all assertions.
    """

    signature = "main(A: Tensor([batch_size + T.int64(1)], int32), B: Tensor([batch_size], int32))"

    batch_size = T.dynamic("batch_size", "int64")

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((batch_size + 1,), "int32"), B: T.Tensor((batch_size,), "int32")):
            T.func_attr({"target": T.target("llvm", host="llvm")})

            for i in range(batch_size):
                B[i] = A[i] + A[i + 1]

    # Should not raise "variable batch_size has been used before definition"
    @I.ir_module
    class Expected:
        @T.function
        def main(
            self_handle: T.Ptr(T.void),
            args: T.Ptr(T.void),
            num_args: T.int32,
            result: T.Ptr(T.void),
        ) -> T.int32:
            T.func_attr({"global_symbol": "__tvm_ffi_main"})
            T.func_attr({"calling_conv": 1, "target": T.target("llvm")})
            assert num_args == 2, (
                "TypeError",
                [
                    "Expected ",
                    "2",
                    " arguments",
                    " when calling:\n  `",
                    signature,
                    "`",
                ],
            )
            assert not T.isnullptr(args), (
                "TypeError",
                [
                    "args pointer is NULL",
                    " when calling:\n  `",
                    signature,
                    "`",
                ],
            )
            A_type_index: T.let[T.int32] = T.abi_field_get(args, 0, 13, ty="int32")
            assert (
                A_type_index == 0 or A_type_index == 4 or A_type_index == 7 or A_type_index >= 64
            ), (
                "TypeError",
                [
                    "Mismatched type on argument #",
                    "0",
                    " when calling:\n  `",
                    signature,
                    "`,\n  expected ",
                    "Tensor",
                ],
            )
            A_handle: T.let[T.Ptr(T.void)] = T.if_then_else(
                A_type_index == 70,
                T.handle_add_byte_offset(T.abi_field_get(args, 0, 15, ty=T.Ptr(T.void)), 24),
                T.abi_field_get(args, 0, 15, ty=T.Ptr(T.void)),
            )
            B_type_index: T.let[T.int32] = T.abi_field_get(args, 1, 13, ty="int32")
            assert (
                B_type_index == 0 or B_type_index == 4 or B_type_index == 7 or B_type_index >= 64
            ), (
                "TypeError",
                [
                    "Mismatched type on argument #",
                    "1",
                    " when calling:\n  `",
                    signature,
                    "`,\n  expected ",
                    "Tensor",
                ],
            )
            B_handle: T.let[T.Ptr(T.void)] = T.if_then_else(
                B_type_index == 70,
                T.handle_add_byte_offset(T.abi_field_get(args, 1, 15, ty=T.Ptr(T.void)), 24),
                T.abi_field_get(args, 1, 15, ty=T.Ptr(T.void)),
            )
            assert not T.isnullptr(A_handle), (
                "TypeError",
                [
                    "Mismatched type on argument #",
                    "0",
                    " when calling:\n  `",
                    signature,
                    "`,\n  expected ",
                    "Tensor",
                ],
            )
            assert 1 == T.abi_field_get(A_handle, 0, 4, ty="int32"), (
                "ValueError",
                [
                    "Mismatched ",
                    "A",
                    ".ndim on argument #",
                    "0",
                    " when calling:\n  `",
                    signature,
                    "`,\n  expected ",
                    "1",
                ],
            )
            assert (
                T.abi_field_get(A_handle, 0, 5, ty="uint8") == T.uint8(0)
                and T.abi_field_get(A_handle, 0, 6, ty="uint8") == T.uint8(32)
                and T.abi_field_get(A_handle, 0, 7, ty="uint16") == T.uint16(1)
            ), (
                "TypeError",
                [
                    "Mismatched ",
                    "A",
                    ".dtype on argument #",
                    "0",
                    " when calling:\n  `",
                    signature,
                    "`,\n  expected ",
                    "int32",
                ],
            )
            main_A_shape: T.let[T.Ptr(T.PrimType("int64"))] = T.abi_field_get(
                A_handle, 0, 2, ty=T.Ptr(T.PrimType("int64"))
            )
            main_A_strides: T.let[T.Ptr(T.PrimType("int64"))] = T.abi_field_get(
                A_handle, 0, 3, ty=T.Ptr(T.PrimType("int64"))
            )
            assert T.abi_field_get(A_handle, 0, 10, ty="int32") == 1, (
                "ValueError",
                [
                    "Mismatched ",
                    "A",
                    ".device_type on argument #",
                    "0",
                    " when calling:\n  `",
                    signature,
                    "`,\n  expected ",
                    "cpu",
                ],
            )
            dev_id: T.let[T.int32] = T.abi_field_get(A_handle, 0, 9, ty="int32")
            assert not T.isnullptr(B_handle), (
                "TypeError",
                [
                    "Mismatched type on argument #",
                    "1",
                    " when calling:\n  `",
                    signature,
                    "`,\n  expected ",
                    "Tensor",
                ],
            )
            assert 1 == T.abi_field_get(B_handle, 0, 4, ty="int32"), (
                "ValueError",
                [
                    "Mismatched ",
                    "B",
                    ".ndim on argument #",
                    "1",
                    " when calling:\n  `",
                    signature,
                    "`,\n  expected ",
                    "1",
                ],
            )
            assert (
                T.abi_field_get(B_handle, 0, 5, ty="uint8") == T.uint8(0)
                and T.abi_field_get(B_handle, 0, 6, ty="uint8") == T.uint8(32)
                and T.abi_field_get(B_handle, 0, 7, ty="uint16") == T.uint16(1)
            ), (
                "TypeError",
                [
                    "Mismatched ",
                    "B",
                    ".dtype on argument #",
                    "1",
                    " when calling:\n  `",
                    signature,
                    "`,\n  expected ",
                    "int32",
                ],
            )
            main_B_shape: T.let[T.Ptr(T.PrimType("int64"))] = T.abi_field_get(
                B_handle, 0, 2, ty=T.Ptr(T.PrimType("int64"))
            )
            batch_size: T.let[T.int64] = T.abi_field_get(main_B_shape, 0, 17, ty="int64")
            main_B_strides: T.let[T.Ptr(T.PrimType("int64"))] = T.abi_field_get(
                B_handle, 0, 3, ty=T.Ptr(T.PrimType("int64"))
            )
            assert T.abi_field_get(B_handle, 0, 10, ty="int32") == 1, (
                "ValueError",
                [
                    "Mismatched ",
                    "B",
                    ".device_type on argument #",
                    "1",
                    " when calling:\n  `",
                    signature,
                    "`,\n  expected ",
                    "cpu",
                ],
            )
            with T.device_context(1, dev_id):
                if not T.isnullptr(main_A_strides):
                    assert batch_size + T.int64(1) == T.int64(1) or T.int64(1) == T.abi_field_get(
                        main_A_strides, 0, 17, ty="int64"
                    ), (
                        "ValueError",
                        [
                            "Mismatched ",
                            "A",
                            ".strides on argument #",
                            "0",
                            " when calling:\n  `",
                            signature,
                            "`,\n  expected to be compact array",
                        ],
                    )
                assert batch_size + T.int64(1) == T.int64(0) or not T.isnullptr(
                    T.reinterpret(
                        T.abi_field_get(A_handle, 0, 1, ty=T.Ptr(T.void)),
                        ty=T.Ptr(T.PrimType("int32")),
                    )
                ), (
                    "ValueError",
                    [
                        "A",
                        " data pointer is NULL on argument #",
                        "0",
                        " when calling:\n  `",
                        signature,
                        "`,\n  expected non-NULL data pointer",
                    ],
                )
                assert batch_size + T.int64(1) == T.int64(0) or T.truncmod(
                    T.reinterpret(
                        T.reinterpret(
                            T.reinterpret(
                                T.abi_field_get(A_handle, 0, 1, ty=T.Ptr(T.void)),
                                ty=T.Ptr(T.PrimType("int32")),
                            ),
                            ty=T.Ptr(T.void),
                        ),
                        ty="uint64",
                    ),
                    T.uint64(64),
                ) == T.uint64(0), (
                    "ValueError",
                    [
                        "Misaligned Tensor data on argument #",
                        "0",
                        " when calling:\n  `",
                        signature,
                        "`,\n  expected data alignment=",
                        "64",
                        " bytes",
                    ],
                )
                if not T.isnullptr(main_B_strides):
                    assert batch_size == T.int64(1) or T.int64(1) == T.abi_field_get(
                        main_B_strides, 0, 17, ty="int64"
                    ), (
                        "ValueError",
                        [
                            "Mismatched ",
                            "B",
                            ".strides on argument #",
                            "1",
                            " when calling:\n  `",
                            signature,
                            "`,\n  expected to be compact array",
                        ],
                    )
                assert dev_id == T.abi_field_get(B_handle, 0, 9, ty="int32"), (
                    "ValueError",
                    [
                        "Mismatched ",
                        "B.device_id",
                        " on argument #",
                        "1",
                        " when calling:\n  `",
                        signature,
                        "`,\n  expected to match ",
                        "A.device_id",
                    ],
                )
                assert batch_size == T.int64(0) or not T.isnullptr(
                    T.reinterpret(
                        T.abi_field_get(B_handle, 0, 1, ty=T.Ptr(T.void)),
                        ty=T.Ptr(T.PrimType("int32")),
                    )
                ), (
                    "ValueError",
                    [
                        "B",
                        " data pointer is NULL on argument #",
                        "1",
                        " when calling:\n  `",
                        signature,
                        "`,\n  expected non-NULL data pointer",
                    ],
                )
                assert batch_size == T.int64(0) or T.truncmod(
                    T.reinterpret(
                        T.reinterpret(
                            T.reinterpret(
                                T.abi_field_get(B_handle, 0, 1, ty=T.Ptr(T.void)),
                                ty=T.Ptr(T.PrimType("int32")),
                            ),
                            ty=T.Ptr(T.void),
                        ),
                        ty="uint64",
                    ),
                    T.uint64(64),
                ) == T.uint64(0), (
                    "ValueError",
                    [
                        "Misaligned Tensor data on argument #",
                        "1",
                        " when calling:\n  `",
                        signature,
                        "`,\n  expected data alignment=",
                        "64",
                        " bytes",
                    ],
                )
                assert batch_size + T.int64(1) == T.abi_field_get(
                    main_A_shape, 0, 17, ty="int64"
                ), (
                    "ValueError",
                    [
                        "Invalid ",
                        "A.shape[0]",
                        " on argument #",
                        "0",
                        " when calling:\n  `",
                        signature,
                        "`,\n  expected ",
                        "B.shape[0] + 1",
                    ],
                )
                assert T.uint64(0) == T.abi_field_get(A_handle, 0, 8, ty="uint64"), (
                    "ValueError",
                    [
                        "Invalid ",
                        "A.byte_offset",
                        " on argument #",
                        "0",
                        " when calling:\n  `",
                        signature,
                        "`,\n  expected ",
                        "0",
                    ],
                )
                assert T.uint64(0) == T.abi_field_get(B_handle, 0, 8, ty="uint64"), (
                    "ValueError",
                    [
                        "Invalid ",
                        "B.byte_offset",
                        " on argument #",
                        "1",
                        " when calling:\n  `",
                        signature,
                        "`,\n  expected ",
                        "0",
                    ],
                )
                A = T.decl_tensor(
                    T.reinterpret(
                        T.abi_field_get(A_handle, 0, 1, ty=T.Ptr(T.void)),
                        ty=T.Ptr(T.PrimType("int32")),
                    ),
                    ty_args=[T.Tensor((batch_size + T.int64(1),), T.int32)],
                )
                if batch_size != T.int64(-1):
                    T.assume_aligned(A, 64)
                B = T.decl_tensor(
                    T.reinterpret(
                        T.abi_field_get(B_handle, 0, 1, ty=T.Ptr(T.void)),
                        ty=T.Ptr(T.PrimType("int32")),
                    ),
                    ty_args=[T.Tensor((batch_size,), T.int32)],
                )
                if batch_size != T.int64(0):
                    T.assume_aligned(B, 64)
                with T.compute_scope("main_compute_"):
                    for i in range(T.int64(0), batch_size):
                        B[i] = A[i] + A[i + T.int64(1)]
                return 0

    After = tvm.tirx.transform.MakePackedAPI()(Before)
    tvm.ir.assert_structural_equal(After, Expected)
    assert len(After["main"].params) == 4


def test_tensor_alignment_attached_to_tensor_var():
    """Packed ABI alignment metadata remains keyed by the logical TensorVar."""

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((16,), "float32", align=64)):
            T.func_attr({"global_symbol": "main", "target": T.target("llvm", host="llvm")})
            T.evaluate(A[0])

    @I.ir_module
    class Expected:
        @T.function
        def main(
            self_handle: T.Ptr(T.void),
            args: T.Ptr(T.void),
            num_args: T.int32,
            result: T.Ptr(T.void),
        ) -> T.int32:
            T.func_attr({"global_symbol": "__tvm_ffi_main"})
            T.func_attr({"calling_conv": 1, "target": T.target("llvm")})
            assert num_args == 1, (
                "TypeError",
                [
                    "Expected ",
                    "1",
                    " arguments",
                    " when calling:\n  `",
                    "main(A: Tensor([16], float32))",
                    "`",
                ],
            )
            assert not T.isnullptr(args), (
                "TypeError",
                [
                    "args pointer is NULL",
                    " when calling:\n  `",
                    "main(A: Tensor([16], float32))",
                    "`",
                ],
            )
            A_type_index: T.let[T.int32] = T.abi_field_get(args, 0, 13, ty="int32")
            assert (
                A_type_index == 0 or A_type_index == 4 or A_type_index == 7 or A_type_index >= 64
            ), (
                "TypeError",
                [
                    "Mismatched type on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(A: Tensor([16], float32))",
                    "`,\n  expected ",
                    "Tensor",
                ],
            )
            A_handle: T.let[T.Ptr(T.void)] = T.if_then_else(
                A_type_index == 70,
                T.handle_add_byte_offset(T.abi_field_get(args, 0, 15, ty=T.Ptr(T.void)), 24),
                T.abi_field_get(args, 0, 15, ty=T.Ptr(T.void)),
            )
            assert not T.isnullptr(A_handle), (
                "TypeError",
                [
                    "Mismatched type on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(A: Tensor([16], float32))",
                    "`,\n  expected ",
                    "Tensor",
                ],
            )
            assert 1 == T.abi_field_get(A_handle, 0, 4, ty="int32"), (
                "ValueError",
                [
                    "Mismatched ",
                    "A",
                    ".ndim on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(A: Tensor([16], float32))",
                    "`,\n  expected ",
                    "1",
                ],
            )
            assert (
                T.abi_field_get(A_handle, 0, 5, ty="uint8") == T.uint8(2)
                and T.abi_field_get(A_handle, 0, 6, ty="uint8") == T.uint8(32)
                and T.abi_field_get(A_handle, 0, 7, ty="uint16") == T.uint16(1)
            ), (
                "TypeError",
                [
                    "Mismatched ",
                    "A",
                    ".dtype on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(A: Tensor([16], float32))",
                    "`,\n  expected ",
                    "float32",
                ],
            )
            main_A_shape: T.let[T.Ptr(T.PrimType("int64"))] = T.abi_field_get(
                A_handle, 0, 2, ty=T.Ptr(T.PrimType("int64"))
            )
            main_A_strides: T.let[T.Ptr(T.PrimType("int64"))] = T.abi_field_get(
                A_handle, 0, 3, ty=T.Ptr(T.PrimType("int64"))
            )
            assert T.abi_field_get(A_handle, 0, 10, ty="int32") == 1, (
                "ValueError",
                [
                    "Mismatched ",
                    "A",
                    ".device_type on argument #",
                    "0",
                    " when calling:\n  `",
                    "main(A: Tensor([16], float32))",
                    "`,\n  expected ",
                    "cpu",
                ],
            )
            dev_id: T.let[T.int32] = T.abi_field_get(A_handle, 0, 9, ty="int32")
            with T.device_context(1, dev_id):
                if not T.isnullptr(main_A_strides):
                    assert 1 == T.Cast(
                        T.abi_field_get(main_A_strides, 0, 17, ty="int64"), "int32"
                    ), (
                        "ValueError",
                        [
                            "Mismatched ",
                            "A",
                            ".strides on argument #",
                            "0",
                            " when calling:\n  `",
                            "main(A: Tensor([16], float32))",
                            "`,\n  expected to be compact array",
                        ],
                    )
                assert not T.isnullptr(
                    T.reinterpret(
                        T.abi_field_get(A_handle, 0, 1, ty=T.Ptr(T.void)),
                        ty=T.Ptr(T.float32),
                    )
                ), (
                    "ValueError",
                    [
                        "A",
                        " data pointer is NULL on argument #",
                        "0",
                        " when calling:\n  `",
                        "main(A: Tensor([16], float32))",
                        "`,\n  expected non-NULL data pointer",
                    ],
                )
                assert T.truncmod(
                    T.reinterpret(
                        T.reinterpret(
                            T.reinterpret(
                                T.abi_field_get(A_handle, 0, 1, ty=T.Ptr(T.void)),
                                ty=T.Ptr(T.float32),
                            ),
                            ty=T.Ptr(T.void),
                        ),
                        ty="uint64",
                    ),
                    T.uint64(64),
                ) == T.uint64(0), (
                    "ValueError",
                    [
                        "Misaligned Tensor data on argument #",
                        "0",
                        " when calling:\n  `",
                        "main(A: Tensor([16], float32))",
                        "`,\n  expected data alignment=",
                        "64",
                        " bytes",
                    ],
                )
                assert T.Cast(T.abi_field_get(main_A_shape, 0, 17, ty="int64"), "int32") == 16, (
                    "ValueError",
                    [
                        "Invalid ",
                        "A.shape[0]",
                        " on argument #",
                        "0",
                        " when calling:\n  `",
                        "main(A: Tensor([16], float32))",
                        "`,\n  expected ",
                        "16",
                    ],
                )
                assert T.uint64(0) == T.abi_field_get(A_handle, 0, 8, ty="uint64"), (
                    "ValueError",
                    [
                        "Invalid ",
                        "A.byte_offset",
                        " on argument #",
                        "0",
                        " when calling:\n  `",
                        "main(A: Tensor([16], float32))",
                        "`,\n  expected ",
                        "0",
                    ],
                )
                A = T.decl_tensor(
                    T.reinterpret(
                        T.abi_field_get(A_handle, 0, 1, ty=T.Ptr(T.void)),
                        ty=T.Ptr(T.float32),
                    ),
                    ty_args=[T.Tensor((16,), T.float32)],
                )
                T.assume_aligned(A, 64)
                with T.compute_scope("main_compute_"):
                    T.evaluate(A[0])
                return 0

    after = tvm.tirx.transform.MakePackedAPI()(Before)
    tvm.ir.assert_structural_equal(after, Expected)
    after = after["main"]
    alignment_nodes = []
    declared_tensors = []

    def collect(node):
        if isinstance(node, tvm.ir.Call) and node.op == tvm.ir.Op.get("tirx.assume_aligned"):
            alignment_nodes.append(node.args[0])
        if _is_tensor_binding(node, "tirx.decl_tensor"):
            declared_tensors.append(node.var)

    tvm_ffi.structural_walk(after.body, collect)
    assert len(alignment_nodes) == 1
    assert any(alignment_nodes[0].same_as(tensor) for tensor in declared_tensors)


if __name__ == "__main__":
    tvm.testing.main()
