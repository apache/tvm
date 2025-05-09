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
import numpy as np
import tvm
import tvm.testing
from tvm import te, tir
from tvm.script import tir as T, ir as I


def _find_assignment(stmt, var_name):
    while not isinstance(stmt, tvm.tir.LetStmt):
        stmt = stmt.body

    if stmt.var.name != var_name:
        return _find_assignment(stmt.body, var_name)

    return stmt


def _find_next(stmt, type):
    search_stack = [stmt]

    while search_stack:
        stmt = search_stack.pop()
        if isinstance(stmt, type):
            return stmt
        elif isinstance(stmt, tvm.tir.SeqStmt):
            search_stack.extend(reversed(stmt))
        else:
            search_stack.append(stmt.body)

    return None


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
        f"The main function's CallNode should use the subroutine's GLobalVar as the operation, "
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


def test_function_call_with_wrong_argument_count():
    """Argument counts must be checked before accessing the type codes"""

    @T.prim_func
    def func(
        A: T.Buffer([16, 16], "int32"),
        B: T.Buffer([16, 16], "int32"),
        C: T.Buffer([16, 16], "int32"),
        D: T.Buffer([16, 16], "int32"),
    ):
        pass

    built = tvm.compile(func, target="llvm")

    with pytest.raises(tvm.TVMError):
        built()


def test_function_call_with_wrong_type_code():
    """Type codes must be checked before accessing the arguments"""

    @T.prim_func
    def func(A: T.Buffer([16, 16], "int32")):
        pass

    built = tvm.compile(func, target="llvm")

    with pytest.raises(tvm.TVMError):
        built(0)


def test_function_call_with_null_data_pointer():
    """The data pointer must be checked before accessing the array"""

    @T.prim_func
    def func(A: T.Buffer([16, 16], "int32"), B: T.Buffer([16, 16], "int32")):
        for i, j in T.grid(16, 16):
            B[i, j] = A[i, j]

    built = tvm.compile(func, target="llvm")

    A = tvm.nd.array(np.zeros([16], dtype="int32"))
    B = tvm.nd.empty([16, 16], "int32", tvm.cpu())

    with pytest.raises(tvm.TVMError):
        built(A, B)


def test_function_call_with_wrong_dimensionality():
    """The dimensionality must be checked before validating the shape"""

    @T.prim_func
    def func(A: T.Buffer([16, 16], "int32"), B: T.Buffer([16, 16], "int32")):
        for i, j in T.grid(16, 16):
            B[i, j] = A[i, j]

    built = tvm.compile(func, target="llvm")

    A = tvm.nd.array(np.zeros([16], dtype="int32"))
    B = tvm.nd.empty([16], "int32", tvm.cpu())

    with pytest.raises(tvm.TVMError):
        built(A, B)


def test_zero_arg_function():
    """Only check non-null args when num_args>0"""

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
            self: T.handle,
            args: T.handle,
            num_args: T.int32,
            result: T.handle("void"),
        ) -> T.int32:
            T.func_attr(
                {
                    "calling_conv": 1,
                    "target": T.target("llvm"),
                }
            )
            assert num_args == 0, "func_without_arg: num_args should be 0"
            with T.attr(0, "compute_scope", "func_without_arg_compute_"):
                T.tvm_struct_set(result, 0, 13, 1)
                T.tvm_struct_set(result, 0, 14, T.Cast("int64", T.int64(42)))
                return 0
            return 0

    After = tvm.tir.transform.MakePackedAPI()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_int_parameter():
    """Boolean may be passed to functions accepting int

    A PackedFunc produced by compiling an IRModule should support the
    same type conversions as the C++ implementation.  When a function
    accepts an integer argument, the caller may call it with a boolean
    value.

    This also provides backwards compatibility for functions that were
    defined as accepting an integer, but are called with a boolean
    argument.  Prior to PackedFunc interface supporting boolean
    arguments directly, the argument would be converted from boolean
    to integer to be stored in a TVMValue.  After adding support for
    boolean arguments, this usage should not cause an error.

    """

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
            self: T.handle,
            args: T.handle,
            num_args: T.int32,
            result: T.handle("void"),
        ) -> T.int32:
            T.func_attr(
                {
                    "calling_conv": 1,
                    "target": T.target("llvm"),
                }
            )
            assert num_args == 1, "main: num_args should be 1"
            assert not T.isnullptr(args), "main: args pointer is NULL"
            arg_type_index: T.int32 = T.tvm_struct_get(args, 0, 13, "int32")
            assert arg_type_index == 1 or arg_type_index == 2, "main: Expect arg[0] to be int"
            arg: T.int32 = T.Cast("int32", T.tvm_struct_get(args, 0, 14, "int64"))
            with T.attr(0, "compute_scope", "main_compute_"):
                if arg > 0:
                    T.tvm_struct_set(result, 0, 13, 1)
                    T.tvm_struct_set(result, 0, 14, T.Cast("int64", 10))
                    return 0
                else:
                    T.tvm_struct_set(result, 0, 13, 1)
                    T.tvm_struct_set(result, 0, 14, T.Cast("int64", 20))
                    return 0
            return 0

    After = tvm.tir.transform.MakePackedAPI()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_bool_parameter():
    """An integer may be passed to a function acccepting Boolean

    A PackedFunc produced by compiling an IRModule should support the
    same type conversions as the C++ implementation.  When a function
    accepts a boolean argument, the caller may call it with an integer
    value.

    """

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
            self: T.handle,
            args: T.handle,
            num_args: T.int32,
            result: T.handle("void"),
        ) -> T.int32:
            T.func_attr(
                {
                    "calling_conv": 1,
                    "target": T.target("llvm"),
                }
            )
            assert num_args == 1, "main: num_args should be 1"
            assert not T.isnullptr(args), "main: args pointer is NULL"
            arg_type_index: T.int32 = T.tvm_struct_get(args, 0, 13, "int32")
            assert arg_type_index == 2 or arg_type_index == 1, "main: Expect arg[0] to be boolean"
            arg: T.bool = T.Cast("bool", T.tvm_struct_get(args, 0, 14, "int64"))
            with T.attr(0, "compute_scope", "main_compute_"):
                if arg:
                    T.tvm_struct_set(result, 0, 13, 1)
                    T.tvm_struct_set(result, 0, 14, T.Cast("int64", 10))
                    return 0
                else:
                    T.tvm_struct_set(result, 0, 13, 1)
                    T.tvm_struct_set(result, 0, 14, T.Cast("int64", 20))
                    return 0
            return 0

    After = tvm.tir.transform.MakePackedAPI()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


if __name__ == "__main__":
    tvm.testing.main()
