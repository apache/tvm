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

import tvm
import tvm.testing
from tvm import te, tir
from tvm.script import tir as T, ir as I
from tvm.driver.build_module import schedule_to_module


def test_makeapi():
    """Not yet working, mock design"""
    n = te.size_var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    s = te.create_schedule(C.op)

    mod = schedule_to_module(s, [n, A, B, C])
    mod = tvm.tir.transform.StorageFlatten(64)(mod)
    mod = tvm.tir.transform.Apply(
        lambda f: f.with_attr(
            {
                "target": tvm.target.Target("llvm", host="llvm"),
                "global_symbol": "main",
            }
        )
    )(mod)

    before = mod
    after = tvm.tir.transform.MakePackedAPI()(before)
    f = after["main"]
    assert len(f.params) == 6


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


def test_variable_passed_from_args():
    ib = tvm.tir.ir_builder.create()

    input_buffer = tvm.tir.decl_buffer(name="input_buffer", shape=[1])
    not_device_context = tvm.tir.Var("not_device_context", dtype="handle")

    ib.emit(
        tvm.tir.call_extern("float32", "some_external_call", input_buffer.data, not_device_context),
    )
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([input_buffer, not_device_context], stmt))
    mod = tvm.tir.transform.Apply(
        lambda f: f.with_attr("target", tvm.target.Target("llvm", host="llvm"))
    )(mod)
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)
    func = tvm.tir.transform.MakePackedAPI()(mod)["main"]

    num_args = func.params[2]

    # num_args assertion
    assert func.body.condition.a == num_args
    assert func.body.condition.b == 2

    # Arguments unpacking
    assignment = _find_assignment(func.body, "input_buffer")
    assert str(assignment.value) == 'T.tvm_struct_get(args, 0, 12, "handle")'

    assignment = _find_assignment(assignment.body, "input_buffer")
    assert str(assignment.value) == 'T.tvm_struct_get(input_buffer, 0, 1, "handle")'
    unpacked_input_buffer = assignment.var

    assignment = _find_assignment(func.body, "not_device_context")
    assert str(assignment.value) == 'T.tvm_struct_get(args, 1, 12, "handle")'
    unpacked_not_device_context = assignment.var

    seq_stmt = _find_next(assignment, tvm.tir.SeqStmt)
    call = _find_next(seq_stmt[1], tvm.tir.Evaluate)
    call_extern = call.value

    assert call_extern.args[1] == unpacked_input_buffer
    assert call_extern.args[2] == unpacked_not_device_context


def test_device_api_context_implicit_resource_handle():
    ib = tvm.tir.ir_builder.create()

    input_buffer = tvm.tir.decl_buffer(name="input_buffer", shape=[1])
    device_context = tvm.tir.Var("device_api_context", dtype="handle")

    ib.emit(
        tvm.tir.call_extern("float32", "some_external_call", input_buffer.data, device_context),
    )
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([input_buffer, device_context], stmt))
    mod = tvm.tir.transform.Apply(
        lambda f: f.with_attr("target", tvm.target.Target("llvm", host="llvm"))
    )(mod)
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)
    func = tvm.tir.transform.MakePackedAPI()(mod)["main"]

    num_args = func.params[2]
    device_context_in_resource_handle = func.params[5]

    # num_args assertion
    assert func.body.condition.a == num_args
    assert func.body.condition.b == 1

    # Arguments unpacking
    assignment = _find_assignment(func.body, "input_buffer")
    assert str(assignment.value) == 'T.tvm_struct_get(args, 0, 12, "handle")'

    assignment = _find_assignment(assignment.body, "input_buffer")
    assert str(assignment.value) == 'T.tvm_struct_get(input_buffer, 0, 1, "handle")'
    unpacked_input_buffer = assignment.var

    seq_stmt = _find_next(assignment, tvm.tir.SeqStmt)
    call = _find_next(seq_stmt[1], tvm.tir.Evaluate)
    call_extern = call.value

    assert call_extern.args[1] == unpacked_input_buffer
    assert call_extern.args[2] == device_context_in_resource_handle


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
        assert len(after.params) == 6
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


if __name__ == "__main__":
    test_makeapi()
