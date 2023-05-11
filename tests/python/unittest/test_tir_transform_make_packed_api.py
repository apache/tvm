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

import tvm
from tvm import te
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
                "target": tvm.target.Target("llvm"),
                "global_symbol": "main",
            }
        )
    )(mod)

    f = tvm.tir.transform.MakePackedAPI()(mod)["main"]
    assert len(f.params) == 6


def _find_assignment(stmt, var_name):
    while not isinstance(stmt, tvm.tir.LetStmt):
        stmt = stmt.body

    if stmt.var.name != var_name:
        return _find_assignment(stmt.body, var_name)

    return stmt


def _find_next(stmt, type):
    while not isinstance(stmt, type):
        stmt = stmt.body
    return stmt


def test_variable_passed_from_args():
    ib = tvm.tir.ir_builder.create()

    input_buffer = tvm.tir.decl_buffer(name="input_buffer", shape=[1])
    not_device_context = tvm.tir.Var("not_device_context", dtype="handle")

    ib.emit(
        tvm.tir.call_extern("float32", "some_external_call", input_buffer.data, not_device_context),
    )
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([input_buffer, not_device_context], stmt))
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", tvm.target.Target("llvm")))(mod)
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)
    func = tvm.tir.transform.MakePackedAPI()(mod)["main"]

    num_args = func.params[2]

    # num_args assertion
    assert func.body.condition.a == num_args
    assert func.body.condition.b == 2

    # Arguments unpacking
    assignment = _find_assignment(func.body, "arg.input_buffer")
    assert str(assignment.value) == 'T.tvm_struct_get(args, 0, 12, "handle")'

    assignment = _find_assignment(func.body, "arg.not_device_context")
    assert str(assignment.value) == 'T.tvm_struct_get(args, 1, 12, "handle")'

    assignment = _find_assignment(func.body, "input_buffer")
    assert str(assignment.value) == 'T.tvm_struct_get(arg_input_buffer, 0, 1, "handle")'
    unpacked_input_buffer = assignment.var

    assignment = _find_assignment(func.body, "not_device_context")
    assert str(assignment.value) == "arg_not_device_context"
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
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", tvm.target.Target("llvm")))(mod)
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)
    func = tvm.tir.transform.MakePackedAPI()(mod)["main"]

    num_args = func.params[2]
    device_context_in_resource_handle = func.params[5]

    # num_args assertion
    assert func.body.condition.a == num_args
    assert func.body.condition.b == 1

    # Arguments unpacking
    assignment = _find_assignment(func.body, "arg.input_buffer")
    assert str(assignment.value) == 'T.tvm_struct_get(args, 0, 12, "handle")'

    assignment = _find_assignment(func.body, "input_buffer")
    assert str(assignment.value) == 'T.tvm_struct_get(arg_input_buffer, 0, 1, "handle")'
    unpacked_input_buffer = assignment.var

    seq_stmt = _find_next(assignment, tvm.tir.SeqStmt)
    call = _find_next(seq_stmt[1], tvm.tir.Evaluate)
    call_extern = call.value

    assert call_extern.args[1] == unpacked_input_buffer
    assert call_extern.args[2] == device_context_in_resource_handle


if __name__ == "__main__":
    test_makeapi()
