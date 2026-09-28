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
import tvm_ffi

import tvm
from tvm import ir, relax, tirx
from tvm.ir import prim


@pytest.mark.parametrize("inplace", [False, True])
def test_bind_value_type_updates_binder_and_uses(inplace):
    var = tirx.Var("x", "int32")
    stmt = tirx.SeqStmt([tirx.Bind(var, tirx.IntImm("int32", 1)), tirx.Evaluate(var)])
    if inplace:
        stmt = stmt._move()

    rewritten = tvm_ffi.structural_map(
        stmt,
        (tirx.IntImm, lambda value: tirx.IntImm("int64", value.value)),
    )
    binder = rewritten.seq[0].var
    assert binder.ty == ir.PrimType("int64")
    assert rewritten.seq[1].value.same_as(binder)
    assert rewritten.seq[0].value.ty.same_as(binder.ty)


@pytest.mark.parametrize("inplace", [False, True])
def test_let_value_type_updates_binder_body_and_result(inplace):
    var = tirx.Var("x", "int32")
    expr = prim.Let(var, tirx.IntImm("int32", 1), var)
    if inplace:
        expr = expr._move()

    rewritten = tvm_ffi.structural_map(
        expr,
        (tirx.IntImm, lambda value: tirx.IntImm("int64", value.value)),
    )
    assert rewritten.var.ty == ir.PrimType("int64")
    assert rewritten.body.same_as(rewritten.var)
    assert rewritten.ty.same_as(rewritten.body.ty)


def test_unchanged_bind_and_let_preserve_identity():
    var = tirx.Var("x", "int32")
    binding = tirx.Bind(var, tirx.IntImm("int32", 1))
    let = prim.Let(var, tirx.IntImm("int32", 1), var)
    assert tvm_ffi.structural_map(binding).same_as(binding)
    assert tvm_ffi.structural_map(let).same_as(let)


def test_bind_allows_missing_value_type():
    var = tirx.Var("x", "int32")
    binding = tirx.Bind(var, tirx.IntImm("int32", 1))
    rewritten = tvm_ffi.structural_map(
        binding,
        (tirx.IntImm, lambda _: ir.Call("tirx.call_extern", [ir.StringImm("f")])),
    )
    assert rewritten.var.ty.same_as(ir.Type.Missing())


def test_symbolic_call_type_changes_structurally_without_reinference():
    n = tirx.Var("n", "int64")
    tensor_type = relax.TensorType([n], "float32")
    x = ir.Var("x", tensor_type)
    y = ir.Var("y", tensor_type)
    call = ir.Call("relax.exp", [x], ret_ty=tensor_type)
    binding = tirx.SeqStmt([tirx.Bind(y, call), tirx.Evaluate(y)])

    rewritten = tvm_ffi.structural_map(
        binding,
        (tirx.Var, lambda var: tirx.IntImm("int64", 4) if var.same_as(n) else var),
    )
    new_binding = rewritten.seq[0]
    ir.assert_structural_equal(new_binding.value.ty, relax.TensorType([4], "float32"))
    ir.assert_structural_equal(new_binding.var.ty, new_binding.value.ty)
    assert rewritten.seq[1].value.same_as(new_binding.var)
    ir.assert_structural_equal(ir.reinfer_type(new_binding.value), tensor_type)


if __name__ == "__main__":
    tvm.testing.main()
