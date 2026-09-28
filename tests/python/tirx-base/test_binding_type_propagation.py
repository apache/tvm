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
def test_bind_rebinds_after_value_type_change(inplace):
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
def test_let_rebinds_before_body_and_updates_result_type(inplace):
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


@pytest.mark.parametrize("inplace", [False, True])
def test_bind_uses_type_identity_after_real_value_mutation(inplace):
    result_ty = relax.TensorType([2], "float32")
    binder_ty = relax.TensorType(result_ty.shape, result_ty.dtype)
    var = ir.Var("y", binder_ty)
    input_var = ir.Var("x", result_ty)
    replacement = ir.Var("replacement", result_ty)
    call = ir.Call("relax.exp", [input_var], ret_ty=result_ty)
    stmt = tirx.SeqStmt([tirx.Bind(var, call), tirx.Evaluate(var)])
    if inplace:
        stmt = stmt._move()

    rewritten = tvm_ffi.structural_map(
        stmt,
        (ir.Var, lambda value: replacement if value.same_as(input_var) else value),
    )
    binder = rewritten.seq[0].var
    assert rewritten.seq[0].value.args[0].same_as(replacement)
    assert not var.ty.same_as(result_ty)
    assert binder.ty.same_as(rewritten.seq[0].value.ty)
    assert rewritten.seq[1].value.same_as(binder)


@pytest.mark.parametrize("inplace", [False, True])
def test_unchanged_bind_and_let_keep_binder_identity(inplace):
    var = tirx.Var("x", "int32")
    binding = tirx.Bind(var, tirx.IntImm("int32", 1))
    let = prim.Let(var, tirx.IntImm("int32", 1), var)
    if inplace:
        binding = binding._move()
        let = let._move()

    rewritten_binding = tvm_ffi.structural_map(binding)
    rewritten_let = tvm_ffi.structural_map(let)
    assert rewritten_binding.var.same_as(var)
    assert rewritten_let.var.same_as(var)


@pytest.mark.parametrize("inplace", [False, True])
def test_unchanged_equal_but_distinct_types_do_not_rebind(inplace):
    result_ty = relax.TensorType([2], "float32")
    binder_ty = relax.TensorType(result_ty.shape, result_ty.dtype)
    var = ir.Var("y", binder_ty)
    x = ir.Var("x", result_ty)
    stmt = tirx.Bind(var, ir.Call("relax.exp", [x], ret_ty=result_ty))
    if inplace:
        stmt = stmt._move()

    assert not binder_ty.same_as(result_ty)
    rewritten = tvm_ffi.structural_map(stmt)
    assert rewritten.var.same_as(var)


@pytest.mark.parametrize("inplace", [False, True])
def test_binder_mutation_synchronizes_type(inplace):
    var = tirx.Var("x", "int32")
    stmt = tirx.SeqStmt([tirx.Bind(var, tirx.IntImm("int32", 1)), tirx.Evaluate(var)])
    if inplace:
        stmt = stmt._move()

    rewritten = tvm_ffi.structural_map(
        stmt,
        (tirx.Var, lambda value: tirx.Var("replacement", "int64") if value.same_as(var) else value),
    )
    binder = rewritten.seq[0].var
    assert not binder.same_as(var)
    assert binder.ty == ir.PrimType("int32")
    assert rewritten.seq[1].value.same_as(binder)


def test_let_mutates_rhs_before_binder_definition():
    source = tirx.Var("source", "int32")
    binder = tirx.Var("binder", "int32")
    visits = []

    def visit_var(var):
        visits.append(var)
        return var

    tvm_ffi.structural_map(prim.Let(binder, source, binder), (tirx.Var, visit_var))
    assert visits[0].same_as(source)
    assert visits[1].same_as(binder)


@pytest.mark.parametrize("inplace", [False, True])
def test_symbolic_call_type_rewrite_remaps_later_use(inplace):
    n = tirx.Var("n", "int64")
    tensor_ty = relax.TensorType([n], "float32")
    x = ir.Var("x", tensor_ty)
    y = ir.Var("y", tensor_ty)
    call = ir.Call("relax.exp", [x], ret_ty=tensor_ty)
    stmt = tirx.SeqStmt([tirx.Bind(y, call), tirx.Evaluate(y)])
    if inplace:
        stmt = stmt._move()

    rewritten = tvm_ffi.structural_map(
        stmt,
        (tirx.Var, lambda var: tirx.IntImm("int64", 4) if var.same_as(n) else var),
    )
    new_binding = rewritten.seq[0]
    ir.assert_structural_equal(new_binding.value.ty, relax.TensorType([4], "float32"))
    assert new_binding.var.ty.same_as(new_binding.value.ty)
    assert rewritten.seq[1].value.same_as(new_binding.var)
    ir.assert_structural_equal(ir.reinfer_type(new_binding.value), tensor_ty)


if __name__ == "__main__":
    tvm.testing.main()
