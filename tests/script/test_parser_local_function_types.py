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
"""Local function references publish their inferred definition types."""

import pytest

import tvm
from tvm import relax
from tvm.script import parser


def _function(params, bindings, result, *, pure=True):
    blocks = [relax.BindingBlock(bindings)] if bindings else []
    body = relax.SeqExpr(blocks, result)
    relax.expr._update_type(body, result.ty)
    return relax.Function(params, body, result.ty, is_pure=pure)


def _call_binding(function, arguments, result_type):
    result = relax.Var("result", result_type)
    call = tvm.ir.Call(function, arguments, ret_ty=result_type)
    return result, relax.VarBinding(result, call)


def _local_binding(function):
    return next(
        binding
        for block in function.body.blocks
        for binding in block.bindings
        if isinstance(binding.value, relax.Function)
    )


def _call_to(function, variable):
    return next(
        binding.value
        for block in function.body.blocks
        for binding in block.bindings
        if isinstance(binding.value, tvm.ir.Call) and binding.value.op.same_as(variable)
    )


def test_local_any_annotation_refines_reference_and_downstream_call():
    source = """
@R.function(pure=False)
def main():
    @R.function(pure=False)
    def inner() -> R.Any:
        return R.print(format="message")
    return inner()
"""
    empty_type = tvm.ir.TupleType([])
    printed = relax.Var("printed", empty_type)
    inner = _function(
        [],
        [
            relax.VarBinding(
                printed, relax.BlockBuilder().normalize(relax.op.print(format="message"))
            )
        ],
        relax.Tuple([]),
        pure=False,
    )
    reference = relax.Var("inner", inner.ty)
    result, call_binding = _call_binding(reference, [], empty_type)
    expected = _function(
        [], [relax.VarBinding(reference, inner), call_binding], result, pure=False
    ).with_attr("global_symbol", "main")
    actual = parser.parse(source)
    tvm.ir.assert_structural_equal(expected, actual)
    binding = _local_binding(actual)
    call = _call_to(actual, binding.var)
    assert isinstance(binding.var.ty.ret, tvm.ir.TupleType)
    assert len(binding.var.ty.ret.fields) == 0
    tvm.ir.assert_structural_equal(binding.var.ty, binding.value.ty)
    tvm.ir.assert_structural_equal(call.ty, binding.value.ret_ty)


@pytest.mark.parametrize("annotation", ["", " -> R.Tensor"])
def test_local_tensor_result_refines_omitted_or_broad_annotation(annotation):
    source = f"""
@R.function
def main(x: R.Tensor((2, 3), "float32")):
    @R.function
    def inner(y: R.Tensor((2, 3), "float32")){annotation}:
        return y
    return inner(x)
"""
    tensor_type = relax.TensorType((2, 3), "float32")
    x, y = relax.Var("x", tensor_type), relax.Var("y", tensor_type)
    inner = _function([y], [], y)
    reference = relax.Var("inner", inner.ty)
    result, call_binding = _call_binding(reference, [x], tensor_type)
    expected = _function([x], [relax.VarBinding(reference, inner), call_binding], result).with_attr(
        "global_symbol", "main"
    )
    actual = parser.parse(source)
    tvm.ir.assert_structural_equal(expected, actual)
    binding = _local_binding(actual)
    tvm.ir.assert_structural_equal(binding.var.ty, binding.value.ty)
    assert [extent.value for extent in binding.var.ty.ret.shape] == [2, 3]
    call = _call_to(actual, binding.var)
    tvm.ir.assert_structural_equal(call.ty, binding.value.ret_ty)


def test_local_higher_order_result_refines_callable_type():
    source = """
@R.function
def main(x: R.Tensor((2, 3), "float32")):
    @R.function
    def outer() -> R.Callable((R.Tensor(None, "float32", ndim=2),),
                              R.Tensor(None, "float32", ndim=2)):
        @R.function
        def inner(y: R.Tensor((2, 3), "float32")):
            return y
        return inner
    closure = outer()
    return closure(x)
"""
    tensor_type = relax.TensorType((2, 3), "float32")
    x, y = relax.Var("x", tensor_type), relax.Var("y", tensor_type)
    inner = _function([y], [], y)
    inner_reference = relax.Var("inner", inner.ty)
    outer = _function([], [relax.VarBinding(inner_reference, inner)], inner_reference)
    outer_reference = relax.Var("outer", outer.ty)
    closure, closure_binding = _call_binding(outer_reference, [], inner.ty)
    result, call_binding = _call_binding(closure, [x], tensor_type)
    expected = _function(
        [x], [relax.VarBinding(outer_reference, outer), closure_binding, call_binding], result
    ).with_attr("global_symbol", "main")
    actual = parser.parse(source)
    tvm.ir.assert_structural_equal(expected, actual)
    binding = _local_binding(actual)
    tvm.ir.assert_structural_equal(binding.var.ty, binding.value.ty)
    assert [extent.value for extent in binding.var.ty.ret.params[0].shape] == [2, 3]


def test_local_recursive_calls_keep_declared_reference_identity():
    source = """
@R.function
def main(x: R.Tensor((2,), "float32")):
    @R.function
    def inner(y: R.Tensor((2,), "float32")) -> R.Tensor((2,), "float32"):
        return inner(y)
    return inner(x)
"""
    tensor_type = relax.TensorType((2,), "float32")
    x, y = relax.Var("x", tensor_type), relax.Var("y", tensor_type)
    reference = relax.Var("inner", relax.FuncType([tensor_type], tensor_type))
    recursive_result, recursive_binding = _call_binding(reference, [y], tensor_type)
    inner = _function([y], [recursive_binding], recursive_result)
    result, call_binding = _call_binding(reference, [x], tensor_type)
    expected = _function([x], [relax.VarBinding(reference, inner), call_binding], result).with_attr(
        "global_symbol", "main"
    )
    actual = parser.parse(source)
    tvm.ir.assert_structural_equal(expected, actual)
    binding = _local_binding(actual)
    assert _call_to(binding.value, binding.var).op.same_as(binding.var)
    assert _call_to(actual, binding.var).op.same_as(binding.var)
    tvm.ir.assert_structural_equal(binding.var.ty, binding.value.ty)
