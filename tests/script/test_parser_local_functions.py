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
"""Local Relax functions retain their enclosing binding region and identities."""

import textwrap

import pytest

import tvm
from tvm import relax
from tvm.script.parser import parse


def _local_binding(function):
    return next(
        binding
        for block in function.body.blocks
        for binding in block.bindings
        if isinstance(binding.value, relax.Function)
    )


@pytest.mark.parametrize("dataflow", [False, True])
@pytest.mark.parametrize("recursive", [False, True])
def test_local_function_binding_region_and_reference_identity(dataflow, recursive):
    inner_return = "inner(y)" if recursive else "R.add(x, y)"
    region = f"""\
@R.function
def inner(y: R.Tensor((2,), "float32")) -> R.Tensor((2,), "float32"):
    return {inner_return}
result = inner(x)
"""
    if dataflow:
        region = "with R.dataflow():\n" + textwrap.indent(region + "R.output(result)\n", "    ")
    source = (
        '@R.function\ndef main(x: R.Tensor((2,), "float32")):\n'
        + textwrap.indent(region, "    ")
        + "    return result\n"
    )
    function = parse(source)
    binding = _local_binding(function)
    assert isinstance(binding.var, relax.DataflowVar) == (dataflow and not recursive)
    call = function.body.blocks[0].bindings[-1].value
    assert call.op.same_as(binding.var)
    inner_call = binding.value.body.blocks[0].bindings[0].value
    if recursive:
        assert inner_call.op.same_as(binding.var)
    else:
        assert inner_call.args[0].same_as(function.params[0])
        assert inner_call.args[1].same_as(binding.value.params[0])
    relax.analysis.well_formed(tvm.IRModule({"main": function}))


def test_explicitly_output_local_function_keeps_recursive_reference():
    function = parse("""
@R.function
def main():
    with R.dataflow():
        @R.function
        def inner(y: R.Tensor((2,), "float32")) -> R.Tensor((2,), "float32"):
            return inner(y)
        R.output(inner)
    return inner
""")
    binding = _local_binding(function)
    assert not isinstance(binding.var, relax.DataflowVar)
    assert function.body.body.same_as(binding.var)
    inner_call = binding.value.body.blocks[0].bindings[0].value
    assert inner_call.op.same_as(binding.var)
    relax.analysis.well_formed(tvm.IRModule({"main": function}))


@pytest.mark.parametrize("recursive", [False, True])
def test_local_dependent_signature_preserves_parameter_and_capture_scopes(recursive):
    inner_return = "inner(current, value)" if recursive else "value"
    function = parse(f"""
@R.function
def main(n: R.Prim("int64"), m: R.Prim("int64"), x: R.Tensor((n, m), "float32")):
    @R.function
    def inner(current: R.Prim("int64"), value: R.Tensor((current, m), "float32")) -> R.Tensor(
        (current, m), "float32"
    ):
        return {inner_return}
    return inner(n, x)
""")
    binding = _local_binding(function)
    current, value = binding.value.params
    assert value.ty.shape[0].same_as(current)
    assert binding.value.ret_ty.shape[0].same_as(current)
    signature_current = binding.var.ty.params[1].shape[0]
    assert signature_current.same_as(current) != recursive
    assert binding.var.ty.ret.shape[0].same_as(signature_current)
    assert binding.var.ty.params[1].shape[1].same_as(function.params[1])
    assert binding.var.ty.ret.shape[1].same_as(function.params[1])
    tvm.ir.assert_structural_equal(binding.var.ty, binding.value.ty)
    call = function.body.blocks[0].bindings[-1].value
    assert call.op.same_as(binding.var)
    assert call.ty.shape[0].same_as(function.params[0])
    if recursive:
        recursive_call = binding.value.body.blocks[0].bindings[0].value
        assert recursive_call.op.same_as(binding.var)
        assert recursive_call.ty.shape[0].same_as(current)
    relax.analysis.well_formed(tvm.IRModule({"main": function}))
