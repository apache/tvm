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
"""Test TIRx function structural mutation."""

import tvm_ffi


def test_structural_map_specializes_function_signature():
    from tvm_tirx.ir import Evaluate, FuncType, IntImm, Return, Var, assert_structural_equal
    from tvm_tirx.tirx import Function, TensorType

    n = Var("n", "int32")
    scale = Var("scale", "int32")
    tensor = Var("tensor", TensorType((n,), "float32"))
    original = Function([tensor, scale], [Evaluate(scale), Return(tensor)], tensor.ty)
    bindings = {n: IntImm("int32", 16), scale: IntImm("int32", 2)}

    # Removing runtime parameters is explicit. structural_map handles the shape
    # definitions embedded in tensor types and remaps the retained tensor's uses.
    signature = Function([tensor], original.body, original.ret_type)
    specialized = tvm_ffi.structural_map(
        signature, (Var, lambda var: bindings.get(var, var)), order="pre"
    )
    expected_tensor = Var("tensor", TensorType((16,), "float32"))
    expected = Function(
        [expected_tensor],
        [Evaluate(IntImm("int32", 2)), Return(expected_tensor)],
        expected_tensor.ty,
    )
    assert_structural_equal(specialized, expected)
    assert specialized.body.seq[1].value.same_as(specialized.params[0])
    assert_structural_equal(
        specialized.ty, FuncType([p.ty for p in specialized.params], specialized.ret_type)
    )
    assert original.params[0].ty.shape[0].same_as(n)
    assert original.ty.arg_types[0].shape[0].same_as(n)

    # A unique declaration allows the parameter array and its Var to remain in
    # place while a nested TensorType changes. The cached FuncType must follow.
    def declaration():
        return Function(
            [Var("tensor", TensorType((n,), "float32"))], None, TensorType((n,), "float32")
        )

    inplace = declaration()
    inplace = tvm_ffi.structural_map(
        inplace._move(), (Var, lambda var: bindings.get(var, var)), order="pre"
    )
    assert int(inplace.params[0].ty.shape[0]) == 16
    assert int(inplace.ret_type.shape[0]) == 16
    assert_structural_equal(inplace.ty, FuncType([inplace.params[0].ty], inplace.ret_type))
