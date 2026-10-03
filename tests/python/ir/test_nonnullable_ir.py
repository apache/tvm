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
"""Required IR references reject absence; optional IR fields preserve it."""

import pytest
import tvm_ffi

import tvm


@pytest.mark.parametrize(
    "name,args",
    [
        ("tirx.Evaluate", (None, None)),
        ("tirx.IfThenElse", (True, None, None, None)),
        ("tirx.IfThenElse", (None, tvm.tirx.Evaluate(0), None, None)),
        ("tirx.While", (True, None, None)),
        ("tirx.SeqStmt", ([None], None)),
        ("ir.Tuple", ([None], None)),
        ("ir.TupleGetItem", (None, 0, None)),
        ("ir.FuncType", ([], None)),
        ("ir.FuncType", ([None], tvm.ir.Type.missing())),
    ],
)
def test_required_reference_rejects_none(name, args):
    with pytest.raises(TypeError):
        tvm_ffi.get_global_func(name)(*args)


def test_optional_statement_fields_and_roundtrip():
    body = tvm.tirx.Evaluate(0)
    conditional = tvm.tirx.IfThenElse(True, body, None)
    assert conditional.else_case is None
    loop = tvm.tirx.For(tvm.tirx.Var("i", "int32"), 0, 4, tvm.tirx.ForKind.SERIAL, body, step=None)
    assert loop.step is None
    declaration = tvm.tirx.PrimFunc([], None)
    assert declaration.body is None
    assert declaration.with_body(body).body.same_as(body)
    for value in [conditional, loop, declaration]:
        restored = tvm.ir.load_json(tvm.ir.save_json(value))
        tvm.ir.assert_structural_equal(value, restored, map_free_vars=True)
    assert "def" in declaration.script()


def test_offset_default_and_missing_type_remain_values():
    tensor = tvm.tirx.decl_tensor((8,), "float32", elem_offset=None)
    assert int(tensor.ty.elem_offset) == 0
    assert tvm.ir.Type.missing().is_missing()
    # False and zero are valid primitive expressions, not absence.
    assert int(tvm.tirx.Evaluate(False).value) == 0
    assert int(tvm.tirx.Evaluate(0).value) == 0


def test_failed_iter_map_has_absent_padding_predicate():
    i = tvm.tirx.Var("i", "int32")
    result = tvm.sym.detect_iter_map([i], {i: tvm.ir.Range(0, i)})
    assert result.errors
    assert result.padding_predicate is None
    restored = tvm.ir.load_json(tvm.ir.save_json(result))
    assert restored.padding_predicate is None


def test_unmapped_python_mutator_var_is_absent():
    @tvm.relax.expr_functor.mutator
    class Mutator(tvm.relax.PyExprMutator):
        pass

    mutator = Mutator()
    old = tvm.relax.Var("old", tvm.ir.PrimType("int32"))
    new = tvm.relax.Var("new", tvm.ir.PrimType("int32"))
    assert mutator.get_var_remap(old) is None
    mutator.set_var_remap(old, new)
    assert mutator.get_var_remap(old).same_as(new)
