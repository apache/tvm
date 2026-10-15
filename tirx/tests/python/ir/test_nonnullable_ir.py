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

import tvm_tirx


@pytest.mark.parametrize(
    "name,args",
    [
        ("ir.Evaluate", (None, None)),
        ("ir.If", (True, None, None, None)),
        ("ir.If", (None, tvm_tirx.ir.Evaluate(0), None, None)),
        ("ir.While", (True, None, None)),
        ("ir.SeqStmt", ([None], None)),
        ("ir.Tuple", ([None], None)),
        ("ir.TupleGetItem", (None, 0, None)),
        ("ir.FuncType", ([], None)),
        ("ir.FuncType", ([None], tvm_tirx.ir.Type.missing())),
    ],
)
def test_required_reference_rejects_none(name, args):
    with pytest.raises(TypeError):
        tvm_ffi.get_global_func(name)(*args)


def test_optional_statement_fields_and_roundtrip():
    body = tvm_tirx.ir.Evaluate(0)
    conditional = tvm_tirx.ir.If(True, body, None)
    assert conditional.else_case is None
    loop = tvm_tirx.ir.For(
        tvm_tirx.ir.Var("i", tvm_tirx.ir.PrimType("int32")), 0, 4, tvm_tirx.ir.ForKind.DEFAULT, body, step=None
    )
    assert loop.step is None
    for value in [conditional, loop]:
        restored = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(value))
        tvm_tirx.ir.assert_structural_equal(value, restored, map_free_vars=True)


def test_missing_type_and_zero_remain_values():
    assert isinstance(tvm_tirx.ir.Type.missing(), tvm_tirx.ir.MissingType)
    # False and zero are valid primitive expressions, not absence.
    assert int(tvm_tirx.ir.Evaluate(False).value) == 0
    assert int(tvm_tirx.ir.Evaluate(0).value) == 0
