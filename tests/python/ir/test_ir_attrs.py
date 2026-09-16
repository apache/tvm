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
# ruff: noqa: F841
import pytest
import tvm_ffi

import tvm


def test_dict_attrs():
    dattr = tvm.ir.make_node("ir.DictAttrs", x=1, y=10, name="xyz", padding=(0, 0))
    assert dattr.x == 1
    datrr = tvm.ir.load_json(tvm.ir.save_json(dattr))
    assert dattr.name == "xyz"
    assert isinstance(dattr, tvm.ir.DictAttrs)
    assert "name" in dattr
    assert dattr["x"] == 1
    assert len(dattr) == 4
    assert len([x for x in dattr.keys()]) == 4
    assert len(dattr.items()) == 4


def test_attrs_equal():
    dattr0 = tvm.ir.make_node("ir.DictAttrs", x=1, y=[10, 20])
    dattr1 = tvm.ir.make_node("ir.DictAttrs", y=[10, 20], x=1)
    dattr2 = tvm.ir.make_node("ir.DictAttrs", x=1, y=None)
    tvm.ir.assert_structural_equal(dattr0, dattr1)
    assert not tvm_ffi.structural_equal(dattr0, dattr2)
    assert not tvm_ffi.structural_equal({"x": 1}, tvm.runtime.convert(1))
    assert not tvm_ffi.structural_equal([1, 2], tvm.runtime.convert(1))


def test_assert_structural_equal_reports_mismatch():
    dattr0 = tvm.ir.make_node("ir.DictAttrs", x=1, y=[10, 20])
    dattr1 = tvm.ir.make_node("ir.DictAttrs", x=1, y=[10, 30])

    with pytest.raises(ValueError) as err:
        tvm.ir.assert_structural_equal(dattr0, dattr1)

    message = str(err.value)
    assert "StructuralEqual check failed" in message
    assert "caused by lhs at" in message
    assert "and rhs at" in message


@pytest.mark.parametrize("kind", ["prim", "relax", "extern"])
def test_function_attribute_copy_preserves_fields(kind):
    span = tvm.ir.Span(tvm.ir.SourceName("attrs"), 1, 2, 3, 4)
    if kind == "prim":
        var = tvm.tirx.Var("x", "int32")
        func = tvm.tirx.PrimFunc([var], tvm.tirx.Evaluate(var), span=span)
        fields = ["params", "body", "ret_type", "ty", "span"]
    elif kind == "relax":
        var = tvm.relax.Var("x", tvm.relax.TensorType([2], "float32"))
        func = tvm.relax.Function([var], var, is_pure=False, span=span)
        fields = ["params", "body", "ret_ty", "ty", "span"]
    else:
        func = tvm.relax.ExternFunc("external_symbol", span=span)
        fields = ["ty", "span"]

    func = func.with_attr("keep", 1)
    shared_attrs = func.attrs
    added = func.with_attr("added", 2)
    updated = func.with_attr({"keep": 3, "added": 4})
    removed = func.without_attr("keep")
    assert func.with_attr({}).same_as(func)
    for result in [added, updated, removed]:
        assert type(result) is type(func)
        assert not result.same_as(func)
        for field in fields:
            assert getattr(result, field).same_as(getattr(func, field))
        if kind == "relax":
            assert result.is_pure is False
        if kind == "extern":
            assert result.global_symbol == "external_symbol"
        assert func.attrs.same_as(shared_attrs)
        assert dict(shared_attrs) == {"keep": 1}
    assert dict(added.attrs) == {"keep": 1, "added": 2}
    assert dict(updated.attrs) == {"keep": 3, "added": 4}
    assert not removed.attrs


if __name__ == "__main__":
    test_dict_attrs()
    test_attrs_equal()
