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

import tvm_tirx


def test_dict_attrs():
    dattr = tvm_tirx.ir.make_node(
        "ir.DictAttrs", dict={"x": 1, "y": 10, "name": "xyz", "padding": (0, 0)}
    )
    assert dattr.x == 1
    assert dattr.dict["x"] == 1
    datrr = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(dattr))
    assert dattr.name == "xyz"
    assert isinstance(dattr, tvm_tirx.ir.DictAttrs)
    assert "name" in dattr
    assert dattr["x"] == 1
    assert len(dattr) == 4
    assert len([x for x in dattr.keys()]) == 4
    assert len(dattr.items()) == 4


def test_attrs_equal():
    dattr0 = tvm_tirx.ir.make_node("ir.DictAttrs", dict={"x": 1, "y": [10, 20]})
    dattr1 = tvm_tirx.ir.make_node("ir.DictAttrs", dict={"y": [10, 20], "x": 1})
    dattr2 = tvm_tirx.ir.make_node("ir.DictAttrs", dict={"x": 1, "y": None})
    tvm_tirx.ir.assert_structural_equal(dattr0, dattr1)
    assert not tvm_ffi.structural_equal(dattr0, dattr2)
    assert not tvm_ffi.structural_equal({"x": 1}, tvm_tirx.runtime.convert(1))
    assert not tvm_ffi.structural_equal([1, 2], tvm_tirx.runtime.convert(1))


def test_assert_structural_equal_reports_mismatch():
    dattr0 = tvm_tirx.ir.make_node("ir.DictAttrs", dict={"x": 1, "y": [10, 20]})
    dattr1 = tvm_tirx.ir.make_node("ir.DictAttrs", dict={"x": 1, "y": [10, 30]})

    with pytest.raises(ValueError) as err:
        tvm_tirx.ir.assert_structural_equal(dattr0, dattr1)

    message = str(err.value)
    assert "StructuralEqual check failed" in message
    assert "caused by lhs at" in message
    assert "and rhs at" in message


def test_op_set_signature_metadata_and_counts():
    name = "test.python_set_signature"
    tvm_tirx.ir._ffi_api.RegisterOp(name, "Python signature test")
    op = tvm_tirx.ir.Op.get(name)
    x = tvm_tirx.ir.Var("x", tvm_tirx.ir.AnyType())
    ty = tvm_tirx.ir.AnyType()

    op.set_signature(
        ["lhs", ("rhs", "Second input")],
        ty_args=[("T", "Result type")],
        var_args=("rest", "Other inputs"),
        var_ty_args="more_types",
    )
    assert [(info.name, info.doc) for info in op.args_info] == [
        ("lhs", ""),
        ("rhs", "Second input"),
    ]
    assert [(info.name, info.doc) for info in op.ty_args_info] == [("T", "Result type")]
    assert (op.var_args_info.name, op.var_args_info.doc) == ("rest", "Other inputs")
    assert op.var_ty_args_info.name == "more_types"
    tvm_tirx.ir.Call(op, [x, x], ty_args=[ty]).validate()
    tvm_tirx.ir.Call(op, [x, x, x], ty_args=[ty, ty]).validate()
    with pytest.raises(TypeError, match=r"Call.args expected at least 2 arguments, got 1"):
        tvm_tirx.ir.Call(op, [x], ty_args=[ty]).validate()
    with pytest.raises(TypeError, match=r"Call.ty_args expected at least 1 type argument, got 0"):
        tvm_tirx.ir.Call(op, [x, x]).validate()

    op.set_signature(["only"])
    assert [info.name for info in op.args_info] == ["only"]
    assert op.var_args_info is None and op.var_ty_args_info is None
    tvm_tirx.ir.Call(op, [x]).validate()
    with pytest.raises(TypeError, match=r"Call.args expected 1 argument, got 2"):
        tvm_tirx.ir.Call(op, [x, x]).validate()
    with pytest.raises(TypeError, match=r"Call.ty_args expected 0 type arguments, got 1"):
        tvm_tirx.ir.Call(op, [x], ty_args=[ty]).validate()
    with pytest.raises(TypeError, match=r"args must be a name string or a \(name, doc\) tuple"):
        op.set_signature(["new", ("bad", 7)])
    with pytest.raises(TypeError, match=r"var_ty_args must be a name string"):
        op.set_signature(["new"], var_ty_args=("bad", 7))
    assert [info.name for info in op.args_info] == ["only"]
    tvm_tirx.ir.Call(op, [x]).validate()


def test_op_set_signature_checks_counts_without_types_or_attrs():
    name = "test.python_set_signature.count_only"
    tvm_tirx.ir._ffi_api.RegisterOp(name, "Count-only Python signature")
    op = tvm_tirx.ir.Op.get(name)
    op.set_signature(["value"], ty_args=["T"])

    call = tvm_tirx.ir.Call(
        op,
        [tvm_tirx.ir.Var("x", tvm_tirx.ir.AnyType())],
        attrs={"note": "permitted"},
        ty_args=[tvm_tirx.ir.StringType()],
    )
    assert call.attrs is not None


def test_generated_op_api_constructs_dictionary_attrs():
    name = "test.python_dict_attrs_api"
    tvm_tirx.ir._ffi_api.RegisterOp(name, "Dictionary attribute API test")
    op = tvm_tirx.ir.Op.get(name)
    op.set_signature(args=[])
    op.set_attrs_type_key("ir.DictAttrs")
    tvm_tirx.ir.register_op_attr(name, "TFixedReturnType", tvm_tirx.ir.PrimType("int32"))
    api = tvm_tirx.ir.op._make_op_api(op, __name__)
    assert len(api().attrs) == 0
    call = api(label="preserved", axis=2)
    assert call.attrs["label"] == "preserved"
    assert call.attrs["axis"] == 2
    assert call.ty == tvm_tirx.ir.PrimType("int32")


if __name__ == "__main__":
    test_dict_attrs()
    test_attrs_equal()
