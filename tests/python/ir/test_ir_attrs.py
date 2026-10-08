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


def test_op_set_signature_metadata_and_counts():
    name = "test.python_set_signature"
    tvm.ir._ffi_api.RegisterOp(name, "Python signature test")
    op = tvm.ir.Op.get(name)
    x = tvm.ir.Var("x", tvm.ir.AnyType())
    ty = tvm.ir.AnyType()

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
    tvm.ir.Call(op, [x, x], ty_args=[ty]).validate()
    tvm.ir.Call(op, [x, x, x], ty_args=[ty, ty]).validate()
    with pytest.raises(TypeError, match=r"Call.args expected at least 2 arguments, got 1"):
        tvm.ir.Call(op, [x], ty_args=[ty]).validate()
    with pytest.raises(TypeError, match=r"Call.ty_args expected at least 1 type argument, got 0"):
        tvm.ir.Call(op, [x, x]).validate()

    op.set_signature(["only"])
    assert [info.name for info in op.args_info] == ["only"]
    assert op.var_args_info is None and op.var_ty_args_info is None
    tvm.ir.Call(op, [x]).validate()
    with pytest.raises(TypeError, match=r"Call.args expected 1 argument, got 2"):
        tvm.ir.Call(op, [x, x]).validate()
    with pytest.raises(TypeError, match=r"Call.ty_args expected 0 type arguments, got 1"):
        tvm.ir.Call(op, [x], ty_args=[ty]).validate()
    with pytest.raises(TypeError, match=r"args must be a name string or a \(name, doc\) tuple"):
        op.set_signature(["new", ("bad", 7)])
    with pytest.raises(TypeError, match=r"var_ty_args must be a name string"):
        op.set_signature(["new"], var_ty_args=("bad", 7))
    assert [info.name for info in op.args_info] == ["only"]
    tvm.ir.Call(op, [x]).validate()


def test_op_set_signature_preserves_custom_validator():
    op = tvm.ir.Op.get("relax.call_pure_packed")
    original_args = [(info.name, info.doc) for info in op.args_info]
    original_ty_args = [(info.name, info.doc) for info in op.ty_args_info]
    original_var_args = (op.var_args_info.name, op.var_args_info.doc)
    original_var_ty_args = (op.var_ty_args_info.name, op.var_ty_args_info.doc)
    try:
        op.set_signature()
        with pytest.raises(TypeError, match="call_pure_packed expects a function argument"):
            tvm.ir.Call(op, []).validate()
    finally:
        op.set_signature(
            original_args,
            ty_args=original_ty_args,
            var_args=original_var_args,
            var_ty_args=original_var_ty_args,
        )


def test_op_set_signature_checks_counts_without_types_or_attrs():
    name = "test.python_set_signature.count_only"
    tvm.ir._ffi_api.RegisterOp(name, "Count-only Python signature")
    op = tvm.ir.Op.get(name)
    op.set_signature(["value"], ty_args=["T"])

    call = tvm.ir.Call(
        op,
        [tvm.ir.Var("x", tvm.ir.AnyType())],
        attrs={"note": "permitted"},
        ty_args=[tvm.ir.StringType()],
    )
    assert call.attrs is not None


def test_ptx_ops_have_variadic_call_signature():
    from tvm.backend.cuda.ptx import TABLE

    op = tvm.ir.Op.get(next(iter(TABLE.values())).op_name)
    assert not op.args_info
    assert op.var_args_info.name == "operands"
    tvm.ir.Call(op, []).validate()
    tvm.ir.Call(op, [tvm.ir.Var("x", tvm.ir.AnyType())]).validate()


if __name__ == "__main__":
    test_dict_attrs()
    test_attrs_equal()
