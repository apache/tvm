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
from tvm import tirx


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


def test_dict_attrs_has_nonzero_attr_accepts_int_imm():
    arg = tirx.Var("arg", "handle")
    func = tirx.PrimFunc([arg], tirx.Evaluate(0)).with_attr(
        {
            "global_symbol": "int_imm_noalias",
            "tirx.noalias": tirx.IntImm("int32", 1),
        }
    )

    tvm.compile(tvm.IRModule({"main": func}), target="c")


if __name__ == "__main__":
    test_dict_attrs()
    test_attrs_equal()
    test_assert_structural_equal_reports_mismatch()
    test_dict_attrs_has_nonzero_attr_accepts_int_imm()
