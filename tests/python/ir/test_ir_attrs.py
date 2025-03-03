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
import tvm
import pytest
import tvm.ir._ffi_api


def test_make_attrs():
    with pytest.raises(AttributeError):
        x = tvm.ir.make_node("attrs.TestAttrs", unknown_key=1, name="xx")

    with pytest.raises(AttributeError):
        x = tvm.ir.make_node("attrs.TestAttrs", axis=100, name="xx")

    x = tvm.ir.make_node("attrs.TestAttrs", name="xx", padding=(3, 4))
    assert x.name == "xx"
    assert x.padding[0].value == 3
    assert x.padding[1].value == 4
    assert x.axis == 10


def test_dict_attrs():
    dattr = tvm.ir.make_node("DictAttrs", x=1, y=10, name="xyz", padding=(0, 0))
    assert dattr.x.value == 1
    datrr = tvm.ir.load_json(tvm.ir.save_json(dattr))
    assert dattr.name == "xyz"
    assert isinstance(dattr, tvm.ir.DictAttrs)
    assert "name" in dattr
    assert dattr["x"].value == 1
    assert len(dattr) == 4
    assert len([x for x in dattr.keys()]) == 4
    assert len(dattr.items()) == 4


def test_attrs_equal():
    dattr0 = tvm.ir.make_node("DictAttrs", x=1, y=[10, 20])
    dattr1 = tvm.ir.make_node("DictAttrs", y=[10, 20], x=1)
    dattr2 = tvm.ir.make_node("DictAttrs", x=1, y=None)
    tvm.ir.assert_structural_equal(dattr0, dattr1)
    assert not tvm.ir.structural_equal(dattr0, dattr2)
    assert not tvm.ir.structural_equal({"x": 1}, tvm.runtime.convert(1))
    assert not tvm.ir.structural_equal([1, 2], tvm.runtime.convert(1))


if __name__ == "__main__":
    test_make_attrs()
    test_dict_attrs()
    test_attrs_equal()
