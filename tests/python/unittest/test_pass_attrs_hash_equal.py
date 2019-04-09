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

def test_attrs_equal():
    x = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3, 4))
    y = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3, 4))
    z = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3,4,1))
    assert tvm.ir_pass.AttrsEqual(x, y)
    assert not tvm.ir_pass.AttrsEqual(x, z)

    dattr = tvm.make.node("DictAttrs", x=1, y=10, name="xyz", padding=(0,0))
    assert not tvm.ir_pass.AttrsEqual(dattr, x)
    dattr2 = tvm.make.node("DictAttrs", x=1, y=10, name="xyz", padding=(0,0))
    assert tvm.ir_pass.AttrsEqual(dattr, dattr2)

    assert tvm.ir_pass.AttrsEqual({"x": x}, {"x": y})
    # array related checks
    assert tvm.ir_pass.AttrsEqual({"x": [x, x]}, {"x": [y, x]})
    assert not tvm.ir_pass.AttrsEqual({"x": [x, 1]}, {"x": [y, 2]})

    n = tvm.var("n")
    assert tvm.ir_pass.AttrsEqual({"x": n+1}, {"x": n+1})





def test_attrs_hash():
    fhash = tvm.ir_pass.AttrsHash
    x = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3, 4))
    y = tvm.make.node("attrs.TestAttrs", name="xx", padding=(3, 4))
    assert fhash({"x": x}) == fhash({"x": y})
    assert fhash({"x": x}) != fhash({"x": [y, 1]})
    assert fhash({"x": [x, 1]}) == fhash({"x": [y, 1]})
    assert fhash({"x": [x, 2]}) == fhash({"x": [y, 2]})


if __name__ == "__main__":
    test_attrs_equal()
    test_attrs_hash()
