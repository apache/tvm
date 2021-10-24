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
""" test bind function."""
import pytest
import tvm
from tvm import te
from tvm import relay
from tvm import TVMError


def test_bind_params():
    x = relay.var("x")
    y = relay.var("y")
    z = relay.add(x, y)
    f = relay.Function([x, y], z)
    fbinded = relay.bind(f, {x: relay.const(1, "float32")})
    fexpected = relay.Function([y], relay.add(relay.const(1, "float32"), y))
    assert tvm.ir.structural_equal(fbinded, fexpected)

    zbinded = relay.bind(z, {y: x})
    zexpected = relay.add(x, x)
    assert tvm.ir.structural_equal(zbinded, zexpected)


def test_bind_duplicated_params():
    a = relay.var("a", shape=(1,))
    aa = relay.var("a", shape=(1,))
    s = a + aa
    func = relay.Function([a, aa], s)

    with pytest.raises(TVMError):
        relay.build_module.bind_params_by_name(func, {"a": [1.0]})


if __name__ == "__main__":
    test_bind_params()
    test_bind_duplicated_params()
