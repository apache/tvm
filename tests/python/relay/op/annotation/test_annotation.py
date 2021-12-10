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
"""Unit tests for annotations."""
import tvm
from tvm import relay
import pytest


def test_on_device_via_string():
    x = relay.Var("x")
    call = relay.annotation.on_device(x, "cuda")
    assert isinstance(call, relay.Call)
    assert len(call.args) == 1
    assert call.args[0] == x
    assert call.attrs.se_scope.device_type_int == 2  # ie kDLCUDA
    assert call.attrs.se_scope.virtual_device_id == 0
    assert call.attrs.se_scope.target is None
    assert call.attrs.se_scope.memory_scope == ""
    assert call.attrs.constrain_body
    assert not call.attrs.constrain_result


def test_on_device_via_device():
    x = relay.Var("x")
    call = relay.annotation.on_device(x, tvm.device("cpu"))
    assert call.attrs.se_scope.device_type_int == 1  # ie kDLCPU


def test_on_device_invalid_device():
    x = relay.Var("x")
    pytest.raises(ValueError, lambda: relay.annotation.on_device(x, "bogus"))


def test_on_device_fixed():
    x = relay.Var("x")
    call = relay.annotation.on_device(x, "cuda", constrain_result=True)
    assert call.attrs.se_scope.device_type_int == 2  # ie kDLCUDA
    assert call.attrs.constrain_body
    assert call.attrs.constrain_result


def test_on_device_free():
    x = relay.Var("x")
    call = relay.annotation.on_device(x, "cuda", constrain_result=False, constrain_body=False)
    assert call.attrs.se_scope.device_type_int == -1  # ie kInvalidDeviceType
    assert not call.attrs.constrain_body
    assert not call.attrs.constrain_result


def test_function_on_device():
    x = relay.Var("x")
    y = relay.Var("y")
    f = relay.Function([x, y], relay.add(x, y))
    func = relay.annotation.function_on_device(f, ["cpu", "cuda"], "cuda")
    assert isinstance(func, relay.Function)
    assert len(func.attrs["param_se_scopes"]) == 2
    assert func.attrs["param_se_scopes"][0].device_type_int == 1  # ie kDLCPU
    assert func.attrs["param_se_scopes"][1].device_type_int == 2  # ie kDLCUDA
    assert func.attrs["result_se_scope"].device_type_int == 2  # ie KDLCUDA


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__] + sys.argv[1:]))
