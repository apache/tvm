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
import tvm.testing
from tvm import relay
import pytest


def test_on_device_via_string():
    x = relay.Var("x")
    call = relay.annotation.on_device(x, "cuda")
    assert isinstance(call, relay.Call)
    assert len(call.args) == 1
    assert call.args[0] == x
    assert call.attrs.virtual_device.device_type_int == 2  # ie kDLCUDA
    assert call.attrs.virtual_device.virtual_device_id == 0
    assert call.attrs.virtual_device.target is None
    assert call.attrs.virtual_device.memory_scope == ""
    assert call.attrs.constrain_body
    assert not call.attrs.constrain_result


def test_on_device_via_device():
    x = relay.Var("x")
    call = relay.annotation.on_device(x, tvm.device("cpu"))
    assert call.attrs.virtual_device.device_type_int == 1  # ie kDLCPU


def test_on_device_invalid_device():
    x = relay.Var("x")
    pytest.raises(ValueError, lambda: relay.annotation.on_device(x, "bogus"))


def test_on_device_fixed():
    x = relay.Var("x")
    call = relay.annotation.on_device(x, "cuda", constrain_result=True)
    assert call.attrs.virtual_device.device_type_int == 2  # ie kDLCUDA
    assert call.attrs.constrain_body
    assert call.attrs.constrain_result


def test_on_device_free():
    x = relay.Var("x")
    call = relay.annotation.on_device(x, "cuda", constrain_result=False, constrain_body=False)
    assert call.attrs.virtual_device.device_type_int == -1  # ie kInvalidDeviceType
    assert not call.attrs.constrain_body
    assert not call.attrs.constrain_result


if __name__ == "__main__":
    tvm.testing.main()
