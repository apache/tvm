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
"""Unit tests for tensor helpers."""
import tvm
from tvm import relay
import pytest


def test_device_copy_via_string():
    x = relay.var("x")
    call = relay.op.device_copy(x, "cuda", "cpu")
    assert isinstance(call, relay.Call)
    assert len(call.args) == 1
    assert call.args[0] == x
    assert call.attrs.src_se_scope.device_type_int == 2  # ie kDLCUDA
    assert call.attrs.src_se_scope.virtual_device_id == 0
    assert call.attrs.src_se_scope.target is None
    assert call.attrs.src_se_scope.memory_scope == ""
    assert call.attrs.dst_se_scope.device_type_int == 1  # ie kDLCPU
    assert call.attrs.dst_se_scope.virtual_device_id == 0
    assert call.attrs.dst_se_scope.target is None
    assert call.attrs.dst_se_scope.memory_scope == ""


def test_device_copy_via_device():
    x = relay.var("x")
    call = relay.op.device_copy(x, tvm.device("cuda"), tvm.device("cpu"))
    assert isinstance(call, relay.Call)
    assert len(call.args) == 1
    assert call.args[0] == x
    assert call.attrs.src_se_scope.device_type_int == 2  # ie kDLCUDA
    assert call.attrs.dst_se_scope.device_type_int == 1  # ie kDLCPU


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__] + sys.argv[1:]))
