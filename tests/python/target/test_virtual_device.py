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
import pytest
import tvm
import tvm.testing


def test_make_virtual_device_for_device():
    virtual_device = tvm.target.VirtualDevice(tvm.device("cuda"))
    assert virtual_device.device_type == 2
    # ie kDLCUDA
    assert virtual_device.virtual_device_id == 0
    assert virtual_device.target is None
    assert virtual_device.memory_scope == ""


def test_make_virtual_device_for_device_and_target():
    target = tvm.target.Target("cuda")
    virtual_device = tvm.target.VirtualDevice(tvm.device("cuda"), target)
    assert virtual_device.device_type == 2  # ie kDLCUDA
    assert virtual_device.target == target
    assert virtual_device.memory_scope == ""


def test_make_virtual_device_for_device_target_and_memory_scope():
    target = tvm.target.Target("cuda")
    scope = "local"
    virtual_device = tvm.target.VirtualDevice(tvm.device("cuda"), target, scope)
    assert virtual_device.device_type == 2  # ie kDLCUDA
    assert virtual_device.target == target
    assert virtual_device.memory_scope == scope


if __name__ == "__main__":
    tvm.testing.main()
