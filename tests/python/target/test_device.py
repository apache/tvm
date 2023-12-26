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
from tvm._ffi.runtime_ctypes import Device


@pytest.mark.parametrize(
    "dev_str, expected_device_type, expect_device_id",
    [
        ("cpu", Device.kDLCPU, 0),
        ("cuda", Device.kDLCUDA, 0),
        ("cuda:0", Device.kDLCUDA, 0),
        ("cuda:3", Device.kDLCUDA, 3),
        ("metal:2", Device.kDLMetal, 2),
    ],
)
def test_device(dev_str, expected_device_type, expect_device_id):
    dev = tvm.device(dev_str)
    assert dev.device_type == expected_device_type
    assert dev.device_id == expect_device_id


@pytest.mark.parametrize(
    "dev_type, dev_id, expected_device_type, expect_device_id",
    [
        ("cpu", 0, Device.kDLCPU, 0),
        ("cuda", 0, Device.kDLCUDA, 0),
        (Device.kDLCUDA, 0, Device.kDLCUDA, 0),
        ("cuda", 3, Device.kDLCUDA, 3),
        (Device.kDLMetal, 2, Device.kDLMetal, 2),
    ],
)
def test_device_with_dev_id(dev_type, dev_id, expected_device_type, expect_device_id):
    dev = tvm.device(dev_type=dev_type, dev_id=dev_id)
    assert dev.device_type == expected_device_type
    assert dev.device_id == expect_device_id


@pytest.mark.parametrize(
    "dev_type, dev_id",
    [
        ("cpu:0:0", None),
        ("cpu:?", None),
        ("cpu:", None),
        (Device.kDLCUDA, "?"),
    ],
)
def test_deive_error(dev_type, dev_id):
    with pytest.raises(ValueError):
        dev = tvm.device(dev_type=dev_type, dev_id=dev_id)


if __name__ == "__main__":
    tvm.testing.main()
