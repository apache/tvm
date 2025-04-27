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

from tvm import ffi as tvm_ffi
import numpy as np


def test_ndarray_attributes():
    data = np.zeros((10, 8, 4, 2), dtype="int16")
    x = tvm_ffi.from_dlpack(data)
    assert isinstance(x, tvm_ffi.NDArray)
    assert x.shape == (10, 8, 4, 2)
    assert x.dtype == tvm_ffi.dtype("int16")
    assert x.device.device_type == tvm_ffi.Device.kDLCPU
    assert x.device.device_id == 0
    x2 = np.from_dlpack(x)
    np.testing.assert_equal(x2, data)
