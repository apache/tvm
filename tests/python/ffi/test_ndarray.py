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

try:
    import torch
except ImportError:
    torch = None

from tvm import ffi as tvm_ffi
import numpy as np


def test_ndarray_attributes():
    data = np.zeros((10, 8, 4, 2), dtype="int16")
    if not hasattr(data, "__dlpack__"):
        return
    x = tvm_ffi.from_dlpack(data)
    assert isinstance(x, tvm_ffi.NDArray)
    assert x.shape == (10, 8, 4, 2)
    assert x.dtype == tvm_ffi.dtype("int16")
    assert x.device.device_type == tvm_ffi.Device.kDLCPU
    assert x.device.device_id == 0
    x2 = np.from_dlpack(x)
    np.testing.assert_equal(x2, data)


def test_shape_object():
    shape = tvm_ffi.Shape((10, 8, 4, 2))
    assert isinstance(shape, tvm_ffi.Shape)
    assert shape == (10, 8, 4, 2)

    fecho = tvm_ffi.convert(lambda x: x)
    shape2 = fecho(shape)
    assert shape2.__tvm_ffi_object__.same_as(shape.__tvm_ffi_object__)
    assert isinstance(shape2, tvm_ffi.Shape)
    assert isinstance(shape2, tuple)

    shape3 = tvm_ffi.convert(shape)
    assert shape3.__tvm_ffi_object__.same_as(shape.__tvm_ffi_object__)
    assert isinstance(shape3, tvm_ffi.Shape)


@pytest.mark.skipif(torch is None, reason="Torch is not installed")
def test_ndarray_auto_dlpack():
    def check(x, y):
        assert isinstance(y, tvm_ffi.NDArray)
        assert y.shape == (128,)
        assert y.dtype == tvm_ffi.dtype("int64")
        assert y.device.device_type == tvm_ffi.Device.kDLCPU
        assert y.device.device_id == 0
        x2 = torch.from_dlpack(y)
        np.testing.assert_equal(x2.numpy(), x.numpy())

    x = torch.arange(128)
    fecho = tvm_ffi.get_global_func("testing.echo")
    y = fecho(x)
    check(x, y)

    # pass in list of tensors
    y = fecho([x])
    check(x, y[0])
