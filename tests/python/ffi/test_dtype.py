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
import pickle
import numpy as np
import tvm
import tvm.testing
from tvm import ffi as tvm_ffi


def test_dtype():
    float32 = tvm_ffi.dtype("float32")
    assert float32.__repr__() == "dtype('float32')"
    assert type(float32) == tvm_ffi.dtype
    x = np.array([1, 2, 3], dtype=float32)
    assert x.dtype == float32


@pytest.mark.parametrize(
    "dtype_str, expected_size",
    [
        ("float32", 4),
        ("float32x4", 16),
        ("float8_e5m2x4", 4),
        ("float6_e2m3fnx4", 3),
        ("float4_e2m1fnx4", 2),
        ("uint8", 1),
        ("bool", 1),
    ],
)
def test_dtype_itemsize(dtype_str, expected_size):
    dtype = tvm_ffi.dtype(dtype_str)
    assert dtype.itemsize == expected_size


@pytest.mark.parametrize("dtype_str", ["int32xvscalex4"])
def test_dtype_itemmize_error(dtype_str):
    with pytest.raises(ValueError):
        tvm_ffi.dtype(dtype_str).itemsize


@pytest.mark.parametrize(
    "dtype_str",
    [
        "float32",
        "float32x4",
        "float8_e5m2x4",
        "float6_e2m3fnx4",
        "float4_e2m1fnx4",
        "uint8",
        "bool",
    ],
)
def test_dtype_pickle(dtype_str):
    dtype = tvm_ffi.dtype(dtype_str)
    dtype_pickled = pickle.loads(pickle.dumps(dtype))
    assert dtype_pickled.type_code == dtype.type_code
    assert dtype_pickled.bits == dtype.bits
    assert dtype_pickled.lanes == dtype.lanes


@pytest.mark.parametrize("dtype_str", ["float32", "bool"])
def test_dtype_with_lanes(dtype_str):
    dtype = tvm_ffi.dtype(dtype_str)
    dtype_with_lanes = dtype.with_lanes(4)
    assert dtype_with_lanes.type_code == dtype.type_code
    assert dtype_with_lanes.bits == dtype.bits
    assert dtype_with_lanes.lanes == 4


if __name__ == "__main__":
    tvm.testing.main()
