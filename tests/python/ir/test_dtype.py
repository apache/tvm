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
"""Test data type related API"""
import tvm
from tvm import DataType
import tvm.testing
import pytest


@pytest.mark.parametrize(
    "dtype_str, expected_size",
    [("float32", 4), ("float32x4", 16), ("e5m2_float8x4", 4), ("uint8", 1)],
)
def test_dtype_itemsize(dtype_str, expected_size):
    dtype = DataType(dtype_str)
    assert dtype.itemsize() == expected_size


@pytest.mark.parametrize("dtype_str", [("int32xvscalex4")])
def test_dtype_itemmize_error(dtype_str):
    with pytest.raises(ValueError):
        size = DataType(dtype_str).itemsize()


if __name__ == "__main__":
    tvm.testing.main()
