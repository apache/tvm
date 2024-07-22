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

"""Test contrib.tvmjs"""

import tempfile

import numpy as np
import pytest

import tvm.testing
from tvm.contrib import tvmjs

dtype = tvm.testing.parameter(
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "float8_e4m3fn",
    "float8_e5m2",
)


def test_save_load_float8(dtype):
    if "float8" in dtype or "bfloat16" in dtype:
        ml_dtypes = pytest.importorskip("ml_dtypes")
        np_dtype = np.dtype(getattr(ml_dtypes, dtype))
    else:
        np_dtype = np.dtype(dtype)

    arr = np.arange(16, dtype=np_dtype)

    with tempfile.TemporaryDirectory(prefix="tvm_") as temp_dir:
        tvmjs.dump_ndarray_cache({"arr": arr}, temp_dir)
        cache, _ = tvmjs.load_ndarray_cache(temp_dir, tvm.cpu())

    after_roundtrip = cache["arr"].numpy()

    np.testing.assert_array_equal(arr, after_roundtrip)


if __name__ == "__main__":
    tvm.testing.main()
