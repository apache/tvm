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
import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T


@pytest.mark.parametrize("dtype", ["float8_e3m4", "float8_e4m3", "float8_e4m3fn", "float8_e5m2"])
def test_fp8_float32_roundtrip(dtype):
    if not tvm_tirx.testing.device_enabled("llvm"):
        pytest.skip("LLVM is not enabled")
    ml_dtypes = pytest.importorskip("ml_dtypes")
    if not hasattr(ml_dtypes, dtype):
        pytest.skip(f"ml_dtypes does not provide {dtype}")
    np_dtype = getattr(ml_dtypes, dtype)

    @T.function
    def roundtrip(
        A: T.Tensor((128,), dtype),
        intermediate: T.Tensor((128,), "float32"),
        B: T.Tensor((128,), dtype),
    ):
        for i in range(128):
            intermediate[i] = A[i]
            B[i] = intermediate[i]

    source = np.random.default_rng(0).standard_normal(128).astype(np_dtype)
    tensors = [
        tvm_tirx.runtime.tensor(source),
        tvm_tirx.runtime.tensor(np.zeros(128, dtype="float32")),
        tvm_tirx.runtime.tensor(np.zeros(128, dtype=np_dtype)),
    ]
    tvm_tirx.tirx.compile(roundtrip, target="llvm")(*tensors)
    np.testing.assert_equal(tensors[2].numpy(), source.astype("float32").astype(np_dtype))
