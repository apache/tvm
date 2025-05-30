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

import tvm
import tvm.testing
import tvm.tir as tir
from tvm import te
from tvm.script import tir as T

try:
    from ml_dtypes import float4_e2m1fn
except ImportError:
    float4_e2m1fn = None


np_dtype, dtype_str = tvm.testing.parameters((float4_e2m1fn, "float4_e2m1fn"))


def test_create_nv_fp4_nd_array(np_dtype, dtype_str):
    if np_dtype is None:
        """Skip test if ml_dtypes is not installed"""
        return
    x = np.random.rand(128, 128).astype(np_dtype)
    x_nd = tvm.nd.array(x)
    assert x_nd.dtype == dtype_str
    np.testing.assert_equal(x_nd.numpy(), x)


def test_nv_fp4_buffer(np_dtype, dtype_str):
    m = te.size_var("m")
    n = te.size_var("n")
    A = tvm.tir.decl_buffer((m, n), dtype_str)
    assert A.dtype == dtype_str


if __name__ == "__main__":
    tvm.testing.main()
