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
    from ml_dtypes import float8_e4m3fn as e4m3_float8, float8_e5m2 as e5m2_float8
except ImportError:
    e4m3_float8, e5m2_float8 = None, None


def fp8_unary(dtype: str):
    @T.prim_func
    def func(
        a: T.handle,
        b: T.handle,
        a_add_b: T.handle,
        a_sub_b: T.handle,
        a_mul_b: T.handle,
        a_fp32: T.handle,
        a_roundtrip: T.handle,
    ) -> None:
        A = T.match_buffer(a, [128], dtype=dtype)
        B = T.match_buffer(b, [128], dtype=dtype)
        A_add_B = T.match_buffer(a_add_b, [128], dtype=dtype)
        A_sub_B = T.match_buffer(a_sub_b, [128], dtype=dtype)
        A_mul_B = T.match_buffer(a_mul_b, [128], dtype=dtype)
        A_fp32 = T.match_buffer(a_fp32, [128], dtype="float32")
        A_roundtrip = T.match_buffer(a_roundtrip, [128], dtype=dtype)
        for i in range(128):
            with T.block("fp8_unary"):
                vi = T.axis.spatial(128, i)
                A_add_B[vi] = A[vi] + B[vi]
                A_sub_B[vi] = A[vi] - B[vi]
                A_mul_B[vi] = A[vi] * B[vi]
                A_fp32[vi] = A[vi]
                A_roundtrip[vi] = A_fp32[vi]

    return func


np_dtype, dtype_str = tvm.testing.parameters(
    (e4m3_float8, "e4m3_float8"), (e5m2_float8, "e5m2_float8")
)


def test_create_nv_fp8_nd_array(np_dtype, dtype_str):
    if np_dtype is None:
        """Skip test if ml_dtypes is not installed"""
        return
    x = np.random.rand(128, 128).astype(np_dtype)
    x_nd = tvm.nd.array(x)
    assert x_nd.dtype == dtype_str


def test_fp8_unary_op(np_dtype, dtype_str):
    func = fp8_unary(dtype_str)
    if not tvm.testing.device_enabled("llvm"):
        return
    if np_dtype is None:
        """Skip test if ml_dtypes is not installed"""
        return

    f = tvm.build(func, target="llvm")
    a = np.random.randn(128).astype(np_dtype)
    b = np.random.randn(128).astype(np_dtype)
    a_add_b = np.zeros(128).astype(np_dtype)
    a_sub_b = np.zeros(128).astype(np_dtype)
    a_mul_b = np.zeros(128).astype(np_dtype)
    a_fp32 = np.zeros(128).astype(np.float32)
    a_roundtrip = np.zeros(128).astype(np_dtype)
    args = list(
        map(lambda _: tvm.nd.array(_), [a, b, a_add_b, a_sub_b, a_mul_b, a_fp32, a_roundtrip])
    )
    f(*args)


def test_nv_fp8_buffer(np_dtype, dtype_str):
    m = te.size_var("m")
    n = te.size_var("n")
    A = tvm.tir.decl_buffer((m, n), dtype_str)
    assert A.dtype == dtype_str


if __name__ == "__main__":
    tvm.testing.main()
