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
import ml_dtypes
import tvm.tir as tir
from tvm import te
from tvm.script import tir as T


def fp8_unary(dtype: str):
    @T.prim_func
    def func(
        a: T.handle,
        b: T.handle,
        a_add_b: T.handle,
        a_sub_b: T.handle,
        a_mul_b: T.handle,
        a_div_b: T.handle,
    ) -> None:
        A = T.match_buffer(a, [128], dtype=dtype)
        B = T.match_buffer(b, [128], dtype=dtype)
        A_add_B = T.match_buffer(a_add_b, [128], dtype=dtype)
        A_sub_B = T.match_buffer(a_sub_b, [128], dtype=dtype)
        A_mul_B = T.match_buffer(a_mul_b, [128], dtype=dtype)
        A_div_B = T.match_buffer(a_div_b, [128], dtype=dtype)
        for i in range(128):
            with T.block("fp8_unary"):
                vi = T.axis.spatial(128, i)
                A_add_B[vi] = A[vi] + B[vi]
                A_sub_B[vi] = A[vi] - B[vi]
                A_mul_B[vi] = A[vi] * B[vi]
                A_div_B[vi] = A[vi] / B[vi]
    return func


# @tvm.testing.fixture
def test_create_nv_fp8_nd_array(dtype):
    x = np.random.rand(128, 128).astype(dtype[0])
    x_nd = tvm.nd.array(x)
    assert x_nd.dtype == dtype[1]
    print(x_nd)


def test_fp8_unary_op(dtype):
    f = fp8_unary(dtype[1])
    print(f)


def test_nv_fp8_buffer(dtype):
    m = te.size_var("m")
    n = te.size_var("n")
    A = tvm.tir.decl_buffer((m, n), dtype[1])
    assert A.dtype == dtype[1]


if __name__ == "__main__":
    test_create_nv_fp8_nd_array((ml_dtypes.float8_e4m3fn, "e4m3_float8"))
    test_create_nv_fp8_nd_array((ml_dtypes.float8_e5m2, "e5m2_float8"))
    test_nv_fp8_buffer((ml_dtypes.float8_e4m3fn, "e4m3_float8"))
    test_fp8_unary_op((ml_dtypes.float8_e4m3fn, "e4m3_float8"))
