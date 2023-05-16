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
import tvm
import tvm.script
from tvm.script import tir as T


def get_before(dtype: str):
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(
            Aptr: T.handle(dtype), Bptr: T.handle(dtype), Dptr: T.handle(dtype)
        ):
            T.func_attr({"global_symbol": "main"})
            A = T.decl_buffer((100,), dtype, data=Aptr)
            B = T.decl_buffer((100,), dtype, data=Bptr)
            D = T.decl_buffer((100,), dtype, data=Dptr)
            C = T.decl_buffer((100,), dtype)
            for i in T.grid(100):
                C[i] = A[i] + B[i]
                D[i] = T.exp(C[i])

    return Before

def promote_f8(f8_dtype: str, promote_dtype: str, v):
    return v

def cast_to_f8(f8_dtype: str, promote_dtype: str, v):
    return v

def get_after_compute_legalize(dtype: str, promote_dtype: str):
    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(
            Aptr: T.handle(dtype), Bptr: T.handle(dtype), Dptr: T.handle(dtype)
        ):
            T.func_attr({"global_symbol": "main"})
            A = T.decl_buffer((100,), dtype, data=Aptr)
            B = T.decl_buffer((100,), dtype, data=Bptr)
            D = T.decl_buffer((100,), dtype, data=Dptr)
            C = T.decl_buffer((100,), promote_dtype)
            for i in T.grid(100):
                C[i] = promote_f8(dtype, promote_dtype, A[i]) + promote_f8(dtype, promote_dtype, B[i])
                D[i] = cast_to_f8(dtype, promote_dtype, T.exp(C[i]))
    return After

def promote_uint8(dtype: str, promote_dtype: str, v):
    return v

def cast_to_uint8(dtype: str, promote_dtype: str, v):
    return v

def get_after_storage_legalize(dtype: str, promote_dtype: str):
    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(Aptr: T.handle("uint8"), Bptr: T.handle("uint8"), Dptr: T.handle("uint8")):
            T.func_attr({"global_symbol": "main"})
            A = T.decl_buffer((100,), "uint8", data=Aptr)
            B = T.decl_buffer((100,), "uint8", data=Bptr)
            D = T.decl_buffer((100,), "uint8", data=Dptr)
            C = T.decl_buffer((100,), promote_dtype)
            for i in T.grid(100):
                C[i] = promote_uint8(dtype, promote_dtype, A[i]) + promote_uint8(dtype, promote_dtype, B[i])
                D[i] = cast_to_uint8(dtype, promote_dtype, T.exp(C[i]))

    return After


def test_fp8_compute_legalize(dtype, promote_dtype):
    before = get_before(dtype)
    expected = get_after_compute_legalize(dtype, promote_dtype)
    # run the transform twice to ensure we can afford to deal
    # with this repeative optimizations
    after = tvm.tir.transform.FP8ComputeLegalize(promote_dtype)(before)
    after = tvm.tir.transform.FP8ComputeLegalize(promote_dtype)(after)

    print(after)
    # tvm.ir.assert_structural_equal(after, expected)


def test_fp8_storage_legalize(dtype, promote_dtype):
    before = get_after_compute_legalize(dtype, promote_dtype)
    after = tvm.tir.transform.FP8StorageLegalize(promote_dtype)(before)
    expected = get_after_storage_legalize()
    # tvm.ir.assert_structural_equal(after, expected)


if __name__ == "__main__":
    test_fp8_compute_legalize("e4m3_float8", "float16")
    test_fp8_compute_legalize("e4m3_float8", "float32")
    test_fp8_compute_legalize("e5m2_float8", "float16")
    test_fp8_compute_legalize("e5m2_float8", "float32")

