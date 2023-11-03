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
import numpy
import tvm
import tvm.testing
from tvm.script import tir as T

# This numpy array is used to test the comparison between the global objects and the
# `tvm.script.tir` submodule.
np_array = numpy.array([0, 1, 2, 3])


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


def test_multi_element_array_in_outmost_namespace():
    func = matmul
    rt_func = tvm.script.from_source(func.script())
    tvm.ir.assert_structural_equal(func, rt_func)


def test_different_dtype_assignment_to_var():
    @T.prim_func
    def test_case():
        a = T.alloc_buffer((10, 10), dtype="int8")

    @T.prim_func
    def func_ref():
        a = T.alloc_buffer([10, 10], dtype="int8")
        T.evaluate(0)

    tvm.ir.assert_structural_equal(
        test_case.with_attr("global_symbol", "main"), func_ref.with_attr("global_symbol", "main")
    )


def test_var_capturing_order():
    b = 2

    @T.prim_func
    def test_case():
        k: T.int32 = b

    @T.prim_func
    def func_ref():
        k: T.int32 = 2
        T.evaluate(0)

    tvm.ir.assert_structural_equal(
        test_case.with_attr("global_symbol", "main"), func_ref.with_attr("global_symbol", "main")
    )


def test_tir_buffer_region_extent_correct_dtype():
    @T.prim_func
    def func(A: T.Buffer((T.int64(16), T.int64(1)), "float32")):
        for i in T.grid(T.int64(16)):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                T.reads(A[vi, T.int64(0) : T.int64(1)])
                T.evaluate(0)

    assert func.body.block.body.body.block.reads[0].region[0].extent.dtype == "int64"


if __name__ == "__main__":
    tvm.testing.main()
