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
"""Test retrieving and applying memory scope constraints to PrimFuncs"""
import tvm
import tvm.testing
from tvm import tir
from tvm import relay
from tvm.script import tir as T


@T.prim_func
def gem(a: T.handle, b: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], scope="scopeA")
    B = T.match_buffer(b, [128, 128], scope="scopeA")
    C = T.match_buffer(c, [128, 128], scope="scopeB")
    D = T.match_buffer(d, [128, 128], scope="scopeC")

    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                D[vi, vj] = C[vi, vj]
            D[vi, vj] = D[vi, vj] + A[vi, vk] * B[vj, vk]


gem_ty = relay.FuncType(
    [
        relay.TupleType(
            [
                relay.TensorType((128, 128), "float32"),
                relay.TensorType((128, 128), "float32"),
            ]
        ),
        relay.TensorType((128, 128), "float32"),
    ],
    relay.TensorType((128, 128), "float32"),
)


def test_get_prim_func_arg_and_result_constraints():
    scopes = tir.analysis.get_prim_func_arg_and_result_memory_constraints(gem, gem_ty)
    assert [x for x in scopes] == ["scopeA", "scopeB", "scopeC"]


def test_apply_prim_func_arg_and_result_memory_constraints():
    rewritten = tir.analysis.apply_prim_func_arg_and_result_memory_constraints(
        gem, gem_ty, ["scopeX", "scopeY", "scopeZ"]
    )
    scopes = tir.analysis.get_prim_func_arg_and_result_memory_constraints(rewritten, gem_ty)
    assert [x for x in scopes] == ["scopeX", "scopeY", "scopeZ"]


if __name__ == "__main__":
    tvm.testing.main()
