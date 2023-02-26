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

import tvm
from tvm import tir
from tvm.script import tir as T


@T.prim_func
def scale_by_two(a: T.Buffer((128,), "int8"), c: T.Buffer((128,), "int8")):
    for i in T.serial(128):
        with T.block("C"):
            c[i] = a[i] * T.int8(2)


@T.prim_func
def scale_by_two_three(a: T.Buffer((128,), "int8"), c: T.Buffer((128,), "int8")):
    B = T.alloc_buffer([128], dtype="int8", scope="global.vtcm")
    for i in T.serial(128):
        with T.block("B"):
            B[i] = a[i] * T.int8(2)
    for i in T.serial(128):
        with T.block("C"):
            c[i] = B[i] * T.int8(3)


@pytest.mark.parametrize("primFunc,size", [(scale_by_two, 128), (scale_by_two_three, 256)])
def test_scale_by(primFunc, size):
    """Test calculate allocated bytes per scope"""
    mod = tvm.IRModule.from_expr(primFunc.with_attr("global_symbol", "main"))
    sch = tir.Schedule(mod, debug_mask="all")
    block_c = sch.get_block("C")
    (flat,) = sch.get_loops(block_c)
    cache_block = sch.cache_read(block_c, 0, storage_scope="global.vtcm")
    sch.compute_at(cache_block, flat)

    mod = sch.mod
    mod = tvm.tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
    sizes = tvm.tir.analysis.calculate_allocated_bytes(mod["main"])
    assert sizes.get("global.vtcm", 0) == size


@T.prim_func
def matmul_mix_scope(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], scope="global")
    B = T.match_buffer(b, [128, 128], scope="global")
    C = T.match_buffer(c, [128, 128], scope="global")
    A_allocated = T.alloc_buffer([128, 128], dtype="float32", scope="global.texture")
    B_allocated = T.alloc_buffer([128, 128], dtype="float32", scope="global.texture")
    C_allocated = T.alloc_buffer([128, 128], dtype="float32", scope="global")

    for i, j in T.grid(128, 128):
        with T.block("A.allocated"):
            A_allocated[i, j] = A[i, j]
    for i, j in T.grid(128, 128):
        with T.block("B.allocated"):
            B_allocated[i, j] = B[i, j]

    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C_allocated[vi, vj] = 0.0
            C_allocated[vi, vj] = C[vi, vj] + A_allocated[vi, vk] * B_allocated[vj, vk]

    for i, j in T.grid(128, 128):
        with T.block("C"):
            C[i, j] = C_allocated[i, j]


@pytest.mark.parametrize(
    "scope,size", [("global", 65536), ("global.texture", 131072), ("global.texture-nhwc", 0)]
)
def test_matmul_mix_scope(scope, size):
    """Test calculate allocated bytes per scope"""
    mod = tvm.IRModule({"main": matmul_mix_scope})
    mod = tvm.tir.transform.LowerInitBlock()(mod)
    mod = tvm.tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
    sizes = tvm.tir.analysis.calculate_allocated_bytes(mod["main"])
    assert sizes.get(scope, 0) == size


if __name__ == "__main__":
    tvm.testing.main()
