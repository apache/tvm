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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import tvm
from tvm.script import tir as T

from tvm.meta_schedule.postproc import RewriteParallelVectorizeUnroll
from tvm.tir.schedule import Schedule

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,not-callable,misplaced-comparison-constant
# fmt: off

@tvm.script.ir_module
class Move_PUV:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, [1024, 1024, 1024], dtype="float32")
        B = T.match_buffer(b, [1024, 1024, 1024], dtype="float32")
        # body
        with T.block("root"):
            T.block_attr({"meta_schedule.parallel":128, "meta_schedule.vectorize":32})
            for i0, j0, i1, j1, k0, i2, j2, k1 in T.grid(128, 64, 4, 4, 64, 4, 8, 32):
                with T.block("move"):
                    vi = T.axis.spatial(1024, i0 * 16 + i1 * 4 + i2)
                    vj = T.axis.spatial(1024, j0 * 32 + j1 * 8 + j2)
                    vk = T.axis.spatial(1024, k0 * 32 + k1)
                    T.where((i0 * 4 + i1) * 4 + i2 < 1024 and (j0 * 4 + j1) * 8 + j2 < 1024 and k0 * 32 + k1 < 1024)
                    T.reads([A[vi, vj, vk]])
                    T.writes([B[vi, vj, vk]])
                    B[vi, vj, vk] = A[vi, vj, vk]


@T.prim_func
def Move_PUV0(a: T.handle, b: T.handle) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main"})
    A = T.match_buffer(a, [1024, 1024, 1024], dtype="float32")
    B = T.match_buffer(b, [1024, 1024, 1024], dtype="float32")
    # body
    with T.block("root"):
        for i0_j0_fused in T.parallel(0, 8192):
            for i1, j1, k0, i2, j2 in T.grid(4, 4, 64, 4, 8):
                for k1_fused in T.vectorized(0, 32):
                    with T.block("move"):
                        vi = T.axis.spatial(1024, i0_j0_fused // 64 * 16 + i1 * 4 + i2)
                        vj = T.axis.spatial(1024, i0_j0_fused % 64 * 32 + j1 * 8 + j2)
                        vk = T.axis.spatial(1024, k0 * 32 + k1_fused)
                        T.where(
                            i0_j0_fused // 64 * 16 + i1 * 4 + i2 < 1024
                            and i0_j0_fused % 64 * 32 + j1 * 8 + j2 < 1024
                            and k0 * 32 + k1_fused < 1024
                        )
                        T.reads([A[vi, vj, vk]])
                        T.writes([B[vi, vj, vk]])
                        B[vi, vj, vk] = A[vi, vj, vk]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,not-callable


def test_meta_schedule_postproc_rewrite_parallel_unroll_vectorize():
    postproc = RewriteParallelVectorizeUnroll()
    sch = Schedule(Move_PUV)
    assert postproc.apply(sch)
    print(sch.mod["main"].script())
    mod = tvm.tir.transform.Simplify()(sch.mod)
    tvm.ir.assert_structural_equal(mod["main"], Move_PUV0)


if __name__ == "__main__":
    test_meta_schedule_postproc_rewrite_parallel_unroll_vectorize()
