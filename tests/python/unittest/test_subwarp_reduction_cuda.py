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
import tvm.testing
import numpy as np
from tvm.script import tir as T


@T.prim_func
def reduce(a: T.handle, b: T.handle, d1: T.int32, d2: T.int32, d3: T.int32) -> None:
    A = T.match_buffer(a, [1, d1, d2, d3])
    B = T.match_buffer(b, [1, d1, d2])

    for i, j, k, l in T.grid(1, d1, d2, d3):
        with T.block("reduce"):
            vi, vj, vk, vl = T.axis.remap("SSSR", [i, j, k, l])
            with T.init():
                B[vi, vj, vk] = 0.0
            B[vi, vj, vk] = B[vi, vj, vk] + A[vi, vj, vk, vl]


@T.prim_func
def reduce_max(a: T.handle, b: T.handle, d1: T.int32, d2: T.int32, d3: T.int32) -> None:
    A = T.match_buffer(a, [1, d1, d2, d3])
    B = T.match_buffer(b, [1, d1, d2])

    for i, j, k, l in T.grid(1, d1, d2, d3):
        with T.block("reduce"):
            vi, vj, vk, vl = T.axis.remap("SSSR", [i, j, k, l])
            with T.init():
                B[vi, vj, vk] = T.float32(-3.4028234663852886e38)
            B[vi, vj, vk] = T.max(B[vi, vj, vk], A[vi, vj, vk, vl])


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_cuda_subwarp_reduction():
    def check_sum(d1: int, d2: int, d3: int):
        _, _, _d1, _d2, _d3 = reduce.params
        mod = reduce.specialize({_d1: d1, _d2: d2, _d3: d3})
        sch = tvm.tir.Schedule(mod)
        blk = sch.get_block("reduce")
        i, j, k, l = sch.get_loops(blk)
        sch.bind(i, "blockIdx.x")
        sch.bind(j, "threadIdx.z")
        sch.bind(k, "threadIdx.y")
        sch.bind(l, "threadIdx.x")
        f = tvm.build(sch.mod["main"], target="cuda")

        # prepare input and output array
        a_np = np.random.rand(1, d1, d2, d3).astype("float32")
        b_np = a_np.sum(axis=-1).astype("float32")
        a = tvm.nd.array(a_np, tvm.cuda(0))
        b = tvm.nd.array(np.zeros_like(b_np), tvm.cuda(0))

        # launch kernel
        f(a, b)
        tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-6, atol=1e-6)

    def check_max(d1: int, d2: int, d3: int):
        _, _, _d1, _d2, _d3 = reduce_max.params
        mod = reduce_max.specialize({_d1: d1, _d2: d2, _d3: d3})
        sch = tvm.tir.Schedule(mod)
        blk = sch.get_block("reduce")
        i, j, k, l = sch.get_loops(blk)
        sch.bind(i, "blockIdx.x")
        sch.bind(j, "threadIdx.z")
        sch.bind(k, "threadIdx.y")
        sch.bind(l, "threadIdx.x")
        f = tvm.build(sch.mod["main"], target="cuda")

        # prepare input and output array
        a_np = -np.random.rand(1, d1, d2, d3).astype("float32")
        b_np = a_np.max(axis=-1).astype("float32")
        a = tvm.nd.array(a_np, tvm.cuda(0))
        b = tvm.nd.array(np.zeros_like(b_np), tvm.cuda(0))

        # launch kernel
        f(a, b)
        tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-6, atol=1e-6)

    for d1 in range(1, 5):
        for d2 in range(1, 5):
            for d3 in range(2, 33):
                check_sum(d1, d2, d3)
                check_max(d1, d2, d3)


if __name__ == "__main__":
    test_cuda_subwarp_reduction()
