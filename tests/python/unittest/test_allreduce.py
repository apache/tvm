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

import pytest


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


def generate_param_sets():
    for d1 in range(1, 5):
        for d2 in range(1, 5):
            for d3 in [2, 4, 8, 12, 16, 32, 48, 64, 100, 128, 201, 256, 512, 1024]:
                if d1 * d2 * d3 < 1024:
                    yield (d1, d2, d3)


dims = tvm.testing.parameter(*generate_param_sets())


@tvm.testing.parametrize_targets("cuda", "metal")
def test_allreduce_sum(dims, target, dev):
    d1, d2, d3 = dims
    _, _, _d1, _d2, _d3 = reduce.params
    mod = reduce.specialize({_d1: d1, _d2: d2, _d3: d3})
    sch = tvm.tir.Schedule(mod)
    blk = sch.get_block("reduce")
    i, j, k, l = sch.get_loops(blk)
    sch.bind(i, "blockIdx.x")
    sch.bind(j, "threadIdx.z")
    sch.bind(k, "threadIdx.y")
    sch.bind(l, "threadIdx.x")
    f = tvm.build(sch.mod["main"], target=target)

    # prepare input and output array
    a_np = np.random.rand(1, d1, d2, d3).astype("float32")
    b_np = a_np.sum(axis=-1).astype("float32")
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(np.zeros_like(b_np), dev)

    # launch kernel
    f(a, b)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-6, atol=1e-6)


define_metal_compile_callback = tvm.testing.parameter(True, False)


@pytest.fixture
def optional_metal_compile_callback(define_metal_compile_callback):
    name = "tvm_callback_metal_compile"
    cached = tvm.get_global_func(name, allow_missing=True)

    if define_metal_compile_callback:

        @tvm.register_func(name, override=True)
        def compile_metal(src, target):
            return tvm.contrib.xcode.compile_metal(src, sdk="macosx")

    yield

    if define_metal_compile_callback:
        if cached is None:
            tvm._ffi.registry.remove_global_func(name)
        else:
            tvm.register_func(name, cached, override=True)


@tvm.testing.requires_metal(support_required="compile-only")
def test_allreduce_sum_compile(optional_metal_compile_callback):
    # Disable the parametrization over dims, at least for now
    dims = (1, 1, 2)
    target = "metal"

    d1, d2, d3 = dims
    _, _, _d1, _d2, _d3 = reduce.params
    mod = reduce.specialize({_d1: d1, _d2: d2, _d3: d3})
    sch = tvm.tir.Schedule(mod)
    blk = sch.get_block("reduce")
    i, j, k, l = sch.get_loops(blk)
    sch.bind(i, "blockIdx.x")
    sch.bind(j, "threadIdx.z")
    sch.bind(k, "threadIdx.y")
    sch.bind(l, "threadIdx.x")
    tvm.build(sch.mod["main"], target=target)


@tvm.testing.parametrize_targets("cuda", "metal")
def test_allreduce_max(dims, target, dev):
    d1, d2, d3 = dims
    _, _, _d1, _d2, _d3 = reduce_max.params
    mod = reduce_max.specialize({_d1: d1, _d2: d2, _d3: d3})
    sch = tvm.tir.Schedule(mod)
    blk = sch.get_block("reduce")
    i, j, k, l = sch.get_loops(blk)
    sch.bind(i, "blockIdx.x")
    sch.bind(j, "threadIdx.z")
    sch.bind(k, "threadIdx.y")
    sch.bind(l, "threadIdx.x")
    f = tvm.build(sch.mod["main"], target=target)

    # prepare input and output array
    a_np = -np.random.rand(1, d1, d2, d3).astype("float32")
    b_np = a_np.max(axis=-1).astype("float32")
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(np.zeros_like(b_np), dev)

    # launch kernel
    f(a, b)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    tvm.testing.main()
