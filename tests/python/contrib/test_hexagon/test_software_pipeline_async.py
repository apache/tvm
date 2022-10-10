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

import sys
import pytest
import numpy as np

import tvm
from tvm import tir
from tvm.contrib.hexagon.session import Session
from tvm.script import tir as T

from .infrastructure import get_hexagon_target

outer = tvm.testing.parameter(8, 16)
inner = tvm.testing.parameter(64, 128)
dtype = tvm.testing.parameter("uint8", "float16")
scope = tvm.testing.parameter("global", "global.vtcm")
# TODO(Straw) Add back "cache_write" schedule type once we have upstreamed
# buffer dependency analysis in InjectSoftwarePipeline pass
# to insert approprite TIR "wait" attributes for this schedule
sched = tvm.testing.parameter("cache_read", "cache_read_write")


@tvm.testing.fixture
def compute(outer, inner, dtype):
    @T.prim_func
    def plus_one_primfunc(A: T.Buffer[(outer, inner), dtype], B: T.Buffer[(outer, inner), dtype]):
        for i in T.serial(outer):
            for j in T.serial(inner):
                with T.block("compute"):
                    with T.block():
                        B[i, j] = A[i, j] + T.cast(1, dtype)

    def plus_one_ref(a):
        return a + 1

    return plus_one_primfunc, plus_one_ref


@tvm.testing.fixture
def schedule(compute, sched, scope):
    sch = tir.Schedule(compute[0])

    compute_block = sch.get_block("compute")
    i, _ = sch.get_loops(compute_block)

    if sched == "cache_read":
        cache_read_block = sch.cache_read(compute_block, 0, scope)
        sch.compute_at(cache_read_block, i)
        sch.annotate(i, "software_pipeline_stage", [0, 1])
        sch.annotate(i, "software_pipeline_order", [0, 1])
        sch.annotate(i, "software_pipeline_async_stages", [0])
    elif sched == "cache_write":
        cache_write_block = sch.cache_write(compute_block, 0, scope)
        sch.reverse_compute_at(cache_write_block, i)
        sch.annotate(i, "software_pipeline_stage", [0, 1])
        sch.annotate(i, "software_pipeline_order", [0, 1])
        sch.annotate(i, "software_pipeline_async_stages", [1])
    elif sched == "cache_read_write":
        cache_read_block = sch.cache_read(compute_block, 0, scope)
        sch.compute_at(cache_read_block, i)
        cache_write_block = sch.cache_write(compute_block, 0, scope)
        sch.reverse_compute_at(cache_write_block, i)
        sch.annotate(i, "software_pipeline_stage", [0, 1, 2])
        sch.annotate(i, "software_pipeline_order", [0, 1, 2])
        sch.annotate(i, "software_pipeline_async_stages", [0, 2])

    return sch


@tvm.testing.requires_hexagon
def test_async_software_pipeline(hexagon_launcher, compute, schedule, outer, inner, dtype, scope):
    sch = schedule

    a_np = np.random.uniform(low=0, high=128, size=(outer, inner)).astype(dtype)
    b_np = np.random.uniform(low=0, high=128, size=(outer, inner)).astype(dtype)
    ref = compute[1](a_np)

    with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
        func = tvm.build(sch.mod["main"], target=get_hexagon_target("v68"))

    with hexagon_launcher.start_session() as hexagon_session:
        dev = hexagon_session.device
        a = tvm.nd.array(a_np, device=dev)
        b = tvm.nd.array(b_np, device=dev)
        mod = hexagon_session.load_module(func)
        mod(a, b)

        if "int" in dtype:
            np.testing.assert_equal(b.numpy(), ref)
        else:
            np.testing.assert_allclose(b.numpy(), ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
