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
"""Async software pipeline tests."""

import numpy as np

import tvm
from tvm import tir
from tvm.script import tir as T

from .infrastructure import get_hexagon_target


def compute(comp_type, outer, inner, dtype):
    """Generate compute function."""
    if comp_type == "single_input":

        @T.prim_func
        def a_plus_1_primfunc(
            a_buffer: T.Buffer[(outer, inner), dtype], out: T.Buffer[(outer, inner), dtype]
        ):
            for i in T.serial(outer):
                for j in T.serial(inner):
                    with T.block("compute"):
                        with T.block():
                            out[i, j] = a_buffer[i, j] + T.cast(1, dtype)

        return a_plus_1_primfunc
    else:

        @T.prim_func
        def a_plus_b_plus_1_primfunc(
            a_buffer: T.Buffer[(outer, inner), dtype],
            b_buffer: T.Buffer[(outer, inner), dtype],
            out: T.Buffer[(outer, inner), dtype],
        ):
            for i in T.serial(outer):
                for j in T.serial(inner):
                    with T.block("compute"):
                        with T.block():
                            out[i, j] = a_buffer[i, j] + b_buffer[i, j] + T.cast(1, dtype)

        return a_plus_b_plus_1_primfunc


class TestAsyncSoftwarePipeline:
    """Async software pipeline test class."""

    outer = tvm.testing.parameter(8, 16)
    inner = tvm.testing.parameter(64, 128)
    dtype = tvm.testing.parameter("uint8", "float16")
    scope = tvm.testing.parameter("global", "global.vtcm")
    # TODO(Joseph) Turn on "multi_input_diffQ" compute type once we have upstreamed
    # changes in the InjectSoftwarePipeline pass to alleviate this restriction:
    # 'a_buffer dependency on multiple async stages is not supported'
    comp_type = tvm.testing.parameter("single_input", "multi_input_sameQ")
    # TODO(Straw) Add back "cache_write" schedule type once we have upstreamed
    # buffer dependency analysis in InjectSoftwarePipeline pass
    # to insert approprite TIR "wait" attributes for this schedule
    sched_type = tvm.testing.parameter("cache_read", "cache_read_write")

    @tvm.testing.fixture
    def data(self, comp_type, outer, inner, dtype):
        out_np = np.random.uniform(low=0, high=128, size=(outer, inner)).astype(dtype)
        a_np = np.random.uniform(low=0, high=128, size=(outer, inner)).astype(dtype)
        if comp_type == "single_input":
            return out_np, a_np
        else:
            b_np = np.random.uniform(low=0, high=128, size=(outer, inner)).astype(dtype)
            return out_np, a_np, b_np

    @tvm.testing.fixture
    def verify(self, dtype):
        def check(out, ref):
            if "int" in dtype:
                np.testing.assert_equal(out.numpy(), ref)
            else:
                np.testing.assert_allclose(out.numpy(), ref, rtol=1e-3, atol=1e-3)

        return check

    @tvm.testing.fixture
    def reference(self, comp_type):
        """Returns reference data."""
        if comp_type == "single_input":

            def a_plus_1_ref(a):
                return a + 1

            return a_plus_1_ref
        else:

            def a_plus_b_plus_1_ref(a, b):
                return a + b + 1

            return a_plus_b_plus_1_ref

    @tvm.testing.fixture
    def schedule(self, comp_type, sched_type, outer, inner, dtype, scope):
        """Generate schedule."""
        sch = tir.Schedule(compute(comp_type, outer, inner, dtype))

        compute_block = sch.get_block("compute")
        i, _ = sch.get_loops(compute_block)

        if "read" in sched_type:
            cache_read_a = sch.cache_read(compute_block, 0, scope)
            sch.compute_at(cache_read_a, i)

            if "multi_input" in comp_type:
                cache_read_b = sch.cache_read(compute_block, 1, scope)
                sch.compute_at(cache_read_b, i)

        if "write" in sched_type:
            cache_write_out = sch.cache_write(compute_block, 0, scope)
            sch.reverse_compute_at(cache_write_out, i)

        if "read" in sched_type and "write" in sched_type:
            if comp_type == "single_input":
                sch.annotate(i, "software_pipeline_stage", [0, 1, 2])
                sch.annotate(i, "software_pipeline_order", [0, 1, 2])
                sch.annotate(i, "software_pipeline_async_stages", [0, 2])
            elif comp_type == "multi_input_sameQ":
                sch.annotate(i, "software_pipeline_stage", [0, 0, 1, 2])
                sch.annotate(i, "software_pipeline_order", [0, 1, 2, 3])
                sch.annotate(i, "software_pipeline_async_stages", [0, 2])
            elif comp_type == "multi_input_diffQ":
                sch.annotate(i, "software_pipeline_stage", [0, 1, 2, 3])
                sch.annotate(i, "software_pipeline_order", [0, 1, 2, 3])
                sch.annotate(i, "software_pipeline_async_stages", [0, 1, 2])

        elif "read" in sched_type:
            if comp_type == "single_input":
                sch.annotate(i, "software_pipeline_stage", [0, 1])
                sch.annotate(i, "software_pipeline_order", [0, 1])
                sch.annotate(i, "software_pipeline_async_stages", [0])
            elif comp_type == "multi_input_sameQ":
                sch.annotate(i, "software_pipeline_stage", [0, 0, 1])
                sch.annotate(i, "software_pipeline_order", [0, 1, 2])
                sch.annotate(i, "software_pipeline_async_stages", [0])
            elif comp_type == "multi_input_diffQ":
                sch.annotate(i, "software_pipeline_stage", [0, 1, 2])
                sch.annotate(i, "software_pipeline_order", [0, 1, 2])
                sch.annotate(i, "software_pipeline_async_stages", [0, 1])

        elif "write" in sched_type:
            sch.annotate(i, "software_pipeline_stage", [0, 1])
            sch.annotate(i, "software_pipeline_order", [0, 1])
            sch.annotate(i, "software_pipeline_async_stages", [1])

        return sch

    @tvm.testing.requires_hexagon
    def test_async_software_pipeline(
        self, hexagon_launcher, comp_type, data, reference, schedule, verify
    ):
        """Async software pipeline test."""
        out_np = data[0]
        a_np = data[1]
        if comp_type == "single_input":
            ref = reference(a_np)
        else:
            b_np = data[2]
            ref = reference(a_np, b_np)

        with tvm.transform.PassContext(
            config={
                "tir.use_async_copy": 1,
                "tir.dma_bypass_cache": 1,
                "tir.merge_async_commit_queue_scope": False,
            }
        ):
            # tvm.lower(schedule.mod["main"]).show()
            func = tvm.build(schedule.mod["main"], target=get_hexagon_target("v68"))

        with hexagon_launcher.create_session() as hexagon_session:
            dev = hexagon_session.device
            mod = hexagon_session.load_module(func)
            out = tvm.nd.array(out_np, device=dev)
            a = tvm.nd.array(a_np, device=dev)
            if comp_type == "single_input":
                mod(a, out)
            else:
                b = tvm.nd.array(b_np, device=dev)
                mod(a, b, out)

            verify(out, ref)


if __name__ == "__main__":
    tvm.testing.main()
