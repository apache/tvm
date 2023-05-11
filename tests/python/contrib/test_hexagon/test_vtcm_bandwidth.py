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

"""Test theoretical bandwith for data transfers to VTCM for different strategies."""

import numpy as np
import pytest

import tvm
from tvm.script import tir as T
from tvm.tir.tensor_intrin.hexagon import DMA_READ_128_i8

from .infrastructure import get_hexagon_target

MB = 1024**2
KB = 1024
TEST_OUTPUT_TEMPLATE = (
    "Test bandwidth with buffer size {}MB... \n"
    "    -Base: {} GBps \n    -Vectorized: {} GBps\n"
    "    -Vectorized and Parallelized: {} GBps\n"
    "    -Sync DMA: {} GBps\n"
    "    -Single DMA Copy: {} GBps\n"
)


def memcopy_operator(size):
    """Generate memory copy operator."""

    @T.prim_func
    def operator(a: T.handle, a_v: T.handle) -> None:
        a_buffer = T.match_buffer(a, size, dtype="int8", align=128, scope="global")
        a_global_vtcm = T.match_buffer(a_v, size, dtype="int8", align=128, scope="global.vtcm")
        for ax0 in T.serial(size):
            with T.block("A_global.vtcm"):
                v0_ind = T.axis.spatial(size, ax0)
                T.reads(a_buffer[v0_ind])
                T.writes(a_global_vtcm[v0_ind])
                a_global_vtcm[v0_ind] = a_buffer[v0_ind]

    return operator


def single_dma_operator(size):
    """Generate single dma operator."""

    @T.prim_func
    def operator(a: T.handle, a_v: T.handle) -> None:
        a_buffer = T.match_buffer(a, size, dtype="int8", align=128, scope="global")
        a_global_vtcm = T.match_buffer(a_v, size, dtype="int8", align=128, scope="global.vtcm")
        T.evaluate(
            T.tvm_call_packed(
                "device_api.hexagon.dma_copy_dltensor",
                T.tvm_stack_make_array(
                    a_global_vtcm.data,
                    T.tvm_stack_make_shape(size, dtype="handle"),
                    0,
                    1,
                    a_global_vtcm.dtype,
                    0,
                    dtype="handle",
                ),
                T.tvm_stack_make_array(
                    a_buffer.data,
                    T.tvm_stack_make_shape(size, dtype="handle"),
                    0,
                    1,
                    a_buffer.dtype,
                    0,
                    dtype="handle",
                ),
                T.cast(size, dtype="int"),
                True,  # bypass cache
                dtype="int32",
            )
        )

    return operator


def evaluate(hexagon_session, sch, size):
    """Evaluate schedule."""
    a_shape = size

    func_tir = tvm.build(sch.mod["main"], target=get_hexagon_target("v69"))
    module = hexagon_session.load_module(func_tir)

    a = np.random.randint(-128, 127, a_shape, dtype="int8")
    a_vtcm = np.zeros(a_shape, dtype="int8")

    a_hexagon = tvm.runtime.ndarray.array(a, device=hexagon_session.device, mem_scope="global")
    a_vtcm_hexagon = tvm.runtime.ndarray.array(
        a_vtcm, device=hexagon_session.device, mem_scope="global.vtcm"
    )

    if tvm.testing.utils.IS_IN_CI:
        # Run with reduced number and repeat for CI
        timer = module.time_evaluator("__tvm_main__", hexagon_session.device, number=1, repeat=1)
    else:
        timer = module.time_evaluator("__tvm_main__", hexagon_session.device, number=10, repeat=10)

    runtime = timer(a_hexagon, a_vtcm_hexagon)

    gbps = round((size / 2**30) / runtime.mean, 4)
    tvm.testing.assert_allclose(a_vtcm_hexagon.asnumpy(), a)

    return gbps


class TestMatMulVec:
    """MatMul test class."""

    # Removed most of these to speedup CI.
    size = tvm.testing.parameter(
        128,
        KB,
        10 * KB,
        100 * KB,
        MB,
    )

    outer_split = tvm.testing.parameter(4)
    unroll_split = tvm.testing.parameter(2)
    vector_split = tvm.testing.parameter(128)

    @tvm.testing.requires_hexagon
    def test_bandwidth(self, hexagon_session, size, outer_split, unroll_split, vector_split):
        """Test bandwidth."""

        if tvm.testing.utils.IS_IN_CI and (size > 128):
            pytest.skip("Skipping test since it takes too long in CI.")

        # Run the base memcopy operator.
        sch = tvm.tir.Schedule(memcopy_operator(size))
        base_gpbs = evaluate(hexagon_session, sch, size)

        # Run with some basic unroll and vectorize scheduling.
        sch = tvm.tir.Schedule(memcopy_operator(size))
        vtcm_block_a = sch.get_block("A_global.vtcm")
        v_block = sch.get_loops(vtcm_block_a)
        _, vio_a, vii_a = sch.split(v_block[0], factors=[None, unroll_split, vector_split])
        sch.unroll(vio_a)
        sch.vectorize(vii_a)
        vectorize_gbps = evaluate(hexagon_session, sch, size)

        # Run with some basic unroll and vectorize scheduling and parallelization.
        sch = tvm.tir.Schedule(memcopy_operator(size))
        vtcm_block_a = sch.get_block("A_global.vtcm")
        v_block = sch.get_loops(vtcm_block_a)
        vbo_a, _, vio_a, vii_a = sch.split(
            v_block[0], factors=[outer_split, None, unroll_split, vector_split]
        )
        sch.unroll(vio_a)
        sch.vectorize(vii_a)
        sch.parallel(vbo_a)
        parallel_gbps = evaluate(hexagon_session, sch, size)

        # Run with some basic unroll and vectorize scheduling and parallelization.
        sch = tvm.tir.Schedule(memcopy_operator(size))
        block = sch.get_block("A_global.vtcm")
        loops = sch.get_loops(block)
        _, inner = sch.split(loops[0], [None, 128])
        sch.tensorize(inner, DMA_READ_128_i8)
        # print(sch.mod.script())
        sync_dma_gbps = evaluate(hexagon_session, sch, size)

        # Run using a single dma copy to transfer the data.
        sch = tvm.tir.Schedule(single_dma_operator(size))
        single_dma_gbps = evaluate(hexagon_session, sch, size)

        mbs = round(size / MB, 2)
        print(
            TEST_OUTPUT_TEMPLATE.format(
                mbs, base_gpbs, vectorize_gbps, parallel_gbps, sync_dma_gbps, single_dma_gbps
            )
        )


if __name__ == "__main__":
    tvm.testing.main()
